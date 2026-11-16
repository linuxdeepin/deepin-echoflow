# %% [markdown]
# # MOSS-TTS-Local 4B — per-SUBLAYER diff (#249 option 2): attn vs MLP
#
# Per-layer diff pinned the prefill backbone divergence to START at block 10
# (blocks 0-9 match to f32 precision). Now split block 10 open: dump the
# attention output and the SwiGLU-MLP output (both pre-residual, last position)
# from ours and the reference, for layers 9/10/11. Block input matches (0-9 exact)
# so whichever sub-op's cosine first drops is the culprit:
#   attn drops  -> attention (QK-norm / RoPE / GQA / softmax f16 precision)
#   mlp  drops  -> SwiGLU MLP (gate/up/down / activation precision)

# %% [code]
import json, os, subprocess, sys, gc, math
from pathlib import Path

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
LAYERS = [0, 1, 2, 4, 6, 8, 9, 10]
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--recursive", "--depth", "1", "--branch", REF,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init",
                           "--recursive", "--depth", "1"], timeout=1800)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start", ref=REF, layers=LAYERS)
TOKEN = kh.resolve_hf_token("HF_TOKEN")
kh.install_build_toolchain()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done")

MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download"):
    F16 = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS))
    CODEC = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS))
TEXT = "Hello world."


def read_dump(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if p:
                out[p[0]] = [float(x) for x in p[1:]]
    return out


# ── our per-sublayer dump (one synth per target layer) ────────────────────────
ours = {}
for L in LAYERS:
    dump = WORK / f"ours_sub_{L}.txt"
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER": str(L),
           "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER_PATH": str(dump)}
    with kh.build_heartbeat(f"ours.synth.{L}"):
        try:
            subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                            "--tts", TEXT, "--tts-output", str(WORK / f"o_{L}.wav"), "--no-prints"],
                           env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
    ours.update(read_dump(dump))
step("ours.done", keys=sorted(ours.keys()))

# ── reference sub-module hooks ────────────────────────────────────────────────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
model = AutoModel.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager", token=TOKEN).eval()
proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
ref = {}


def mk(name):
    def hook(_m, _i, o):
        hs = o[0] if isinstance(o, (tuple, list)) else o
        ref[name] = hs[:, -1, :].detach().float().reshape(-1).tolist()
    return hook


layers = model.transformer.layers
hooked = []
for L in LAYERS:
    lyr = layers[L]
    # Qwen3DecoderLayer: .self_attn returns (attn_out, ...); .mlp returns tensor
    attn_mod = getattr(lyr, "self_attn", None) or getattr(lyr, "attn", None)
    mlp_mod = getattr(lyr, "mlp", None) or getattr(lyr, "feed_forward", None)
    if attn_mod is not None:
        hooked.append(attn_mod.register_forward_hook(mk(f"sub_attn_{L}")))
    if mlp_mod is not None:
        hooked.append(mlp_mod.register_forward_hook(mk(f"sub_mlp_{L}")))
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.gen"):
    model.generate(input_ids=input_ids, max_new_frames=1, do_sample=False)
step("ref.done", keys=sorted(ref.keys()))
del model
gc.collect()


# ── cosine per sub-op ─────────────────────────────────────────────────────────
def cos(a, b):
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return round(dot / (na * nb + 1e-9), 6)


def l2rel(a, b):
    if not a or not b or len(a) != len(b):
        return None
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b)) + 1e-9
    return round(num / den, 6)


def top_div(a, b, n=3):
    # top-n channels by |a-b|, with the reference magnitude there (outlier test)
    if not a or not b or len(a) != len(b):
        return None
    idx = sorted(range(len(a)), key=lambda i: abs(a[i] - b[i]), reverse=True)[:n]
    return [{"ch": i, "ours": round(a[i], 3), "ref": round(b[i], 3),
             "|d|": round(abs(a[i] - b[i]), 3)} for i in idx]


def maxabs(v):
    return round(max(abs(x) for x in v), 3) if v else None


table = []
for L in LAYERS:
    for which in ("sub_attn", "sub_mlp"):
        k = f"{which}_{L}"
        o, r = ours.get(k), ref.get(k)
        table.append({"key": k, "cos": cos(o, r), "l2rel": l2rel(o, r),
                      "ref_maxabs": maxabs(r), "ours_maxabs": maxabs(o),
                      "top_div": top_div(o, r),
                      "have_ours": k in ours, "have_ref": k in ref})

# verdict: at the first diverging layer, which sub-op drops first
verdict = "inconclusive"
for L in LAYERS:
    ca = cos(ours.get(f"sub_attn_{L}"), ref.get(f"sub_attn_{L}"))
    cm = cos(ours.get(f"sub_mlp_{L}"), ref.get(f"sub_mlp_{L}"))
    if ca is not None and ca < 0.997:
        verdict = f"ATTENTION diverges first at layer {L} (attn cos {ca}, mlp cos {cm})"
        break
    if cm is not None and cm < 0.997:
        verdict = f"MLP diverges first at layer {L} (attn cos {ca} ok, mlp cos {cm})"
        break

(WORK / "sublayer_diff.json").write_text(json.dumps({"verdict": verdict, "table": table}, indent=2))
step("diff", verdict=verdict, table=table)
print("DONE", verdict, json.dumps(table), flush=True)
