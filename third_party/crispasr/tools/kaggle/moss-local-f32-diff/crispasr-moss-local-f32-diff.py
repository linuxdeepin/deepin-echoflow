# %% [markdown]
# # MOSS-TTS-Local 4B — f32-compute sublayer diff (#249 option 2): precision or bug?
#
# The prefill backbone drift is broad & distributed (not outliers), grows smoothly
# 0.3%->4.6% over layers 0-9, then JUMPS 5x at layer 10 (cos 0.972). That's the
# fingerprint of ggml f16-compute numeric drift hitting a high-gain layer — OR a
# genuine graph bug. DECISIVE TEST: convert the backbone to an f32 GGUF (no upload,
# convert on-node) and re-run our sublayer diff. If f32 collapses the layer-10 jump
# back to ~0.999, it's pure precision (fix = ship the stop path at higher precision);
# if it still jumps, it's an algorithmic/graph difference and we keep hunting.

# %% [code]
import json, os, subprocess, sys, gc, math
from pathlib import Path

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
LAYERS = [8, 9, 10, 11]
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
TEXT = "Hello world."
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
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer", "gguf"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done")

# ── convert backbone -> f32 GGUF on-node (no upload) ──────────────────────────
MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("hf.snapshot"):
    HF_DIR = snapshot_download(HF, local_dir=str(MODELS / "hf"), token=TOKEN,
                               allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.py"])
F32 = MODELS / "moss-tts-local-v1.5-f32.gguf"
step("convert.f32")
with kh.build_heartbeat("convert.f32"):
    subprocess.check_call([sys.executable, str(REPO / "models" / "convert-moss-tts-local-to-gguf.py"),
                           "--input", HF_DIR, "--output", str(F32), "--dtype", "f32"], timeout=3600)
with kh.build_heartbeat("download.codec"):
    CODEC = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS))
step("convert.done", f32_gb=round(F32.stat().st_size / 1e9, 2))


def read_dump(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if p:
                out[p[0]] = [float(x) for x in p[1:]]
    return out


# ── our f32-GGUF per-sublayer dump (subprocess frees its RAM on exit) ──────────
ours = {}
for L in LAYERS:
    dump = WORK / f"ours_f32_sub_{L}.txt"
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER": str(L),
           "CRISPASR_MOSS_TTS_LOCAL_DUMP_SUBLAYER_PATH": str(dump)}
    with kh.build_heartbeat(f"ours.synth.{L}"):
        try:
            subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", str(F32), "--codec-model", CODEC,
                            "--tts", TEXT, "--tts-output", str(WORK / f"o_{L}.wav"), "--no-prints"],
                           env=env, timeout=900, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
    ours.update(read_dump(dump))
step("ours.done", keys=sorted(ours.keys()))
# free the 16GB f32 GGUF from page cache before loading the torch reference
try:
    F32.unlink()
except OSError:
    pass
gc.collect()

# ── reference (f32) sub-module hooks ──────────────────────────────────────────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
model = AutoModel.from_pretrained(HF_DIR, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager").eval()
proc = AutoProcessor.from_pretrained(HF_DIR, trust_remote_code=True)
ref = {}


def mk(name):
    def hook(_m, _i, o):
        hs = o[0] if isinstance(o, (tuple, list)) else o
        ref[name] = hs[:, -1, :].detach().float().reshape(-1).tolist()
    return hook


layers = model.transformer.layers
for L in LAYERS:
    lyr = layers[L]
    attn_mod = getattr(lyr, "self_attn", None) or getattr(lyr, "attn", None)
    mlp_mod = getattr(lyr, "mlp", None) or getattr(lyr, "feed_forward", None)
    if attn_mod is not None:
        attn_mod.register_forward_hook(mk(f"sub_attn_{L}"))
    if mlp_mod is not None:
        mlp_mod.register_forward_hook(mk(f"sub_mlp_{L}"))
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.gen"):
    model.generate(input_ids=input_ids, max_new_frames=1, do_sample=False)
step("ref.done", keys=sorted(ref.keys()))
del model
gc.collect()


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


table = [{"key": f"{w}_{L}", "cos": cos(ours.get(f"{w}_{L}"), ref.get(f"{w}_{L}")),
          "l2rel": l2rel(ours.get(f"{w}_{L}"), ref.get(f"{w}_{L}"))}
         for L in LAYERS for w in ("sub_attn", "sub_mlp")]
c10 = cos(ours.get("sub_attn_10"), ref.get("sub_attn_10"))
verdict = ("PRECISION: f32 compute collapses the layer-10 jump (attn cos %s >= 0.997) -> ship higher precision" % c10
           if c10 is not None and c10 >= 0.997
           else "GRAPH BUG: layer-10 still diverges at f32 (attn cos %s) -> algorithmic difference remains" % c10)
(WORK / "f32_diff.json").write_text(json.dumps({"verdict": verdict, "table": table}, indent=2))
step("diff", verdict=verdict, table=table)
print("DONE", verdict, json.dumps(table), flush=True)
