# %% [markdown]
# # MOSS-TTS-Local 4B — per-LAYER backbone diff (#249): pin the op
#
# frame-0 backbone diverges (cos 0.989) though hparams/RoPE/QK-norm are all
# correct. Dump each Qwen3 block's last-position hidden (prompt prefill) from
# both and find the layer where cosine first drops:
#   sharp drop at one layer -> that layer/op is the bug.
#   gradual decline         -> systematic per-layer numerics (harder / f16-ish).

# %% [code]
import json, os, subprocess, sys, gc, math
from pathlib import Path

REPO = Path("/kaggle/temp/CrispASR")
WORK = Path("/kaggle/working")
REF = os.environ.get("CRISPASR_REF", "fix/249-moss")
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--recursive", "--depth", "1", "--branch", REF,
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init",
                           "--recursive", "--depth", "1"], timeout=1800)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start", ref=REF)
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

# ── our per-layer dump ────────────────────────────────────────────────────────
dump = WORK / "ours_layers.txt"
env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DUMP_LAYERS": str(dump)}
with kh.build_heartbeat("ours.synth"):
    try:
        subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                        "--tts", TEXT, "--tts-output", str(WORK / "o.wav"), "--no-prints"],
                       env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
ours = {}
if dump.exists():
    for line in dump.read_text().splitlines():
        p = line.split()
        if p and p[0].startswith("blk_"):
            ours[int(p[0][4:])] = [float(x) for x in p[1:]]
step("ours.done", n_layers=len(ours))

# ── reference per-layer hooks ─────────────────────────────────────────────────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
model = AutoModel.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager", token=TOKEN).eval()
proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
ref = {}
def mk(i):
    def hook(_m, _i, o):
        hs = o[0] if isinstance(o, (tuple, list)) else o
        ref[i] = hs[:, -1, :].detach().float().reshape(-1).tolist()
    return hook
layers = model.transformer.layers
for i, lyr in enumerate(layers):
    lyr.register_forward_hook(mk(i))
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.gen"):
    model.generate(input_ids=input_ids, max_new_frames=1, do_sample=False)
step("ref.done", n_layers=len(ref))
del model; gc.collect()

# ── per-layer cosine ──────────────────────────────────────────────────────────
def cos(a, b):
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b)); na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return round(dot / (na * nb + 1e-9), 6)
per = [{"layer": i, "cos": cos(ours.get(i), ref.get(i))} for i in range(max(len(ours), len(ref)))]
# first layer whose cos drops meaningfully below the previous
drop = None
prev = 1.0
for e in per:
    if e["cos"] is not None:
        if prev - e["cos"] > 0.003 and drop is None:
            drop = {"layer": e["layer"], "from": prev, "to": e["cos"]}
        prev = e["cos"]
(WORK / "layer_diff.json").write_text(json.dumps({"first_drop": drop, "per_layer": per}, indent=2))
step("diff", first_drop=drop, per_layer=[{"l": e["layer"], "cos": e["cos"]} for e in per if e["layer"] % 4 == 0 or e["layer"] < 4])
print("DONE first_drop", drop, flush=True)
