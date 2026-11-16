# %% [markdown]
# # MOSS-TTS-Local 4B — frame-0 per-component diff (#249): backbone vs depth
#
# Confirmed a real forward bug (reference f16 stops @~18, ours @~55). frame-0 is
# a single prompt forward (no accumulation) — the cleanest place to localize it.
# Compare our global_hidden (Qwen3 backbone out) and lh (local/depth transformer
# out) against the reference's for the SAME prompt:
#   global matches, local differs -> bug in the DEPTH (local) transformer.
#   global differs                -> bug in the Qwen3 BACKBONE forward.
# cosine ~1 tolerates f16 rounding; a real bug drops it well below 1.

# %% [code]
import json, os, re, subprocess, sys, gc, math
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

# ── our C++: dump frame-0 global + local ──────────────────────────────────────
dump = WORK / "ours_hidden.txt"
env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DUMP_HIDDEN": str(dump)}
with kh.build_heartbeat("ours.synth"):
    try:
        subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                        "--tts", TEXT, "--tts-output", str(WORK / "o.wav"), "--no-prints"],
                       env=env, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass  # dump is written at frame 0, long before any timeout
og = ol = None
if dump.exists():
    for line in dump.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "GLOBAL":
            og = [float(x) for x in parts[1:]]
        elif parts and parts[0] == "LOCAL":
            ol = [float(x) for x in parts[1:]]
step("ours.done", global_n=len(og) if og else 0, local_n=len(ol) if ol else 0)

# ── reference: frame-0 global (transformer out) + local (stop-head input) ─────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
model = AutoModel.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager", token=TOKEN).eval()
proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
rg = {}; rl = {}
def gh(_m, _i, o):
    hs = getattr(o, "last_hidden_state", None)
    if hs is None:
        hs = o[0] if isinstance(o, (tuple, list)) else o
    rg["h"] = hs[:, -1, :].detach().float().reshape(-1).tolist()
def lpre(_m, inp):
    x = inp[0]; rl["h"] = x.reshape(-1, x.shape[-1])[-1].detach().float().tolist()
h1 = model.transformer.register_forward_hook(gh)
h2 = model.local_text_lm_head.register_forward_pre_hook(lpre)
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.gen"):
    model.generate(input_ids=input_ids, max_new_frames=1, do_sample=False)
h1.remove(); h2.remove()
step("ref.done", global_n=len(rg.get("h", [])), local_n=len(rl.get("h", [])))
del model; gc.collect()

# ── compare ───────────────────────────────────────────────────────────────────
def cmp(a, b):
    if not a or not b or len(a) != len(b):
        return {"len_a": len(a) if a else 0, "len_b": len(b) if b else 0, "mismatch": True}
    dot = sum(x * y for x, y in zip(a, b)); na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb + 1e-9)
    maxd = max(abs(x - y) for x, y in zip(a, b))
    return {"cos": round(cos, 6), "maxdiff": round(maxd, 4), "norm_ours": round(na, 3), "norm_ref": round(nb, 3)}
gc_ = cmp(og, rg.get("h"))
lc_ = cmp(ol, rl.get("h"))
if gc_.get("cos", 1) < 0.999:
    verdict = f"BACKBONE bug (global cos={gc_.get('cos')}) — Qwen3 backbone forward diverges at the prompt"
elif lc_.get("cos", 1) < 0.999:
    verdict = f"DEPTH-TRANSFORMER bug (global cos={gc_.get('cos')} OK, local cos={lc_.get('cos')})"
else:
    verdict = f"both match at f0 (global cos={gc_.get('cos')}, local cos={lc_.get('cos')}) — divergence is accumulation-only"
(WORK / "f0_diff.json").write_text(json.dumps({"verdict": verdict, "global": gc_, "local": lc_}, indent=2))
step("diff", verdict=verdict, glob=gc_, loc=lc_)
print("DONE", verdict, flush=True)
