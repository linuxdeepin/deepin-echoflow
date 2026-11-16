# %% [markdown]
# # MOSS-TTS-Local 4B — stop-logit TRAJECTORY DIFF: our port vs HF reference (#249)
#
# Our F16 stops @frame 55; the reference @~15. Head/structure verified identical,
# so it's a per-frame forward DRIFT that delays our stop. This dumps the RAW
# per-frame stop logits [continue, stop] from BOTH, run GREEDY (deterministic),
# on the same prompt, and localizes the divergence:
#   frame-0 logits differ  -> immediate forward bug (prompt/backbone/local at f0).
#   frame-0 match, gap-slope slower -> accumulation drift (backbone not winding
#                                      down as fast as the reference).
#
# Order: our C++ subprocess first (frees RAM on exit), THEN the float32 reference
# (~16GB) — never both resident. CPU (correct numerics).

# %% [code]
import json, os, re, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/kaggle/temp/CrispASR")  # NOT /kaggle/working (keeps the output small)
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
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "huggingface_hub", "hf_transfer", "safetensors"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

# ── build our instrumented C++ ────────────────────────────────────────────────
BUILD = REPO / "build"
step("cmake.configure")
subprocess.run(["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
                "-DCMAKE_BUILD_TYPE=Release"] + kh.crispasr_cmake_flags(), check=True)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}")
CLI = (BUILD / "bin" / "crispasr") if (BUILD / "bin" / "crispasr").exists() else next(iter(BUILD.rglob("crispasr")))
os.environ["LD_LIBRARY_PATH"] = f"{BUILD/'src'}:{BUILD/'ggml'/'src'}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("build.done", cli=str(CLI))

MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download.gguf"):
    F16 = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS))
    CODEC = hf_hub_download("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS))

TEXT = "Hello world."
MAXF = 128

# ── OUR C++ greedy trajectory (subprocess; frees RAM after) ────────────────────
step("ours.synth")
env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_GREEDY_TEXT": "1",
       "CRISPASR_MOSS_TTS_LOCAL_GREEDY_AUDIO": "1", "CRISPASR_MOSS_TTS_LOCAL_DUMP_STOP": "1"}
log = WORK / "ours.log"
with open(log, "w") as fh, kh.build_heartbeat("ours.synth"):
    p = subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16, "--codec-model", CODEC,
                        "--tts", TEXT, "--tts-output", str(WORK / "ours.wav"), "--no-prints"],
                       stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=1800)
logtxt = log.read_text(errors="replace")
ours = [{"f": int(f), "cont": float(c), "stop": float(s), "gap": float(g)}
        for f, c, s, g in re.findall(r"DUMPSTOP frame=(\d+) cont=([\-\d.]+) stop=([\-\d.]+) gap=([\-\d.]+)", logtxt)]
step("ours.done", rc=p.returncode, n=len(ours), first=ours[:3],
     tail="" if ours else logtxt[-500:])

# ── HF reference greedy trajectory ────────────────────────────────────────────
step("ref.load")
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402
torch.set_num_threads(os.cpu_count() or 4)
HF = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
model = AutoModel.from_pretrained(HF, trust_remote_code=True, torch_dtype=torch.float32,
                                  attn_implementation="eager", token=TOKEN).eval()
proc = AutoProcessor.from_pretrained(HF, trust_remote_code=True)
ref_logits = []


def hook(_m, _i, o):
    t = o.detach().float().reshape(-1, o.shape[-1])
    ref_logits.append(t[-1].tolist())


h = model.local_text_lm_head.register_forward_hook(hook)
msg = proc.build_user_message(text=TEXT)
feat = proc([msg], mode="generation")
input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
with kh.build_heartbeat("ref.generate"):
    model.generate(input_ids=input_ids, max_new_frames=MAXF, do_sample=False)  # GREEDY
h.remove()
ref = [{"f": i, "cont": round(c, 6), "stop": round(s, 6), "gap": round(c - s, 6)}
       for i, (c, s) in enumerate(ref_logits)]
step("ref.done", n=len(ref), first=ref[:3])

# ── diff ──────────────────────────────────────────────────────────────────────
def stop_frame(traj):
    return next((t["f"] for t in traj if t["gap"] < 0), None)


diff = {"ours_frames": len(ours), "ref_frames": len(ref),
        "ours_stop_frame": stop_frame(ours), "ref_stop_frame": stop_frame(ref),
        "frame0": {"ours": ours[0] if ours else None, "ref": ref[0] if ref else None},
        "gap_at": {str(f): {"ours": next((t["gap"] for t in ours if t["f"] == f), None),
                            "ref": next((t["gap"] for t in ref if t["f"] == f), None)}
                   for f in (0, 1, 2, 5, 10, 15, 20, 30, 50)}}
(WORK / "traj_diff.json").write_text(json.dumps({"diff": diff, "ours": ours, "ref": ref}, indent=2))
step("diff", **diff)
print("DONE", json.dumps(diff), flush=True)
