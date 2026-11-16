# %% [markdown]
# # MOSS-TTS-Local 4B — does OUR F16 (unquantized) stop reliably? (#249)
#
# The HF reference stops reliably (3/3 seeds, frame ~15). Our quants are
# coin-flippy (q4_k/q6_k runaway depends on text). Structural port is faithful
# (head/projection/feedback/audio-sum all match the reference). Decisive fork:
#   F16 stops reliably  -> port is correct; QUANTIZATION erodes the marginal stop.
#   F16 runs away too   -> genuine port numerics bug -> trajectory diff needed.
#
# Prebuilt v0.8.23 CPU binary (correct numerics; GPU could confound). Synth two
# texts with CRISPASR_MOSS_TTS_LOCAL_DEBUG=1 and report the stop verdict + the
# first stop-head logits.

# %% [code]
import json, os, re, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("start")
TOKEN = kh.resolve_hf_token("HF_TOKEN")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

# prebuilt binary
BIN = WORK / "bin"; BIN.mkdir(exist_ok=True)
CLI = BIN / "crispasr"
subprocess.check_call(
    "wget -q https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.23/crispasr-linux-x86_64.tar.gz "
    f"-O /tmp/c.tgz && tar -xzf /tmp/c.tgz -C {BIN} --strip-components=1", shell=True)
CLI.chmod(0o755)
os.environ["LD_LIBRARY_PATH"] = f"{BIN}:{os.environ.get('LD_LIBRARY_PATH','')}"
step("binary.ready")

MODELS = Path("/kaggle/temp/models"); MODELS.mkdir(parents=True, exist_ok=True)
REPO_ID = "cstr/moss-tts-local-v1.5-GGUF"
with kh.build_heartbeat("download"):
    F16 = hf_hub_download(REPO_ID, "moss-tts-local-v1.5-f16.gguf", local_dir=str(MODELS))
    CODEC = hf_hub_download(REPO_ID, "moss-tts-local-v1.5-codec.gguf", local_dir=str(MODELS))
step("downloaded")

TEXTS = {"short": "Hello world.",
         "long": "The quick brown fox jumps over the lazy dog. "
                 "Speech synthesis should stay intelligible over a longer passage."}
results = {}
for tag, text in TEXTS.items():
    wav = WORK / f"f16_{tag}.wav"
    log = WORK / f"f16_{tag}.log"
    env = {**os.environ, "CRISPASR_MOSS_TTS_LOCAL_DEBUG": "1"}
    with open(log, "w") as fh, kh.build_heartbeat(f"synth.{tag}"):
        try:
            subprocess.run([str(CLI), "--backend", "moss-tts-local", "-m", F16,
                            "--codec-model", CODEC, "--tts", text,
                            "--tts-output", str(wav), "--no-prints"],
                           stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=900)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
    txt = log.read_text(errors="replace")
    gen = re.search(r"generated (\d+) frames .*?(stopped naturally|runaway)", txt)
    logits = re.findall(r"frame (\d+): stop_head continue=([\-\d.]+) stop=([\-\d.]+)", txt)
    r = {"verdict": gen.group(2) if gen else ("runaway(timeout)" if timed_out else "unknown"),
         "frames": int(gen.group(1)) if gen else None,
         "first_logits": [{"f": int(f), "cont": float(c), "stop": float(s)} for f, c, s in logits[:12]]}
    results[tag] = r
    step(f"synth.{tag}", verdict=r["verdict"], frames=r["frames"], first=r["first_logits"][:4])

(WORK / "f16_stop.json").write_text(json.dumps(results, indent=2))
step("done", verdicts={k: v["verdict"] for k, v in results.items()})
print("DONE", json.dumps(results)[:1500], flush=True)
