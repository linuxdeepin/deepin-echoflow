# %% [markdown]
# # CrispASR — produce + validate + upload Q8_0 GGUFs (vibevoice-asr, voxcpm2)
#
# HF requests:
#   - cstr/vibevoice-asr-GGUF/discussions/2  "Could you please share Q8_0 GGUF?"
#   - cstr/voxcpm2-GGUF/discussions/1        (Q8_0 too)
#
# Q8_0 is a CPU-only re-quantization of the published F16 (crispasr-quantize
# streams tensors, ~500 MB RAM) — no GPU compute needed, but Kaggle CPU workers
# have no internet, so this runs on a GPU kernel purely for the fast HF up/down
# link (16.7 GB F16 down + 8.85 GB Q8 up for vibevoice).
#
# For each model: download F16 → quantize Q8_0 → VALIDATE → upload only on pass
# → delete local files. vibevoice-asr is validated by transcribing jfk.wav;
# voxcpm2 (TTS) by synthesizing then whisper-roundtripping. voxcpm2 only runs if
# vibevoice's Q8 passed (the "if it works, then also do voxcpm2" gate).

# %% [code]
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")

# ── Kaggle regime: clone CrispASR + import harness (bundled fallback) ──────────
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = WORK / "CrispASR"
if not REPO.exists():
    try:
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
            CRISPASR_URL, str(REPO)])
        subprocess.check_call(
            f"git -C {REPO} checkout HEAD -- samples/ tools/kaggle/", shell=True)
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start")

# ── HF auth: env → Kaggle Secret → mounted crispasr-hf-token dataset ──────────
TOKEN = kh.resolve_hf_token("HF_TOKEN")
step("hf_token.resolved", have=bool(TOKEN))

# ── deps ──────────────────────────────────────────────────────────────────────
step("install-deps.begin")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download, HfApi
step("install-deps.done")

# ── pre-built binary (v0.8.6 CPU tarball ships crispasr + crispasr-quantize) ──
RELEASE = "v0.8.6"
TARBALL = "crispasr-linux-x86_64.tar.gz"
BIN = WORK / "bin"
BIN.mkdir(exist_ok=True)
CRISPASR = BIN / "crispasr"
QUANT = BIN / "crispasr-quantize"
step("binary-download.begin", release=RELEASE)
subprocess.check_call(
    f"wget -q https://github.com/CrispStrobe/CrispASR/releases/download/{RELEASE}/{TARBALL} "
    f"-O /tmp/c.tar.gz && tar -xzf /tmp/c.tar.gz -C {BIN} --strip-components=1", shell=True)
CRISPASR.chmod(0o755)
QUANT.chmod(0o755)
assert CRISPASR.is_file() and QUANT.is_file(), "binaries missing"
step("binary-download.done")

WAV = REPO / "samples" / "jfk.wav"

# /kaggle/working is capped ~20 GB, too small for vibevoice F16(16.7)+Q8(8.85)=
# 25.5 GB. Pick the mount with the most free space (Kaggle GPU instances have
# ~50-60 GB on / or /tmp). Files are uploaded straight to HF, so they don't need
# to live in /kaggle/working.
def _free(p):
    try:
        return shutil.disk_usage(p).free / 1e9
    except Exception:
        return 0.0


_cands = [("/tmp/models", "/tmp"), ("/kaggle/temp/models", "/kaggle/temp"),
          (str(WORK / "models"), str(WORK)), ("/models", "/")]
_best = max(_cands, key=lambda c: _free(c[1]))
MODELS = Path(_best[0])
MODELS.mkdir(parents=True, exist_ok=True)
step("models-dir.chosen", dir=str(MODELS), free_gb=round(_free(_best[1]), 1),
     probe={d: round(_free(d), 1) for _, d in _cands})

# whisper-tiny.en for the TTS round-trip check (small; only for voxcpm2)
WHISPER = None


def _get_whisper():
    global WHISPER
    if WHISPER is None:
        step("whisper-download.begin")
        WHISPER = hf_hub_download(repo_id="ggerganov/whisper.cpp",
                                  filename="ggml-tiny.en.bin", local_dir=str(MODELS))
        step("whisper-download.done")
    return WHISPER


def _run(cmd, timeout=1200):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).split()


def free_gb():
    return round(shutil.disk_usage(str(MODELS)).free / 1e9, 1)


def quantize(f16_path, q8_path):
    step("quantize.begin", f16=Path(f16_path).name, free_gb=free_gb())
    with kh.build_heartbeat("quantize"):
        p = _run([str(QUANT), str(f16_path), str(q8_path), "q8_0"], timeout=3600)
    ok = p.returncode == 0 and Path(q8_path).is_file()
    step("quantize.done", exit=p.returncode, ok=ok,
         q8_gb=round(Path(q8_path).stat().st_size / 1e9, 2) if ok else 0,
         tail=p.stdout[-300:] if not ok else "")
    return ok


def upload(q8_path, repo, msg):
    step("upload.begin", repo=repo, file=Path(q8_path).name,
         gb=round(Path(q8_path).stat().st_size / 1e9, 2))
    HfApi(token=TOKEN).upload_file(
        path_or_fileobj=str(q8_path), path_in_repo=Path(q8_path).name,
        repo_id=repo, repo_type="model", commit_message=msg)
    step("upload.done", repo=repo)


# jfk gold key content-words that a working ASR must produce
JFK_KEYS = ["fellow", "americans", "country", "you"]


def do_vibevoice():
    repo = "cstr/vibevoice-asr-GGUF"
    f16 = MODELS / "vibevoice-asr-f16.gguf"
    q8 = MODELS / "vibevoice-asr-q8_0.gguf"
    need = 16.7 + 8.85 + 1.0
    if free_gb() < need:
        step("vibevoice.skip", reason=f"disk {free_gb()}GB < {need}GB")
        return False
    step("vibevoice.download-f16.begin")
    hf_hub_download(repo_id=repo, filename="vibevoice-asr-f16.gguf", local_dir=str(MODELS))
    step("vibevoice.download-f16.done", gb=round(f16.stat().st_size / 1e9, 2))
    if not quantize(f16, q8):
        return False
    # Validate: transcribe jfk with the Q8
    step("vibevoice.transcribe.begin")
    p = _run([str(CRISPASR), "-m", str(q8), "-f", str(WAV), "--backend", "vibevoice"], timeout=900)
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    txt = lines[-1] if lines else ""
    words = set(_norm(txt))
    hits = sum(k in words for k in JFK_KEYS)
    ok = p.returncode == 0 and hits >= 3
    step("vibevoice.transcribe.done", exit=p.returncode, hits=hits, transcript=txt[:200])
    if not ok:
        step("vibevoice.VALIDATION-FAILED", reason=f"only {hits}/4 key words", tail=p.stderr[-300:])
        return False
    upload(q8, repo, "Add Q8_0 (bit-quality re-quant of F16; ASR-validated on jfk) (discussions/2)")
    f16.unlink(missing_ok=True)
    q8.unlink(missing_ok=True)
    step("vibevoice.DONE-uploaded", free_gb=free_gb())
    return True


def do_voxcpm2():
    repo = "cstr/voxcpm2-GGUF"
    f16 = MODELS / "voxcpm2-f16.gguf"
    q8 = MODELS / "voxcpm2-q8_0.gguf"
    wav_out = WORK / "voxcpm2_out.wav"
    step("voxcpm2.download-f16.begin", free_gb=free_gb())
    hf_hub_download(repo_id=repo, filename="voxcpm2-f16.gguf", local_dir=str(MODELS))
    step("voxcpm2.download-f16.done", gb=round(f16.stat().st_size / 1e9, 2))
    if not quantize(f16, q8):
        return False
    # Validate (TTS): zero-shot synthesize, then whisper round-trip.
    syn = "The quick brown fox jumps over the lazy dog."
    step("voxcpm2.synth.begin")
    p = _run([str(CRISPASR), "-m", str(q8), "--backend", "voxcpm2-tts",
              "--tts", syn, "--tts-output", str(wav_out)], timeout=900)
    ok_synth = p.returncode == 0 and wav_out.is_file() and wav_out.stat().st_size > 20000
    step("voxcpm2.synth.done", exit=p.returncode,
         wav_kb=round(wav_out.stat().st_size / 1e3) if wav_out.is_file() else 0,
         tail=p.stderr[-300:] if not ok_synth else "")
    if not ok_synth:
        step("voxcpm2.VALIDATION-FAILED", reason="synthesis produced no/empty audio")
        return False
    w = _get_whisper()
    pr = _run([str(CRISPASR), "-m", w, "-f", str(wav_out)], timeout=600)
    rt = " ".join(re.sub(r"\[[^\]]*\]", "", ln).strip() for ln in pr.stdout.splitlines())
    rtw = set(_norm(rt))
    keys = ["quick", "brown", "fox", "lazy", "dog"]
    hits = sum(k in rtw for k in keys)
    ok = hits >= 3
    step("voxcpm2.roundtrip.done", hits=hits, transcript=rt[:200])
    if not ok:
        step("voxcpm2.VALIDATION-FAILED", reason=f"round-trip only {hits}/5 key words")
        return False
    upload(q8, repo, "Add Q8_0 (re-quant of F16; TTS round-trip-validated) (discussions/1)")
    f16.unlink(missing_ok=True)
    q8.unlink(missing_ok=True)
    step("voxcpm2.DONE-uploaded", free_gb=free_gb())
    return True


# %% [code]
vibe_ok = False
try:
    vibe_ok = do_vibevoice()
except Exception as exc:
    step("vibevoice.EXCEPTION", error=f"{type(exc).__name__}: {exc}")

if vibe_ok:
    try:
        do_voxcpm2()
    except Exception as exc:
        step("voxcpm2.EXCEPTION", error=f"{type(exc).__name__}: {exc}")
else:
    step("voxcpm2.skip", reason="vibevoice Q8 did not validate/upload — gate not met")

step("script.done", free_gb=free_gb())
print("DONE", flush=True)
