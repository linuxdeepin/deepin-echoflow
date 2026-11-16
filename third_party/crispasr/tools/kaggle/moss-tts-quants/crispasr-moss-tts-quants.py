# %% [markdown]
# # CrispASR — produce + validate + upload MOSS-TTS quants (#249)
#
# Issue #249 (subof): "Q6 and Q8 quanta are available for MOSS-TTS. Perhaps this
# would be another step towards improving the quality of generation."
#
# For each model, download F16 (+ codec) → quantize → VALIDATE via TTS→whisper
# round-trip → upload only on pass → delete local files. crispasr-quantize is a
# CPU-only streaming re-quant of the published F16; synthesis is run on CPU too
# (always numerically correct — GPU could confound a quant-quality check).
#
#   cstr/moss-tts-v1.5-GGUF        (8B):  add q6_k, q8_0   (already has q4_k)
#   cstr/moss-tts-local-v1.5-GGUF  (4B):  add q4_k, q6_k, q8_0  (was f16-only)
#
# Prebuilt v0.8.23 CPU tarball (ships crispasr + crispasr-quantize, MOSS-TTS
# support landed v0.8.21). gpu=false, internet=true.

# %% [code]
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")

# ── Kaggle regime: clone CrispASR + import the harness FROM the clone ──────────
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = WORK / "CrispASR"
if not REPO.exists():
    try:
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
            CRISPASR_URL, str(REPO)])
        subprocess.check_call(
            f"git -C {REPO} checkout HEAD -- tools/kaggle/", shell=True)
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start")

TOKEN = kh.resolve_hf_token("HF_TOKEN")
step("hf_token.resolved", have=bool(TOKEN))

step("install-deps.begin")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402
step("install-deps.done")

# ── prebuilt binary: v0.8.23 CPU tarball (crispasr + crispasr-quantize) ───────
RELEASE = "v0.8.23"
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
assert CRISPASR.is_file() and QUANT.is_file(), "binaries missing from tarball"
os.environ["LD_LIBRARY_PATH"] = f"{BIN}:{os.environ.get('LD_LIBRARY_PATH', '')}"
step("binary-download.done")

# ── pick the mount with the most free space (F16 8B ~16GB + a quant) ──────────
def _free(p):
    try:
        return shutil.disk_usage(p).free / 1e9
    except Exception:
        return 0.0


_cands = [("/tmp/models", "/tmp"), ("/kaggle/temp/models", "/kaggle/temp"),
          (str(WORK / "models"), str(WORK))]
_best = max(_cands, key=lambda c: _free(c[1]))
MODELS = Path(_best[0])
MODELS.mkdir(parents=True, exist_ok=True)
step("models-dir.chosen", dir=str(MODELS), free_gb=round(_free(_best[1]), 1))

WHISPER = None


def _get_whisper():
    global WHISPER
    if WHISPER is None:
        step("whisper-download.begin")
        WHISPER = hf_hub_download(repo_id="ggerganov/whisper.cpp",
                                  filename="ggml-tiny.en.bin", local_dir=str(MODELS))
        step("whisper-download.done")
    return WHISPER


def _run(cmd, timeout=2400):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).split()


def free_gb():
    return round(shutil.disk_usage(str(MODELS)).free / 1e9, 1)


SYN_TEXT = "The quick brown fox jumps over the lazy dog."
SYN_KEYS = ["quick", "brown", "fox", "lazy", "dog"]


def quantize(f16_path, out_path, ftype):
    step("quantize.begin", src=Path(f16_path).name, ftype=ftype, free_gb=free_gb())
    with kh.build_heartbeat(f"quantize.{ftype}"):
        p = _run([str(QUANT), str(f16_path), str(out_path), ftype], timeout=3600)
    ok = p.returncode == 0 and Path(out_path).is_file()
    step("quantize.done", ftype=ftype, exit=p.returncode, ok=ok,
         gb=round(Path(out_path).stat().st_size / 1e9, 2) if ok else 0,
         tail=p.stdout[-300:] if not ok else "")
    return ok


def validate_tts(backbone, codec, backend):
    """Synthesize a fixed sentence with the quant backbone + F16 codec, then
    whisper round-trip. Returns (ok, info)."""
    wav = WORK / f"{Path(backbone).stem}_out.wav"
    with kh.build_heartbeat(f"synth.{Path(backbone).stem}"):
        p = _run([str(CRISPASR), "--backend", backend, "-m", str(backbone),
                  "--codec-model", str(codec), "--tts", SYN_TEXT,
                  "--tts-output", str(wav), "--no-prints"], timeout=2400)
    ok_synth = p.returncode == 0 and wav.is_file() and wav.stat().st_size > 20000
    if not ok_synth:
        return False, {"stage": "synth", "exit": p.returncode,
                       "wav_kb": round(wav.stat().st_size / 1e3) if wav.is_file() else 0,
                       "tail": (p.stdout or p.stderr or "")[-400:]}
    w = _get_whisper()
    with kh.build_heartbeat("roundtrip"):
        pr = _run([str(CRISPASR), "-m", w, "-f", str(wav)], timeout=900)
    rt = " ".join(re.sub(r"\[[^\]]*\]", "", ln).strip() for ln in pr.stdout.splitlines())
    hits = sum(k in set(_norm(rt)) for k in SYN_KEYS)
    wav.unlink(missing_ok=True)
    return hits >= 3, {"stage": "roundtrip", "hits": hits, "transcript": rt[:200]}


def upload(path, repo, msg):
    step("upload.begin", repo=repo, file=Path(path).name,
         gb=round(Path(path).stat().st_size / 1e9, 2))
    HfApi(token=TOKEN).upload_file(
        path_or_fileobj=str(path), path_in_repo=Path(path).name,
        repo_id=repo, repo_type="model", commit_message=msg)
    step("upload.done", repo=repo, file=Path(path).name)


def do_model(repo, base, backend, quants):
    tag = base
    step(f"{tag}.begin", repo=repo, quants=quants, free_gb=free_gb())
    f16 = MODELS / f"{base}-f16.gguf"
    codec = MODELS / f"{base}-codec.gguf"
    with kh.build_heartbeat(f"{tag}.download"):
        hf_hub_download(repo_id=repo, filename=f"{base}-f16.gguf", local_dir=str(MODELS))
        hf_hub_download(repo_id=repo, filename=f"{base}-codec.gguf", local_dir=str(MODELS))
    step(f"{tag}.downloaded", f16_gb=round(f16.stat().st_size / 1e9, 2), free_gb=free_gb())
    produced = []
    for ftype in quants:
        out = MODELS / f"{base}-{ftype}.gguf"
        if not quantize(f16, out, ftype):
            step(f"{tag}.{ftype}.QUANT-FAILED")
            continue
        ok, info = validate_tts(out, codec, backend)
        step(f"{tag}.{ftype}.validate", ok=ok, **info)
        if not ok:
            step(f"{tag}.{ftype}.VALIDATION-FAILED", **info)
            out.unlink(missing_ok=True)
            continue
        upload(out, repo, f"Add {ftype} (re-quant of F16; TTS round-trip validated) (#249)")
        produced.append(f"{base}-{ftype}.gguf")
        out.unlink(missing_ok=True)
    f16.unlink(missing_ok=True)
    codec.unlink(missing_ok=True)
    step(f"{tag}.DONE", produced=produced, free_gb=free_gb())
    return produced


# %% [code]
RESULT = {}
for repo, base, backend, quants in [
    ("cstr/moss-tts-v1.5-GGUF", "moss-tts-v1.5", "moss-tts", ["q6_k", "q8_0"]),
    ("cstr/moss-tts-local-v1.5-GGUF", "moss-tts-local-v1.5", "moss-tts-local",
     ["q4_k", "q6_k", "q8_0"]),
]:
    try:
        RESULT[base] = do_model(repo, base, backend, quants)
    except Exception as exc:
        step(f"{base}.EXCEPTION", error=f"{type(exc).__name__}: {exc}")
        RESULT[base] = f"EXCEPTION: {exc}"

step("script.done", result=RESULT, free_gb=free_gb())
print("DONE", RESULT, flush=True)
