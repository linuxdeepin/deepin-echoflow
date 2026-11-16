"""
CrispASR — importance-matrix (imatrix) quantization + HF upload.

For each target ASR-LLM decoder model:
  1. download the F16 GGUF,
  2. calibrate an imatrix by running the CC0 Common Voice EN+DE clips through
     the model with CRISPASR_IMATRIX_OUT set,
  3. quantize baseline + imatrix at q4_k and q3_k,
  4. validate: transcribe held-out clips, report CER of baseline vs imatrix
     against the F16 gold (imatrix should not worsen; helps most at q3_k),
  5. upload the imatrix quants (+ the imatrix file) to the model's HF repo.

GPU kernel: GPU workers get internet (needed for HF), and CUDA accelerates the
calibration passes. Big files stage under /tmp (~70 GB), not /kaggle/working
(~20 GB cap). See kaggle_usage.md.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
STAGE = Path("/tmp/imx")          # big files live here (ephemeral ~70 GB layer)
REPO = WORK / "CrispASR"
BUILD = REPO / "build"
BRANCH = os.environ.get("CRISPASR_REF", "main")
STAGE.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    (WORK / "status.txt").write_text(msg + "\n")


# ── 1. clone + harness ──────────────────────────────────────────────────────
log(f"[1/4] Cloning CrispASR ({BRANCH})")
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, "--recursive",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], timeout=180)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
from huggingface_hub import hf_hub_download, snapshot_download, HfApi  # noqa: E402
api = HfApi(token=hf_token)
kh.step("cloned")

# ── 2. CUDA build (crispasr for the imatrix producer + crispasr-quantize) ────
log("[2/4] Building crispasr + crispasr-quantize (CUDA)")
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
cfg = ["cmake", "-G", "Ninja", "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"] \
    + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
subprocess.check_call(cfg, cwd=str(REPO), timeout=180)
with kh.build_heartbeat("cmake.build"):
    subprocess.check_call(
        ["cmake", "--build", str(BUILD), "--target", "crispasr", "crispasr-quantize",
         f"-j{kh.safe_build_jobs(gpu=True)}"], cwd=str(REPO), timeout=3600)
CRISPASR = str(BUILD / "bin" / "crispasr")
QUANTIZE = str(BUILD / "bin" / "crispasr-quantize")
assert Path(CRISPASR).exists() and Path(QUANTIZE).exists(), "build produced no binaries"
kh.step("built")

# ── 3. calibration corpus (CC0 Common Voice EN+DE) ──────────────────────────
log("[3/4] Downloading CC0 calibration set")
calib_dir = STAGE / "calib"
snapshot_download("cstr/crispasr-imatrix-calib", repo_type="dataset",
                  local_dir=str(calib_dir), allow_patterns=["*.mp3"])
en = sorted((calib_dir / "en").glob("*.mp3"))
de = sorted((calib_dir / "de").glob("*.mp3"))
CALIB = [str(p) for p in (en[:8] + de[:8])]           # 8 EN + 8 DE
VAL = [str(p) for p in (en[8:10] + de[8:10])]         # held out for CER
log(f"  calib={len(CALIB)} val={len(VAL)}")
kh.step("corpus")


def cer(hyp, ref):
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
        prev = cur
    return prev[n] / m


def transcribe(model, wav, timeout=600):
    r = subprocess.run([CRISPASR, "-m", model, "-f", wav],
                       capture_output=True, text=True, timeout=timeout)
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ── 4. per-model: calibrate → quantize → validate → upload ──────────────────
MODELS = [
    # small → large, so early wins bank before any timeout
    dict(repo="cstr/qwen3-asr-0.6b-GGUF", src="qwen3-asr-0.6b.gguf", prefix="qwen3-asr-0.6b"),
    dict(repo="cstr/mega-asr-GGUF", src="mega-asr-1.7b-f16.gguf", prefix="mega-asr-1.7b"),
    dict(repo="cstr/MOSS-Transcribe-preview-2B-GGUF", src="moss-transcribe-preview-2b-f16.gguf",
         prefix="moss-transcribe-preview-2b"),
    dict(repo="cstr/higgs-audio-v3-stt-GGUF", src="higgs-stt-f16.gguf", prefix="higgs-stt"),
    dict(repo="cstr/ark-asr-3b-GGUF", src="ark-asr-3b-f16.gguf", prefix="ark-asr-3b"),
]
QTYPES = ["q4_k", "q3_k"]
summary = []

for mi, M in enumerate(MODELS):
    tag = M["prefix"]
    log(f"[4/4] ({mi + 1}/{len(MODELS)}) {tag}")
    mdir = STAGE / tag
    mdir.mkdir(exist_ok=True)
    try:
        # download F16 source
        f16 = mdir / M["src"]
        p = hf_hub_download(M["repo"], M["src"], cache_dir=str(STAGE / "hfcache"))
        shutil.copy2(p, f16)
        print(f"  src {f16.stat().st_size / 1e6:.0f} MB", flush=True)

        # gold transcripts (F16) for the val clips
        gold = {}
        for w in VAL:
            gold[w] = transcribe(str(f16), w)

        # calibrate imatrix
        imat = mdir / f"{tag}-en-de.imatrix.gguf"
        if imat.exists():
            imat.unlink()
        t0 = time.time()
        for ci, w in enumerate(CALIB):
            env = dict(os.environ, CRISPASR_IMATRIX_OUT=str(imat))
            subprocess.run([CRISPASR, "-m", str(f16), "-f", w], env=env,
                           capture_output=True, text=True, timeout=600)
        ok_imat = imat.exists() and imat.stat().st_size > 4096
        print(f"  imatrix {'OK' if ok_imat else 'MISSING'} "
              f"({imat.stat().st_size // 1024 if imat.exists() else 0} KB, {time.time() - t0:.0f}s)",
              flush=True)
        if not ok_imat:
            summary.append(f"{tag}: SKIP (no imatrix — producer failed on this backend/CUDA)")
            continue

        # upload the imatrix file (reproducibility)
        api.upload_file(path_or_fileobj=str(imat), path_in_repo=imat.name,
                        repo_id=M["repo"], repo_type="model",
                        commit_message="Add importance matrix (CC0 Common Voice EN+DE)")

        for q in QTYPES:
            base = mdir / f"{tag}-{q}-base.gguf"       # baseline (not uploaded)
            imx = mdir / f"{tag}-{q}-imatrix.gguf"      # imatrix (uploaded)
            for out, extra in ((base, []), (imx, ["--imatrix", str(imat)])):
                if out.exists():
                    out.unlink()
                subprocess.run([QUANTIZE, str(f16), str(out), q] + extra,
                               capture_output=True, text=True, timeout=1200)
            if not imx.exists():
                summary.append(f"{tag} {q}: quantize FAILED")
                continue
            # A/B: CER of baseline vs imatrix against the F16 gold
            ca = cb = 0.0
            for w in VAL:
                if base.exists():
                    ca += cer(transcribe(str(base), w), gold[w])
                cb += cer(transcribe(str(imx), w), gold[w])
            ca /= max(1, len(VAL))
            cb /= max(1, len(VAL))
            sz = imx.stat().st_size / 1e6
            note = f"{tag} {q}: {sz:.0f}MB  CER base={ca:.3f} imatrix={cb:.3f} d={cb - ca:+.3f}"
            print("  " + note, flush=True)
            # sanity gate: don't publish garbage (CER≈1 means broken)
            if cb > 0.8:
                summary.append(note + "  → NOT UPLOADED (garbage)")
            else:
                api.upload_file(path_or_fileobj=str(imx), path_in_repo=imx.name,
                                repo_id=M["repo"], repo_type="model",
                                commit_message=f"Add {q} imatrix quant (CC0 Common Voice EN+DE calibration)")
                summary.append(note + "  → uploaded")
            if base.exists():
                base.unlink()
    except Exception as e:
        summary.append(f"{tag}: ERROR {repr(e)[:160]}")
        print(f"  ERROR: {e}", flush=True)
    finally:
        shutil.rmtree(mdir, ignore_errors=True)   # free disk before next model
    kh.step(f"done:{tag}")

# ── summary ─────────────────────────────────────────────────────────────────
report = "\n".join(summary)
print("\n===== IMATRIX QUANT SUMMARY =====\n" + report, flush=True)
(WORK / "summary.txt").write_text(report + "\n")
(WORK / "status.txt").write_text("DONE\n")
