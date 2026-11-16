# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — higgs/ark beam-vs-greedy on NOISY audio (PLAN beam spot-check)
#
# Closes the PLAN checkbox: "Run higgs/ark -bs 4 on a noisy/accented clip to
# see if beam improves WER (only proven no-regression == greedy on easy JFK)".
# Both models breach the 2 GB VPS rule (higgs q4 2.4 GB, ark-3b q4 3.5 GB),
# so this runs on Kaggle.
#
# Design: jfk.wav + additive white noise at 10 dB and 5 dB SNR (deterministic,
# seed 42) — canonical text known, so WER is exact. Each backend x condition x
# {greedy, -bs 4}: transcript + wall time. Beam is only worth its cost if it
# beats greedy WER on the degraded conditions.

# ─────────────────────────── cell 1 (code) ───────────────────────────
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
_T0 = time.time()

def step(msg):
    print(f"\n{'='*60}\n[{time.time()-_T0:.0f}s] {msg}\n{'='*60}", flush=True)

# ─────────────────────────── cell 2 (code) ───────────────────────────
step("Clone + build CrispASR (CUDA)")
CRISPASR_DIR = WORK / "CrispASR"
if not CRISPASR_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", "--shallow-submodules",
                           "https://github.com/CrispStrobe/CrispASR.git", str(CRISPASR_DIR)])
sys.path.insert(0, str(CRISPASR_DIR / "tools" / "kaggle"))
import kaggle_harness as kh
kh.init_progress()
kh.install_build_toolchain()
build_dir = CRISPASR_DIR / "build"
os.makedirs(build_dir, exist_ok=True)
os.chdir(str(CRISPASR_DIR))
env = {**os.environ, "CCACHE_DIR": str(WORK / ".ccache")}
subprocess.check_call([
    "cmake", "-G", "Ninja", "-B", str(build_dir),
    "-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_NO_C2PA_NATIVE=ON", "-DGGML_CUDA=ON",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
], env=env)
with kh.build_heartbeat("cuda-build"):
    subprocess.check_call(["cmake", "--build", str(build_dir), "-j4",
                           "--target", "crispasr-cli"], env=env)
BIN = build_dir / "bin" / "crispasr"
assert BIN.is_file()

# ─────────────────────────── cell 3 (code) ───────────────────────────
step("Fetch models via CLI auto-download (registry)")
import numpy as np
MODELS = {}
for backend in ("higgs-stt", "ark-asr"):
    # Warm-up run downloads the registry model into the cache; the A/B matrix
    # then just passes -m auto (cache hit, no re-download).
    r = subprocess.run([str(BIN), "--backend", backend, "-m", "auto", "--auto-download",
                        "-f", str(CRISPASR_DIR / "samples" / "jfk.wav"), "-nt", "-l", "en"],
                       capture_output=True, text=True, timeout=3600)
    print(f"{backend} warmup rc={r.returncode} :: {r.stdout.strip()[-120:]}", flush=True)
    assert r.returncode == 0, f"{backend} warmup failed: {r.stderr[-400:]}"
    MODELS[backend] = "auto"

# ─────────────────────────── cell 4 (code) ───────────────────────────
step("Build noisy clips (seed 42)")
JFK = CRISPASR_DIR / "samples" / "jfk.wav"
CANON = ("and so my fellow americans ask not what your country can do for you "
         "ask what you can do for your country")

def add_noise(src, dst, snr_db):
    with wave.open(str(src)) as r:
        pcm = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32)
        params = r.getparams()
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(pcm)).astype(np.float32)
    sig_p = float(np.mean(pcm ** 2))
    noise = noise * np.sqrt(sig_p / (10 ** (snr_db / 10.0)) / float(np.mean(noise ** 2)))
    out = np.clip(pcm + noise, -32768, 32767).astype(np.int16)
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(out.tobytes())

CLIPS = {"clean": JFK}
for snr in (10, 5):
    p = WORK / f"jfk_snr{snr}.wav"
    add_noise(JFK, p, snr)
    CLIPS[f"snr{snr}"] = p

def norm(t):
    return re.sub(r"[^a-z' ]", "", t.lower()).split()

def wer(hyp, ref):
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return float(d[len(r), len(h)]) / max(1, len(r))

# ─────────────────────────── cell 5 (code) ───────────────────────────
step("A/B matrix")
RESULTS = {"meta": {"seed": 42, "canonical": CANON}}
for backend, model in MODELS.items():
    for clip_name, clip in CLIPS.items():
        for arm, extra in (("greedy", []), ("beam4", ["-bs", "4"])):
            cmd = [str(BIN), "--backend", backend, "-m", str(model),
                   "-f", str(clip), "-nt", "-l", "en"] + extra
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            dt = time.time() - t0
            text = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
            key = f"{backend}/{clip_name}/{arm}"
            RESULTS[key] = {"rc": r.returncode, "wall_s": round(dt, 1),
                            "wer": round(wer(text, CANON), 4), "text": text[:200]}
            print(f"{key}: rc={r.returncode} wer={RESULTS[key]['wer']} {dt:.0f}s :: {text[:80]}", flush=True)

# ─────────────────────────── cell 6 (code) ───────────────────────────
step("Verdict")
for b in MODELS:
    for c in CLIPS:
        g = RESULTS.get(f"{b}/{c}/greedy", {}).get("wer")
        bm = RESULTS.get(f"{b}/{c}/beam4", {}).get("wer")
        if g is not None and bm is not None:
            print(f"{b:10s} {c:6s}: greedy WER {g:.3f} vs beam4 {bm:.3f} -> "
                  f"{'BEAM WINS' if bm < g else 'no win' if bm == g else 'BEAM WORSE'}", flush=True)
(WORK / "beam_noisy_results.json").write_text(json.dumps(RESULTS, indent=1))
print("DONE", flush=True)
