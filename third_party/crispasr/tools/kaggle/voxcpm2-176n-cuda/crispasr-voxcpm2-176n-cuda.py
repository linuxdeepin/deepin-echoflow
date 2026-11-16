"""
CrispASR — VoxCPM2 GPU (CUDA) validation + CPU A/B (PLAN §176n)

§176n was closed on M1 Metal: VoxCPM2 already runs on the GPU via the
VOXCPM2_USE_GRAPH fused-graph path (default ON for a GPU backend), verified
3.75x faster than CPU with a correct TTS->ASR round-trip. Metal uses unified
memory; a discrete GPU (CUDA) instead takes the `needs_gpu_mirror` path
(device-local VRAM + a GPU weight mirror). That path was NOT exercised for
correctness, and CUDA has stricter per-op contiguity asserts than Metal
(LEARNING 35), so a Metal-clean GPU default can still crash / mis-decode on
CUDA. This kernel closes that gap on a Kaggle P100/T4.

Matrix (same text + seed):
  1. cpu — --no-gpu (trusted baseline audio)
  2. gpu — default (VOXCPM2_USE_GRAPH=1 on CUDA; the shipped GPU path)

Acceptance:
  * both rc=0 and produce a non-trivial WAV
  * GPU log shows the GPU backend + weight mirror (not a silent CPU fallback)
  * BOTH round-trip through parakeet ASR to the input text (word overlap) —
    the only real correctness test for TTS (component cos means nothing here)
  * report GPU-vs-CPU synth speedup (informational; not an acceptance gate)

Build/report plumbing from the shared harness tools/kaggle/kaggle_harness.py.
enable_gpu=true in kernel-metadata.json.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)
# Short, phonetically-clean sentence so the ASR round-trip is unambiguous.
TTS_TEXT = "And so my fellow Americans, ask not what your country can do for you."


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build ──────────────────────────────────────────────
print(f"[start] ref={CRISPASR_REF}", flush=True)
print(f"  disk: {shutil.disk_usage('/kaggle/working')}", flush=True)
Path("/kaggle/working/started.txt").write_text("started\n")

if REPO.exists():
    shutil.rmtree(REPO)
run(
    [
        "git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
        "--recursive", CRISPASR_REPO, str(REPO),
    ]
)

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
try:
    import kaggle_harness as kh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).strip()
kh.step("gpu", gpu_name=gpu_name)

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    [
        "cmake", "-S", str(REPO), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DCRISPASR_BUILD_TESTS=OFF",
    ]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli"
        f" -j{kh.safe_build_jobs(gpu=True)}"
    )

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [
        c for c in BUILD.rglob("crispasr")
        if c.is_file() and os.access(c, os.X_OK)
    ]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))

# ── Download voxcpm2 model + parakeet (ASR round-trip) ──────────────
kh.step("downloading models")
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
    )
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

tts_model = Path(hf_hub_download(
    "cstr/voxcpm2-GGUF", "voxcpm2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF", "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")


# ── Run one VoxCPM2 synthesis config ────────────────────────────────
def run_tts(label, extra_args, env_overrides=None, timeout=900):
    kh.step(f"{label}.start")
    out_wav = WORK / f"tts-{label}.wav"
    if out_wav.exists():
        out_wav.unlink()
    env = {"VOXCPM2_TIMING": "1"}
    if env_overrides:
        env.update(env_overrides)
    cmd = [
        str(CLI), "--backend", "voxcpm2-tts",
        "-m", str(tts_model),
        "--tts", TTS_TEXT,
        "--tts-output", str(out_wav),
        "--seed", "42", "-v",
    ] + extra_args
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        rc, combined = r.returncode, (r.stdout + "\n" + r.stderr)
    except subprocess.TimeoutExpired as ex:
        rc = -1
        so = ex.stdout or ""
        se = ex.stderr or ""
        combined = (so.decode(errors="replace") if isinstance(so, bytes) else so) + "\n" + (
            se.decode(errors="replace") if isinstance(se, bytes) else se
        )
    elapsed = round(time.time() - t0, 1)
    (RESULTS / f"{label}_log.txt").write_text(combined)

    wav_ok = out_wav.exists() and out_wav.stat().st_size > 1000
    wav_size = out_wav.stat().st_size if out_wav.exists() else 0
    wav_md5 = hashlib.md5(out_wav.read_bytes()).hexdigest() if wav_ok else None

    # Health markers + synth timing.
    gpu_mirror = "GPU weight mirror" in combined
    backend_gpu = bool(re.search(r"backend\s*=\s*(?!CPU)", combined)) or "MTL0" in combined or gpu_mirror
    m_total = re.search(r"voxcpm2:\s*total\s+([\d.]+)\s*ms", combined)
    synth_ms = float(m_total.group(1)) if m_total else None
    crashed = any(s in combined for s in (
        "Segmentation fault", "GGML_ASSERT", "CUDA error", "illegal memory",
        "unsupported op", "NaN",
    ))

    print(f"\n{'='*64}", flush=True)
    print(f"Run: {label}  rc={rc}  elapsed={elapsed}s  "
          f"wav={'OK' if wav_ok else 'MISSING'}  size={wav_size}  md5={wav_md5}",
          flush=True)
    if synth_ms:
        print(f"  synth total: {synth_ms:.0f} ms", flush=True)
    print(f"  gpu_mirror={gpu_mirror}  backend_gpu={backend_gpu}  crashed={crashed}",
          flush=True)
    if rc != 0 or crashed:
        print("  --- output tail ---", flush=True)
        for ln in combined.splitlines()[-30:]:
            print(f"   {ln}", flush=True)
    kh.step(f"{label}.done", rc=rc, elapsed=elapsed, wav_ok=wav_ok,
            wav_size=wav_size, md5=wav_md5, synth_ms=synth_ms, crashed=crashed)
    return {
        "label": label, "rc": rc, "wav_ok": wav_ok, "wav_size": wav_size,
        "md5": wav_md5, "synth_ms": synth_ms, "crashed": crashed,
        "gpu_mirror": gpu_mirror, "backend_gpu": backend_gpu,
        "wav_path": str(out_wav),
    }


MATRIX = [
    ("cpu", ["--no-gpu"], None),
    ("gpu", [], None),  # default: VOXCPM2_USE_GRAPH=1 on CUDA
]
results = {}
for label, extra, env in MATRIX:
    results[label] = run_tts(label, extra, env)


# ── ASR round-trip (parakeet) — the only real correctness test ──────
def normalize(s):
    return re.sub(r"[^a-z ]", "", s.lower()).split()


def asr_roundtrip(label, wav_path, timeout=180):
    kh.step(f"asr_{label}.start")
    out_stem = WORK / f"asr-{label}"
    try:
        subprocess.run(
            [str(CLI), "--backend", "parakeet", "-m", str(asr_model),
             "-f", wav_path, "-of", str(out_stem), "-otxt", "--no-prints"],
            env=os.environ, capture_output=True, text=True, timeout=timeout,
        )
        txt = out_stem.with_suffix(".txt")
        text = txt.read_text().strip() if txt.exists() and txt.stat().st_size else ""
    except subprocess.TimeoutExpired:
        text = ""
    kh.step(f"asr_{label}.done", chars=len(text))
    return text


ref_words = set(normalize(TTS_TEXT))
asr = {}
for label in ("cpu", "gpu"):
    r = results[label]
    text = asr_roundtrip(label, r["wav_path"]) if r["wav_ok"] else ""
    hit = set(normalize(text)) & ref_words
    recall = len(hit) / max(1, len(ref_words))
    asr[label] = {"text": text, "recall": round(recall, 2)}


# ── Summary + acceptance ────────────────────────────────────────────
print("\n" + "=" * 64, flush=True)
print(f"SUMMARY — VoxCPM2 §176n CUDA validation — {sha[:8]} on {gpu_name}", flush=True)
print("=" * 64, flush=True)
for label in ("cpu", "gpu"):
    r = results[label]
    a = asr[label]
    print(f"  {label:4s}  rc={r['rc']}  wav={'OK' if r['wav_ok'] else 'X'}  "
          f"synth={r['synth_ms']}ms  gpu_mirror={r['gpu_mirror']}  "
          f"recall={a['recall']}  asr='{a['text'][:70]}'", flush=True)

speedup = None
if results["cpu"]["synth_ms"] and results["gpu"]["synth_ms"]:
    speedup = round(results["cpu"]["synth_ms"] / results["gpu"]["synth_ms"], 2)
    print(f"  GPU speedup (synth): {speedup}x", flush=True)

# Acceptance: GPU must not crash, must produce a WAV, must round-trip
# intelligibly (recall >= 0.6 like the CPU baseline), and must actually run
# on the GPU (mirror created — not a silent CPU fallback).
ACCEPT_RECALL = 0.6
gpu_ok = (
    results["gpu"]["rc"] == 0
    and results["gpu"]["wav_ok"]
    and not results["gpu"]["crashed"]
    and results["gpu"]["gpu_mirror"]
    and asr["gpu"]["recall"] >= ACCEPT_RECALL
)
cpu_ok = (
    results["cpu"]["rc"] == 0
    and results["cpu"]["wav_ok"]
    and asr["cpu"]["recall"] >= ACCEPT_RECALL
)
verdict = "PASS" if (gpu_ok and cpu_ok) else "FAIL"
print(f"\n  VERDICT: {verdict}  (cpu_ok={cpu_ok}  gpu_ok={gpu_ok}  speedup={speedup}x)",
      flush=True)
kh.step("verdict", verdict=verdict, cpu_ok=cpu_ok, gpu_ok=gpu_ok, speedup=speedup)
if verdict != "PASS":
    raise SystemExit(f"VoxCPM2 §176n CUDA validation FAILED: {verdict}")
print("\n§176n CUDA validation PASS — GPU path is correct + intelligible on "
      f"{gpu_name}.", flush=True)
