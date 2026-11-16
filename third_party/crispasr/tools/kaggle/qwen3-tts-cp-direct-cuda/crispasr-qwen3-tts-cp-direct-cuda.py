"""
CrispASR — Qwen3-TTS CP_DIRECT CUDA validation (PLAN §232, #245)

Tests the sched-free persistent code_pred / talker-bucket graphs on CUDA.
CP_DIRECT dispatches the code predictor through two gallocr-allocated
persistent graphs with no ggml_backend_sched involved, so the #56-class
sched-plan-reuse breakage (GGML_ASSERT in ggml_backend_tensor_set on
Jetson sm_87; nil-buffer inputs on Metal) cannot occur by construction.
This kernel verifies that claim on a Kaggle P100/T4 and A/Bs performance.

Matrix (all seed 42, same text):
  1. base      — no env (validated default path)
  2. o15       — QWEN3_TTS_O15=1 (the historical #56 crasher; context)
  3. direct    — QWEN3_TTS_CP_DIRECT=1
  4. direct_lk — QWEN3_TTS_CP_DIRECT=1 + QWEN3_TTS_LK_BUCKET=1

Acceptance: every run rc=0 + WAV md5 identical to base + ASR roundtrip
intelligible. Perf: ms/frame per config.

Build/report plumbing from the shared harness tools/kaggle/kaggle_harness.py.
enable_gpu=true in kernel-metadata.json.
"""

import hashlib
import os
import re
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
TTS_TEXT = "Please call Stella. Ask her to bring these things with her from the store."


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
import shutil
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
        c
        for c in BUILD.rglob("crispasr")
        if c.is_file() and os.access(c, os.X_OK)
    ]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))

# ── Download qwen3-tts model + tokenizer + parakeet for ASR roundtrip ──
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
    "cstr/qwen3-tts-0.6b-base-GGUF",
    "qwen3-tts-12hz-0.6b-base-q8_0.gguf",
    cache_dir=str(MODELS), token=token,
))
tts_codec = Path(hf_hub_download(
    "cstr/qwen3-tts-tokenizer-12hz-GGUF",
    "qwen3-tts-tokenizer-12hz.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF",
    "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")


# ── Run TTS under one env config ────────────────────────────────────
def run_tts(label, env_overrides, timeout=600):
    """Run qwen3-tts synthesis and return dict with results."""
    kh.step(f"{label}.start")
    out_wav = WORK / f"tts-{label}.wav"
    if out_wav.exists():
        out_wav.unlink()

    env = {"QWEN3_TTS_BENCH": "1"}
    env.update(env_overrides)

    # Use jfk.wav from the repo as voice reference (qwen3-tts requires 24kHz)
    voice_ref_16k = REPO / "samples" / "jfk.wav"
    voice_ref = WORK / "jfk_24k.wav"
    if not voice_ref.exists():
        try:
            import scipy.io.wavfile as swav
            from scipy.signal import resample_poly
            sr_in, data = swav.read(str(voice_ref_16k))
            if sr_in != 24000:
                data_24k = resample_poly(data.astype("float32"), 24000, sr_in)
                swav.write(str(voice_ref), 24000, data_24k.astype("int16"))
            else:
                shutil.copy(str(voice_ref_16k), str(voice_ref))
        except ImportError:
            subprocess.run(["ffmpeg", "-y", "-i", str(voice_ref_16k),
                            "-ar", "24000", str(voice_ref)],
                           capture_output=True, timeout=30)
    ref_text = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

    cmd = [
        str(CLI), "--backend", "qwen3-tts",
        "-m", str(tts_model),
        "--codec-model", str(tts_codec),
        "--voice", str(voice_ref),
        "--ref-text", ref_text,
        "--i-have-rights",
        "--tts", TTS_TEXT,
        "--tts-output", str(out_wav),
        "--seed", "42",
        "-v",
    ]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout,
        )
        rc, stdout, stderr = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as ex:
        rc = -1
        stdout = (
            (ex.stdout or b"").decode(errors="replace")
            if isinstance(ex.stdout, bytes)
            else (ex.stdout or "")
        )
        stderr = (
            (ex.stderr or b"").decode(errors="replace")
            if isinstance(ex.stderr, bytes)
            else (ex.stderr or "")
        )
    elapsed = round(time.time() - t0, 1)
    combined = stdout + "\n" + stderr
    (RESULTS / f"{label}_log.txt").write_text(combined)

    wav_exists = out_wav.exists() and out_wav.stat().st_size > 1000
    wav_size = out_wav.stat().st_size if out_wav.exists() else 0
    wav_md5 = (
        hashlib.md5(out_wav.read_bytes()).hexdigest() if wav_exists else None
    )

    bench_lines = [
        ln for ln in combined.splitlines()
        if "qwen3_tts:" in ln and ("ms" in ln or "bench" in ln.lower())
    ]

    ar_match = re.search(
        r"ar_loop\s+([\d.]+)\s+ms\s+\((\d+)\s+frames?,\s+([\d.]+)\s+ms/frame\)",
        combined,
    )
    ms_per_frame = float(ar_match.group(3)) if ar_match else None
    n_frames = int(ar_match.group(2)) if ar_match else None

    # Direct-path health markers
    direct_active = "cp_direct active" in combined
    fell_back = ("using sched path" in combined
                 or "cp_direct compute failed" in combined)

    print(f"\n{'='*64}", flush=True)
    print(
        f"Run: {label}  rc={rc}  elapsed={elapsed}s  "
        f"wav={'OK' if wav_exists else 'MISSING'}  size={wav_size}  "
        f"md5={wav_md5}",
        flush=True,
    )
    if ms_per_frame:
        print(f"  ar_loop: {ms_per_frame:.1f} ms/frame ({n_frames} frames)", flush=True)
    if direct_active:
        print("  cp_direct: ACTIVE", flush=True)
    if fell_back:
        print("  cp_direct/bucket: FELL BACK to sched path", flush=True)
    for bl in bench_lines[-8:]:
        print(f"  {bl.strip()}", flush=True)
    if rc != 0:
        print("  --- output tail ---", flush=True)
        for ln in combined.splitlines()[-30:]:
            print(f"   {ln}", flush=True)

    kh.step(
        f"{label}.done",
        rc=rc, elapsed=elapsed, wav_ok=wav_exists,
        wav_size=wav_size, ms_per_frame=ms_per_frame, md5=wav_md5,
    )
    return {
        "label": label, "rc": rc, "wav_ok": wav_exists,
        "wav_size": wav_size, "ms_per_frame": ms_per_frame,
        "md5": wav_md5, "wav_path": str(out_wav), "elapsed": elapsed,
        "direct_active": direct_active, "fell_back": fell_back,
    }


# v2 (post-3584fac0): CP_DIRECT and codec FASTCONV are now DEFAULT-ON for
# GPU backends — this run validates the shipped defaults against the full
# legacy path on CUDA. "base" pins both gates OFF; "direct" is the shipped
# default; "direct_lk" adds the opt-in talker bucketing (fastest on P100
# in v1). Output equivalence: md5, else PCM cosine (CPU showed a 1-LSB
# realization drift on the FASTCONV K=1 matmul; cos 1.00000000).
MATRIX = [
    ("base", {"QWEN3_TTS_CODEC_FASTCONV": "0", "QWEN3_TTS_CP_DIRECT": "0"}),
    ("direct", {}),
    ("direct_lk", {"QWEN3_TTS_LK_BUCKET": "1"}),
]
results = {}
for label, env_overrides in MATRIX:
    results[label] = run_tts(label, env_overrides)


# ── ASR roundtrip: transcribe base + direct WAVs with parakeet ──────
def asr_roundtrip(label, wav_path, timeout=120):
    kh.step(f"asr_{label}.start")
    out_stem = WORK / f"asr-{label}"
    cmd = [
        str(CLI), "--backend", "parakeet",
        "-m", str(asr_model),
        "-f", wav_path,
        "-of", str(out_stem), "-otxt",
        "--no-prints",
    ]
    try:
        subprocess.run(
            cmd, env=os.environ, capture_output=True, text=True, timeout=timeout
        )
        txt_path = out_stem.with_suffix(".txt")
        text = (
            txt_path.read_text().strip()
            if txt_path.exists() and txt_path.stat().st_size > 0
            else ""
        )
    except subprocess.TimeoutExpired:
        text = ""
    kh.step(f"asr_{label}.done", chars=len(text))
    return text


asr_base = (
    asr_roundtrip("base", results["base"]["wav_path"])
    if results["base"]["wav_ok"] else ""
)
asr_direct = (
    asr_roundtrip("direct", results["direct"]["wav_path"])
    if results["direct"]["wav_ok"] else ""
)

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 64, flush=True)
print(f"SUMMARY — qwen3-tts CP_DIRECT CUDA test — {sha[:8]} on {gpu_name}", flush=True)
print("=" * 64, flush=True)
base_md5 = results["base"]["md5"]
for label, _ in MATRIX:
    r = results[label]
    md5_ok = (r["md5"] == base_md5) if (r["md5"] and base_md5) else False
    print(
        f"  {label:10s} rc={r['rc']:3d}  wav={'OK ' if r['wav_ok'] else 'FAIL'}  "
        f"md5={'==base' if md5_ok else (r['md5'] or 'n/a')}  "
        f"{r['ms_per_frame'] or '?'} ms/frame  {r['elapsed']}s wall"
        f"{'  [fell back]' if r['fell_back'] else ''}",
        flush=True,
    )
print(f"  ASR roundtrip base:   {asr_base[:100]!r}", flush=True)
print(f"  ASR roundtrip direct: {asr_direct[:100]!r}", flush=True)

def pcm_cos(p1, p2):
    """Sample-wise cosine of two 16-bit WAVs (None on any failure)."""
    try:
        import wave
        import numpy as np
        def rd(p):
            w = wave.open(p)
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            w.close()
            return x.astype(np.float64)
        a, b = rd(p1), rd(p2)
        n = min(len(a), len(b))
        if n == 0 or len(a) != len(b):
            return 0.0
        a, b = a[:n], b[:n]
        return float(a @ b / ((a @ a) ** 0.5 * (b @ b) ** 0.5 + 1e-12))
    except Exception:
        return None


d = results["direct"]
same_md5 = bool(d["md5"]) and d["md5"] == base_md5
cos = None
if not same_md5 and d["wav_ok"] and results["base"]["wav_ok"]:
    cos = pcm_cos(results["base"]["wav_path"], d["wav_path"])
out_equiv = same_md5 or (cos is not None and cos > 0.99999)
if cos is not None:
    print(f"  PCM cos(base, direct) = {cos:.8f}", flush=True)
direct_ok = (
    d["rc"] == 0 and d["wav_ok"] and out_equiv
    and d["direct_active"] and not d["fell_back"]
)
direct_faster = (
    d["ms_per_frame"] is not None
    and results["base"]["ms_per_frame"] is not None
    and d["ms_per_frame"] < results["base"]["ms_per_frame"]
)

if direct_ok and direct_faster:
    b = results["base"]
    eq = "md5-identical" if same_md5 else f"PCM cos {cos:.8f}"
    print(f"\n  VERDICT: shipped defaults (CP_DIRECT + FASTCONV) WORK on CUDA "
          f"({gpu_name}), {eq}. "
          f"{b['ms_per_frame']:.1f} -> {d['ms_per_frame']:.1f} ms/frame "
          f"({(1 - d['ms_per_frame']/b['ms_per_frame'])*100:.0f}% faster).", flush=True)
elif direct_ok:
    print("\n  VERDICT: shipped defaults correct on CUDA (output equivalent) "
          "but no speedup here — check bench lines.", flush=True)
elif d["rc"] != 0:
    print(f"\n  VERDICT: shipped defaults CRASH on CUDA (rc={d['rc']}). "
          "REGRESSION — investigate before release.", flush=True)
elif d["fell_back"]:
    print("\n  VERDICT: CP_DIRECT fell back to the sched path on CUDA "
          "(unsupported op / placement). Output still valid; no CUDA win.", flush=True)
else:
    print("\n  VERDICT: shipped-defaults output mismatch on CUDA "
          f"(md5 differs, cos={cos}). REGRESSION — investigate.", flush=True)

kh.step(
    "summary",
    direct_ok=direct_ok, direct_faster=direct_faster, same_md5=same_md5,
    pcm_cos=cos,
    base_ms=results["base"]["ms_per_frame"], direct_ms=d["ms_per_frame"],
    base_md5=base_md5, direct_md5=d["md5"], sha=sha,
)
kh._push_progress_to_hf(force=True)
kh.step("script.end")
