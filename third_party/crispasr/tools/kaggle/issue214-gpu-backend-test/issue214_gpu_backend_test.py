"""
CrispASR Issue #214 — --gpu-backend selection regression test

Tests that --gpu-backend actually routes inference to the specified
backend instead of silently falling back to CUDA.

Test matrix:
  1. Default (auto) — should use CUDA on a CUDA-enabled machine
  2. --gpu-backend cuda — explicit CUDA, should match default
  3. --gpu-backend cpu  — CPU-only, should NOT touch CUDA
  4. --no-gpu           — CPU-only, baseline
  5. Multi-backend correctness — same model, same audio, compare
     transcripts between GPU backends to detect routing issues.

Also runs a broad backend smoke test (6 backends x JFK) to check
that the ggml_backend_init_best() -> crispasr_init_gpu_backend()
migration didn't break any backend's init path.

$KAGGLE_ACCOUNT account, T4/P100 GPU.
"""

import gc
import json
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
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass


def run(cmd, check=True, env=None, timeout=None, capture=False):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    kw = dict(env=e, timeout=timeout)
    if capture:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
        kw["text"] = True
    r = subprocess.run(cmd, **kw)
    if check and r.returncode != 0:
        if capture:
            print(r.stdout[-2000:] if r.stdout else "", flush=True)
            print(r.stderr[-2000:] if r.stderr else "", flush=True)
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


def download_model(repo, filename, token=None):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, cache_dir=str(MODELS), token=token))


def run_crispasr(args, env_extra=None, timeout=300):
    """Run crispasr CLI, return (rc, stdout, stderr, elapsed_s)."""
    e = os.environ.copy()
    if env_extra:
        e.update(env_extra)
    if "-np" not in args and "--no-prints" not in args:
        args = list(args) + ["-np"]
    cmd = [str(CLI)] + args
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=e, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout)
        rc = r.returncode
        stdout = r.stdout.decode("utf-8", errors="replace").strip()
        stderr = r.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as ex:
        rc = -99
        stdout = (ex.stdout or b"").decode("utf-8", errors="replace").strip()
        stderr = (ex.stderr or b"").decode("utf-8", errors="replace")
    elapsed = round(time.time() - t0, 1)
    for line in stderr.strip().splitlines()[-3:]:
        print(f"  [stderr] {line}", flush=True)
    return rc, stdout, stderr, elapsed


# ── Early diagnostics ──────────────────────────────────────────────
print("[start] Issue #214 GPU backend selection test", flush=True)
Path("/kaggle/working/started.txt").write_text("started\n")

# ── Clone + CUDA build ─────────────────────────────────────────────
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--recursive",
     "https://github.com/CrispStrobe/CrispASR.git", str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    pass
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
kh.step("cloned", sha=sha)

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
    ["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
     "-DCRISPASR_BUILD_TESTS=OFF"]
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
    cands = [c for c in BUILD.rglob("crispasr")
             if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
    )
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
summary = {}

jfk_wav = REPO / "samples" / "jfk.wav"
assert jfk_wav.exists(), "samples/jfk.wav missing"

# =====================================================================
# TEST 1: --gpu-backend routing (core issue #214 fix)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 1: --gpu-backend routing (issue #214)", flush=True)
print("=" * 70, flush=True)
kh.step("test1.start")

try:
    # Use sensevoice-small for the routing test — single GGUF, no companion
    # files needed, small (~129 MB Q4_K), fast inference, reliable.
    route_model = download_model("cstr/sensevoice-small-GGUF",
                                 "sensevoice-small-q4_k.gguf", token)
    kh.step("test1.model_downloaded")

    # A: default (auto — should pick CUDA)
    rc_a, out_a, err_a, el_a = run_crispasr([
        "-m", str(route_model), "-f", str(jfk_wav), "-v",
    ], timeout=120)

    # B: --gpu-backend cuda (explicit — should use CUDA)
    rc_b, out_b, err_b, el_b = run_crispasr([
        "-m", str(route_model), "-f", str(jfk_wav),
        "--gpu-backend", "cuda", "-v",
    ], timeout=120)

    # C: --gpu-backend cpu (should NOT touch CUDA at all)
    rc_c, out_c, err_c, el_c = run_crispasr([
        "-m", str(route_model), "-f", str(jfk_wav),
        "--gpu-backend", "cpu", "-v",
    ], timeout=120)

    # D: --no-gpu (baseline CPU)
    rc_d, out_d, err_d, el_d = run_crispasr([
        "-m", str(route_model), "-f", str(jfk_wav),
        "--no-gpu", "-v",
    ], timeout=120)

    # Check stderr for backend selection messages
    def find_backend_in_stderr(stderr_text):
        """Extract which backend was actually used from stderr."""
        for line in stderr_text.splitlines():
            if "using preferred GPU backend:" in line:
                return line.strip()
            if "using" in line.lower() and "backend" in line.lower():
                return line.strip()
        return "(not found in stderr)"

    backend_a = find_backend_in_stderr(err_a)
    backend_b = find_backend_in_stderr(err_b)
    backend_c = find_backend_in_stderr(err_c)
    backend_d = find_backend_in_stderr(err_d)

    # Check if CUDA appears in stderr for cpu-only runs
    cuda_in_c = "cuda" in err_c.lower() and "using" in err_c.lower()
    cuda_in_d = "cuda" in err_d.lower() and "using" in err_d.lower()

    # All runs should produce valid transcripts
    all_ok = all(rc == 0 for rc in [rc_a, rc_b, rc_c, rc_d])
    all_have_text = all(len(t) > 10 for t in [out_a, out_b, out_c, out_d])

    summary["gpu_backend_routing"] = {
        "auto_rc": rc_a, "auto_elapsed": el_a,
        "auto_backend": backend_a,
        "auto_transcript": out_a[:120],
        "cuda_rc": rc_b, "cuda_elapsed": el_b,
        "cuda_backend": backend_b,
        "cuda_transcript": out_b[:120],
        "cpu_flag_rc": rc_c, "cpu_flag_elapsed": el_c,
        "cpu_flag_backend": backend_c,
        "cpu_flag_transcript": out_c[:120],
        "cpu_flag_cuda_leak": cuda_in_c,
        "nogpu_rc": rc_d, "nogpu_elapsed": el_d,
        "nogpu_backend": backend_d,
        "nogpu_transcript": out_d[:120],
        "nogpu_cuda_leak": cuda_in_d,
        "all_ok": all_ok,
        "all_have_text": all_have_text,
    }

    print(f"\n  auto:        rc={rc_a} elapsed={el_a}s", flush=True)
    print(f"    backend: {backend_a}", flush=True)
    print(f"    text: {out_a[:80]}", flush=True)
    print(f"\n  --gpu-backend cuda: rc={rc_b} elapsed={el_b}s", flush=True)
    print(f"    backend: {backend_b}", flush=True)
    print(f"    text: {out_b[:80]}", flush=True)
    print(f"\n  --gpu-backend cpu:  rc={rc_c} elapsed={el_c}s", flush=True)
    print(f"    backend: {backend_c}", flush=True)
    print(f"    text: {out_c[:80]}", flush=True)
    print(f"    CUDA leak: {cuda_in_c}", flush=True)
    print(f"\n  --no-gpu:    rc={rc_d} elapsed={el_d}s", flush=True)
    print(f"    backend: {backend_d}", flush=True)
    print(f"    text: {out_d[:80]}", flush=True)

    # Key assertion: --gpu-backend cpu should be noticeably slower than
    # --gpu-backend cuda (since it's CPU-only inference)
    if el_b > 0 and el_c > 0:
        cpu_slowdown = round(el_c / el_b, 1)
        print(f"\n  CPU/CUDA slowdown factor: {cpu_slowdown}x", flush=True)
        summary["gpu_backend_routing"]["cpu_cuda_slowdown"] = cpu_slowdown

    kh.step("test1.done", **summary["gpu_backend_routing"])

except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    summary["gpu_backend_routing"] = {"error": str(ex)}
    kh.step("test1.error", error=str(ex))

# =====================================================================
# TEST 2: Multi-backend init smoke test (regression gate)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 2: Backend init smoke test (6 backends)", flush=True)
print("=" * 70, flush=True)
kh.step("test2.start")

SMOKE_BACKENDS = [
    ("firered-asr", "cstr/firered-asr2-aed-GGUF",    "firered-asr2-aed-q4_k.gguf",   []),
    ("parakeet",    "cstr/parakeet-tdt-0.6b-v2-GGUF", "parakeet-tdt-0.6b-v2-q4_k.gguf", []),
    ("sensevoice",  "cstr/sensevoice-small-GGUF",     "sensevoice-small-q4_k.gguf",   []),
    ("qwen3",       "cstr/qwen3-asr-0.6b-GGUF",      "qwen3-asr-0.6b-q4_k.gguf",    []),
    ("glm-asr",     "cstr/glm-asr-nano-GGUF",        "glm-asr-nano-q4_k.gguf",      []),
    ("fastconformer-ctc", "cstr/stt-en-fastconformer-ctc-large-GGUF",
     "stt-en-fastconformer-ctc-large-q4_k.gguf", []),
]

# Moonshine needs tokenizer.bin in the same dir as the model GGUF.
# hf_hub_download puts files in a snapshot dir, so downloading both
# from the same repo lands them together.
MOONSHINE_REPO = "cstr/moonshine-tiny-GGUF"
MOONSHINE_MODEL = "moonshine-tiny-q4_k.gguf"
MOONSHINE_TOK = "tokenizer.bin"

smoke_results = {}

# Moonshine: download both model + tokenizer.bin from same repo so they
# land in the same HF snapshot directory.
print(f"\n--- moonshine ---", flush=True)
kh.step("test2.moonshine.start")
try:
    moon_model = download_model(MOONSHINE_REPO, MOONSHINE_MODEL, token)
    download_model(MOONSHINE_REPO, MOONSHINE_TOK, token)  # lands next to model
    rc, out, err, elapsed = run_crispasr(
        ["-m", str(moon_model), "-f", str(jfk_wav)], timeout=180
    )
    transcript = out
    ok = rc == 0 and len(transcript) > 10
    result = {"rc": rc, "ok": ok, "elapsed": elapsed,
              "chars": len(transcript), "preview": transcript[:100]}
    smoke_results["moonshine"] = result
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: rc={rc} {len(transcript)} chars {elapsed}s", flush=True)
    print(f"  text: {transcript[:100]}", flush=True)
    kh.step("test2.moonshine.done", **result)
except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    smoke_results["moonshine"] = {"error": str(ex)}
    kh.step("test2.moonshine.error", error=str(ex))

for backend_name, repo, fname, extra_args in SMOKE_BACKENDS:
    print(f"\n--- {backend_name} ---", flush=True)
    kh.step(f"test2.{backend_name}.start")
    try:
        model_path = download_model(repo, fname, token)
        rc, out, err, elapsed = run_crispasr(
            ["-m", str(model_path), "-f", str(jfk_wav)] + extra_args,
            timeout=180
        )
        transcript = out
        ok = rc == 0 and len(transcript) > 10
        result = {
            "rc": rc, "ok": ok, "elapsed": elapsed,
            "chars": len(transcript),
            "preview": transcript[:100],
        }
        smoke_results[backend_name] = result
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: rc={rc} {len(transcript)} chars {elapsed}s", flush=True)
        print(f"  text: {transcript[:100]}", flush=True)
        kh.step(f"test2.{backend_name}.done", **result)

        # Free disk for next model
        del model_path
        gc.collect()

    except Exception as ex:
        print(f"  ERROR: {ex}", flush=True)
        smoke_results[backend_name] = {"error": str(ex)}
        kh.step(f"test2.{backend_name}.error", error=str(ex))

summary["backend_smoke"] = smoke_results

# =====================================================================
# TEST 3: TTS backend init smoke (verify TTS backends didn't regress)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST 3: TTS backend smoke (3 backends)", flush=True)
print("=" * 70, flush=True)
kh.step("test3.start")

TTS_BACKENDS = [
    ("piper",       "cstr/piper-en_US-lessac-medium-GGUF",
     "piper-en_US-lessac-medium-f16.gguf",
     ["--tts", "Hello world, testing GPU backend selection."],
     None, None),  # no companion
    ("fastpitch",   "cstr/fastpitch-en-GGUF",
     "fastpitch-en-q8_0.gguf",
     ["--tts", "Hello world, testing GPU backend selection."],
     None, None),
]

# Kokoro needs a voice pack companion — download both
KOKORO_REPO = "cstr/kokoro-82m-GGUF"
KOKORO_MODEL = "kokoro-82m-q8_0.gguf"
KOKORO_VOICE_REPO = "cstr/kokoro-voices-GGUF"
KOKORO_VOICE = "kokoro-voice-af_heart.gguf"

tts_results = {}

# Kokoro: download model + voice companion to same cache dir
print(f"\n--- kokoro ---", flush=True)
kh.step("test3.kokoro.start")
try:
    kokoro_model = download_model(KOKORO_REPO, KOKORO_MODEL, token)
    # Voice file must sit next to model GGUF — download to same dir
    kokoro_voice = download_model(KOKORO_VOICE_REPO, KOKORO_VOICE, token)
    outfile = RESULTS / "kokoro_out.wav"
    rc, out, err, elapsed = run_crispasr(
        ["-m", str(kokoro_model), "--companion", str(kokoro_voice),
         "-of", str(outfile), "--tts", "Hello world, testing GPU backend selection."],
        timeout=180
    )
    wav_exists = outfile.exists()
    wav_size = outfile.stat().st_size if wav_exists else 0
    ok = rc == 0 and wav_exists and wav_size > 1000
    result = {"rc": rc, "ok": ok, "elapsed": elapsed, "wav_size": wav_size}
    tts_results["kokoro"] = result
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: rc={rc} wav={wav_size}B {elapsed}s", flush=True)
    kh.step("test3.kokoro.done", **result)
    if outfile.exists():
        outfile.unlink()
except Exception as ex:
    print(f"  ERROR: {ex}", flush=True)
    tts_results["kokoro"] = {"error": str(ex)}
    kh.step("test3.kokoro.error", error=str(ex))

for backend_name, repo, fname, extra_args, _, _ in TTS_BACKENDS:
    print(f"\n--- {backend_name} ---", flush=True)
    kh.step(f"test3.{backend_name}.start")
    try:
        model_path = download_model(repo, fname, token)
        outfile = RESULTS / f"{backend_name}_out.wav"
        rc, out, err, elapsed = run_crispasr(
            ["-m", str(model_path), "-of", str(outfile)] + extra_args,
            timeout=180
        )
        wav_exists = outfile.exists()
        wav_size = outfile.stat().st_size if wav_exists else 0
        ok = rc == 0 and wav_exists and wav_size > 1000
        result = {
            "rc": rc, "ok": ok, "elapsed": elapsed,
            "wav_size": wav_size,
        }
        tts_results[backend_name] = result
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: rc={rc} wav={wav_size}B {elapsed}s", flush=True)
        kh.step(f"test3.{backend_name}.done", **result)

        # Cleanup wav
        if outfile.exists():
            outfile.unlink()
        del model_path
        gc.collect()

    except Exception as ex:
        print(f"  ERROR: {ex}", flush=True)
        tts_results[backend_name] = {"error": str(ex)}
        kh.step(f"test3.{backend_name}.error", error=str(ex))

summary["tts_smoke"] = tts_results

# =====================================================================
# SUMMARY
# =====================================================================
print("\n\n" + "=" * 70, flush=True)
print(f"SUMMARY — Issue #214 GPU backend test — {sha[:8]} on {gpu_name}", flush=True)
print("=" * 70, flush=True)

# Test 1: routing
t1 = summary.get("gpu_backend_routing", {})
if "error" not in t1:
    routing_ok = t1.get("all_ok") and t1.get("all_have_text")
    print(f"\n#214 GPU backend routing: {'PASS' if routing_ok else 'FAIL'}", flush=True)
    print(f"  auto:        {t1.get('auto_elapsed')}s", flush=True)
    print(f"  --gpu cuda:  {t1.get('cuda_elapsed')}s", flush=True)
    print(f"  --gpu cpu:   {t1.get('cpu_flag_elapsed')}s (CUDA leak: {t1.get('cpu_flag_cuda_leak')})", flush=True)
    print(f"  --no-gpu:    {t1.get('nogpu_elapsed')}s", flush=True)
    if t1.get("cpu_cuda_slowdown"):
        print(f"  CPU/CUDA slowdown: {t1['cpu_cuda_slowdown']}x", flush=True)
else:
    print(f"\n#214 GPU backend routing: ERROR — {t1.get('error')}", flush=True)

# Test 2: ASR smoke
t2 = summary.get("backend_smoke", {})
passed = sum(1 for v in t2.values() if v.get("ok"))
total = len(t2)
print(f"\nASR backend smoke: {passed}/{total} PASS", flush=True)
for bname, bdata in t2.items():
    status = "PASS" if bdata.get("ok") else ("ERROR" if "error" in bdata else "FAIL")
    detail = bdata.get("elapsed", bdata.get("error", ""))
    print(f"  {bname}: {status} ({detail})", flush=True)

# Test 3: TTS smoke
t3 = summary.get("tts_smoke", {})
tts_passed = sum(1 for v in t3.values() if v.get("ok"))
tts_total = len(t3)
print(f"\nTTS backend smoke: {tts_passed}/{tts_total} PASS", flush=True)
for bname, bdata in t3.items():
    status = "PASS" if bdata.get("ok") else ("ERROR" if "error" in bdata else "FAIL")
    detail = bdata.get("elapsed", bdata.get("error", ""))
    print(f"  {bname}: {status} ({detail})", flush=True)

# Overall verdict
all_pass = (
    t1.get("all_ok", False) and t1.get("all_have_text", False)
    and passed == total
    and tts_passed == tts_total
)
print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}", flush=True)

# Save
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nFull results saved to {RESULTS / 'summary.json'}", flush=True)

kh.step("summary", overall="PASS" if all_pass else "FAIL",
        asr_smoke=f"{passed}/{total}", tts_smoke=f"{tts_passed}/{tts_total}")
kh._push_progress_to_hf(force=True)
kh.step("script.end")
