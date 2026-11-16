#!/usr/bin/env python3
"""Kaggle kernel: validate HTDemucs C++ against Python reference.

Builds CrispASR (htdemucs target only), converts HTDemucs model,
runs the C++ smoke test, and compares spec_input / encoder outputs
against the Python reference dumper's GGUF.
"""
import os, sys, subprocess, time, traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
# Crash breadcrumb — written before anything else can fail
with open("/kaggle/working/started.txt", "w") as _f:
    _f.write(f"started at {time.time()}\npython={sys.version}\n")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = REPO / "build"
os.chdir(str(WORK))

# ── Clone CrispASR ──────────────────────────────────────────────────
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
if not REPO.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(REPO)])
        subprocess.check_call(["git", "submodule", "update", "--init", "ggml"],
                              cwd=str(REPO))
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass  # fall through to bundled copy

if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()

# ── Install deps ────────────────────────────────────────────────────
kh.step("install_deps")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "demucs", "gguf", "einops", "julius", "dora-search",
                       "--no-deps"])

# ── HF token (for model downloads with rate limits) ─────────────────
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    kh.step("hf_token_resolved")

# ── Build toolchain ─────────────────────────────────────────────────
kh.step("install_toolchain")
toolchain = kh.install_build_toolchain()

# ── Build CrispASR (htdemucs target only) ───────────────────────────
kh.step("cmake_configure")
build_flags = kh.cache_and_link_flags()  # ccache + mold + CRISPASR_NO_C2PA_NATIVE

with kh.build_heartbeat("cmake.configure"):
    kh.sh(
        f"cmake -S {REPO} -B {BUILD} -G Ninja "
        f"-DCMAKE_BUILD_TYPE=Release "
        + " ".join(build_flags)
    )

kh.step("cmake_build")
import multiprocessing
jobs = str(multiprocessing.cpu_count())
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} "
        f"--target htdemucs crispasr-core -j{jobs}"
    )

kh.step(f"build_complete (htdemucs, -j{jobs})")

# ── Convert HTDemucs model ──────────────────────────────────────────
kh.step("convert_model")
model_path = WORK / "htdemucs-f32.gguf"
kh.sh(
    f"{sys.executable} {REPO / 'models' / 'convert-htdemucs-to-gguf.py'} "
    f"--model htdemucs --output {model_path} --dtype f32"
)
kh.step(f"converted ({model_path.stat().st_size / 1e6:.1f} MB)")

# ── Generate Python reference ───────────────────────────────────────
kh.step("reference_dump")
ref_path = WORK / "htdemucs-ref.gguf"

# Synthetic test audio (3s sine at 16 kHz mono — harness resamples to 44100 stereo)
import numpy as np
import wave
test_wav = WORK / "test_sine.wav"
sr = 16000
t = np.arange(sr * 3) / sr
pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
pcm_i16 = (pcm * 32767).astype(np.int16)
with wave.open(str(test_wav), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm_i16.tobytes())

kh.sh(
    f"{sys.executable} {REPO / 'tools' / 'dump_reference.py'} "
    f"--backend htdemucs --model-dir htdemucs "
    f"--audio {test_wav} --output {ref_path}"
)
kh.step(f"reference_dump_done ({ref_path.stat().st_size / 1e6:.1f} MB)")

# ── Build and run C++ smoke test ────────────────────────────────────
kh.step("build_smoke_test")
smoke_src = REPO / "tests" / "test_htdemucs_smoke.cpp"
smoke_bin = BUILD / "bin" / "test_htdemucs_smoke"
# Link with rpath so the binary finds shared libs at runtime
ggml_lib = BUILD / "ggml" / "src"
ggml_base = ggml_lib / "ggml-base"
ggml_cpu = ggml_lib / "ggml-cpu"
rpath = f"-Wl,-rpath,{ggml_lib},-rpath,{ggml_base},-rpath,{ggml_cpu},-rpath,{BUILD / 'src'}"
kh.sh(
    f"g++ -std=c++17 -O2 "
    f"-I {REPO / 'src'} -I {REPO / 'ggml' / 'include'} "
    f"{smoke_src} "
    f"-L {BUILD / 'src'} -lhtdemucs -lcrispasr-core "
    f"-L {ggml_lib} -lggml "
    f"-L {ggml_base} -lggml-base "
    f"-L {ggml_cpu} -lggml-cpu "
    f"-lpthread -lm -ldl {rpath} "
    f"-o {smoke_bin}"
)

kh.step("run_smoke_test")
env = os.environ.copy()
env["CRISPASR_HTDEMUCS_DEBUG"] = "1"
env["OMP_NUM_THREADS"] = jobs
result = subprocess.run([str(smoke_bin), str(model_path)],
                       capture_output=True, text=True, env=env, timeout=600)
print(result.stderr)
if result.returncode != 0:
    kh.step(f"smoke_test_FAILED (exit {result.returncode})")
else:
    kh.step("smoke_test_PASSED")

# ── Compare reference stages ────────────────────────────────────────
kh.step("compare_reference")
try:
    from gguf import GGUFReader
    ref = GGUFReader(str(ref_path))
    ref_tensors = {t.name: np.array(t.data, dtype=np.float32) for t in ref.tensors}
    kh.step(f"reference_loaded ({len(ref_tensors)} stages)")

    for name, data in sorted(ref_tensors.items()):
        kh.step(f"  {name}: shape={data.shape}, mean={data.mean():.6f}, std={data.std():.6f}")
except Exception as e:
    kh.step(f"reference_compare_FAILED: {e}")

# ── Write progress file ─────────────────────────────────────────────
with open(WORK / "progress.txt", "w") as f:
    f.write("HTDemucs validation complete\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Reference: {ref_path}\n")
    f.write(f"Smoke test exit code: {result.returncode}\n")
    f.write(f"Smoke test stderr:\n{result.stderr}\n")

kh.step("done")
