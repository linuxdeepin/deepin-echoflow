#!/usr/bin/env python3
"""
Kaggle kernel: end-to-end dots.tts synthesis test on GPU.

Builds CrispASR with CUDA, downloads Q4_K GGUFs, runs synthesis,
validates output WAV.
"""

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

PROGRESS = WORK / "progress.txt"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")

log("Kernel started")

try:
    # ── Clone CrispASR ──
    _CRISPASR_DIR = Path("/tmp/CrispASR")
    if not _CRISPASR_DIR.exists():
        log("Cloning CrispASR...")
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(_CRISPASR_DIR)])
    sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))

    import kaggle_harness as kh
    kh.init_progress()
    log("kaggle_harness imported OK")

    # ── Build CrispASR with CUDA ──
    log("Installing build toolchain...")
    kh.install_build_toolchain()

    log("Detecting CUDA arch...")
    cuda_arch = kh.detect_cuda_arch()
    log(f"CUDA arch: {cuda_arch}")

    cmake_flags = kh.cuda_build_flags(cuda_arch)
    cache_flags = kh.cache_and_link_flags()
    n_jobs = kh.safe_build_jobs(gpu=True)

    build_dir = _CRISPASR_DIR / "build"
    cmake_env = os.environ.copy()
    cmake_env["CCACHE_DIR"] = "/kaggle/working/.ccache"

    # Configure
    log("CMake configure...")
    cmake_args = [
        "cmake", "-G", "Ninja", "-B", str(build_dir), "-S", str(_CRISPASR_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    # Add cache/link flags
    if isinstance(cache_flags, list):
        cmake_args.extend(cache_flags)
    elif isinstance(cache_flags, str) and cache_flags.strip():
        cmake_args.extend(cache_flags.strip().split())
    # Add CUDA flags
    if isinstance(cmake_flags, list):
        cmake_args.extend(cmake_flags)
    elif isinstance(cmake_flags, str) and cmake_flags.strip():
        cmake_args.extend(cmake_flags.strip().split())

    log(f"cmake args: {cmake_args}")
    r = subprocess.run(cmake_args, capture_output=True, text=True, env=cmake_env,
                       cwd=str(_CRISPASR_DIR), timeout=120)
    log(f"CMake rc={r.returncode}")
    if r.returncode != 0:
        log(f"stderr: {r.stderr[-800:]}")

    # Build just the shared library (not the full CLI — that takes 20+ min)
    log(f"Building crispasr-lib + dots-tts with {n_jobs} jobs...")
    with kh.build_heartbeat("dots-tts CUDA build"):
        r2 = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "crispasr-lib", f"-j{n_jobs}"],
            capture_output=True, text=True, env=cmake_env, cwd=str(_CRISPASR_DIR), timeout=1800)
    log(f"Build rc={r2.returncode}")
    if r2.returncode != 0:
        log(f"Build stderr (last 800): {r2.stderr[-800:]}")

    # Check that the shared library exists
    import glob
    libs = glob.glob(str(build_dir / "src" / "libcrispasr*"))
    log(f"Built libs: {libs}")
    if not libs:
        log("ERROR: libcrispasr not found")
        sys.exit(1)
    log("Library built OK")

    # ── Download GGUFs ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    hf_token = kh.resolve_hf_token()

    from huggingface_hub import hf_hub_download

    model_dir = WORK / "models"
    model_dir.mkdir(exist_ok=True)

    files = [
        "dots-tts-soar-q4_k.gguf",
        "dots-tts-soar-vocoder-f16.gguf",
    ]
    for fname in files:
        log(f"Downloading {fname}...")
        hf_hub_download("cstr/dots-tts-soar-GGUF", fname,
                       local_dir=str(model_dir), token=hf_token if hf_token else None)
        log(f"  {fname}: {(model_dir / fname).stat().st_size / (1024*1024):.1f} MB")

    # ── Check GPU ──
    log("GPU check:")
    os.system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")

    # ── Run synthesis via Python ctypes (no CLI needed) ──
    core_model = str(model_dir / "dots-tts-soar-q4_k.gguf")
    voc_model = str(model_dir / "dots-tts-soar-vocoder-f16.gguf")

    log("\n=== Synthesis via ctypes ===")
    try:
        import ctypes
        lib_path = glob.glob(str(build_dir / "src" / "libcrispasr.so*"))[0]
        log(f"Loading {lib_path}")
        lib = ctypes.CDLL(lib_path)

        # Use session API (correct names from crispasr_session.h)
        lib.crispasr_session_open_explicit.restype = ctypes.c_void_p
        lib.crispasr_session_open_explicit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_synthesize.restype = ctypes.POINTER(ctypes.c_float)
        lib.crispasr_session_synthesize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        lib.crispasr_session_close.argtypes = [ctypes.c_void_p]
        lib.crispasr_pcm_free.argtypes = [ctypes.POINTER(ctypes.c_float)]

        os.environ["DOTS_TTS_BENCH"] = "1"
        os.environ["CRISPASR_DOTS_TTS_DEBUG"] = "1"

        log(f"Opening session: {core_model}")
        t0 = time.time()
        sess = lib.crispasr_session_open_explicit(core_model.encode(), b"dots-tts", 4)
        if not sess:
            log("ERROR: session open failed")
        else:
            log(f"Session opened in {time.time()-t0:.1f}s")

            text = "Hello world."
            log(f"Synthesizing: '{text}'")
            t0 = time.time()
            n_samples = ctypes.c_int(0)
            pcm = lib.crispasr_session_synthesize(sess, text.encode(), ctypes.byref(n_samples))
            elapsed = time.time() - t0

            if pcm and n_samples.value > 0:
                log(f"Synthesis done: {n_samples.value} samples ({elapsed:.1f}s)")
                # Read first few samples
                samples = [pcm[i] for i in range(min(1000, n_samples.value))]
                max_val = max(abs(s) for s in samples)
                log(f"  Max amplitude (first 1000): {max_val:.6f}")
                if max_val > 0.001:
                    log("  ✓ Audio has non-trivial content")
                else:
                    log("  ✗ Audio appears silent")

                # Save as WAV
                import struct, wave
                output_wav = WORK / "dots_tts_output.wav"
                all_samples = [pcm[i] for i in range(n_samples.value)]
                with wave.open(str(output_wav), 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(48000)
                    pcm_i16 = b''.join(struct.pack('<h', max(-32768, min(32767, int(s * 32767)))) for s in all_samples)
                    wf.writeframes(pcm_i16)
                log(f"  Saved WAV: {output_wav} ({output_wav.stat().st_size / 1024:.1f} KB)")

                lib.crispasr_pcm_free(pcm)
            else:
                log(f"Synthesis returned null/empty ({elapsed:.1f}s)")

            lib.crispasr_session_close(sess)
    except Exception as e:
        log(f"Synthesis test failed: {e}")
        import traceback
        log(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Python reference dump (dots.tts upstream)
    # ══════════════════════════════════════════════════════════════════
    log("\n=== Phase 2: Python reference dump ===")
    try:
        log("Installing dots_tts Python package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/rednote-hilab/dots.tts.git",
                               "gguf"], timeout=300)

        ref_dir = WORK / "ref_dump"
        ref_dir.mkdir(exist_ok=True)

        # Run the reference backend
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "reference_backends"))

        ref_env = os.environ.copy()
        ref_env["DOTS_TEXT"] = "Hello world, this is a test."
        ref_env["DOTS_MAX_PATCHES"] = "2"
        ref_env["DOTS_SEED"] = "42"
        ref_env["DOTS_ODE_STEPS"] = "4"

        log("Running dots_tts reference dump...")
        r = subprocess.run([
            sys.executable, "-c",
            f"""
import os, sys, numpy as np
from pathlib import Path
os.environ['DOTS_TEXT'] = '{ref_env["DOTS_TEXT"]}'
os.environ['DOTS_MAX_PATCHES'] = '{ref_env["DOTS_MAX_PATCHES"]}'
os.environ['DOTS_SEED'] = '{ref_env["DOTS_SEED"]}'
os.environ['DOTS_ODE_STEPS'] = '{ref_env["DOTS_ODE_STEPS"]}'
sys.path.insert(0, '{_CRISPASR_DIR / "tools" / "reference_backends"}')
import dots_tts_reference as ref
results = ref.run(None, 0, Path('{ref_dir}'))
for k, v in results.items():
    print(f'  {{k}}: {{v.shape}} min={{v.min():.4f}} max={{v.max():.4f}}')
"""
        ], capture_output=True, text=True, timeout=600, env=ref_env)

        log(f"Reference dump rc={r.returncode}")
        log(f"stdout:\n{r.stdout[-2000:]}")
        if r.stderr:
            log(f"stderr:\n{r.stderr[-1000:]}")

        # List dumped files
        log("Reference files:")
        for f in sorted(ref_dir.iterdir()):
            log(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        log(f"Reference dump failed: {e}")
        log(traceback.format_exc())

    log("\nDone!")

except Exception as e:
    log(f"\nFATAL ERROR: {e}")
    log(traceback.format_exc())
