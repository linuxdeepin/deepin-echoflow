#!/usr/bin/env python3
"""
Kaggle kernel: quantize dots-tts-soar GGUF files (Q8_0 + Q4_K).

Downloads F16 GGUFs from cstr/dots-tts-soar-GGUF, builds crispasr-quantize
via kaggle_harness, runs quantization, and uploads results back to HF.
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
    CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
    _CRISPASR_DIR = Path("/tmp/CrispASR")
    if not _CRISPASR_DIR.exists():
        log("Cloning CrispASR...")
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(_CRISPASR_DIR)])
    sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))

    import kaggle_harness as kh
    kh.init_progress()
    log("kaggle_harness imported OK")

    # ── Install build toolchain + build crispasr-quantize ──
    log("Installing build toolchain...")
    kh.install_build_toolchain()

    log("Building crispasr-quantize...")
    build_dir = _CRISPASR_DIR / "build"

    # Simple CPU-only cmake configure + build
    os.makedirs(str(build_dir), exist_ok=True)
    cmake_env = os.environ.copy()
    cmake_env["CCACHE_DIR"] = "/kaggle/working/.ccache"

    r = subprocess.run([
        "cmake", "-G", "Ninja", "-B", str(build_dir), "-S", str(_CRISPASR_DIR),
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_BUILD_TYPE=Release",
    ], capture_output=True, text=True, env=cmake_env, cwd=str(_CRISPASR_DIR), timeout=120)
    log(f"CMake configure: rc={r.returncode}")
    if r.returncode != 0:
        log(f"CMake stderr (last 500): {r.stderr[-500:]}")

    n_jobs = kh.safe_build_jobs(gpu=True)
    r2 = subprocess.run([
        "cmake", "--build", str(build_dir),
        "--target", "crispasr-quantize",
        f"-j{n_jobs}",
    ], capture_output=True, text=True, env=cmake_env, cwd=str(_CRISPASR_DIR), timeout=600)
    log(f"Build: rc={r2.returncode}")
    if r2.returncode != 0:
        log(f"Build stderr (last 500): {r2.stderr[-500:]}")

    quantize_bin = build_dir / "bin" / "crispasr-quantize"
    if not quantize_bin.exists():
        log(f"ERROR: crispasr-quantize not found at {quantize_bin}")
        log("Listing bin/:")
        for f in (build_dir / "bin").iterdir() if (build_dir / "bin").exists() else []:
            log(f"  {f.name}")
        sys.exit(1)
    log(f"crispasr-quantize built OK")

    # ── Install HF deps ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "huggingface_hub"])

    # ── Get HF token ──
    hf_token = kh.resolve_hf_token()

    # ── Download F16 GGUFs from HF ──
    log("Downloading F16 GGUFs from cstr/dots-tts-soar-GGUF...")
    from huggingface_hub import hf_hub_download

    dl_dir = Path("/tmp/dots-tts-gguf")
    dl_dir.mkdir(exist_ok=True)

    files_to_quantize = [
        "dots-tts-soar-f16.gguf",
        "dots-tts-soar-vocoder-f16.gguf",
    ]

    for fname in files_to_quantize:
        log(f"Downloading {fname}...")
        hf_hub_download(
            "cstr/dots-tts-soar-GGUF",
            fname,
            local_dir=str(dl_dir),
            token=hf_token if hf_token else None,
        )
        fpath = dl_dir / fname
        log(f"  {fname}: {fpath.stat().st_size / (1024*1024):.1f} MB")

    # ── Quantize ──
    out_dir = WORK / "gguf_output"
    out_dir.mkdir(exist_ok=True)

    for quant_type in ["q8_0", "q4_k"]:
        for fname in files_to_quantize:
            src = dl_dir / fname
            dst_name = fname.replace("-f16.gguf", f"-{quant_type}.gguf")
            dst = out_dir / dst_name

            log(f"Quantizing {fname} → {dst_name} ({quant_type})...")
            t0 = time.time()
            result = subprocess.run(
                [str(quantize_bin), str(src), str(dst), quant_type],
                capture_output=True, text=True, timeout=600
            )
            elapsed = time.time() - t0

            if result.returncode != 0:
                log(f"  FAILED (rc={result.returncode}): {result.stderr[-500:]}")
                log(f"  stdout: {result.stdout[-500:]}")
            else:
                sz = dst.stat().st_size / (1024*1024)
                log(f"  → {dst_name}: {sz:.1f} MB ({elapsed:.1f}s)")
                for line in result.stdout.split('\n'):
                    if 'quantized' in line.lower() or 'total' in line.lower():
                        log(f"  {line.strip()}")

    # List all output files
    log("\nAll output files:")
    for f in sorted(out_dir.iterdir()):
        log(f"  {f.name}: {f.stat().st_size / (1024*1024):.1f} MB")

    # ── Upload to HuggingFace ──
    log("\nUploading to HuggingFace...")
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token if hf_token else None)
    repo_id = "cstr/dots-tts-soar-GGUF"

    for f in sorted(out_dir.iterdir()):
        if f.suffix == ".gguf" and f.stat().st_size > 0:
            log(f"Uploading {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...")
            try:
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=repo_id,
                    commit_message=f"Add {f.name}",
                )
                log(f"  Uploaded {f.name}")
            except Exception as e:
                log(f"  Upload failed: {e}")

    log("\nDone!")

except Exception as e:
    log(f"\nFATAL ERROR: {e}")
    log(traceback.format_exc())
