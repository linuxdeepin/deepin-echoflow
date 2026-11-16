#!/usr/bin/env python3
"""
Kaggle kernel: re-convert dots.tts core GGUF (tokenizer fix) + quantize Q8_0/Q4_K.

The first conversion stored tokenizer as GGUF string arrays which the C reader
can't handle. This version stores as newline-joined strings. Also quantizes
the core model to Q8_0 and Q4_K.
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

    # ── Install deps ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "gguf", "safetensors", "huggingface_hub"])

    # ── Build crispasr-quantize ──
    log("Installing build toolchain...")
    kh.install_build_toolchain()
    log("Building crispasr-quantize...")
    build_dir = _CRISPASR_DIR / "build"
    cmake_env = os.environ.copy()
    cmake_env["CCACHE_DIR"] = "/kaggle/working/.ccache"
    subprocess.run(["cmake", "-G", "Ninja", "-B", str(build_dir), "-S", str(_CRISPASR_DIR),
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_BUILD_TYPE=Release"], capture_output=True, env=cmake_env,
        cwd=str(_CRISPASR_DIR), timeout=120)
    n_jobs = kh.safe_build_jobs(gpu=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "crispasr-quantize",
        f"-j{n_jobs}"], capture_output=True, env=cmake_env, cwd=str(_CRISPASR_DIR), timeout=600)
    quantize_bin = build_dir / "bin" / "crispasr-quantize"
    log(f"crispasr-quantize: {'OK' if quantize_bin.exists() else 'MISSING'}")

    # ── HF token ──
    hf_token = kh.resolve_hf_token()

    # ── Download source model ──
    log("Downloading dots.tts-soar from HuggingFace...")
    from huggingface_hub import snapshot_download, HfApi, hf_hub_download
    model_dir = Path(snapshot_download("rednote-hilab/dots.tts-soar",
        cache_dir="/tmp/hf_cache", token=hf_token if hf_token else None))
    log(f"Model at: {model_dir}")

    # ── Re-convert core GGUF with fixed tokenizer ──
    convert_script = _CRISPASR_DIR / "models" / "convert-dots-tts-to-gguf.py"
    out_dir = WORK / "gguf_output"
    out_dir.mkdir(exist_ok=True)

    log("Converting core model (F16, fixed tokenizer)...")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(convert_script),
        "--model-dir", str(model_dir), "--output-dir", str(out_dir),
        "--name", "dots-tts-soar", "--quant", "f16",
        "--skip-vocoder", "--skip-speaker"],
        capture_output=True, text=True, timeout=3600)
    log(f"Convert rc={r.returncode}, {time.time()-t0:.0f}s")
    if r.stdout: log(f"stdout:\n{r.stdout[-1000:]}")
    if r.returncode != 0 and r.stderr: log(f"stderr:\n{r.stderr[-500:]}")

    f16_path = out_dir / "dots-tts-soar-f16.gguf"
    if f16_path.exists():
        log(f"Core F16: {f16_path.stat().st_size/(1024*1024):.1f} MB")
    else:
        log("ERROR: Core F16 not produced")
        sys.exit(1)

    # Free disk: delete HF cache (we have the GGUF now)
    import shutil
    shutil.rmtree("/tmp/hf_cache", ignore_errors=True)
    total, used, free = shutil.disk_usage(str(WORK))
    log(f"Disk after cache cleanup: {free/(1024**3):.1f} GB free")

    # ── Quantize core ──
    if quantize_bin.exists():
        for qt in ["q8_0", "q4_k"]:
            dst = out_dir / f"dots-tts-soar-{qt}.gguf"
            log(f"Quantizing core → {qt}...")
            t0 = time.time()
            r = subprocess.run([str(quantize_bin), str(f16_path), str(dst), qt],
                capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and dst.exists():
                log(f"  → {dst.name}: {dst.stat().st_size/(1024*1024):.1f} MB ({time.time()-t0:.1f}s)")
                for line in r.stdout.split('\n'):
                    if 'quantized' in line.lower(): log(f"  {line.strip()}")
            else:
                log(f"  FAILED rc={r.returncode}: {r.stderr[-300:]}")
                log(f"  stdout: {r.stdout[-300:]}")
            # Delete F16 after Q4_K to save disk for upload
            if qt == "q4_k" and f16_path.exists():
                pass  # Keep F16 for upload too

    # ── Upload all to HF ──
    log("\nUploading to HuggingFace...")
    api = HfApi(token=hf_token if hf_token else None)
    repo_id = "cstr/dots-tts-soar-GGUF"

    for f in sorted(out_dir.iterdir()):
        if f.suffix == ".gguf" and f.stat().st_size > 0:
            log(f"Uploading {f.name} ({f.stat().st_size/(1024*1024):.1f} MB)...")
            try:
                api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=repo_id, commit_message=f"Add {f.name} (fixed tokenizer)")
                log(f"  Uploaded {f.name}")
            except Exception as e:
                log(f"  Upload failed: {e}")

    log("\nDone!")

except Exception as e:
    log(f"\nFATAL ERROR: {e}")
    log(traceback.format_exc())
