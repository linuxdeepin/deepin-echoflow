#!/usr/bin/env python3
"""
Kaggle kernel: convert rednote-hilab/dots.tts-soar → GGUF for CrispASR.
"""

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

# Write progress immediately so we always have output
PROGRESS = WORK / "progress.txt"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")

log("Kernel started")

try:
    # ── Bootstrap CrispASR (outside /kaggle/working to avoid huge output download) ──
    CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
    _CRISPASR_DIR = Path("/tmp/CrispASR")  # /tmp is fine for code, not for large files
    if not _CRISPASR_DIR.exists():
        log("Cloning CrispASR...")
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(_CRISPASR_DIR)])
    sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))

    # Also try bundled fallback
    try:
        import kaggle_harness as kh
        kh.init_progress()
        log("kaggle_harness imported OK")
    except Exception as e:
        log(f"kaggle_harness import failed: {e}")

    # ── Install deps ──
    log("Installing gguf + safetensors + huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "gguf", "safetensors", "huggingface_hub"])
    log("Deps installed")

    # ── Get HF token ──
    try:
        hf_token = kh.resolve_hf_token()
    except Exception:
        hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        log(f"HF token: {hf_token[:8]}...")
    else:
        log("No HF token — dots.tts-soar is Apache-2.0 public, should be fine")

    # ── Download model ──
    log("Downloading rednote-hilab/dots.tts-soar...")
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(
        "rednote-hilab/dots.tts-soar",
        cache_dir="/tmp/hf_cache",  # /tmp for cache, keep /kaggle/working clean
        token=hf_token if hf_token else None,
    ))
    log(f"Model at: {model_dir}")

    # List files
    for f in sorted(model_dir.iterdir()):
        sz = f.stat().st_size / (1024*1024) if f.is_file() else 0
        log(f"  {f.name}: {sz:.1f} MB")

    # ── Check disk space ──
    import shutil
    total, used, free = shutil.disk_usage(str(WORK))
    log(f"Disk: {free / (1024**3):.1f} GB free of {total / (1024**3):.1f} GB")

    # ── Run conversion ──
    convert_script = _CRISPASR_DIR / "models" / "convert-dots-tts-to-gguf.py"
    out_dir = WORK / "gguf_output"
    out_dir.mkdir(exist_ok=True)

    log("Running GGUF conversion (F16)...")
    t0 = time.time()
    result = subprocess.run([
        sys.executable, str(convert_script),
        "--model-dir", str(model_dir),
        "--output-dir", str(out_dir),
        "--name", "dots-tts-soar",
        "--quant", "f16",
    ], capture_output=True, text=True, timeout=3600)

    log(f"stdout:\n{result.stdout}")
    if result.stderr:
        log(f"stderr:\n{result.stderr}")
    if result.returncode != 0:
        log(f"Conversion FAILED with exit code {result.returncode}")
    else:
        elapsed = time.time() - t0
        log(f"Conversion took {elapsed:.1f}s")

    # List output files
    log("Output files:")
    for f in sorted(out_dir.iterdir()):
        sz = f.stat().st_size / (1024*1024)
        log(f"  {f.name}: {sz:.1f} MB")

    # ── Inspect GGUF (list tensors) ──
    log("\n=== Core model tensors ===")
    try:
        from gguf import GGUFReader
        core_gguf = out_dir / "dots-tts-soar-f16.gguf"
        if core_gguf.exists():
            reader = GGUFReader(str(core_gguf))
            log(f"Tensors: {len(reader.tensors)}")
            for t in reader.tensors[:30]:
                log(f"  {t.name}: {list(t.shape)} {t.tensor_type.name}")
            if len(reader.tensors) > 30:
                log(f"  ... ({len(reader.tensors) - 30} more)")
    except Exception as e:
        log(f"GGUF inspection failed: {e}")

    # ── Upload to HuggingFace ──
    log("\nUploading to HuggingFace...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token if hf_token else None)
        repo_id = "cstr/dots-tts-soar-GGUF"

        try:
            api.create_repo(repo_id, exist_ok=True, repo_type="model")
        except Exception as e:
            log(f"Repo create: {e}")

        for f in sorted(out_dir.iterdir()):
            if f.suffix == ".gguf" and f.stat().st_size > 0:
                log(f"Uploading {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...")
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f.name,
                    repo_id=repo_id,
                    commit_message=f"Add {f.name}",
                )
                log(f"  Uploaded {f.name}")

        log(f"All files uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        log(f"HF upload failed: {e}")
        log("GGUF files are in kernel output — download manually")

    log("\nDone!")

except Exception as e:
    log(f"\nFATAL ERROR: {e}")
    log(traceback.format_exc())
