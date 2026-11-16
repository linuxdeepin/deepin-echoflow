#!/usr/bin/env python3
"""Kaggle kernel: convert KugelAudio-0-Open → F16 GGUF → quantize → upload to HF.

CPU-only (30 GB RAM is enough for the 18.7 GB model).
Produces kugelaudio-0-open-f16.gguf and kugelaudio-0-open-q4_k.gguf,
uploads both to cstr/kugelaudio-0-open-GGUF on HuggingFace.

Push: python -m kaggle kernels push -p tools/kaggle/kugelaudio-convert
"""

import os, sys, subprocess, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
# Use /kaggle/temp for large artifacts — /kaggle/working is capped at ~20 GB
# and gets saved as kernel output. We upload to HF directly from temp.
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BUILD = TEMP / "build"

# ── Phase 0: Clone repo (harness lives in it) ──────────────────────────────
print("=== Phase 0: clone repo ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "feature/kugelaudio-tts",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])

# Import harness AFTER clone
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()

# ── Phase 1: Install deps ──────────────────────────────────────────────────
kh.step("install deps")
kh.install_build_toolchain()  # ninja + ccache + mold
kh.sh_with_progress("pip install -q torch transformers safetensors gguf huggingface_hub hf_transfer")

# ── Phase 2: Resolve HF token ──────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("  HF_TOKEN resolved OK")
else:
    print("  WARNING: no HF_TOKEN — will stage for local pickup")

# ── Phase 3: Download source model ─────────────────────────────────────────
kh.step("download model")
from huggingface_hub import snapshot_download

# Use /kaggle/temp for source cache (not /kaggle/working — capped at ~20 GB)
for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "kugelaudio-src"
        break
scratch.mkdir(parents=True, exist_ok=True)
free = kh.free_gb(str(scratch))
print(f"  source cache: {scratch} (free: {free:.1f} GiB)" if free else f"  source cache: {scratch}")

src = snapshot_download(
    repo_id="kugelaudio/kugelaudio-0-open",
    cache_dir=str(scratch),
    token=hf_token,
)
print(f"  source dir: {src}")

# ── Phase 4: Convert to F16 GGUF ───────────────────────────────────────────
kh.step("convert F16 GGUF")
f16_path = TEMP / "kugelaudio-0-open-f16.gguf"

kh.sh_with_progress(
    f"python models/convert-kugelaudio-to-gguf.py "
    f"--input {src} --output {f16_path} --no-encoders --type f16",
    cwd=str(REPO),
)
print(f"  F16 GGUF: {f16_path} ({f16_path.stat().st_size / (1024**3):.1f} GiB)")

# ── Phase 5: Upload F16 to HF (before quantize — free disk after) ──────────
kh.step("upload F16 to HF")
HF_REPO = "cstr/kugelaudio-0-open-GGUF"

if hf_token:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  repo create: {e}")

    print(f"  uploading F16 ({f16_path.stat().st_size / (1024**3):.1f} GiB)...")
    api.upload_file(
        path_or_fileobj=str(f16_path),
        path_in_repo="kugelaudio-0-open-f16.gguf",
        repo_id=HF_REPO, repo_type="model",
        commit_message="Add F16 GGUF (KugelAudio-0-Open TTS, --no-encoders)",
    )
    print("  uploaded F16")
else:
    print("  no HF_TOKEN — skipping upload")

# ── Phase 6: Build crispasr-quantize ────────────────────────────────────────
kh.step("build quantizer")
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
    + " ".join(flags),
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-quantize")

quantize_bin = BUILD / "bin" / "crispasr-quantize"
print(f"  quantizer: {quantize_bin}")

# ── Phase 7: Quantize F16 → Q4_K ───────────────────────────────────────────
kh.step("quantize Q4_K")
q4k_path = TEMP / "kugelaudio-0-open-q4_k.gguf"

kh.sh_with_progress(f"{quantize_bin} {f16_path} {q4k_path} q4_k")
print(f"  Q4_K GGUF: {q4k_path} ({q4k_path.stat().st_size / (1024**3):.1f} GiB)")

# Delete F16 to free disk
f16_path.unlink(missing_ok=True)
print("  deleted F16 (uploaded already)")

# ── Phase 8: Upload Q4_K to HF ─────────────────────────────────────────────
kh.step("upload Q4_K to HF")
if hf_token:
    print(f"  uploading Q4_K ({q4k_path.stat().st_size / (1024**3):.1f} GiB)...")
    api.upload_file(
        path_or_fileobj=str(q4k_path),
        path_in_repo="kugelaudio-0-open-q4_k.gguf",
        repo_id=HF_REPO, repo_type="model",
        commit_message="Add Q4_K GGUF (KugelAudio-0-Open TTS, --no-encoders)",
    )
    print("  uploaded Q4_K")
    print(f"\n  All uploaded to https://huggingface.co/{HF_REPO}")
else:
    print("  no HF_TOKEN — staged for local pickup")

kh.step("done")
print("=== All done ===")
