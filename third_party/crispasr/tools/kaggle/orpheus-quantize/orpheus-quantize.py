#!/usr/bin/env python3
"""Kaggle kernel: download orpheus-3b-0.1-ft F16 GGUF, quantize to Q4_K, upload to HF.

CPU-only (30 GB RAM is enough for the 6.2 GB F16 model).
Downloads orpheus-3b-0.1-ft-f16.gguf from cstr/orpheus-3b-0.1-ft-GGUF,
quantizes to Q4_K, uploads back to the same HF repo.

Push: python -m kaggle kernels push -p tools/kaggle/orpheus-quantize
"""

import os, sys, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else WORK
BUILD = TEMP / "build"

HF_REPO = "cstr/orpheus-3b-0.1-ft-GGUF"
F16_FILE = "orpheus-3b-0.1-ft-f16.gguf"
Q4K_FILE = "orpheus-3b-0.1-ft-q4_k.gguf"

# -- Phase 0: Clone repo --
print("=== Phase 0: clone repo ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()

# -- Phase 1: Install deps --
kh.step("install deps")
kh.install_build_toolchain()
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")

# -- Phase 2: Resolve HF token --
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("  HF_TOKEN resolved OK", flush=True)
else:
    print("  WARNING: no HF_TOKEN", flush=True)

# -- Phase 3: Download F16 GGUF --
kh.step("download F16 GGUF")
from huggingface_hub import hf_hub_download

f16_path = TEMP / F16_FILE
if not f16_path.exists():
    hf_hub_download(
        repo_id=HF_REPO,
        filename=F16_FILE,
        local_dir=str(TEMP),
        token=hf_token,
    )
print(f"  F16 GGUF: {f16_path} ({f16_path.stat().st_size / (1024**3):.2f} GiB)", flush=True)

# -- Phase 4: Build crispasr-quantize --
kh.step("build quantizer")
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
    + " ".join(flags),
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} "
        f"--target crispasr-quantize"
    )

quantize_bin = BUILD / "bin" / "crispasr-quantize"
print(f"  quantizer: {quantize_bin}", flush=True)

# -- Phase 5: Quantize F16 -> Q4_K --
kh.step("quantize Q4_K")
q4k_path = TEMP / Q4K_FILE

kh.sh_with_progress(f"{quantize_bin} {f16_path} {q4k_path} q4_k")
print(f"  Q4_K GGUF: {q4k_path} ({q4k_path.stat().st_size / (1024**3):.2f} GiB)", flush=True)

# -- Phase 6: Upload Q4_K to HF --
kh.step("upload Q4_K to HF")
if hf_token:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    print(f"  uploading Q4_K ({q4k_path.stat().st_size / (1024**3):.2f} GiB)...", flush=True)
    api.upload_file(
        path_or_fileobj=str(q4k_path),
        path_in_repo=Q4K_FILE,
        repo_id=HF_REPO, repo_type="model",
        commit_message="Add Q4_K GGUF (orpheus-3b-0.1-ft, crispasr-quantize)",
    )
    print(f"  uploaded to https://huggingface.co/{HF_REPO}", flush=True)
else:
    print("  no HF_TOKEN — skipping upload", flush=True)

kh.step("done")
print("=== All done ===", flush=True)
