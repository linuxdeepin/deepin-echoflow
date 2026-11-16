#!/usr/bin/env python3
"""Kaggle kernel: convert google/gemma-4-E4B-it → F16/Q8_0/Q4_K GGUF → upload to HF.

E4B is the larger sibling of the already-shipped E2B (issue #196 asked for a
bigger Gemma-4 ASR model). It is the SAME `gemma4` architecture as E2B — a
byte-identical USM Conformer audio tower (1024-dim) plus a larger decoder
(42L×2560, kv_heads=2, use_double_wide_mlp=False). The existing converter and
C++ gemma4 backend handle it with no code changes: all dims are read
dynamically and the backend's single-MLP GeGLU path matches E4B's standard MLP.

  (The 12B model is a DIFFERENT architecture, `gemma4_unified`, and is rejected
   by the converter — see #196.)

Conversion is CPU-bound (streams tensors one at a time, RAM stays bounded), but
the kernel runs with enable_gpu=true because Kaggle CPU-only workers get NO
internet — and we must download E4B from HF and upload the GGUFs back. Large
artifacts go to /kaggle/temp (not /kaggle/working, capped at ~20 GB).

REQUIREMENTS:
  - ${KAGGLE_ACCOUNT}/crispasr-hf-token dataset mounted (HF token); the token's HF
    account must have accepted the google/gemma-4-E4B-it gated license.

Push (under $KAGGLE_ACCOUNT):
  export KAGGLE_API_TOKEN=<$KAGGLE_ACCOUNT token>
  python -m kaggle kernels push -p tools/kaggle/gemma4-e4b-convert
"""

import os
import sys
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BUILD = TEMP / "build"

SRC_REPO = "google/gemma-4-E4B-it"
HF_REPO = "cstr/gemma4-e4b-it-GGUF"
NAME = "gemma4-e4b-it"

# ── Phase 0: Clone repo (harness + converter live in it) ────────────────────
print("=== Phase 0: clone repo ===", flush=True)
if not REPO.exists():
    try:
        subprocess.check_call([
            "git", "clone", "--depth", "1", "-b", "main",
            "https://github.com/CrispStrobe/CrispASR", str(REPO),
        ])
    except Exception as e:
        print(f"  git clone failed: {e}")

# Import harness from the clone; fall back to the copy bundled in the push dir.
if (REPO / "tools" / "kaggle").is_dir():
    sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

# ── Phase 1: Install deps ───────────────────────────────────────────────────
# torch + transformers are pre-installed and NOT needed by the converter
# (it reads safetensors directly) — don't reinstall torch (wastes time / OOM).
kh.step("install deps")
kh.install_build_toolchain()  # ninja + ccache + mold (for crispasr-quantize)
kh.sh_with_progress("pip install -q safetensors gguf huggingface_hub hf_transfer")

# ── Phase 2: Resolve HF token ───────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("  HF_TOKEN resolved OK")
else:
    print("  FATAL: no HF_TOKEN — gemma-4-E4B-it is gated and cannot download")
    sys.exit(1)

# ── Phase 3: Download source model ──────────────────────────────────────────
kh.step("download model")
from huggingface_hub import snapshot_download  # noqa: E402

scratch = TEMP / "gemma4-e4b-src"
scratch.mkdir(parents=True, exist_ok=True)
free = kh.free_gb(str(scratch))
print(f"  source cache: {scratch}" + (f" (free: {free:.1f} GiB)" if free else ""))

src = snapshot_download(
    repo_id=SRC_REPO,
    cache_dir=str(scratch),
    token=hf_token,
    allow_patterns=["*.safetensors", "config.json", "tokenizer.json",
                    "tokenizer_config.json", "generation_config.json"],
)
print(f"  source dir: {src}")

# ── Phase 4: Convert to F16 GGUF ────────────────────────────────────────────
kh.step("convert F16 GGUF")
f16_path = TEMP / f"{NAME}-f16.gguf"
# Keep the gguf-writer temp file off /kaggle/working (capped) and on /kaggle/temp.
os.environ["TMPDIR"] = str(TEMP)
kh.sh_with_progress(
    f"python models/convert-gemma4-e2b-to-gguf.py "
    f"--input {src} --output {f16_path} --outtype f16",
    cwd=str(REPO),
)
print(f"  F16 GGUF: {f16_path} ({f16_path.stat().st_size / (1024**3):.1f} GiB)")

# ── Phase 5: Upload F16 to HF ───────────────────────────────────────────────
kh.step("upload F16 to HF")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=hf_token)
try:
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"  repo create: {e}")

print(f"  uploading F16 ({f16_path.stat().st_size / (1024**3):.1f} GiB)...")
api.upload_file(
    path_or_fileobj=str(f16_path),
    path_in_repo=f"{NAME}-f16.gguf",
    repo_id=HF_REPO, repo_type="model",
    commit_message="Add F16 GGUF (Gemma-4 E4B audio ASR)",
)
print("  uploaded F16")

# ── Phase 6: Build crispasr-quantize ────────────────────────────────────────
kh.step("build quantizer")
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF " + " ".join(flags),
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-quantize"
    )
quantize_bin = BUILD / "bin" / "crispasr-quantize"
print(f"  quantizer: {quantize_bin}")

# ── Phase 7: Quantize + upload (Q8_0, then Q4_K) ────────────────────────────
for quant in ("q8_0", "q4_k"):
    kh.step(f"quantize {quant}")
    out = TEMP / f"{NAME}-{quant}.gguf"
    kh.sh_with_progress(f"{quantize_bin} {f16_path} {out} {quant}")
    print(f"  {quant} GGUF: {out} ({out.stat().st_size / (1024**3):.1f} GiB)")
    kh.step(f"upload {quant}")
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo=f"{NAME}-{quant}.gguf",
        repo_id=HF_REPO, repo_type="model",
        commit_message=f"Add {quant.upper()} GGUF (Gemma-4 E4B audio ASR)",
    )
    print(f"  uploaded {quant}")
    out.unlink(missing_ok=True)

f16_path.unlink(missing_ok=True)
kh.step("done")
print(f"\n=== All done — https://huggingface.co/{HF_REPO} ===")
