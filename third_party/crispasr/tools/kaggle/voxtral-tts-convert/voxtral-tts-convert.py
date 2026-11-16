#!/usr/bin/env python3
"""Kaggle kernel: convert mistralai/Voxtral-4B-TTS-2603 → F16/Q8_0/Q4_K GGUF → upload to HF.

The model is ~8 GB in BF16 (consolidated.safetensors). The converter handles:
  - LLM backbone (26L Ministral 3B)
  - FM acoustic transformer (3L)
  - Codec decoder (weight-norm fusing)
  - 20 preset voice embeddings
  - Tekken tokenizer (131K vocab)

Push (under $KAGGLE_ACCOUNT):
  export KAGGLE_API_TOKEN=<$KAGGLE_ACCOUNT token>
  python -m kaggle kernels push -p tools/kaggle/voxtral-tts-convert
"""

import os
import sys
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

SRC_REPO = "mistralai/Voxtral-4B-TTS-2603"
HF_REPO = "cstr/voxtral-4b-tts-GGUF"
NAME = "voxtral-4b-tts"

# ── Phase 0: Clone repo ──────────────────────────────────────────────────────
print("=== Phase 0: clone repo ===", flush=True)
if not REPO.exists():
    try:
        subprocess.check_call([
            "git", "clone", "--depth", "1", "-b", "main",
            "https://github.com/CrispStrobe/CrispASR", str(REPO),
        ])
    except Exception as e:
        print(f"  git clone failed: {e}")

# Init ggml submodule (needed by crispasr-quantize build)
if (REPO / "ggml").is_dir() and not (REPO / "ggml" / "CMakeLists.txt").exists():
    try:
        subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))
    except Exception as e:
        print(f"  submodule init failed: {e}")

if (REPO / "tools" / "kaggle").is_dir():
    sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

# ── Phase 1: Install deps ────────────────────────────────────────────────────
kh.step("install deps")
kh.sh_with_progress("pip install -q safetensors sentencepiece gguf huggingface_hub hf_transfer")
kh.sh_with_progress("pip install -q torch --index-url https://download.pytorch.org/whl/cpu")

# ── Phase 2: Resolve HF token ────────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("  HF_TOKEN resolved OK")
else:
    print("  No HF_TOKEN — upload will fail")

# ── Phase 3: Download source model ───────────────────────────────────────────
kh.step("download model")
from huggingface_hub import snapshot_download  # noqa: E402

scratch = TEMP / "voxtral-tts-src"
scratch.mkdir(parents=True, exist_ok=True)
free = kh.free_gb(str(scratch))
print(f"  source cache: {scratch}" + (f" (free: {free:.1f} GiB)" if free else ""))

src = snapshot_download(
    repo_id=SRC_REPO,
    cache_dir=str(scratch),
    token=hf_token,
    allow_patterns=["*.safetensors", "params.json", "tekken.json", "voice_embedding/*.pt"],
)
print(f"  source dir: {src}")

# ── Phase 4: Convert to F16 GGUF ─────────────────────────────────────────────
kh.step("convert F16 GGUF")
f16_path = TEMP / f"{NAME}-f16.gguf"
os.environ["TMPDIR"] = str(TEMP)
kh.sh_with_progress(
    f"python models/convert-voxtral-tts-to-gguf.py "
    f"--input {src} --output {f16_path}",
    cwd=str(REPO),
)
print(f"  F16 GGUF: {f16_path} ({f16_path.stat().st_size / (1024**3):.1f} GiB)")

# ── Phase 5: Upload F16 to HF ────────────────────────────────────────────────
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
    commit_message="Add F16 GGUF (Voxtral 4B TTS)",
)
print("  uploaded F16")

# ── Phase 6: Build crispasr-quantize ─────────────────────────────────────────
kh.step("build quantizer")
BUILD = TEMP / "build"
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

# ── Phase 7: Quantize + upload ────────────────────────────────────────────────
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
        commit_message=f"Add {quant.upper()} GGUF (Voxtral 4B TTS)",
    )
    print(f"  uploaded {quant}")
    out.unlink(missing_ok=True)

print(f"\n=== Done — GGUFs uploaded to {HF_REPO} ===", flush=True)
