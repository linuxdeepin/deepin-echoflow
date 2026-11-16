#!/usr/bin/env python3
"""Kaggle kernel: convert kyutai/stt-2.6b-en → F16/Q8_0/Q4_K GGUF → upload to HF.

The 2.6B model is the larger English-only sibling of the already-shipped
stt-1b-en_fr. Same Mimi codec + causal LM architecture; the backend handles
it dynamically (all dims and hparams read from GGUF KV). The key differences:
  - 48 layers × 32 heads (vs 16 layers × 16 heads)
  - head_dim = 64 (2048/32) vs 128 (2048/16)
  - text_card = 4000 (English-only vocab, vs 8000)
  - audio_silence_prefix_seconds = 1.0 (prepend 1 s silence before Mimi encode)
  - audio_delay_seconds = 2.5 (vs 0.5)

The existing converter (models/convert-kyutai-stt-to-gguf.py) handles all of
this via config.json; no code changes needed beyond what was added in #238.

Conversion is CPU-bound. The Mimi weights are shared with the 1B model
(same file: mimi-pytorch-e351c8d8@125.safetensors, ~100 MB), so the bulk is
the LM safetensors (~5.2 GB in bfloat16 → ~2.6 GB in F16).

REQUIREMENTS:
  - ${KAGGLE_ACCOUNT}/crispasr-hf-token dataset mounted (HF token for upload only;
    kyutai/stt-2.6b-en is publicly available, no gating).

Push (under $KAGGLE_ACCOUNT):
  export KAGGLE_API_TOKEN=<$KAGGLE_ACCOUNT token>
  python -m kaggle kernels push -p tools/kaggle/kyutai-stt-2.6b-convert
"""

import os
import sys
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

SRC_REPO = "kyutai/stt-2.6b-en"
HF_REPO = "cstr/kyutai-stt-2.6b-en-GGUF"
NAME = "kyutai-stt-2.6b"

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

# ── Phase 2: Resolve HF token ────────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("  HF_TOKEN resolved OK")
else:
    print("  No HF_TOKEN — upload will fail; download OK (model is public)")

# ── Phase 3: Download source model ───────────────────────────────────────────
kh.step("download model")
from huggingface_hub import snapshot_download  # noqa: E402

scratch = TEMP / "kyutai-stt-2.6b-src"
scratch.mkdir(parents=True, exist_ok=True)
free = kh.free_gb(str(scratch))
print(f"  source cache: {scratch}" + (f" (free: {free:.1f} GiB)" if free else ""))

src = snapshot_download(
    repo_id=SRC_REPO,
    cache_dir=str(scratch),
    token=hf_token,
    allow_patterns=["*.safetensors", "config.json", "tokenizer_*.model"],
)
print(f"  source dir: {src}")

# ── Phase 4: Convert to F16 GGUF ─────────────────────────────────────────────
kh.step("convert F16 GGUF")
f16_path = TEMP / f"{NAME}-f16.gguf"
os.environ["TMPDIR"] = str(TEMP)
kh.sh_with_progress(
    f"python models/convert-kyutai-stt-to-gguf.py "
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
    commit_message="Add F16 GGUF (Kyutai STT 2.6B English)",
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
        commit_message=f"Add {quant.upper()} GGUF (Kyutai STT 2.6B English)",
    )
    print(f"  uploaded {quant}")
    out.unlink(missing_ok=True)

# ── Phase 8: Generate reference GGUF for crispasr-diff ──────────────────────
#
# Dumps Mimi + LM intermediates (seanet_output, enc_tfm_output, rvq_codes,
# lm_frame0_logits, generated_text) to a GGUF archive so the C++ diff harness
# can validate element-wise that crispasr's runtime matches PyTorch.
#
# The JFK sample is baked into the repo so no extra audio download is needed.
kh.step("generate reference GGUF")
kh.sh_with_progress(
    "pip install -q moshi sentencepiece scipy gguf",
)
ref_path = TEMP / f"{NAME}-ref.gguf"
jfk_wav = REPO / "samples" / "jfk.wav"
kh.sh_with_progress(
    f"python tools/dump_reference.py --backend kyutai-stt "
    f"--model-dir {src} --audio {jfk_wav} --output {ref_path}",
    cwd=str(REPO),
)
print(f"  ref GGUF: {ref_path} ({ref_path.stat().st_size / 1024:.0f} KiB)")

kh.step("upload ref GGUF to HF")
api.upload_file(
    path_or_fileobj=str(ref_path),
    path_in_repo=f"{NAME}-ref.gguf",
    repo_id=HF_REPO, repo_type="model",
    commit_message="Add reference activation dump GGUF for crispasr-diff (kyutai-stt 2.6B)",
)
print("  uploaded ref GGUF")
ref_path.unlink(missing_ok=True)

print(f"\n=== Done — GGUFs uploaded to {HF_REPO} ===", flush=True)
