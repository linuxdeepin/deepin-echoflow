#!/usr/bin/env python3
"""Re-quantize qwen3-asr-1.7b with audio-tower Q8_0 floor (#240).

The existing Q4_K on HF was quantized before commit 3c3ba2c7 which adds
a Q8_0 floor for audio.* tensors. Without it, the audio encoder's
quantization drift compounds to cos_min=0.75, causing empty transcripts.
"""

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
HF_REPO = "cstr/qwen3-asr-1.7b-GGUF"

print("=== Phase 0: clone + build ===", flush=True)
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", "main",
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
if (REPO / "ggml").is_dir() and not (REPO / "ggml" / "CMakeLists.txt").exists():
    subprocess.check_call(["git", "submodule", "update", "--init", "ggml"], cwd=str(REPO))

if (REPO / "tools" / "kaggle").is_dir():
    sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF " + " ".join(flags))
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-quantize")
quantize = BUILD / "bin" / "crispasr-quantize"

kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

kh.step("download F16 source")
from huggingface_hub import hf_hub_download, HfApi
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
f16_path = hf_hub_download(HF_REPO, "qwen3-asr-1.7b-f16.gguf", cache_dir=str(TEMP / "hf"))
print(f"  F16: {f16_path}")

kh.step("re-quantize Q4_K (with audio Q8_0 floor)")
q4k_path = TEMP / "qwen3-asr-1.7b-q4_k.gguf"
kh.sh_with_progress(f"{quantize} {f16_path} {q4k_path} q4_k")
print(f"  Q4_K: {q4k_path} ({q4k_path.stat().st_size / (1024**2):.0f} MB)")

kh.step("upload fixed Q4_K")
api = HfApi(token=hf_token)
api.upload_file(
    path_or_fileobj=str(q4k_path),
    path_in_repo="qwen3-asr-1.7b-q4_k.gguf",
    repo_id=HF_REPO, repo_type="model",
    commit_message="Re-quantize Q4_K with audio-tower Q8_0 floor (fixes #240 empty transcripts)",
)
print("  uploaded Q4_K")

# Also re-bake Q4_0 and Q5_K if they exist
for quant in ("q4_0", "q5_k"):
    kh.step(f"re-quantize {quant}")
    out = TEMP / f"qwen3-asr-1.7b-{quant}.gguf"
    try:
        kh.sh_with_progress(f"{quantize} {f16_path} {out} {quant}")
        print(f"  {quant}: {out} ({out.stat().st_size / (1024**2):.0f} MB)")
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=f"qwen3-asr-1.7b-{quant}.gguf",
            repo_id=HF_REPO, repo_type="model",
            commit_message=f"Re-quantize {quant.upper()} with audio-tower Q8_0 floor (fixes #240)",
        )
        print(f"  uploaded {quant}")
        out.unlink(missing_ok=True)
    except Exception as e:
        print(f"  {quant} skipped: {e}")

print("\n=== Done ===", flush=True)
