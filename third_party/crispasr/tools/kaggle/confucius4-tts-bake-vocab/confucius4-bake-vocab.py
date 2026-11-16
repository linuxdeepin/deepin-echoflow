#!/usr/bin/env python3
"""Kaggle kernel: bake SentencePiece vocab into Confucius4-TTS T2S GGUF.

Downloads the existing T2S F16 GGUF, adds tokenizer.ggml.tokens +
tokenizer.ggml.scores from the HF tokenizer.json, re-writes as a new
GGUF, quantizes, and uploads. The runtime can then use
core/sentencepiece.h to tokenize at inference time.

Push (under $KAGGLE_ACCOUNT):
  # Kaggle CLI auth: ~/.kaggle/access_token (never commit a token; see tests/test_no_secrets.py)
  python -m kaggle kernels push -p tools/kaggle/confucius4-tts-bake-vocab
"""

import os
import sys
import json
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

HF_REPO = "cstr/confucius4-tts-GGUF"

# ── Phase 0: Clone + deps ───────────────────────────────────────────────────
print("=== Phase 0: setup ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", "main",
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
subprocess.check_call(
    ["git", "submodule", "update", "--init", "--recursive"],
    cwd=str(REPO),
)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q safetensors gguf huggingface_hub hf_transfer tokenizers")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# ── Phase 1: Download tokenizer.json ─────────────────────────────────────────
kh.step("download tokenizer")
from huggingface_hub import hf_hub_download, HfApi
import numpy as np

tok_path = hf_hub_download(
    "netease-youdao/Confucius4-TTS",
    "tokenizer.json",
    local_dir=str(TEMP / "tok"),
    token=hf_token,
)
with open(tok_path) as f:
    tok_data = json.load(f)

model = tok_data["model"]
vocab_dict = model["vocab"]  # {token: id}
merges_list = model.get("merges", [])  # list of "token1 token2" strings
print(f"  model type: {model.get('type', 'unknown')}")
print(f"  vocab entries: {len(vocab_dict)}")
print(f"  merges: {len(merges_list)}")

# Build ordered token list (by ID)
tokens = [""] * len(vocab_dict)
scores = [0.0] * len(vocab_dict)
for token, idx in vocab_dict.items():
    tokens[idx] = token
    scores[idx] = -float(idx)  # BPE: use negative ID as score (lower ID = higher priority)

print(f"  tokens[0:5]: {tokens[:5]}")
print(f"  merges[0:3]: {merges_list[:3]}")

# ── Phase 2: Download existing T2S F16 GGUF ──────────────────────────────────
kh.step("download T2S F16 GGUF")
src_path = hf_hub_download(
    HF_REPO,
    "confucius4-tts-t2s-f16.gguf",
    local_dir=str(TEMP / "models"),
    token=hf_token,
)
print(f"  source: {src_path} ({os.path.getsize(src_path) / 1024**3:.2f} GiB)")

# ── Phase 3: Read existing GGUF, add vocab, write new GGUF ──────────────────
kh.step("bake vocab into GGUF")
import gguf
from safetensors import safe_open

# Read existing GGUF
reader = gguf.GGUFReader(src_path)

# Create new GGUF with the same arch + KV + vocab
out_path = TEMP / "confucius4-tts-t2s-f16-v2.gguf"
writer = gguf.GGUFWriter(str(out_path), "confucius4-tts")

# Copy all existing KV metadata
for field_name in reader.fields:
    field = reader.fields[field_name]
    # Skip internal GGUF fields
    if field_name.startswith("GGUF.") or field_name == "general.quantization_version" or field_name == "general.file_type":
        continue
    # Read the value
    if field.types and field.types[0] == gguf.GGUFValueType.UINT32:
        val = field.parts[field.data[0]][0]
        writer.add_uint32(field_name, int(val))
    elif field.types and field.types[0] == gguf.GGUFValueType.STRING:
        val = str(bytes(field.parts[field.data[0]]), "utf-8")
        writer.add_string(field_name, val)
    elif field.types and field.types[0] == gguf.GGUFValueType.ARRAY:
        # Copy arrays (tokenizer.model, w2v_bert.mean, w2v_bert.var)
        if "mean" in field_name or "var" in field_name:
            vals = [float(field.parts[p][0]) for p in field.data]
            writer.add_array(field_name, vals)
        elif "tokenizer.model" in field_name:
            vals = [int(field.parts[p][0]) for p in field.data]
            writer.add_array(field_name, vals)

# Add the new vocab + merges arrays
writer.add_array("tokenizer.ggml.tokens", tokens)
writer.add_array("tokenizer.ggml.scores", scores)
if merges_list:
    writer.add_array("tokenizer.ggml.merges", merges_list)
    print(f"  added tokenizer.ggml.merges ({len(merges_list)} entries)")
print(f"  added tokenizer.ggml.tokens ({len(tokens)} entries)")
print(f"  added tokenizer.ggml.scores ({len(scores)} entries)")

# Copy all tensors
n_copied = 0
for tensor_info in reader.tensors:
    name = tensor_info.name
    data = tensor_info.data
    # The data is already in the right format (F16/F32 etc)
    writer.add_tensor(name, data, raw_dtype=tensor_info.tensor_type)
    n_copied += 1
    if n_copied % 50 == 0:
        print(f"  copied {n_copied} tensors...", flush=True)

print(f"  copied {n_copied} tensors total")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"  output: {out_path} ({os.path.getsize(str(out_path)) / 1024**3:.2f} GiB)")

# ── Phase 4: Upload new F16 ──────────────────────────────────────────────────
kh.step("upload F16 with vocab")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi(token=hf_token)
api.upload_file(
    path_or_fileobj=str(out_path),
    path_in_repo="confucius4-tts-t2s-f16.gguf",
    repo_id=HF_REPO, repo_type="model",
    commit_message="Add tokenizer.ggml.tokens + scores to T2S F16 GGUF",
)
print("  uploaded F16 with vocab")

# ── Phase 5: Build quantizer + requantize ────────────────────────────────────
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

for quant in ("q8_0", "q4_k"):
    kh.step(f"quantize {quant}")
    qout = TEMP / f"confucius4-tts-t2s-{quant}.gguf"
    kh.sh_with_progress(f"{quantize_bin} {out_path} {qout} {quant}")
    print(f"  {quant}: {qout} ({os.path.getsize(str(qout)) / 1024**3:.2f} GiB)")
    kh.step(f"upload {quant}")
    api.upload_file(
        path_or_fileobj=str(qout),
        path_in_repo=f"confucius4-tts-t2s-{quant}.gguf",
        repo_id=HF_REPO, repo_type="model",
        commit_message=f"Add {quant.upper()} GGUF with baked vocab",
    )
    print(f"  uploaded {quant}")
    qout.unlink(missing_ok=True)

print(f"\n=== Done — T2S GGUFs with vocab uploaded to {HF_REPO} ===", flush=True)
