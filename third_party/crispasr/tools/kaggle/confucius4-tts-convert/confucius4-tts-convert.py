#!/usr/bin/env python3
"""Kaggle kernel: Confucius4-TTS — inspect + convert T2S to GGUF + upload to HF.

Phase 1 kernel: downloads the model, dumps all tensor names/shapes/dtypes for
the T2S model (t2s_model.safetensors) and S2A model (s2a_model.pt), then
converts the T2S model to a GGUF. The S2A conversion is deferred to a follow-up
kernel once the T2S runtime is validated.

The T2S model is a GPT-2 backbone (24L, d=1280, 20h) with:
  - TextEmbeddingProjector: Embedding(32000,4096) + MLP(4096→1280)
  - Qwen3TTSSpeakerEncoder: ECAPA-TDNN (mel_dim=1024, enc_dim=1280)
  - Semantic embedding: Embedding(8194, 1280)
  - Learned positional embeddings for text (520) and semantic (1520)
  - Semantic head: Linear(1280, 8194)

REQUIREMENTS:
  - ${KAGGLE_ACCOUNT}/crispasr-hf-token dataset mounted (HF token for upload)
  - No GPU needed (CPU conversion only)

Push (under $KAGGLE_ACCOUNT):
  # Kaggle CLI auth: ~/.kaggle/access_token (never commit a token; see tests/test_no_secrets.py)
  python -m kaggle kernels push -p tools/kaggle/confucius4-tts-convert
"""

import os
import sys
import json
import struct
import subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")

SRC_REPO = "netease-youdao/Confucius4-TTS"
HF_REPO = "cstr/confucius4-tts-GGUF"
NAME = "confucius4-tts"

# ── Phase 0: Clone repo ─────────────────────────────────────────────────────
print("=== Phase 0: clone repo ===", flush=True)
if not REPO.exists():
    try:
        subprocess.check_call([
            "git", "clone", "--depth", "1", "-b", "main",
            "https://github.com/CrispStrobe/CrispASR", str(REPO),
        ])
    except Exception as e:
        print(f"  git clone failed: {e}")

# Init ALL submodules (ggml + c2pa-audio — cmake requires both)
if REPO.exists():
    try:
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(REPO),
        )
    except Exception as e:
        print(f"  submodule init failed: {e}")

if (REPO / "tools" / "kaggle").is_dir():
    sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

# ── Phase 1: Install deps ───────────────────────────────────────────────────
kh.step("install deps")
kh.sh_with_progress("pip install -q safetensors gguf huggingface_hub hf_transfer sentencepiece")

# ── Phase 2: Resolve HF token ───────────────────────────────────────────────
kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("  HF_TOKEN resolved OK")
else:
    print("  No HF_TOKEN -- upload will fail")

# ── Phase 3: Download source model ──────────────────────────────────────────
kh.step("download model")
from huggingface_hub import snapshot_download, HfApi  # noqa: E402

scratch = TEMP / "confucius4-src"
scratch.mkdir(parents=True, exist_ok=True)
free = kh.free_gb(str(scratch))
print(f"  cache: {scratch} (free: {free:.1f} GiB)" if free else f"  cache: {scratch}")

src = snapshot_download(
    repo_id=SRC_REPO,
    cache_dir=str(scratch),
    token=hf_token,
    allow_patterns=[
        "t2s_model.safetensors",
        "s2a_model.pt",
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "wav2vec2bert_stats.pt",
    ],
)
print(f"  source dir: {src}")

# ── Phase 4: Inspect T2S tensor names ────────────────────────────────────────
kh.step("inspect T2S tensors")
import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402

t2s_path = os.path.join(src, "t2s_model.safetensors")
print(f"  T2S file: {t2s_path} ({os.path.getsize(t2s_path) / 1024**3:.2f} GiB)")

t2s_names = []
with safe_open(t2s_path, framework="pt") as f:
    for name in sorted(f.keys()):
        t = f.get_tensor(name)
        t2s_names.append((name, list(t.shape), str(t.dtype)))
        print(f"  {name:65s} {str(list(t.shape)):25s} {t.dtype}")

# Write diagnostic to working dir
with open(WORK / "t2s_tensors.txt", "w") as fp:
    for name, shape, dtype in t2s_names:
        fp.write(f"{name}\t{shape}\t{dtype}\n")

# ── Phase 5: Inspect S2A tensor names ────────────────────────────────────────
kh.step("inspect S2A tensors")
import torch  # noqa: E402

s2a_path = os.path.join(src, "s2a_model.pt")
print(f"  S2A file: {s2a_path} ({os.path.getsize(s2a_path) / 1024**3:.2f} GiB)")

s2a_state = torch.load(s2a_path, map_location="cpu", weights_only=True)
s2a_names = []
for name in sorted(s2a_state.keys()):
    t = s2a_state[name]
    s2a_names.append((name, list(t.shape), str(t.dtype)))
    print(f"  {name:65s} {str(list(t.shape)):25s} {t.dtype}")

with open(WORK / "s2a_tensors.txt", "w") as fp:
    for name, shape, dtype in s2a_names:
        fp.write(f"{name}\t{shape}\t{dtype}\n")

# Free S2A state dict
del s2a_state
import gc; gc.collect()

# ── Phase 6: Inspect wav2vec2bert_stats ──────────────────────────────────────
kh.step("inspect w2v-bert stats")
stats_path = os.path.join(src, "wav2vec2bert_stats.pt")
stats = torch.load(stats_path, map_location="cpu", weights_only=True)
for k, v in stats.items():
    print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")

# ── Phase 7: Read config and tokenizer info ──────────────────────────────────
kh.step("read config + tokenizer")
with open(os.path.join(src, "config.json")) as f:
    config = json.load(f)
print(f"  config.json: {json.dumps(config, indent=2)}")

with open(os.path.join(src, "tokenizer_config.json")) as f:
    tok_config = json.load(f)
print(f"  tokenizer type: {tok_config.get('tokenizer_class', 'unknown')}")
print(f"  vocab_size: {tok_config.get('vocab_size', 'not set')}")
print(f"  model_max_length: {tok_config.get('model_max_length', 'not set')}")

# Count tokens in tokenizer.json
with open(os.path.join(src, "tokenizer.json")) as f:
    tok_data = json.load(f)
vocab = tok_data.get("model", {}).get("vocab", [])
if isinstance(vocab, list):
    print(f"  tokenizer.json vocab entries: {len(vocab)}")
elif isinstance(vocab, dict):
    print(f"  tokenizer.json vocab entries: {len(vocab)}")

# ── Phase 8: Convert T2S to GGUF ────────────────────────────────────────────
kh.step("convert T2S to GGUF")
os.environ["TMPDIR"] = str(TEMP)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gguf  # noqa: E402

f16_path = TEMP / f"{NAME}-t2s-f16.gguf"
writer = gguf.GGUFWriter(str(f16_path), "confucius4-tts")

# KV metadata
writer.add_uint32("confucius4.t2s.num_layers", 24)
writer.add_uint32("confucius4.t2s.model_dim", 1280)
writer.add_uint32("confucius4.t2s.num_heads", 20)
writer.add_uint32("confucius4.t2s.max_text_seq_lens", 520)
writer.add_uint32("confucius4.t2s.max_semantic_seq_lens", 1520)
writer.add_uint32("confucius4.t2s.vocab_size", 32000)
writer.add_uint32("confucius4.t2s.semantic_vocab_size", 8194)
writer.add_uint32("confucius4.t2s.text_embedding_dim", 4096)
writer.add_uint32("confucius4.t2s.speaker_embedding_dim", 1024)
writer.add_uint32("confucius4.t2s.start_semantic_token", 8192)
writer.add_uint32("confucius4.t2s.stop_semantic_token", 8193)
writer.add_uint32("confucius4.sample_rate", 22050)

# Add tokenizer model (raw SentencePiece bytes)
tok_model_path = os.path.join(src, "tokenizer.model")
if os.path.exists(tok_model_path):
    with open(tok_model_path, "rb") as f:
        tok_data_raw = f.read()
    writer.add_array("tokenizer.model", list(tok_data_raw))
    print(f"  baked tokenizer.model ({len(tok_data_raw)} bytes)")

# Bake BPE vocab + merges from tokenizer.json (core/bpe.h compatible)
tok_json_path = os.path.join(src, "tokenizer.json")
if os.path.exists(tok_json_path):
    import json as _json
    with open(tok_json_path) as f:
        tok_j = _json.load(f)
    model = tok_j.get("model", {})
    vocab_dict = model.get("vocab", {})
    merges_list = model.get("merges", [])
    # Newer tokenizer.json stores merges as ["a", "b"] pairs; the C GGUF
    # reader rejects nested arrays (invalid type 9), and core/bpe keys its
    # rank table on the textual "a b" form — join.
    merges_list = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges_list]
    # Build ordered token list by ID
    tokens = [""] * len(vocab_dict)
    for token, idx in vocab_dict.items():
        tokens[idx] = token
    writer.add_array("tokenizer.ggml.tokens", tokens)
    if merges_list:
        writer.add_array("tokenizer.ggml.merges", merges_list)
    print(f"  baked BPE vocab: {len(tokens)} tokens, {len(merges_list)} merges")

# Add wav2vec2bert stats
writer.add_array("confucius4.w2v_bert.mean", stats["mean"].float().numpy().tolist())
writer.add_array("confucius4.w2v_bert.var", stats["var"].float().numpy().tolist())

# Write tensors from T2S safetensors (lazy per-tensor loading)
# GPT-2 Conv1D stores weights as (in_features, out_features) — the TRANSPOSE
# of PyTorch's normal nn.Linear (out_features, in_features). ggml mul_mat(A,B)
# computes B @ A.T and requires A.ne[0] == B.ne[0] == in_features. So Conv1D
# weights must be transposed before writing to GGUF: numpy (in, out) → (out, in)
# → ggml ne[0]=in, ne[1]=out.
GPT2_CONV1D_WEIGHTS = {
    "transformer.h.", ".attn.c_attn.weight",
    "transformer.h.", ".attn.c_proj.weight",
    "transformer.h.", ".mlp.c_fc.weight",
    "transformer.h.", ".mlp.c_proj.weight",
}
def is_gpt2_conv1d(name):
    """Check if this tensor is a GPT-2 Conv1D weight that needs transposing."""
    return (name.startswith("transformer.h.") and
            (name.endswith(".attn.c_attn.weight") or
             name.endswith(".attn.c_proj.weight") or
             name.endswith(".mlp.c_fc.weight") or
             name.endswith(".mlp.c_proj.weight")))

n_written = 0
n_transposed = 0
with safe_open(t2s_path, framework="pt") as f:
    for name in sorted(f.keys()):
        t = f.get_tensor(name).float().numpy()
        # Transpose GPT-2 Conv1D weights: (in, out) → (out, in)
        if t.ndim == 2 and is_gpt2_conv1d(name):
            t = t.T.copy()
            n_transposed += 1
        # 1-D tensors (biases, norms) stay F32; everything else → F16
        if t.ndim == 1:
            writer.add_tensor(name, t)
        else:
            writer.add_tensor(name, t.astype(np.float16))
        n_written += 1
        if n_written % 50 == 0:
            print(f"  wrote {n_written} tensors...", flush=True)

print(f"  transposed {n_transposed} GPT-2 Conv1D weights")

print(f"  wrote {n_written} tensors total")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"  T2S F16 GGUF: {f16_path} ({f16_path.stat().st_size / 1024**3:.2f} GiB)")

# ── Phase 9: Upload F16 to HF ───────────────────────────────────────────────
kh.step("upload F16 to HF")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi(token=hf_token)
try:
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"  repo create: {e}")

print(f"  uploading T2S F16 ({f16_path.stat().st_size / 1024**3:.1f} GiB)...")
api.upload_file(
    path_or_fileobj=str(f16_path),
    path_in_repo=f"{NAME}-t2s-f16.gguf",
    repo_id=HF_REPO, repo_type="model",
    commit_message="Add T2S F16 GGUF (Confucius4-TTS, GPT-2 24L/1280d/20h)",
)
print("  uploaded T2S F16")

# ── Phase 10: Build quantizer + quantize ─────────────────────────────────────
kh.step("build quantizer")
# Submodules already initialised in Phase 0.
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

for quant in ("q8_0", "q4_k"):
    kh.step(f"quantize T2S {quant}")
    out = TEMP / f"{NAME}-t2s-{quant}.gguf"
    kh.sh_with_progress(f"{quantize_bin} {f16_path} {out} {quant}")
    print(f"  {quant}: {out} ({out.stat().st_size / 1024**3:.2f} GiB)")

    kh.step(f"upload T2S {quant}")
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo=f"{NAME}-t2s-{quant}.gguf",
        repo_id=HF_REPO, repo_type="model",
        commit_message=f"Add T2S {quant.upper()} GGUF (Confucius4-TTS)",
    )
    print(f"  uploaded T2S {quant}")
    out.unlink(missing_ok=True)

# Free the F16 now that quants are uploaded
f16_path.unlink(missing_ok=True)

# ── Phase 11: Convert S2A to GGUF ───────────────────────────────────────────
kh.step("convert S2A to GGUF")
s2a_state = torch.load(s2a_path, map_location="cpu", weights_only=True)

s2a_f16_path = TEMP / f"{NAME}-s2a-f16.gguf"
s2a_writer = gguf.GGUFWriter(str(s2a_f16_path), "confucius4-tts-s2a")

# S2A KV metadata
s2a_writer.add_uint32("confucius4.s2a.input_size", 512)
s2a_writer.add_uint32("confucius4.s2a.output_size", 80)
s2a_writer.add_uint32("confucius4.s2a.spk_embed_dim", 192)
s2a_writer.add_uint32("confucius4.s2a.semantic_embed_dim", 1024)
s2a_writer.add_uint32("confucius4.s2a.lm_latent_dim", 1280)
s2a_writer.add_uint32("confucius4.s2a.estimator_depth", 13)
s2a_writer.add_uint32("confucius4.s2a.estimator_num_heads", 8)
s2a_writer.add_uint32("confucius4.s2a.estimator_hidden_dim", 512)
s2a_writer.add_uint32("confucius4.s2a.wavenet_num_layers", 8)

n_written = 0
for name in sorted(s2a_state.keys()):
    t = s2a_state[name].float().numpy()
    if t.ndim == 1:
        s2a_writer.add_tensor(name, t)
    else:
        s2a_writer.add_tensor(name, t.astype(np.float16))
    n_written += 1
    if n_written % 50 == 0:
        print(f"  wrote {n_written} tensors...", flush=True)

print(f"  wrote {n_written} S2A tensors total")

# ── CAMPPlus style encoder baked into the S2A GGUF ──────────────────────────
# The reference's style_embedding comes from funasr/campplus
# (campplus_cn_common.bin, CAMPPlus(feat_dim=80, embedding_size=192)); the
# runtime rebinds it through chatterbox_campplus with the `campplus.` prefix
# (dots-tts pattern). BN running stats and 1-D tensors stay F32; the F16
# 2-D+ weights must be excluded from quantization (quantize rule: keep
# `campplus.` at F16 — the whole encoder is ~28 MB).
from huggingface_hub import hf_hub_download as _dl
campplus_path = _dl("funasr/campplus", filename="campplus_cn_common.bin", token=hf_token)
spk_state = torch.load(campplus_path, map_location="cpu", weights_only=True)
if isinstance(spk_state, dict) and "state_dict" in spk_state:
    spk_state = spk_state["state_dict"]
n_spk = 0
for name in sorted(spk_state.keys()):
    if name.endswith("num_batches_tracked"):
        continue
    t = spk_state[name].float().numpy()
    if t.ndim == 1:
        s2a_writer.add_tensor("campplus." + name, t)
    else:
        s2a_writer.add_tensor("campplus." + name, t.astype(np.float16))
    n_spk += 1
print(f"  baked {n_spk} CAMPPlus tensors under campplus.*")
del spk_state

s2a_writer.write_header_to_file()
s2a_writer.write_kv_data_to_file()
s2a_writer.write_tensors_to_file()
s2a_writer.close()

del s2a_state; gc.collect()
print(f"  S2A F16 GGUF: {s2a_f16_path} ({s2a_f16_path.stat().st_size / 1024**3:.2f} GiB)")

kh.step("upload S2A F16")
api.upload_file(
    path_or_fileobj=str(s2a_f16_path),
    path_in_repo=f"{NAME}-s2a-f16.gguf",
    repo_id=HF_REPO, repo_type="model",
    commit_message="Add S2A F16 GGUF (Confucius4-TTS, Flow-matching DiT+WaveNet)",
)
print("  uploaded S2A F16")

for quant in ("q8_0", "q4_k"):
    kh.step(f"quantize S2A {quant}")
    out = TEMP / f"{NAME}-s2a-{quant}.gguf"
    kh.sh_with_progress(f"{quantize_bin} {s2a_f16_path} {out} {quant}")
    print(f"  {quant}: {out} ({out.stat().st_size / 1024**3:.2f} GiB)")

    kh.step(f"upload S2A {quant}")
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo=f"{NAME}-s2a-{quant}.gguf",
        repo_id=HF_REPO, repo_type="model",
        commit_message=f"Add S2A {quant.upper()} GGUF (Confucius4-TTS)",
    )
    print(f"  uploaded S2A {quant}")
    out.unlink(missing_ok=True)

s2a_f16_path.unlink(missing_ok=True)

print(f"\n=== Done -- GGUFs uploaded to {HF_REPO} ===", flush=True)
