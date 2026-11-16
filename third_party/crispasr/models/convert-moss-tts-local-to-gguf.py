#!/usr/bin/env python3
"""Convert OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (MossTTSLocal, 4B) to GGUF.

Second deliverable of #249 (the first, the 8B MossTTSDelay, ships as `moss-tts`).
Architecture decoded line-by-line from the HF modeling code — see
docs/moss-tts/STUDY-4B.md. NO C++/GGUF reference exists (openmoss did only the 8B).

  Backbone  : Qwen3-4B — 36 layers, hidden 2560, 32 Q-heads / 8 KV-heads,
              head_dim 128, SwiGLU intermediate 9728, QK-norm, RoPE (NEOX) theta
              1e6, rms_eps 1e-6, vocab 151936, TIED word embeddings.
  Local     : a 1-layer depth transformer (GPT2-style: LayerNorm+bias, fused-QKV
              c_attn/c_proj+bias, RoPE NEOX base 1e6, MLP fc_in->SiLU->fc_out,
              n_inner 9728, 32 heads head_dim 80) that AR-generates the 12
              codebooks *within* each frame — replaces the 8B's delay pattern.
  Audio     : 12 RVQ codebooks (audio_vocab 1024 + pad 1024). Input embedding =
              text_emb + Sum_k audio_emb_k. Heads: text_lm_head (tied to
              embed_tokens), audio_lm_heads.k (tied to audio_embeddings.k), and a
              binary local_text_lm_head (2-way: assistant_slot vs audio_end).
  Codec     : OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 (48 kHz, 12 codebooks) — a
              separate companion GGUF; its architecture is studied in Phase 3
              (--codec is accepted but the v2 tensor map may need adjustment).

Output layout (mirrors the 8B moss-tts convention):

  <output>.gguf            backbone GGUF, arch "moss-tts-local":
      llm.*                Qwen3-4B backbone (quantizable); tied -> llm.lm_head
                           emitted from embed_tokens
      local.*              the 1-layer local/depth transformer (kept F16)
      moss.audio_embed.{k} 12 audio embedding tables         (kept F16)
      moss.audio_head.{k}  12 audio LM heads (= embeds, tied) (kept F16)
      moss.local_text_head binary continue/stop head         (kept F16)
      moss_tts_local.* KV  backbone + local hparams
      moss.* KV            n_vq=12, audio vocab, token ids, 48 kHz frame rate
      gpt2 BPE tokenizer   (Qwen3 vocab + merges)

Streams tensors one-at-a-time via safe_open (BF16->F16); peak RAM stays modest.

Usage:
    python models/convert-moss-tts-local-to-gguf.py \\
        --input OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \\
        --output moss-tts-local-v1.5-f16.gguf
    crispasr-quantize moss-tts-local-v1.5-f16.gguf moss-tts-local-v1.5-q4_k.gguf q4_k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("Error: torch not found. Install torch (BF16->F16 needs it).")
    sys.exit(1)
try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. pip install gguf")
    sys.exit(1)
try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package not found. pip install safetensors")
    sys.exit(1)
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub package not found. pip install huggingface_hub")
    sys.exit(1)


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading model from HuggingFace: {model_id}")
    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json",
                        "tokenizer.json", "added_tokens.json"]))


# --- Backbone Qwen3 name map: transformer.* -> llm.* (same layout as the 8B,
#     different prefix; QK-norm present, embeddings TIED). ---
def map_backbone_name(hf_name: str) -> str:
    name = hf_name
    name = name.replace("transformer.embed_tokens.", "llm.embed.")
    name = name.replace("transformer.norm.", "llm.final_norm.")
    name = name.replace("transformer.layers.", "llm.blk.")
    name = name.replace(".mlp.gate_proj.", ".ffn.gate.")
    name = name.replace(".mlp.up_proj.", ".ffn.up.")
    name = name.replace(".mlp.down_proj.", ".ffn.down.")
    name = name.replace(".self_attn.q_proj.", ".attn.q.")
    name = name.replace(".self_attn.k_proj.", ".attn.k.")
    name = name.replace(".self_attn.v_proj.", ".attn.v.")
    name = name.replace(".self_attn.o_proj.", ".attn.o.")
    name = name.replace(".self_attn.q_norm.", ".attn.q_norm.")
    name = name.replace(".self_attn.k_norm.", ".attn.k_norm.")
    name = name.replace(".input_layernorm.", ".attn_norm.")
    name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
    return name


# --- Local (depth) transformer name map: local_transformer.* -> local.* ---
def map_local_name(hf_name: str) -> str:
    name = hf_name
    name = name.replace("local_transformer.h.", "local.blk.")
    name = name.replace("local_transformer.ln_f.", "local.final_norm.")
    name = name.replace(".attn.c_attn.", ".attn.qkv.")   # fused QKV (+bias)
    name = name.replace(".attn.c_proj.", ".attn.o.")
    name = name.replace(".ln_1.", ".attn_norm.")
    name = name.replace(".ln_2.", ".ffn_norm.")
    name = name.replace(".mlp.fc_in.", ".ffn.in.")
    name = name.replace(".mlp.fc_out.", ".ffn.out.")
    return name


def classify_tensor(hf_name: str):
    """(kind, gguf_name); kind in {backbone, local, audio_embed, local_text_head}.
    Tied heads (text_lm_head, audio_lm_heads) are SKIPPED in the safetensors and
    re-emitted from their embeddings below (llm.lm_head, moss.audio_head.k)."""
    if hf_name.startswith("transformer."):
        return "backbone", map_backbone_name(hf_name)
    if hf_name.startswith("local_transformer."):
        return "local", map_local_name(hf_name)
    if hf_name.startswith("audio_embeddings.") and hf_name.endswith(".weight"):
        idx = int(hf_name.split(".")[1])
        return "audio_embed", f"moss.audio_embed.{idx}.weight"
    if hf_name == "local_text_lm_head.weight":
        return "local_text_head", "moss.local_text_head.weight"
    # tied / redundant heads — skip (re-emitted from embeddings)
    if hf_name == "text_lm_head.weight" or hf_name.startswith("audio_lm_heads."):
        return "skip_tied", None
    return None, None


def bake_tokenizer(writer, model_dir: Path, vocab_size: int):
    tj = model_dir / "tokenizer.json"
    if tj.exists():
        with open(tj, encoding="utf-8") as f:
            tok = json.load(f)
        model = tok.get("model", {})
        vocab = model.get("vocab", {})
        merges = model.get("merges", [])
        merges = [" ".join(m) if isinstance(m, list) else m for m in merges]
        tokens = [f"[PAD{i}]" for i in range(vocab_size)]
        for token, idx in vocab.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = token
        for at in tok.get("added_tokens", []):
            idx = at.get("id")
            if idx is not None and 0 <= idx < vocab_size:
                tokens[idx] = at.get("content", tokens[idx])
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        if merges:
            writer.add_token_merges(merges)
        print(f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges")
        return
    print("  WARNING: no tokenizer.json — tokenizer not baked")


def to_gguf_array(tensor, out_dtype, ggml_type, keep_f16=False):
    if tensor.dtype == torch.bfloat16:
        arr = tensor.to(torch.float32).numpy()
    else:
        arr = tensor.numpy()
    if arr.ndim >= 2:
        if keep_f16:
            return arr.astype(np.float16), GGMLQuantizationType.F16
        return arr.astype(out_dtype), ggml_type
    return arr.astype(np.float32), GGMLQuantizationType.F32


def write_backbone_gguf(model_dir: Path, config: dict, out_path: Path, out_dtype, ggml_type):
    qc = config.get("qwen3_config", config.get("language_config", {}))
    gc = config.get("gpt2_config", {})
    n_vq = int(config.get("n_vq", 12))
    audio_vocab_size = int(config.get("audio_vocab_size", 1024))
    audio_pad_code = int(config.get("audio_pad_code", audio_vocab_size))
    sampling_rate = int(config.get("sampling_rate", 48000))
    # MOSS-Audio-Tokenizer-v2 hop = 3840 (48000/3840 = 12.5 Hz frame rate); STEREO.
    downsample_rate = int(config.get("downsample_rate", 3840))
    audio_channels = int(config.get("number_channels", 2))
    vocab_size = int(qc.get("vocab_size", 151936))

    print("\nMOSS-TTS-Local-Transformer-v1.5 (MossTTSLocal, 4B)")
    print(f"  Backbone: {qc.get('num_hidden_layers', 36)}L hidden={qc.get('hidden_size', 2560)} "
          f"heads={qc.get('num_attention_heads', 32)}/{qc.get('num_key_value_heads', 8)} "
          f"head_dim={qc.get('head_dim', 128)} ffn={qc.get('intermediate_size', 9728)} tied={qc.get('tie_word_embeddings')}")
    print(f"  Local: {int(config.get('local_transformer_layers', gc.get('n_layer', 1)))}L "
          f"n_embd={gc.get('n_embd', 2560)} heads={gc.get('n_head', 32)} n_inner={gc.get('n_inner', 9728)} "
          f"rope_base={gc.get('rope_base', 1e6)} act={gc.get('activation_function', 'silu')}")
    print(f"  Audio: n_vq={n_vq}, audio_vocab={audio_vocab_size}(+1 pad), vocab={vocab_size}, {sampling_rate} Hz")

    writer = GGUFWriter(str(out_path), "moss-tts-local", use_temp_file=True)
    writer.add_name("MOSS-TTS-Local-Transformer-v1.5")

    # Backbone (Qwen3-4B) hparams
    writer.add_uint32("moss_tts_local.llm.hidden_size", int(qc.get("hidden_size", 2560)))
    writer.add_uint32("moss_tts_local.llm.num_layers", int(qc.get("num_hidden_layers", 36)))
    writer.add_uint32("moss_tts_local.llm.num_heads", int(qc.get("num_attention_heads", 32)))
    writer.add_uint32("moss_tts_local.llm.num_kv_heads", int(qc.get("num_key_value_heads", 8)))
    writer.add_uint32("moss_tts_local.llm.head_dim", int(qc.get("head_dim", 128)))
    writer.add_uint32("moss_tts_local.llm.intermediate_size", int(qc.get("intermediate_size", 9728)))
    writer.add_uint32("moss_tts_local.llm.vocab_size", vocab_size)
    writer.add_uint32("moss_tts_local.llm.max_position_embeddings",
                      int(qc.get("max_position_embeddings", 32768)))
    writer.add_float32("moss_tts_local.llm.rope_theta", float(qc.get("rope_theta", 1000000.0)))
    writer.add_float32("moss_tts_local.llm.rms_norm_eps", float(qc.get("rms_norm_eps", 1e-6)))
    writer.add_bool("moss_tts_local.llm.tied_embeddings", bool(qc.get("tie_word_embeddings", True)))

    # Local (depth) transformer hparams
    writer.add_uint32("moss_tts_local.local.num_layers",
                      int(config.get("local_transformer_layers", gc.get("n_layer", 1))))
    writer.add_uint32("moss_tts_local.local.hidden_size", int(gc.get("n_embd", 2560)))
    writer.add_uint32("moss_tts_local.local.num_heads", int(gc.get("n_head", 32)))
    writer.add_uint32("moss_tts_local.local.intermediate_size", int(gc.get("n_inner", 9728)))
    writer.add_float32("moss_tts_local.local.rope_base", float(gc.get("rope_base", 1000000.0)))
    writer.add_float32("moss_tts_local.local.layer_norm_eps", float(gc.get("layer_norm_epsilon", 1e-6)))

    # Audio hparams
    writer.add_uint32("moss.n_vq", n_vq)
    writer.add_uint32("moss.audio_vocab_size", audio_vocab_size)
    writer.add_uint32("moss.audio_pad_code", audio_pad_code)
    writer.add_uint32("moss.sampling_rate", sampling_rate)
    writer.add_uint32("moss.downsample_rate", downsample_rate)
    writer.add_uint32("moss.audio_channels", audio_channels)
    writer.add_float32("moss.frame_rate", float(sampling_rate) / float(downsample_rate))
    writer.add_string("moss.local_text_head_mode", str(config.get("local_text_head_mode", "binary")))

    # Special token ids (MossTTSLocal config.json)
    writer.add_uint32("moss.token.audio_start", int(config.get("audio_start_token_id", 151669)))
    writer.add_uint32("moss.token.audio_end", int(config.get("audio_end_token_id", 151670)))
    writer.add_uint32("moss.token.audio_user_slot", int(config.get("audio_user_slot_token_id", 151654)))
    writer.add_uint32("moss.token.audio_gen_slot",
                      int(config.get("audio_assistant_gen_slot_token_id",
                                     config.get("audio_assistant_slot_token_id", 151656))))
    writer.add_uint32("moss.token.im_start", int(config.get("im_start_token_id", 151644)))
    writer.add_uint32("moss.token.im_end", int(config.get("im_end_token_id", 151645)))
    writer.add_uint32("moss.token.pad", int(config.get("pad_token_id", 151643)))

    bake_tokenizer(writer, model_dir, vocab_size)

    st_files = sorted(model_dir.glob("*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_handle = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            name_to_handle[name] = idx
    print(f"  Safetensors: {len(name_to_handle)} tensors in {len(st_files)} file(s)")

    counts = {"backbone": 0, "local": 0, "audio_embed": 0, "local_text_head": 0}
    skipped = []
    for hf_name in sorted(name_to_handle):
        kind, gguf_name = classify_tensor(hf_name)
        if kind is None:
            skipped.append(hf_name)
            continue
        if kind == "skip_tied":
            continue
        tensor = handles[name_to_handle[hf_name]].get_tensor(hf_name)
        # local transformer + audio embeds/heads + local_text_head are
        # precision-sensitive -> keep F16 (quantize only the big Qwen3 backbone).
        keep_f16 = kind in ("local", "audio_embed", "local_text_head")
        arr, dtype = to_gguf_array(tensor, out_dtype, ggml_type, keep_f16=keep_f16)
        writer.add_tensor(gguf_name, arr, raw_dtype=dtype)
        counts[kind] += 1

        # Re-emit tied heads from their embeddings so the runtime has explicit
        # tensors (embeddings are TIED: text_lm_head=embed_tokens,
        # audio_lm_heads.k=audio_embeddings.k).
        if gguf_name == "llm.embed.weight":
            arr2, d2 = to_gguf_array(tensor, out_dtype, ggml_type, keep_f16=False)
            writer.add_tensor("llm.lm_head.weight", arr2, raw_dtype=d2)
        elif kind == "audio_embed":
            k = gguf_name.split(".")[2]
            arr2, d2 = to_gguf_array(tensor, out_dtype, ggml_type, keep_f16=True)
            writer.add_tensor(f"moss.audio_head.{k}.weight", arr2, raw_dtype=d2)
        del tensor

    print(f"  Wrote: {counts}")
    if skipped:
        print(f"  Skipped {len(skipped)} unmapped tensors, e.g. {skipped[:5]}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  -> {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


# ===========================================================================
# Codec (MOSS-Audio-Tokenizer-v2) — DECODE-only companion GGUF
# ===========================================================================
#
# We emit ONLY the tensors the decode path needs (see docs/moss-tts/STUDY-4B.md
# "P3 codec" + PLAN NOW):
#   - quantizer: input_proj/output_proj + the FIRST 12 LFQ quantizers
#     (codebook[1024,8] + in_proj 512->8 + out_proj 8->512). Quantizers 12..31
#     are skipped (the LM only predicts the top 12).
#   - decoder: the 6 ProjectedTransformer stages (gguf idx 0,2,4,6,8,10). The 6
#     PatchedPretransform upsamplers (odd idx) are weightless pure reshapes.
#   - encoder (voice cloning): the mirrored 6 stages (gguf idx 1,3,5,7,9,11);
#     the even indices are the weightless downsamplers. Layer counts mirror the
#     decoder in reverse — enc.11 has 32 layers against dec.0's 32, the rest 12.
#     Emitted only when the checkpoint carries them, and flagged in the KV as
#     `encoder_present` so a decode-only GGUF still loads and simply reports
#     cloning as unavailable.
# WNConv1d weight-norm params are stored raw (wp0=original0, wp1=original1) and
# reconstructed at load (w = wp0 * wp1 / ||wp1||) — mirrors the v1 codec runtime.
#
# ⚠ v2 has input_proj AND output_proj on EVERY stage (even when d_model==out_dim,
# e.g. dec.0 output_proj is a real 1280->1280 matrix) — both are always emitted.

CODEC_N_VQ_USED = 12          # the LM predicts the top 12 of 32 quantizers
CODEC_DECODER_STAGES = [0, 2, 4, 6, 8, 10]  # ProjectedTransformer gguf indices
CODEC_ENCODER_STAGES = [1, 3, 5, 7, 9, 11]  # mirrored analysis stages


def _codec_add(writer, name, tensor, force_f32):
    """Add a codec tensor. 2D matmul weights -> F16 unless force_f32; 1D -> F32."""
    if tensor.dtype == torch.bfloat16:
        arr = tensor.to(torch.float32).numpy()
    else:
        arr = tensor.numpy()
    arr = np.ascontiguousarray(arr)
    if force_f32 or arr.ndim < 2:
        writer.add_tensor(name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
    else:
        writer.add_tensor(name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)


def _wn_prefix(handles, name_to_handle, hf_prefix, gguf_prefix, writer):
    """Emit a WNConv1d (k=1) as wp0/wp1/bias. hf_prefix e.g.
    'quantizer.output_proj'; gguf_prefix e.g. 'codec.quant.oproj'."""
    g0 = handles[name_to_handle[hf_prefix + ".parametrizations.weight.original0"]].get_tensor(
        hf_prefix + ".parametrizations.weight.original0")
    g1 = handles[name_to_handle[hf_prefix + ".parametrizations.weight.original1"]].get_tensor(
        hf_prefix + ".parametrizations.weight.original1")
    # original0: (out,1,1) -> (out,);  original1: (out,in,1) -> (out,in)
    _codec_add(writer, gguf_prefix + ".wp0", g0.reshape(g0.shape[0]), force_f32=True)
    _codec_add(writer, gguf_prefix + ".wp1", g1.reshape(g1.shape[0], g1.shape[1]), force_f32=True)
    bkey = hf_prefix + ".bias"
    if bkey in name_to_handle:
        b = handles[name_to_handle[bkey]].get_tensor(bkey)
        _codec_add(writer, gguf_prefix + ".bias", b, force_f32=True)


def write_codec_gguf(codec_dir: Path, out_path: Path):
    with open(codec_dir / "config.json", encoding="utf-8") as f:
        ccfg = json.load(f)
    qk = ccfg.get("quantizer_kwargs", {})

    print("\nMOSS-Audio-Tokenizer-v2 codec (decode-only)")
    print(f"  {ccfg.get('sample_rate', 48000)} Hz, {ccfg.get('number_channels', 2)}ch, "
          f"hop {ccfg.get('downsample_rate', 3840)}, n_vq_used {CODEC_N_VQ_USED}")

    writer = GGUFWriter(str(out_path), "moss-tts-local-codec", use_temp_file=True)
    writer.add_name("MOSS-Audio-Tokenizer-v2")
    writer.add_uint32("moss-tts-local-codec.num_quantizers", CODEC_N_VQ_USED)
    writer.add_uint32("moss-tts-local-codec.codebook_size", int(qk.get("codebook_size", 1024)))
    writer.add_uint32("moss-tts-local-codec.codebook_dim", int(qk.get("codebook_dim", 8)))
    writer.add_uint32("moss-tts-local-codec.rvq_dim", int(qk.get("rvq_dim", 512)))
    writer.add_uint32("moss-tts-local-codec.output_dim", int(qk.get("output_dim", 768)))
    writer.add_uint32("moss-tts-local-codec.sampling_rate", int(ccfg.get("sample_rate", 48000)))
    writer.add_uint32("moss-tts-local-codec.downsample_rate", int(ccfg.get("downsample_rate", 3840)))
    writer.add_uint32("moss-tts-local-codec.num_channels", int(ccfg.get("number_channels", 2)))
    writer.add_bool("moss-tts-local-codec.enable_channel_interleave",
                    bool(ccfg.get("enable_channel_interleave", True)))
    writer.add_float32("moss-tts-local-codec.rope_max_period", 10000.0)
    writer.add_bool("moss-tts-local-codec.present", True)

    st_files = sorted(codec_dir.glob("*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_handle = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            name_to_handle[name] = idx
    print(f"  Safetensors: {len(name_to_handle)} tensors in {len(st_files)} file(s)")

    n_emit = 0

    # --- Quantizer: global output_proj (512->768) + 12 LFQ quantizers ---
    _wn_prefix(handles, name_to_handle, "quantizer.output_proj", "codec.quant.oproj", writer)
    n_emit += 1
    for i in range(CODEC_N_VQ_USED):
        cb = handles[name_to_handle[f"quantizer.quantizers.{i}.codebook.weight"]].get_tensor(
            f"quantizer.quantizers.{i}.codebook.weight")  # (1024, 8)
        _codec_add(writer, f"codec.quant.q.{i}.codebook", cb, force_f32=False)
        _wn_prefix(handles, name_to_handle, f"quantizer.quantizers.{i}.out_proj",
                   f"codec.quant.q.{i}.oproj", writer)
        n_emit += 1

    # --- ProjectedTransformer stages (encoder and decoder share the layout) ---
    def _emit_stages(hf_root, gguf_root, stages, label):
        """Emit one side's ProjectedTransformer stages. Encoder and decoder have
        identical per-stage structure — only the index set and the layer counts
        differ (they mirror in reverse: enc.11 and dec.0 both carry 32)."""
        emitted = 0
        for S in stages:
            base = f"{hf_root}.{S}."
            gb = f"{gguf_root}.{S}."
            # input_proj / output_proj ALWAYS present (bias=False linears).
            _codec_add(writer, gb + "iproj.weight",
                       handles[name_to_handle[base + "input_proj.weight"]].get_tensor(base + "input_proj.weight"),
                       force_f32=False)
            _codec_add(writer, gb + "oproj.weight",
                       handles[name_to_handle[base + "output_proj.weight"]].get_tensor(base + "output_proj.weight"),
                       force_f32=False)
            # count layers present for this stage
            li = 0
            while (base + f"transformer.layers.{li}.norm1.weight") in name_to_handle:
                lb = base + f"transformer.layers.{li}."
                glb = gb + f"l.{li}."
                for src, dst, f32 in [
                    ("norm1.weight", "norm1.weight", True), ("norm1.bias", "norm1.bias", True),
                    ("norm2.weight", "norm2.weight", True), ("norm2.bias", "norm2.bias", True),
                    ("self_attn.in_proj.weight", "attn_in.weight", False),
                    ("self_attn.out_proj.weight", "attn_out.weight", False),
                    ("ffn.0.weight", "ffn1.weight", False), ("ffn.2.weight", "ffn2.weight", False),
                    ("layer_scale_1.scale", "ls1.scale", True), ("layer_scale_2.scale", "ls2.scale", True),
                ]:
                    key = lb + src
                    if key not in name_to_handle:
                        raise RuntimeError(f"codec: missing {key}")
                    _codec_add(writer, glb + dst, handles[name_to_handle[key]].get_tensor(key), force_f32=f32)
                li += 1
            print(f"  {label}.{S}: {li} layers")
            emitted += 1
        return emitted

    n_emit += _emit_stages("decoder", "codec.dec", CODEC_DECODER_STAGES, "dec")

    # --- Encoder (voice cloning): emitted only when the checkpoint has it ---
    have_encoder = all(f"encoder.{S}.input_proj.weight" in name_to_handle
                       for S in CODEC_ENCODER_STAGES)
    if have_encoder:
        # Analysis side of the quantizer: input_proj (768->512) and the per-
        # quantizer in_proj (512->8). Without these the encoder output cannot be
        # turned into codes, so they are part of the same all-or-nothing gate.
        _wn_prefix(handles, name_to_handle, "quantizer.input_proj", "codec.quant.iproj", writer)
        n_emit += 1
        for i in range(CODEC_N_VQ_USED):
            _wn_prefix(handles, name_to_handle, f"quantizer.quantizers.{i}.in_proj",
                       f"codec.quant.q.{i}.iproj", writer)
            n_emit += 1
        n_emit += _emit_stages("encoder", "codec.enc", CODEC_ENCODER_STAGES, "enc")
        print("  encoder: present -> voice cloning available")
    else:
        print("  encoder: ABSENT in checkpoint -> decode-only GGUF (no voice cloning)")
    writer.add_bool("moss-tts-local-codec.encoder_present", bool(have_encoder))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"  -> {out_path} ({out_path.stat().st_size / 1e9:.2f} GB), {n_emit} modules")


def main():
    ap = argparse.ArgumentParser(description="Convert MOSS-TTS-Local-Transformer-v1.5 (4B) to GGUF")
    ap.add_argument("--input", required=True, help="HF id or local dir of MOSS-TTS-Local-Transformer-v1.5")
    ap.add_argument("--output", required=True, help="output backbone GGUF path")
    ap.add_argument("--codec", default=None, help="(Phase 3) HF id/dir of MOSS-Audio-Tokenizer-v2")
    ap.add_argument("--codec-output", default=None, help="codec GGUF path (default: <output>-codec.gguf)")
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()

    out_dtype = np.float16 if args.dtype == "f16" else np.float32
    ggml_type = GGMLQuantizationType.F16 if args.dtype == "f16" else GGMLQuantizationType.F32

    model_dir = load_model_dir(args.input)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_type") != "moss_tts_local":
        print(f"WARNING: model_type={config.get('model_type')!r} (expected 'moss_tts_local')")

    write_backbone_gguf(model_dir, config, Path(args.output), out_dtype, ggml_type)

    if args.codec:
        codec_dir = load_model_dir(args.codec)
        codec_out = (Path(args.codec_output) if args.codec_output
                     else Path(args.output).with_name(Path(args.output).stem + "-codec.gguf"))
        write_codec_gguf(codec_dir, codec_out)


if __name__ == "__main__":
    main()
