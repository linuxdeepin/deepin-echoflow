#!/usr/bin/env python3
"""
Convert Aratako/Irodori-TTS safetensors → GGUF for the CrispASR
`irodori-tts` backend.

Supports both base (v3) and VoiceDesign (v3-VoiceDesign) checkpoints.
VoiceDesign adds a caption encoder for style/emotion conditioning.

Irodori-TTS pipeline:
  - TextEncoder: Embedding + N TextBlock layers (RoPE self-attn + SwiGLU)
  - CaptionEncoder (VoiceDesign): same TextEncoder arch for style text
  - ReferenceLatentEncoder: Linear + N TextBlock layers
  - TimestepEmbedding: sinusoidal → MLP
  - DiT: N DiffusionBlock layers (LowRankAdaLN + JointAttention + SwiGLU)
    JointAttention has text + speaker + caption (VoiceDesign) context KV
  - DurationPredictor: token_sum_adarn_zero / token_sum_dual_adarn_zero
  - Euler RF ODE solver
  - DAC-VAE decoder: Semantic-DACVAE-Japanese-32dim (48kHz)

Usage:
    python models/convert-irodori-tts-to-gguf.py \\
        --hf-repo Aratako/Irodori-TTS-600M-v3-VoiceDesign \\
        --output /mnt/storage/gguf-models/irodori-tts-600m-v3-voicedesign-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


# ── Default hyperparameters (v3 500M base) ──────────────────────────

IRODORI_HPARAMS = dict(
    arch="irodori-tts",
    latent_dim=128,
    latent_patch_size=1,
    model_dim=2048,
    num_layers=24,
    num_heads=16,
    mlp_ratio=2.875,
    text_dim=1280,
    text_layers=14,
    text_heads=10,
    text_mlp_ratio=2.6,
    text_vocab_size=102400,
    speaker_dim=1280,
    speaker_layers=14,
    speaker_heads=10,
    speaker_mlp_ratio=2.6,
    speaker_patch_size=1,
    timestep_embed_dim=512,
    adaln_rank=256,
    norm_eps=1e-5,
    # Duration predictor (v3)
    use_duration_predictor=1,
    duration_aux_dim=14,
    duration_hidden_dim=1024,
    duration_layers=3,
    # Sampling defaults
    ode_steps=40,
    cfg_scale_text=3.0,
    cfg_scale_speaker=5.0,
    sample_rate=48000,
    # From Semantic-DACVAE-Japanese-32dim: strides [12,10,8,2] = 1920
    codec_hop_length=1920,
)


# ── Helpers ──────────────────────────────────────────────────────────

def to_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).numpy()


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()


def write_tensor(writer: GGUFWriter, name: str, data: np.ndarray, desc: str = ""):
    """Write tensor, choosing F16 for large weights, F32 for small/bias."""
    if data.size <= 512 or data.ndim == 1:
        writer.add_tensor(name, data.astype(np.float32))
    else:
        writer.add_tensor(name, data.astype(np.float16))


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load all tensors from a safetensors file."""
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


# ── Tensor name mapping ──────────────────────────────────��───────────

def map_text_encoder_generic(state: dict, writer: GGUFWriter,
                             src_prefix: str, gguf_prefix: str):
    """Map TextEncoder weights (shared by text and caption encoders)."""
    # Embedding
    t = state[f"{src_prefix}text_embedding.weight"]
    write_tensor(writer, f"{gguf_prefix}.emb", to_f16(t))

    # TextBlock layers
    n_layers = 0
    while f"{src_prefix}blocks.{n_layers}.attention.wq.weight" in state:
        n_layers += 1

    for i in range(n_layers):
        bp = f"{src_prefix}blocks.{i}."
        op = f"{gguf_prefix}.blk.{i}."

        write_tensor(writer, f"{op}attn_norm", to_f32(state[f"{bp}attention_norm.weight"]))
        write_tensor(writer, f"{op}attn.wq", to_f16(state[f"{bp}attention.wq.weight"]))
        write_tensor(writer, f"{op}attn.wk", to_f16(state[f"{bp}attention.wk.weight"]))
        write_tensor(writer, f"{op}attn.wv", to_f16(state[f"{bp}attention.wv.weight"]))
        write_tensor(writer, f"{op}attn.wo", to_f16(state[f"{bp}attention.wo.weight"]))
        write_tensor(writer, f"{op}attn.gate", to_f16(state[f"{bp}attention.gate.weight"]))
        write_tensor(writer, f"{op}attn.q_norm", to_f32(state[f"{bp}attention.q_norm.weight"]))
        write_tensor(writer, f"{op}attn.k_norm", to_f32(state[f"{bp}attention.k_norm.weight"]))
        write_tensor(writer, f"{op}mlp_norm", to_f32(state[f"{bp}mlp_norm.weight"]))
        write_tensor(writer, f"{op}mlp.w1", to_f16(state[f"{bp}mlp.w1.weight"]))
        write_tensor(writer, f"{op}mlp.w2", to_f16(state[f"{bp}mlp.w2.weight"]))
        write_tensor(writer, f"{op}mlp.w3", to_f16(state[f"{bp}mlp.w3.weight"]))

    return n_layers


def map_text_encoder(state: dict, writer: GGUFWriter, prefix: str = "text_encoder."):
    return map_text_encoder_generic(state, writer, prefix, "irodori.text_enc")


def map_caption_encoder(state: dict, writer: GGUFWriter, prefix: str = "caption_encoder."):
    """Map CaptionEncoder weights (VoiceDesign). Same structure as TextEncoder."""
    if f"{prefix}text_embedding.weight" not in state:
        return 0
    return map_text_encoder_generic(state, writer, prefix, "irodori.cap_enc")


def map_speaker_encoder(state: dict, writer: GGUFWriter, prefix: str = "speaker_encoder."):
    """Map ReferenceLatentEncoder weights."""
    # Input projection
    write_tensor(writer, "irodori.spk_enc.in_proj.w", to_f16(state[f"{prefix}in_proj.weight"]))
    write_tensor(writer, "irodori.spk_enc.in_proj.b", to_f32(state[f"{prefix}in_proj.bias"]))

    # TextBlock layers (same structure as text encoder)
    n_layers = 0
    while f"{prefix}blocks.{n_layers}.attention.wq.weight" in state:
        n_layers += 1

    for i in range(n_layers):
        bp = f"{prefix}blocks.{i}."
        op = f"irodori.spk_enc.blk.{i}."

        write_tensor(writer, f"{op}attn_norm", to_f32(state[f"{bp}attention_norm.weight"]))
        write_tensor(writer, f"{op}attn.wq", to_f16(state[f"{bp}attention.wq.weight"]))
        write_tensor(writer, f"{op}attn.wk", to_f16(state[f"{bp}attention.wk.weight"]))
        write_tensor(writer, f"{op}attn.wv", to_f16(state[f"{bp}attention.wv.weight"]))
        write_tensor(writer, f"{op}attn.wo", to_f16(state[f"{bp}attention.wo.weight"]))
        write_tensor(writer, f"{op}attn.gate", to_f16(state[f"{bp}attention.gate.weight"]))
        write_tensor(writer, f"{op}attn.q_norm", to_f32(state[f"{bp}attention.q_norm.weight"]))
        write_tensor(writer, f"{op}attn.k_norm", to_f32(state[f"{bp}attention.k_norm.weight"]))
        write_tensor(writer, f"{op}mlp_norm", to_f32(state[f"{bp}mlp_norm.weight"]))
        write_tensor(writer, f"{op}mlp.w1", to_f16(state[f"{bp}mlp.w1.weight"]))
        write_tensor(writer, f"{op}mlp.w2", to_f16(state[f"{bp}mlp.w2.weight"]))
        write_tensor(writer, f"{op}mlp.w3", to_f16(state[f"{bp}mlp.w3.weight"]))

    return n_layers


def map_dit_blocks(state: dict, writer: GGUFWriter, prefix: str = "blocks."):
    """Map DiffusionBlock layers."""
    n_layers = 0
    while f"{prefix}{n_layers}.attention.wq.weight" in state:
        n_layers += 1

    for i in range(n_layers):
        bp = f"{prefix}{i}."
        op = f"irodori.dit.blk.{i}."

        # JointAttention
        write_tensor(writer, f"{op}attn.wq", to_f16(state[f"{bp}attention.wq.weight"]))
        write_tensor(writer, f"{op}attn.wk", to_f16(state[f"{bp}attention.wk.weight"]))
        write_tensor(writer, f"{op}attn.wv", to_f16(state[f"{bp}attention.wv.weight"]))
        write_tensor(writer, f"{op}attn.wo", to_f16(state[f"{bp}attention.wo.weight"]))
        write_tensor(writer, f"{op}attn.gate", to_f16(state[f"{bp}attention.gate.weight"]))
        write_tensor(writer, f"{op}attn.q_norm", to_f32(state[f"{bp}attention.q_norm.weight"]))
        write_tensor(writer, f"{op}attn.k_norm", to_f32(state[f"{bp}attention.k_norm.weight"]))
        # Text context KV projections
        write_tensor(writer, f"{op}attn.wk_text", to_f16(state[f"{bp}attention.wk_text.weight"]))
        write_tensor(writer, f"{op}attn.wv_text", to_f16(state[f"{bp}attention.wv_text.weight"]))
        # Speaker context KV projections
        if f"{bp}attention.wk_speaker.weight" in state:
            write_tensor(writer, f"{op}attn.wk_spk", to_f16(state[f"{bp}attention.wk_speaker.weight"]))
            write_tensor(writer, f"{op}attn.wv_spk", to_f16(state[f"{bp}attention.wv_speaker.weight"]))
        # Caption context KV projections (VoiceDesign)
        if f"{bp}attention.wk_caption.weight" in state:
            write_tensor(writer, f"{op}attn.wk_cap", to_f16(state[f"{bp}attention.wk_caption.weight"]))
            write_tensor(writer, f"{op}attn.wv_cap", to_f16(state[f"{bp}attention.wv_caption.weight"]))

        # LowRankAdaLN for attention
        for comp in ("shift", "scale", "gate"):
            write_tensor(writer, f"{op}adaln_attn.{comp}_down",
                         to_f16(state[f"{bp}attention_adaln.{comp}_down.weight"]))
            write_tensor(writer, f"{op}adaln_attn.{comp}_up.w",
                         to_f16(state[f"{bp}attention_adaln.{comp}_up.weight"]))
            write_tensor(writer, f"{op}adaln_attn.{comp}_up.b",
                         to_f32(state[f"{bp}attention_adaln.{comp}_up.bias"]))

        # LowRankAdaLN for MLP
        for comp in ("shift", "scale", "gate"):
            write_tensor(writer, f"{op}adaln_mlp.{comp}_down",
                         to_f16(state[f"{bp}mlp_adaln.{comp}_down.weight"]))
            write_tensor(writer, f"{op}adaln_mlp.{comp}_up.w",
                         to_f16(state[f"{bp}mlp_adaln.{comp}_up.weight"]))
            write_tensor(writer, f"{op}adaln_mlp.{comp}_up.b",
                         to_f32(state[f"{bp}mlp_adaln.{comp}_up.bias"]))

        # SwiGLU MLP
        write_tensor(writer, f"{op}mlp.w1", to_f16(state[f"{bp}mlp.w1.weight"]))
        write_tensor(writer, f"{op}mlp.w2", to_f16(state[f"{bp}mlp.w2.weight"]))
        write_tensor(writer, f"{op}mlp.w3", to_f16(state[f"{bp}mlp.w3.weight"]))

    return n_layers


def map_duration_predictor(state: dict, writer: GGUFWriter, prefix: str = "duration_predictor."):
    """Map DurationPredictor weights (token_sum_adarn_zero architecture)."""
    if f"{prefix}token_input_proj.weight" not in state:
        return False

    write_tensor(writer, "irodori.dur.input_proj.w",
                 to_f16(state[f"{prefix}token_input_proj.weight"]))
    write_tensor(writer, "irodori.dur.input_proj.b",
                 to_f32(state[f"{prefix}token_input_proj.bias"]))

    # Null speaker embedding
    write_tensor(writer, "irodori.dur.null_speaker",
                 to_f32(state[f"{prefix}null_speaker"]))

    # Null caption embedding (VoiceDesign dual architecture)
    if f"{prefix}null_caption" in state:
        write_tensor(writer, "irodori.dur.null_caption",
                     to_f32(state[f"{prefix}null_caption"]))

    # DurationSwiGLUBlock layers
    n_layers = 0
    while f"{prefix}token_blocks.{n_layers}.norm.weight" in state:
        n_layers += 1

    for i in range(n_layers):
        bp = f"{prefix}token_blocks.{i}."
        op = f"irodori.dur.blk.{i}."

        write_tensor(writer, f"{op}norm", to_f32(state[f"{bp}norm.weight"]))
        write_tensor(writer, f"{op}mlp.w1", to_f16(state[f"{bp}mlp.w1.weight"]))
        write_tensor(writer, f"{op}mlp.w2", to_f16(state[f"{bp}mlp.w2.weight"]))
        write_tensor(writer, f"{op}mlp.w3", to_f16(state[f"{bp}mlp.w3.weight"]))
        # Speaker modulation
        write_tensor(writer, f"{op}mod.w", to_f16(state[f"{bp}modulation.weight"]))
        write_tensor(writer, f"{op}mod.b", to_f32(state[f"{bp}modulation.bias"]))
        # Caption modulation (VoiceDesign dual_adarn_zero)
        if f"{bp}caption_modulation.weight" in state:
            write_tensor(writer, f"{op}cap_mod.w", to_f16(state[f"{bp}caption_modulation.weight"]))
            write_tensor(writer, f"{op}cap_mod.b", to_f32(state[f"{bp}caption_modulation.bias"]))

    # Output
    write_tensor(writer, "irodori.dur.out_norm", to_f32(state[f"{prefix}token_out_norm.weight"]))
    write_tensor(writer, "irodori.dur.out_proj.w", to_f32(state[f"{prefix}token_out_proj.weight"]))
    write_tensor(writer, "irodori.dur.out_proj.b", to_f32(state[f"{prefix}token_out_proj.bias"]))

    return True


def map_top_level(state: dict, writer: GGUFWriter):
    """Map top-level model weights (timestep MLP, in/out projections, norms)."""
    # Timestep conditioning MLP: Linear(512, 2048) -> SiLU -> Linear(2048, 2048) -> SiLU -> Linear(2048, 6144)
    write_tensor(writer, "irodori.cond.0.w", to_f16(state["cond_module.0.weight"]))
    write_tensor(writer, "irodori.cond.2.w", to_f16(state["cond_module.2.weight"]))
    write_tensor(writer, "irodori.cond.4.w", to_f16(state["cond_module.4.weight"]))

    # Input projection: Linear(latent_dim * patch_size, model_dim)
    write_tensor(writer, "irodori.in_proj.w", to_f16(state["in_proj.weight"]))
    write_tensor(writer, "irodori.in_proj.b", to_f32(state["in_proj.bias"]))

    # Output norm + projection
    write_tensor(writer, "irodori.out_norm", to_f32(state["out_norm.weight"]))
    write_tensor(writer, "irodori.out_proj.w", to_f16(state["out_proj.weight"]))
    write_tensor(writer, "irodori.out_proj.b", to_f32(state["out_proj.bias"]))

    # Text norm (after text encoder)
    write_tensor(writer, "irodori.text_norm", to_f32(state["text_norm.weight"]))

    # Speaker norm (after speaker encoder)
    if "speaker_norm.weight" in state:
        write_tensor(writer, "irodori.spk_norm", to_f32(state["speaker_norm.weight"]))

    # Caption norm (after caption encoder, VoiceDesign)
    if "caption_norm.weight" in state:
        write_tensor(writer, "irodori.cap_norm", to_f32(state["caption_norm.weight"]))


# ── Tokenizer ─────────────────────────────────────���──────────────────

def load_tokenizer_data(tokenizer_repo: str) -> tuple[list[str], list[float], list[str], int, int]:
    """Load tokenizer vocab, scores, BPE merges, BOS and PAD IDs from HuggingFace repo."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install transformers sentencepiece")

    tok = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=True)
    vocab_size = len(tok)

    # Extract tokenizer model info from the fast tokenizer backend
    bt = tok.backend_tokenizer
    tj = json.loads(bt.to_str())
    model_info = tj.get("model", {})
    model_type = model_info.get("type", "")
    print(f"  Tokenizer model type: {model_type}")

    scores = [0.0] * vocab_size
    merges = []

    if model_type == "Unigram":
        # Unigram: extract scores directly from tokenizer.json vocab
        hf_vocab = model_info.get("vocab", [])
        for i, entry in enumerate(hf_vocab):
            if i < vocab_size and isinstance(entry, list) and len(entry) >= 2:
                scores[i] = float(entry[1])
        print(f"  Loaded {len(hf_vocab)} Unigram scores from fast tokenizer")
    elif model_type == "BPE":
        # BPE: extract merges from tokenizer.json
        raw_merges = model_info.get("merges", [])
        for m in raw_merges:
            if isinstance(m, list):
                merges.append(" ".join(m))
            else:
                merges.append(str(m))
        print(f"  Loaded {len(merges)} BPE merges from fast tokenizer")
        # Also try to load SP scores for backward compat
        try:
            import sentencepiece as spm
            sp_model = None
            if hasattr(tok, 'vocab_file') and tok.vocab_file:
                sp_model = spm.SentencePieceProcessor()
                sp_model.Load(tok.vocab_file)
            if sp_model:
                for i in range(min(vocab_size, sp_model.GetPieceSize())):
                    scores[i] = sp_model.GetScore(i)
                print(f"  Loaded {sp_model.GetPieceSize()} SentencePiece scores")
        except Exception as e:
            print(f"  WARNING: could not load SP scores: {e}")
    else:
        # Fallback: try raw SentencePiece model
        try:
            import sentencepiece as spm
            sp_model = None
            if hasattr(tok, 'vocab_file') and tok.vocab_file:
                sp_model = spm.SentencePieceProcessor()
                sp_model.Load(tok.vocab_file)
            if sp_model:
                for i in range(min(vocab_size, sp_model.GetPieceSize())):
                    scores[i] = sp_model.GetScore(i)
                print(f"  Loaded {sp_model.GetPieceSize()} SentencePiece scores")
        except Exception as e:
            print(f"  WARNING: could not load SP scores: {e}")

    # Extract normalizer info for the C++ runtime
    normalizer = tj.get("normalizer")
    has_prepend_space = False
    if normalizer:
        # Check for ▁ prepend pattern in the normalizer sequence
        norms = normalizer.get("normalizers", [normalizer])
        for n in norms:
            if n.get("type") == "Replace":
                pattern = n.get("pattern", {})
                content = n.get("content", "")
                regex = pattern.get("Regex", "")
                if content == "\u2581" and "^" in regex:
                    has_prepend_space = True
    if has_prepend_space:
        print(f"  Normalizer: prepend ▁ at start of text")

    vocab = []
    for i in range(vocab_size):
        try:
            token = tok.convert_ids_to_tokens(i)
            if token is None:
                token = f"<unk_{i}>"
            vocab.append(token)
        except Exception:
            vocab.append(f"<unk_{i}>")

    # Also get the raw tokenizer.model path for embedding
    sp_model_path = getattr(tok, 'vocab_file', None)
    if sp_model_path and not Path(sp_model_path).exists():
        sp_model_path = None

    bos_id = tok.bos_token_id if tok.bos_token_id is not None else -1
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
        tok.eos_token_id if tok.eos_token_id is not None else -1)

    return vocab, scores, merges, bos_id, pad_id, sp_model_path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Irodori-TTS to GGUF")
    parser.add_argument("--checkpoint", type=str, help="Path to .safetensors checkpoint")
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repo ID (downloads automatically)")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    parser.add_argument("--tokenizer-repo", type=str, default=None,
                        help="Tokenizer HF repo (auto-detected from checkpoint metadata if omitted)")
    parser.add_argument("--no-vocab", action="store_true",
                        help="Skip embedding tokenizer vocab (saves space)")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f16", "f32"],
                        help="Weight dtype (default: f16)")
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if args.hf_repo is None:
            sys.exit("Specify either --checkpoint or --hf-repo")
        from huggingface_hub import hf_hub_download
        # Try to find the safetensors file
        try:
            ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename="model.safetensors")
        except Exception:
            # Some repos use different names
            from huggingface_hub import list_repo_files
            files = list_repo_files(args.hf_repo)
            st_files = [f for f in files if f.endswith(".safetensors")]
            if not st_files:
                sys.exit(f"No .safetensors files found in {args.hf_repo}")
            ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=st_files[0])
        print(f"Downloaded checkpoint: {ckpt_path}")

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    state = load_safetensors(ckpt_path)

    # Extract config from safetensors metadata
    config = None
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        config_json = metadata.get("config_json")
        if config_json:
            config = json.loads(config_json)
            print(f"Found config in checkpoint metadata: {list(config.keys())}")

    # Override hparams from config if available
    hparams = dict(IRODORI_HPARAMS)
    if config:
        for key in ("latent_dim", "latent_patch_size", "model_dim", "num_layers",
                    "num_heads", "mlp_ratio", "text_dim", "text_layers", "text_heads",
                    "text_mlp_ratio", "text_vocab_size", "speaker_dim", "speaker_layers",
                    "speaker_heads", "speaker_mlp_ratio", "speaker_patch_size",
                    "timestep_embed_dim", "adaln_rank", "norm_eps",
                    "use_duration_predictor", "duration_aux_dim",
                    "duration_hidden_dim", "duration_layers",
                    # Caption (VoiceDesign)
                    "use_caption_condition", "use_speaker_condition",
                    "caption_vocab_size", "caption_dim", "caption_layers",
                    "caption_heads", "caption_mlp_ratio",
                    "duration_architecture",
                    "max_text_len", "max_caption_len"):
            if key in config:
                hparams[key] = config[key]

    # Convert booleans to ints for GGUF
    for bkey in ("use_caption_condition", "use_speaker_condition",
                 "use_duration_predictor"):
        if bkey in hparams and isinstance(hparams[bkey], bool):
            hparams[bkey] = int(hparams[bkey])

    has_caption = bool(hparams.get("use_caption_condition", 0))
    print(f"Model config: dim={hparams['model_dim']}, layers={hparams['num_layers']}, "
          f"heads={hparams['num_heads']}, text_dim={hparams['text_dim']}, "
          f"text_layers={hparams['text_layers']}, caption={has_caption}")

    # Create GGUF writer
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output_path), "irodori-tts")

    # Write hyperparameters
    for k, v in hparams.items():
        if isinstance(v, int):
            writer.add_uint32(f"irodori.{k}", v)
        elif isinstance(v, float):
            writer.add_float32(f"irodori.{k}", v)
        elif isinstance(v, str):
            writer.add_string(f"irodori.{k}", v)

    # Write tokenizer vocab + scores + BPE merges
    if not args.no_vocab:
        tokenizer_repo = args.tokenizer_repo
        if tokenizer_repo is None:
            # Auto-detect from checkpoint metadata
            tokenizer_repo = (config or {}).get("text_tokenizer_repo",
                                           "sbintuitions/sarashina2.2-0.5b")
        print(f"Loading tokenizer: {tokenizer_repo}")
        vocab, scores, merges, bos_id, pad_id, sp_model_path = load_tokenizer_data(tokenizer_repo)
        writer.add_array("tokenizer.ggml.tokens", vocab)
        writer.add_array("tokenizer.ggml.scores", [float(s) for s in scores])
        if merges:
            writer.add_array("tokenizer.ggml.merges", merges)
        writer.add_uint32("tokenizer.ggml.bos_token_id", bos_id if bos_id >= 0 else 0)
        writer.add_uint32("tokenizer.ggml.pad_token_id", pad_id if pad_id >= 0 else 0)
        writer.add_uint32(f"irodori.tokenizer.vocab_size", len(vocab))
        # Embed raw tokenizer.model as a tensor for exact SP tokenization in C++
        if sp_model_path:
            with open(sp_model_path, "rb") as f:
                sp_model_data = f.read()
            sp_arr = np.frombuffer(sp_model_data, dtype=np.int8)
            writer.add_tensor("tokenizer.ggml.raw_model", sp_arr)
            print(f"  embedded tokenizer.model as tensor ({len(sp_model_data)} bytes)")
        print(f"  vocab_size={len(vocab)}, merges={len(merges)}, bos_id={bos_id}, pad_id={pad_id}")

    # Write model weights
    print("Converting text encoder...")
    n_text = map_text_encoder(state, writer)
    print(f"  {n_text} text encoder layers")

    print("Converting speaker encoder...")
    n_spk = map_speaker_encoder(state, writer)
    print(f"  {n_spk} speaker encoder layers")

    # Caption encoder (VoiceDesign)
    if has_caption:
        print("Converting caption encoder...")
        n_cap = map_caption_encoder(state, writer)
        print(f"  {n_cap} caption encoder layers")

    print("Converting top-level weights...")
    map_top_level(state, writer)

    print("Converting DiT blocks...")
    n_dit = map_dit_blocks(state, writer)
    print(f"  {n_dit} DiT layers")

    print("Converting duration predictor...")
    has_dur = map_duration_predictor(state, writer)
    print(f"  duration predictor: {'yes' if has_dur else 'no'}")

    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = output_path.stat().st_size
    print(f"\nWrote {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  {len(state)} source tensors → GGUF")


if __name__ == "__main__":
    main()
