"""TADA encoder reference dump backend for crispasr-diff.

Runs the TADA encoder pipeline (Aligner + WavEncoder + LocalAttentionEncoder)
and captures intermediate activations for parity testing against the C++
implementation.

Pipeline:
  1. Aligner: audio (24kHz→16kHz) → wav2vec2 CTC logits → DP alignment → positions
  2. WavEncoder: audio (24kHz, padded) → strided conv → 50Hz features (T, 1024)
  3. LocalAttentionEncoder: features + pos_emb → segment attention → (T, 1024)
  4. hidden_linear: (T, 1024) → (T, 512)
  5. Post-processing: zero non-token, add noise, gather, normalize → token_values

Environment variables:
  TADA_ENCODER_TEXT     — transcript of the reference audio
  TADA_ENCODER_LANG     — language code (ar, ch, de, es, fr, it, ja, pl, pt; omit for en)
  TADA_DEVICE           — "cpu" or "cuda" (default: "cpu")
  TADA_SEED             — random seed (default: 42)
  TADA_CODEC_DIR        — local path to HumeAI/tada-codec (optional)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    # Aligner stages
    "aligner_logits",       # CTC logits (T_16k, vocab_size) — truncated to save space
    "aligner_positions",    # DP alignment positions (N_tokens,)
    "aligner_token_masks",  # Binary mask (T_50hz,) — 1 at token positions

    # WavEncoder stages
    "encoder_wav_out",      # WavEncoder output (T_50hz, 1024)

    # LocalAttentionEncoder stages
    "encoder_attn_input",   # Input to attention (after pos_emb addition) (T_50hz, 1024)
    "encoder_attn_out",     # LocalAttentionEncoder output (T_50hz, 1024)

    # Post-processing
    "encoder_hidden",       # After hidden_linear (T_50hz, 512)
    "encoder_token_values", # Final gathered+normalized (N_tokens, 512)
]

DEFAULT_TRANSCRIPT = "Please call Stella."


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int = 256) -> Dict[str, np.ndarray]:
    import torch
    import torchaudio
    torch.set_grad_enabled(False)

    out: Dict[str, np.ndarray] = {}

    device = os.environ.get("TADA_DEVICE", "cpu")
    transcript = os.environ.get("TADA_ENCODER_TEXT", DEFAULT_TRANSCRIPT)
    language = os.environ.get("TADA_ENCODER_LANG") or None
    seed = int(os.environ.get("TADA_SEED", "42"))
    codec_dir = os.environ.get("TADA_CODEC_DIR")

    torch.manual_seed(seed)

    print(f"  transcript: {transcript!r}")
    print(f"  language: {language or 'en'}")

    # Patch for transformers >= 5.x: all_tied_weights_keys was renamed/removed
    import transformers
    if not hasattr(transformers.PreTrainedModel, 'all_tied_weights_keys'):
        def _get_tied(self):
            return getattr(self, '_all_tied_weights_keys_store', None) or \
                   getattr(self, '_tied_weights_keys', None) or {}
        def _set_tied(self, val):
            self._all_tied_weights_keys_store = val
        transformers.PreTrainedModel.all_tied_weights_keys = property(_get_tied, _set_tied)

    # Patch tokenizer for gated Llama — redirect to unsloth mirror
    from transformers import AutoTokenizer
    _orig_tok = AutoTokenizer.from_pretrained.__func__
    @classmethod
    def _patched_tok(cls, name, *a, **kw):
        try:
            return _orig_tok(cls, name, *a, **kw)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e):
                return _orig_tok(cls, "unsloth/Llama-3.2-1B", *a, **kw)
            raise
    AutoTokenizer.from_pretrained = _patched_tok

    import gc
    from tada.modules.aligner import Aligner
    from tada.modules.encoder import Encoder
    from tada.utils.text import normalize_text

    # ── Process audio ──
    audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    audio_24k = torchaudio.functional.resample(audio_t, 16000, 24000)
    print(f"  audio: {audio_24k.shape[-1]} samples @ 24kHz ({audio_24k.shape[-1]/24000:.2f}s)")

    # ── Step 1: Aligner (load separately, run, then free to save RAM) ──
    repo = codec_dir or "HumeAI/tada-codec"
    aligner_sub = f"aligner-{language}" if language else "aligner"
    print(f"  loading Aligner from {repo}/{aligner_sub}")
    aligner = Aligner.from_pretrained(repo, subfolder=aligner_sub).float().to(device)
    aligner.eval()

    norm_text = normalize_text(transcript)
    text_tokens = aligner.tokenizer.encode(norm_text, add_special_tokens=False)
    text_tokens_t = torch.tensor([text_tokens], device=device)
    audio_length = torch.tensor([audio_24k.shape[-1]], device=device)

    with torch.no_grad():
        align_output = aligner(
            audio_24k,
            text_tokens=text_tokens_t,
            audio_length=audio_length,
            sample_rate=24000,
            return_logits=True,
        )

    token_positions = align_output.token_positions.cpu()
    token_masks = align_output.token_masks.cpu()

    if "aligner_logits" in stages and align_output.logits is not None:
        logits = align_output.logits[0].cpu().float().numpy()
        max_frames = min(100, logits.shape[0])
        max_vocab = min(1000, logits.shape[1])
        out["aligner_logits"] = logits[:max_frames, :max_vocab].copy()

    if "aligner_positions" in stages:
        out["aligner_positions"] = token_positions[0].float().numpy()

    if "aligner_token_masks" in stages:
        out["aligner_token_masks"] = token_masks[0].float().numpy()

    print(f"  aligner: {len(text_tokens)} tokens → {int(token_masks[0].sum())} aligned positions")

    # Free aligner to reclaim ~1.3 GB before loading encoder
    del aligner, align_output
    gc.collect()
    print(f"  aligner freed")

    # ── Step 2: Load Encoder (WavEncoder + LocalAttention only, no aligner) ──
    print(f"  loading Encoder from {repo}/encoder")
    encoder = Encoder.from_pretrained(repo, subfolder="encoder").float().to(device)
    encoder.eval()

    # Run WavEncoder directly
    x = audio_24k
    with torch.no_grad():
        wav_out = encoder.wav_encoder(
            torch.nn.functional.pad(x.unsqueeze(1), (0, 960), value=0)
        ).transpose(1, 2)  # (1, T, hidden_dim=1024)

    if "encoder_wav_out" in stages:
        out["encoder_wav_out"] = wav_out[0].cpu().float().numpy()
    print(f"  wav_encoder: {wav_out.shape[1]} frames × {wav_out.shape[2]}d")

    # ── Step 3: Add pos_emb + LocalAttentionEncoder ──
    seq_len = wav_out.shape[1]
    padded_masks = torch.nn.functional.pad(token_masks.to(device), (0, seq_len - token_masks.shape[1]), value=0)

    attn_input = wav_out + encoder.pos_emb(padded_masks)

    if "encoder_attn_input" in stages:
        out["encoder_attn_input"] = attn_input[0].cpu().float().numpy()

    # Build segment attention mask
    from tada.modules.encoder import _create_segment_attention_mask
    attn_mask = _create_segment_attention_mask(padded_masks, version=encoder.config.block_attention)

    with torch.no_grad():
        attn_out = encoder.local_attention_encoder(attn_input, mask=attn_mask)

    if "encoder_attn_out" in stages:
        out["encoder_attn_out"] = attn_out[0].cpu().float().numpy()
    print(f"  local_attention: {attn_out.shape[1]} × {attn_out.shape[2]}d")

    # ── Step 4: hidden_linear ──
    hidden_out = encoder.hidden_linear(attn_out)

    if "encoder_hidden" in stages:
        out["encoder_hidden"] = hidden_out[0].cpu().float().numpy()

    # ── Step 5: Post-processing ──
    # Zero out non-token frames
    encoded_expanded = torch.where(
        padded_masks.unsqueeze(-1) == 0,
        torch.zeros_like(hidden_out),
        hidden_out,
    )

    # Sample (add noise)
    torch.manual_seed(seed)  # Reset for deterministic noise
    encoded_expanded_sampled = torch.where(
        padded_masks.unsqueeze(-1) == 0,
        encoded_expanded,
        encoder.sample(encoded_expanded, dist_type=encoder.config.dist_type),
    )

    # Gather at token positions
    token_positions_dev = token_positions.to(device)
    token_values = torch.gather(
        encoded_expanded_sampled,
        1,
        (token_positions_dev - 1).clamp(min=0).unsqueeze(-1).expand(-1, -1, encoded_expanded_sampled.shape[-1]),
    )
    # Normalize
    token_values = (token_values - encoder.config.acoustic_mean) / encoder.config.acoustic_std

    if "encoder_token_values" in stages:
        out["encoder_token_values"] = token_values[0].cpu().float().numpy()
    print(f"  token_values: {token_values.shape[1]} tokens × {token_values.shape[2]}d")

    # Store metadata
    out["tada_encoder_text"] = transcript

    return out
