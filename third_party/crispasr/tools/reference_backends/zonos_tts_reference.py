#!/usr/bin/env python3
"""
Reference backend for Zonos TTS -- runs the upstream Zyphra/Zonos model
and dumps per-layer intermediate activations + output codes/PCM for
diff-testing against the C++ zonos_tts runtime.

Architecture (Zyphra/Zonos-v0.1-transformer):
  - Text encoder: character-level eSpeak phonemizer -> embedding (PLAN #130)
  - Prefix conditioner: concatenated conditioning embeddings
    (espeak, speaker, emotion, fmax, pitch_std, speaking_rate, language_id)
  - AR backbone: 26-layer transformer (d_model=2048, n_heads=16, n_kv=4, GQA)
    generating DAC audio codes (9 codebooks x 1024 entries, ~86 Hz)
  - DAC codec decoder: descript/dac_44khz (9 codebooks, 44.1 kHz output)
  - Speaker conditioning: ResNet293 + ECAPA-TDNN -> 128-d LDA embedding

Usage:
  # Full pipeline with activation dump:
  python tools/reference_backends/zonos_tts_reference.py \\
      --text "Hello world" \\
      --output /mnt/storage/zonos-tts/ref_out.wav \\
      --dump-dir /mnt/storage/zonos-tts/ref_activations/ \\
      --seed 42

  # Just generate audio (no activation dump):
  python tools/reference_backends/zonos_tts_reference.py \\
      --text "Hello world" \\
      --output /mnt/storage/zonos-tts/ref_out.wav

  # With reference speaker audio:
  python tools/reference_backends/zonos_tts_reference.py \\
      --text "Hello world" \\
      --output /mnt/storage/zonos-tts/ref_out.wav \\
      --ref-audio /path/to/speaker.wav

Prerequisites:
  pip install zonos torch torchaudio numpy soundfile
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# dump_reference.py plugin API
# ---------------------------------------------------------------------------

_N_AR_STEPS = int(os.environ.get("ZONOS_DIFF_N_STEPS", "10"))

DEFAULT_STAGES = [
    "conditioning_prefix",   # (2, prefix_len, d_model): cond+uncond stacked
    "phoneme_ids",           # (prefix_len_ph,): int32 token IDs
    "prefill_hidden",        # (2, d_model): last-token backbone hidden state at prefill [cond, uncond]
    "prefill_logits",        # (n_codebooks, head_vocab_size): CFG-blended logits from prefill
    # Per-AR-step CFG-blended logits (post-logit_bias masking), same as what the
    # sampler sees. Shape (n_codebooks, head_vocab_size) each.
    # Count controlled by ZONOS_DIFF_N_STEPS env var (default 10).
    *[f"ar_step_{k}_logits" for k in range(_N_AR_STEPS)],
    "output_codes",          # (n_codebooks, seq_len): final generated codes
]


def dump(model_dir: Path, audio: np.ndarray, stages: set, **kwargs) -> dict[str, np.ndarray]:
    """
    Run the Zonos model and capture intermediates for diff testing.

    model_dir: HF id or local path to Zyphra/Zonos-v0.1-transformer.
    audio: unused (Zonos is TTS; text comes from ZONOS_TTS_TEXT env var).
    stages: subset of DEFAULT_STAGES to capture.

    Env vars:
      ZONOS_TTS_TEXT            synthesis text (default "Hello world.")
      ZONOS_TTS_SEED            RNG seed (default 42)
      ZONOS_TTS_MAX_TOKENS      max AR steps (default 200)
      ZONOS_TTS_LANGUAGE        eSpeak language code (default "en-us")
      ZONOS_SPEAKER_EMB_PATH    path to 128-d float32 binary (int32 dim + 128 floats),
                                matches the C++ ZONOS_SPEAKER_EMB_PATH env var.
                                If not set, uses torch.randn (will differ from C++).
    """
    try:
        import torch
        import torchaudio  # noqa: F401 – needed by zonos internals
    except ImportError:
        raise SystemExit("pip install torch torchaudio")
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
    except ImportError:
        raise SystemExit("pip install git+https://github.com/Zyphra/Zonos.git")

    text = os.environ.get("ZONOS_TTS_TEXT", "Hello world.")
    seed = int(os.environ.get("ZONOS_TTS_SEED", "42"))
    max_tokens = int(os.environ.get("ZONOS_TTS_MAX_TOKENS", "200"))
    language = os.environ.get("ZONOS_TTS_LANGUAGE", "en-us")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_str = str(model_dir)
    model = Zonos.from_pretrained(model_str, device=device)

    # Load speaker embedding from file if ZONOS_SPEAKER_EMB_PATH is set.
    # File format: int32 dim (must be 128) followed by 128 float32 values.
    # This matches the C++ runtime's ZONOS_SPEAKER_EMB_PATH convention so
    # both sides use an identical embedding for apples-to-apples comparison.
    spk_path = os.environ.get("ZONOS_SPEAKER_EMB_PATH", "")
    speaker: torch.Tensor
    if spk_path and os.path.exists(spk_path):
        import struct
        with open(spk_path, "rb") as f:
            (dim,) = struct.unpack("<i", f.read(4))
            assert dim == 128, f"expected dim=128, got {dim}"
            emb = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
        speaker = torch.from_numpy(emb).to(device=device, dtype=torch.bfloat16).view(1, 1, 128)
        print(f"zonos-ref: loaded speaker embedding from {spk_path}", file=sys.stderr)
    else:
        torch.manual_seed(int(os.environ.get("ZONOS_TTS_SEED", "42")))
        speaker = torch.randn(1, 1, 128, device=device, dtype=torch.bfloat16)
        print("zonos-ref: using random speaker embedding (set ZONOS_SPEAKER_EMB_PATH for reproducible comparison)",
              file=sys.stderr)
    cond_dict = make_cond_dict(text=text, language=language, speaker=speaker, device=device)

    captures: dict[str, np.ndarray] = {}

    # conditioning_prefix — shape (2, prefix_len, d_model)
    conditioning = model.prepare_conditioning(cond_dict)
    if "conditioning_prefix" in stages:
        captures["conditioning_prefix"] = conditioning.detach().cpu().float().numpy()

    # phoneme_ids: extract from the conditioner for the same text
    if "phoneme_ids" in stages:
        try:
            from zonos.conditioning import phonemize, get_symbol_ids, BOS_ID, EOS_ID
            ph = phonemize([text], [language])[0]
            ids = [BOS_ID] + get_symbol_ids(ph) + [EOS_ID]
            captures["phoneme_ids"] = np.array(ids, dtype=np.int32)
        except Exception as e:
            print(f"phoneme_ids capture failed: {e}", file=sys.stderr)

    # prefill_hidden — last-token backbone hidden state at prefill: shape (2, d_model)
    # Captured by hooking model.backbone during _prefill, so the batch-of-2 output
    # (index 0 = cond, index 1 = uncond) is sliced at the final position.
    if "prefill_hidden" in stages:
        _backbone_hidden: list = []
        def _backbone_hook(module, inp, out):
            # out: [2, T, d_model]; take last token per batch entry
            _backbone_hidden.append(out.detach()[:, -1, :].cpu().float().numpy())
        handle = model.backbone.register_forward_hook(_backbone_hook)
        try:
            from zonos.codebook_pattern import apply_delay_pattern as _adp
            _seed_codes = torch.full((1, 9, max_tokens), -1, device=device)
            _delayed = _adp(_seed_codes, model.masked_token_id)
            _delayed_prefix = _delayed[..., :1]
            _inf_params = model.setup_cache(batch_size=2, max_seqlen=conditioning.shape[1] + max_tokens + 9)
            with torch.no_grad():
                model._prefill(conditioning, _delayed_prefix, _inf_params, cfg_scale=2.0)
        finally:
            handle.remove()
        if _backbone_hidden:
            captures["prefill_hidden"] = _backbone_hidden[0]  # shape (2, d_model)

    # prefill_logits + ar_step_K_logits + output_codes — run generate() with hook
    n_steps = _N_AR_STEPS
    need_generate = (
        "prefill_logits" in stages
        or "output_codes" in stages
        or any(f"ar_step_{k}_logits" in stages for k in range(n_steps))
    )
    if need_generate:
        torch.manual_seed(seed)

        # Hook sample_from_logits (imported in zonos.model) to capture the exact
        # distribution used for sampling at each call:
        #   call index 0 → prefill_logits   (CFG-blended, no logit_bias yet)
        #   call index k (k ≥ 1) → ar_step_{k-1}_logits  (CFG-blended + logit_bias)
        # The logits tensor has shape (1, n_codebooks, vocab_size) at each call.
        import zonos.model as _zonos_model
        _call_idx: list[int] = [0]
        _ar_captures: dict[str, np.ndarray] = {}
        _orig_sample = _zonos_model.sample_from_logits

        def _hooked_sample(logits, **kwargs):
            idx = _call_idx[0]
            _call_idx[0] += 1
            arr = logits.detach().squeeze(0).cpu().float().numpy()  # (n_cb, vocab)
            if idx == 0:
                if "prefill_logits" in stages:
                    _ar_captures["prefill_logits"] = arr
            else:
                k = idx - 1
                stage = f"ar_step_{k}_logits"
                if stage in stages:
                    _ar_captures[stage] = arr
            return _orig_sample(logits, **kwargs)

        _zonos_model.sample_from_logits = _hooked_sample
        try:
            codes = model.generate(
                conditioning,
                max_new_tokens=max_tokens,
                cfg_scale=2.0,
                progress_bar=False,
                disable_torch_compile=True,
            )
        finally:
            _zonos_model.sample_from_logits = _orig_sample

        captures.update(_ar_captures)
        if "output_codes" in stages:
            captures["output_codes"] = codes.detach().cpu().numpy().astype(np.int32)

    return captures


# ---------------------------------------------------------------------------
# Standalone script (unchanged from original)
# ---------------------------------------------------------------------------


def _hook_factory(dump_dir: str, layer_name: str) -> callable:
    """Create a forward hook that saves output activations as .npy."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if hasattr(out, 'detach'):
            arr = out.detach().cpu().float().numpy()
        else:
            arr = np.array(out)
        path = os.path.join(dump_dir, f"{layer_name}.npy")
        np.save(path, arr)
    return hook


def main():
    ap = argparse.ArgumentParser(description="Zonos TTS reference backend for diff testing")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--output", required=True, help="Output WAV path")
    ap.add_argument("--model", default="Zyphra/Zonos-v0.1-transformer",
                    help="HF model ID or local path")
    ap.add_argument("--ref-audio", default="",
                    help="Reference audio for speaker cloning")
    ap.add_argument("--dump-dir", default="",
                    help="Directory to dump per-layer activations as .npy")
    ap.add_argument("--dump-codes", default="",
                    help="Dump raw DAC codes to text file")
    ap.add_argument("--language", default="en-us",
                    help="Language code (eSpeak)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=86 * 30,
                    help="Maximum number of AR decode steps")
    ap.add_argument("--cfg-scale", type=float, default=2.0,
                    help="Classifier-free guidance scale")

    # Conditioning parameters
    ap.add_argument("--pitch-std", type=float, default=20.0)
    ap.add_argument("--speaking-rate", type=float, default=15.0)
    ap.add_argument("--fmax", type=float, default=22050.0)
    ap.add_argument("--emotion", type=float, nargs=8,
                    default=[0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077],
                    help="8-element emotion vector (Happiness, Sadness, Disgust, Fear, "
                         "Surprise, Anger, Other, Neutral)")
    args = ap.parse_args()

    try:
        import torch
        import torchaudio
        import soundfile as sf
    except ImportError:
        sys.exit("pip install torch torchaudio soundfile")

    # Import Zonos
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
    except ImportError:
        sys.exit("pip install zonos  (or: pip install git+https://github.com/Zyphra/Zonos.git)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} on {device}...", file=sys.stderr)
    model = Zonos.from_pretrained(args.model, device=device)

    # Speaker embedding
    speaker = None
    if args.ref_audio:
        print(f"Extracting speaker embedding from {args.ref_audio}...", file=sys.stderr)
        wav, sr = torchaudio.load(args.ref_audio)
        speaker = model.make_speaker_embedding(wav, sr)
    else:
        print("No ref audio; using random speaker embedding", file=sys.stderr)
        speaker = torch.randn(1, 1, 128, device=device, dtype=torch.bfloat16)

    # Build conditioning
    cond_dict = make_cond_dict(
        text=args.text,
        language=args.language,
        speaker=speaker,
        emotion=args.emotion,
        fmax=args.fmax,
        pitch_std=args.pitch_std,
        speaking_rate=args.speaking_rate,
        device=device,
    )

    # Set up activation hooks if dump_dir is requested
    hooks = []
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

        # Hook the prefix conditioner output
        hooks.append(model.prefix_conditioner.register_forward_hook(
            _hook_factory(args.dump_dir, "prefix_conditioner_output")))

        # Hook each backbone transformer layer
        for i, layer in enumerate(model.backbone.layers):
            hooks.append(layer.register_forward_hook(
                _hook_factory(args.dump_dir, f"backbone_layer_{i:02d}")))

        # Hook the backbone final norm
        hooks.append(model.backbone.norm_f.register_forward_hook(
            _hook_factory(args.dump_dir, "backbone_norm_f")))

        # Hook individual attention and MLP sub-modules in each layer
        for i, layer in enumerate(model.backbone.layers):
            hooks.append(layer.mixer.register_forward_hook(
                _hook_factory(args.dump_dir, f"backbone_layer_{i:02d}_attn")))
            hooks.append(layer.mlp.register_forward_hook(
                _hook_factory(args.dump_dir, f"backbone_layer_{i:02d}_mlp")))

    # Prepare conditioning
    conditioning = model.prepare_conditioning(cond_dict)

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "conditioning_prefix.npy"),
                conditioning.detach().cpu().float().numpy())

    # Generate
    torch.manual_seed(args.seed)
    print(f"Generating (max_tokens={args.max_tokens}, cfg={args.cfg_scale})...",
          file=sys.stderr)
    codes = model.generate(
        conditioning,
        max_new_tokens=args.max_tokens,
        cfg_scale=args.cfg_scale,
        progress_bar=True,
    )

    # Remove hooks
    for h in hooks:
        h.remove()

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "output_codes.npy"),
                codes.detach().cpu().numpy())

    # Dump codes as text
    if args.dump_codes:
        codes_np = codes.detach().cpu().numpy()
        with open(args.dump_codes, "w") as f:
            # Shape: (batch, n_codebooks, seq_len)
            for cb in range(codes_np.shape[1]):
                f.write(f"codebook_{cb}: ")
                f.write(" ".join(str(int(c)) for c in codes_np[0, cb]))
                f.write("\n")
        print(f"Wrote codes to {args.dump_codes}", file=sys.stderr)

    # Decode to audio
    print("Decoding codes to audio...", file=sys.stderr)
    wavs = model.autoencoder.decode(codes).cpu()
    wav_np = wavs[0, 0].numpy()
    sr = model.autoencoder.sampling_rate  # 44100

    if args.dump_dir:
        np.save(os.path.join(args.dump_dir, "output_pcm.npy"), wav_np)

    # Save output WAV
    sf.write(args.output, wav_np, sr)
    print(f"Wrote {args.output} ({len(wav_np)/sr:.2f}s, {sr} Hz)", file=sys.stderr)

    # Print summary
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Model: {args.model}", file=sys.stderr)
    print(f"  Text: {args.text!r}", file=sys.stderr)
    print(f"  Language: {args.language}", file=sys.stderr)
    print(f"  Codes shape: {list(codes.shape)}", file=sys.stderr)
    print(f"  Audio: {len(wav_np)} samples @ {sr} Hz = {len(wav_np)/sr:.2f}s",
          file=sys.stderr)
    if args.dump_dir:
        n_files = len([f for f in os.listdir(args.dump_dir) if f.endswith('.npy')])
        print(f"  Activation dumps: {n_files} files in {args.dump_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
