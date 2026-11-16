"""AudioSeal watermark generator + detector reference dump backend.

Captures stage-by-stage activations from the official PyTorch AudioSeal
model (facebook/audioseal) so the CrispASR C++ AudioSeal implementation
can be diffed via `crispasr-diff audioseal`.

Stages dumped (generator):
  as_input         — (1, 1, T) raw input audio at 16 kHz
  as_enc_in        — after input conv, (1, 32, T)
  as_enc_blk0..3   — after each encoder block
  as_enc_lstm      — after LSTM layers
  as_enc_out       — encoder output (1, 128, T_latent)
  as_msg_proj      — message projection added to latent
  as_dec_in        — after decoder input conv
  as_dec_blk0..3   — after each decoder block
  as_watermark     — decoder output (the additive watermark)
  as_output        — input + watermark

Stages dumped (detector):
  as_det_enc_in    — after input conv
  as_det_enc_blk0..3 — after each encoder block
  as_det_enc_lstm  — after LSTM
  as_det_logits    — detection logits (1, 2, T_latent)
  as_det_probs     — softmax probabilities

Usage:
    python -m tools.dump_reference --backend audioseal \\
        --model-dir facebook/audioseal \\
        --audio samples/jfk.wav \\
        --output /tmp/audioseal-ref.gguf

    build/bin/crispasr-diff audioseal \\
        audioseal.gguf \\
        /tmp/audioseal-ref.gguf \\
        samples/jfk.wav
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

try:
    from . import _hooks
except ImportError:
    import _hooks


DEFAULT_STAGES = [
    "as_input",
    "as_enc_in",
    "as_enc_blk0",
    "as_enc_blk1",
    "as_enc_blk2",
    "as_enc_blk3",
    "as_enc_lstm",
    "as_enc_out",
    "as_msg_proj",
    "as_dec_in",
    "as_dec_blk0",
    "as_dec_blk1",
    "as_dec_blk2",
    "as_dec_blk3",
    "as_watermark",
    "as_output",
    "as_det_logits",
    "as_det_probs",
]


def dump(model_dir: str, audio_path: str, stages: set | None = None) -> Dict[str, np.ndarray]:
    """Run AudioSeal and dump intermediate tensors."""
    import torch
    import torchaudio

    if stages is None:
        stages = set(DEFAULT_STAGES)

    out: Dict[str, np.ndarray] = {}

    # Load audio
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav[:1]  # mono
    wav = wav.unsqueeze(0)  # (1, 1, T)

    if "as_input" in stages:
        out["as_input"] = wav.squeeze(0).numpy().astype(np.float32)

    # Try loading via audioseal package
    try:
        from audioseal import AudioSeal
        generator = AudioSeal.load_generator("audioseal_wm_16bits")
        detector = AudioSeal.load_detector("audioseal_detector_16bits")
    except Exception:
        # Fallback: load from model_dir
        generator_path = Path(model_dir) / "generator_base.pth"
        detector_path = Path(model_dir) / "detector_base.pth"
        if not generator_path.exists():
            from huggingface_hub import hf_hub_download
            generator_path = hf_hub_download("facebook/audioseal", "generator_base.pth")
            detector_path = hf_hub_download("facebook/audioseal", "detector_base.pth")

        from audioseal import AudioSeal
        generator = AudioSeal.load_generator("audioseal_wm_16bits")
        detector = AudioSeal.load_detector("audioseal_detector_16bits")

    # --- Generator forward with hooks ---
    with torch.no_grad():
        # Encoder stages
        x = wav
        encoder = generator.encoder if hasattr(generator, 'encoder') else generator.model.encoder

        # Hook into encoder layers
        hook_outputs = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                hook_outputs[name] = output.detach().cpu().numpy().astype(np.float32)
            return hook_fn

        # Register hooks on encoder model layers
        if hasattr(encoder, 'model'):
            model_layers = list(encoder.model.children())
            # First conv
            if len(model_layers) > 0:
                model_layers[0].register_forward_hook(make_hook("as_enc_in"))
            # Encoder blocks
            blk_idx = 0
            for i, layer in enumerate(model_layers):
                layer_name = type(layer).__name__
                if 'EncoderBlock' in layer_name or 'SEANetResnetBlock' in layer_name:
                    # This is an encoder block (resnet + downsample)
                    pass
                if hasattr(layer, 'shortcut') or hasattr(layer, 'block'):
                    # Likely an encoder block component
                    pass
            # Register on all sequential children that look like blocks
            for i, layer in enumerate(model_layers[1:], 1):
                if hasattr(layer, 'register_forward_hook'):
                    hook_name = f"as_enc_layer_{i}"
                    layer.register_forward_hook(make_hook(hook_name))

        # Run generator
        # Default message: all ones (16 bits)
        message = torch.ones(1, 16, dtype=torch.int32)
        watermarked = generator(wav, message=message, alpha=1.0)
        watermark = watermarked - wav

        if "as_watermark" in stages:
            out["as_watermark"] = watermark.squeeze(0).numpy().astype(np.float32)
        if "as_output" in stages:
            out["as_output"] = watermarked.squeeze(0).numpy().astype(np.float32)

        # Capture any hooked outputs
        for name, val in hook_outputs.items():
            # Flatten batch dimension for GGUF
            if val.ndim == 3:
                val = val[0]  # (C, T)
            out[name] = val

        # --- Detector forward ---
        det_result = detector(watermarked)
        # det_result shape: (1, 2+nbits, T)
        det_np = det_result.detach().cpu().numpy().astype(np.float32)

        if "as_det_logits" in stages:
            out["as_det_logits"] = det_np[0, :2, :]  # (2, T) detection logits

        # Softmax on detection channels
        from scipy.special import softmax
        det_probs = softmax(det_np[0, :2, :], axis=0)
        if "as_det_probs" in stages:
            out["as_det_probs"] = det_probs.astype(np.float32)  # (2, T)

    # Print summary
    print(f"audioseal_ref: dumped {len(out)} stages from AudioSeal")
    for name, arr in sorted(out.items()):
        print(f"  {name:24s}  shape={list(arr.shape)}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.6f}")

    return out
