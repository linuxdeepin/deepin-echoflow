#!/usr/bin/env python3
"""
Decode an Irodori-TTS raw latent file using the DACVAE decoder (Python).

Usage:
    # First generate latent with CrispASR:
    crispasr --backend irodori-tts -m model.gguf --tts "テスト" --tts-output out.wav

    # Then decode with this script (for validation):
    python tools/irodori_decode_latent.py --latent out.latent --output decoded.wav

The .latent file is written by irodori_tts when CRISPASR_IRODORI_DUMP_LATENT=1.
Format: raw float32, shape (T_frames, 32).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=str, required=True, help="Path to .latent file (raw float32)")
    parser.add_argument("--output", type=str, required=True, help="Output WAV path")
    parser.add_argument("--model", type=str, default="Aratako/Semantic-DACVAE-Japanese-32dim",
                        help="DACVAE model (HF repo or local path)")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimensionality")
    args = parser.parse_args()

    # Load latent
    data = np.fromfile(args.latent, dtype=np.float32)
    n_frames = len(data) // args.latent_dim
    if n_frames * args.latent_dim != len(data):
        sys.exit(f"Latent file size ({len(data)}) not divisible by latent_dim ({args.latent_dim})")
    latent = torch.from_numpy(data.reshape(1, n_frames, args.latent_dim))  # (1, T, D)
    print(f"Loaded latent: {latent.shape} ({n_frames} frames)")

    # Load DACVAE
    try:
        from dacvae import DACVAE
    except ImportError:
        sys.exit("pip install dacvae (or git+https://github.com/facebookresearch/dacvae.git)")

    from huggingface_hub import hf_hub_download

    model_path = args.model
    if not Path(model_path).exists() and "/" in model_path:
        model_path = hf_hub_download(repo_id=model_path, filename="weights.pth")

    model = DACVAE.load(model_path).eval()
    # Disable watermark (same as Irodori's codec.py)
    decoder = model.decoder
    decoder.alpha = 0.0
    if hasattr(decoder, 'wm_model'):
        def _watermark_passthrough(x, message=None, _decoder=decoder):
            del message
            return _decoder.wm_model.encoder_block.forward_no_conv(x)
        decoder.watermark = _watermark_passthrough

    print(f"DACVAE loaded: sr={model.sample_rate}, hop={model.hop_length}")

    # Decode: latent (B, T, D) → transpose to (B, D, T) → out_proj → decoder
    with torch.inference_mode():
        z = latent.transpose(1, 2).contiguous()  # (1, D, T)
        audio = model.decode(z)  # (1, 1, samples)

    audio = audio.squeeze().cpu()
    # Ensure mono 1D
    if audio.ndim > 1:
        audio = audio[0] if audio.shape[0] < audio.shape[-1] else audio
        audio = audio.squeeze()
    print(f"Decoded audio: {audio.shape} ({len(audio) / model.sample_rate:.2f}s @ {model.sample_rate} Hz)")

    # Save using soundfile (torchaudio's backend may be unavailable)
    try:
        import soundfile as sf
        sf.write(args.output, audio.numpy(), model.sample_rate)
    except ImportError:
        torchaudio.save(args.output, audio.unsqueeze(0), model.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
