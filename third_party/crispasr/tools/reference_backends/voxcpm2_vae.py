"""Official VoxCPM2 AudioVAE encode/decode reference for the S2S upscaler.

This intentionally uses the upstream ``VoxCPM2Model`` helper that the full
VoxCPM2 reference backend already relies on. It captures the encoder patch
layout and the 48 kHz AudioVAE round trip for comparison with the isolated
native ``voxcpm2-vae`` backend.

Usage::

    python tools/dump_reference.py --backend voxcpm2-vae \
      --model-dir /path/to/VoxCPM2 --audio samples/jfk.wav \
      --output /tmp/voxcpm2-vae-ref.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np
import torch

DEFAULT_STAGES = ["input_16k", "vae_latent_patches", "output_48k"]


def dump(
    *,
    model_dir: Path,
    audio: np.ndarray,
    stages: Set[str],
    max_new_tokens: int = 0,
) -> Dict[str, np.ndarray]:
    del max_new_tokens
    if audio is None or len(audio) == 0:
        raise ValueError("voxcpm2-vae reference requires 16 kHz input audio")

    voxcpm_path = "/tmp/voxcpm_src"
    if voxcpm_path not in sys.path:
        sys.path.insert(0, voxcpm_path)
    try:
        from voxcpm.model.voxcpm2 import VoxCPM2Model
    except ImportError as exc:
        raise ImportError(
            "install the pinned upstream voxcpm source in /tmp/voxcpm_src "
            "(see tools/reference_envs/voxcpm2-tts/requirements.txt)"
        ) from exc

    model = VoxCPM2Model.from_local(str(model_dir), optimize=False, device="cpu")
    model.eval()

    with torch.inference_mode():
        # Mirror VoxCPM2Model._encode_wav after its 16 kHz mono file load. Feed
        # the caller's float PCM directly so the reference does not introduce
        # an unrelated 16-bit WAV quantization round-trip.
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        patch_len = model.patch_size * model.chunk_size
        remainder = waveform.size(1) % patch_len
        if remainder:
            waveform = torch.nn.functional.pad(waveform, (0, patch_len - remainder))
        encoded = model.audio_vae.encode(waveform.to(model.device), model._encode_sample_rate).cpu()
        patches = encoded.view(model.audio_vae.latent_dim, -1, model.patch_size).permute(1, 2, 0)
        if patches.ndim != 3:
            raise RuntimeError(f"unexpected AudioVAE patch shape: {tuple(patches.shape)}")

        latent = patches.reshape(-1, patches.shape[-1]).transpose(0, 1).unsqueeze(0)
        output = model.audio_vae.decode(latent).detach().float().cpu().reshape(-1)

    exact = len(audio) * 3
    if output.numel() < exact:
        raise RuntimeError(f"AudioVAE returned {output.numel()} samples, expected at least {exact}")

    results: Dict[str, np.ndarray] = {}
    if "input_16k" in stages:
        results["input_16k"] = np.ascontiguousarray(audio, dtype=np.float32)
    if "vae_latent_patches" in stages:
        results["vae_latent_patches"] = patches.detach().float().cpu().numpy()
    if "output_48k" in stages:
        results["output_48k"] = output[:exact].numpy().astype(np.float32)
    return results
