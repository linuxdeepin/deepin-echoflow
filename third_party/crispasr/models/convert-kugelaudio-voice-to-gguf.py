#!/usr/bin/env python3
"""Convert a voice reference WAV to a KugelAudio voice GGUF.

Runs the acoustic encoder on a reference audio clip to produce
pre-encoded voice embeddings (acoustic_mean), then stores them
in a small GGUF file that can be loaded by kugelaudio_load_voice().

Requires the full model (with encoders) — run on Kaggle GPU or a
machine with 19+ GB VRAM.

Usage (on Kaggle):
  python models/convert-kugelaudio-voice-to-gguf.py \\
      --model kugelaudio/kugelaudio-0-open \\
      --audio reference_voice.wav \\
      --output kugelaudio-voice-custom.gguf

Usage (from existing .pt voice):
  python models/convert-kugelaudio-voice-to-gguf.py \\
      --voice-pt voices/default.pt \\
      --output kugelaudio-voice-default.gguf
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def encode_voice_from_audio(model_id: str, audio_path: str, device: str = "cuda"):
    """Encode a WAV file through the acoustic encoder to get voice embeddings."""
    import torchaudio

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 24000:
        waveform = torchaudio.functional.resample(waveform, sr, 24000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)  # mono

    # Load model with encoders (don't strip them!)
    try:
        from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
    except ImportError:
        raise RuntimeError("kugelaudio_open package required for voice encoding")

    # Encode through acoustic tokenizer
    with torch.no_grad():
        audio_input = waveform.to(device=device, dtype=torch.bfloat16)
        encoder_output = model.model.acoustic_tokenizer.encode(audio_input.unsqueeze(0))
        acoustic_mean = encoder_output.mean  # [1, T_frames, vae_dim]

    acoustic_mean_f32 = acoustic_mean.cpu().float().numpy()

    # Get scaling factors
    scaling = model.model.speech_scaling_factor.cpu().float().item()
    bias = model.model.speech_bias_factor.cpu().float().item()

    return acoustic_mean_f32, scaling, bias


def load_voice_pt(pt_path: str):
    """Load a pre-encoded .pt voice file."""
    voice = torch.load(pt_path, map_location="cpu", weights_only=True)
    acoustic_mean = voice["acoustic_mean"].float().numpy()
    acoustic_std = voice.get("acoustic_std")
    if acoustic_std is not None:
        acoustic_std = float(acoustic_std) if isinstance(acoustic_std, (int, float)) else acoustic_std.float().item()
    return acoustic_mean, acoustic_std


def write_voice_gguf(output_path: str, acoustic_mean: np.ndarray,
                     scaling: float = 0.0, bias: float = 0.0,
                     acoustic_std: float = 0.5, voice_name: str = "custom",
                     cloned_from_recording: bool = False,
                     consent_attestation: str = ""):
    """Write voice embeddings to a GGUF file."""
    writer = gguf.GGUFWriter(output_path, "kugelaudio-voice")
    writer.add_name(f"kugelaudio-voice-{voice_name}")

    # Voice-clone provenance — see examples/cli/crispasr_voice_clone_policy.h.
    #
    # Stamped per pack rather than declared for the whole `kugelaudio-voice`
    # architecture, because this script has two modes: --audio bakes a real
    # person's recording (a clone), --voice-pt converts an upstream pre-encoded
    # voice (a preset). Listing the architecture as recording-derived would gate
    # the presets too; the stamp is the only predicate that can tell them apart.
    if cloned_from_recording:
        writer.add_bool("crispasr.voice.cloned_from_recording", True)
        if consent_attestation:
            writer.add_string("crispasr.voice.consent_attestation", consent_attestation)

    # Voice metadata
    writer.add_string("kugelaudio.voice.name", voice_name)
    if acoustic_mean.ndim == 3:
        n_frames = acoustic_mean.shape[1]
        vae_dim = acoustic_mean.shape[2]
    elif acoustic_mean.ndim == 2:
        n_frames = acoustic_mean.shape[0]
        vae_dim = acoustic_mean.shape[1]
    else:
        raise ValueError(f"unexpected acoustic_mean shape: {acoustic_mean.shape}")

    writer.add_uint32("kugelaudio.voice.n_frames", n_frames)
    writer.add_uint32("kugelaudio.voice.vae_dim", vae_dim)
    writer.add_float32("kugelaudio.voice.acoustic_std", acoustic_std)
    if scaling != 0.0:
        writer.add_float32("kugelaudio.voice.scaling_factor", scaling)
        writer.add_float32("kugelaudio.voice.bias_factor", bias)

    # Voice embedding tensor
    data = acoustic_mean.reshape(-1).astype(np.float32)
    writer.add_tensor("voice.acoustic_mean", data,
                      raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path)
    print(f"wrote {output_path}: {n_frames} frames, vae_dim={vae_dim}, {file_size} bytes")


def main():
    parser = argparse.ArgumentParser(description="Convert voice to KugelAudio GGUF")
    parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open",
                        help="HF model ID (needed for --audio encoding)")
    parser.add_argument("--audio", help="WAV file to encode as voice reference")
    parser.add_argument("--voice-pt", help="Pre-encoded .pt voice file to convert")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--name", default="custom", help="Voice name")
    parser.add_argument("--device", default="cuda", help="Device for encoding")
    parser.add_argument("--i-have-rights", dest="i_have_rights", action="store_true",
                        help="attest that you have the consent of the speaker in --audio, or "
                             "that it is your own voice. Required with --audio: encoding a "
                             "recording bakes a real person's voice into a reusable clone. "
                             "Not needed for --voice-pt, which converts an upstream preset.")
    parser.add_argument("--consent-attestation", default="",
                        help="free-text attestation recorded in the pack metadata (audit trail; "
                             "does not replace --i-have-rights at synthesis time)")
    args = parser.parse_args()

    if args.voice_pt:
        # Upstream pre-encoded voice: no recording of a natural person enters
        # here, so no attestation and no clone stamp. Same reasoning that keeps
        # kokoro and vibevoice packs ungated.
        acoustic_mean, acoustic_std = load_voice_pt(args.voice_pt)
        write_voice_gguf(args.output, acoustic_mean,
                         acoustic_std=acoustic_std or 0.5,
                         voice_name=args.name)
    elif args.audio:
        # Consent gate, mirroring the CLI's --i-have-rights and the other
        # bakers. Baking IS the cloning step: the acoustic encoder turns the
        # recording into a reusable speaker identity, and everything downstream
        # just replays it.
        if not args.i_have_rights:
            raise SystemExit(
                "encoding a voice from --audio requires --i-have-rights.\n"
                "\n"
                "  By passing --i-have-rights you attest:\n"
                '  "I have the consent of the speaker whose voice this clones,\n'
                '   or it is my own voice."\n'
                "\n"
                "  --voice-pt converts an upstream preset instead and needs no attestation.\n"
            )
        acoustic_mean, scaling, bias = encode_voice_from_audio(
            args.model, args.audio, args.device)
        write_voice_gguf(args.output, acoustic_mean,
                         scaling=scaling, bias=bias,
                         voice_name=args.name,
                         cloned_from_recording=True,
                         consent_attestation=args.consent_attestation)
    else:
        parser.error("either --audio or --voice-pt is required")


if __name__ == "__main__":
    main()
