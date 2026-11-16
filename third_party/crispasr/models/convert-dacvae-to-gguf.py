#!/usr/bin/env python3
"""
Convert Semantic-DACVAE (Aratako/Semantic-DACVAE-Japanese-32dim) → GGUF
for the CrispASR irodori-tts backend's audio decoder.

The DACVAE decoder has the same architecture as descript/dac_44khz:
  out_proj: Conv1d(codebook_dim → 1024, k=1)   ← VAE bottleneck projection
  in_conv:  Conv1d(1024 → 1536, k=7)
  4 DecoderBlocks: strides [8, 8, 4, 2]
  out_snake + out_conv + Tanh → PCM

Usage:
    python models/convert-dacvae-to-gguf.py \\
        --model Aratako/Semantic-DACVAE-Japanese-32dim \\
        --output /mnt/storage/gguf-models/dacvae-ja-32dim-f16.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


def to_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).numpy()


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()


def write_tensor(writer: GGUFWriter, name: str, data: np.ndarray):
    if data.size <= 512 or data.ndim == 1:
        writer.add_tensor(name, data.astype(np.float32))
    else:
        writer.add_tensor(name, data.astype(np.float16))


def main():
    parser = argparse.ArgumentParser(description="Convert DACVAE to GGUF")
    parser.add_argument("--model", type=str, required=True,
                        help="HF repo ID or local path to DACVAE weights.pth")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    args = parser.parse_args()

    # Load DACVAE model
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dacvae import DACVAE
    except ImportError:
        sys.exit("pip install dacvae (or git+https://github.com/facebookresearch/dacvae.git)")

    # Resolve model path: HF repo ID → download weights.pth first
    model_path = args.model
    if not Path(model_path).exists() and "/" in model_path:
        from huggingface_hub import hf_hub_download
        print(f"Downloading DACVAE from HuggingFace: {model_path}")
        model_path = hf_hub_download(repo_id=model_path, filename="weights.pth")
        print(f"  Downloaded: {model_path}")

    print(f"Loading DACVAE: {model_path}")
    model = DACVAE.load(model_path).eval()
    state = model.state_dict()

    # Extract config
    sample_rate = int(model.sample_rate)
    hop_length = int(model.hop_length)
    latent_dim = int(model.latent_dim)
    decoder_dim = int(model.decoder_dim)
    decoder_rates = list(model.decoder_rates)
    codebook_dim = int(model.codebook_dim)

    print(f"  sample_rate={sample_rate}, hop_length={hop_length}")
    print(f"  latent_dim={latent_dim}, codebook_dim={codebook_dim}")
    print(f"  decoder_dim={decoder_dim}, decoder_rates={decoder_rates}")

    # Infer decoder channels: decoder_dim // 2^i for each block
    n_blocks = len(decoder_rates)
    decoder_channels = [decoder_dim // (2 ** i) for i in range(n_blocks + 1)]
    print(f"  decoder_channels={decoder_channels}")

    # Create GGUF
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output_path), "dacvae")

    # Write config
    writer.add_uint32("dacvae.sample_rate", sample_rate)
    writer.add_uint32("dacvae.hop_length", hop_length)
    writer.add_uint32("dacvae.latent_dim", latent_dim)
    writer.add_uint32("dacvae.codebook_dim", codebook_dim)
    writer.add_uint32("dacvae.decoder_dim", decoder_dim)
    writer.add_uint32("dacvae.n_decoder_blocks", n_blocks)

    # ── VAE bottleneck out_proj: NormConv1d(codebook_dim → latent_dim, k=1) ──
    def get_conv_weight(prefix):
        """Get effective weight from weight-normed conv."""
        if f"{prefix}.weight" in state:
            return state[f"{prefix}.weight"], state.get(f"{prefix}.bias")
        # Weight norm decomposition: weight = g * v / ||v||
        v = state[f"{prefix}.weight_v"]
        g = state[f"{prefix}.weight_g"]
        norm = torch.linalg.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
        norm = norm.unsqueeze(-1) if v.ndim == 3 else norm
        weight = g * v / (norm + 1e-12)
        bias = state.get(f"{prefix}.bias")
        return weight, bias

    # out_proj (NormConv1d stores weight at module level, not under .conv)
    w, b = get_conv_weight("quantizer.out_proj")
    print(f"  out_proj: weight {list(w.shape)}, bias {list(b.shape) if b is not None else None}")
    write_tensor(writer, "dacvae.out_proj.w", to_f16(w))
    if b is not None:
        write_tensor(writer, "dacvae.out_proj.b", to_f32(b))

    # ── Decoder ──
    # in_conv: NormConv1d(latent_dim → decoder_dim, k=7)
    w, b = get_conv_weight("decoder.model.0")
    print(f"  in_conv: weight {list(w.shape)}")
    write_tensor(writer, "dacvae.dec.in_conv.w", to_f16(w))
    if b is not None:
        write_tensor(writer, "dacvae.dec.in_conv.b", to_f32(b))

    # DecoderBlocks
    for blk_idx in range(n_blocks):
        blk_prefix = f"decoder.model.{blk_idx + 1}.block"

        # The DecoderBlock structure in DACVAE is more complex than standard DAC:
        # block[0] = Snake1d (input_dim)
        # block[1] = NormConvTranspose1d (input_dim → output_dim, k=2*stride, stride=stride)
        # block[2] = ELU
        # block[3] = NormConvTranspose1d (input_dim/3 → output_dim/3, k=2*wm_stride, stride=wm_stride)
        # block[4] = ResidualUnit(output_dim, d=1, Snake)
        # block[5] = ResidualUnit(output_dim, d=3, Snake)
        # block[6] = ResidualUnit(output_dim/3, k=3, ELU, causal, true_skip)
        # block[7] = ResidualUnit(output_dim/3, k=3, ELU, causal, true_skip)
        # block[8] = ResidualUnit(output_dim, d=9, Snake)
        # block[9] = ResidualUnit or Identity (last_kernel_size)
        # block[10] = ELU
        # block[11] = NormConv1d (downsample)

        # For now, write all decoder block weights with their original structure
        # The C++ code will need to handle the DACVAE-specific block layout
        prefix = f"decoder.model.{blk_idx + 1}"
        block_tensors = {k: v for k, v in state.items() if k.startswith(prefix)}
        for tname, tval in sorted(block_tensors.items()):
            short = tname.replace(prefix + ".", "")
            gguf_name = f"dacvae.dec.blk.{blk_idx}.{short}"
            # Reconstruct weight-normed weights
            if "weight_v" in tname:
                base = tname.replace("weight_v", "")
                g_key = base + "weight_g"
                if g_key in state:
                    v = tval
                    g = state[g_key]
                    norm = torch.linalg.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
                    if v.ndim == 3:
                        norm = norm.unsqueeze(-1)
                    eff_w = g * v / (norm + 1e-12)
                    gguf_name = gguf_name.replace(".weight_v", ".weight")
                    write_tensor(writer, gguf_name, to_f16(eff_w))
                    continue
            if "weight_g" in tname:
                continue  # handled with weight_v above
            if "alpha" in short:
                write_tensor(writer, gguf_name, to_f32(tval))
            elif tval.ndim <= 1 or tval.numel() <= 512:
                write_tensor(writer, gguf_name, to_f32(tval))
            else:
                write_tensor(writer, gguf_name, to_f16(tval))

    # Final layers: wm_model.encoder_block (used for watermark-free output path)
    # forward_no_conv runs: Snake1d(96) → Conv1d(96→1, k=7) → Tanh → Identity
    # We need: pre.0.alpha (Snake), pre.1.weight+bias (Conv 96→1)
    # pre.3 (Conv 1→32, k=7) is replaced by Identity, so skip it
    wm_prefix = "decoder.wm_model.encoder_block.pre"
    for tname, tval in sorted(state.items()):
        if not tname.startswith(wm_prefix):
            continue
        # Skip pre.3 (replaced by Identity in forward_no_conv)
        if ".pre.3." in tname:
            continue
        short = tname.replace("decoder.", "")
        # Reconstruct weight-normed convs
        if "weight_v" in tname:
            base = tname.replace("weight_v", "")
            g_key = base + "weight_g"
            if g_key in state:
                v = tval
                g = state[g_key]
                norm = torch.linalg.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
                if v.ndim == 3:
                    norm = norm.unsqueeze(-1)
                eff_w = g * v / (norm + 1e-12)
                gguf_name = f"dacvae.{short}".replace(".weight_v", ".weight")
                write_tensor(writer, gguf_name, to_f16(eff_w))
                print(f"  wm {gguf_name}: {list(eff_w.shape)}")
                continue
        if "weight_g" in tname:
            continue  # handled above
        gguf_name = f"dacvae.{short}"
        if "alpha" in tname:
            write_tensor(writer, gguf_name, to_f32(tval))
        else:
            write_tensor(writer, gguf_name, to_f32(tval))
        print(f"  wm {gguf_name}: {list(tval.shape)}")

    # ── Encoder (voice cloning: reference audio → 32-dim latent) ──
    # Mirror of the decoder; DACVAE.encode deterministic path is
    #   z = encoder(pad(audio));  mean = quantizer.in_proj(z)[:codebook_dim]
    # The encoder is a stack of NormConv1d (pad_mode="none" ⇒ symmetric
    # padding (k-stride)*dil//2 baked into the conv) + Snake1d, with 4
    # EncoderBlocks (3 ResidualUnits d=1,3,9 → Snake → strided downsample).
    encoder_rates = [int(r) for r in model.encoder_rates]
    encoder_dim = int(model.encoder_dim)
    writer.add_uint32("dacvae.has_encoder", 1)
    writer.add_uint32("dacvae.encoder_dim", encoder_dim)
    writer.add_array("dacvae.encoder_rates", encoder_rates)
    print(f"  encoder: dim={encoder_dim} rates={encoder_rates}")

    def emit_conv(state_prefix, gguf_base):
        w, b = get_conv_weight(state_prefix)
        write_tensor(writer, f"{gguf_base}.w", to_f16(w))
        if b is not None:
            write_tensor(writer, f"{gguf_base}.b", to_f32(b))
        print(f"  enc {gguf_base}: {list(w.shape)}")

    def emit_alpha(state_key, gguf_name):
        write_tensor(writer, gguf_name, to_f32(state[state_key]))

    # block.0: input conv (1 → encoder_dim, k=7)
    emit_conv("encoder.block.0", "dacvae.enc.in_conv")
    # block.1..N: EncoderBlocks
    for i, _stride in enumerate(encoder_rates):
        bp = f"encoder.block.{i + 1}.block"
        # 3 ResidualUnits (dilation 1, 3, 9)
        for r in range(3):
            rp = f"{bp}.{r}.block"
            emit_alpha(f"{rp}.0.alpha", f"dacvae.enc.blk{i}.res{r}.alpha0")
            emit_conv(f"{rp}.1", f"dacvae.enc.blk{i}.res{r}.conv0")
            emit_alpha(f"{rp}.2.alpha", f"dacvae.enc.blk{i}.res{r}.alpha1")
            emit_conv(f"{rp}.3", f"dacvae.enc.blk{i}.res{r}.conv1")
        # Snake + strided downsample conv
        emit_alpha(f"{bp}.3.alpha", f"dacvae.enc.blk{i}.down_snake.alpha")
        emit_conv(f"{bp}.4", f"dacvae.enc.blk{i}.down")
    # Final Snake + conv (encoder_final_dim → encoder_final_dim, k=3)
    n_enc = len(encoder_rates)
    emit_alpha(f"encoder.block.{n_enc + 1}.alpha", "dacvae.enc.out_snake.alpha")
    emit_conv(f"encoder.block.{n_enc + 2}", "dacvae.enc.out_conv")
    # VAE bottleneck in_proj: NormConv1d(latent_dim → 2*codebook_dim, k=1);
    # deterministic encode takes the first codebook_dim channels (the mean).
    emit_conv("quantizer.in_proj", "dacvae.enc.in_proj")

    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = output_path.stat().st_size
    print(f"\nWrote {output_path} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
