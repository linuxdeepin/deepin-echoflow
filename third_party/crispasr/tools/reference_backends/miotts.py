#!/usr/bin/env python3
"""
MioTTS reference dumper for the CrispASR diff harness.

Runs the MioTTS-0.6B LLM (Qwen3ForCausalLM) to generate speech tokens from
a text prompt, then runs MioCodec decode to produce a waveform. Dumps
per-stage intermediates to a GGUF archive for comparison with the C++ port.

Stages dumped:
  1. input_ids        — tokenized prompt (int32)
  2. logits_step_0    — first LLM logits (sanity check embedding + first layer)
  3. generated_ids    — full generated token sequence (int32)
  4. speech_tokens    — extracted speech token indices for codec (int32)
  5. fsq_embedding    — FSQ dequantized content embedding (float32)
  6. wave_prenet_out  — after wave_prenet transformer (float32)
  7. wave_decoder_out — after wave_decoder transformer (float32)
  8. audio_output     — final 24kHz waveform (float32)

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    TMPDIR=/mnt/volume1/tmp-overflow \\
    python tools/reference_backends/miotts.py \\
        --llm-dir /mnt/storage/models/miotts-0.6b \\
        --codec-dir /mnt/storage/models/miocodec-25hz-24khz \\
        --text "Hello world" \\
        --output /mnt/volume1/tmp-overflow/miotts-ref.gguf \\
        --max-tokens 50
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


DEFAULT_STAGES = [
    "input_ids",
    "logits_step_0",
    "generated_ids",
    "speech_tokens",
    "fsq_embedding",
    "wave_prenet_out",
    "wave_decoder_out",
    "audio_output",
]


def dump(args):
    """Main dump entry point for the diff harness."""
    writer = GGUFWriter(args.output, "miotts-ref", use_temp_file=True)

    # ─── Stage 1: LLM inference ─────────────────────────────────────
    print("[miotts-ref] Loading LLM...")
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.llm_dir) / "tokenizer.json"),
    )
    # Load chat template and special tokens
    import json as _json
    _tok_cfg_path = Path(args.llm_dir) / "tokenizer_config.json"
    if _tok_cfg_path.exists():
        with open(_tok_cfg_path) as _f:
            _tok_cfg = _json.load(_f)
        if "chat_template" in _tok_cfg and _tok_cfg["chat_template"]:
            tokenizer.chat_template = _tok_cfg["chat_template"]
        if "eos_token" in _tok_cfg:
            tokenizer.eos_token = _tok_cfg["eos_token"]
        if "pad_token" in _tok_cfg:
            tokenizer.pad_token = _tok_cfg["pad_token"]
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
    # Fallback: load chat_template.jinja if not in tokenizer_config
    if not getattr(tokenizer, 'chat_template', None):
        _jinja_path = Path(args.llm_dir) / "chat_template.jinja"
        if _jinja_path.exists():
            with open(_jinja_path) as _f:
                tokenizer.chat_template = _f.read()
    # Final fallback: hardcode ChatML template
    if not getattr(tokenizer, 'chat_template', None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    # Build ChatML prompt: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    messages = [{"role": "user", "content": args.text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"[miotts-ref] Prompt: {repr(prompt[:80])}...")
    print(f"[miotts-ref] Input IDs shape: {input_ids.shape}")

    # Dump input_ids (stored as float32 for diff harness compatibility)
    writer.add_tensor("input_ids", input_ids[0].numpy().astype(np.int32).astype(np.float32))

    # Load model
    print("[miotts-ref] Loading model (this may take a moment on 8GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_dir,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    # Dump the embedding lookup for the input_ids (structural gate)
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)  # (1, T, d_model)
        writer.add_tensor("token_embed", embeds[0].float().numpy())
        print(f"[miotts-ref] token_embed shape: {embeds[0].shape}")

    # Get first-step logits
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        logits_step_0 = outputs.logits[0, -1, :].float().numpy()
    writer.add_tensor("logits_step_0", logits_step_0)
    print(f"[miotts-ref] logits_step_0 shape: {logits_step_0.shape}")

    # Generate speech tokens
    print(f"[miotts-ref] Generating (max_tokens={args.max_tokens})...")
    max_new = args.max_tokens
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else 1.0,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    generated_ids = generated[0].numpy().astype(np.int32)
    writer.add_tensor("generated_ids", generated_ids.astype(np.float32))
    print(f"[miotts-ref] Generated {len(generated_ids)} tokens total")

    # Extract speech token indices
    # Speech tokens are in range [151669, 164469) → codec indices [0, 12800)
    speech_start = 151669
    speech_end = 164469
    new_tokens = generated_ids[input_ids.shape[1]:]
    speech_mask = (new_tokens >= speech_start) & (new_tokens < speech_end)
    speech_token_ids = new_tokens[speech_mask] - speech_start
    writer.add_tensor("speech_tokens", speech_token_ids.astype(np.float32))
    print(f"[miotts-ref] Speech tokens: {len(speech_token_ids)}")

    if len(speech_token_ids) == 0:
        print("[miotts-ref] WARNING: no speech tokens generated!")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return

    # Free LLM memory
    del model, outputs, generated
    gc.collect()

    # ─── Stage 2: MioCodec decode ───────────────────────────────────
    print("[miotts-ref] Loading MioCodec...")

    # We need to load the codec manually since `miocodec` pip package
    # may not be available. Load from safetensors directly and implement
    # the decode path step by step.
    from safetensors import safe_open

    codec_path = Path(args.codec_dir) / "model.safetensors"
    sf = safe_open(str(codec_path), framework="pt")

    # FSQ dequantize: indices → codes → embedding
    # FSQ levels = [8, 8, 8, 5, 5], basis = [1, 8, 64, 512, 2560]
    levels = torch.tensor([8, 8, 8, 5, 5], dtype=torch.long)
    basis = torch.tensor([1, 8, 64, 512, 2560], dtype=torch.long)

    indices = torch.tensor(speech_token_ids, dtype=torch.long).unsqueeze(-1)  # (T, 1)
    # indices_to_codes: (index // basis) % levels → (T, 5)
    codes_non_centered = (indices // basis) % levels
    # scale_and_shift_inverse: (codes - levels//2) / (levels//2)
    half_width = levels // 2
    codes_normalized = (codes_non_centered.float() - half_width.float()) / half_width.float()

    # Project from FSQ dim (5) to embedding dim (768)
    # Weight: local_quantizer.out_proj.weight (768, 5)
    # Bias: local_quantizer.out_proj.bias (768,)
    has_proj_out = "local_quantizer.proj_out.weight" in sf.keys()
    if has_proj_out:
        proj_out_w = sf.get_tensor("local_quantizer.proj_out.weight").float()
        proj_out_b = sf.get_tensor("local_quantizer.proj_out.bias").float()
        fsq_embedding = codes_normalized @ proj_out_w.T + proj_out_b  # (T, 768)
    else:
        # Maybe the quantizer uses a different projection name
        print("[miotts-ref] WARNING: no proj_out found, using raw codes")
        fsq_embedding = codes_normalized

    writer.add_tensor("fsq_embedding", fsq_embedding.numpy())
    print(f"[miotts-ref] FSQ embedding shape: {fsq_embedding.shape}")

    # ─── wave_prenet forward (6 layers, 768d, windowed attention) ──────
    # This is a standard bidirectional transformer with SwiGLU FFN.
    # We implement it step-by-step against the safetensors weights.
    print("[miotts-ref] Running wave_prenet (6 layers)...")

    def layer_norm(x, w, b, eps=1e-5):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + eps).sqrt() * w + b

    def swiglu_ffn(x, w1, w2, w3):
        # SwiGLU: gate = silu(x @ w1.T), up = x @ w3.T, out = (gate * up) @ w2.T
        gate = torch.nn.functional.silu(x @ w1.T)
        up = x @ w3.T
        return (gate * up) @ w2.T

    def rope_freqs(seq_len, dim, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return torch.complex(torch.cos(freqs), torch.sin(freqs))

    def apply_rope(x, freqs_cis):
        # x: (T, n_heads, head_dim) → complex view
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = freqs_cis[:x_.shape[0]].unsqueeze(1)  # (T, 1, head_dim//2)
        return torch.view_as_real(x_ * freqs).flatten(-2).type_as(x)

    def windowed_attention(x, wq, wk, wv, wo, n_heads, window_size=65):
        T, d = x.shape
        hd = d // n_heads
        Q = (x @ wq.T).reshape(T, n_heads, hd)
        K = (x @ wk.T).reshape(T, n_heads, hd)
        V = (x @ wv.T).reshape(T, n_heads, hd)
        # RoPE
        freqs = rope_freqs(T, hd, theta=10000.0)
        Q = apply_rope(Q, freqs)
        K = apply_rope(K, freqs)
        # Attention with window mask
        Q = Q.permute(1, 0, 2)  # (n_heads, T, hd)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)
        scale = hd ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (n_heads, T, T)
        # Window mask: only attend within window_size//2 on each side
        w = window_size // 2
        mask = torch.ones(T, T, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=-w) & torch.tril(mask, diagonal=w)
        scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (n_heads, T, hd)
        out = out.permute(1, 0, 2).reshape(T, d)  # (T, d)
        return out @ wo.T

    x = fsq_embedding  # (T, 768)
    for il in range(6):
        prefix = f"wave_prenet.layers.{il}"
        # Pre-attention LayerNorm
        an_w = sf.get_tensor(f"{prefix}.attention_norm.weight").float()
        an_b = sf.get_tensor(f"{prefix}.attention_norm.bias").float()
        wq = sf.get_tensor(f"{prefix}.attention.wq.weight").float()
        wk = sf.get_tensor(f"{prefix}.attention.wk.weight").float()
        wv = sf.get_tensor(f"{prefix}.attention.wv.weight").float()
        wo = sf.get_tensor(f"{prefix}.attention.wo.weight").float()
        fn_w = sf.get_tensor(f"{prefix}.ffn_norm.weight").float()
        fn_b = sf.get_tensor(f"{prefix}.ffn_norm.bias").float()
        w1 = sf.get_tensor(f"{prefix}.feed_forward.w1.weight").float()
        w2 = sf.get_tensor(f"{prefix}.feed_forward.w2.weight").float()
        w3 = sf.get_tensor(f"{prefix}.feed_forward.w3.weight").float()
        # Attention
        h = layer_norm(x, an_w, an_b)
        attn_out = windowed_attention(h, wq, wk, wv, wo, n_heads=12, window_size=125)
        x = x + attn_out
        # FFN
        h = layer_norm(x, fn_w, fn_b)
        ffn_out = swiglu_ffn(h, w1, w2, w3)
        x = x + ffn_out

    # Output projection (768 → 512)
    proj_w = sf.get_tensor("wave_prenet.output_proj.weight").float()
    proj_b = sf.get_tensor("wave_prenet.output_proj.bias").float()
    x = x @ proj_w.T + proj_b  # (T, 512)

    writer.add_tensor("wave_prenet_out", x.numpy())
    print(f"[miotts-ref] wave_prenet_out shape: {x.shape}")

    # ─── conv_upsample (ConvTranspose1d, 512→512, k=2, s=2) ─────────
    print("[miotts-ref] Running conv_upsample...")
    conv_w = sf.get_tensor("wave_conv_upsample.weight").float()  # (512, 512, 2)
    conv_b = sf.get_tensor("wave_conv_upsample.bias").float()    # (512,)
    # ConvTranspose1d: x is (T, 512) → reshape to (1, 512, T) for conv
    x_conv = x.unsqueeze(0).permute(0, 2, 1)  # (1, 512, T)
    x_conv = torch.nn.functional.conv_transpose1d(x_conv, conv_w, conv_b, stride=2)  # (1, 512, T*2)
    x = x_conv.squeeze(0).permute(1, 0)  # (T*2, 512)
    print(f"[miotts-ref] after conv_upsample: {x.shape}")

    # ─── Interpolate to STFT frame length ────────────────────────────
    # target_audio_length = T_codec * (sample_rate / codec_frame_rate)
    # T_codec = 29, sr=24000, frame_rate=25 → audio_length = 29 * 960 = 27840
    # stft_length = (audio_length + hop_length - 1) // hop_length
    # with hop=480: stft_length = (27840 + 479) // 480 = 59
    T_codec = len(speech_token_ids)
    audio_length = T_codec * (24000 // 25)  # = T_codec * 960
    stft_length = (audio_length + 480 - 1) // 480
    x_interp = x.unsqueeze(0).permute(0, 2, 1)  # (1, 512, T*2)
    x_interp = torch.nn.functional.interpolate(x_interp, size=stft_length, mode='linear', align_corners=False)
    x = x_interp.squeeze(0).permute(1, 0)  # (stft_length, 512)
    print(f"[miotts-ref] after interpolate: {x.shape} (stft_length={stft_length})")

    # ─── wave_prior_net (ResNetStack, 2 blocks) ─────────────────────
    print("[miotts-ref] Running wave_prior_net...")
    def group_norm(x_in, w, b, num_groups=32, eps=1e-6):
        # x_in: (T, C) → (1, C, T) for group_norm
        return torch.nn.functional.group_norm(
            x_in.unsqueeze(0).permute(0, 2, 1), num_groups, w, b, eps
        ).permute(0, 2, 1).squeeze(0)

    def resnet_block(x_in, prefix):
        n1_w = sf.get_tensor(f"{prefix}.norm1.weight").float()
        n1_b = sf.get_tensor(f"{prefix}.norm1.bias").float()
        c1_w = sf.get_tensor(f"{prefix}.conv1.weight").float()
        c1_b = sf.get_tensor(f"{prefix}.conv1.bias").float()
        n2_w = sf.get_tensor(f"{prefix}.norm2.weight").float()
        n2_b = sf.get_tensor(f"{prefix}.norm2.bias").float()
        c2_w = sf.get_tensor(f"{prefix}.conv2.weight").float()
        c2_b = sf.get_tensor(f"{prefix}.conv2.bias").float()
        residual = x_in
        h = group_norm(x_in, n1_w, n1_b)
        h = torch.nn.functional.silu(h)
        # Conv1d: (T, 512) → (1, 512, T) → conv → (1, 512, T) → (T, 512)
        h = h.unsqueeze(0).permute(0, 2, 1)
        h = torch.nn.functional.conv1d(h, c1_w, c1_b, padding=1)
        h = h.permute(0, 2, 1).squeeze(0)
        h = group_norm(h, n2_w, n2_b)
        h = torch.nn.functional.silu(h)
        h = h.unsqueeze(0).permute(0, 2, 1)
        h = torch.nn.functional.conv1d(h, c2_w, c2_b, padding=1)
        h = h.permute(0, 2, 1).squeeze(0)
        return residual + h

    for i in range(2):
        x = resnet_block(x, f"wave_prior_net.blocks.{i}")
    writer.add_tensor("wave_prior_net_out", x.numpy())
    print(f"[miotts-ref] wave_prior_net_out shape: {x.shape}")

    # ─── wave_decoder (8 layers, 512d, AdaLN-Zero) ──────────────────
    print("[miotts-ref] Running wave_decoder (8 layers, AdaLN-Zero)...")
    # Use zero global embedding (no voice cloning reference)
    global_emb = torch.zeros(128)

    def adaln_zero_modulate(x_in, cond, proj_w, proj_b):
        # AdaLN-Zero: SiLU(cond) → Linear → (shift, scale, gate)
        c = torch.nn.functional.silu(cond)
        c = c @ proj_w.T + proj_b  # (3*dim,)
        shift, scale, gate = c.chunk(3, dim=-1)
        # LayerNorm (no learned params) + modulate
        mean = x_in.mean(-1, keepdim=True)
        var = x_in.var(-1, keepdim=True, unbiased=False)
        x_norm = (x_in - mean) / (var + 1e-5).sqrt()
        x_mod = x_norm * (1 + scale) + shift
        return x_mod, gate

    for il in range(8):
        prefix = f"wave_decoder.layers.{il}"
        wq = sf.get_tensor(f"{prefix}.attention.wq.weight").float()
        wk = sf.get_tensor(f"{prefix}.attention.wk.weight").float()
        wv = sf.get_tensor(f"{prefix}.attention.wv.weight").float()
        wo = sf.get_tensor(f"{prefix}.attention.wo.weight").float()
        w1 = sf.get_tensor(f"{prefix}.feed_forward.w1.weight").float()
        w2 = sf.get_tensor(f"{prefix}.feed_forward.w2.weight").float()
        w3 = sf.get_tensor(f"{prefix}.feed_forward.w3.weight").float()
        adaln_attn_w = sf.get_tensor(f"{prefix}.attention_norm.condition_proj.1.weight").float()
        adaln_attn_b = sf.get_tensor(f"{prefix}.attention_norm.condition_proj.1.bias").float()
        adaln_ffn_w = sf.get_tensor(f"{prefix}.ffn_norm.condition_proj.1.weight").float()
        adaln_ffn_b = sf.get_tensor(f"{prefix}.ffn_norm.condition_proj.1.bias").float()

        residual = x
        # AdaLN-Zero attention norm
        x_mod, gate_attn = adaln_zero_modulate(x, global_emb, adaln_attn_w, adaln_attn_b)

        # Windowed attention (8 heads, hd=64, window=65)
        attn_out = windowed_attention(x_mod, wq, wk, wv, wo, n_heads=8, window_size=65)
        x = residual + gate_attn * attn_out

        # AdaLN-Zero FFN norm
        residual = x
        x_mod, gate_ffn = adaln_zero_modulate(x, global_emb, adaln_ffn_w, adaln_ffn_b)
        ffn_out = swiglu_ffn(x_mod, w1, w2, w3)
        x = residual + gate_ffn * ffn_out

    writer.add_tensor("wave_decoder_out", x.numpy())
    print(f"[miotts-ref] wave_decoder_out shape: {x.shape}")

    # ─── wave_post_net (ResNetStack, 2 blocks) ──────────────────────
    print("[miotts-ref] Running wave_post_net...")
    for i in range(2):
        x = resnet_block(x, f"wave_post_net.blocks.{i}")
    writer.add_tensor("wave_post_net_out", x.numpy())
    print(f"[miotts-ref] wave_post_net_out shape: {x.shape}")

    # ─── iSTFT head (Linear 512→1922, split mag+phase, iSTFT) ───────
    print("[miotts-ref] Running iSTFT head...")
    istft_w = sf.get_tensor("istft_head.out.weight").float()  # (1922, 512)
    istft_b = sf.get_tensor("istft_head.out.bias").float()    # (1922,)
    stft_out = x @ istft_w.T + istft_b  # (stft_length, 1922)
    # Split into magnitude and phase (each n_fft//2 + 1 = 961 bins)
    mag_log, phase = stft_out[:, :961], stft_out[:, 961:]
    mag = torch.exp(mag_log).clamp(max=100.0)

    # Reconstruct complex STFT
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    spec = torch.complex(real, imag)  # (stft_length, 961)

    # iSTFT with "same" padding
    n_fft = 1920
    hop_length = 480
    win_length = n_fft
    window = torch.hann_window(win_length)
    pad = (win_length - hop_length) // 2  # = 720

    spec_t = spec.T.unsqueeze(0)  # (1, 961, stft_length)
    # Use torch.fft.irfft for each frame
    ifft = torch.fft.irfft(spec_t, n_fft, dim=1, norm="backward")  # (1, n_fft, stft_length)
    ifft = ifft * window[None, :, None]

    # Overlap-add via fold
    T_stft = spec_t.shape[2]
    output_size = (T_stft - 1) * hop_length + win_length
    y = torch.nn.functional.fold(
        ifft, output_size=(1, output_size),
        kernel_size=(1, win_length), stride=(1, hop_length)
    )[:, 0, 0, pad:-pad]

    # Window envelope normalization
    window_sq = window.square().expand(1, T_stft, -1).transpose(1, 2)
    window_envelope = torch.nn.functional.fold(
        window_sq, output_size=(1, output_size),
        kernel_size=(1, win_length), stride=(1, hop_length)
    ).squeeze()[pad:-pad]
    y = y / window_envelope

    audio = y.squeeze(0)
    writer.add_tensor("audio_output", audio.numpy())
    print(f"[miotts-ref] audio_output shape: {audio.shape} ({audio.shape[0]/24000:.2f}s at 24kHz)")

    print(f"[miotts-ref] Full codec decode complete!")
    print(f"[miotts-ref] Writing to {args.output}...")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("[miotts-ref] Done.")


def main():
    parser = argparse.ArgumentParser(description="MioTTS reference dumper")
    parser.add_argument("--llm-dir", required=True, help="MioTTS-0.6B model directory")
    parser.add_argument("--codec-dir", required=True, help="MioCodec-25Hz-24kHz directory")
    parser.add_argument("--text", default="Hello world", help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    args = parser.parse_args()
    dump(args)


if __name__ == "__main__":
    main()
