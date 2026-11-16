"""TADA TTS reference dump backend for crispasr-diff.

Uses the official model.generate() API with a reference audio prompt
for voice conditioning. This produces audible, intelligible speech
that can be validated via ASR roundtrip.

Pipeline:
  1. Encoder: reference audio → aligned acoustic features (voice prompt)
  2. model.generate(prompt, text): AR + FM → acoustic features + time durations
  3. Decoder: expand + local-attention + DAC upsampler → 24 kHz PCM

Environment variables:
  TADA_SYN_TEXT       — text to synthesize (default: "Please call Stella.")
  TADA_PROMPT_TEXT    — transcript of the reference audio (optional, for alignment)
  TADA_DEVICE         — "cpu" or "cuda" (default: "cpu")
  TADA_SEED           — random seed (default: 42)
  TADA_CODEC_DIR      — local path to HumeAI/tada-codec (optional)
  TADA_WAV_OUTPUT     — path for output WAV (default: /tmp/tada-ref-output.wav)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_tokens",
    "prompt_token_values",
    "prompt_token_positions",
    "prompt_time_before",
    "prompt_time_after",
    "acoustic_features",
    "time_before",
    "fm_step_index",
    "fm_speech_in",
    "fm_cond",
    "fm_neg_cond",
    "fm_speech_out",
    "fm_time_bits",
    # codec stages — codec_input + codec_token_masks are required by the diff
    # harness to drive codec extraction; codec_proj/codec_attn_out are optional
    # intermediates inside the codec decoder
    "codec_input",
    "codec_token_masks",
    "codec_proj",
    "codec_attn_out",
    "codec_pcm",
]

DEFAULT_SYN_TEXT = "Please call Stella."


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int = 256) -> Dict[str, np.ndarray]:
    import torch
    import torchaudio
    torch.set_grad_enabled(False)

    out: Dict[str, np.ndarray] = {}

    device = os.environ.get("TADA_DEVICE", "cpu")
    syn_text = os.environ.get("TADA_SYN_TEXT", DEFAULT_SYN_TEXT)
    seed = int(os.environ.get("TADA_SEED", "42"))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Load model ──
    print(f"  loading TADA from {model_dir}")
    from transformers import AutoTokenizer
    from tada.modules.tada import TadaForCausalLM, InferenceOptions
    from tada.modules.encoder import Encoder
    from tada.modules.decoder import Decoder

    want_fm_debug = bool(stages & {
        "fm_step_index",
        "fm_speech_in",
        "fm_cond",
        "fm_neg_cond",
        "fm_speech_out",
        "fm_time_bits",
    })
    fm_debug = []
    orig_solve_flow_matching = None
    if want_fm_debug:
        orig_solve_flow_matching = TadaForCausalLM._solve_flow_matching

        def _capture(x: torch.Tensor) -> np.ndarray:
            y = x.detach().cpu().float()
            if y.ndim > 1 and y.shape[0] == 1:
                y = y.squeeze(0)
            if y.ndim > 1 and y.shape[0] == 1:
                y = y.squeeze(0)
            return y.numpy()

        def _patched_solve(self, *args, **kwargs):
            speech = kwargs.get("speech", args[0] if len(args) > 0 else None)
            cond = kwargs.get("cond", args[1] if len(args) > 1 else None)
            neg_cond = kwargs.get("neg_cond", args[2] if len(args) > 2 else None)
            rec = {
                "speech_in": _capture(speech),
                "cond": _capture(cond),
                "neg_cond": _capture(neg_cond) if neg_cond is not None else np.zeros_like(_capture(cond)),
            }
            result = orig_solve_flow_matching(self, *args, **kwargs)
            rec["speech_out"] = _capture(result)
            rec["time_bits"] = rec["speech_out"][-int(self.time_dim):]
            fm_debug.append(rec)
            return result

        TadaForCausalLM._solve_flow_matching = _patched_solve

    # Patch tokenizer for gated Llama
    _orig = AutoTokenizer.from_pretrained.__func__
    @classmethod
    def _patched(cls, name, *a, **kw):
        try:
            return _orig(cls, name, *a, **kw)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e):
                return _orig(cls, "unsloth/Llama-3.2-1B", *a, **kw)
            raise
    AutoTokenizer.from_pretrained = _patched

    codec_dir = os.environ.get("TADA_CODEC_DIR")
    if codec_dir:
        _orig_dec = Decoder.from_pretrained.__func__
        @classmethod
        def _patched_dec(cls, name, *a, **kw):
            if "tada-codec" in str(name):
                return _orig_dec(cls, codec_dir, *a, **kw)
            return _orig_dec(cls, name, *a, **kw)
        Decoder.from_pretrained = _patched_dec

    model = TadaForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    tokenizer = model.tokenizer

    # ── Load Encoder for voice conditioning ──
    print(f"  loading Encoder")
    codec_path = codec_dir or "HumeAI/tada-codec"
    encoder = Encoder.from_pretrained(codec_path, subfolder="encoder").to(device)
    encoder.eval()

    # ── Process reference audio ──
    # audio is 16 kHz mono float32 from dump_reference.py
    audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    # Resample 16k → 24k for the encoder
    audio_24k = torchaudio.functional.resample(audio_t, 16000, 24000)

    prompt_text = os.environ.get("TADA_PROMPT_TEXT")
    if prompt_text:
        prompt_texts = [prompt_text]
    else:
        prompt_texts = None  # auto-transcribe via parakeet

    print(f"  encoding reference audio ({audio_24k.shape[-1]/24000:.1f}s @ 24kHz)")
    with torch.no_grad():
        prompt = encoder(
            audio_24k,
            text=prompt_texts,
            sample_rate=24000,
        )

    print(f"  prompt: {prompt.token_values.shape[1]} aligned tokens")

    # ── Dump prompt features for C++ to use as pre-computed input ──
    if "prompt_token_values" in stages:
        out["prompt_token_values"] = prompt.token_values[0].cpu().float().numpy()
    if "prompt_token_positions" in stages:
        out["prompt_token_positions"] = prompt.token_positions[0].cpu().float().numpy()

    # Store the (auto-transcribed) reference-audio transcript so the C++ diff
    # uses the SAME prompt-region tokens Python feeds. Without it, C++ pads the
    # whole voice-replay region with <pad>, but generate(use_text_in_prompt=False)
    # only masks the first prompt_acoustic_features.shape[1] positions — the tail
    # of the transcript stays unmasked and conditions the first AR/FM steps.
    # Missing those tokens makes fm_cond #0/#1 ≈ 0 and collapses the timing.
    if prompt.text and prompt.text[0]:
        out["tada_tts_prompt_text"] = prompt.text[0]

    # ── Tokenize synth text ──
    from tada.utils.text import normalize_text
    norm_syn_text = normalize_text(syn_text)
    text_tokens_raw = tokenizer.encode(norm_syn_text, add_special_tokens=False)
    if "text_tokens" in stages:
        out["text_tokens"] = np.array(text_tokens_raw, dtype=np.float32)

    # ── Patch _decode_wav to intercept expanded codec input ──
    # Also patch Decoder.forward to capture proj and attn intermediates.
    # We store results here so they're accessible after generate() returns.
    _captured: Dict[str, np.ndarray] = {}

    need_codec_input = any(s in stages for s in
                           ("codec_input", "codec_token_masks",
                            "codec_proj", "codec_attn_out"))

    if need_codec_input:
        from tada.modules.decoder import _create_segment_attention_mask

        orig_decode_wav = TadaForCausalLM._decode_wav

        def patched_decode_wav(self, encoded, time_before):
            tb = time_before[: encoded.shape[0] + 1]
            expanded = []
            for pos in range(encoded.shape[0]):
                expanded.append(
                    torch.zeros(
                        (tb[pos] - 1).clamp(min=0),
                        encoded.shape[-1],
                        device=self.device,
                        dtype=encoded.dtype,
                    )
                )
                expanded.append(encoded[pos].unsqueeze(0))
            expanded.append(
                torch.zeros(tb[-1], encoded.shape[-1], device=self.device, dtype=encoded.dtype)
            )
            encoded_expanded = torch.cat(expanded, dim=0).unsqueeze(0)  # (1, T, 512)
            masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()   # (1, T)
            _captured["codec_input"] = encoded_expanded[0].detach().cpu().float().numpy()       # (T, 512)
            _captured["codec_token_masks"] = masks[0].detach().cpu().float().numpy()           # (T,)
            _captured["decode_time_before"] = tb.detach().cpu().float().numpy()
            return orig_decode_wav(self, encoded, time_before)

        TadaForCausalLM._decode_wav = patched_decode_wav

        # Patch Decoder.forward to capture proj and attn outputs
        orig_decoder_forward = Decoder.forward

        def patched_decoder_forward(self, encoded_expanded, token_masks):
            # proj: (1, T, 512) → (1, T, 1024)
            decoder_proj_out = self.decoder_proj(encoded_expanded)
            _captured["codec_proj"] = decoder_proj_out[0].detach().cpu().float().numpy()  # (T, 1024)

            # Build attn mask same as original
            attn_mask = _create_segment_attention_mask(token_masks, version="v2")
            attn_out = self.local_attention_decoder(decoder_proj_out, mask=attn_mask)
            _captured["codec_attn_out"] = attn_out[0].detach().cpu().float().numpy()      # (T, 1024)

            # wav_decoder expects (B, C, T) — transpose before passing
            x_rec = self.wav_decoder(attn_out.transpose(1, 2))
            return x_rec

        Decoder.forward = patched_decoder_forward

    # ── Generate with official API ──
    print(f"  generating: {syn_text!r}")
    opts = InferenceOptions(
        text_do_sample=False,
        acoustic_cfg_scale=1.6,
        duration_cfg_scale=1.0,
        noise_temperature=0.9,
        num_flow_matching_steps=10,
        num_acoustic_candidates=1,
        cfg_schedule="cosine",
        time_schedule="logsnr",
    )

    # Reseed RIGHT before generate() so Python's flow-matching noise starts from
    # a fresh seed-`seed` MT19937 stream. The C++ engine seeds fresh and never
    # runs the encoder, but Python's encoder (parakeet auto-transcribe + the
    # codec encoder forward) consumes RNG above, which would otherwise offset
    # Python's FM noise relative to C++ and make every FM stage (speech_in,
    # speech_out, time_bits, time_before) look uncorrelated in the diff even
    # though the math is identical. Reseeding makes the comparison fair.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        with torch.no_grad():
            gen_output = model.generate(
                prompt=prompt,
                text=syn_text,
                inference_options=opts,
                verbose=True,
            )
    finally:
        if orig_solve_flow_matching is not None:
            TadaForCausalLM._solve_flow_matching = orig_solve_flow_matching

    # Restore patches
    if need_codec_input:
        TadaForCausalLM._decode_wav = orig_decode_wav
        Decoder.forward = orig_decoder_forward

    # ── Extract codec intermediates from capture dict ──
    for key in ("codec_input", "codec_token_masks", "codec_proj", "codec_attn_out",
                "decode_time_before"):
        if key in stages and key in _captured:
            out[key] = _captured[key]
        elif key in _captured:
            out[key] = _captured[key]  # always include for harness alignment

    if _captured.get("codec_input") is not None:
        n_frames = _captured["codec_input"].shape[0]
        n_voiced = int(_captured["codec_token_masks"].sum()) if "codec_token_masks" in _captured else "?"
        print(f"  codec: {n_frames} expanded frames, {n_voiced} voiced")

    # ── Extract generation outputs ──
    if gen_output.input_text_ids is not None:
        print(f"  input_ids: {gen_output.input_text_ids.shape}")
        if "input_ids" in stages:
            out["input_ids"] = gen_output.input_text_ids[0].cpu().float().numpy()

    if gen_output.acoustic_features is not None and "acoustic_features" in stages:
        out["acoustic_features"] = gen_output.acoustic_features[0].cpu().float().numpy()

    if gen_output.time_before is not None and "time_before" in stages:
        out["time_before"] = gen_output.time_before[0].cpu().float().numpy()

    if fm_debug:
        n = len(fm_debug)
        if "fm_step_index" in stages:
            out["fm_step_index"] = np.stack(
                [np.array([float(i), float(i)], dtype=np.float32) for i in range(n)],
                axis=0,
            )
        for name in ("speech_in", "cond", "neg_cond", "speech_out", "time_bits"):
            stage_name = f"fm_{name}"
            if stage_name in stages:
                out[stage_name] = np.stack([rec[name].astype(np.float32) for rec in fm_debug], axis=0)
        print(f"  captured FM calls: {n}")

    if gen_output.audio is not None and gen_output.audio[0] is not None:
        pcm = gen_output.audio[0].cpu().float().numpy()
        if "codec_pcm" in stages:
            out["codec_pcm"] = pcm
        wav_path = Path(os.environ.get("TADA_WAV_OUTPUT", "/tmp/tada-ref-output.wav"))
        import wave
        pcm16 = (pcm * 32767).clip(-32767, 32767).astype(np.int16)
        with wave.open(str(wav_path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm16.tobytes())
        print(f"  WAV: {wav_path} ({len(pcm16)} samples, "
              f"RMS={np.sqrt(np.mean(pcm.astype(float)**2)):.4f})")

    # Store synthesis text in metadata so diff harness can replay the same text
    out["tada_tts_syn_text"] = syn_text  # routed to crispasr.ref.tada_tts_syn_text

    generated_text = ""
    if gen_output.output_str:
        generated_text = gen_output.output_str[0]
    print(f"  generated text: {generated_text!r}")

    return out
