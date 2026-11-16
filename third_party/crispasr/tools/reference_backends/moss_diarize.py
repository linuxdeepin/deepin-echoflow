"""MOSS-Transcribe-Diarize-0.9B reference dump backend.

Captures the architectural boundaries the C++ runtime reproduces:

  raw_audio         16 kHz mono PCM fed in
  mel_spectrogram   80-bin log-mel from WhisperFeatureExtractor (per 30s chunk)
  encoder_output    Whisper encoder output (concatenated valid frames from all chunks)
  audio_embeds      after 4x merge + VQAdaptor (the audio→LLM boundary)
  generated_text    greedy transcript from model.generate()

`audio_embeds` is the gate stage: if this matches, the audio pipeline is correct
and any decode divergence is in the prompt template or LLM.

The model is 1.82 GB BF16 (~3.6 GB in f32). On 8 GB RAM, load with
low_cpu_mem_usage=True + torch.float32 (no bf16→f32 expansion spike).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "conv_stem_out",
    "encoder_layer_0",
    "encoder_output",
    "audio_embeds",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch

    # Stub torch_compilable_check if missing (model requires bleeding-edge transformers)
    import transformers.utils as _tu
    if not hasattr(_tu, "torch_compilable_check"):
        def _tcc(cond, msg=""):
            assert cond, msg
        _tu.torch_compilable_check = _tcc

    from transformers import AutoModelForCausalLM, AutoProcessor

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = np.asarray(audio, dtype=np.float32)

    need_model = bool(stages - {"raw_audio"})
    if not need_model:
        return out

    print(f"  loading MOSS-Transcribe-Diarize from {model_dir}")

    # Workaround: transformers 5.x expects extra_special_tokens as a dict,
    # but this model ships it as a list. Patch before loading.
    import json, shutil
    tc_path = Path(model_dir) / "tokenizer_config.json"
    tc_bak = Path(model_dir) / "tokenizer_config.json.bak"
    patched = False
    if tc_path.exists():
        with open(tc_path) as f:
            tc = json.load(f)
        if isinstance(tc.get("extra_special_tokens"), list):
            shutil.copy2(tc_path, tc_bak)
            tc["extra_special_tokens"] = {t: t for t in tc["extra_special_tokens"]}
            with open(tc_path, "w") as f:
                json.dump(tc, f, ensure_ascii=False, indent=2)
            patched = True

    try:
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True,
            dtype=torch.float32, low_cpu_mem_usage=True,
        ).eval()
    finally:
        if patched and tc_bak.exists():
            shutil.move(str(tc_bak), str(tc_path))

    try:
        from . import _reload_guard
    except ImportError:
        import _reload_guard
    _reload_guard.reload_if_random_init(model, model_dir)

    audio = np.asarray(audio, dtype=np.float32)

    # ---- Build inputs manually (bypass chat template which may not exist) ----
    # The exact format is:
    #   <|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\nPROMPT<|im_end|>\n<|im_start|>assistant\n
    # The processor's __call__ expands the single <|audio_pad|> to N pads + time markers.
    DEFAULT_PROMPT = (
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
        "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        "并在段末标注结束时间戳，以清晰标明该段语音范围。"
    )
    text = (
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>\n"
        f"{DEFAULT_PROMPT}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(text=text, audio=[audio], return_tensors="pt")

    input_features = inputs["input_features"]  # (num_chunks, 80, 3000)
    audio_feature_lengths = inputs["audio_feature_lengths"]
    audio_chunk_mapping = inputs.get("audio_chunk_mapping")
    input_ids = inputs["input_ids"]

    print(f"  input_features: {input_features.shape}, audio_feature_lengths: {audio_feature_lengths.tolist()}")
    print(f"  input_ids: {input_ids.shape}")

    # ---- mel_spectrogram: first chunk's mel features ----
    if "mel_spectrogram" in stages:
        mel = input_features[0]  # (80, 3000) — first 30s chunk
        out["mel_spectrogram"] = mel.detach().cpu().float().numpy()

    # ---- encoder stages with hooks for per-layer capture ----
    need_enc = bool(stages & {"encoder_output", "audio_embeds", "conv_stem_out",
                              "encoder_layer_0"})
    if need_enc:
        enc = model.model.whisper_encoder
        captures = {}

        handles = []
        # Conv stem output: capture after conv2 (before gelu, permute, pos embed).
        # We apply gelu + permute manually to match the C++ conv_stem_out which is
        # after conv1+gelu+conv2+gelu+reshape+transpose (= after both gelus, in (T,D) layout).
        if "conv_stem_out" in stages:
            def _cap_conv2(_mod, _inp, output):
                # output is conv2 raw output: (B, D, T) before gelu
                captures["conv_stem_out_raw"] = output.detach().clone()
            handles.append(enc.conv2.register_forward_hook(_cap_conv2))

        # Per-layer hooks
        if "encoder_layer_0" in stages and len(enc.layers) > 0:
            def _cap_l0(_mod, _inp, output):
                t = output[0] if isinstance(output, tuple) else output
                captures["encoder_layer_0"] = t.detach().clone()
            handles.append(enc.layers[0].register_forward_hook(_cap_l0))

        with torch.no_grad():
            whisper_out = enc(input_features, return_dict=True).last_hidden_state

        for h in handles:
            h.remove()

        # conv_stem_out: apply gelu (hook captured pre-gelu), then permute to (T, D)
        if "conv_stem_out" in stages and "conv_stem_out_raw" in captures:
            import torch.nn.functional as F
            cs = F.gelu(captures["conv_stem_out_raw"])[0]  # (D, T) after gelu
            valid_T = whisper_out.shape[1]  # 1500
            cs_np = cs[:, :valid_T].permute(1, 0).contiguous().cpu().float().numpy()
            out["conv_stem_out"] = cs_np

        # encoder_layer_0: (B, T, D) — take batch 0 valid frames
        if "encoder_layer_0" in stages and "encoder_layer_0" in captures:
            l0 = captures["encoder_layer_0"][0]  # (T, D)
            out["encoder_layer_0"] = l0.cpu().float().numpy()

        # Concatenate valid frames per chunk
        parts = []
        for i in range(whisper_out.shape[0]):
            valid_len = int(audio_feature_lengths[i].item()) * 4
            parts.append(whisper_out[i, :valid_len].detach().cpu().float().numpy())
        enc_concat = np.concatenate(parts, axis=0)  # (T_total, 1024)

        if "encoder_output" in stages:
            out["encoder_output"] = enc_concat

        # ---- audio_embeds: 4x merge + VQAdaptor ----
        if "audio_embeds" in stages:
            with torch.no_grad():
                audio_features = model.model.get_audio_features(
                    input_features=input_features,
                    audio_feature_lengths=audio_feature_lengths,
                    audio_chunk_mapping=audio_chunk_mapping,
                )
            adapted = torch.cat([f.squeeze(0) for f in audio_features], dim=0)
            out["audio_embeds"] = adapted.detach().cpu().float().numpy()

    # ---- generated_text: greedy decode ----
    if "generated_text" in stages:
        try:
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            prompt_len = input_ids.shape[1]
            text = processor.tokenizer.decode(
                gen_ids[0, prompt_len:], skip_special_tokens=True
            )
            out["generated_text"] = text
            print(f"  generated_text: {text[:200]!r}")
        except Exception as e:
            print(f"  generated_text skipped: {e}")
            out["generated_text"] = ""

    return out
