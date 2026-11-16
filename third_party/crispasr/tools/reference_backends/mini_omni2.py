"""Mini-Omni2 reference dump backend (gpt-omni/mini-omni2).

Loads the Mini-Omni2 model (Whisper-small encoder + SwiGLU adapter +
Qwen2-0.5B LLM) and captures intermediates at each architectural
boundary for crispasr-diff comparison.

Architecture:
  Whisper-small encoder (80 mel, 12L, 768d) →
  whisperMLP adapter (768→4864→896 SwiGLU) →
  Qwen2-0.5B LLM (896d, 24L, 14 heads, 2 KV groups, RMSNorm, SwiGLU) →
  text logits (152000) + 7×audio logits (4160 each)

For ASR we only exercise the text path: audio → mel → encoder →
adapter → LLM → text tokens.

The model uses a custom litgpt framework (NOT standard HF), so we
clone the repo and load weights from lit_model.pth + small.pt.

Stages:

  raw_audio               (N,)              input PCM @ 16 kHz
  mel_spectrogram         (80, T_mel)       Whisper log-mel features
  whisper_encoder_output  (T_enc, 768)      encoder last hidden state
  adapter_output          (T_enc, 896)      after whisperMLP projection
  input_embeds            (T_total, 896)    full LLM input (audio + text tokens)
  llm_logits              (T_gen, V_text)   text logits from greedy decode
  generated_text          (string)          decoded transcript

Memory note: Whisper-small (~484 MB) and the full GPT model (~2.8 GB)
are loaded sequentially, not simultaneously. Whisper is freed before
loading the GPT model to keep peak RSS under ~4 GB.

Env vars:
  MINI_OMNI2_REPO   path to cloned gpt-omni/mini-omni2 repo (optional;
                    if absent, tries to import litgpt from model_dir)
  MINI_OMNI2_MAX_TOKENS  max new tokens for ASR decode (default: 200)

Usage:

  python tools/dump_reference.py --backend mini-omni2 \
      --model-dir /path/to/mini-omni2-hf \
      --audio samples/jfk.wav \
      --output /tmp/mini-omni2-ref.gguf
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "whisper_encoder_output",
    "adapter_output",
]


# ---------------------------------------------------------------------------
# Token constants (must match inference.py)
# ---------------------------------------------------------------------------

_TEXT_VOCABSIZE = 151936
_TEXT_SPECIAL = 64
_AUDIO_VOCABSIZE = 4096
_AUDIO_SPECIAL = 64
_PADDED_TEXT = _TEXT_VOCABSIZE + _TEXT_SPECIAL   # 152000
_PADDED_AUDIO = _AUDIO_VOCABSIZE + _AUDIO_SPECIAL  # 4160

_EOT = _TEXT_VOCABSIZE        # 151936
_PAD_T = _TEXT_VOCABSIZE + 1
_INPUT_T = _TEXT_VOCABSIZE + 2
_ANSWER_T = _TEXT_VOCABSIZE + 3

_EOA = _AUDIO_VOCABSIZE       # 4096
_PAD_A = _AUDIO_VOCABSIZE + 1
_INPUT_A = _AUDIO_VOCABSIZE + 2
_ANSWER_A = _AUDIO_VOCABSIZE + 3


def _layershift(input_id: int, layer: int,
                stride: int = _PADDED_AUDIO,
                shift: int = _PADDED_TEXT) -> int:
    return input_id + shift + layer * stride


# ---------------------------------------------------------------------------
# Public dump() entry point
# ---------------------------------------------------------------------------

def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Mini-Omni2 reference forward and return stage captures."""
    import torch
    import whisper

    model_dir = Path(model_dir)
    max_tok = int(os.environ.get("MINI_OMNI2_MAX_TOKENS", str(max_new_tokens or 200)))

    # --- Add litgpt to path ---
    repo_dir = os.environ.get("MINI_OMNI2_REPO")
    if repo_dir:
        sys.path.insert(0, repo_dir)
    else:
        for candidate in [model_dir, model_dir.parent]:
            litgpt_init = candidate / "litgpt" / "__init__.py"
            if litgpt_init.exists():
                sys.path.insert(0, str(candidate))
                break

    out: Dict[str, np.ndarray | str] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # Determine which phases we need
    need_whisper = bool(stages & {"mel_spectrogram", "whisper_encoder_output",
                                   "adapter_output", "input_embeds",
                                   "llm_logits", "generated_text"})
    need_gpt = bool(stages & {"adapter_output", "input_embeds",
                               "llm_logits", "generated_text"})

    enc_out_np = None  # saved across whisper→gpt boundary

    # =====================================================================
    # Phase 1: Whisper-small (mel + encoder)
    # =====================================================================
    if need_whisper:
        whisper_path = model_dir / "small.pt"
        if whisper_path.exists():
            print(f"  loading Whisper-small from {whisper_path}")
            whisper_model = whisper.load_model(str(whisper_path), device="cpu")
        else:
            print("  loading Whisper-small from OpenAI (downloading)")
            whisper_model = whisper.load_model("small", device="cpu")
        whisper_model.eval()

        audio_t = torch.from_numpy(audio.astype(np.float32))
        audio_padded = whisper.pad_or_trim(audio_t)
        mel = whisper.log_mel_spectrogram(audio_padded)  # (80, 3000)

        if "mel_spectrogram" in stages:
            out["mel_spectrogram"] = mel.numpy().astype(np.float32)

        # Actual audio length in encoder frames (20ms per frame)
        duration_ms = (len(audio) / 16000) * 1000
        audio_len = int(duration_ms / 20) + 1

        with torch.no_grad():
            mel_input = mel.unsqueeze(0)  # (1, 80, 3000)
            encoder_output = whisper_model.encoder(mel_input)  # (1, 1500, 768)

        enc_out = encoder_output[0, :audio_len, :].contiguous()  # (T_enc, 768)

        if "whisper_encoder_output" in stages:
            out["whisper_encoder_output"] = enc_out.numpy().astype(np.float32)

        # Save encoder output as numpy for the GPT phase
        enc_out_np = enc_out.numpy().astype(np.float32)

        # Free Whisper to reclaim ~484 MB before loading GPT
        del whisper_model, encoder_output, enc_out, mel_input, mel, audio_padded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  freed Whisper-small")

    # =====================================================================
    # Phase 2: LitGPT (adapter + LLM)
    # =====================================================================
    if need_gpt:
        assert enc_out_np is not None
        print(f"  loading Mini-Omni2 LitGPT from {model_dir}")
        from litgpt.config import Config
        from litgpt.model import GPT

        config = Config.from_file(str(model_dir / "model_config.yaml"))
        config.post_adapter = False

        model = GPT(config)
        state_dict = torch.load(str(model_dir / "lit_model.pth"),
                                map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        model.eval()

        enc_out_t = torch.from_numpy(enc_out_np)  # (T_enc, 768)
        T_enc = enc_out_t.shape[0]

        # --- Run whisperMLP adapter ---
        if "adapter_output" in stages or "input_embeds" in stages or "llm_logits" in stages or "generated_text" in stages:
            adapter = model.whisper_adapter
            with torch.no_grad():
                enc_3d = enc_out_t.unsqueeze(0).float()  # (1, T_enc, 768)
                adapted = adapter(enc_3d)  # (1, T_enc, 896)

            adapted_2d = adapted[0].contiguous()  # (T_enc, 896)

            if "adapter_output" in stages:
                out["adapter_output"] = adapted_2d.numpy().astype(np.float32)

        # --- Build 8-stream input_ids (ASR task) ---
        input_ids = []
        for i in range(7):
            ids = [_layershift(_INPUT_A, i)]
            ids += [_layershift(_PAD_A, i)] * T_enc
            ids += [_layershift(_EOA, i), _layershift(_ANSWER_A, i)]
            input_ids.append(torch.tensor(ids, dtype=torch.int32).unsqueeze(0))
        text_ids = [_INPUT_T] + [_PAD_T] * T_enc + [_EOT, _ANSWER_T]
        input_ids.append(torch.tensor(text_ids, dtype=torch.int32).unsqueeze(0))

        # --- Capture input_embeds ---
        if "input_embeds" in stages:
            with torch.no_grad():
                x_a = adapted.clone()  # (1, T_enc, 896)
                embeds = []
                for ids in input_ids:
                    embeds.append(model.transformer.wte(ids))
                # Audio features replace pad positions in audio streams
                # (0..6) only — stream 7 (text) keeps its token embeddings.
                for i in range(7):
                    embeds[i][:, 1:T_enc+1, :] = x_a
                x = sum(embeds) / 8.0
                out["input_embeds"] = x[0].numpy().astype(np.float32)

        # --- ASR generation (greedy) ---
        if "llm_logits" in stages or "generated_text" in stages:
            print(f"  running ASR generation (max {max_tok} tokens) ...")
            model.set_kv_cache(batch_size=1)
            audio_feat = enc_out_t.unsqueeze(0).float()
            T = input_ids[0].size(1)
            input_pos = torch.arange(0, T)
            all_logits = []
            generated_tokens = []

            with torch.no_grad():
                logits_a, logit_t = model(
                    audio_feat, input_ids, None, input_pos,
                    whisper_lens=[T - 3], task=["asr"],
                )
                last_logits = logit_t[0, -1, :_PADDED_TEXT]
                token = last_logits.argmax().item()
                generated_tokens.append(token)
                all_logits.append(last_logits.numpy().astype(np.float32))

                cur_pos = torch.tensor([T])
                for step in range(max_tok - 1):
                    step_ids = []
                    for i in range(7):
                        step_ids.append(
                            torch.tensor([_layershift(_EOA, i)],
                                         dtype=torch.int32).unsqueeze(0))
                    step_ids.append(
                        torch.tensor([token], dtype=torch.int32).unsqueeze(0))
                    logits_a, logit_t = model(
                        None, step_ids, None, cur_pos,
                        whisper_lens=None, task=None,
                    )
                    last_logits = logit_t[0, -1, :_PADDED_TEXT]
                    token = last_logits.argmax().item()
                    if token == _EOT:
                        break
                    generated_tokens.append(token)
                    all_logits.append(last_logits.numpy().astype(np.float32))
                    cur_pos = cur_pos.add_(1)

            model.clear_kv_cache()

            if "llm_logits" in stages and all_logits:
                out["llm_logits"] = np.stack(all_logits, axis=0)

            if "generated_text" in stages:
                from litgpt.tokenizer import Tokenizer
                tokenizer = Tokenizer(model_dir)
                text = tokenizer.decode(torch.tensor(generated_tokens))
                out["generated_text"] = text.strip()
                print(f"  ASR result: {out['generated_text']!r}")

        del model
        gc.collect()

    return out
