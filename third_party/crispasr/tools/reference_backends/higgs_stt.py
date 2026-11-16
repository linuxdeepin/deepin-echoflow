"""bosonai/higgs-audio-v3-stt reference dump backend.

Captures the architectural boundary the C++ runtime (src/higgs_stt.cpp)
reproduces, so crispasr-diff can localize a divergence:

  raw_audio       the 16 kHz mono PCM fed in
  audio_embeds    chunked Whisper tower + projector output, concatenated over
                  the model's chunk_size_seconds (4 s) chunks  (N_total, 2048)
  generated_text  greedy transcript (optional)

`audio_embeds` is THE stage to gate: higgs-audio splits the waveform into 4 s
chunks and encodes each independently (chunk-local positions, within-chunk
attention) before concatenating — encoding one padded 30 s window corrupts the
result and derails the decoder. This dumps exactly the model's own
`_apply_audio_tower_whisper` path (the same one used during generation, which
produces the verbatim transcript), with the valid token count per chunk taken
from `audio_feat_out_lengths`. Matches `higgs_stt_encode_audio` in the runtime.

The model is loaded in bf16 (≈5.4 GB) to stay within a 16 GB box.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "audio_embeds",
    "generated_text",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    # transformers 4.51 pulls in timm_wrapper which raises "unsupported platform"
    # on macOS via an eventlet import; stub it out before importing transformers.
    sys.modules.setdefault("eventlet", None)  # type: ignore
    import torch
    from dataclasses import asdict
    from transformers import AutoModel, AutoTokenizer

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = np.asarray(audio, dtype=np.float32)

    need_model = bool({"audio_embeds", "generated_text"} & stages)
    if not need_model:
        return out

    print(f"  loading HiggsAudio3 (bf16) from {model_dir}")
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Reuse the repo's transcribe.py helpers to build the chunked audio batch
    # exactly as inference does (collator splits into 4 s chunks).
    import importlib.util
    tp = Path(model_dir) / "transcribe.py"
    spec = importlib.util.spec_from_file_location("higgs_transcribe", str(tp))
    T = importlib.util.module_from_spec(spec)
    sys.modules["higgs_transcribe"] = T
    spec.loader.exec_module(T)  # type: ignore

    audio = np.asarray(audio, dtype=np.float32)

    if "audio_embeds" in stages:
        collator = T._create_collator(model.config)
        ids = T._build_input_tokens(tok, T.DEFAULT_PROMPT, enable_thinking=True)
        sample = T._build_sample(audio, ids, sample_rate=16000)
        batch = asdict(collator([sample]))
        af = batch["audio_features"].to(torch.bfloat16)
        am = batch["audio_feature_attention_mask"]
        with torch.no_grad():
            embed, out_lengths = model._apply_audio_tower_whisper(af, am)
        # embed: (num_chunks, max_tokens, 2048); take the valid prefix per chunk
        # and concatenate (what merge_input_ids_with_audio_features splices).
        parts = []
        for i in range(embed.shape[0]):
            n = int(out_lengths[i].item())
            parts.append(embed[i, :n].detach().cpu().float().numpy())
        out["audio_embeds"] = np.concatenate(parts, axis=0)  # (N_total, 2048)

    if "generated_text" in stages:
        try:
            text = T.transcribe(model, tok, audio, sample_rate=16000)
            out["generated_text"] = text or ""
        except Exception as e:  # noqa: BLE001
            print(f"  generated_text skipped: {e}")
            out["generated_text"] = ""

    return out
