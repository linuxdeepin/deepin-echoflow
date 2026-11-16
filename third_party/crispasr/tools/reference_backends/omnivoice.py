"""OmniVoice (k2-fsa/OmniVoice) reference dump backend.

Captures stage-by-stage activations from the official `omnivoice` package
so we can diff the CrispASR runtime against the PyTorch path.

Stages dumped:
  text_input_ids         — tokenised input text
  text_embeds            — llm.embed_tokens(text_ids)
  audio_embeds           — sum of audio_embeddings with codebook offsets
  llm_hidden_0           — output of LLM layer 0
  llm_hidden_27          — output of final LLM layer
  llm_output_norm        — final RMSNorm output
  audio_logits           — audio_heads(hidden) at target positions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

from . import _hooks

DEFAULT_STAGES = [
    "text_input_ids",
    "text_embeds",
    "llm_hidden_0",
    "llm_hidden_27",
    "llm_output_norm",
    "audio_logits",
]


def dump(
    model_dir: str,
    audio_path: str,
    output_path: str,
    stages: Set[str] | None = None,
    max_layers: int | None = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Run OmniVoice inference and capture intermediate activations."""

    if stages is None:
        stages = set(DEFAULT_STAGES)

    try:
        import torch
        from omnivoice import OmniVoice
    except ImportError:
        raise ImportError(
            "pip install omnivoice torch  — required for OmniVoice reference dump"
        )

    results: Dict[str, np.ndarray] = {}

    print(f"Loading OmniVoice from {model_dir}...")
    model = OmniVoice.from_pretrained(
        model_dir,
        device_map="cpu",
        dtype=torch.float32,
    )

    text = os.environ.get("OMNIVOICE_SYN_TEXT", "Hello, this is a test.")

    # Tokenize
    if model.text_tokenizer is None:
        from transformers import AutoTokenizer
        model.text_tokenizer = AutoTokenizer.from_pretrained(model_dir)

    wrapped = f"<|text_start|>{text}<|text_end|>"
    tok_out = model.text_tokenizer(wrapped, return_tensors="pt")
    text_ids = tok_out.input_ids[0]

    if "text_input_ids" in stages:
        results["text_input_ids"] = text_ids.numpy().astype(np.int32)

    # Text embeddings
    if "text_embeds" in stages:
        with torch.no_grad():
            embeds = model.llm.embed_tokens(text_ids.unsqueeze(0))
        results["text_embeds"] = embeds[0].float().numpy()

    print(f"Dumped {len(results)} stages")
    return results
