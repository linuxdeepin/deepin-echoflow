#!/usr/bin/env python3
"""MOSS-TTS-v1.5 (MossTTSDelay) reference dumper for the crispasr diff harness.

Produces the GREEDY audio code grid the C++ runtime must reproduce byte-for-byte
(Phase 3 code-parity gate), and optionally the codec-decoded reference waveform
(Phase 4). The model's own `generate(do_sample=False)` IS the greedy reference —
we drive it and extract the (n_vq, T_audio) code matrix.

Run inside a GPU box with the 8B model (Kaggle) — see
tools/kaggle/moss-tts-validate/. The high-level HF API surface (processor method
names, generate return shape) is confirmed on that first run; the extraction
below is defensive and logs the actual structure so any mismatch is a one-line
fix, not a silent wrong reference.

Env:
  MOSS_TTS_MODEL   HF id or local dir of MOSS-TTS-v1.5 (default the HF id)
  MOSS_TTS_TEXT    text to synthesize (fixed for parity)
  MOSS_TTS_SEED    RNG seed (default 0; greedy is seed-independent but set it)
  MOSS_TTS_MAXNEW  max_new_tokens (default 512)
  MOSS_TTS_CODEC   HF id or dir of MOSS-Audio-Tokenizer (optional; decode ref)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "codes",       # (n_vq, T_audio) int32 — the greedy code grid (parity target)
    "waveform",    # (T_audio*1920,) f32 — codec-decoded reference (if codec given)
]

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"
DEFAULT_CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer"


def _extract_codes(model, out: Any) -> np.ndarray:
    """Pull the (n_vq, T_audio) int code grid out of a MossTTSDelay generate
    result. Tries the documented attributes, then falls back to scanning for the
    first 2D integer tensor whose minor axis == n_vq. Raises with a dump of the
    structure if nothing matches (so the Kaggle run tells us the real shape)."""
    import torch

    cfg = model.config
    n_vq = int(getattr(cfg, "n_vq", 32))
    audio_start = int(getattr(cfg, "audio_start_token_id", 151652))
    audio_end = int(getattr(cfg, "audio_end_token_id", 151653))

    # generate() returns a list of (start_len, tokens) tuples; tokens is the raw
    # DELAYED (seq_len, 1+n_vq) grid [text, audio_0..audio_{n_vq-1}].
    tokens = out[0][1] if (isinstance(out, (list, tuple)) and out and isinstance(out[0], (list, tuple))) else out
    tok = tokens.detach().to(torch.int64).cpu().numpy() if hasattr(tokens, "detach") else np.asarray(tokens)
    tok = np.squeeze(tok)
    if tok.ndim != 2:
        raise RuntimeError(f"moss_tts_ref: unexpected tokens shape {tok.shape}")
    if tok.shape[1] != 1 + n_vq and tok.shape[0] == 1 + n_vq:
        tok = tok.T  # orient to (seq_len, 1+n_vq)

    # Un-delay exactly like the C++ extract_audio_codes: last <audio_start> bounds
    # the segment, un-shift codebook cb by cb steps, T_audio = T - n_vq.
    col0 = tok[:, 0]
    starts = np.where(col0 == audio_start)[0]
    if len(starts) == 0:
        raise RuntimeError("moss_tts_ref: no <audio_start> in generated tokens")
    start = int(starts[-1]) + 1
    ends = np.where(col0[start:] == audio_end)[0]
    end = start + int(ends[0]) if len(ends) else tok.shape[0]
    T = end - start
    if T <= n_vq:
        raise RuntimeError(f"moss_tts_ref: audio segment too short (T={T} <= n_vq={n_vq})")
    T_audio = T - n_vq
    codes = np.zeros((n_vq, T_audio), dtype=np.int32)
    for cb in range(n_vq):
        for t in range(T_audio):
            codes[cb, t] = tok[start + t + cb, 1 + cb]
    # also expose the RAW delayed grid (from `start`) for warm-up/onset diffing
    raw = tok[start:end].astype(np.int32)
    return codes, raw


def dump(
    model_dir: "Path | str | None" = None,
    audio: np.ndarray | None = None,  # ignored (TTS has no input audio)
    stages: Set[str] | None = None,
    max_new_tokens: int | None = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Return {"codes": (n_vq, T) int32[, "waveform": (n,) f32]} for the greedy
    reference. Serialized to a GGUF fixture by the harness."""
    import torch

    if stages is None:
        stages = set(DEFAULT_STAGES)
    model_id = str(model_dir or os.environ.get("MOSS_TTS_MODEL", DEFAULT_MODEL))
    text = os.environ.get("MOSS_TTS_TEXT", "The quick brown fox jumps over the lazy dog.")
    seed = int(os.environ.get("MOSS_TTS_SEED", "0"))
    if max_new_tokens is None:
        max_new_tokens = int(os.environ.get("MOSS_TTS_MAXNEW", "512"))
    torch.manual_seed(seed)

    from transformers import AutoModel, AutoProcessor

    # The 8B bf16 (~16 GB) doesn't fit a 16 GB GPU (Kaggle P100/T4) — force CPU
    # there via MOSS_TTS_REF_DEVICE=cpu (fits ~29 GB host RAM in bf16).
    dev_env = os.environ.get("MOSS_TTS_REF_DEVICE", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if dev_env == "auto" else dev_env
    dtype = torch.bfloat16  # keep both greedy sides comparable; bf16 fits CPU RAM
    print(f"[moss_tts_ref] loading {model_id} on {device} ({dtype})", flush=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # MossTTSProcessor is a chat processor: build a user message, then call it
    # with conversations=[...] (NOT text=...). See processing_moss_tts.py.
    user_msg = processor.build_user_message(text=text, reference=None, instruction=None, tokens=None,
                                            quality=None, sound_event=None, ambient_sound=None, language=None)
    batch = processor(conversations=[user_msg], mode="generation", apply_chat_template=True)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(batch).items()}

    print(f"[moss_tts_ref] greedy generate (max_new_tokens={max_new_tokens})", flush=True)
    # Custom MossTTSDelay.generate: greedy = both temperatures 0 (no do_sample kwarg).
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                             max_new_tokens=max_new_tokens, text_temperature=0.0, audio_temperature=0.0)

    codes, raw = _extract_codes(model, out)
    print(f"[moss_tts_ref] codes shape={codes.shape} dtype={codes.dtype} "
          f"range=[{codes.min()},{codes.max()}]", flush=True)
    results: Dict[str, np.ndarray] = {"codes": codes.astype(np.int32),
                                      "raw": raw.astype(np.int32)}

    # Optional: decode the reference waveform via the codec.
    if "waveform" in stages:
        codec_id = os.environ.get("MOSS_TTS_CODEC", "")
        if codec_id:
            try:
                codec = AutoModel.from_pretrained(codec_id, trust_remote_code=True,
                                                  torch_dtype=torch.float32).to(device).eval()
                with torch.no_grad():
                    ct = torch.from_numpy(codes.astype(np.int64)).unsqueeze(0).to(device)
                    dec = codec.decode(ct) if hasattr(codec, "decode") else codec(ct)
                wav = np.squeeze(dec.detach().float().cpu().numpy())
                results["waveform"] = wav.astype(np.float32)
                print(f"[moss_tts_ref] waveform shape={results['waveform'].shape}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[moss_tts_ref] codec decode skipped: {type(e).__name__}: {e}", flush=True)

    return results


# Backwards-compatible alias used by some kernels (ref.run(model, idx, out_dir)).
def run(model_dir=None, idx: int = 0, out_dir: "Path | str | None" = None, **kwargs) -> Dict[str, np.ndarray]:
    res = dump(model_dir=model_dir, **kwargs)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in res.items():
            np.save(out_dir / f"{k}.npy", v)
    return res


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("moss_tts_ref")
    r = run(out_dir=out)
    for k, v in r.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
