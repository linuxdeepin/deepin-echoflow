"""mistralai/Voxtral-4B-TTS-2603 reference dump backend.

Uses vllm-omni (or manual PyTorch) to run the full TTS pipeline and
captures intermediate activations for crispasr-diff comparison.

Stages:

  raw_text           str           input text (→ GGUF metadata)
  voice_name         str           voice preset name (→ GGUF metadata)
  text_token_ids     (N,)          F32 Tekken BPE token IDs
  voice_embedding    (T_voice, D)  F32 pre-summed voice embeddings
  llm_hidden_frame0  (D,)          F32 LLM hidden state at first generated frame
  semantic_codes     (T_gen,)      F32 semantic token IDs per frame
  acoustic_codes     (T_gen, 36)   F32 acoustic FSQ values per frame
  generated_audio    (N_pcm,)      F32 24 kHz mono PCM output
  generated_text     str           echo of input text (→ GGUF metadata)

Usage:

  python tools/dump_reference.py --backend voxtral-tts \\
      --model-dir /hf/Voxtral-4B-TTS-2603 \\
      --audio samples/jfk.wav \\
      --output /mnt/volume1/tmp-overflow/voxtral-tts-ref.gguf

  Env vars:
    VOXTRAL_TTS_TEXT   — synthesis text (default "Hello world.")
    VOXTRAL_TTS_VOICE  — voice preset (default "fr_female")

Requires: torch, safetensors, sentencepiece
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

# LLM backbone params (Ministral-3B; from voxtral_tts hparams). vllm-omni is NOT
# required — the LLM is a standard Mistral stack loadable straight from the native
# consolidated.safetensors and run with a manual forward (this is what the mudler C
# reference does). The FM + codec are TTS-custom; the codec is already validated
# (cos 0.9999), so this dumper covers the per-layer LLM path where a structural bug
# would live. Frame-0 stage names match the runtime's CRISPASR_VOXTRAL_TTS_DIFF_DUMP.
_D, _NL, _NH, _NKV, _HD = 3072, 26, 32, 8, 128
_THETA, _EPS = 1_000_000.0, 1e-5

DEFAULT_STAGES = ["text_token_ids", "voice_embedding", "embed"] \
    + [f"llm_L{i}" for i in range(_NL)] + ["hidden", "generated_text"]


def _rms_norm(x, w, eps):
    import torch
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rope_normal(x, pos, head_dim, theta):
    """GGML_ROPE_TYPE_NORMAL: rotate ADJACENT pairs (2i, 2i+1). x: (T, H, head_dim)."""
    import torch
    half = head_dim // 2
    inv = theta ** (-torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim)
    ang = pos[:, None].float() * inv[None, :]           # (T, half)
    cos, sin = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Manual PyTorch forward of the LLM backbone → per-layer frame-0 hidden.

    Runs WITHOUT vllm/transformers: loads the native consolidated.safetensors keys
    lazily (one layer at a time, ~sub-2 GB peak) and runs a single causal forward over
    [BOS][BEGIN_AUDIO][voice][/INST=36]text[INST=35][BEGIN_AUDIO][AUDIO=24]; the last
    position (the AUDIO token) is the frame-0 h0 decode. Dumps per-layer residual +
    final hidden for a per-layer cos diff vs the runtime.
    """
    import torch
    from safetensors import safe_open

    out: Dict[str, np.ndarray] = {}
    md = Path(model_dir)
    text = os.environ.get("VOXTRAL_TTS_TEXT", "Hello world.")
    voice = os.environ.get("VOXTRAL_TTS_VOICE", "neutral_female")
    out["generated_text"] = text
    print(f"  text={text!r} voice={voice}  (manual PyTorch LLM forward, no vllm)")

    sf = safe_open(str(md / "consolidated.safetensors"), framework="pt")

    def load(name):
        return sf.get_tensor(name).float()

    # Tekken tokenize (class name / ctor vary across mistral_common versions)
    import mistral_common.tokens.tokenizers.tekken as _tk
    _Tek = getattr(_tk, "Tekkenizer", None) or getattr(_tk, "Tekken")
    tekp = str(md / "tekken.json")
    tok = _Tek.from_file(tekp) if hasattr(_Tek, "from_file") else _Tek(tekp)
    text_ids = tok.encode(text, False, False)  # (s, bos, eos)
    out["text_token_ids"] = np.array(text_ids, dtype=np.float32)

    # Preset voice embedding (pre-summed, spliced into the prompt)
    ve = torch.load(str(md / "voice_embedding" / f"{voice}.pt"),
                    map_location="cpu", weights_only=True).float()
    if ve.dim() == 1:
        ve = ve.view(-1, _D)
    out["voice_embedding"] = ve.numpy()

    emb_w = load("mm_audio_embeddings.tok_embeddings.weight")  # (V, D)

    def e(tid):
        return emb_w[tid]

    # [BOS=1][BEGIN_AUDIO=25][voice…][/INST=36]text[INST=35][BEGIN_AUDIO=25][AUDIO=24]
    seq = [e(1), e(25)] + [ve[i] for i in range(ve.shape[0])] + [e(36)] \
        + [e(t) for t in text_ids] + [e(35), e(25), e(24)]
    x = torch.stack(seq, 0).float()  # (T, D); last row = AUDIO token = frame-0 position
    T = x.shape[0]
    pos = torch.arange(T)
    out["embed"] = x[-1].numpy()
    print(f"  seq len={T} (frame-0 decode at last position, pos={T - 1})")

    scale = 1.0 / (_HD ** 0.5)
    grp = _NH // _NKV
    cmask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    for il in range(_NL):
        xn = _rms_norm(x, load(f"layers.{il}.attention_norm.weight"), _EPS)
        q = (xn @ load(f"layers.{il}.attention.wq.weight").T).view(T, _NH, _HD)
        k = (xn @ load(f"layers.{il}.attention.wk.weight").T).view(T, _NKV, _HD)
        v = (xn @ load(f"layers.{il}.attention.wv.weight").T).view(T, _NKV, _HD)
        q = _rope_normal(q, pos, _HD, _THETA)
        k = _rope_normal(k, pos, _HD, _THETA)
        k = k.repeat_interleave(grp, dim=1)  # GQA → (T, NH, HD)
        v = v.repeat_interleave(grp, dim=1)
        qh, kh, vh = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        att = torch.softmax((qh @ kh.transpose(1, 2)) * scale + cmask, dim=-1)
        o = (att @ vh).permute(1, 0, 2).reshape(T, _NH * _HD)
        x = x + (o @ load(f"layers.{il}.attention.wo.weight").T)
        xn = _rms_norm(x, load(f"layers.{il}.ffn_norm.weight"), _EPS)
        gate = torch.nn.functional.silu(xn @ load(f"layers.{il}.feed_forward.w1.weight").T) \
            * (xn @ load(f"layers.{il}.feed_forward.w3.weight").T)
        x = x + (gate @ load(f"layers.{il}.feed_forward.w2.weight").T)
        out[f"llm_L{il}"] = x[-1].numpy()

    out["hidden"] = _rms_norm(x, load("norm.weight"), _EPS)[-1].numpy()
    print(f"  dumped {_NL} layers + embed + hidden  |h0|={float(np.linalg.norm(out['hidden'])):.3f}")
    return out
