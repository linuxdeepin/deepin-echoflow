"""kyutai/stt-1b-en_fr and kyutai/stt-2.6b-en reference dump backend.

Runs the full Mimi + LM pipeline in PyTorch and captures intermediate
activations at every architectural boundary for crispasr-diff.

Architecture:
  16 kHz PCM → append silence tail → resample 24 kHz
             → prepend silence prefix → SEANet CNN encoder (4 stride convs)
             → 8-layer encoder transformer → stride-2 downsample
             → RVQ (1 semantic + 31 acoustic codebooks)
             → causal LM → SentencePiece decode → text

Audio conditioning (matches the C++ runtime exactly):
  1. Silence tail: append (audio_delay + silence_prefix) seconds at 16 kHz
     so the causal LM can flush its final pending tokens.
  2. Resample 16 → 24 kHz (scipy.signal.resample).
  3. Silence prefix: prepend silence_prefix seconds at 24 kHz before Mimi
     encode (stt-2.6b-en uses 1.0 s; stt-1b-en_fr uses 0.0 s).

Without steps 1+3 the 2.6B model truncates the last ~3 words due to its
2.5 s audio_delay + 1.0 s silence_prefix lookahead.

Stages (all optional, controlled by --stages):

  raw_audio          (N,)            F32 input PCM at 16 kHz
  pcm_24k            (N24,)          F32 resampled+conditioned 24 kHz (Mimi input)
  seanet_output      (512, T_enc)    F32 after SEANet CNN stack
  enc_tfm_output     (T_enc, 512)    F32 after 8L encoder transformer
  downsampled        (512, T_fr)     F32 after stride-2 downsample conv
  rvq_codes          (n_q, T_fr)     F32 (cast from int32) RVQ code indices
  lm_frame0_logits   (text_card,)    F32 LM logits at frame 0
  generated_text     str             full transcript (→ GGUF metadata)

Usage:

  python tools/dump_reference.py --backend kyutai-stt \\
      --model-dir /hf/stt-2.6b-en \\
      --audio samples/jfk.wav \\
      --output /mnt/volume1/tmp-overflow/kyutai-stt-2.6b-ref.gguf

Requires:
  pip install safetensors sentencepiece scipy moshi
  (moshi for the LM decode stage; scipy for resampling)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "pcm_24k",
    "seanet_output",
    "enc_tfm_output",
    "downsampled",
    "rvq_codes",
    "lm_frame0_logits",
    "generated_text",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resample_24k(pcm_16k: np.ndarray) -> np.ndarray:
    """Resample 16 kHz float32 PCM to 24 kHz using scipy."""
    import scipy.signal
    n_out = int(len(pcm_16k) * 24000 / 16000)
    return scipy.signal.resample(pcm_16k.astype(np.float32), n_out)


def _streaming_conv1d(x, weight, bias, stride: int):
    """Causal-padded Conv1d matching moshi StreamingConv1d (left-pad only)."""
    import torch
    k = weight.shape[2]
    pad = k - stride
    if pad > 0:
        x = torch.nn.functional.pad(x, (pad, 0))
    return torch.nn.functional.conv1d(x, weight, bias, stride=stride, padding=0)


def _resblock(x, sd, idx: int):
    """One SEANet residual block (ELU + causal conv + ELU + causal conv + skip)."""
    import torch
    p1 = f"encoder.model.{idx}.block.1.conv.conv"
    p3 = f"encoder.model.{idx}.block.3.conv.conv"
    h = torch.nn.functional.elu(x)
    h = _streaming_conv1d(h, sd[p1 + ".weight"], sd.get(p1 + ".bias"), 1)
    h = torch.nn.functional.elu(h)
    h = _streaming_conv1d(h, sd[p3 + ".weight"], sd.get(p3 + ".bias"), 1)
    return x + h


def _apply_rope_interleaved(x, head_dim: int):
    """Interleaved RoPE ([r0,i0,r1,i1,...]) for the encoder transformer."""
    import torch
    T = x.shape[0]
    pos = torch.arange(T, dtype=torch.float32)
    ds = torch.arange(head_dim // 2, dtype=torch.float32)
    freqs = torch.exp(ds * (-np.log(10000.0) * 2 / head_dim))
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [T, hd/2]
    cos_v = torch.cos(angles)
    sin_v = torch.sin(angles)
    # x: [T, n_heads, head_dim]
    x2 = x.view(*x.shape[:-1], head_dim // 2, 2)
    xr, xi = x2[..., 0], x2[..., 1]
    cos_v = cos_v.unsqueeze(1)  # [T, 1, hd/2]
    sin_v = sin_v.unsqueeze(1)
    or_ = xr * cos_v - xi * sin_v
    oi = xr * sin_v + xi * cos_v
    return torch.stack([or_, oi], dim=-1).view(*x.shape)


def _nearest_neighbor_quantize(residual, codebook):
    """Nearest-neighbor VQ: residual [T, d], codebook [V, d] → codes [T]."""
    import torch
    dists = torch.cdist(residual.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
    return dists.argmin(dim=-1)


# ---------------------------------------------------------------------------
# main dump function
# ---------------------------------------------------------------------------

def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = np.asarray(audio, dtype=np.float32)

    # ── Resolve model files ──────────────────────────────────────────────────
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        # HF id → download
        from huggingface_hub import hf_hub_download
        cache = os.environ.get("HF_HOME", "/tmp") + "/hub"
        config_path = Path(hf_hub_download(str(model_dir), "config.json",
                                            cache_dir=cache))
        model_dir = config_path.parent

    with open(config_path) as f:
        config = json.load(f)

    mimi_name = config.get("mimi_name", "mimi-pytorch-e351c8d8@125.safetensors")
    tok_name = config.get("tokenizer_name", "tokenizer_en_fr_audio_8000.model")

    # Audio conditioning parameters from stt_config.
    stt_cfg = config.get("stt_config", {})
    audio_delay_s = float(stt_cfg.get("audio_delay_seconds", 0.5))
    silence_prefix_s = float(stt_cfg.get("audio_silence_prefix_seconds", 0.0))
    total_lookahead_s = audio_delay_s + silence_prefix_s

    print(f"  config: dim={config['dim']} layers={config['num_layers']} "
          f"heads={config['num_heads']} text_card={config['text_card']}")
    print(f"  audio_delay={audio_delay_s}s  silence_prefix={silence_prefix_s}s  "
          f"total_lookahead={total_lookahead_s}s")

    from safetensors import safe_open

    def _load_sd(path: Path):
        sd = {}
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k).float()
        return sd

    print("  loading Mimi weights …")
    mimi_sd = _load_sd(model_dir / mimi_name)

    # Pre-compute EMA codebook embeddings (embedding_sum / cluster_usage)
    codebooks: Dict[str, torch.Tensor] = {}
    for key in list(mimi_sd.keys()):
        if key.endswith("._codebook.embedding_sum"):
            prefix = key[: -len("embedding_sum")]
            usage = mimi_sd[prefix + "cluster_usage"].clamp(min=1e-5)
            codebooks[prefix + "embedding"] = mimi_sd[key] / usage.unsqueeze(-1)

    # ── Audio conditioning (must match C++ kyutai_stt_transcribe_impl) ───────
    #
    # Step 1: Append silence tail at 16 kHz so the causal LM can flush all
    # pending tokens. The tail length = total_lookahead = audio_delay +
    # silence_prefix. Without this, the 2.6B model (3.5 s lookahead) drops
    # the last ~3 words of the JFK clip.
    tail_16k = max(8000, int(total_lookahead_s * 16000))
    audio_with_tail = np.concatenate([
        audio.astype(np.float32),
        np.zeros(tail_16k, dtype=np.float32),
    ])
    print(f"  appended {tail_16k} silence tail samples at 16 kHz "
          f"({tail_16k / 16000:.2f} s)")

    # Step 2: Resample 16 → 24 kHz.
    print("  resampling to 24 kHz …")
    pcm_24k = _resample_24k(audio_with_tail)

    # Step 3: Prepend silence prefix at 24 kHz (1.0 s for 2.6B, 0.0 s for 1B).
    if silence_prefix_s > 0.0:
        n_prefix_24k = int(silence_prefix_s * 24000)
        pcm_24k = np.concatenate([
            np.zeros(n_prefix_24k, dtype=np.float32),
            pcm_24k,
        ])
        print(f"  prepended {n_prefix_24k} silence prefix samples at 24 kHz "
              f"({silence_prefix_s} s)")

    if "pcm_24k" in stages:
        out["pcm_24k"] = pcm_24k

    # Pad PCM to a multiple of 1920 (SEANet total stride)
    frame_size = 1920
    extra = (frame_size - len(pcm_24k) % frame_size) % frame_size
    pcm_t = torch.from_numpy(np.concatenate([pcm_24k,
                                              np.zeros(extra, dtype=np.float32)]))
    x = pcm_t.unsqueeze(0).unsqueeze(0)  # [1, 1, T]

    # ── SEANet CNN encoder ───────────────────────────────────────────────────
    print("  running SEANet …")
    x = _streaming_conv1d(x, mimi_sd["encoder.model.0.conv.conv.weight"],
                          mimi_sd.get("encoder.model.0.conv.conv.bias"), 1)

    for i, (rb_idx, st_idx, stride, kernel) in enumerate(
            zip([1, 4, 7, 10], [3, 6, 9, 12], [4, 5, 6, 8], [8, 10, 12, 16])):
        x = _resblock(x, mimi_sd, rb_idx)
        x = torch.nn.functional.elu(x)
        pfx = f"encoder.model.{st_idx}.conv.conv"
        x = _streaming_conv1d(x, mimi_sd[pfx + ".weight"],
                              mimi_sd.get(pfx + ".bias"), stride)

    x = torch.nn.functional.elu(x)
    x = _streaming_conv1d(x, mimi_sd["encoder.model.14.conv.conv.weight"],
                          mimi_sd.get("encoder.model.14.conv.conv.bias"), 1)
    # x: [1, 512, T_enc]
    if "seanet_output" in stages:
        out["seanet_output"] = x[0].detach().numpy()  # (512, T_enc)

    # ── Encoder transformer (8 layers) ──────────────────────────────────────
    print("  running encoder transformer …")
    enc = x.squeeze(0).permute(1, 0)  # [T_enc, 512]
    T_enc, dim = enc.shape
    n_heads, head_dim = 8, 64  # always for Mimi

    with torch.no_grad():
        for li in range(8):
            pfx = f"encoder_transformer.transformer.layers.{li}"
            res = enc
            nw = mimi_sd[f"{pfx}.norm1.weight"]
            nb = mimi_sd.get(f"{pfx}.norm1.bias")
            h = torch.nn.functional.layer_norm(enc, [dim], nw, nb)

            qkv_w = mimi_sd[f"{pfx}.self_attn.in_proj_weight"]
            qkv = h @ qkv_w.T
            Q, K, V = qkv.split(dim, dim=-1)
            Q = Q.view(T_enc, n_heads, head_dim)
            K = K.view(T_enc, n_heads, head_dim)
            V = V.view(T_enc, n_heads, head_dim)
            Q = _apply_rope_interleaved(Q, head_dim)
            K = _apply_rope_interleaved(K, head_dim)

            Q_t = Q.permute(1, 0, 2).unsqueeze(0)
            K_t = K.permute(1, 0, 2).unsqueeze(0)
            V_t = V.permute(1, 0, 2).unsqueeze(0)
            pos_q = torch.arange(T_enc).view(-1, 1)
            pos_k = torch.arange(T_enc).view(1, -1)
            delta = pos_q - pos_k
            mask = (delta >= 0) & (delta < 250)
            mask = mask.unsqueeze(0).unsqueeze(0)
            attn = torch.nn.functional.scaled_dot_product_attention(
                Q_t, K_t, V_t, attn_mask=mask, dropout_p=0.0)
            attn = attn.squeeze(0).permute(1, 0, 2).reshape(T_enc, dim)
            out_w = mimi_sd[f"{pfx}.self_attn.out_proj.weight"]
            attn = attn @ out_w.T
            ls1 = mimi_sd[f"{pfx}.layer_scale_1.scale"]
            enc = res + attn * ls1

            res2 = enc
            nw2 = mimi_sd[f"{pfx}.norm2.weight"]
            nb2 = mimi_sd.get(f"{pfx}.norm2.bias")
            h2 = torch.nn.functional.layer_norm(enc, [dim], nw2, nb2)
            ff1 = mimi_sd[f"{pfx}.linear1.weight"]
            ff2 = mimi_sd[f"{pfx}.linear2.weight"]
            h2 = torch.nn.functional.gelu(h2 @ ff1.T) @ ff2.T
            ls2 = mimi_sd[f"{pfx}.layer_scale_2.scale"]
            enc = res2 + h2 * ls2

    if "enc_tfm_output" in stages:
        out["enc_tfm_output"] = enc.detach().numpy()  # (T_enc, 512)

    # ── Downsample conv ──────────────────────────────────────────────────────
    feat_map = enc.permute(1, 0).unsqueeze(0)  # [1, 512, T_enc]
    ds_w = mimi_sd["downsample.conv.conv.conv.weight"]
    ds_b = mimi_sd.get("downsample.conv.conv.conv.bias")
    groups = 512 if (ds_w.shape[1] == 1 and ds_w.shape[0] == 512) else 1
    feat_map = torch.nn.functional.pad(feat_map, (2, 0))
    feat_map = torch.nn.functional.conv1d(feat_map, ds_w, ds_b, stride=2, groups=groups)
    T_fr = feat_map.shape[2]
    if "downsampled" in stages:
        out["downsampled"] = feat_map[0].detach().numpy()  # (512, T_fr)

    # ── RVQ quantizer ────────────────────────────────────────────────────────
    n_q_semantic = 1
    n_q_acoustic = int(config["n_q"]) - 1

    def _rvq_encode(x_map, group_prefix, n_cb):
        """x_map: [1, d, T] → codes [n_cb, T] (int64)."""
        pw = mimi_sd[f"{group_prefix}.input_proj.weight"]
        pb = mimi_sd.get(f"{group_prefix}.input_proj.bias")
        proj = torch.nn.functional.conv1d(x_map, pw, pb)  # [1, d_cb, T]
        residual = proj.squeeze(0).permute(1, 0)  # [T, d_cb]
        codes_list = []
        for q in range(n_cb):
            cb_key = f"{group_prefix}.vq.layers.{q}._codebook.embedding"
            cb = codebooks.get(cb_key, mimi_sd.get(cb_key))
            if cb is None:
                codes_list.append(torch.zeros(T_fr, dtype=torch.long))
                continue
            c = _nearest_neighbor_quantize(residual, cb)
            residual = residual - cb[c]
            codes_list.append(c)
        return torch.stack(codes_list, dim=0).numpy()  # (n_cb, T)

    codes_semantic = _rvq_encode(feat_map, "quantizer.rvq_first", n_q_semantic)
    codes_acoustic = _rvq_encode(feat_map, "quantizer.rvq_rest", n_q_acoustic)
    all_codes = np.concatenate([codes_semantic, codes_acoustic], axis=0)  # (n_q, T_fr)
    if "rvq_codes" in stages:
        out["rvq_codes"] = all_codes.astype(np.float32)  # store as F32 for GGUF

    del mimi_sd  # free memory before LM load

    # ── LM decode (via moshi) ────────────────────────────────────────────────
    need_lm = bool({"lm_frame0_logits", "generated_text"} & stages)
    if need_lm:
        print("  loading LM weights …")
        try:
            from moshi.models import lm as lm_module
        except ImportError as e:
            raise SystemExit(
                "moshi package required for LM stages.\n"
                "Install: pip install moshi\n"
                f"(import error: {e})")

        import sentencepiece as spm

        tok_path = model_dir / tok_name
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tok_path))

        lm_kwargs = {
            "dim": config["dim"],
            "text_card": config["text_card"],
            "existing_text_padding_id": config["existing_text_padding_id"],
            "n_q": config["n_q"],
            "dep_q": 0,
            "card": config["card"],
            "num_heads": config["num_heads"],
            "num_layers": config["num_layers"],
            "hidden_scale": config["hidden_scale"],
            "causal": True,
            "context": config["context"],
            "max_period": config["max_period"],
            "gating": "silu",
            "norm": "rms_norm_f32",
            "positional_embedding": "rope",
            "delays": config["delays"],
            "cross_attention": False,
        }
        lm_model = lm_module.LMModel(**lm_kwargs)
        lm_sd = {}
        with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                lm_sd[k] = f.get_tensor(k)
        lm_model.load_state_dict(lm_sd, strict=False)
        lm_model.eval()
        lm_model = lm_model.float()
        del lm_sd

        padding_id = int(config["existing_text_padding_id"])
        gen = lm_module.LMGen(lm_model, temp=0.0, temp_text=0.0,
                               top_k=250, top_k_text=50)

        codes_torch = torch.from_numpy(all_codes.astype(np.int64)).unsqueeze(0)  # [1, n_q, T]
        result_tokens: list[int] = []
        frame0_logits = None

        with torch.no_grad(), gen.streaming(1):
            for t in range(T_fr):
                audio_tok = codes_torch[:, :, t:t + 1]
                out_step = gen.step(audio_tok)

                if "lm_frame0_logits" in stages and t == 0 and frame0_logits is None:
                    # Capture raw logits before argmax on first frame.
                    # LMGen exposes the distribution on the text stream via the
                    # internal sampler — we re-derive logits from the hidden state
                    # only if the step returns a tuple with a logits field.
                    # Fall back to None if not available.
                    try:
                        # step() returns text_token (and optionally logits as
                        # a second element depending on moshi version)
                        if isinstance(out_step, tuple) and len(out_step) >= 2:
                            logit_cand = out_step[1]
                            if hasattr(logit_cand, "shape"):
                                frame0_logits = logit_cand.squeeze().float().numpy()
                    except Exception:  # noqa: BLE001
                        pass

                if out_step is not None:
                    tok = (out_step[0] if isinstance(out_step, tuple) else out_step)
                    result_tokens.append(int(tok.squeeze().item()))
                else:
                    result_tokens.append(0)

        if frame0_logits is not None and "lm_frame0_logits" in stages:
            out["lm_frame0_logits"] = frame0_logits.astype(np.float32)

        if "generated_text" in stages:
            text = ""
            for tok in result_tokens:
                if tok != 0 and tok != padding_id and tok < sp.GetPieceSize():
                    piece = sp.IdToPiece(tok).replace("\u2581", " ")
                    text += piece
            out["generated_text"] = text.strip()
            print(f"  transcript: {text.strip()!r}")

    return out
