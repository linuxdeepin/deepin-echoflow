#!/usr/bin/env python3
"""
Convert ai-sage/GigaAM-v3 (HuggingFace, trust_remote_code) → GGUF F16.

Five revisions live in the one HF repo, one per checkpoint; four of them are
ASR models and are supported here:

  ctc       Conformer + CTC head,  charwise vocab (33 Cyrillic chars + blank)
  rnnt      Conformer + RNN-T head, charwise vocab
  e2e_ctc   Conformer + CTC head,  SentencePiece 256, punctuation + ITN
  e2e_rnnt  Conformer + RNN-T head, SentencePiece 1024, punctuation + ITN

(the fifth, `ssl`, is the bare HuBERT-CTC pretrained encoder with no head — it
produces no transcript, so it is not converted.)

Architecture — read off `modeling_gigaam.py` + `config.json`, NOT inferred:

  preprocessor          torchaudio MelSpectrogram(n_mels=64, n_fft=320,
                        win_length=320, hop_length=160, center=False,
                        power=2, mel_scale='htk', norm=None) then
                        log(clamp(x, 1e-9, 1e9)).  NO per-feature z-norm.
                        The Hann window and the mel filterbank both live in
                        the checkpoint, so they are copied out verbatim
                        rather than rebuilt (removes a whole class of
                        window-convention / mel-norm scale bugs).

  encoder (16× Conformer, d_model=768, 16 heads, ff=3072):
    pre_encode          StridingSubsampling(conv1d): two
                        Conv1d(k=5, stride=2, pad=2) + ReLU stages
                        (64→768, 768→768) = 4× time downsample.
                        NOT the dw_striding Conv2d stack NeMo uses.
    layer i             FFN1(½) → MHA(rotary) → conv(dw k=5, LayerNorm)
                        → FFN2(½) → LayerNorm
    attention           RotaryPositionMultiHeadAttention. Two quirks worth
                        spelling out because both are easy to get wrong:
                          * RoPE is applied to the layer INPUT reshaped as
                            heads, BEFORE linear_q / linear_k — so
                            Q = Wq·RoPE(x), K = Wk·RoPE(x), V = Wv·x
                            (V is NOT rotated).
                          * the rotary base is `pos_emb_max_len`, i.e. 5000,
                            not the usual 10000 — RotaryPositionalEmbedding
                            is constructed as (dim=d_model//n_heads,
                            base=pos_emb_max_len).
                        rotate_half / NEOX pairing over head_dim=48.
    conv norm           conv_norm_type='layer_norm', so the module's
                        `batch_norm` submodule is really a LayerNorm — its
                        weight/bias are the LN affine, and there are no
                        running stats to fold.

  head (ctc)            Conv1d(768 → num_classes, k=1) then log_softmax
  head (rnnt)           predictor: Embedding(num_classes, 320) + 1-layer
                        LSTM(320, 320); joint: enc(768→320) + pred(320→320)
                        → ReLU → Linear(320 → num_classes).
                        blank_id = num_classes - 1.

GGUF tensor naming (mirrors the nemotron converter so the shared
core_conformer::BlockWeights loader maps 1:1):

  preprocessor.fb                                    F32  [n_mels, n_freqs]
  preprocessor.window                                F32  [win_length]

  encoder.pre.conv.{0,2}.{weight,bias}               F32

  encoder.layers.{i}.norm_ff1.{weight,bias}          F32
  encoder.layers.{i}.ff1.linear{1,2}.weight          F16
  encoder.layers.{i}.ff1.linear{1,2}.bias            F32
  encoder.layers.{i}.norm_attn.{weight,bias}         F32
  encoder.layers.{i}.attn.{q,k,v,out}.weight         F16
  encoder.layers.{i}.attn.{q,k,v,out}.bias           F32
  encoder.layers.{i}.norm_conv.{weight,bias}         F32
  encoder.layers.{i}.conv.pw1.weight                 F16
  encoder.layers.{i}.conv.dw.{weight,bias}           F16/F32
  encoder.layers.{i}.conv.ln.{weight,bias}           F32
  encoder.layers.{i}.conv.pw2.weight                 F16
  encoder.layers.{i}.norm_ff2.{weight,bias}          F32
  encoder.layers.{i}.norm_out.{weight,bias}          F32

  head.ctc.{weight,bias}                             F16/F32   (ctc heads)
  decoder.embed.weight                               F16       (rnnt heads)
  decoder.lstm.0.{w_ih,w_hh,b_ih,b_hh}               F16/F32
  joint.{enc,pred,out}.{weight,bias}                 F16/F32

GGUF metadata keys (under `gigaam.*`) — see the writer block below.

Usage:
  python models/convert-gigaam-to-gguf.py \
      --model /path/to/GigaAM-v3/e2e_rnnt --output gigaam-v3-e2e-rnnt-f16.gguf
  # or straight from the hub (downloads the revision):
  python models/convert-gigaam-to-gguf.py --model ai-sage/GigaAM-v3 \
      --revision e2e_rnnt --output gigaam-v3-e2e-rnnt-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    import torch
except ImportError:
    sys.exit("pip install torch")


# ---------------------------------------------------------------------------
# Source resolution — a local snapshot dir or an HF repo id + revision
# ---------------------------------------------------------------------------


def resolve_source(model: str, revision: str | None) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"'{model}' is not a directory and huggingface_hub is not installed")
    return Path(snapshot_download(model, revision=revision))


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------


def remap_name(src: str) -> str | None:
    """HF state-dict key → GGUF tensor name (None = drop)."""
    n = src
    # Every tensor is under the `model.` PreTrainedModel prefix.
    if n.startswith("model."):
        n = n[len("model."):]

    if n.endswith("num_batches_tracked"):
        return None
    # conv_norm_type='layer_norm' → the `batch_norm` submodule is an
    # nn.LayerNorm and carries no running stats. If a future checkpoint
    # switches to batch_norm these would need folding, not dropping.
    if "running_mean" in n or "running_var" in n:
        print(f"  [WARN] dropping BatchNorm running stat {src} — this build "
              f"assumes conv_norm_type=layer_norm", file=sys.stderr)
        return None

    # ---- preprocessor: Hann window + mel filterbank ----
    if n == "preprocessor.featurizer.0.spectrogram.window":
        return "preprocessor.window"
    if n == "preprocessor.featurizer.0.mel_scale.fb":
        return "preprocessor.fb"

    # ---- pre-encoder (conv1d striding subsampling) ----
    if n.startswith("encoder.pre_encode."):
        return n.replace("encoder.pre_encode.", "encoder.pre.")

    # ---- conformer layers ----
    if n.startswith("encoder.layers."):
        layer_id, sub = n[len("encoder.layers."):].split(".", 1)
        sub = (
            sub.replace("norm_feed_forward1", "norm_ff1")
            .replace("norm_feed_forward2", "norm_ff2")
            .replace("feed_forward1", "ff1")
            .replace("feed_forward2", "ff2")
            .replace("norm_self_att", "norm_attn")
            .replace("self_attn.linear_q", "attn.q")
            .replace("self_attn.linear_k", "attn.k")
            .replace("self_attn.linear_v", "attn.v")
            .replace("self_attn.linear_out", "attn.out")
            .replace("conv.pointwise_conv1", "conv.pw1")
            .replace("conv.depthwise_conv", "conv.dw")
            .replace("conv.pointwise_conv2", "conv.pw2")
            .replace("conv.batch_norm", "conv.ln")  # LayerNorm despite the name
        )
        return f"encoder.layers.{layer_id}.{sub}"

    # ---- CTC head ----
    if n == "head.decoder_layers.0.weight":
        return "head.ctc.weight"
    if n == "head.decoder_layers.0.bias":
        return "head.ctc.bias"

    # ---- RNN-T predictor ----
    if n == "head.decoder.embed.weight":
        return "decoder.embed.weight"
    lstm_map = {
        "weight_ih_l0": "lstm.0.w_ih",
        "weight_hh_l0": "lstm.0.w_hh",
        "bias_ih_l0": "lstm.0.b_ih",
        "bias_hh_l0": "lstm.0.b_hh",
    }
    if n.startswith("head.decoder.lstm."):
        suf = n[len("head.decoder.lstm."):]
        if suf in lstm_map:
            return f"decoder.{lstm_map[suf]}"

    # ---- RNN-T joint ----
    joint_map = {
        "head.joint.enc.weight": "joint.enc.weight",
        "head.joint.enc.bias": "joint.enc.bias",
        "head.joint.pred.weight": "joint.pred.weight",
        "head.joint.pred.bias": "joint.pred.bias",
        "head.joint.joint_net.1.weight": "joint.out.weight",
        "head.joint.joint_net.1.bias": "joint.out.bias",
    }
    if n in joint_map:
        return joint_map[n]

    print(f"  [WARN unmapped] {src}", file=sys.stderr)
    return None


def is_f32_tensor(name: str, shape: tuple[int, ...]) -> bool:
    if name.startswith("preprocessor."):
        return True
    if name.endswith(".bias"):
        return True
    if "norm" in name or ".ln." in name:
        return True
    # Pre-encode stays F32 for the same reason as nemotron (#81): the mel is
    # un-normalized log-mel in roughly [-20.7, +5], so the subsampling convs
    # produce large intermediates whose F16 rounding error cascades through
    # all 16 conformer layers. Both convs together are ~14 MB.
    if name.startswith("encoder.pre."):
        return True
    if len(shape) <= 1:
        return True
    return False


_QUANT_TYPE_MAP: dict[str, gguf.GGMLQuantizationType] = {
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(src_dir: Path, out_path: Path, quant: str | None = None) -> None:
    quant_type = _QUANT_TYPE_MAP.get(quant.lower()) if quant else None
    if quant and quant_type is None:
        sys.exit(f"Unknown --quant type '{quant}'. Choices: {list(_QUANT_TYPE_MAP)}")

    cfg_path = src_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"no config.json in {src_dir}")
    cfg = json.loads(cfg_path.read_text())["cfg"]["model"]["cfg"]

    prep = cfg["preprocessor"]
    enc = cfg["encoder"]
    head = cfg.get("head")
    decoding = cfg.get("decoding")
    if head is None or decoding is None:
        sys.exit(f"{src_dir} has no ASR head (the `ssl` revision is encoder-only)")

    model_class = cfg["model_class"]  # "ctc" | "rnnt"
    model_name = cfg.get("model_name", f"v3_{model_class}")
    if model_class not in ("ctc", "rnnt"):
        sys.exit(f"unsupported model_class '{model_class}'")

    weights_path = src_dir / "pytorch_model.bin"
    if not weights_path.exists():
        sys.exit(f"no pytorch_model.bin in {src_dir}")
    print(f"Loading: {weights_path}")
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)

    # ---- vocabulary: charwise list, or the SentencePiece pieces ----
    charwise = decoding.get("vocabulary") is not None
    if charwise:
        vocab = list(decoding["vocabulary"])
        tokenizer_type = "charwise"
    else:
        try:
            import sentencepiece as spm
        except ImportError:
            sys.exit("pip install sentencepiece (needed for the e2e_* revisions)")
        spm_path = src_dir / decoding.get("model_path", "tokenizer.model")
        if not spm_path.exists():
            spm_path = src_dir / "tokenizer.model"
        sp = spm.SentencePieceProcessor()
        sp.load(str(spm_path))
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        tokenizer_type = "spm"
    print(f"  vocab: {len(vocab)} entries ({tokenizer_type})")

    # ---- hyper-parameters (cross-checked against the tensors) ----
    sr = int(prep["sample_rate"])
    n_mels = int(prep["features"])
    n_fft = int(prep["n_fft"])
    win_length = int(prep["win_length"])
    hop_length = int(prep["hop_length"])
    center = bool(prep.get("center", True))
    if center:
        sys.exit("this build assumes preprocessor center=False (GigaAM-v3 default)")
    if prep.get("mel_scale", "htk") != "htk" or prep.get("mel_norm") is not None:
        print(f"  [WARN] mel_scale={prep.get('mel_scale')} mel_norm={prep.get('mel_norm')}"
              f" — the filterbank is copied from the checkpoint so this is informational",
              file=sys.stderr)

    d_model = int(enc["d_model"])
    n_layers = int(enc["n_layers"])
    n_heads = int(enc["n_heads"])
    head_dim = d_model // n_heads
    ff_dim = int(enc["ff_expansion_factor"]) * d_model
    subsampling = enc.get("subsampling", "conv1d")
    if subsampling != "conv1d":
        sys.exit(f"unsupported subsampling '{subsampling}' (only conv1d is implemented)")
    subsampling_factor = int(enc["subsampling_factor"])
    subs_kernel = int(enc["subs_kernel_size"])
    conv_kernel = int(enc["conv_kernel_size"])
    conv_norm_type = enc.get("conv_norm_type", "batch_norm")
    if conv_norm_type != "layer_norm":
        sys.exit(f"unsupported conv_norm_type '{conv_norm_type}'")
    self_attention_model = enc.get("self_attention_model", "rotary")
    if self_attention_model != "rotary":
        sys.exit(f"unsupported self_attention_model '{self_attention_model}'")
    pos_emb_max_len = int(enc["pos_emb_max_len"])
    # RotaryPositionalEmbedding(dim=d_model//n_heads, base=pos_emb_max_len):
    # the second positional arg of PositionalEncoding is `base`, so the rotary
    # base really is pos_emb_max_len (5000), not the customary 10000.
    rope_base = pos_emb_max_len

    if model_class == "ctc":
        num_classes = int(head["num_classes"])
        pred_hidden = 0
        joint_hidden = 0
    else:
        num_classes = int(head["decoder"]["num_classes"])
        pred_hidden = int(head["decoder"]["pred_hidden"])
        pred_layers = int(head["decoder"]["pred_rnn_layers"])
        joint_hidden = int(head["joint"]["joint_hidden"])
        if pred_layers != 1:
            sys.exit(f"unsupported pred_rnn_layers={pred_layers} (only 1 is implemented)")
    blank_id = num_classes - 1
    if blank_id != len(vocab):
        print(f"  [WARN] blank_id={blank_id} != len(vocab)={len(vocab)}", file=sys.stderr)

    frame_dur_cs = int(round(hop_length * subsampling_factor * 100 / sr))
    print(f"  hparams: d_model={d_model} layers={n_layers} heads={n_heads} ff={ff_dim} "
          f"head={model_class} classes={num_classes} n_mels={n_mels} "
          f"rope_base={rope_base} frame={frame_dur_cs}cs")

    # ---- write GGUF ----
    print(f"Writing: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="gigaam")

    writer.add_string("gigaam.model_name", model_name)
    writer.add_uint32("gigaam.sample_rate", sr)
    writer.add_uint32("gigaam.n_mels", n_mels)
    writer.add_uint32("gigaam.n_fft", n_fft)
    writer.add_uint32("gigaam.win_length", win_length)
    writer.add_uint32("gigaam.hop_length", hop_length)
    writer.add_bool("gigaam.mel_center", center)
    writer.add_uint32("gigaam.d_model", d_model)
    writer.add_uint32("gigaam.n_layers", n_layers)
    writer.add_uint32("gigaam.n_heads", n_heads)
    writer.add_uint32("gigaam.head_dim", head_dim)
    writer.add_uint32("gigaam.ff_dim", ff_dim)
    writer.add_uint32("gigaam.subsampling_factor", subsampling_factor)
    writer.add_uint32("gigaam.subsampling_channels", d_model)
    writer.add_uint32("gigaam.subs_kernel", subs_kernel)
    writer.add_uint32("gigaam.conv_kernel", conv_kernel)
    writer.add_string("gigaam.conv_norm_type", conv_norm_type)
    writer.add_string("gigaam.self_attention_model", self_attention_model)
    writer.add_uint32("gigaam.pos_emb_max_len", pos_emb_max_len)
    writer.add_float32("gigaam.rope_base", float(rope_base))
    writer.add_string("gigaam.head_type", model_class)
    writer.add_uint32("gigaam.num_classes", num_classes)
    writer.add_uint32("gigaam.blank_id", blank_id)
    writer.add_uint32("gigaam.vocab_size", len(vocab))
    writer.add_string("gigaam.tokenizer_type", tokenizer_type)
    writer.add_uint32("gigaam.frame_dur_cs", frame_dur_cs)
    if model_class == "rnnt":
        writer.add_uint32("gigaam.pred_hidden", pred_hidden)
        writer.add_uint32("gigaam.pred_layers", 1)
        writer.add_uint32("gigaam.joint_hidden", joint_hidden)
    writer.add_array("tokenizer.ggml.tokens", vocab)

    # ---- tensors ----
    n_written = n_f16 = n_f32 = n_quant = n_unmapped = 0
    for src in sorted(sd.keys()):
        name = remap_name(src)
        if name is None:
            if not src.endswith("num_batches_tracked") and "running_" not in src:
                n_unmapped += 1
            continue
        t = sd[src].float().numpy()  # .float() first: numpy has no bf16/f16 path we want

        # torchaudio's MelScale.fb is (n_freqs, n_mels); core_mel wants the
        # MelsFreqs layout fb[m * n_freqs + k].
        if name == "preprocessor.fb":
            assert t.shape == (n_fft // 2 + 1, n_mels), f"unexpected fb shape {t.shape}"
            t = np.ascontiguousarray(t.T)
        # CTC head is a k=1 Conv1d — drop the trailing kernel axis so the
        # runtime can mul_mat it directly.
        if name == "head.ctc.weight":
            t = t.reshape(t.shape[0], t.shape[1])

        raw_dtype = None
        if is_f32_tensor(name, t.shape):
            t = t.astype(np.float32)
            n_f32 += 1
        elif quant_type is not None:
            try:
                t = gguf.quantize(t.astype(np.float32), quant_type)
                raw_dtype = quant_type
                n_quant += 1
            except Exception:
                t = t.astype(np.float16)
                n_f16 += 1
        else:
            t = t.astype(np.float16)
            n_f16 += 1

        writer.add_tensor(name, t, raw_dtype=raw_dtype)
        n_written += 1
        if n_written <= 24 or n_written % 60 == 0:
            label = quant_type.name if raw_dtype else str(t.dtype)
            print(f"  {name:52s}  {str(t.shape):24s}  {label}")

    quant_label = f", {quant_type.name}: {n_quant}" if quant_type else ""
    print(f"\n  total tensors: {n_written}  (F16: {n_f16}, F32: {n_f32}{quant_label})")
    if n_unmapped:
        sys.exit(f"{n_unmapped} tensor(s) were unmapped — see the [WARN unmapped] lines")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert ai-sage/GigaAM-v3 → GGUF")
    p.add_argument("--model", required=True,
                   help="local snapshot dir, or the HF repo id 'ai-sage/GigaAM-v3'")
    p.add_argument("--revision", default=None,
                   help="HF revision when --model is a repo id: ctc | rnnt | e2e_ctc | e2e_rnnt")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    p.add_argument("--quant", default=None, help="q4_k | q8_0 (default: F16)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(resolve_source(args.model, args.revision), args.output, quant=args.quant)
