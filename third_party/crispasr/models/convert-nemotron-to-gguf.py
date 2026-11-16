#!/usr/bin/env python3
"""
Convert nvidia/nemotron-3.5-asr-streaming-0.6b (a NeMo .nemo checkpoint) → GGUF F16.

Architecture (from model_config.yaml + tensor inspection):

  preprocessor (mel):     128 mel bins, n_fft=512, win=25ms, stride=10ms (Hann)
                          normalize="NA" (no normalization applied by preprocessor)
  encoder (24× Cache-Aware FastConformer):
    pre_encode:           3-stage dw_striding Conv2d (8× time downsample, causal)
                          out: linear(4352 → 1024)  [4352 = 256ch * 17 freq bins]
    layer i:              FFN1(½) → MHA(rel_pos, untied bias) → conv(dw, k=9, LN) → FFN2(½) → LN
    d_model = 1024  n_heads = 8  ff = 4096
    conv uses LayerNorm (not BatchNorm) — weight + bias, no running stats
    streaming: att_context_size=[[56,3],[56,0],[56,6],[56,13]]
               att_context_style=chunked_limited, conv_context_size=causal
  prompt_kernel:          MLP(1152 → 2048 → 1024) for language prompt features
  decoder.prediction:     embed(13088, 640) + 2-layer LSTM(640, 640)
  joint:                  enc(1024→640) + pred(640→640) → relu → linear(640 → 13088)
                          13088 = 13087 vocab + 1 blank (pure RNNT, no TDT durations)

GGUF tensor naming:
  preprocessor.fb                                    F32
  preprocessor.window                                F32

  encoder.pre.conv.{0,2,3,5,6}.{weight,bias}         F16/F32
  encoder.pre.out.{weight,bias}                      F16/F32

  encoder.layers.{i}.norm_ff1.{weight,bias}          F32
  encoder.layers.{i}.ff1.linear1.weight              F16
  encoder.layers.{i}.ff1.linear2.weight              F16
  ... (analogous for ff2)

  encoder.layers.{i}.norm_attn.{weight,bias}         F32
  encoder.layers.{i}.attn.{q,k,v,out,pos}.weight     F16
  encoder.layers.{i}.attn.pos_bias_u                 F32
  encoder.layers.{i}.attn.pos_bias_v                 F32

  encoder.layers.{i}.norm_conv.{weight,bias}         F32
  encoder.layers.{i}.conv.pw1.weight                 F16
  encoder.layers.{i}.conv.dw.weight                  F16
  encoder.layers.{i}.conv.ln.{weight,bias}           F32  (LayerNorm, NOT BatchNorm)
  encoder.layers.{i}.conv.pw2.weight                 F16

  encoder.layers.{i}.norm_ff2.{weight,bias}          F32
  encoder.layers.{i}.norm_out.{weight,bias}          F32

  prompt_kernel.0.{weight,bias}                      F16/F32
  prompt_kernel.2.{weight,bias}                      F16/F32

  decoder.embed.weight                               F16
  decoder.lstm.{0,1}.{w_ih,w_hh,b_ih,b_hh}           F16/F32

  joint.enc.{weight,bias}                            F16/F32
  joint.pred.{weight,bias}                           F16/F32
  joint.out.{weight,bias}                            F16/F32

GGUF metadata keys (under `nemotron.*`):
  nemotron.sample_rate           = 16000
  nemotron.n_mels                = 128
  nemotron.n_fft                 = 512
  nemotron.win_length            = 400
  nemotron.hop_length            = 160
  nemotron.d_model               = 1024
  nemotron.n_layers              = 24
  nemotron.n_heads               = 8
  nemotron.head_dim              = 128
  nemotron.ff_dim                = 4096
  nemotron.subsampling_factor    = 8
  nemotron.subsampling_channels  = 256
  nemotron.conv_kernel           = 9
  nemotron.pred_hidden           = 640
  nemotron.pred_layers           = 2
  nemotron.joint_hidden          = 640
  nemotron.vocab_size            = 13087
  nemotron.blank_id              = 13087
  nemotron.n_tdt_durations       = 0     (pure RNNT)
  nemotron.frame_dur_cs          = 8
  nemotron.causal_downsampling   = true
  nemotron.conv_norm_type        = "layer_norm"
  nemotron.att_context_style     = "chunked_limited"
  nemotron.n_att_context_presets  = 4
  nemotron.att_context_left.N    = [56, 56, 56, 56]
  nemotron.att_context_right.N   = [3, 0, 6, 13]
  nemotron.num_prompts           = 128
  nemotron.prompt_kernel_in      = 1152
  nemotron.prompt_kernel_mid     = 2048

  tokenizer.ggml.tokens          = [<13087 strings from SentencePiece>]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
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
try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")


# ---------------------------------------------------------------------------
# .nemo loading — in-memory path (no disk extraction)
# ---------------------------------------------------------------------------


def load_nemo_inmem(nemo_path: Path) -> dict:
    """Load .nemo tarball entirely in RAM."""
    import io as _io
    result = {}
    with tarfile.open(nemo_path, "r") as tf:
        for m in tf.getmembers():
            n = m.name
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            data = fobj.read()
            if n.endswith("model_weights.ckpt"):
                buf = _io.BytesIO(data)
                del data
                sd = torch.load(buf, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                result["weights"] = sd
            elif n.endswith("model_config.yaml"):
                result["config_str"] = data.decode()
            elif n.endswith("_tokenizer.model"):
                result["spm_bytes"] = data
            elif n.endswith("_vocab.txt"):
                result["vocab_bytes"] = data
    if "weights" not in result or "spm_bytes" not in result:
        sys.exit(f"could not find weights / tokenizer in {nemo_path}")
    return result


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------


def remap_name(nemo_name: str) -> str | None:
    n = nemo_name

    # Skip BatchNorm stats counter
    if n.endswith("num_batches_tracked"):
        return None
    # Skip running_mean/running_var — this model uses LayerNorm in conv
    if "running_mean" in n or "running_var" in n:
        return None

    # ---- preprocessor (mel filterbank + Hann window) ----
    if n == "preprocessor.featurizer.fb":
        return "preprocessor.fb"
    if n == "preprocessor.featurizer.window":
        return "preprocessor.window"

    # ---- pre-encoder (subsampling Conv2d stack) ----
    if n.startswith("encoder.pre_encode."):
        return n.replace("encoder.pre_encode.", "encoder.pre.")

    # ---- conformer layers ----
    if n.startswith("encoder.layers."):
        rest = n[len("encoder.layers."):]
        layer_id, sub = rest.split(".", 1)
        sub = (
            sub.replace("feed_forward1", "ff1")
            .replace("feed_forward2", "ff2")
            .replace("norm_feed_forward1", "norm_ff1")
            .replace("norm_feed_forward2", "norm_ff2")
            .replace("norm_self_att", "norm_attn")
            .replace("self_attn.linear_q", "attn.q")
            .replace("self_attn.linear_k", "attn.k")
            .replace("self_attn.linear_v", "attn.v")
            .replace("self_attn.linear_out", "attn.out")
            .replace("self_attn.linear_pos", "attn.pos")
            .replace("self_attn.pos_bias_u", "attn.pos_bias_u")
            .replace("self_attn.pos_bias_v", "attn.pos_bias_v")
            .replace("conv.pointwise_conv1", "conv.pw1")
            .replace("conv.depthwise_conv", "conv.dw")
            .replace("conv.pointwise_conv2", "conv.pw2")
            .replace("conv.batch_norm", "conv.ln")  # LayerNorm stored as batch_norm in NeMo
        )
        return f"encoder.layers.{layer_id}.{sub}"

    # ---- prompt kernel (language prompt MLP) ----
    if n.startswith("prompt_kernel."):
        return n

    # ---- decoder (predictor) ----
    if n == "decoder.prediction.embed.weight":
        return "decoder.embed.weight"
    if n.startswith("decoder.prediction.dec_rnn.lstm."):
        suf = n[len("decoder.prediction.dec_rnn.lstm."):]
        for key, gguf_key in [
            ("weight_ih_l0", "lstm.0.w_ih"),
            ("weight_hh_l0", "lstm.0.w_hh"),
            ("bias_ih_l0", "lstm.0.b_ih"),
            ("bias_hh_l0", "lstm.0.b_hh"),
            ("weight_ih_l1", "lstm.1.w_ih"),
            ("weight_hh_l1", "lstm.1.w_hh"),
            ("bias_ih_l1", "lstm.1.b_ih"),
            ("bias_hh_l1", "lstm.1.b_hh"),
        ]:
            if suf == key:
                return f"decoder.{gguf_key}"

    # ---- joint ----
    if n == "joint.enc.weight":
        return "joint.enc.weight"
    if n == "joint.enc.bias":
        return "joint.enc.bias"
    if n == "joint.pred.weight":
        return "joint.pred.weight"
    if n == "joint.pred.bias":
        return "joint.pred.bias"
    if n == "joint.joint_net.2.weight":
        return "joint.out.weight"
    if n == "joint.joint_net.2.bias":
        return "joint.out.bias"

    print(f"  [WARN unmapped] {n}", file=sys.stderr)
    return None


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    if gguf_name.startswith("preprocessor."):
        return True
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name:
        return True
    if "bn" in gguf_name or "ln" in gguf_name:
        return True
    if "pos_bias_u" in gguf_name or "pos_bias_v" in gguf_name:
        return True
    # Pre-encode weights: keep all in F32. The unnormalized mel values
    # (range [-16, +2.5]) produce large conv intermediates (up to ~60).
    # F16 rounding errors accumulate across 256 channels x 17 freq bins
    # = 4352-dim linear projection, causing up to 1.6 absolute error per
    # frame vs NeMo F32 — which cascades through 24 conformer layers into
    # completely wrong encoder output. Pre-encode is only ~20 MB. (#81)
    if gguf_name.startswith("encoder.pre."):
        return True
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


_QUANT_TYPE_MAP: dict[str, gguf.GGMLQuantizationType] = {
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}


def convert(nemo_path: Path, out_path: Path, quant: str | None = None) -> None:
    quant_type = _QUANT_TYPE_MAP.get(quant.lower()) if quant else None
    if quant and quant_type is None:
        sys.exit(f"Unknown --quant type '{quant}'. Choices: {list(_QUANT_TYPE_MAP)}")

    print(f"Loading: {nemo_path}  (in-memory, no disk extraction)")
    nemo_data = load_nemo_inmem(nemo_path)
    sd = nemo_data["weights"]

    import yaml
    cfg = yaml.safe_load(nemo_data["config_str"])

    import io as _io
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(nemo_data["spm_bytes"])
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print(f"  vocab:  {len(vocab)} pieces")

    # ----- write GGUF -----
    print(f"Writing: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="nemotron")

    # Hyper-parameters from model_config.yaml
    prep = cfg.get("preprocessor", {}) if cfg else {}
    enc_cfg = cfg.get("encoder", {}) if cfg else {}
    dec_cfg = cfg.get("decoder", {}) if cfg else {}
    pred_cfg = dec_cfg.get("prednet", {}) if cfg else {}
    joint_cfg = cfg.get("joint", {}) if cfg else {}
    joint_net = joint_cfg.get("jointnet", {}) if cfg else {}
    model_defaults = cfg.get("model_defaults", {}) if cfg else {}

    feat_in = enc_cfg.get("feat_in", prep.get("features", 128))
    sr = prep.get("sample_rate", cfg.get("sample_rate", 16000))
    n_fft = prep.get("n_fft", 512)
    ws = prep.get("window_size", 0.025)
    wst = prep.get("window_stride", 0.01)
    d_model = enc_cfg.get("d_model", 1024)
    n_layers = enc_cfg.get("n_layers", 24)
    n_heads = enc_cfg.get("n_heads", 8)
    head_dim = d_model // n_heads
    ff_dim = enc_cfg.get("ff_expansion_factor", 4) * d_model
    subsampling_factor = enc_cfg.get("subsampling_factor", 8)
    subsampling_channels = enc_cfg.get("subsampling_conv_channels", 256)
    conv_kernel = enc_cfg.get("conv_kernel_size", 9)
    xscaling = bool(enc_cfg.get("xscaling", False))
    causal_downsampling = bool(enc_cfg.get("causal_downsampling", True))
    conv_norm_type = enc_cfg.get("conv_norm_type", "layer_norm")
    att_context_style = enc_cfg.get("att_context_style", "chunked_limited")
    pred_hidden = pred_cfg.get("pred_hidden", 640)
    pred_layers = pred_cfg.get("pred_rnn_layers", 2)
    joint_hidden = (joint_net.get("joint_hidden")
                    or joint_cfg.get("encoder_hidden")
                    or joint_net.get("encoder_hidden")
                    or 640)

    # Streaming context presets
    att_context_size = enc_cfg.get("att_context_size", [[56, 3]])
    if isinstance(att_context_size[0], int):
        att_context_size = [att_context_size]
    n_presets = len(att_context_size)
    att_left = [p[0] for p in att_context_size]
    att_right = [p[1] for p in att_context_size]

    # Prompt features
    num_prompts = model_defaults.get("num_prompts", 128)

    # Cross-check: joint.out rows should be vocab+1 (pure RNNT)
    joint_out_w = sd.get("joint.joint_net.2.weight")
    actual_joint_out = int(joint_out_w.shape[0]) if joint_out_w is not None else 0
    vocab_plus_blank = len(vocab) + 1
    n_tdt_durations = 0
    tdt_durations = []
    if actual_joint_out and actual_joint_out != vocab_plus_blank:
        print(
            f"  [warn] joint.out.weight rows={actual_joint_out} != vocab+blank={vocab_plus_blank}",
            file=sys.stderr,
        )

    # Cross-check pred_hidden from LSTM weight
    lstm0_w_ih = sd.get("decoder.prediction.dec_rnn.lstm.weight_ih_l0")
    if lstm0_w_ih is not None:
        actual = int(lstm0_w_ih.shape[1])
        if actual != pred_hidden:
            print(f"  [warn] pred_hidden={pred_hidden} vs lstm shape={actual}; using {actual}", file=sys.stderr)
            pred_hidden = actual

    # Cross-check joint_hidden
    joint_pred_w = sd.get("joint.pred.weight")
    if joint_pred_w is not None:
        actual = int(joint_pred_w.shape[0])
        if actual != joint_hidden:
            print(f"  [warn] joint_hidden={joint_hidden} vs tensor={actual}; using {actual}", file=sys.stderr)
            joint_hidden = actual

    # Prompt kernel dimensions
    pk0_w = sd.get("prompt_kernel.0.weight")
    prompt_kernel_in = int(pk0_w.shape[1]) if pk0_w is not None else 1152
    prompt_kernel_mid = int(pk0_w.shape[0]) if pk0_w is not None else 2048

    print(
        f"  hparams: d_model={d_model} layers={n_layers} heads={n_heads} "
        f"ff={ff_dim} pred_hidden={pred_hidden} joint_hidden={joint_hidden} "
        f"n_mels={feat_in} vocab={len(vocab)} causal_ds={causal_downsampling} "
        f"conv_norm={conv_norm_type} att_presets={n_presets}"
    )

    writer.add_uint32("nemotron.sample_rate", sr)
    writer.add_uint32("nemotron.n_mels", feat_in)
    writer.add_uint32("nemotron.n_fft", n_fft)
    writer.add_uint32("nemotron.win_length", int(ws * sr))
    writer.add_uint32("nemotron.hop_length", int(wst * sr))
    writer.add_uint32("nemotron.d_model", d_model)
    writer.add_uint32("nemotron.n_layers", n_layers)
    writer.add_uint32("nemotron.n_heads", n_heads)
    writer.add_uint32("nemotron.head_dim", head_dim)
    writer.add_uint32("nemotron.ff_dim", ff_dim)
    writer.add_uint32("nemotron.subsampling_factor", subsampling_factor)
    writer.add_uint32("nemotron.subsampling_channels", subsampling_channels)
    writer.add_uint32("nemotron.conv_kernel", conv_kernel)
    writer.add_bool("nemotron.xscaling", xscaling)
    writer.add_bool("nemotron.causal_downsampling", causal_downsampling)
    writer.add_string("nemotron.conv_norm_type", conv_norm_type)
    writer.add_string("nemotron.att_context_style", att_context_style)
    writer.add_uint32("nemotron.n_att_context_presets", n_presets)
    writer.add_array("nemotron.att_context_left", att_left)
    writer.add_array("nemotron.att_context_right", att_right)
    writer.add_uint32("nemotron.pred_hidden", pred_hidden)
    writer.add_uint32("nemotron.pred_layers", pred_layers)
    writer.add_uint32("nemotron.joint_hidden", joint_hidden)
    writer.add_uint32("nemotron.vocab_size", len(vocab))
    writer.add_uint32("nemotron.blank_id", len(vocab))  # blank is vocab_size
    writer.add_uint32("nemotron.n_tdt_durations", 0)     # pure RNNT
    writer.add_uint32("nemotron.frame_dur_cs", int(round(wst * subsampling_factor * 100)))
    writer.add_uint32("nemotron.num_prompts", num_prompts)
    writer.add_uint32("nemotron.prompt_kernel_in", prompt_kernel_in)
    writer.add_uint32("nemotron.prompt_kernel_mid", prompt_kernel_mid)

    # Prompt dictionary (language -> prompt_id mapping)
    prompt_dict = model_defaults.get("prompt_dictionary", {})
    if prompt_dict:
        lang_keys = sorted(prompt_dict.keys())
        lang_vals = [prompt_dict[k] for k in lang_keys]
        writer.add_array("nemotron.prompt_langs", lang_keys)
        writer.add_array("nemotron.prompt_ids", lang_vals)

    writer.add_array("tokenizer.ggml.tokens", vocab)

    # ----- inventory -----
    decoder_keys = sorted(k for k in sd.keys() if k.startswith("decoder.prediction."))
    joint_keys = sorted(k for k in sd.keys() if k.startswith("joint."))
    prompt_keys = sorted(k for k in sd.keys() if k.startswith("prompt_kernel"))
    print("  decoder.prediction.* tensors:")
    for k in decoder_keys:
        print(f"    {k}  shape={tuple(sd[k].shape)}")
    print("  joint.* tensors:")
    for k in joint_keys:
        print(f"    {k}  shape={tuple(sd[k].shape)}")
    print("  prompt_kernel.* tensors:")
    for k in prompt_keys:
        print(f"    {k}  shape={tuple(sd[k].shape)}")

    # ----- tensors -----
    n_written = 0
    n_f16 = 0
    n_f32 = 0
    n_quant = 0
    n_unmapped = 0
    for name in sorted(sd.keys()):
        gguf_name = remap_name(name)
        if gguf_name is None:
            if not name.endswith("num_batches_tracked") and "running_" not in name:
                n_unmapped += 1
            continue
        t = sd[name].cpu().numpy()
        if t.dtype == np.float64:
            t = t.astype(np.float32)

        raw_dtype = None
        if is_f32_tensor(gguf_name, t.shape):
            t = t.astype(np.float32)
            n_f32 += 1
        elif quant_type is not None:
            t = t.astype(np.float32)
            try:
                t = gguf.quantize(t, quant_type)
                raw_dtype = quant_type
                n_quant += 1
            except Exception:
                t = t.astype(np.float16)
                n_f16 += 1
        else:
            t = t.astype(np.float16)
            n_f16 += 1

        writer.add_tensor(gguf_name, t, raw_dtype=raw_dtype)
        n_written += 1
        if n_written <= 30 or n_written % 50 == 0:
            dtype_label = str(quant_type.name) if raw_dtype else str(t.dtype)
            print(f"  {gguf_name:60s}  {str(t.shape):28s}  {dtype_label}")

    # Inject zero dw.bias per encoder layer (nemotron uses LayerNorm in conv,
    # not BatchNorm, so there's no native dw.bias — but the shared build_block
    # expects one. The zero bias is a no-op add; the actual normalization comes
    # from the conv.ln.weight/bias tensors.)
    layers_seen = set()
    for name in sorted(sd.keys()):
        if name.startswith("encoder.layers.") and ".depthwise_conv.weight" in name:
            li = int(name.split(".")[2])
            layers_seen.add(li)
    for li in sorted(layers_seen):
        gguf_name = f"encoder.layers.{li}.conv.dw.bias"
        bias = np.zeros(d_model, dtype=np.float32)
        writer.add_tensor(gguf_name, bias)
        n_written += 1
        n_f32 += 1

    quant_label = f", {quant_type.name}: {n_quant}" if quant_type else ""
    print(
        f"\n  total tensors: {n_written}  (F16: {n_f16}, F32: {n_f32}{quant_label})"
        f"  (+{len(layers_seen)} synthetic conv.dw.bias)"
    )
    if n_unmapped:
        print(
            f"\n  WARNING: {n_unmapped} tensor(s) were unmapped — see "
            f"[WARN unmapped] lines above.",
            file=sys.stderr,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Nemotron .nemo → GGUF (F16 or quantized)")
    p.add_argument("--nemo", required=True, type=Path, help="path to .nemo file")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    p.add_argument("--quant", default=None, help="quantize linear weights (e.g. q4_k, q8_0); default: F16")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.nemo, args.output, quant=args.quant)
