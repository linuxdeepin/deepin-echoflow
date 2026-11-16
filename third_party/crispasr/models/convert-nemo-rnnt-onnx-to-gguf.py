#!/usr/bin/env python3
"""Convert a NeMo FastConformer-RNNT **ONNX export** to a parakeet-family GGUF.

Written for #387 (hojreh/Quds-v4-onnx, Persian, fine-tune of
nvidia/stt_fa_fastconformer_hybrid_large) but generic over the standard
NeMo/onnx_asr export pair:

    encoder-model.onnx          mel [B, n_mels, T] -> encoder states
    decoder_joint-model.onnx    fused prediction-net (LSTM) + joint
    tokens.txt                  "<piece> <id>" per line, <blk> last

Why ONNX and not .nemo: some authors only publish the export (#387). The
torch module names survive in most initializers; the ones the exporter
anonymised (onnx::MatMul_*, onnx::Conv_*, onnx::LSTM_*) are recovered by
tracing each initializer to its consumer node, whose name carries the full
module scope (e.g. /layers.16/self_attn/linear_pos/MatMul). Two exporter
transforms are undone on the way:
  * MatMul B-matrices are stored [in, out] — transposed back to Linear's
    [out, in];
  * ONNX LSTM tensors are W/R/B with gate order i,o,f,c — reordered to
    torch's i,f,g,o and split back into w_ih/w_hh/b_ih/b_hh.
BatchNorm is pre-folded into the exported depthwise convs, so identity BN
stats are emitted for the runtime's fold to no-op over.

The mel filterbank + Hann window are NOT in the export (it takes mel
features). They are copied verbatim from the BASE model's .nemo checkpoint
buffers (--base-nemo, default: auto-download the CC-BY-4.0 NVIDIA base) —
never recomputed, per the copy-the-shipped-filterbank rule.

Usage:
  python models/convert-nemo-rnnt-onnx-to-gguf.py \
      --onnx-dir /path/with/onnx/files \
      --base-nemo nvidia/stt_fa_fastconformer_hybrid_large \
      --output quds-v4-fa-f16.gguf --dtype f16
"""

import argparse
import os
import re
import sys
import tarfile
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("pip install onnx")

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# ONNX loading: resolve every initializer to a stable, module-scoped name.
# ---------------------------------------------------------------------------
def scoped_initializers(model):
    """name -> np.array, with onnx::* names replaced by consumer-node scope."""
    init = {t.name: t for t in model.graph.initializer}
    consumer = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp in init and inp not in consumer:
                consumer[inp] = node
    out = {}
    for name, tensor in init.items():
        arr = numpy_helper.to_array(tensor)
        if name.startswith("onnx::"):
            node = consumer.get(name)
            if node is None:
                sys.exit(f"anonymous initializer {name} has no consumer — unmappable export")
            # '/layers.0/conv/depthwise_conv/Conv' -> 'layers.0.conv.depthwise_conv'
            scope = node.name.strip("/").split("/")
            scope = [s for s in scope if s not in ("MatMul", "Conv", "LSTM")]
            key = ".".join(scope)
            kind = name.split("_")[0].split("::")[1]  # MatMul / Conv / LSTM
            # LSTM has three tensors on one node: keep W/R/B identity by rank+order
            out.setdefault((key, kind), []).append((name, arr))
        else:
            out[(name, "named")] = [(name, arr)]
    return out


def flatten_scoped(scoped):
    """(scope, kind) -> single array for MatMul/Conv, list for LSTM."""
    flat = {}
    for (key, kind), items in scoped.items():
        if kind == "named":
            flat[key] = items[0][1]
        elif kind == "LSTM":
            # sort by the numeric suffix of the original onnx:: name so the
            # W (iofc x input), R (iofc x hidden), B ([Wb;Rb]) order the
            # exporter emitted is preserved
            items = sorted(items, key=lambda kv: int(kv[0].rsplit("_", 1)[1]))
            flat[key + "#LSTM"] = [a for _, a in items]
        elif kind == "Conv":
            # a Conv scope can carry weight AND bias (folded-BN exports):
            # disambiguate by rank — weight is 3-D [C,1,K], bias is 1-D.
            for _, arr in items:
                suffix = "#Conv" if arr.ndim > 1 else "#ConvBias"
                if key + suffix in flat:
                    sys.exit(f"scope {key}{suffix}: duplicate")
                flat[key + suffix] = arr
        else:
            if len(items) != 1:
                sys.exit(f"scope {key} ({kind}) has {len(items)} initializers — ambiguous")
            flat[key + "#MatMul"] = items[0][1]
    return flat


def lstm_unpack(tensors, hidden):
    """ONNX LSTM (W[1,4H,I], R[1,4H,H], B[1,8H]) -> torch w_ih/w_hh/b_ih/b_hh.

    ONNX gate order is i,o,f,c; torch is i,f,g(=c),o.
    """
    W, R, B = (t[0] for t in tensors)  # drop the num_directions axis

    def reorder(m):
        i, o, f, c = np.split(m, 4, axis=0)
        return np.concatenate([i, f, c, o], axis=0)

    w_ih = reorder(W)
    w_hh = reorder(R)
    b = B.reshape(2, 4 * hidden)
    b_ih = reorder(b[0].reshape(4 * hidden, 1)).reshape(-1)
    b_hh = reorder(b[1].reshape(4 * hidden, 1)).reshape(-1)
    return w_ih, w_hh, b_ih, b_hh


# ---------------------------------------------------------------------------
# Base .nemo: mel filterbank + window buffers (copied, never recomputed).
# ---------------------------------------------------------------------------
def load_base_preproc(base, tmpdir):
    path = base
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download

        # NVIDIA NeMo repos ship "<name>.nemo" at the top level.
        repo = base
        fname = repo.rsplit("/", 1)[-1] + ".nemo"
        log(f"  downloading {repo}/{fname} …")
        path = hf_hub_download(repo, fname, local_dir=tmpdir)
    import torch

    with tarfile.open(path) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt"))
        f = tar.extractfile(member)
        sd = torch.load(f, map_location="cpu", weights_only=True)
    fb = sd.get("preprocessor.featurizer.fb")
    win = sd.get("preprocessor.featurizer.window")
    if fb is None or win is None:
        sys.exit("base checkpoint lacks preprocessor.featurizer.{fb,window} buffers")
    return fb.float().numpy(), win.float().numpy()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--base-nemo", default="nvidia/stt_fa_fastconformer_hybrid_large")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    ap.add_argument("--arch-name", default=None, help="model name string for gguf metadata")
    args = ap.parse_args()

    d = args.onnx_dir
    log("loading encoder-model.onnx …")
    enc = onnx.load(os.path.join(d, "encoder-model.onnx"))
    log("loading decoder_joint-model.onnx …")
    dj = onnx.load(os.path.join(d, "decoder_joint-model.onnx"))
    E = flatten_scoped(scoped_initializers(enc))
    D = flatten_scoped(scoped_initializers(dj))

    # ---- hparams from shapes -------------------------------------------------
    n_layers = 1 + max(int(m.group(1)) for k in E for m in [re.match(r"layers\.(\d+)\.", k)] if m)
    d_model = E["layers.0.norm_out.weight"].shape[0]
    pbu = E["layers.0.self_attn.pos_bias_u"]
    n_heads, head_dim = pbu.shape
    ff_dim = E["layers.0.feed_forward1.linear1.bias"].shape[0]
    conv_kernel = E["layers.0.conv.depthwise_conv#Conv"].shape[-1]
    sub_ch = E["pre_encode.conv.0.weight"].shape[0]
    pre_out_w = E["pre_encode.out#MatMul"]  # [flat_in, d_model]
    n_mels = 8 * pre_out_w.shape[0] // sub_ch // 8  # flat_in = sub_ch * ceil(n_mels/8)
    # exact: flat_in / sub_ch = n_mels/8 for divisible mels
    n_mels = pre_out_w.shape[0] * 8 // sub_ch

    embed = D["decoder.prediction.embed.weight"]
    vocab_p1, pred_hidden = embed.shape
    joint_out_w = D["joint.joint_net.joint_net.2#MatMul"]  # [joint_hidden, vocab+1]
    joint_hidden = joint_out_w.shape[0]
    vocab_size = joint_out_w.shape[1] - 1  # blank last

    toks = []
    with open(os.path.join(d, "tokens.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, idx = line.rsplit(" ", 1)
            toks.append((int(idx), piece))
    toks.sort()
    tokens = [p for _, p in toks]
    if tokens and tokens[-1] == "<blk>":
        tokens = tokens[:-1]  # runtime appends blank at vocab_size
    assert len(tokens) == vocab_size, (len(tokens), vocab_size)

    log(
        f"  hparams: layers={n_layers} d_model={d_model} heads={n_heads}x{head_dim} "
        f"ff={ff_dim} k={conv_kernel} mels={n_mels} pred_hidden={pred_hidden} "
        f"joint={joint_hidden} vocab={vocab_size}"
    )

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", None)) as tmpdir:
        fb, window = load_base_preproc(args.base_nemo, tmpdir)
        log(f"  preprocessor: fb {fb.shape} window {window.shape} (copied from base)")

    # ---- write GGUF ----------------------------------------------------------
    ftype = np.float16 if args.dtype == "f16" else np.float32
    writer = GGUFWriter(args.output, "parakeet")
    name = args.arch_name or os.path.basename(args.output).rsplit(".", 1)[0]
    writer.add_name(name)

    sr, n_fft, win_len, hop = 16000, 512, len(window), 160
    writer.add_uint32("parakeet.sample_rate", sr)
    writer.add_uint32("parakeet.n_mels", n_mels)
    writer.add_uint32("parakeet.n_fft", n_fft)
    writer.add_uint32("parakeet.win_length", win_len)
    writer.add_uint32("parakeet.hop_length", hop)
    writer.add_uint32("parakeet.d_model", d_model)
    writer.add_uint32("parakeet.n_layers", n_layers)
    writer.add_uint32("parakeet.n_heads", n_heads)
    writer.add_uint32("parakeet.head_dim", head_dim)
    writer.add_uint32("parakeet.ff_dim", ff_dim)
    writer.add_uint32("parakeet.subsampling_factor", 8)
    writer.add_uint32("parakeet.subsampling_channels", sub_ch)
    writer.add_uint32("parakeet.conv_kernel", conv_kernel)
    writer.add_uint32("parakeet.pred_hidden", pred_hidden)
    writer.add_uint32("parakeet.pred_layers", 1)
    writer.add_uint32("parakeet.joint_hidden", joint_hidden)
    writer.add_uint32("parakeet.vocab_size", vocab_size)
    writer.add_uint32("parakeet.blank_id", vocab_size)
    writer.add_uint32("parakeet.n_tdt_durations", 0)
    writer.add_array("parakeet.tdt_durations", [])
    writer.add_uint32("parakeet.frame_dur_cs", 8)
    writer.add_bool("parakeet.has_ctc", False)
    writer.add_array("tokenizer.ggml.tokens", tokens)

    tensors = {}

    def put(name_, arr, keep_f32=False):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if not keep_f32 and args.dtype == "f16" and a.ndim >= 2 and a.size >= 256:
            a = a.astype(ftype)
        tensors[name_] = a

    put("preprocessor.fb", fb, keep_f32=True)
    put("preprocessor.window", window, keep_f32=True)

    # pre-encoder subsampling stack
    for i in (0, 2, 3, 5, 6):
        put(f"encoder.pre.conv.{i}.weight", E[f"pre_encode.conv.{i}.weight"])
        put(f"encoder.pre.conv.{i}.bias", E[f"pre_encode.conv.{i}.bias"], keep_f32=True)
    put("encoder.pre.out.weight", pre_out_w.T)  # MatMul [in,out] -> Linear [out,in]
    put("encoder.pre.out.bias", E["pre_encode.out.bias"], keep_f32=True)

    def mm(key):
        return E[key + "#MatMul"].T

    for i in range(n_layers):
        L = f"layers.{i}"
        G = f"encoder.layers.{i}"
        for ff in ("feed_forward1", "feed_forward2"):
            gg = "ff1" if ff.endswith("1") else "ff2"
            put(f"{G}.{gg}.linear1.weight", mm(f"{L}.{ff}.linear1"))
            put(f"{G}.{gg}.linear1.bias", E[f"{L}.{ff}.linear1.bias"], keep_f32=True)
            put(f"{G}.{gg}.linear2.weight", mm(f"{L}.{ff}.linear2"))
            put(f"{G}.{gg}.linear2.bias", E[f"{L}.{ff}.linear2.bias"], keep_f32=True)
        put(f"{G}.norm_ff1.weight", E[f"{L}.norm_feed_forward1.weight"], keep_f32=True)
        put(f"{G}.norm_ff1.bias", E[f"{L}.norm_feed_forward1.bias"], keep_f32=True)
        put(f"{G}.norm_ff2.weight", E[f"{L}.norm_feed_forward2.weight"], keep_f32=True)
        put(f"{G}.norm_ff2.bias", E[f"{L}.norm_feed_forward2.bias"], keep_f32=True)
        put(f"{G}.norm_attn.weight", E[f"{L}.norm_self_att.weight"], keep_f32=True)
        put(f"{G}.norm_attn.bias", E[f"{L}.norm_self_att.bias"], keep_f32=True)
        put(f"{G}.norm_conv.weight", E[f"{L}.norm_conv.weight"], keep_f32=True)
        put(f"{G}.norm_conv.bias", E[f"{L}.norm_conv.bias"], keep_f32=True)
        put(f"{G}.norm_out.weight", E[f"{L}.norm_out.weight"], keep_f32=True)
        put(f"{G}.norm_out.bias", E[f"{L}.norm_out.bias"], keep_f32=True)

        for qkv in ("q", "k", "v", "out", "pos"):
            src = {"pos": "linear_pos"}.get(qkv, f"linear_{qkv}")
            put(f"{G}.attn.{qkv}.weight", mm(f"{L}.self_attn.{src}"))
        for qkv in ("q", "k", "v", "out"):
            put(f"{G}.attn.{qkv}.bias", E[f"{L}.self_attn.linear_{qkv}.bias"], keep_f32=True)
        put(f"{G}.attn.pos_bias_u", E[f"{L}.self_attn.pos_bias_u"], keep_f32=True)
        put(f"{G}.attn.pos_bias_v", E[f"{L}.self_attn.pos_bias_v"], keep_f32=True)

        put(f"{G}.conv.pw1.weight", E[f"{L}.conv.pointwise_conv1.weight"])
        put(f"{G}.conv.pw1.bias", E[f"{L}.conv.pointwise_conv1.bias"], keep_f32=True)
        put(f"{G}.conv.dw.weight", E[f"{L}.conv.depthwise_conv#Conv"])
        put(f"{G}.conv.dw.bias", E.get(f"{L}.conv.depthwise_conv#ConvBias", np.zeros(d_model, np.float32)),
            keep_f32=True)
        put(f"{G}.conv.pw2.weight", E[f"{L}.conv.pointwise_conv2.weight"])
        put(f"{G}.conv.pw2.bias", E[f"{L}.conv.pointwise_conv2.bias"], keep_f32=True)
        # BN was folded into the exported dw conv: identity stats.
        put(f"{G}.conv.bn.weight", np.ones(d_model, np.float32), keep_f32=True)
        put(f"{G}.conv.bn.bias", np.zeros(d_model, np.float32), keep_f32=True)
        put(f"{G}.conv.bn.running_mean", np.zeros(d_model, np.float32), keep_f32=True)
        put(f"{G}.conv.bn.running_var", np.ones(d_model, np.float32), keep_f32=True)

    # decoder (single-layer LSTM) + joint
    put("decoder.embed.weight", embed)
    w_ih, w_hh, b_ih, b_hh = lstm_unpack(D["decoder.dec_rnn.lstm#LSTM"], pred_hidden)
    put("decoder.lstm.0.w_ih", w_ih)
    put("decoder.lstm.0.w_hh", w_hh)
    put("decoder.lstm.0.b_ih", b_ih, keep_f32=True)
    put("decoder.lstm.0.b_hh", b_hh, keep_f32=True)

    put("joint.enc.weight", D["joint.enc#MatMul"].T)
    put("joint.enc.bias", D["joint.enc.bias"], keep_f32=True)
    put("joint.pred.weight", D["joint.pred#MatMul"].T)
    put("joint.pred.bias", D["joint.pred.bias"], keep_f32=True)
    put("joint.out.weight", D["joint.joint_net.joint_net.2#MatMul"].T)
    put("joint.out.bias", D["joint.joint_net.2.bias"], keep_f32=True)

    for name_, arr in tensors.items():
        writer.add_tensor(name_, arr)
    log(f"  writing {len(tensors)} tensors → {args.output}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log("done")


if __name__ == "__main__":
    main()
