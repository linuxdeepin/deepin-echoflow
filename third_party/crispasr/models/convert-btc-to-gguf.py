#!/usr/bin/env python3
"""Convert BTC chord-recognition checkpoints (jayg996/BTC-ISMIR19) to GGUF.

    python models/convert-btc-to-gguf.py \
        --model btc_model.pt --output btc-chords-f32.gguf --dtype f32

Two variants ship upstream and differ ONLY in the classifier width:
  btc_model.pt             -> 25 chord classes  (maj/min vocabulary)
  btc_model_large_voca.pt  -> 170 chord classes (large vocabulary)

LICENCE. The upstream code is MIT, but these checkpoints were trained on
Isophonics / Robbie Williams / UsPop2002 annotations, which are CC BY-NC-SA.
The GGUF therefore carries `general.license = cc-by-nc-sa-4.0` so CrispASR's
acceptance gate refuses to download it without --accept-license. See
docs/music-transcription/PLAN.md.

Details that are NOT the obvious default (see BTC_BLUEPRINT.md):
  * `mean`/`std` are stored ALONGSIDE the weights and normalise the log-CQT.
  * Each layer has a FORWARD block (causal mask) and a BACKWARD block
    (transposed mask), concatenated then projected 256->128.
  * The FFN is Conv(k=3) -> ReLU -> Conv(k=3) with SYMMETRIC (1,1) padding.
  * output_layer.lstm.* is DEAD WEIGHT (never called) and is skipped.
  * Attention linears are bias-free; LayerNorm eps is 1e-6.
"""

import argparse
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("error: pip install gguf")
try:
    import torch
except ImportError:
    sys.exit("error: pip install torch")

ARCH = "btc"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="btc_model.pt or btc_model_large_voca.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    if "model" not in ck:
        sys.exit("error: checkpoint has no 'model' key — is this a BTC checkpoint?")
    sd = ck["model"]

    # Normalisation stats live next to the weights, not in the config.
    mean = float(ck["mean"])
    std = float(ck["std"])

    n_chords = int(sd["output_layer.output_projection.weight"].shape[0])
    hidden = int(sd["self_attn_layers.embedding_proj.weight"].shape[0])
    feat = int(sd["self_attn_layers.embedding_proj.weight"].shape[1])
    n_layers = 1 + max(
        int(k.split(".")[2])
        for k in sd
        if k.startswith("self_attn_layers.self_attn_layers.")
    )
    ffn = int(sd["self_attn_layers.self_attn_layers.0.attn_block.positionwise_convolution.layers.0.conv.weight"].shape[0])

    print(f"btc: {feat}->{hidden}, {n_layers} layers, ffn {ffn}, "
          f"{n_chords} chords, mean {mean:.6f} std {std:.6f}")

    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_uint32("btc.feature_size", feat)
    w.add_uint32("btc.hidden_size", hidden)
    w.add_uint32("btc.n_layers", n_layers)
    w.add_uint32("btc.n_heads", 4)          # run_config.yaml
    w.add_uint32("btc.filter_size", ffn)
    w.add_uint32("btc.n_chords", n_chords)
    w.add_uint32("btc.timestep", 108)       # training sequence length
    w.add_float32("btc.norm_mean", mean)
    w.add_float32("btc.norm_std", std)
    w.add_float32("btc.layer_norm_eps", 1e-6)
    # CQT front end — must match core/cqt.h / librosa.
    w.add_uint32("btc.cqt_n_bins", 144)
    w.add_uint32("btc.cqt_bins_per_octave", 24)
    w.add_uint32("btc.cqt_hop_length", 2048)
    w.add_uint32("btc.sample_rate", 22050)
    # Chunk length the reference splits audio into before the CQT (mp3.inst_len).
    # Also fixes the output frame rate: feature_per_second = inst_len / timestep,
    # which is NOT hop/sample_rate -- see docs/music-transcription/BTC_BLUEPRINT.md.
    w.add_float32("btc.inst_len_sec", 10.0)
    w.add_string("general.license", "cc-by-nc-sa-4.0")
    w.add_string(
        "general.license.description",
        "Weights trained on Isophonics/RobbieWilliams/UsPop2002 chord "
        "annotations (CC BY-NC-SA). NON-COMMERCIAL USE ONLY. Upstream code "
        "jayg996/BTC-ISMIR19 is MIT.",
    )

    emitted = []

    def emit(name: str, t: torch.Tensor, force_f32: bool = False) -> None:
        a = t.detach().cpu().float().numpy()
        if args.dtype == "f16" and not force_f32 and a.ndim >= 2:
            a = a.astype(np.float16)
        else:
            a = a.astype(np.float32)
        w.add_tensor(name, a)
        emitted.append(name)

    emit("embedding_proj.weight", sd["self_attn_layers.embedding_proj.weight"])

    seen = {"self_attn_layers.embedding_proj.weight"}
    pfx = "self_attn_layers.self_attn_layers."
    for i in range(n_layers):
        for src_blk, dst_blk in (("attn_block", "fwd"), ("backward_attn_block", "bwd")):
            b = f"{pfx}{i}.{src_blk}."
            for src, dst in (
                ("multi_head_attention.query_linear.weight", "attn.q.weight"),
                ("multi_head_attention.key_linear.weight", "attn.k.weight"),
                ("multi_head_attention.value_linear.weight", "attn.v.weight"),
                ("multi_head_attention.output_linear.weight", "attn.o.weight"),
                ("positionwise_convolution.layers.0.conv.weight", "ffn.0.weight"),
                ("positionwise_convolution.layers.0.conv.bias", "ffn.0.bias"),
                ("positionwise_convolution.layers.1.conv.weight", "ffn.1.weight"),
                ("positionwise_convolution.layers.1.conv.bias", "ffn.1.bias"),
                ("layer_norm_mha.gamma", "norm_mha.gamma"),
                ("layer_norm_mha.beta", "norm_mha.beta"),
                ("layer_norm_ffn.gamma", "norm_ffn.gamma"),
                ("layer_norm_ffn.beta", "norm_ffn.beta"),
            ):
                k = b + src
                # Norms and biases stay F32 — they are tiny and precision-
                # sensitive, matching the quantiser rules elsewhere in the repo.
                emit(f"layers.{i}.{dst_blk}.{dst}", sd[k],
                     force_f32=("norm" in dst or dst.endswith(".bias")))
                seen.add(k)
        for src, dst in (("linear.weight", "proj.weight"), ("linear.bias", "proj.bias")):
            k = f"{pfx}{i}.{src}"
            emit(f"layers.{i}.{dst}", sd[k], force_f32=dst.endswith("bias"))
            seen.add(k)

    # Final LayerNorm after the 8 layers, before the output LSTM. Not in the
    # architecture docs — the strict unconverted-tensor check below is what
    # surfaced it.
    for src, dst in (("self_attn_layers.layer_norm.gamma", "final_norm.gamma"),
                     ("self_attn_layers.layer_norm.beta", "final_norm.beta")):
        emit(dst, sd[src], force_f32=True)
        seen.add(src)

    # output_layer.lstm.* is DEAD WEIGHT: nn.LSTM is constructed in
    # OutputLayer.__init__ (transformer_modules.py:64) but SoftmaxOutputLayer
    # .forward never calls it — line 75 is just output_projection(hidden).
    # Skipped deliberately; converting it would invite someone to implement a
    # layer the reference never runs.
    for k in list(sd):
        if k.startswith("output_layer.lstm."):
            seen.add(k)

    for src, dst in (
        ("output_layer.output_projection.weight", "output.proj.weight"),
        ("output_layer.output_projection.bias", "output.proj.bias"),
    ):
        emit(dst, sd[src], force_f32=dst.endswith("bias"))
        seen.add(src)

    # Fail loudly rather than silently dropping a tensor — a missing weight
    # would show up much later as an unexplained parity failure.
    missing = [k for k in sd if k not in seen]
    if missing:
        sys.exit(f"error: {len(missing)} unconverted tensors, e.g. {missing[:5]}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: {len(emitted)} tensors emitted "
          f"({len(seen) - len(emitted)} skipped as unused), {n_chords} chords, dtype {args.dtype}")
    print("NOTE: weights are CC BY-NC-SA — non-commercial use only.")


if __name__ == "__main__":
    main()
