#!/usr/bin/env python3
"""Convert a Beat This! checkpoint to GGUF (§251b, beat/downbeat tracking).

Usage:
    python models/convert-beat-this-to-gguf.py --model final0.ckpt \
        --output beat-this-f16.gguf [--dtype f16]

Weights: CPJKU/beat_this, **MIT** — the upstream README states code AND published
weights are MIT. Checkpoint `final0` from the authors' cloud (see
`beat_this/inference.py` CHECKPOINT_URL). Paper: "Beat This! Accurate Beat
Tracking Without DBN Postprocessing" (ISMIR 2024).

Why this model matters here: it fills `RhythmGrid`, the one CometBeat seam type
with no model behind it, and it is **madmom-free** — its dependency list is
numpy/torch/torchaudio/einops/rotary-embedding-torch/soxr. The whole point of the
paper is avoiding DBN postprocessing, so no Böck-patented step enters the chain
even by distillation. That is a hard requirement downstream.

ARCHITECTURE (traced from beat_this/model/beat_tracker.py, not from the ONNX
export — that export has anonymous initializers like `onnx::MatMul_5756` plus
1760 Constants of dynamic-shape plumbing, so it is the wrong porting basis):

  frontend.stem   : BN1d(128) -> Conv2d(1->32, k=(4,3), s=(4,1), p=(0,1), NO bias)
                    -> BN2d(32) -> GELU          [freq 128 -> 32]
  frontend.blocks : 3x [ PartialFTTransformer(dim, head_dim=32)
                         -> Conv2d(d->2d, k=(2,3), s=(2,1), p=(0,1), NO bias)
                         -> BN2d -> GELU ]
                    dims 32->64->128->256, freq 32->16->8->4
  frontend.concat : (b c f t) -> (b t (c f))     [256*4 = 1024]
  frontend.linear : Linear(1024 -> 512)
  transformer     : 6x roformer layers, dim 512, 16 heads (512/32), ff_mult 4,
                    shared RotaryEmbedding(32), norm_output
  task_heads      : SumHead -> Linear(512 -> 2)  [beat, downbeat]

BATCHNORM: the stem is `conv2d(bias=False) -> bn2d -> GELU`, i.e. BN comes
IMMEDIATELY after the conv with no activation between, so it folds exactly:
    s  = gamma / sqrt(var + eps);  w' = w * s;  b' = beta - mean * s
Folded in float64. NOTE this is the opposite of CREPE, where a ReLU sits between
conv and BN and folding is INVALID (see docs/music-transcription/PLAN.md). Do not
generalise either way — read the module order every time.

`frontend.stem.bn1d` is NOT foldable: it is a per-FREQUENCY affine applied to the
input spectrogram before any conv, so it ships as a standalone scale/offset pair.

I/O CONTRACT (from the musetric export's config.json, which records it exactly):
  22050 Hz mono, downmix = ARITHMETIC MEAN (L+R)/2 -- NOT ffmpeg -ac 1, which
  rematrixes to (L+R)/sqrt(2); because features are log1p(1000*mel) that gain is
  non-constant and it moves beats.
  STFT n_fft 1024, hop 441, periodic Hann, normalized='frame_length';
  128 mel bins; log1p(1000 * x); 50 fps.
  Inference: 1500-frame windows, borderSize 6, overlap 'keep_first'.
  Peak-pick: maxima within +/-3 frames (kernel 7), threshold 0, dedupe <=1 frame,
  then snap each downbeat to its nearest beat.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

BN_EPS = 1e-5  # torch nn.BatchNorm default


def fold_bn_into_conv(w, gamma, beta, mean, var, eps=BN_EPS):
    """conv(bias=False) -> BN  ==>  conv(w', b'). All math in float64.

    Valid ONLY because nothing sits between the conv and the BN here.
    """
    w = w.astype(np.float64)
    s = gamma.astype(np.float64) / np.sqrt(var.astype(np.float64) + eps)
    # w is (out, in, kf, kt); scale is per-output-channel.
    return w * s[:, None, None, None], beta.astype(np.float64) - mean.astype(np.float64) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="final0.ckpt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--filterbank", type=Path,
                    help="optional mel-filterbank.bin [513,128] f32 from the musetric "
                         "export; baked in so the runtime never re-derives it")
    args = ap.parse_args()

    import torch
    from gguf import GGUFWriter, GGMLQuantizationType

    if not args.model.exists():
        sys.exit(f"error: no checkpoint at {args.model}")
    ck = torch.load(str(args.model), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    sd = {k: v.detach().cpu().numpy() for k, v in sd.items()
          if hasattr(v, "detach") and not k.endswith("num_batches_tracked")}

    n_params = sum(v.size for v in sd.values())
    print(f"beat-this: {len(sd)} tensors, {n_params/1e6:.2f} M params")

    w = GGUFWriter(str(args.output), "beat-this", use_temp_file=True)
    w.add_name("Beat This! (final0)")
    # Geometry the runtime must not re-derive.
    for k, v in (("sample_rate", 22050), ("n_fft", 1024), ("hop_length", 441),
                 ("mel_bins", 128), ("transformer_dim", 512), ("n_layers", 6),
                 ("head_dim", 32), ("n_heads", 16), ("stem_dim", 32),
                 ("ff_mult", 4), ("chunk_size", 1500), ("border_size", 6),
                 ("peak_kernel", 7), ("dedup_width", 1)):
        w.add_uint32(f"beat-this.{k}", v)
    w.add_float32("beat-this.fps", 50.0)
    w.add_float32("beat-this.log_multiplier", 1000.0)
    w.add_float32("beat-this.peak_threshold", 0.0)
    w.add_string("beat-this.overlap_mode", "keep_first")
    w.add_string("beat-this.downmix", "mean")

    f32 = GGMLQuantizationType.F32
    f16 = GGMLQuantizationType.F16
    wdt = f16 if args.dtype == "f16" else f32
    nwdt = np.float16 if args.dtype == "f16" else np.float32

    if args.filterbank:
        fb = np.fromfile(str(args.filterbank), dtype=np.float32)
        if fb.size != 513 * 128:
            sys.exit(f"error: filterbank has {fb.size} floats, expected {513*128}")
        # Ship verbatim: re-deriving a mel filterbank is a classic source of
        # silent drift (slaney vs htk, layout). The export ships the exact one.
        w.add_tensor("aux.mel_filterbank", fb.reshape(513, 128), raw_dtype=f32)
        print("  baked mel filterbank [513,128]")

    # --- fold the four conv+BN pairs -------------------------------------
    folded = set()
    # NOTE the BN module is named differently in the two places: the stem calls
    # it `bn2d`, the frontend blocks call it `norm`. Verified against the actual
    # checkpoint keys, not assumed from the module source -- make_frontend_block
    # constructs it inline, so the attribute name is not obvious from reading it.
    conv_bn = [("frontend.stem.conv2d", "frontend.stem.bn2d")]
    for i in range(3):
        conv_bn.append((f"frontend.blocks.{i}.conv2d", f"frontend.blocks.{i}.norm"))

    n_w = 0
    for conv, bn in conv_bn:
        wk, bnk = f"{conv}.weight", f"{bn}."
        if wk not in sd or f"{bnk}weight" not in sd:
            sys.exit(f"error: expected {wk} and {bnk}* in the checkpoint — "
                     f"architecture changed, re-read beat_tracker.py")
        wf, bf = fold_bn_into_conv(sd[wk], sd[f"{bnk}weight"], sd[f"{bnk}bias"],
                                   sd[f"{bnk}running_mean"], sd[f"{bnk}running_var"])
        w.add_tensor(f"{conv}.weight", np.ascontiguousarray(wf.astype(nwdt)), raw_dtype=wdt)
        w.add_tensor(f"{conv}.bias", bf.astype(np.float32), raw_dtype=f32)
        n_w += 2
        folded.update({wk, f"{bnk}weight", f"{bnk}bias",
                       f"{bnk}running_mean", f"{bnk}running_var"})

    # --- stem bn1d: NOT foldable (per-frequency, pre-conv) ---------------
    g = sd["frontend.stem.bn1d.weight"].astype(np.float64)
    b = sd["frontend.stem.bn1d.bias"].astype(np.float64)
    m = sd["frontend.stem.bn1d.running_mean"].astype(np.float64)
    v = sd["frontend.stem.bn1d.running_var"].astype(np.float64)
    s = g / np.sqrt(v + BN_EPS)
    w.add_tensor("frontend.stem.bn1d.scale", s.astype(np.float32), raw_dtype=f32)
    w.add_tensor("frontend.stem.bn1d.offset", (b - m * s).astype(np.float32), raw_dtype=f32)
    n_w += 2
    folded.update({f"frontend.stem.bn1d.{x}" for x in
                   ("weight", "bias", "running_mean", "running_var")})

    # --- everything else verbatim ----------------------------------------
    for name, arr in sd.items():
        if name in folded:
            continue
        keep_f32 = arr.ndim < 2 or name.endswith(("gamma", "beta", "bias"))
        if args.dtype == "f16" and not keep_f32:
            w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float16)), raw_dtype=f16)
        else:
            w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)), raw_dtype=f32)
        n_w += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output} ({args.output.stat().st_size/1e6:.1f} MB, {n_w} tensors)")


if __name__ == "__main__":
    main()
