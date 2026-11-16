#!/usr/bin/env python3
"""
Convert WeSpeaker ResNet34-LM (speaker embedding) → GGUF.

Source: https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM
        (`avg_model`, a plain torch state dict — 219 tensors, 11.25 M params)

⚠ LICENSE: the WeSpeaker *weights* are **CC-BY-4.0**, not Apache-2.0 as some
downstream projects state. Redistributing a converted GGUF therefore REQUIRES
attribution — the HF model card must carry `license: cc-by-4.0` plus credit to
`Wespeaker/wespeaker-voxceleb-resnet34-LM`, and the entry must be listed in
THIRD_PARTY_NOTICES.txt. (The wenet-e2e/wespeaker *code* is Apache-2.0; this
converter is a clean-room implementation from the architecture, not a
translation of it.)

Architecture — traced through `wespeaker/models/resnet.py` +
`wespeaker/cli/speaker.py`, which is the code that actually runs the model:

  features   torchaudio.compliance.kaldi.fbank(num_mel_bins=80,
             frame_length=25, frame_shift=10, window_type='hamming'),
             on an int16-SCALE waveform (`torchaudio.load(normalize=False)`,
             i.e. wavform_norm defaults to False), then per-utterance CMN:
             `feat = feat - feat.mean(dim=0)`.

  input      (T, 80) -> permute -> (80, T) -> unsqueeze(1) -> (1, 80, T)
             so the 2-D map is HEIGHT=freq(80), WIDTH=time(T).

  stem       Conv2d(1 -> 32, k=3, s=1, p=1, bias=False) -> BN -> ReLU
  layer1     3 x BasicBlock(32  -> 32,  stride 1)   (no shortcut projection)
  layer2     4 x BasicBlock(32  -> 64,  first stride 2 + 1x1 shortcut)
  layer3     6 x BasicBlock(64  -> 128, first stride 2 + 1x1 shortcut)
  layer4     3 x BasicBlock(128 -> 256, first stride 2 + 1x1 shortcut)

  BasicBlock relu(bn1(conv1(x))); bn2(conv2(.)); += shortcut(x); relu

  TSTP       over the LAST (time) axis of (B, 256, 10, T'):
               mean = x.mean(-1).flatten(1)            -> (B, 2560)
               std  = sqrt(var(x, -1) + 1e-7).flatten(1) -> (B, 2560)
             ⚠ torch.var defaults to the UNBIASED (n-1) estimator.
             stats = cat(mean, std) -> (B, 5120), ordering c*10 + f.

  seg_1      Linear(5120 -> 256)  ->  the embedding.
             two_emb_layer=False, so seg_bn_1 / seg_2 are Identity: there is
             NO ReLU, NO BatchNorm and NO L2 normalisation on the output.
             (FoxNose's clustering L2-normalises later, at its own layer.)

  projection ArcMargin classifier head (17982 x 256) — TRAINING ONLY.
             Dropped here, which takes 11.25 M params down to ~6.6 M.

BatchNorm folding: every BN is folded into the convolution that precedes it
(w *= gamma/sqrt(var+eps); b = beta - gamma*mean/sqrt(var+eps)). This is
mathematically exact up to float rounding and removes 100+ tensors plus a
per-graph normalisation from the runtime.

GGUF tensor naming:
  stem.conv.{weight,bias}                        F16/F32
  layers.{s}.{b}.conv1.{weight,bias}             F16/F32
  layers.{s}.{b}.conv2.{weight,bias}             F16/F32
  layers.{s}.{b}.shortcut.{weight,bias}          F16/F32  (only where present)
  seg1.{weight,bias}                             F16/F32

Usage:
  python models/convert-wespeaker-to-gguf.py \
      --model Wespeaker/wespeaker-voxceleb-resnet34-LM \
      --output wespeaker-resnet34-lm-f16.gguf
"""

from __future__ import annotations

import argparse
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

# nn.BatchNorm2d default.
BN_EPS = 1e-5

# ResNet34 stage depths and the channel width multiplier used by the
# published checkpoint (m_channels=32, feat_dim=80, embed_dim=256).
NUM_BLOCKS = [3, 4, 6, 3]
M_CHANNELS = 32
FEAT_DIM = 80
EMBED_DIM = 256


def resolve_source(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    if p.is_file():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(f"'{model}' is not a path and huggingface_hub is not installed")
    return Path(snapshot_download(model))


def load_state_dict(src: Path) -> dict:
    ckpt = src if src.is_file() else src / "avg_model"
    if not ckpt.exists():
        sys.exit(f"no checkpoint at {ckpt} (expected the WeSpeaker `avg_model`)")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def fold_bn(conv_w: torch.Tensor, bn_prefix: str, sd: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fold a BatchNorm2d into the conv that precedes it.

    y = gamma * (conv(x) - mean)/sqrt(var + eps) + beta
      = conv_folded(x) + b   with   scale = gamma/sqrt(var + eps)
    """
    gamma = sd[f"{bn_prefix}.weight"].float().numpy()
    beta = sd[f"{bn_prefix}.bias"].float().numpy()
    mean = sd[f"{bn_prefix}.running_mean"].float().numpy()
    var = sd[f"{bn_prefix}.running_var"].float().numpy()
    scale = gamma / np.sqrt(var + BN_EPS)
    w = conv_w.float().numpy() * scale[:, None, None, None]
    b = beta - mean * scale
    return w.astype(np.float32), b.astype(np.float32)


_QUANT_TYPE_MAP: dict[str, gguf.GGMLQuantizationType] = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}


def convert(src: Path, out_path: Path, quant: str | None = None, all_f32: bool = False) -> None:
    quant_type = _QUANT_TYPE_MAP.get(quant.lower()) if quant else None
    if quant and quant_type is None:
        sys.exit(f"Unknown --quant '{quant}'. Choices: {list(_QUANT_TYPE_MAP)}")

    sd = load_state_dict(src)
    print(f"Loaded {len(sd)} tensors ({sum(v.numel() for v in sd.values())/1e6:.2f} M params)")

    # Cross-check the checkpoint really is the 80-mel / 256-dim ResNet34 this
    # converter describes, rather than trusting the filename.
    seg1_w = sd.get("seg_1.weight")
    if seg1_w is None:
        sys.exit("no seg_1.weight — this is not a WeSpeaker ResNet embedding model")
    embed_dim, pool_out = int(seg1_w.shape[0]), int(seg1_w.shape[1])
    stats_dim = (FEAT_DIM // 8) * M_CHANNELS * 8
    if embed_dim != EMBED_DIM or pool_out != stats_dim * 2:
        sys.exit(f"unexpected seg_1 shape {tuple(seg1_w.shape)}; "
                 f"expected ({EMBED_DIM}, {stats_dim*2}) for feat_dim=80/emb=256")
    if tuple(sd["conv1.weight"].shape) != (M_CHANNELS, 1, 3, 3):
        sys.exit(f"unexpected conv1 shape {tuple(sd['conv1.weight'].shape)}")

    print(f"Writing: {out_path}")
    w = gguf.GGUFWriter(str(out_path), arch="wespeaker")

    w.add_string("wespeaker.variant", "resnet34-lm")
    w.add_uint32("wespeaker.sample_rate", 16000)
    w.add_uint32("wespeaker.n_mels", FEAT_DIM)
    w.add_uint32("wespeaker.frame_length_ms", 25)
    w.add_uint32("wespeaker.frame_shift_ms", 10)
    w.add_string("wespeaker.window_type", "hamming")
    # torchaudio.load(normalize=False) hands kaldi.fbank an int16-scale
    # waveform; our core_kaldi_fbank expresses that as int16_scale.
    w.add_bool("wespeaker.int16_scale", True)
    # `feat = feat - feat.mean(dim=0)` in compute_features().
    w.add_bool("wespeaker.cmn", True)
    w.add_uint32("wespeaker.m_channels", M_CHANNELS)
    w.add_array("wespeaker.num_blocks", NUM_BLOCKS)
    w.add_uint32("wespeaker.embed_dim", EMBED_DIM)
    w.add_uint32("wespeaker.stats_dim", stats_dim)
    w.add_float32("wespeaker.tstp_eps", 1e-7)
    w.add_bool("wespeaker.tstp_unbiased_var", True)

    n_f16 = n_f32 = n_quant = 0

    def emit(name: str, arr: np.ndarray, force_f32: bool = False) -> None:
        nonlocal n_f16, n_f32, n_quant
        raw = None
        # Conv kernels stay F32 as a SPEED choice, not a correctness one.
        # ggml used to assert (src1->type == GGML_TYPE_F32) here, but our fork's
        # 00285218 builds the conv_2d im2col patch in vec_dot_type, so an F16
        # kernel now runs and is numerically fine: cosine 0.99999724 against the
        # upstream oracle, and the GGUF drops 23.9 MB -> 13.3 MB.
        # It is just much slower. Measured back to back in one loop over the
        # same 352 windows of a 215 s file, M1, CPU backend:
        #     F32 kernels  133 ms/window
        #     F16 kernels  297 ms/window   (2.2x)
        # ResNet34 is ~94% of the embedder's time and the embedder is ~70% of a
        # diarization run, so paying 2.2x there to save 10 MB on disk is the
        # wrong trade. Re-measure before flipping this; if the CPU F16 conv path
        # ever gets a direct kernel the arithmetic could reverse.
        # The kernels are 4-D; the only 2-D weight is seg1.
        if force_f32 or all_f32 or arr.ndim <= 1 or arr.ndim == 4:
            arr = arr.astype(np.float32)
            n_f32 += 1
        elif quant_type is not None and arr.ndim == 2:
            try:
                arr = gguf.quantize(arr.astype(np.float32), quant_type)
                raw = quant_type
                n_quant += 1
            except Exception:
                arr = arr.astype(np.float16)
                n_f16 += 1
        else:
            arr = arr.astype(np.float16)
            n_f16 += 1
        w.add_tensor(name, arr, raw_dtype=raw)

    # ---- stem ----
    sw, sb = fold_bn(sd["conv1.weight"], "bn1", sd)
    emit("stem.conv.weight", sw)
    emit("stem.conv.bias", sb)

    # ---- residual stages ----
    n_shortcuts = 0
    for stage, n_blocks in enumerate(NUM_BLOCKS, start=1):
        for b in range(n_blocks):
            src_p = f"layer{stage}.{b}"
            dst_p = f"layers.{stage - 1}.{b}"
            for i in (1, 2):
                cw, cb = fold_bn(sd[f"{src_p}.conv{i}.weight"], f"{src_p}.bn{i}", sd)
                emit(f"{dst_p}.conv{i}.weight", cw)
                emit(f"{dst_p}.conv{i}.bias", cb)
            # The projection shortcut exists only where the block changes
            # shape (first block of stages 2-4): Conv2d(1x1) + BatchNorm2d,
            # stored as shortcut.0 / shortcut.1.
            if f"{src_p}.shortcut.0.weight" in sd:
                pw, pb = fold_bn(sd[f"{src_p}.shortcut.0.weight"], f"{src_p}.shortcut.1", sd)
                emit(f"{dst_p}.shortcut.weight", pw)
                emit(f"{dst_p}.shortcut.bias", pb)
                n_shortcuts += 1

    # ---- embedding head ----
    emit("seg1.weight", sd["seg_1.weight"].float().numpy())
    emit("seg1.bias", sd["seg_1.bias"].float().numpy())

    dropped = [k for k in sd if k.startswith("projection.")]
    total = n_f16 + n_f32 + n_quant
    qlabel = f", {quant_type.name}: {n_quant}" if quant_type else ""
    print(f"  {total} tensors (F16: {n_f16}, F32: {n_f32}{qlabel}), "
          f"{n_shortcuts} projection shortcuts, BN folded into every conv")
    print(f"  dropped (training-only): {dropped}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"\nDone: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert WeSpeaker ResNet34-LM → GGUF")
    p.add_argument("--model", required=True,
                   help="HF repo id, a snapshot dir, or a path to `avg_model`")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--quant", default=None, help="q8_0 (default: F16)")
    p.add_argument("--f32", action="store_true",
                   help="keep every tensor at F32 (reference-precision artifact; the whole "
                        "model is only ~26 MB, so this is a cheap parity baseline)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    convert(resolve_source(a.model), a.output, quant=a.quant, all_f32=a.f32)
