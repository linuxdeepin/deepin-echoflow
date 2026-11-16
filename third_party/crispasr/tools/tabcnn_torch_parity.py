#!/usr/bin/env python3
"""TabCNN numpy spec vs the torch reference — the acceptance gate for src/tabcnn.

This file is the SPEC the C++ mirrors. Every stage is implemented here in plain
numpy from the geometry, then checked against the real `amt_tools` model, so a
disagreement localises to one stage instead of showing up as "the tab is wrong".

Run:

    pip install amt-tools librosa            # MIT / ISC
    python tools/tabcnn_torch_parity.py \\
        --model /path/to/best_TabCNN_tablature_trancription_model \\
        --audio samples/jfk.wav

    # negative control — perturb one stage and confirm the gate FAILS
    python tools/tabcnn_torch_parity.py --model ... --audio ... --break-stage conv1

WHY THE MAGNITUDE RATIO IS ASSERTED, NOT JUST COSINE
----------------------------------------------------
Cosine, correlation and peak-bin agreement all divide magnitude out, so a stage
wrong by a uniform factor passes every one of them. This repo has been bitten
three times: the htdemucs iSTFT scale inversion (cos stayed 1.000000), the
`core/cqt.h` missing `scale=True` (correlation 0.9999 while every bin was low by
up to 152x), and — while writing THIS backend's reference dumper — omitting
amt_tools' `/80 + 1` rescale, which produced cos = -0.544 and a magnitude ratio
of 0.0047. The magnitudes are what caught it: |mine| 15895 vs |ref| 88.

So every stage asserts BOTH cos >= COS_TOL AND |ratio - 1| <= RATIO_TOL, and the
report prints |spec| and |ref| side by side.

THE FRONT END IS PART OF THE SPEC
---------------------------------
`model.frontend` is an empty Sequential — the CQT lives outside the network — so
a harness that starts at features never tests it. That is exactly how a
front-end mismatch survived on BTC (86.63% -> 98.56% mir_eval once fixed). This
tool therefore starts from the WAVEFORM.

⚠️ `amplitude_to_db(ref=np.max)` is a PER-CLIP normalisation: the dB reference is
the maximum of the whole clip's CQT. The features cannot be computed chunked
without changing them, so the C++ must do a two-pass (CQT the clip, take the
max, then normalise) rather than streaming. Chunking here would reproduce the
BTC chunked-CQT bug.
"""

import argparse
import sys

import numpy as np

COS_TOL = 0.9999
RATIO_TOL = 0.01          # |median(|ref|/|spec|) - 1|
SR = 22050
HOP = 512
N_BINS = 192
BINS_PER_OCTAVE = 24
FRAME_WIDTH = 9
DB_FLOOR = 80.0


# ── the spec: plain numpy, mirroring what src/tabcnn.cpp must do ────────────

def spec_cqt(audio, sample_rate):
    """Waveform -> [n_bins, T] in [0, 1]. Two-pass by construction."""
    import librosa
    if sample_rate != SR:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=SR)
    fmin = librosa.note_to_hz("C1")
    spec = np.abs(librosa.vqt(y=audio, sr=SR, hop_length=HOP, fmin=fmin,
                              n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                              gamma=0))
    # amplitude_to_db(ref=np.max) with librosa's default top_db=80, written out
    # so the C++ can follow it literally.
    ref = spec.max()
    log_spec = 20.0 * np.log10(np.maximum(spec, 1e-10) / max(ref, 1e-10))
    log_spec = np.maximum(log_spec, -DB_FLOOR)      # top_db clamp
    return (log_spec / DB_FLOOR + 1.0).astype(np.float32)


def spec_windows(cqt, frame_width=FRAME_WIDTH):
    """[bins, T] -> [T, 1, bins, frame_width], zero-padded both edges."""
    pad = frame_width // 2
    padded = np.pad(cqt, ((0, 0), (pad, pad)), mode="constant")
    T = padded.shape[1] - 2 * pad
    idx = np.arange(frame_width)[None, :] + np.arange(T)[:, None]
    return padded[:, idx].transpose(1, 0, 2)[:, None, :, :].astype(np.float32)


def spec_conv2d(x, w, b):
    """Valid 3x3 conv, stride 1. x [N,Ci,H,W], w [Co,Ci,kh,kw] -> [N,Co,H-2,W-2]."""
    N, Ci, H, W = x.shape
    Co, _, kh, kw = w.shape
    oh, ow = H - kh + 1, W - kw + 1
    s = x.strides
    patches = np.lib.stride_tricks.as_strided(
        x, shape=(N, Ci, oh, ow, kh, kw),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    out = np.einsum("nchwij,ocij->nohw", patches, w, optimize=True)
    return out + b[None, :, None, None]


def spec_maxpool2x2(x):
    N, C, H, W = x.shape
    H2, W2 = H // 2, W // 2
    return x[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2).max(axis=(3, 5))


def spec_forward(cqt, weights):
    """Full forward from the CQT. Returns {stage: array}."""
    out = {}
    x = spec_windows(cqt)
    out["conv0"] = spec_conv2d(x, weights["conv0.weight"], weights["conv0.bias"])
    out["conv0_relu"] = np.maximum(out["conv0"], 0)
    out["conv1"] = spec_conv2d(out["conv0_relu"], weights["conv1.weight"], weights["conv1.bias"])
    out["conv1_relu"] = np.maximum(out["conv1"], 0)
    out["conv2"] = spec_conv2d(out["conv1_relu"], weights["conv2.weight"], weights["conv2.bias"])
    out["conv2_relu"] = np.maximum(out["conv2"], 0)
    out["pool"] = spec_maxpool2x2(out["conv2_relu"])
    # torch flattens NCHW in C-order; the C++ must use the same axis order or the
    # 5952-wide dense layer silently reads a permuted vector.
    flat = out["pool"].reshape(out["pool"].shape[0], -1)
    out["dense0"] = flat @ weights["dense0.weight"].T + weights["dense0.bias"]
    out["dense0_relu"] = np.maximum(out["dense0"], 0)
    out["logits"] = out["dense0_relu"] @ weights["head.weight"].T + weights["head.bias"]
    return out


def log_softmax_groups(logits, n_groups=6, n_classes=21):
    """[T, 126] -> [T, 6, 21] log-probabilities (what the C++ ABI emits)."""
    z = logits.reshape(-1, n_groups, n_classes)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


# ── comparison ─────────────────────────────────────────────────────────────

def compare(name, spec, ref):
    a = np.asarray(spec, dtype=np.float64).reshape(-1)
    b = np.asarray(ref, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return {"stage": name, "ok": False, "why": f"shape {a.shape} vs {b.shape}"}
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0
    m = np.abs(a) > 1e-8
    ratio = float(np.median(np.abs(b[m] / a[m]))) if m.any() else float("nan")
    ok = cos >= COS_TOL and abs(ratio - 1.0) <= RATIO_TOL
    return {"stage": name, "ok": ok, "cos": cos, "ratio": ratio,
            "spec_norm": float(na), "ref_norm": float(nb),
            "max_abs": float(np.abs(a - b).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="truncate audio (numpy conv is O(T); 4 s is plenty)")
    ap.add_argument("--break-stage", default=None,
                    help="negative control: corrupt this stage's weights and "
                         "confirm the gate fails")
    args = ap.parse_args()

    import wave
    import torch
    try:
        import amt_tools  # noqa: F401
    except ImportError:
        sys.exit("need amt-tools to unpickle the checkpoint: pip install amt-tools")

    with wave.open(args.audio, "rb") as wv:
        sr = wv.getframerate()
        audio = np.frombuffer(wv.readframes(wv.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    audio = audio[: int(args.seconds * sr)]

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model.eval()
    model.device = "cpu"
    sd = model.state_dict()
    weights = {
        "conv0.weight": sd["conv.0.weight"].numpy(), "conv0.bias": sd["conv.0.bias"].numpy(),
        "conv1.weight": sd["conv.2.weight"].numpy(), "conv1.bias": sd["conv.2.bias"].numpy(),
        "conv2.weight": sd["conv.4.weight"].numpy(), "conv2.bias": sd["conv.4.bias"].numpy(),
        "dense0.weight": sd["dense.0.weight"].numpy(), "dense0.bias": sd["dense.0.bias"].numpy(),
        "head.weight": sd["dense.3.output_layer.weight"].numpy(),
        "head.bias": sd["dense.3.output_layer.bias"].numpy(),
    }
    if args.break_stage:
        key = f"{args.break_stage}.weight"
        if key not in weights:
            sys.exit(f"--break-stage must be one of "
                     f"{sorted(k.split('.')[0] for k in weights if k.endswith('.weight'))}")
        weights[key] = weights[key] * 1.05   # 5% scale error: cosine-invisible
        print(f"NEGATIVE CONTROL: scaled {key} by 1.05\n")

    # front end first — it is outside the model, so nothing else tests it
    cqt = spec_cqt(audio, sr)
    from amt_tools.features import CQT
    import librosa
    a44 = librosa.resample(y=audio, orig_sr=sr, target_sr=SR) if sr != SR else audio
    ref_cqt = np.squeeze(CQT(sample_rate=SR, hop_length=HOP, decibels=True,
                             fmin=librosa.note_to_hz("C1"), n_bins=N_BINS,
                             bins_per_octave=BINS_PER_OCTAVE).process_audio(a44))
    n = min(cqt.shape[1], ref_cqt.shape[1])
    rows = [compare("cqt_db", cqt[:, :n], ref_cqt[:, :n])]

    # then the network, torch-forward with hooks as ground truth
    ref = {}
    name_map = {"conv.0": "conv0", "conv.1": "conv0_relu", "conv.2": "conv1",
                "conv.3": "conv1_relu", "conv.4": "conv2", "conv.5": "conv2_relu",
                "conv.6": "pool", "dense.0": "dense0", "dense.1": "dense0_relu",
                "dense.3.output_layer": "logits"}

    def mk(k):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            if torch.is_tensor(t):
                ref[k] = t.detach().cpu().numpy()
        return fn

    for nm, mod in model.named_modules():
        if nm in name_map:
            mod.register_forward_hook(mk(name_map[nm]))
    with torch.no_grad():
        model(torch.from_numpy(spec_windows(cqt)))

    got = spec_forward(cqt, weights)
    for stage in ["conv0", "conv0_relu", "conv1", "conv1_relu", "conv2",
                  "conv2_relu", "pool", "dense0", "dense0_relu", "logits"]:
        if stage in ref:
            rows.append(compare(stage, got[stage].reshape(ref[stage].shape), ref[stage]))

    print(f"{'stage':14s} {'cos':>12s} {'|ref|/|spec|':>14s} {'|spec|':>12s} "
          f"{'|ref|':>12s} {'max|d|':>10s}  verdict")
    print("-" * 92)
    failed = []
    for r in rows:
        if "cos" not in r:
            print(f"{r['stage']:14s} {r['why']}   FAIL")
            failed.append(r["stage"])
            continue
        verdict = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            failed.append(r["stage"])
        print(f"{r['stage']:14s} {r['cos']:12.9f} {r['ratio']:14.8f} "
              f"{r['spec_norm']:12.3f} {r['ref_norm']:12.3f} {r['max_abs']:10.2e}  {verdict}")

    lp = log_softmax_groups(got["logits"].reshape(-1, 126))
    print(f"\nlog_softmax groups -> {lp.shape}; per-frame sum(exp) = "
          f"{np.exp(lp).sum(-1).mean():.6f} (should be 1.0)")
    print(f"tolerances: cos >= {COS_TOL}, |ratio - 1| <= {RATIO_TOL}")

    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nAll stages PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
