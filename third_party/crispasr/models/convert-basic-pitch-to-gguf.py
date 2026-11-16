#!/usr/bin/env python3
"""
Convert Spotify Basic Pitch (ICASSP 2022) nmp.onnx → GGUF (arch "basic-pitch").

Source model (Apache-2.0), shipped inside the pip package and the git repo:
    basic_pitch/saved_models/icassp_2022/nmp.onnx
    sha256 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec

Why the ONNX and not the TF SavedModel: the ONNX has (a) the BatchNorm layers
already folded into the preceding convolutions and (b) the nnAudio CQT2010v2
kernels present as plain initializers. Taking the CQT kernels straight out of
the graph means the C++ front end never has to re-derive them — in particular
it never has to reimplement `scipy.signal.firwin2`, which designs the 256-tap
decimation lowpass. A derivation mismatch there would be invisible until the
per-octave parity numbers came back wrong.

Tensor roles are resolved by walking the graph (op type, kernel_shape, weight
shape, and "does this conv's output feed a Neg"), NOT by the `const_fold_opt__NNN`
initializer names, which are an artefact of one particular tf2onnx run.

GGUF tensor naming:
    basic_pitch.cqt.kernel_real     (36, 256)   F32  top-octave CQT kernels
    basic_pitch.cqt.kernel_imag     (36, 256)   F32
    basic_pitch.cqt.lowpass         (256,)      F32  x2 decimation FIR
    basic_pitch.cqt.sqrt_lengths    (309,)      F32  librosa-matching rescale
    basic_pitch.contour_conv.{weight,bias}   (8,8,3,39)  / (8,)
    basic_pitch.contour_out.{weight,bias}    (1,8,5,5)   / (1,)
    basic_pitch.note_conv.{weight,bias}      (32,1,7,7)  / (32,)
    basic_pitch.note_out.{weight,bias}       (1,32,7,3)  / (1,)
    basic_pitch.onset_conv.{weight,bias}     (32,8,5,5)  / (32,)
    basic_pitch.onset_out.{weight,bias}      (1,33,3,3)  / (1,)

Conv weights keep the ONNX layout (out_ch, in_ch, kh, kw), contiguous, so the
C++ side indexes w[((oc*IC + ic)*KH + kh)*KW + kw].

Usage:
    python models/convert-basic-pitch-to-gguf.py \
        --input /mnt/storage/gguf-models/basic-pitch-src/nmp.onnx \
        --output /mnt/storage/gguf-models/basic-pitch-f16.gguf
    # --f32 keeps the convolutions in F32 (default is F16)
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("pip install onnx", file=sys.stderr)
    sys.exit(1)

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


# ── hyperparameters, from basic_pitch/constants.py + models.py ──────────────
# See /mnt/volume1/tmp-overflow/basic-pitch-notes.md for the derivations.

SAMPLE_RATE = 22050
FFT_HOP = 256
AUDIO_WINDOW_LENGTH = 2
ANNOTATIONS_FPS = SAMPLE_RATE // FFT_HOP                 # 86
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH   # 172
AUDIO_N_SAMPLES = SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844

ANNOTATIONS_BASE_FREQUENCY = 27.5
ANNOTATIONS_N_SEMITONES = 88
CONTOURS_BINS_PER_SEMITONE = 3
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE  # 264
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES                                  # 88

N_HARMONICS = 8
HARMONICS = [0.5] + list(range(1, N_HARMONICS))          # [0.5,1,...,7]
HARMONIC_SHIFTS = [
    int(round(12 * CONTOURS_BINS_PER_SEMITONE * np.log2(float(h)))) for h in HARMONICS
]                                                        # [-36,0,36,57,72,84,93,101]

CQT_BINS_PER_OCTAVE = 12 * CONTOURS_BINS_PER_SEMITONE    # 36
MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))  # 103
CQT_N_SEMITONES = min(int(np.ceil(12.0 * np.log2(N_HARMONICS)) + ANNOTATIONS_N_SEMITONES), MAX_N_SEMITONES)
CQT_N_BINS = CQT_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE  # 309
CQT_N_OCTAVES = int(np.ceil(CQT_N_BINS / CQT_BINS_PER_OCTAVE))  # 9
CQT_N_FILTERS = min(CQT_BINS_PER_OCTAVE, CQT_N_BINS)            # 36
CQT_N_FFT = 256

MIDI_OFFSET = 21
OVERLAPPING_FRAMES = 30
NORM_LOG_EPS = 1e-10


# ── graph walking ───────────────────────────────────────────────────────────

def build_index(graph):
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    consumers = {}
    for node in graph.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    return init, consumers


def attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return None


def conv_weight(node, init):
    """Return (weight_array, bias_array_or_None) for a Conv node."""
    w = init.get(node.input[1])
    if w is None:
        raise RuntimeError(f"Conv {node.name}: weight {node.input[1]} is not an initializer")
    b = init.get(node.input[2]) if len(node.input) > 2 else None
    return w, b


def extract(model):
    g = model.graph
    init, consumers = build_index(g)
    convs = [n for n in g.node if n.op_type == "Conv"]

    out = {}
    meta = {}

    # ── CQT kernels ────────────────────────────────────────────────────────
    # The 36-bin top-octave kernel bank is applied twice per octave: once for
    # the real part and once for the imaginary part, whose Conv output is
    # immediately negated (nnaudio.get_cqt_complex: `CQT_imag = -conv1d(...)`).
    real_k = imag_k = None
    lowpass = None
    for n in convs:
        ks = attr(n, "kernel_shape")
        if list(ks) != [1, 256]:
            continue
        w, _ = conv_weight(n, init)
        if w.shape == (1, 1, 1, 256):
            if lowpass is None:
                lowpass = w.reshape(256)
            continue
        if w.shape != (36, 1, 1, 256):
            raise RuntimeError(f"unexpected CQT kernel shape {w.shape}")
        negated = any(c.op_type == "Neg" for c in consumers.get(n.output[0], []))
        if negated:
            imag_k = w.reshape(36, 256)
        else:
            real_k = w.reshape(36, 256)
    if real_k is None or imag_k is None or lowpass is None:
        raise RuntimeError("could not locate CQT kernels / lowpass in the graph")
    out["basic_pitch.cqt.kernel_real"] = real_k
    out["basic_pitch.cqt.kernel_imag"] = imag_k
    out["basic_pitch.cqt.lowpass"] = lowpass

    # sqrt(lengths) rescale — the only (n_bins,1,1) initializer.
    sq = [a for a in init.values() if a.dtype == np.float32 and a.shape == (CQT_N_BINS, 1, 1)]
    if len(sq) != 1:
        raise RuntimeError(f"expected 1 sqrt(lengths) tensor, found {len(sq)}")
    out["basic_pitch.cqt.sqrt_lengths"] = sq[0].reshape(CQT_N_BINS)

    # ── CQT BatchNorm (1 channel, folded to x*scale + shift) ───────────────
    # Find the Mul/Add pair that sits between the NormalizedLog output and the
    # harmonic-stacking slices.
    hs_producers = set()
    for n in g.node:
        if n.op_type in ("Slice", "Pad") and "harmonic-stacking" in n.name:
            hs_producers.update(n.input)
    bn_add = None
    for n in g.node:
        if n.op_type == "Add" and n.output[0] in hs_producers:
            bn_add = n
    if bn_add is None:
        raise RuntimeError("could not locate the CQT BatchNorm Add")
    shift_arr = next(init[i] for i in bn_add.input if i in init)
    producer = {o: n for n in g.node for o in n.output}
    bn_mul = producer[next(i for i in bn_add.input if i not in init)]
    if bn_mul.op_type != "Mul":
        raise RuntimeError(f"expected Mul before the BatchNorm Add, got {bn_mul.op_type}")
    scale_arr = next(init[i] for i in bn_mul.input if i in init)
    meta["cqt_bn_scale"] = float(np.asarray(scale_arr).reshape(-1)[0])
    meta["cqt_bn_shift"] = float(np.asarray(shift_arr).reshape(-1)[0])

    # ── model convolutions ─────────────────────────────────────────────────
    # Keyed on (kernel_shape, weight shape) — unique for all six, and readable
    # against the table in models.py.
    want = {
        (3, 39): "contour_conv",
        (7, 7): "note_conv",
        (7, 3): "note_out",
        (3, 3): "onset_out",
    }
    for n in convs:
        ks = tuple(attr(n, "kernel_shape"))
        if ks == (1, 256):
            continue
        w, b = conv_weight(n, init)
        if ks == (5, 5):
            key = "contour_out" if w.shape[0] == 1 else "onset_conv"
        else:
            key = want.get(ks)
        if key is None:
            raise RuntimeError(f"unmapped Conv kernel_shape {ks}")
        if key + ".weight" in [k.split("basic_pitch.")[-1] for k in out]:
            raise RuntimeError(f"duplicate conv role {key}")
        out[f"basic_pitch.{key}.weight"] = w
        if b is None:
            raise RuntimeError(f"conv {key} has no bias")
        out[f"basic_pitch.{key}.bias"] = b
        meta.setdefault("pads", {})[key] = list(attr(n, "pads"))
        meta.setdefault("strides", {})[key] = list(attr(n, "strides"))

    expected_shapes = {
        "basic_pitch.contour_conv.weight": (8, 8, 3, 39),
        "basic_pitch.contour_out.weight": (1, 8, 5, 5),
        "basic_pitch.note_conv.weight": (32, 1, 7, 7),
        "basic_pitch.note_out.weight": (1, 32, 7, 3),
        "basic_pitch.onset_conv.weight": (32, 8, 5, 5),
        "basic_pitch.onset_out.weight": (1, 33, 3, 3),
    }
    for name, shape in expected_shapes.items():
        if name not in out:
            raise RuntimeError(f"missing {name}")
        if out[name].shape != shape:
            raise RuntimeError(f"{name}: expected {shape}, got {out[name].shape}")

    # Padding/stride sanity — these are what the C++ hard-codes.
    expected_pads = {
        "contour_conv": [1, 19, 1, 19],
        "contour_out": [2, 2, 2, 2],
        "note_conv": [3, 2, 3, 2],
        "note_out": [3, 1, 3, 1],
        "onset_conv": [2, 1, 2, 1],
        "onset_out": [1, 1, 1, 1],
    }
    expected_strides = {
        "contour_conv": [1, 1], "contour_out": [1, 1], "note_conv": [1, 3],
        "note_out": [1, 1], "onset_conv": [1, 3], "onset_out": [1, 1],
    }
    for k, v in expected_pads.items():
        if meta["pads"][k] != v:
            raise RuntimeError(f"{k}: pads {meta['pads'][k]} != expected {v}")
    for k, v in expected_strides.items():
        if meta["strides"][k] != v:
            raise RuntimeError(f"{k}: strides {meta['strides'][k]} != expected {v}")

    return out, meta


def main():
    ap = argparse.ArgumentParser(description="Convert Basic Pitch nmp.onnx to GGUF")
    ap.add_argument("--input", "-i", required=True, help="Path to nmp.onnx")
    ap.add_argument("--output", "-o", required=True, help="Output GGUF path")
    ap.add_argument("--f32", action="store_true", help="Keep convolutions in F32 (default F16)")
    args = ap.parse_args()

    with open(args.input, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(f"Loading ONNX: {args.input}\n  sha256 {digest}")

    model = onnx.load(args.input)
    tensors, meta = extract(model)
    print(f"  extracted {len(tensors)} tensors, "
          f"cqt bn scale={meta['cqt_bn_scale']:.7g} shift={meta['cqt_bn_shift']:.7g}")

    writer = GGUFWriter(args.output, "basic-pitch")

    writer.add_string("general.name", "basic-pitch")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.source.url", "https://github.com/spotify/basic-pitch")
    writer.add_string("basic_pitch.source.sha256", digest)

    writer.add_uint32("basic_pitch.sample_rate", SAMPLE_RATE)
    writer.add_uint32("basic_pitch.fft_hop", FFT_HOP)
    writer.add_uint32("basic_pitch.audio_n_samples", AUDIO_N_SAMPLES)
    writer.add_uint32("basic_pitch.annot_n_frames", ANNOT_N_FRAMES)
    writer.add_uint32("basic_pitch.annotations_fps", ANNOTATIONS_FPS)
    writer.add_uint32("basic_pitch.overlapping_frames", OVERLAPPING_FRAMES)
    writer.add_uint32("basic_pitch.midi_offset", MIDI_OFFSET)
    writer.add_uint32("basic_pitch.n_freq_bins_notes", N_FREQ_BINS_NOTES)
    writer.add_uint32("basic_pitch.n_freq_bins_contours", N_FREQ_BINS_CONTOURS)
    writer.add_uint32("basic_pitch.contours_bins_per_semitone", CONTOURS_BINS_PER_SEMITONE)

    writer.add_uint32("basic_pitch.cqt.n_bins", CQT_N_BINS)
    writer.add_uint32("basic_pitch.cqt.bins_per_octave", CQT_BINS_PER_OCTAVE)
    writer.add_uint32("basic_pitch.cqt.n_octaves", CQT_N_OCTAVES)
    writer.add_uint32("basic_pitch.cqt.n_filters", CQT_N_FILTERS)
    writer.add_uint32("basic_pitch.cqt.n_fft", CQT_N_FFT)
    writer.add_uint32("basic_pitch.cqt.hop", FFT_HOP)
    writer.add_float32("basic_pitch.cqt.fmin", ANNOTATIONS_BASE_FREQUENCY)
    writer.add_float32("basic_pitch.cqt.bn_scale", meta["cqt_bn_scale"])
    writer.add_float32("basic_pitch.cqt.bn_shift", meta["cqt_bn_shift"])
    writer.add_float32("basic_pitch.norm_log_eps", NORM_LOG_EPS)

    writer.add_uint32("basic_pitch.n_harmonics", N_HARMONICS)
    writer.add_array("basic_pitch.harmonics", [float(h) for h in HARMONICS])
    # Written as float32 because core_gguf only exposes a float array reader;
    # the values are exact small integers, so nothing is lost.
    writer.add_array("basic_pitch.harmonic_shifts", [float(s) for s in HARMONIC_SHIFTS])

    f16_convs = not args.f32
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        # The CQT front end and all biases stay F32: they are tiny, and the
        # front end is where a rounding error compounds across nine octaves of
        # recursive decimation.
        as_f16 = f16_convs and name.endswith(".weight") and ".cqt." not in name
        if as_f16:
            writer.add_tensor(name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
        print(f"  {name}: {list(arr.shape)} {'F16' if as_f16 else 'F32'}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
