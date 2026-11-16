#!/usr/bin/env python3
"""Convert the TabCNN guitar-tablature model to GGUF.

TabCNN (Wiggins & Kim, ISMIR 2019) emits, per frame, six independent softmaxes
over 21 fret classes — one per string — with no decoding of any kind. It is the
emission scorer for CrispASR's `--tab` surface; the constrained Viterbi/DP that
turns emissions into a playable fingering lives in the caller.

  weights   `best_TabCNN_tablature_trancription_model` (sic) from the EGSet12
            Zenodo record https://zenodo.org/records/11406378 — **CC BY 4.0**.
            This is the GuitarProFX-augmented variant, which is the one worth
            shipping: the vanilla GuitarSet-trained model collapses from
            tablature F1 0.748 to 0.447 on real electric guitar, while the
            augmented one recovers to 0.585 (DAFx-24). Attribution required.
  code      `amt-tools` (github.com/cwitkowitz/amt-tools) — **MIT**.
            `pip install amt-tools` is needed to unpickle the checkpoint, which
            is a full `amt_tools.models.tabcnn.TabCNN` object, not a state_dict.

⚠️ The checkpoint is NOT the Keras model from `andywiggins/tab-cnn` — that repo
carries no licence file and is irrelevant here. Do not read or port from it.

Usage:

    pip install amt-tools
    python models/convert-tabcnn-to-gguf.py \\
        --model /path/to/best_TabCNN_tablature_trancription_model \\
        --output tabcnn-f16.gguf --dtype f16

Geometry (read from the loaded object, see tools/reference_backends/tabcnn.py):

    front end   librosa.vqt(sr=22050, hop=512, fmin=C1, n_bins=192,
                            bins_per_octave=24, gamma=0) -> abs
                -> amplitude_to_db(ref=np.max)   -> [-80, 0]
                -> /80 + 1                       -> [0, 1]
    input       [1, 192, 9]  (192 CQT bins x 9-frame centred context window)
    conv        Conv2d(1,32,3) ReLU Conv2d(32,64,3) ReLU Conv2d(64,64,3) ReLU
                MaxPool2d(2,2) Dropout(0.25)      192x9 -> 186x3 -> 93x1
    dense       flatten 64*93 = 5952 -> Linear(5952,128) ReLU Dropout(0.5)
    head        Linear(128, 126) -> reshape [6, 21] -> per-string softmax

Dropout is inference-time identity and is not emitted.

⚠️ `ref=np.max` makes the front end a PER-CLIP normalisation: the dB reference
is the maximum of the whole clip's CQT, so features cannot be computed chunked
without changing them. `tabcnn.sample_rate` etc. are written into the GGUF so
the runtime cannot drift from the reference.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

# Front-end constants. These are NOT free parameters — they are read from
# the DAFx-24 paper ("resampled to the 22050Hz sampling rate expected by
# TabCNN") and the checkpoint (dim_in=192, frame_width=9, GuitarProfile).
# fmin is C1, not the guitar's low E: from C1 the 192nd bin sits at 8372 Hz and
# fits under the 11025 Hz Nyquist. Verified end-to-end -- see the converter's
# module docstring and tools/reference_backends/tabcnn.py.
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 192
BINS_PER_OCTAVE = 24
FRAME_WIDTH = 9
FMIN_NOTE = "C1"
DB_FLOOR = 80.0  # librosa amplitude_to_db top_db, and the /80 rescale divisor

# The EXACT upstream artifact. CC BY 4.0 attribution has to be CHECKABLE: a
# reader must be able to fetch these bytes and hash them.
#
# ⚠️ The record has NO Zenodo DOI. An earlier version of this file cited
# `10.5281/zenodo.11406378`, which was invented by pattern-matching Zenodo's
# usual DOI format -- it 404s. The record's real DOI is the arXiv one below
# (verified HTTP 200), and the authoritative locator is the record URL. Cite
# what resolves, not what looks plausible.
SOURCE_RECORD_URL = "https://zenodo.org/records/11406378"
SOURCE_FILE_URL = ("https://zenodo.org/records/11406378/files/"
                   "best_TabCNN_tablature_trancription_model?download=1")
SOURCE_DOI = "10.48550/arXiv.2405.14679"
# Upstream-published checksum, from the Zenodo API for this file. Independent of
# anything we compute locally, so a reader can verify the chain end to end.
SOURCE_MD5 = "ce168b2cd426f81a2a78499214e40605"

# name in the torch state_dict -> name in the GGUF
TENSOR_MAP = {
    "conv.0.weight": "conv0.weight",
    "conv.0.bias": "conv0.bias",
    "conv.2.weight": "conv1.weight",
    "conv.2.bias": "conv1.bias",
    "conv.4.weight": "conv2.weight",
    "conv.4.bias": "conv2.bias",
    "dense.0.weight": "dense0.weight",
    "dense.0.bias": "dense0.bias",
    "dense.3.output_layer.weight": "head.weight",
    "dense.3.output_layer.bias": "head.bias",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="EGSet12 checkpoint (best_TabCNN_tablature_trancription_model)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16",
                    help="dtype for the two large matrices; biases stay f32")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("need torch: pip install torch")
    try:
        import amt_tools  # noqa: F401  (required to unpickle the checkpoint)
    except ImportError:
        sys.exit("need amt-tools to unpickle this checkpoint: pip install amt-tools")
    from gguf import GGUFWriter, GGMLQuantizationType

    model = torch.load(str(args.model), map_location="cpu", weights_only=False)
    model.eval()
    sd = model.state_dict()

    missing = [k for k in TENSOR_MAP if k not in sd]
    if missing:
        sys.exit(f"checkpoint is missing expected tensors: {missing}")

    # Read the head geometry from the model rather than hardcoding it, so a
    # differently-tuned profile fails loudly instead of silently mis-shaping.
    profile = model.profile
    num_strings = profile.get_num_dofs()
    num_classes = profile.num_pitches + 1
    tuning = list(getattr(profile, "tuning", []))
    head_out = sd["dense.3.output_layer.weight"].shape[0]
    if head_out != num_strings * num_classes:
        sys.exit(f"head emits {head_out} but profile implies "
                 f"{num_strings}x{num_classes}={num_strings * num_classes}")
    dim_in = int(getattr(model, "dim_in", N_BINS))
    if dim_in != N_BINS:
        sys.exit(f"model dim_in={dim_in} but this converter assumes {N_BINS} CQT bins")
    # TabCNN derives the flattened dense width from dim_in; check it end to end
    # so a geometry change cannot slip through as a silent reshape.
    expect_flat = 64 * ((dim_in - 6) // 2)
    if sd["dense.0.weight"].shape[1] != expect_flat:
        sys.exit(f"dense0 expects {sd['dense.0.weight'].shape[1]} inputs, "
                 f"geometry implies {expect_flat}")

    import librosa
    fmin_hz = float(librosa.note_to_hz(FMIN_NOTE))

    w = GGUFWriter(str(args.output), "tabcnn", use_temp_file=True)
    w.add_uint32("tabcnn.sample_rate", SAMPLE_RATE)
    w.add_uint32("tabcnn.hop_length", HOP_LENGTH)
    w.add_uint32("tabcnn.n_bins", N_BINS)
    w.add_uint32("tabcnn.bins_per_octave", BINS_PER_OCTAVE)
    w.add_uint32("tabcnn.frame_width", FRAME_WIDTH)
    w.add_uint32("tabcnn.num_strings", num_strings)
    w.add_uint32("tabcnn.num_classes", num_classes)
    w.add_float32("tabcnn.fmin_hz", fmin_hz)
    w.add_float32("tabcnn.db_floor", DB_FLOOR)
    if tuning:
        w.add_array("tabcnn.tuning", [str(t) for t in tuning])
    # The last class is "not played" — a decoder must know which index that is
    # rather than assuming it is the highest.
    w.add_uint32("tabcnn.silent_class", num_classes - 1)
    # Provenance must be VERIFIABLE, not merely cited. A DOI points at the
    # whole EGSet12 record (37 files); it does not say which artifact this GGUF
    # was converted from, so a downstream user cannot check the claim. Record
    # the exact file URL and the sha256 of the bytes we read, computed here from
    # the actual input rather than pasted.
    src_sha = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    w.add_string("general.license", "cc-by-4.0")
    w.add_string("general.source.url", SOURCE_FILE_URL)
    w.add_string("general.source.record_url", SOURCE_RECORD_URL)
    w.add_string("general.source.doi", SOURCE_DOI)
    w.add_string("general.source.md5_upstream", SOURCE_MD5)
    w.add_string("general.source.filename", Path(args.model).name)
    w.add_string("general.source.sha256", src_sha)
    w.add_description(
        "TabCNN guitar tablature emission scorer (Wiggins & Kim, ISMIR 2019); "
        "GuitarProFX-augmented weights from EGSet12 (CC BY 4.0). Emits six "
        "per-string distributions over 21 fret classes per frame; decoding is "
        "the caller's responsibility.")

    f16 = GGMLQuantizationType.F16
    f32 = GGMLQuantizationType.F32
    big_dt = np.float16 if args.dtype == "f16" else np.float32
    big_qt = f16 if args.dtype == "f16" else f32

    n_written = 0
    for src, dst in TENSOR_MAP.items():
        t = sd[src].detach().cpu().numpy()
        # Keep biases and the tiny conv kernels at f32; only the two large
        # matrices (dense0 761k, head 16k) are worth narrowing. The conv stack
        # is 55k params total, so f16 there buys nothing and costs accuracy.
        if dst.endswith(".bias") or dst.startswith("conv"):
            w.add_tensor(dst, np.ascontiguousarray(t.astype(np.float32)), raw_dtype=f32)
        else:
            w.add_tensor(dst, np.ascontiguousarray(t.astype(big_dt)), raw_dtype=big_qt)
        n_written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    total = sum(int(np.prod(sd[k].shape)) for k in TENSOR_MAP)
    print(f"wrote {args.output}  ({n_written} tensors, {total:,} params, "
          f"{args.output.stat().st_size / 1e6:.2f} MB)")
    print(f"  strings={num_strings} classes={num_classes} silent_class={num_classes - 1}")
    print(f"  source sha256: {src_sha}")
    print(f"  front end: {SAMPLE_RATE} Hz, hop {HOP_LENGTH}, {N_BINS} bins @ "
          f"{BINS_PER_OCTAVE}/oct from {FMIN_NOTE} ({fmin_hz:.2f} Hz)")
    if tuning:
        print(f"  tuning: {' '.join(str(t) for t in tuning)}")


if __name__ == "__main__":
    main()
