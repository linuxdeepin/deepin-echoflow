"""TabCNN reference backend — amt_tools ground truth for crispasr-diff.

TabCNN (Wiggins & Kim, ISMIR 2019) is a ~0.83 M-parameter CNN that emits, per
frame, six independent softmaxes over 21 fret classes — one per string. There
is no decoding, no temporal model and no autoregression, so this dumper is a
plain forward pass with leaf-module hooks.

Provenance (see docs/music-transcription/GUITAR_TAB_SPEC.md §0):

  weights   the GuitarProFX-augmented model from the EGSet12 Zenodo record
            (https://zenodo.org/records/11406378), **CC BY 4.0** — attribution required.
            ⚠ There is NO Zenodo DOI; `10.5281/zenodo.11406378` 404s.
            The record DOI is the arXiv one, 10.48550/arXiv.2405.14679.
            File: `best_TabCNN_tablature_trancription_model` (sic, 3.3 MB).
  code      `amt-tools` (github.com/cwitkowitz/amt-tools), **MIT**.
            `pip install amt-tools`.

  ⚠️ The shipped weights are a pickled `amt_tools.models.tabcnn.TabCNN`, NOT
  the Keras model from `andywiggins/tab-cnn`. That repo has no licence file and
  is irrelevant here — do not read or port from it.

Usage:

    python tools/dump_reference.py --backend tabcnn \\
        --model-dir /path/to/best_TabCNN_tablature_trancription_model \\
        --audio samples/guitar.wav \\
        --output /tmp/tabcnn-ref.gguf

    build/bin/crispasr-diff tabcnn tabcnn-f16.gguf \\
        /tmp/tabcnn-ref.gguf samples/guitar.wav

Geometry, read from the loaded object (not inferred):

    input   [1, 192, 9]  — 192 CQT bins x a 9-frame context window
    conv    Conv2d(1,32,3x3) -> ReLU -> Conv2d(32,64,3x3) -> ReLU
            -> Conv2d(64,64,3x3) -> ReLU -> MaxPool2d(2,2) -> Dropout(0.25)
            192x9 -> 190x7 -> 188x5 -> 186x3 -> pool -> 93x1
    dense   flatten 64*93*1 = 5952 -> Linear(5952,128) -> ReLU -> Dropout(0.5)
    head    SoftmaxGroups(dim_in=128, num_groups=6, num_classes=21)
            = Linear(128,126), reshaped to [6, 21]

⚠️ THE FRONT END IS NOT IN THE MODEL. `model.frontend` is an EMPTY Sequential
(identity). The CQT lives in `amt_tools.features.CQT`, so a diff that starts at
`cqt_db` proves nothing about our own CQT — exactly the blind spot that let a
front-end mismatch through on BTC (86.63% -> 98.56% mir_eval once fixed) and on
piano-transcription. This dumper therefore ALSO emits the raw `audio` and the
`cqt_db` input, so the C++ side can be diffed from the waveform, not from
replayed features.

The exact front end, read from amt_tools source:

    librosa.vqt(y, sr=22050, hop=512, fmin=C1(32.70 Hz), n_bins=192,
                bins_per_octave=24, gamma=0)
    -> np.abs(...)
    -> librosa.core.amplitude_to_db(feats, ref=np.max)   # -> [-80, 0]
    -> feats / 80 + 1                                    # -> [0, 1]

⚠️⚠️ `ref=np.max` is a PER-CLIP normalisation: the dB scale is relative to the
maximum of the WHOLE clip's CQT. Two consequences the C++ must honour:
  (1) it cannot be computed streaming/chunked without changing the features —
      chunk the audio and every chunk gets its own reference;
  (2) librosa's `amplitude_to_db` also clamps at `top_db=80` by default.
A cosine check cannot see either mistake (both are monotone/scale effects), so
`tools/tabcnn_torch_parity.py` must assert the median per-bin magnitude ratio,
per the `core/cqt.h` 152x scale bug.

Frame count is `1 + len(audio) // hop_length` (amt_tools
`FeatureModule.get_expected_frames`), NOT `len(audio) // hop_length`.
"""

import os

import numpy as np

DEFAULT_STAGES = [
    "audio",
    "cqt_db",
    "conv0", "conv0_relu",
    "conv1", "conv1_relu",
    "conv2", "conv2_relu",
    "pool", "dropout_conv",
    "dense0", "dense0_relu",
    "logits",
    "tablature",
]

# Read from the loaded checkpoint + amt_tools sources, NOT inferred:
#   amt_tools/datasets/GuitarSet.py  -> sample_rate=44100, hop_length=512
#   model.profile (GuitarProfile)    -> tuning E2 A2 D3 G3 B3 E4, num_pitches=20
#                                       -> 6 groups x (20+1) = 126 outputs
#   model.dim_in=192, frame_width=9; TabCNN's (dim_in-6)//2 = 93 matches the
#   observed pool height, so 192 bins is confirmed end-to-end.
# fmin is C1 (32.70 Hz), NOT E2. Assuming the guitar's lowest string is the
# obvious guess and it is wrong: 192 bins at 24/octave from E2 reaches 21096 Hz,
# which cannot fit under the 22050 Hz Nyquist that the DAFx-24 paper states
# TabCNN expects ("resampled to the 22050Hz sampling rate expected by TabCNN").
# From C1 the top bin is 8372 Hz and it fits. Measured end-to-end on EGSet12
# track 01 against its JAMS ground truth: fmin C1 -> tablature F1 0.771,
# E1 -> 0.040, E2 at 44.1 kHz -> 0.0008. The whole chain is worthless if this
# constant is wrong, and every wrong value still RUNS.
_SR = 22050
_HOP = 512
_N_BINS = 192
_BINS_PER_OCTAVE = 24
_FRAME_WIDTH = 9


def _resolve_model_path(model_dir):
    """--model-dir may be the model FILE or a directory containing it."""
    p = str(model_dir)
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        for name in sorted(os.listdir(p)):
            if "TabCNN" in name or name.endswith((".pt", ".pth")):
                return os.path.join(p, name)
    raise FileNotFoundError(
        f"could not resolve a TabCNN checkpoint from --model-dir {model_dir!r}. "
        "Pass the file from the EGSet12 Zenodo record "
        "(best_TabCNN_tablature_trancription_model)."
    )


def _cqt_db(audio, sample_rate):
    """Reproduce amt_tools.features.CQT exactly.

    Kept as an explicit local function rather than instantiating the amt_tools
    module so the parity target is visible in one place and can be compared
    line-by-line against core/cqt.h.
    """
    import librosa

    if sample_rate != _SR:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=_SR)
    fmin = librosa.note_to_hz("C1")
    vqt = librosa.vqt(y=audio, sr=_SR, hop_length=_HOP, fmin=fmin,
                      n_bins=_N_BINS, bins_per_octave=_BINS_PER_OCTAVE, gamma=0)
    feats = np.abs(vqt)
    # ⚠️ per-clip reference — see module docstring
    feats = librosa.core.amplitude_to_db(feats, ref=np.max)
    # ⚠️⚠️ amt_tools FeatureModule.post_proc does NOT stop at dB. It affinely
    # rescales the assumed [-80, 0] dB range into [0, 1]. Omitting these two
    # lines feeds the CNN inputs ~180x out of range: measured cos = -0.544 and
    # a median per-bin magnitude ratio of 0.0047 against the real module.
    # Cosine alone would have partly hidden a pure scale error — it was the
    # |mine| vs |ref| magnitudes (15895 vs 88) that made it obvious.
    feats = feats / 80
    feats = feats + 1
    return feats.astype(np.float32)


def _windows(cqt, frame_width=_FRAME_WIDTH):
    """[bins, T] -> [T, 1, bins, frame_width], zero-padded at both edges.

    amt_tools frames the CQT with a centred context window; the model consumes
    one window per output frame.
    """
    pad = frame_width // 2
    padded = np.pad(cqt, ((0, 0), (pad, pad)), mode="constant")
    n_bins, total = padded.shape
    T = total - 2 * pad
    out = np.empty((T, 1, n_bins, frame_width), dtype=np.float32)
    for t in range(T):
        out[t, 0] = padded[:, t:t + frame_width]
    return out


def dump(model_dir, audio, stages, max_new_tokens=None, sample_rate=16000, **kwargs):
    """Run the amt_tools TabCNN forward and return per-stage intermediates.

    Args:
        model_dir: path to the EGSet12 checkpoint (file or containing dir).
        audio:     float32 numpy array, mono PCM (resampled to 22.05 kHz here).
        stages:    set of stage names to capture.
        max_new_tokens: unused (no autoregressive decoding).

    Returns:
        dict of {stage_name: numpy_array}
    """
    import torch

    want = set(stages) if stages else set(DEFAULT_STAGES)
    out = {}

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if "audio" in want:
        out["audio"] = audio

    cqt = _cqt_db(audio, sample_rate)
    if "cqt_db" in want:
        # Emitted TRANSPOSED to [T, n_bins]. librosa and amt_tools produce
        # [n_bins, T], but core/cqt.h emits frame-major [T, n_bins], and this
        # archive exists to be diffed against the C++ runtime. Comparing the two
        # layouts flat is apples-to-oranges: it reads cos 0.66 with the norms
        # matching to 0.3%, which is the transpose signature and is exactly how
        # it presented the first time. Keep this aligned with the runtime, not
        # with librosa.
        out["cqt_db"] = np.ascontiguousarray(cqt.T)

    model = torch.load(_resolve_model_path(model_dir), map_location="cpu",
                       weights_only=False)
    model.eval()
    # The checkpoint was saved from a CUDA box; amt_tools reads model.device
    # when moving inputs, so force it to CPU or the forward tries cuda:0.
    model.device = "cpu"

    # Hook every LEAF module. Names below mirror DEFAULT_STAGES so the C++ side
    # can key on stable identifiers rather than torch's module paths.
    name_map = {
        "conv.0": "conv0", "conv.1": "conv0_relu",
        "conv.2": "conv1", "conv.3": "conv1_relu",
        "conv.4": "conv2", "conv.5": "conv2_relu",
        "conv.6": "pool", "conv.7": "dropout_conv",
        "dense.0": "dense0", "dense.1": "dense0_relu",
        "dense.3.output_layer": "logits",
    }
    captured = {}

    def _hook(key):
        def fn(_mod, _inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(t):
                captured[key] = t.detach().cpu().numpy()
        return fn

    handles = []
    for name, mod in model.named_modules():
        if name in name_map and name_map[name] in want:
            handles.append(mod.register_forward_hook(_hook(name_map[name])))

    x = torch.from_numpy(_windows(cqt))
    with torch.no_grad():
        result = model(x)
    for h in handles:
        h.remove()
    out.update(captured)

    if "tablature" in want:
        tab = result["tablature"] if isinstance(result, dict) else result
        tab = tab.detach().cpu().numpy() if torch.is_tensor(tab) else np.asarray(tab)
        # [T, 1, 126] -> [T, 6, 21]; the C++ emits log-probs, so the diff should
        # compare pre-softmax logits (`logits`) AND this reshaped view.
        out["tablature"] = tab.reshape(tab.shape[0], 6, 21)

    return out
