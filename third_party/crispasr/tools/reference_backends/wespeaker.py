"""WeSpeaker ResNet34-LM speaker-embedding reference dump backend.

Runs the UPSTREAM model as an oracle and captures per-stage activations, so
`crispasr-diff wespeaker` is comparing the ggml runtime against the real
WeSpeaker forward rather than against a second implementation of my own.
Reimplementing the ResNet here would make the diff circular and prove nothing.

The upstream package (wenet-e2e/wespeaker, Apache-2.0) is used only as a
measurement instrument — it is imported, never vendored, and nothing it
returns is copied into the C++ runtime. Point `WESPEAKER_REPO` at a checkout
if the package is not pip-installed:

    git clone --depth 1 https://github.com/wenet-e2e/wespeaker /tmp/wespeaker
    export WESPEAKER_REPO=/tmp/wespeaker

`model_dir` is a snapshot of `Wespeaker/wespeaker-voxceleb-resnet34-LM`
(containing `avg_model`), or a direct path to that checkpoint.

Stages:

  raw_audio        (N,)             input PCM, 16 kHz mono
  fbank            (T, 80)          Kaldi fbank AFTER per-utterance CMN —
                                    the actual model input
  stem_out         (T', 80, 32)     after conv1 + bn1 + relu
  layer1_out       (T', 80, 32)
  layer2_out       (T'/2, 40, 64)
  layer3_out       (T'/4, 20, 128)
  layer4_out       (T'/8, 10, 256)
  stats            (5120,)          TSTP mean||std
  embedding        (256,)           seg_1(stats) — the model output

Feature-map captures are emitted in the ggml layout (width=time fastest, then
height=freq, then channel) so `ref.compare()` sees the same flat ordering the
C++ produces. torch's (C, F, T) is permuted to (T, F, C) for that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "fbank",
    "stem_out",
    "layer1_out",
    "layer2_out",
    "layer3_out",
    "layer4_out",
    "stats",
    "embedding",
]

FEAT_DIM = 80
EMBED_DIM = 256


def _import_wespeaker():
    """Import wespeaker's ResNet34, from the installed package or WESPEAKER_REPO.

    `wespeaker/__init__.py` pulls in the CLI, which imports `silero_vad` and a
    stack of serving deps this dump does not need. `models/resnet.py` itself
    needs only torch plus its sibling `pooling_layers`, so when running from a
    checkout we register synthetic package entries and let the submodule import
    resolve normally — the package __init__ never executes. Same idea as the
    `sys.modules["eventlet"] = None` trick the NeMo backends use, but without
    having to guess which transitive deps to stub.
    """
    repo = os.environ.get("WESPEAKER_REPO")
    if repo:
        import types

        root = Path(repo).resolve() / "wespeaker"
        if not (root / "models" / "resnet.py").exists():
            raise SystemExit(f"WESPEAKER_REPO={repo} has no wespeaker/models/resnet.py")
        for name, path in (("wespeaker", root), ("wespeaker.models", root / "models")):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = [str(path)]  # type: ignore[attr-defined]
                sys.modules[name] = mod
    try:
        from wespeaker.models.resnet import ResNet34  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "wespeaker not importable. Either `pip install wespeaker`, or:\n"
            "  git clone --depth 1 https://github.com/wenet-e2e/wespeaker /tmp/wespeaker\n"
            "  export WESPEAKER_REPO=/tmp/wespeaker\n"
            f"(import error: {exc})"
        ) from exc
    return ResNet34


def _to_ggml_layout(t) -> np.ndarray:
    """(1, C, F, T) torch feature map -> (C, F, T) numpy, contiguous.

    The C++ builds these maps with ne = [W=time, H=freq, C=channels], so the
    flat index is c*(H*W) + f*W + t — TIME fastest, channel slowest. That is
    exactly C-contiguous (C, F, T), so the right move is to drop the batch dim
    and make it contiguous, NOT to permute.

    (An earlier version permuted to (T, F, C), which puts *channel* fastest —
    the exact transpose. The tell was that `stats` and `embedding` compared at
    cos 0.999999 while every feature map read 0.008-0.5: a real compute error
    cannot produce correct outputs from wrong intermediates, so the mismatch
    had to be in the comparison, not the runtime.)
    """
    arr = t.detach().cpu().float()
    if arr.ndim == 4:
        arr = arr[0]  # (C, F, T)
    return arr.contiguous().numpy()


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    import torchaudio.compliance.kaldi as kaldi

    ResNet34 = _import_wespeaker()

    src = Path(model_dir)
    ckpt = src if src.is_file() else src / "avg_model"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} (expected WeSpeaker `avg_model`)")

    print(f"  loading WeSpeaker ResNet34-LM from {ckpt}")
    model = ResNet34(feat_dim=FEAT_DIM, embed_dim=EMBED_DIM,
                     pooling_func="TSTP", two_emb_layer=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # `projection.*` is the ArcMargin training head; the embedding model has
    # no such parameter, so it is expected to be unused here.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("projection.")]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # The shared loader hands us float32 in [-1, 1]; wespeaker's CLI reads with
    # torchaudio.load(normalize=False), i.e. an int16-scale waveform. Rescale
    # so the fbank sees what the real pipeline feeds it.
    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0) * 32768.0

    with torch.no_grad():
        feat = kaldi.fbank(
            wav,
            num_mel_bins=FEAT_DIM,
            frame_length=25,
            frame_shift=10,
            sample_frequency=16000,
            window_type="hamming",
        )  # (T, 80)
        feat = feat - torch.mean(feat, dim=0)  # per-utterance CMN
        if "fbank" in stages:
            out["fbank"] = feat.numpy()

        x = feat.unsqueeze(0)  # (1, T, 80)

        # Capture the residual stages by walking the model explicitly rather
        # than with hooks: _get_frame_level_feat() is a single fused call, so
        # hooks on it would only ever yield the last stage.
        import torch.nn.functional as F

        h = x.permute(0, 2, 1).unsqueeze(1)  # (1, 1, 80, T)
        h = F.relu(model.bn1(model.conv1(h)))
        if "stem_out" in stages:
            out["stem_out"] = _to_ggml_layout(h)
        for i, layer in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), start=1):
            h = layer(h)
            name = f"layer{i}_out"
            if name in stages:
                out[name] = _to_ggml_layout(h)

        stats = model.pool(h)  # (1, 5120)
        if "stats" in stages:
            out["stats"] = stats[0].numpy()

        emb = model.seg_1(stats)  # (1, 256)
        if "embedding" in stages:
            out["embedding"] = emb[0].numpy()

        # Cross-check the oracle against the model's own entry point, so a
        # mistake in this hand-walked forward cannot silently become the
        # reference the C++ is validated against.
        _, ref_emb = model(x)
        delta = float((ref_emb[0] - emb[0]).abs().max())
        if delta > 1e-4:
            raise SystemExit(f"hand-walked forward disagrees with model.forward (max|d|={delta:.3e})")
        print(f"  oracle self-check OK (max|d| vs model.forward = {delta:.2e})")
        print(f"  embedding: dim={emb.shape[1]}  |emb|={float(emb[0].norm()):.4f}")

    return out
