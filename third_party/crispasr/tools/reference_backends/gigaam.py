"""GigaAM-v3 (ai-sage/GigaAM-v3) reference dump backend.

Loads the HF checkpoint with `trust_remote_code=True` — the driving
inference code IS `modeling_gigaam.py` in the snapshot, so every stage is
captured from the module the model actually runs, not a re-implementation.

`model_dir` is a local snapshot of ONE revision (ctc / rnnt / e2e_ctc /
e2e_rnnt) or the repo id `ai-sage/GigaAM-v3` plus `GIGAAM_REVISION=<rev>`.

Stages:

  raw_audio          (N,)                input PCM (16 kHz mono)
  mel_spectrogram    (n_mels, T_mel)     log-mel, MelsTime layout — matches
                                         gigaam_compute_mel()'s row-major
                                         (n_mels, T) output
  pre_encode_output  (T_enc, d_model)    conv1d striding subsampling
  encoder_layer_K    (T_enc, d_model)    after conformer block K (K < 16)
  encoder_output     (T_enc, d_model)    final encoder hidden state

  CTC heads:
  ctc_log_probs      (T_enc, n_classes)  head output (already log_softmax'd)

  RNN-T heads:
  joint_enc_proj     (T_enc, joint_hidden)  joint.enc(encoder_out)
  pred_initial       (pred_hidden,)         predictor output for the
                                            all-zeros initial embedding
  joint_logits_t0    (n_classes,)           joint.joint_net at frame 0 with
                                            the initial predictor state
                                            (pre-log_softmax is what the C++
                                            emits, so this is the raw
                                            joint_net output — see below)

  generated_text     str                 the blueprint's own transcribe()
                                         output. This is the acceptance
                                         target of HARD RULE #3, so it is
                                         produced by calling
                                         `model.transcribe(wav_file)` on the
                                         SAME decoding object, not by a
                                         re-implemented greedy loop.

Usage:

  python tools/dump_reference.py --backend gigaam \\
      --model-dir /models/GigaAM-v3/e2e_rnnt \\
      --audio example.wav --output gigaam-e2e-rnnt-ref.gguf
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Dict, Set

import numpy as np

N_LAYERS = 16

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "pre_encode_output",
] + [f"encoder_layer_{i}" for i in range(N_LAYERS)] + [
    "encoder_output",
    "ctc_log_probs",
    "joint_enc_proj",
    "pred_initial",
    "joint_logits_t0",
    "generated_text",
]


def _write_temp_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """`transcribe()` takes a path and re-decodes it through ffmpeg, so the
    acceptance transcript has to come from a file. Round-tripping the exact
    int16 the shared loader produced keeps it bit-identical to `audio`."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import torch
    from transformers import AutoModel

    revision = os.environ.get("GIGAAM_REVISION")
    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    print(f"  loading GigaAM-v3 from {model_dir}"
          + (f" (revision {revision})" if revision else ""))
    wrapper = AutoModel.from_pretrained(str(model_dir), **kwargs)
    wrapper.eval()
    model = wrapper.model  # GigaAMASR
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    head_type = model.cfg["model_class"]  # "ctc" | "rnnt"
    print(f"  head={head_type} dtype={dtype}")

    out: Dict[str, np.ndarray] = {}
    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    wav = torch.from_numpy(audio.astype(np.float32)).to(dev).to(dtype).unsqueeze(0)
    length = torch.full([1], wav.shape[-1], device=dev)

    from . import _hooks
    captured: Dict[str, torch.Tensor] = {}
    enc = model.encoder
    stage_modules = []
    if "pre_encode_output" in stages:
        stage_modules.append(("pre_encode_output", enc.pre_encode))
    for i, layer in enumerate(enc.layers):
        name = f"encoder_layer_{i}"
        if name in stages:
            stage_modules.append((name, layer))
    handles = _hooks.capture_modules(captured, stage_modules)

    with torch.inference_mode():
        feats, feat_len = model.preprocessor(wav, length)
        # feats: (B=1, n_mels, T_mel). The C++ mel is row-major (n_mels, T)
        # because the conv1d subsampling consumes 64 channels over time, so
        # no transpose here (unlike the NeMo backends).
        if "mel_spectrogram" in stages:
            T_mel = int(feat_len.item())
            out["mel_spectrogram"] = feats[0, :, :T_mel].detach().cpu().float().numpy()

        encoded, enc_len = model.encoder(feats, feat_len)
        T_enc = int(enc_len.item())
        # encoder() returns (B, d_model, T_enc); crispasr keeps (T_enc, d_model).
        enc_td = encoded[0, :, :T_enc].transpose(0, 1).contiguous()
        if "encoder_output" in stages:
            out["encoder_output"] = enc_td.detach().cpu().float().numpy()

        if head_type == "ctc" and "ctc_log_probs" in stages:
            # CTCHead already applies log_softmax and returns (B, T, C).
            lp = model.head(encoder_output=encoded)
            out["ctc_log_probs"] = lp[0, :T_enc].detach().cpu().float().numpy()

        if head_type == "rnnt":
            joint = model.head.joint
            if "joint_enc_proj" in stages:
                pe = joint.enc(enc_td)
                out["joint_enc_proj"] = pe.detach().cpu().float().numpy()
            # predict(None, None) feeds the zeros embedding through the LSTM
            # with zero state — the decoder's t=0 entry point.
            g, _ = model.head.decoder.predict(None, None)
            if "pred_initial" in stages:
                out["pred_initial"] = g[0, 0].detach().cpu().float().numpy()
            if "joint_logits_t0" in stages:
                # joint.joint() log_softmaxes its output; the C++ joint_step
                # returns raw logits (argmax is invariant, but the diff is
                # not), so reproduce joint_net without the log_softmax.
                f0 = enc_td[0:1].unsqueeze(0)                     # (1, 1, d_model)
                lg = joint.joint_net(joint.enc(f0) + joint.pred(g))
                out["joint_logits_t0"] = lg[0, 0].detach().cpu().float().numpy()

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=T_enc))

    if "generated_text" in stages:
        wav_path = _write_temp_wav(audio)
        try:
            out["generated_text"] = model.transcribe(wav_path)
            print(f"  reference transcript: {out['generated_text']!r}")
        finally:
            os.unlink(wav_path)

    return out
