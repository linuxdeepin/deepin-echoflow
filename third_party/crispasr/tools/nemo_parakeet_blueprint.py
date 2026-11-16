#!/usr/bin/env python3
"""Run upstream NeMo's inference modes on a parakeet .nemo checkpoint —
the blueprint-parity benchmark from the issue #89 audit.

Purpose: before hunting a "port bug", establish what the ORIGINAL Python
stack produces on the exact same audio. If the GGUF runtime's output is
char-identical to NeMo's in a given mode (it was, in every comparable mode),
the loss is a model pathology and the fix is pipeline policy, not graph math.

Modes (comma-separated via --modes):
  plain      model.transcribe() — full attention, TDT decode
  ctc        CTC branch decode (hybrid TDT+CTC models), full attention
  local      change_attention_model("rel_pos_local_attn", [C, C]) — NeMo's
             documented long-form switch; C from --context (encoder frames,
             1 frame = 80 ms). CrispASR equivalent:
             CRISPASR_PARAKEET_ATT_CONTEXT="C,C"
  local-ctc  local attention + CTC branch
  buffered   BatchedFrameASRTDT chunked long-form inference (NeMo's
             recommended long-form path; --chunk-len/--total-buffer;
             stock example params are 1.6/4.0)

Measured on the issue #89 reporter's 60 s clip (recall vs whisper-large-v3):
plain 11 %, ctc 1 %, local[128] 46 %, buffered 15-51 % — all worse than
CrispASR's shipped VAD-cap + gap-fill pipeline (97 %). Score outputs with
tools/asr_coverage_score.py.

macOS note: NeMo imports eventlet-adjacent code that breaks; the
sys.modules["eventlet"] = None stub below is required before importing nemo.

Usage:
  python tools/nemo_parakeet_blueprint.py \
      --nemo /path/parakeet-tdt_ctc-0.6b-ja.nemo \
      --audio clip1.wav clip2.wav --outdir /tmp/nemo-bench \
      --modes plain,local,buffered --context 128
"""

import sys

sys.modules["eventlet"] = None  # macOS import workaround — must precede nemo

import argparse
import math
import os


def transcribe_standard(model, mode, context, wavs, outdir):
    local = "local" in mode
    ctc = "ctc" in mode
    if local:
        model.change_attention_model("rel_pos_local_attn", [context, context])
        model.change_subsampling_conv_chunking_factor(1)
    else:
        model.change_attention_model("rel_pos")
    if hasattr(model, "change_decoding_strategy"):
        try:
            model.change_decoding_strategy(decoder_type="ctc" if ctc else "rnnt")
        except Exception as e:
            print(f"{mode}: change_decoding_strategy failed: {e}")
            if ctc:
                return
    for wav in wavs:
        base = os.path.splitext(os.path.basename(wav))[0]
        try:
            out = model.transcribe([wav], batch_size=1)
            text = out[0].text if hasattr(out[0], "text") else str(out[0])
        except Exception as e:
            text = ""
            print(f"{mode} {base}: FAILED {e}")
        write(outdir, mode, base, text)


def transcribe_buffered(model, chunk_len, total_buffer, wavs, outdir):
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.asr.parts.utils.streaming_utils import BatchedFrameASRTDT
    from nemo.collections.asr.parts.utils.transcribe_utils import get_buffered_pred_feat_rnnt

    model_cfg = model._cfg
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0
    OmegaConf.set_struct(model_cfg.preprocessor, True)
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy_batch"
        decoding_cfg.preserve_alignments = True
        if hasattr(model, "cur_decoder"):
            model.change_decoding_strategy(decoding_cfg, decoder_type="rnnt")
        else:
            model.change_decoding_strategy(decoding_cfg)

    feature_stride = model_cfg.preprocessor["window_stride"]
    model_stride_in_secs = feature_stride * 8  # fastconformer 8x subsampling
    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)

    frame_asr = BatchedFrameASRTDT(asr_model=model, frame_len=chunk_len, total_buffer=total_buffer, batch_size=1)
    hyps = get_buffered_pred_feat_rnnt(
        asr=frame_asr,
        tokens_per_chunk=tokens_per_chunk,
        delay=mid_delay,
        model_stride_in_secs=model_stride_in_secs,
        batch_size=1,
        manifest=None,
        filepaths=list(wavs),
        accelerator="cpu",
    )
    tag = f"buffered{chunk_len:g}x{total_buffer:g}"
    for wav, h in zip(wavs, hyps):
        base = os.path.splitext(os.path.basename(wav))[0]
        write(outdir, tag, base, h.text if hasattr(h, "text") else str(h))


def write(outdir, mode, base, text):
    path = os.path.join(outdir, f"nemo_{mode}_{base}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"{mode} {base}: {len(text)} chars -> {path}")
    print(f"  {text[:120]}...")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True, help="path to .nemo checkpoint")
    ap.add_argument("--audio", required=True, nargs="+", help="16 kHz mono wav file(s)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--modes", default="plain", help="comma list: plain,ctc,local,local-ctc,buffered")
    ap.add_argument("--context", type=int, default=128, help="local-attention window (encoder frames, 80 ms each)")
    ap.add_argument("--chunk-len", type=float, default=1.6, help="buffered: new speech per step (s)")
    ap.add_argument("--total-buffer", type=float, default=4.0, help="buffered: window incl. context (s)")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr

    os.makedirs(args.outdir, exist_ok=True)
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    model.eval()

    for mode in args.modes.split(","):
        if mode == "buffered":
            transcribe_buffered(model, args.chunk_len, args.total_buffer, args.audio, args.outdir)
        else:
            transcribe_standard(model, mode, args.context, args.audio, args.outdir)


if __name__ == "__main__":
    main()
