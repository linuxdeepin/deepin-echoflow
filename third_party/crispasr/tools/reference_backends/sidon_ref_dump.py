#!/usr/bin/env python3
"""Dump SaruLab Sidon v0.1 reference intermediates for the CrispASR diff harness.

Runs the upstream *TorchScript* modules (self-contained; no sidon source needed):

  * feature_extractor_cpu.pt  — 16 kHz mono PCM -> predictor features (the handoff)
  * decoder_cpu.pt            — predictor features -> 48 kHz mono PCM

and writes the input, the predictor handoff, and the final 48 kHz output to a
GGUF (`sidon-ref.gguf`) plus a WAV of the reference output. The GGUF is the
reference the native `sidon_diff` compares against per stage, and the WAV answers
"is our output equal to the original model's" directly (ASR round-trip / corr).

Usage:
  python tools/reference_backends/sidon_ref_dump.py \
      --fe   /path/feature_extractor_cpu.pt \
      --dec  /path/decoder_cpu.pt \
      --wav  samples/jfk.wav \
      --out  sidon-ref.gguf \
      --out-wav sidon-ref-48k.wav
"""

from __future__ import annotations

import argparse
import sys
import wave

import numpy as np
import torch


def load_wav_16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getnchannels() == 1, "expected mono"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    x = np.clip(pcm, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767.0).astype(np.int16).tobytes())


def as_np(t) -> np.ndarray:
    if isinstance(t, (tuple, list)):
        t = t[0]
    return np.ascontiguousarray(t.detach().float().cpu().numpy())


def seamless_mel(pcm: np.ndarray) -> torch.Tensor:
    """160-dim SeamlessM4T log-mel features [1, T, 160] — the w2v-BERT input."""
    from transformers import SeamlessM4TFeatureExtractor

    fx = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    out = fx(pcm, sampling_rate=16000, return_tensors="pt")
    return out["input_features"]


def pick_hidden(d):
    """From the w2v-BERT dict output, pick the [.,.,1024] predictor handoff."""
    if torch.is_tensor(d):
        return d
    items = list(d.items())
    for k, v in items:
        if torch.is_tensor(v) and v.dim() == 3 and v.shape[-1] == 1024:
            return v
    # fall back to the first 3-D tensor
    for k, v in items:
        if torch.is_tensor(v) and v.dim() == 3:
            return v
    raise RuntimeError(f"no hidden-state tensor in fe output: {[(k, tuple(v.shape)) for k, v in items]}")


def run_decoder(dec, hidden: torch.Tensor):
    """Call the decoder trying a few conventions on the predictor handoff."""
    cands = [hidden, hidden.transpose(1, 2)]
    last = None
    for c in cands:
        try:
            with torch.no_grad():
                return as_np(dec(c)), tuple(c.shape)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"decoder conventions failed; last: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--wav", default="samples/jfk.wav")
    ap.add_argument("--out", default="sidon-ref.gguf")
    ap.add_argument("--out-wav", default="sidon-ref-48k.wav")
    args = ap.parse_args()

    torch.manual_seed(0)
    fe = torch.jit.load(args.fe, map_location="cpu").eval()
    dec = torch.jit.load(args.dec, map_location="cpu").eval()

    # Introspect the traced forward schemas so mismatches are obvious.
    for name, m in (("feature_extractor", fe), ("decoder", dec)):
        try:
            print(f"[schema] {name}.forward: {m.forward.schema}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            print(f"[schema] {name}: (no schema)", file=sys.stderr)

    pcm = load_wav_16k(args.wav)
    print(f"[input] {pcm.shape[0]} samples @16k", file=sys.stderr)

    mel = seamless_mel(pcm)
    print(f"[mel] input_features {tuple(mel.shape)} ({mel.dtype})", file=sys.stderr)

    with torch.no_grad():
        fe_out = fe(mel)
    if isinstance(fe_out, dict):
        print(f"[fe] dict keys: {[(k, tuple(v.shape)) for k, v in fe_out.items() if torch.is_tensor(v)]}",
              file=sys.stderr)
    hidden = pick_hidden(fe_out)
    feats = as_np(hidden)
    print(f"[fe] handoff features {feats.shape}", file=sys.stderr)

    out, dec_in = run_decoder(dec, hidden)
    out = out.reshape(-1)
    print(f"[dec] in={dec_in} -> out{out.shape} 48k, {out.shape[0] / 48000:.2f}s", file=sys.stderr)

    write_wav(args.out_wav, out, 48000)

    try:
        import gguf
    except ImportError:
        np.savez(args.out.replace(".gguf", ".npz"), input_16k=pcm, predictor_feats=feats.reshape(-1),
                 feats_shape=np.array(feats.shape, dtype=np.int32), output_48k=out)
        print(f"[out] gguf pkg missing — wrote {args.out.replace('.gguf', '.npz')}", file=sys.stderr)
        return 0

    w = gguf.GGUFWriter(args.out, "sidon-ref")
    w.add_tensor("input_16k", pcm.astype(np.float32))
    w.add_tensor("predictor_feats", feats.astype(np.float32))
    w.add_tensor("output_48k", out.astype(np.float32))
    w.add_uint32("sidon_ref.feats_rank", len(feats.shape))
    for i, d in enumerate(feats.shape):
        w.add_uint32(f"sidon_ref.feats_dim.{i}", int(d))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[out] wrote {args.out} + {args.out_wav}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
