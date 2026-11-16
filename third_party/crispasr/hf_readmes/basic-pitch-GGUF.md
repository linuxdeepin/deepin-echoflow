---
license: apache-2.0
base_model:
- spotify/basic-pitch
pipeline_tag: audio-classification
tags:
- music-transcription
- note-events
- basic-pitch
- amt
- gguf
- crispasr
library_name: ggml
---

# Basic Pitch — GGUF (ggml)

GGUF / ggml conversion of [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) (ICASSP 2022 `nmp` model, Apache-2.0) for **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (§250): lightweight polyphonic audio → note-event transcription (any instrument, 88-key MIDI range).

Converted from the `nmp.onnx` that ships inside the basic-pitch repo (sha256 `2c3c1d14…`, 230 KB) with [`models/convert-basic-pitch-to-gguf.py`](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-basic-pitch-to-gguf.py). The nnAudio CQT2010v2 kernels, the 256-tap decimation FIR, and the rescale vector are copied **bit-for-bit from the ONNX initializers** — nothing DSP-critical is reimplemented.

## Files

| File | Size | Notes |
|---|---:|---|
| `basic-pitch-f32.gguf` | 146 KB | exact — note events match the Python reference 27/27 and 11/11 on the validation clips |
| `basic-pitch-f16.gguf` | 113 KB | default; two note *ends* may shift by one frame (11.6 ms) at threshold boundaries |

## Usage

```bash
crispasr --piano --backend basic-pitch -m auto --auto-download -f input.wav
crispasr --piano -m basic-pitch-f32.gguf -f input.wav --piano-format json
```

Parity vs the upstream ONNX (per-stage cosine ≥ 0.9991 on real audio, 1.000000 on native-rate input; end-to-end note events exact at F32) is enforced by the `crispasr-diff basic-pitch` harness with `tools/reference_backends/basic_pitch.py` as the oracle.
