---
license: cc-by-4.0
base_model: nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0
language:
  - ar
tags:
  - automatic-speech-recognition
  - forced-alignment
  - gguf
  - crispasr
  - fastconformer
  - ctc
  - nemo
pipeline_tag: automatic-speech-recognition
---

# stt-ar-fastconformer-hybrid-ctc-large-GGUF

GGUF conversions of the **CTC branch** of [nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0) for [CrispASR](https://github.com/CrispStrobe/CrispASR). The upstream model is a hybrid transducer+CTC Arabic ASR release; the shared FastConformer encoder plus the auxiliary CTC head are extracted here as a standalone CTC model (the RNNT prediction network and joint are dropped), giving a compact Arabic ASR **and forced-alignment** model with punctuation + capitalisation.

| Quant | Size | Description |
|---|---|---|
| F16 | 219 MB | Full precision |
| Q8_0 | 130 MB | 8-bit |
| Q4_K | 82 MB | 4-bit K-quant (recommended) |

## Architecture

17-layer NeMo FastConformer encoder + Conv1d CTC head. d_model=512, 8 heads, SentencePiece vocab, 80 log-mel features, ~115M params.

## Usage

```bash
# Arabic ASR:
crispasr --backend fastconformer-ctc -m stt-ar-fastconformer-hybrid-ctc-large-q4_k.gguf -f audio.wav

# Forced alignment (word timestamps for known text, or re-timing an .srt):
crispasr --align-only -am stt-ar-fastconformer-hybrid-ctc-large-q4_k.gguf \
    -f audio.wav --text-file subtitles.srt --align-output retimed.srt
```

## Attribution

All credit for the model goes to NVIDIA's NeMo team; this repository only repackages the CTC branch in GGUF form under the same CC-BY-4.0 license. Conversion: `models/convert-stt-fastconformer-ctc-to-gguf.py` in CrispASR.
