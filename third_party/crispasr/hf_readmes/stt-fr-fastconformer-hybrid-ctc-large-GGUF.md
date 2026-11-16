---
license: cc-by-4.0
base_model: nvidia/stt_fr_fastconformer_hybrid_large_pc
language:
  - fr
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

# stt-fr-fastconformer-hybrid-ctc-large-GGUF

GGUF conversions of the **CTC branch** of [nvidia/stt_fr_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc) for [CrispASR](https://github.com/CrispStrobe/CrispASR). The upstream model is a hybrid transducer+CTC French ASR release; the shared FastConformer encoder plus the auxiliary CTC head are extracted here as a standalone CTC model (the RNNT prediction network and joint are dropped), giving a compact French ASR **and forced-alignment** model with punctuation + capitalisation.

| Quant | Size | Description |
|---|---|---|
| F16 | 219 MB | Full precision |
| Q8_0 | 130 MB | 8-bit |
| Q4_K | 82 MB | 4-bit K-quant (recommended) |

## Architecture

17-layer NeMo FastConformer encoder + Conv1d CTC head. d_model=512, 8 heads, SentencePiece vocab, 80 log-mel features, ~115M params.

## Usage

```bash
# French ASR:
crispasr --backend fastconformer-ctc -m stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf -f audio.wav

# Forced alignment (word timestamps for known text, or re-timing an .srt):
crispasr --align-only -am stt-fr-fastconformer-hybrid-ctc-large-q4_k.gguf \
    -f audio.wav --text-file subtitles.srt --align-output retimed.srt
```

## Attribution

All credit for the model goes to NVIDIA's NeMo team; this repository only repackages the CTC branch in GGUF form under the same CC-BY-4.0 license. Conversion: `models/convert-stt-fastconformer-ctc-to-gguf.py` in CrispASR.
