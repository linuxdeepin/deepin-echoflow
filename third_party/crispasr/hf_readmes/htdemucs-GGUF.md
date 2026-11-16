---
license: mit
tags:
  - audio
  - source-separation
  - demucs
  - htdemucs
  - gguf
  - crispasr
base_model: facebook/htdemucs
pipeline_tag: audio-to-audio
---

# HTDemucs — GGUF

GGUF conversions of [Meta's HTDemucs](https://github.com/facebookresearch/demucs) (Hybrid Transformer Demucs) for use with [CrispASR](https://github.com/CrispStrobe/CrispASR).

HTDemucs is a state-of-the-art music source separation model that splits audio into 4 stems: **drums**, **bass**, **other**, and **vocals**.

## Files

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `htdemucs-f16.gguf` | F16 | 81 MB | Full precision (conv weights F16, norms/biases F32) |
| `htdemucs-q8_0.gguf` | Q8_0 | 53 MB | Transformer weights quantized, conv weights F16 |
| `htdemucs-q4_k.gguf` | Q4_K | 38 MB | Aggressive quantization of transformer weights |

40 of 533 tensors are quantized (the CrossTransformer attention/FFN projections). Encoder/decoder conv weights remain at F16 (required by the ggml Conv2d kernel). Norms, biases, LayerScale, and frequency embeddings stay at F32.

## Usage with CrispASR

```bash
# Separate audio into stems
crispasr --separate -m htdemucs-f16.gguf -f input.wav

# Auto-download
crispasr --separate --backend htdemucs --auto-download -f input.wav

# Select specific stems
crispasr --separate -m htdemucs-f16.gguf --stems vocals,drums -f input.wav
```

## Python

```python
import crispasr
s = crispasr.Session("htdemucs-f16.gguf", backend="htdemucs")
stems = s.separate(stereo_pcm_44100hz)
# stems = {"drums": array, "bass": array, "other": array, "vocals": array}
```

## Model Details

- **Architecture**: Hybrid Transformer Demucs (HTDemucs)
- **Parameters**: 42M
- **Input**: Stereo audio at 44100 Hz
- **Output**: 4 stereo stems (drums, bass, other, vocals)
- **Segment length**: 7.8 seconds (with overlap-add for longer audio)
- **License**: MIT (Meta)

### Architecture

- Dual-branch U-Net: frequency (Conv2d on STFT spectrograms) + time (Conv1d on waveforms)
- 4-layer encoder/decoder with skip connections
- 5-layer CrossTransformer at the bottleneck (alternating self-attention and cross-attention between freq and time branches)
- DConv residual blocks (dilated convolutions) in the encoder
- Complex-as-Channels (CaC) for spectrogram processing
- Frequency embeddings (ScaledEmbedding with smooth initialization)

### Conversion

Converted from the official `htdemucs` pretrained model using:
```bash
python models/convert-htdemucs-to-gguf.py --model htdemucs --output htdemucs-f16.gguf --dtype f16
crispasr-quantize htdemucs-f16.gguf htdemucs-q8_0.gguf q8_0
crispasr-quantize htdemucs-f16.gguf htdemucs-q4_k.gguf q4_k
```
