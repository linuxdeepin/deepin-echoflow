---
license: cc-by-4.0
base_model: FermionResearch/Phonon-2
base_model_relation: quantized
language:
  - en
tags:
  - automatic-speech-recognition
  - gguf
  - crispasr
  - parakeet
pipeline_tag: automatic-speech-recognition
---

# Phonon-2 GGUF

GGUF conversions of [Fermion Research's Phonon-2](https://huggingface.co/FermionResearch/Phonon-2)
for [CrispASR](https://github.com/CrispStrobe/CrispASR), using its existing
Parakeet TDT backend. Phonon-2 is an English speech recognizer derived from
NVIDIA Parakeet TDT 0.6B v3, with retrained five-value encoder weights and
6-bit remaining tables.

| File | Size (decimal MB) | Use |
|---|---:|---|
| `phonon2-f16.gguf` | 1,255 | Reference fidelity, ordinary F16 rounding |
| `phonon2-q8_0.gguf` | 674 | Default balance of fidelity and size |
| `phonon2-q4_k.gguf` | 402 | Smaller download; more transcript drift |

These are standard ggml formats. They do **not** preserve the upstream
164 MB transport's custom base-3/five-value packing. Upstream likewise
expands weights at load time in its runtimes; download size is not runtime RAM.

## Usage

With a CrispASR build containing issue #481:

```sh
crispasr -m phonon2 -f recording.wav
crispasr -m phonon2 --model-quant f16 -f recording.wav
crispasr -m phonon2 --model-quant q4_k -f recording.wav
```

Downloaded files also work with the existing Parakeet backend in earlier
compatible CrispASR builds:

```sh
crispasr --backend parakeet -m phonon2-q8_0.gguf -f recording.wav
```

Python bindings auto-detect the GGUF architecture:

```python
import soundfile as sf
from crispasr import Session

audio, rate = sf.read("recording.wav", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)
with Session("phonon2-q8_0.gguf") as session:
    for segment in session.transcribe(audio, sample_rate=rate):
        print(segment.text)
```

## Validation and limits

The converter's expansions matched all 723 source tensors exactly against
Fermion's independent reader. F16 passed all 28 compared activation stages on
JFK and the upstream LibriSpeech sample (minimum cosine 0.999996 / 0.999796;
norm-ratio error bounded by 0.065% / 0.259%).

On 21 English clips through CPU C ABI sessions with an arbitrary model filename:

| Export | Exact reference transcripts | Normalized word edits versus reference |
|---|---:|---:|
| F16 | 21/21 | 0/298 |
| Q8_0 | 19/21 | 1/298 |
| Q4_K | 15/21 | 7/298 |

Reference: stock Transformers `ParakeetForTDT` using the unmodified upstream
container reader. The clips are JFK, the upstream LibriSpeech sample, and every
fourth clip (19 of 73) in `hf-internal-testing/librispeech_asr_dummy`'s clean
validation split. Normalization lowercases and ignores punctuation, retaining
apostrophes inside words. These are edits against the model reference, **not
WER against human ground truth**. Q8 changed one capitalization and one name.
This small check does not reproduce the seven-set Open ASR Leaderboard results.
No CrispASR Metal/CUDA/Vulkan speed or accuracy claim is made here.

## Reproduce conversion

```sh
python models/convert-parakeet-to-gguf.py --hf FermionResearch/Phonon-2 --output phonon2-f16.gguf
crispasr-quantize phonon2-f16.gguf phonon2-q8_0.gguf q8_0
crispasr-quantize phonon2-f16.gguf phonon2-q4_k.gguf q4_k
python tools/dump_reference.py --backend phonon2 --model-dir FermionResearch/Phonon-2 \
  --audio samples/jfk.wav --output phonon2-jfk-ref.gguf
crispasr-diff parakeet phonon2-f16.gguf phonon2-jfk-ref.gguf samples/jfk.wav
```

Source revision: `9c7fef3584499a88fe8d394427f45851bbb8b446`.
Source archive SHA-256: `98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e`.
Expanded container SHA-256: `4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468`.

## Attribution

Phonon-2: Copyright 2026 Fermion Research. Base model: NVIDIA Parakeet TDT
0.6B v3. Weights remain **CC-BY-4.0**. Changes here: exact unpacking of the
Fermion container, tensor renaming/layout conversion to GGUF, F16 rounding,
and additional ggml quantization for Q8/Q4. The original `NOTICE` and weights
license accompany these artifacts. Fermion's container reference code is
Apache-2.0 and is vendored with its license in CrispASR.
