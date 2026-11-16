---
license: cc-by-4.0
language:
- en
- de
base_model:
- Helsinki-NLP/opus-mt-en-de
pipeline_tag: translation
tags:
- translation
- marian
- opus-mt
- encoder-decoder
- gguf
- crispasr
library_name: ggml
---

# Opus-MT English → German — GGUF (ggml)

GGUF / ggml conversion of [`Helsinki-NLP/opus-mt-en-de`](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) for use with **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)** (`--backend marian`).

A small (~75M parameters, 6+6 layers, d=512) MarianMT model for one language pair. In CrispASR it is the fastest translator for live transcription + translation (`--live-translate`): tens of milliseconds per sentence on a laptop CPU.

## Licence and attribution

**CC-BY-4.0.** The weights are the `opus-2020-02-26` release of the [OPUS-MT project](https://github.com/Helsinki-NLP/Opus-MT) (Jörg Tiedemann and Santhosh Thottingal, University of Helsinki; *OPUS-MT — Building open translation services for the World*, EAMT 2020), trained on [OPUS](https://opus.nlpl.eu/) data. The project distributes its pre-trained models under CC-BY 4.0 ([Opus-MT README](https://github.com/Helsinki-NLP/Opus-MT), [OPUS-MT-train README](https://github.com/Helsinki-NLP/OPUS-MT-train)); that statement, not the tag on an individual model card, is what this repository follows. This is a format conversion: the weights are unchanged at f16 and quantized at q8_0. Redistribution requires attribution to the OPUS-MT project.

## Files

| File | Size | Output vs. the reference implementation |
|---|---:|---|
| `opus-mt-en-de-f16.gguf` | 153 MB | identical: 8/8 sentences greedy, 8/8 with beam 4 |
| `opus-mt-en-de-q8_0.gguf` | 84 MB | 8/8 sentences greedy, 7/8 with beam 4 — the others differ in wording, not in meaning. Fastest; recommended. |

"Reference" is `MarianMTModel.generate` from Hugging Face `transformers` on the original checkpoint; token ids are identical for every sentence and for 24 tokenizer stress strings. A q4_k file is deliberately not published: it changed about half the test sentences.

## Quick start

```bash
git clone https://github.com/CrispStrobe/CrispASR && cd CrispASR
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Text → text. The backend is detected from the file; -bs 1 is greedy,
# without it the checkpoint's own beam size (4) is used.
./build/bin/crispasr --backend marian -m auto -sl en -tl de -bs 1 \
    --text "Please remember to pack your documents in time."
# Bitte denken Sie daran, Ihre Dokumente rechtzeitig zu packen.

# Live: microphone in, transcript + translation out, sentence by sentence
./build/bin/crispasr --live-translate -l en --tr-tl de \
    -m auto --backend parakeet --translate-model opus-mt-en-de
```

`-m auto` / `--translate-model opus-mt-en-de` download `opus-mt-en-de-q8_0.gguf` on first use.

## Notes

- One model per direction; the opposite direction is [`cstr/opus-mt-de-en-GGUF`](https://huggingface.co/cstr/opus-mt-de-en-GGUF).
- Beam search in this runtime has no KV cache and is several times slower than greedy; live mode always decodes greedy.
- Literal `</s>`, `<unk>`, `<pad>` in the input are treated as ordinary text here (the reference treats them as special tokens).

## Conversion

```bash
python models/convert-marian-to-gguf.py --input <dir>/opus-mt-en-de --output opus-mt-en-de-f16.gguf
./build/bin/crispasr-quantize opus-mt-en-de-f16.gguf opus-mt-en-de-q8_0.gguf q8_0
python tools/marian_parity.py --hf-dir <dir>/opus-mt-en-de --gguf opus-mt-en-de-f16.gguf --lang en
```
