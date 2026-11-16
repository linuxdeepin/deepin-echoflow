# Fuzzing `crispasr_audio_load`

`crispasr_audio_load()` runs an attacker-controllable file through the whole
decoder dispatch — miniaudio (WAV/MP3/FLAC/OGG) plus the hand-rolled Sun-AU,
AMR, WebM/EBML and MP4 fallbacks. Those hand-rolled parsers are the audio
attack surface; the harness (`fuzz_audio_load.cpp`) fuzzes the one entry point
that reaches all of them.

## Build (clang required)

```bash
cmake -B build-fuzz \
  -DCRISPASR_FUZZ=ON \
  -DCRISPASR_SANITIZE_ADDRESS=ON \
  -DCRISPASR_SANITIZE_UNDEFINED=ON \
  -DCRISPASR_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-fuzz --target crispasr-fuzz-audio
```

`-DCRISPASR_FUZZ=ON` adds `-fsanitize=fuzzer-no-link` (coverage) to every TU;
the harness target adds `-fsanitize=fuzzer` (the libFuzzer driver). Combine
with the sanitizers so out-of-bounds accesses abort instead of reading garbage.

## Run

Seed from the committed samples (valid inputs for every format the loader
handles), so the mutator starts from real container structure:

```bash
mkdir -p /tmp-nonsmall/crispasr-fuzz-corpus   # any writable dir off a real fs
cp samples/jfk.wav samples/jfk.au samples/jfk.amr samples/jfk.webm \
   samples/jfk-vorbis.webm samples/jfk.m4a /tmp-nonsmall/crispasr-fuzz-corpus/

./build-fuzz/bin/crispasr-fuzz-audio -max_len=1048576 /tmp-nonsmall/crispasr-fuzz-corpus
```

A crash writes `crash-<hash>`; reproduce with
`./build-fuzz/bin/crispasr-fuzz-audio crash-<hash>`. The harness writes each
input to `crispasr_fuzz_input.bin` in the cwd and caps size at 8 MB (the
parsers have their own 500 MB caps; the cap just keeps the fuzzer itself fast).

## What it exercises

Every format-detection + fallback path in `crispasr_audio_load`, i.e. the
`crispasr_au_decode` / `crispasr_amr_decode` / `crispasr_webm_decode` /
`crispasr_m4a_decode` parsers plus miniaudio's WAV/MP3/FLAC/OGG handling.

## GGUF metadata harness (`crispasr-fuzz-gguf`)

`fuzz_gguf_meta.cpp` fuzzes GGUF **model-file** parsing — a model is untrusted
input too. It drives `core_gguf::open_metadata()` plus the KV/array accessors
(`kv_str_array` is the vocab path: count + per-string lengths straight from the
file). GGUF has a magic + structured header, so **seed from a real small
`.gguf`** or the fuzzer never gets past the magic:

```bash
cmake --build build-fuzz --target crispasr-fuzz-gguf
mkdir -p gguf-corpus && cp /path/to/small-model.gguf gguf-corpus/
./build-fuzz/bin/crispasr-fuzz-gguf -max_len=2097152 gguf-corpus
```

The audio harness is the one wired into CI (`linux-fuzz-smoke`) because it has
committed seeds (`samples/`); the GGUF harness needs a model seed, so run it out
of band.

## Text tokenizer harness (`crispasr-fuzz-tokenizer`)

`fuzz_tokenizer.cpp` fuzzes the shared text tokenizers `core_bpe::tokenize_simple`
(GPT-2 byte-level BPE) and `core_wordpiece::Tokenizer::tokenize` (BERT WordPiece)
over arbitrary **text** — the untrusted prompt / `--ref-text` / caption surface.
Vocab/merges are pinned benign (the model-supplied vocab is covered by the GGUF
harness); the fuzzed bytes drive the byte→unicode map, the whitespace/punctuation
pre-tokenizer, the BPE merge loop, and the WordPiece greedy match on invalid
UTF-8 / lone continuation bytes / embedded NULs. No seed needed (any bytes are
valid text):

```bash
cmake --build build-fuzz --target crispasr-fuzz-tokenizer
./build-fuzz/bin/crispasr-fuzz-tokenizer -max_len=65536 tok-corpus
```

Clean at ~139K runs/16 s under `-fsanitize=fuzzer,address,undefined` (2026-07-12).
