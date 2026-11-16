# c2pa-audio

**Native C2PA (Content Credentials) signing & verification for WAV, MP3, M4A + FLAC audio — no c2pa-rs, no Rust, no OpenSSL.**

[![CI](https://github.com/CrispStrobe/c2pa-audio/actions/workflows/ci.yml/badge.svg)](https://github.com/CrispStrobe/c2pa-audio/actions/workflows/ci.yml)


A tiny (~160 KB), dependency-light implementation of the [C2PA](https://c2pa.org)
manifest format: canonical CBOR, JUMBF boxes, COSE_Sign1, and **ES256**
(ECDSA P-256 + SHA-256, via the vendored [micro-ecc](https://github.com/kmackay/micro-ecc)).
It signs and verifies Content Credentials that are **fully interoperable with the
c2pa-rs reference implementation** — in both directions.

It began as the C2PA layer inside [CrispASR](https://github.com/CrispStrobe/CrispASR)
(marking AI-generated TTS output as `c2pa.created` / `trainedAlgorithmicMedia`),
extracted here as a standalone library because it needs none of the ASR stack.

## Why

The official c2pa-rs is excellent but heavy (~10 MB) — awkward for the browser
(wasm) and for embedding in mobile/edge apps. This library is a clean-room
reimplementation of the published C2PA spec + RFCs, small enough to compile
anywhere (desktop, mobile, **wasm**), with a pure-JS variant needing no native
code at all.

## Interop matrix (all verified)

|                     | verify: c2pa-rs | verify: our JS | verify: our C++ |
|---------------------|:---------------:|:--------------:|:---------------:|
| **sign: c2pa-rs**   | ✅ ref          | ✅             | ✅              |
| **sign: our JS**    | ✅              | ✅             | ✅              |
| **sign: our C++**   | ✅              | ✅             | ✅              |
| **sign: our wasm**  | ✅              | ✅             | ✅              |

Every signer's output validates in every verifier for **WAV, MP3, M4A, and FLAC**,
including the c2pa-rs reference reader; tampering the audio breaks the hard-binding hash, and a
mangled signature fails COSE verification.

## Languages

| Language        | How              | Location                   | Status |
|-----------------|------------------|----------------------------|--------|
| C / C++         | link the lib     | `include/c2pa_audio.h`  | ✅ tested |
| JavaScript / TS | pure WebCrypto   | `js/c2pa.mjs`, `js/c2pa-verify.mjs` | ✅ tested |
| Dart / Flutter  | `dart:ffi`       | `bindings/dart/`           | ✅ tested |
| Python          | `ctypes`         | `bindings/python/`         | ✅ tested |
| Go              | `cgo`            | `bindings/go/`             | ✅ tested |
| C#              | P/Invoke         | `bindings/csharp/`         | ⚠️ code (no local dotnet) |

The JS variant is special: `js/c2pa.mjs` + `js/c2pa-verify.mjs` are **pure
WebCrypto** and need no native library — ideal for browsers and Workers.

## Demo

`demo/index.html` is a self-contained, dependency-free page that signs, inspects,
verifies, and tamper-tests audio entirely in the browser using the pure-JS
library. Open the file directly, or regenerate it from source with
`node demo/build.mjs`. See `demo/README.md`.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure     # C ABI round-trip + reference vector
```

Produces `libc2pa_audio.{dylib,so,dll}` (shared, for FFI) and `.a` (static).

## C ABI

```c
#include "c2pa_audio.h"

// sign (NULL cert/key -> bundled self-signed default cert)
unsigned char* out; size_t out_len;
c2pa_audio_sign(wav, wav_len, "audio/wav", NULL, NULL, &out, &out_len);

// verify -> bit flags (0xF == fully valid)
int flags = c2pa_audio_verify(out, out_len);

c2pa_audio_free(out);
```

## Quick starts

**JavaScript (no native lib):**
```js
import { c2paSignWav } from './js/c2pa.mjs';
import { c2paVerifyWav } from './js/c2pa-verify.mjs';
const signed = await c2paSignWav(wav, certPem, keyPem);
const result = await c2paVerifyWav(signed);   // { valid, signatureValid, ... }
```

**Dart:**
```dart
final c2pa = C2paAudio.open();               // loads libc2pa_audio
final signed = c2pa.signWav(wav);            // bundled default cert
final r = c2pa.verify(signed);               // r.valid == true
```

**Python / Go / C#** — see `bindings/`.

## Scope

**WAV** (RIFF `C2PA` chunk), **MP3** (ID3v2.4 GEOB frame), **M4A/MP4**
(ISO BMFF `uuid` box + `c2pa.hash.bmff.v3`), and **FLAC** (ID3v2 GEOB prepend,
like c2pa-rs), sign + verify — all fully interoperable with c2pa-rs. Trust-anchor evaluation is out of scope (a
self-signed cert verifies cryptographically but is "untrusted" to a full
validator).

The BMFF path is **codec-agnostic**, so **AAC-in-MP4 and Opus-in-MP4** work out
of the box (pass `audio/mp4`). Only the *raw streaming* containers — ADTS AAC
(`.aac`) and Ogg Opus (`.opus`/`.ogg`) — have no C2PA embedding path; neither
does c2pa-rs (it refuses `audio/aac` / `audio/ogg`). Remux those into MP4 to
sign them.

Signing API takes a MIME type: `"audio/wav"`, `"audio/mpeg"`, `"audio/mp4"`, or `"audio/flac"`.
Verification auto-detects the container (RIFF / ID3 / ISO-BMFF).

## Licensing

- **This code**: MIT (see `LICENSE`). It is an original, clean-room
  implementation of the *published* C2PA spec + RFCs (CBOR 8949, COSE 9052,
  JUMBF ISO 19566-5) — it does **not** derive from c2pa-rs source.
- **Vendored micro-ecc** (`third_party/uecc/`): BSD-2-Clause.
- **SHA-256** (`src/sha256.h`): public domain.
- C2PA is an open standard with a royalty-free patent policy for conformant
  implementations. This is not legal advice — do your own IP review before
  shipping.
