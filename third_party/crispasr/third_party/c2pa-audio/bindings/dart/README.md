# c2pa_audio

Dart FFI bindings for [c2pa-audio](https://github.com/CrispStrobe/c2pa-audio) —
native **C2PA (Content Credentials)** signing and verification for **WAV**,
**MP3**, and **M4A/MP4** audio. ES256 / CBOR / JUMBF / COSE, **no c2pa-rs**, interoperable with
the c2pa-rs reference implementation in both directions.

## Native library

These are *bindings* — you provide the native shared library
(`libc2pa_audio.{dylib,so}` / `c2pa_audio.dll`), built from the
[c2pa-audio](https://github.com/CrispStrobe/c2pa-audio) repo with CMake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

Point the binding at it via the `C2PA_AUDIO_LIB` environment variable, or place
it where the OS loader can find it (the package falls back to the platform
default name).

## Usage

```dart
import 'dart:typed_data';
import 'package:c2pa_audio/c2pa_audio.dart';

void main() {
  final c2pa = C2paAudio.open(); // uses C2PA_AUDIO_LIB or the default lib name

  final Uint8List wav = /* your WAV bytes */ Uint8List(0);
  final signed = c2pa.signWav(wav);           // bundled self-signed default cert
  // or: c2pa.signMp3(mp3) / c2pa.signM4a(m4a) / c2pa.signFlac(flac)

  final r = c2pa.verify(signed);               // auto-detects WAV/MP3
  print('valid: ${r.valid}');                  // signature + hash bindings OK
}
```

Pass your own `certPem` / `keyPem` to `sign*` for a custom signer identity.

## Scope

WAV, MP3, and M4A/MP4, sign + verify. Trust-anchor evaluation is out of scope (a self-signed
cert verifies cryptographically but is "untrusted" to a full validator). AAC/Opus are supported when muxed into MP4.

Independent project — not affiliated with the C2PA / Content Authenticity
Initiative. MIT licensed.
