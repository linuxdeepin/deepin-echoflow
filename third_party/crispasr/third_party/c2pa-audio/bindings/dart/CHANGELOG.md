## 0.2.0

- M4A/MP4 (ISO BMFF, c2pa.hash.bmff.v3) and FLAC (ID3v2 GEOB) sign + verify, interoperable with c2pa-rs.
- `signM4a` / `signFlac` conveniences; `verify` auto-detects WAV/MP3/M4A/FLAC.

## 0.1.0

- Initial release: FFI bindings for the native c2pa-audio library.
- Sign and verify C2PA Content Credentials in WAV and MP3 audio.
- `signWav` / `signMp3` / `sign(mime: ...)` and `verify` (auto-detects container).
- Interoperable with the c2pa-rs reference implementation.
