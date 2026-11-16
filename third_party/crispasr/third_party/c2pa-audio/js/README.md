# c2pa-audio (JS)

**Pure-WebCrypto C2PA (Content Credentials) signing & verification for audio — no
native library, no c2pa-rs.** ES256 / CBOR / JUMBF / COSE, interoperable with the
c2pa-rs reference implementation. Runs in browsers, Node ≥16, Deno, and Workers.

```js
import { c2paSignWav, c2paSignMp3, c2paSignM4a, c2paSignFlac } from 'c2pa-audio';
import { c2paVerifyWav } from 'c2pa-audio/verify';

const signed = await c2paSignWav(wavBytes, certPem, keyPem);   // also Mp3/M4a/Flac
const result = await c2paVerifyWav(signed);   // { valid, signatureValid, ... } — auto-detects container
```

Containers: WAV (RIFF chunk), MP3 (ID3v2 GEOB), M4A/MP4 (ISO BMFF, bmff.v3),
FLAC (ID3v2 GEOB). Signing takes a PEM cert + key; verification is self-contained.

Independent project — not affiliated with the C2PA / Content Authenticity
Initiative. MIT. Full docs + C/Dart/Python/Go/C# bindings:
https://github.com/CrispStrobe/c2pa-audio
