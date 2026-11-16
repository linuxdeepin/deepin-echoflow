#!/usr/bin/env python3
"""Live parity: sign each container with c2pa-audio (via the C ABI / ctypes
binding), then validate the output in the c2pa-rs reference reader (c2pa-python).
Also confirms tampering the audio is rejected. Run in CI with C2PA_AUDIO_LIB set
to the built shared library.
"""
import io
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
from c2pa_audio import C2paAudio  # noqa: E402

from c2pa import Reader  # noqa: E402

HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "..", "test", "assets")


def make_wav(n=4800, sr=24000):
    data = b"".join(struct.pack("<h", int(3000 * (i % 100 - 50) / 50)) for i in range(n))
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


def read_status(signed, mime):
    m = json.loads(Reader(mime, io.BytesIO(signed)).json())
    return [v["code"] for v in m.get("validation_status", [])]


def main():
    c = C2paAudio()
    print("c2pa-audio version", c.version)
    cases = [
        ("wav", make_wav(), "audio/wav"),
    ]
    for f, mime in [("sample.mp3", "audio/mpeg"), ("sample.m4a", "audio/mp4"),
                    ("sample.flac", "audio/flac")]:
        p = os.path.join(ASSETS, f)
        if os.path.exists(p):
            cases.append((f, open(p, "rb").read(), mime))

    failures = 0
    for name, raw, mime in cases:
        signed = c.sign(raw, mime)
        status = read_status(signed, mime)
        bad = [s for s in status if s != "signingCredential.untrusted"]
        if bad:
            print(f"FAIL {name}: c2pa-rs reported {bad}")
            failures += 1
        else:
            print(f"ok   {name}: our signer -> c2pa-rs reader VALID ({len(raw)} -> {len(signed)})")
        # tamper the audio payload -> c2pa-rs must flag a hash mismatch
        t = bytearray(signed)
        t[-64] ^= 0xFF
        tstatus = read_status(bytes(t), mime)
        if not any("mismatch" in s for s in tstatus):
            print(f"FAIL {name}: tamper not detected by c2pa-rs ({tstatus})")
            failures += 1
        else:
            print(f"ok   {name}: tamper rejected by c2pa-rs")

    print("PASSED" if not failures else f"FAILED ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
