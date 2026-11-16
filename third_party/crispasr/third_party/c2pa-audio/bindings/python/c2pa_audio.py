"""c2pa_audio — Python ctypes binding for the standalone C2PA signer/verifier.

Loads libc2pa_audio and exposes sign_wav / verify_wav. No c2pa-rs.

    from c2pa_audio import C2paAudio
    c = C2paAudio()                       # uses C2PA_AUDIO_LIB or default name
    signed = c.sign_wav(wav_bytes)        # -> bytes (bundled default cert)
    r = c.verify_wav(signed)              # -> VerifyResult(valid=True, ...)
"""
import ctypes
import os
import platform
from dataclasses import dataclass

SIG_VALID, DATA_VALID, ASSERT_VALID, VALID = 0x1, 0x2, 0x4, 0x8


def _default_libname() -> str:
    s = platform.system()
    if s == "Darwin":
        return "libc2pa_audio.dylib"
    if s == "Windows":
        return "c2pa_audio.dll"
    return "libc2pa_audio.so"


@dataclass
class VerifyResult:
    signature_valid: bool
    data_hash_valid: bool
    assertions_valid: bool
    valid: bool


class C2paAudio:
    def __init__(self, path: str | None = None):
        path = path or os.environ.get("C2PA_AUDIO_LIB") or _default_libname()
        lib = ctypes.CDLL(path)
        self._lib = lib
        lib.c2pa_audio_sign.restype = ctypes.c_int
        lib.c2pa_audio_sign.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.c2pa_audio_verify.restype = ctypes.c_int
        lib.c2pa_audio_verify.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.c2pa_audio_free.restype = None
        lib.c2pa_audio_free.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
        lib.c2pa_audio_version.restype = ctypes.c_char_p

    @property
    def version(self) -> str:
        return self._lib.c2pa_audio_version().decode()

    def sign(self, data: bytes, mime: str = "audio/wav", cert_pem: str | None = None,
             key_pem: str | None = None) -> bytes:
        out = ctypes.POINTER(ctypes.c_ubyte)()
        out_len = ctypes.c_size_t(0)
        rc = self._lib.c2pa_audio_sign(
            data, len(data), mime.encode(),
            cert_pem.encode() if cert_pem else None,
            key_pem.encode() if key_pem else None,
            ctypes.byref(out), ctypes.byref(out_len),
        )
        if rc != 0:
            raise RuntimeError(f"c2pa_audio_sign failed (rc={rc})")
        try:
            return bytes(ctypes.cast(out, ctypes.POINTER(ctypes.c_ubyte * out_len.value)).contents)
        finally:
            self._lib.c2pa_audio_free(out)

    def sign_wav(self, wav: bytes, cert_pem=None, key_pem=None) -> bytes:
        return self.sign(wav, "audio/wav", cert_pem, key_pem)

    def sign_mp3(self, mp3: bytes, cert_pem=None, key_pem=None) -> bytes:
        return self.sign(mp3, "audio/mpeg", cert_pem, key_pem)

    def sign_m4a(self, m4a: bytes, cert_pem=None, key_pem=None) -> bytes:
        return self.sign(m4a, "audio/mp4", cert_pem, key_pem)

    def sign_flac(self, flac: bytes, cert_pem=None, key_pem=None) -> bytes:
        return self.sign(flac, "audio/flac", cert_pem, key_pem)

    def verify(self, data: bytes) -> VerifyResult:
        f = self._lib.c2pa_audio_verify(data, len(data))
        return VerifyResult(bool(f & SIG_VALID), bool(f & DATA_VALID), bool(f & ASSERT_VALID), bool(f & VALID))
