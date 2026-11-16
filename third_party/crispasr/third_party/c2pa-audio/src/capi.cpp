// capi.cpp — C ABI implementation, wrapping the C++ signer/verifier.
#include "c2pa_audio.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "c2pa_native.h"
#include "default_cert.h"

using crispasr::c2pa_native::Bytes;
using crispasr::c2pa_native::sign_flac;
using crispasr::c2pa_native::sign_m4a;
using crispasr::c2pa_native::sign_mp3;
using crispasr::c2pa_native::sign_wav;
using crispasr::c2pa_native::verify_wav;
using crispasr::c2pa_native::VerifyResult;

extern "C" int c2pa_audio_sign(const unsigned char* in, size_t in_len, const char* mime, const char* cert_pem,
                               const char* key_pem, unsigned char** out, size_t* out_len) {
    if (!in || !out || !out_len)
        return 1;
    *out = nullptr;
    *out_len = 0;
    try {
        std::string cert = cert_pem ? std::string(cert_pem) : std::string(c2pa_audio_default_cert_pem());
        std::string key = key_pem ? std::string(key_pem) : std::string(c2pa_audio_default_key_pem());
        Bytes data(in, in + in_len);
        std::string fmt = mime ? std::string(mime) : std::string("audio/wav");
        Bytes signed_;
        if (fmt == "audio/mpeg" || fmt == "audio/mp3")
            signed_ = sign_mp3(data, cert, key);
        else if (fmt == "audio/mp4" || fmt == "audio/m4a" || fmt == "audio/x-m4a")
            signed_ = sign_m4a(data, cert, key);
        else if (fmt == "audio/flac" || fmt == "audio/x-flac")
            signed_ = sign_flac(data, cert, key);
        else if (fmt == "audio/wav" || fmt == "audio/x-wav" || fmt == "audio/wave")
            signed_ = sign_wav(data, cert, key);
        else
            return 4; // unsupported container
        if (signed_.empty())
            return 2;
        unsigned char* buf = static_cast<unsigned char*>(std::malloc(signed_.size()));
        if (!buf)
            return 3;
        std::memcpy(buf, signed_.data(), signed_.size());
        *out = buf;
        *out_len = signed_.size();
        return 0;
    } catch (...) {
        // Never let a C++ exception unwind into a C / FFI caller — that is UB
        // and takes the host process down. Report allocation failure instead.
        *out = nullptr;
        *out_len = 0;
        return 3;
    }
}

extern "C" int c2pa_audio_verify(const unsigned char* in, size_t in_len) {
    if (!in)
        return 0;
    try {
        Bytes data(in, in + in_len);
        VerifyResult r = verify_wav(data); // auto-detects WAV/MP3
        int flags = 0;
        if (r.signature_valid)
            flags |= C2PA_AUDIO_SIG_VALID;
        if (r.data_hash_valid)
            flags |= C2PA_AUDIO_DATA_VALID;
        if (r.assertions_valid)
            flags |= C2PA_AUDIO_ASSERT_VALID;
        if (r.valid)
            flags |= C2PA_AUDIO_VALID;
        return flags;
    } catch (...) {
        // A hostile file must verify as "not valid", never abort the caller.
        return 0;
    }
}

extern "C" void c2pa_audio_free(unsigned char* p) { std::free(p); }

extern "C" const char* c2pa_audio_version(void) { return "0.1.0"; }
