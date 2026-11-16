// c2pa_audio_native.h — native C++ C2PA signer for WAV (no c2pa-rs / Rust).
//
// The C++ twin of bindings/javascript/c2pa.mjs. Hand-builds canonical CBOR, the
// JUMBF box tree, COSE_Sign1, and RIFF chunk embedding; signs with ES256
// (ECDSA P-256 + SHA-256) using the vendored BSD-2 micro-ecc (deterministic
// RFC 6979 — no RNG needed for the signature) and a header-only SHA-256. The
// output validates in the c2pa-rs reference reader (only status:
// signingCredential.untrusted for a self-signed cert). WAV only — matches the
// TTS output path; MP3/M4A stay on the native c2pa-rs lib when it is present.
//
// This header is a C++11-clean DECLARATION only (so it can be included from the
// C++11 crispasr-lib TUs). The implementation lives in c2pa_audio_native.cpp
// and is compiled at C++17.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace crispasr {
namespace c2pa_native {

using Bytes = std::vector<uint8_t>;

struct SignOptions {
    std::string generator_name = "CrispASR";
    std::string generator_version = "0.6";
    std::string software_agent = "CrispASR TTS";
};

// Sign a WAV container with a C2PA manifest. cert_pem/key_pem: PEM strings
// (X.509 P-256 cert + PKCS#8 or SEC1 EC private key). Returns the signed WAV,
// or an empty vector on any failure (caller keeps the unsigned WAV — still
// watermarked / metadata-tagged upstream).
Bytes sign_wav(const Bytes& wav, const std::string& cert_pem, const std::string& key_pem,
               const SignOptions& opts = SignOptions());

// Sign an MP3 by embedding the manifest in an ID3v2.4 GEOB frame. Same manifest
// as WAV; only the container differs. Returns the signed MP3 or empty on failure.
Bytes sign_mp3(const Bytes& mp3, const std::string& cert_pem, const std::string& key_pem,
               const SignOptions& opts = SignOptions());

// Sign an M4A/MP4 (ISO BMFF) by inserting a C2PA 'uuid' box and binding with a
// c2pa.hash.bmff.v3 assertion. Returns the signed file or empty on failure.
Bytes sign_m4a(const Bytes& m4a, const std::string& cert_pem, const std::string& key_pem,
               const SignOptions& opts = SignOptions());

// Sign a FLAC by prepending an ID3v2.4 GEOB manifest tag — the same container
// mechanism c2pa-rs uses for FLAC (identical to the MP3 path; the fLaC audio is
// preserved untouched after the tag). Returns the signed file or empty.
Bytes sign_flac(const Bytes& flac, const std::string& cert_pem, const std::string& key_pem,
                const SignOptions& opts = SignOptions());

// Result of verifying a signed WAV's C2PA manifest.
struct VerifyResult {
    bool valid = false;            // signature + data hash + assertions all OK
    bool signature_valid = false;  // COSE ES256 verified vs the embedded cert
    bool data_hash_valid = false;  // hard binding matches the audio
    bool assertions_valid = false; // every claimed assertion hash matches
    bool trusted = false;          // always false here (no trust-anchor check)
    std::string generator_name;    // claim_generator_info.name (if present)
    std::vector<std::string> errors;
};

// Verify a C2PA-signed WAV natively (no c2pa-rs): parse the JUMBF tree, verify
// the COSE_Sign1 ES256 signature against the embedded certificate's public key
// (uECC), and recompute the hard-binding data hash + assertion hashedURIs.
VerifyResult verify_wav(const Bytes& wav);

} // namespace c2pa_native
} // namespace crispasr
