// crispasr_c2pa.h — C2PA (Content Credentials) manifest signing for TTS output.
//
// Compile-time gated on CRISPASR_HAVE_C2PA. When enabled, signs synthesized
// audio (WAV / MP3 / M4A / FLAC — whatever the c2pa runtime supports) with a
// C2PA manifest declaring AI-generated provenance, in memory (no temp files).
//
// When CRISPASR_HAVE_C2PA is not defined, the functions are no-ops that return
// false and leave the buffer unchanged (the watermark + container metadata tag
// still provide provenance).
//
// Targets the c2pa-rs C ABI (c2pa.h from the prebuilt native lib, v0.89+):
//   c2pa_signer_from_info() / c2pa_builder_from_json() / c2pa_builder_sign()
//   with callback-based streams (c2pa_create_stream). Sign in memory over a
//   std::string buffer for any supported format.
//
// Certificate: a self-signed X.509 (P-256 / ES256) is sufficient for
// machine-readable AI marking (EU AI Act Art. 50). C2PA verifiers show
// "unverified signer" for self-signed certs; the manifest is still valid.
// Generate with scripts/generate-c2pa-cert.sh, or let CrispASR auto-provision a
// per-install self-signed cert (see crispasr_c2pa_autocert() below).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "crispasr_c2pa_default_cert.h" // bundled self-signed default cert (baked in)

// Native pure-C++ C2PA signer for WAV (no c2pa-rs). Always available; it is the
// primary path for WAV on every platform. Define CRISPASR_NO_C2PA_NATIVE to opt
// out (e.g. a build that only wants MP3/M4A via c2pa-rs).
#ifndef CRISPASR_NO_C2PA_NATIVE
#include "crispasr_c2pa_native.h"
#endif

#ifdef CRISPASR_HAVE_C2PA
#include <c2pa.h>
#endif

// The C2PA manifest JSON. digitalSourceType per IPTC vocabulary marks this as
// trained-algorithmic (AI-generated) media.
inline const char* crispasr_c2pa_manifest_json() {
    return R"({
  "claim_generator": "CrispASR",
  "claim_generator_info": [{
    "name": "CrispASR",
    "version": "0.6"
  }],
  "assertions": [{
    "label": "c2pa.actions",
    "data": {
      "actions": [{
        "action": "c2pa.created",
        "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        "softwareAgent": "CrispASR TTS"
      }]
    }
  }]
})";
}

// Map an output file extension (lowercased, no dot) to the C2PA MIME/format
// string, or "" if c2pa cannot embed a manifest in that container. c2pa-rs
// supports WAV/MP3/M4A/MP4/FLAC for audio; it does NOT support Ogg/Opus or raw
// ADTS AAC, so those return "" (caller skips signing, keeps watermark + tag).
inline const char* crispasr_c2pa_format_for_ext(const std::string& ext) {
    if (ext == "wav")
        return "audio/wav";
    if (ext == "mp3")
        return "audio/mpeg";
    if (ext == "m4a")
        return "audio/mp4";
    if (ext == "mp4")
        return "audio/mp4";
    if (ext == "flac")
        return "audio/flac";
    return ""; // aac (adts), opus/ogg — not embeddable by c2pa
}

// read_file is always available (needs only <fstream>) so the native WAV path
// can load a user cert/key without c2pa-rs compiled in.
namespace crispasr_c2pa_detail {
inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
} // namespace crispasr_c2pa_detail

#ifdef CRISPASR_HAVE_C2PA

namespace crispasr_c2pa_detail {

// In-memory stream backing for the c2pa callback stream API. `buf` is owned by
// the caller; `pos` is the stream cursor.
struct membuf {
    std::string* buf;
    size_t pos;
};

inline intptr_t mem_read(StreamContext* ctx, uint8_t* data, intptr_t len) {
    auto* m = reinterpret_cast<membuf*>(ctx);
    if (len < 0)
        return -1;
    size_t avail = m->buf->size() > m->pos ? m->buf->size() - m->pos : 0;
    size_t n = std::min(static_cast<size_t>(len), avail);
    if (n)
        std::memcpy(data, m->buf->data() + m->pos, n);
    m->pos += n;
    return static_cast<intptr_t>(n);
}

inline intptr_t mem_seek(StreamContext* ctx, intptr_t offset, C2paSeekMode mode) {
    auto* m = reinterpret_cast<membuf*>(ctx);
    intptr_t base = (mode == Start)     ? 0
                    : (mode == Current) ? static_cast<intptr_t>(m->pos)
                                        : static_cast<intptr_t>(m->buf->size());
    intptr_t np = base + offset;
    if (np < 0)
        np = 0;
    m->pos = static_cast<size_t>(np);
    return static_cast<intptr_t>(m->pos);
}

inline intptr_t mem_write(StreamContext* ctx, const uint8_t* data, intptr_t len) {
    auto* m = reinterpret_cast<membuf*>(ctx);
    if (len < 0)
        return -1;
    if (m->pos + static_cast<size_t>(len) > m->buf->size())
        m->buf->resize(m->pos + static_cast<size_t>(len));
    if (len)
        std::memcpy(&(*m->buf)[m->pos], data, static_cast<size_t>(len));
    m->pos += static_cast<size_t>(len);
    return len;
}

inline intptr_t mem_flush(StreamContext*) {
    return 0;
}

} // namespace crispasr_c2pa_detail

#endif // CRISPASR_HAVE_C2PA

// Sign an in-memory audio buffer with a C2PA manifest from in-memory PEM
// content (no filesystem — works in the browser/mobile sandbox). `format` is a
// C2PA MIME string (see crispasr_c2pa_format_for_ext). On success `data` is
// replaced with the signed asset and true is returned; false (data unchanged)
// when C2PA is unavailable, inputs are empty, the format is unsupported, or
// signing fails.
inline bool crispasr_c2pa_sign_pem(std::string& data, const char* format, const std::string& cert_pem,
                                   const std::string& key_pem) {
    if (!format || !*format || cert_pem.empty() || key_pem.empty())
        return false;

#ifndef CRISPASR_NO_C2PA_NATIVE
    // Native path — pure C++ (uECC ES256 + hand-built CBOR/JUMBF/COSE) from the
    // vendored c2pa-audio lib, no c2pa-rs; works on every platform including the
    // wasm/mobile sandbox. WAV (RIFF chunk), MP3 (ID3v2 GEOB), M4A/MP4 (ISO BMFF,
    // c2pa.hash.bmff.v3), and FLAC (ID3v2 GEOB prepend) are ALL native — c2pa-rs
    // is no longer needed for any audio container CrispASR emits.
    {
        crispasr::c2pa_native::Bytes in(data.begin(), data.end());
        crispasr::c2pa_native::Bytes out;
        if (std::strcmp(format, "audio/wav") == 0)
            out = crispasr::c2pa_native::sign_wav(in, cert_pem, key_pem);
        else if (std::strcmp(format, "audio/mpeg") == 0)
            out = crispasr::c2pa_native::sign_mp3(in, cert_pem, key_pem);
        else if (std::strcmp(format, "audio/mp4") == 0)
            out = crispasr::c2pa_native::sign_m4a(in, cert_pem, key_pem);
        else if (std::strcmp(format, "audio/flac") == 0)
            out = crispasr::c2pa_native::sign_flac(in, cert_pem, key_pem);
        if (!out.empty()) {
            data.assign(reinterpret_cast<const char*>(out.data()), out.size());
            return true;
        }
        // fall through to c2pa-rs (if compiled in) on native failure / other formats
    }
#endif

#ifdef CRISPASR_HAVE_C2PA
    using namespace crispasr_c2pa_detail;

    // c2pa 0.89 marks c2pa_builder_from_json / c2pa_signer_free deprecated in
    // favor of newer context/free APIs; the deprecated ones remain functional
    // and are the stable path across the prebuilt lib versions we pin. Silence
    // the warning locally so it doesn't trip a -Werror build.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    const std::string& cert = cert_pem;
    const std::string& key = key_pem;

    // Self-signed P-256 EC cert (ES256).
    C2paSignerInfo info;
    info.alg = "es256";
    info.sign_cert = cert.c_str();
    info.private_key = key.c_str();
    info.ta_url = nullptr;
    C2paSigner* signer = c2pa_signer_from_info(&info);
    if (!signer) {
        fprintf(stderr, "crispasr: C2PA signer init failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        return false;
    }

    C2paBuilder* builder = c2pa_builder_from_json(crispasr_c2pa_manifest_json());
    if (!builder) {
        fprintf(stderr, "crispasr: C2PA builder init failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        c2pa_signer_free(signer);
        return false;
    }

    membuf src{&data, 0};
    std::string out;
    membuf dst{&out, 0};
    C2paStream* ss =
        c2pa_create_stream(reinterpret_cast<StreamContext*>(&src), mem_read, mem_seek, mem_write, mem_flush);
    C2paStream* ds =
        c2pa_create_stream(reinterpret_cast<StreamContext*>(&dst), mem_read, mem_seek, mem_write, mem_flush);

    const unsigned char* manifest_bytes = nullptr;
    int64_t rc = c2pa_builder_sign(builder, format, ss, ds, signer, &manifest_bytes);
    if (manifest_bytes)
        c2pa_manifest_bytes_free(manifest_bytes);
    c2pa_release_stream(ss);
    c2pa_release_stream(ds);
    c2pa_builder_free(builder);
    c2pa_signer_free(signer);

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    if (rc < 0) {
        fprintf(stderr, "crispasr: C2PA signing failed (%s): %s\n", format, c2pa_error() ? c2pa_error() : "unknown");
        return false;
    }
    data.swap(out);
    return true;
#else
    (void)data;
    (void)cert_pem;
    (void)key_pem;
    return false;
#endif
}

// Sign from PEM FILE PATHS (reads the files, then delegates to sign_pem).
inline bool crispasr_c2pa_sign_buffer(std::string& data, const char* format, const std::string& cert_path,
                                      const std::string& key_path) {
    if (!format || !*format || cert_path.empty() || key_path.empty())
        return false;
    // Reading the PEM files needs only <fstream> (always available); the native
    // WAV path then works even without c2pa-rs. read_file is defined regardless
    // of CRISPASR_HAVE_C2PA (see crispasr_c2pa_detail below).
    std::string cert = crispasr_c2pa_detail::read_file(cert_path);
    std::string key = crispasr_c2pa_detail::read_file(key_path);
    if (cert.empty() || key.empty()) {
        fprintf(stderr, "crispasr: C2PA cert/key unreadable ('%s' / '%s')\n", cert_path.c_str(), key_path.c_str());
        return false;
    }
    return crispasr_c2pa_sign_pem(data, format, cert, key);
}

// Back-compat WAV wrapper (existing call sites).
inline bool crispasr_c2pa_sign_wav(std::string& wav, const std::string& cert_path, const std::string& key_path) {
    return crispasr_c2pa_sign_buffer(wav, "audio/wav", cert_path, key_path);
}

// Sign `data` (format = C2PA MIME) with the EFFECTIVE credentials: the user's
// --c2pa-cert/--c2pa-key (file paths) if given, otherwise the built-in bundled
// self-signed default cert (on-by-default provenance on EVERY platform — the
// bundled cert is baked in, so this works in the browser/mobile sandbox with no
// filesystem or openssl). Best-effort — returns false (data unchanged) if C2PA
// is unavailable, the format is unsupported, or signing fails.
inline bool crispasr_c2pa_sign_auto(std::string& data, const char* format, const std::string& user_cert,
                                    const std::string& user_key, const std::string& cache_dir) {
    (void)cache_dir; // reserved (unused since the default cert is bundled, not cached)
    if (!format || !*format)
        return false;
    if (!user_cert.empty() && !user_key.empty())
        return crispasr_c2pa_sign_buffer(data, format, user_cert, user_key);
    return crispasr_c2pa_sign_pem(data, format, crispasr_c2pa_default_cert_pem(), crispasr_c2pa_default_key_pem());
}

// Print a one-time startup note about C2PA capabilities.
inline void crispasr_c2pa_startup_check() {
#if !defined(CRISPASR_HAVE_C2PA) && defined(CRISPASR_NO_C2PA_NATIVE)
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "crispasr: C2PA signing disabled (no native signer and c2pa-c not found; "
                        "run scripts/fetch-c2pa.sh and rebuild, or enable the native signer)\n");
        warned = true;
    }
#elif !defined(CRISPASR_HAVE_C2PA)
    // The native signer covers WAV, MP3 (ID3v2.4 GEOB), M4A/MP4 (ISO BMFF
    // uuid) and FLAC — see crispasr_c2pa_sign_pem() above. c2pa-rs is only a
    // fallback for formats outside that set. No warning: signing works out of
    // the box for every container CrispASR emits a manifest into, and the
    // watertight marking floor relies on that being true.
#endif
}
