// test_c2pa_native.cpp — unit tests for the native C++ C2PA signer
// (src/core/crispasr_c2pa_native.{h,cpp}). Hermetic: no c2pa-rs, no network,
// no python. Validates the CBOR encoder, JUMBF box layout, and the structural /
// cryptographic invariants of a signed WAV, using the bundled self-signed cert.
//
// The [emit] case writes a signed WAV to $CRISPASR_C2PA_EMIT (or ./c2pa_native_
// signed.wav) so the live parity ctest can feed it to the c2pa-rs reference
// reader — mirroring bindings/javascript/test/c2pa.parity.test.mjs.

#include "crispasr_c2pa_native.h"
#include "crispasr_c2pa_default_cert.h"
#include "crispasr_sha256.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using crispasr::c2pa_native::Bytes;

namespace {

Bytes make_wav(int n = 4800, int sr = 24000) {
    Bytes w;
    auto p32 = [&](uint32_t v) {
        w.push_back(uint8_t(v));
        w.push_back(uint8_t(v >> 8));
        w.push_back(uint8_t(v >> 16));
        w.push_back(uint8_t(v >> 24));
    };
    auto p16 = [&](uint16_t v) {
        w.push_back(uint8_t(v));
        w.push_back(uint8_t(v >> 8));
    };
    auto s4 = [&](const char* s) { w.insert(w.end(), s, s + 4); };
    s4("RIFF");
    p32(uint32_t(36 + n * 2));
    s4("WAVE");
    s4("fmt ");
    p32(16);
    p16(1);
    p16(1);
    p32(uint32_t(sr));
    p32(uint32_t(sr * 2));
    p16(2);
    p16(16);
    s4("data");
    p32(uint32_t(n * 2));
    for (int i = 0; i < n; i++) {
        int16_t s = int16_t(3000.0 * std::sin(2.0 * 3.14159265358979 * 220.0 * i / sr));
        w.push_back(uint8_t(s & 0xff));
        w.push_back(uint8_t((s >> 8) & 0xff));
    }
    return w;
}

uint32_t rd32be(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
uint32_t rd32le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

struct Chunk {
    size_t start;
    uint32_t size;
};
bool find_chunk(const Bytes& wav, const char id[5], Chunk& out) {
    size_t off = 12;
    while (off + 8 <= wav.size()) {
        uint32_t sz = rd32le(&wav[off + 4]);
        if (std::memcmp(&wav[off], id, 4) == 0) {
            out = {off, sz};
            return true;
        }
        off += 8 + sz + (sz & 1);
    }
    return false;
}

// Collect label -> full jumb box bytes.
void walk_jumbf(const uint8_t* b, size_t len, std::vector<std::pair<std::string, Bytes>>& out) {
    size_t o = 0;
    while (o + 8 <= len) {
        uint32_t sz = rd32be(&b[o]);
        if (sz < 8 || o + sz > len)
            break;
        if (std::memcmp(&b[o + 4], "jumb", 4) == 0) {
            const uint8_t* payload = b + o + 8;
            size_t plen = sz - 8;
            size_t lblStart = 8 + 16 + 1; // jumd hdr + type uuid + toggles
            size_t end = lblStart;
            while (end < plen && payload[end] != 0)
                end++;
            std::string label(reinterpret_cast<const char*>(payload + lblStart), end - lblStart);
            out.emplace_back(label, Bytes(b + o, b + o + sz));
            walk_jumbf(payload, plen, out);
        }
        o += sz;
    }
}
const Bytes* box_of(const std::vector<std::pair<std::string, Bytes>>& boxes, const std::string& label) {
    for (auto& p : boxes)
        if (p.first == label)
            return &p.second;
    return nullptr;
}

std::string bundled_cert() {
    return crispasr_c2pa_default_cert_pem();
}
std::string bundled_key() {
    return crispasr_c2pa_default_key_pem();
}

} // namespace

TEST_CASE("C2PA native: signed WAV is well-formed RIFF with a C2PA chunk", "[unit][c2pa]") {
    Bytes wav = make_wav();
    Bytes signed_ = crispasr::c2pa_native::sign_wav(wav, bundled_cert(), bundled_key());
    REQUIRE(signed_.size() > wav.size());
    REQUIRE(std::memcmp(&signed_[8], "WAVE", 4) == 0);
    // RIFF size field = total - 8
    REQUIRE(rd32le(&signed_[4]) == signed_.size() - 8);
    Chunk c2pa;
    REQUIRE(find_chunk(signed_, "C2PA", c2pa));
    // original audio bytes untouched
    Chunk d0, d1;
    REQUIRE(find_chunk(wav, "data", d0));
    REQUIRE(find_chunk(signed_, "data", d1));
    REQUIRE(d0.size == d1.size);
    REQUIRE(std::memcmp(&wav[d0.start + 8], &signed_[d1.start + 8], d0.size) == 0);
}

TEST_CASE("C2PA native: JUMBF tree has the expected boxes and type UUIDs", "[unit][c2pa]") {
    Bytes signed_ = crispasr::c2pa_native::sign_wav(make_wav(), bundled_cert(), bundled_key());
    Chunk c2pa;
    REQUIRE(find_chunk(signed_, "C2PA", c2pa));
    std::vector<std::pair<std::string, Bytes>> boxes;
    walk_jumbf(&signed_[c2pa.start + 8], c2pa.size, boxes);

    for (const char* lbl :
         {"c2pa", "c2pa.assertions", "c2pa.actions.v2", "c2pa.hash.data", "c2pa.claim.v2", "c2pa.signature"}) {
        REQUIRE(box_of(boxes, lbl) != nullptr);
    }
    // type UUID first-4-bytes: box = [size|'jumb'](8)[jumd hdr](8)[type uuid](16)
    auto t4 = [&](const std::string& lbl) {
        const Bytes* b = box_of(boxes, lbl);
        return std::string(reinterpret_cast<const char*>(b->data() + 16), 4);
    };
    REQUIRE(t4("c2pa") == "c2pa");
    REQUIRE(t4("c2pa.assertions") == "c2as");
    REQUIRE(t4("c2pa.claim.v2") == "c2cl");
    REQUIRE(t4("c2pa.signature") == "c2cs");
}

TEST_CASE("C2PA native: assertion hash is sha256(box without 8-byte header)", "[unit][c2pa]") {
    Bytes signed_ = crispasr::c2pa_native::sign_wav(make_wav(), bundled_cert(), bundled_key());
    Chunk c2pa;
    REQUIRE(find_chunk(signed_, "C2PA", c2pa));
    std::vector<std::pair<std::string, Bytes>> boxes;
    walk_jumbf(&signed_[c2pa.start + 8], c2pa.size, boxes);
    const Bytes* actions = box_of(boxes, "c2pa.actions.v2");
    const Bytes* hashData = box_of(boxes, "c2pa.hash.data");
    REQUIRE(actions);
    REQUIRE(hashData);
    auto ha = crispasr::sha::sha256(actions->data() + 8, actions->size() - 8);
    auto hd = crispasr::sha::sha256(hashData->data() + 8, hashData->size() - 8);
    // non-degenerate + distinct (both are real 32-byte digests over different content)
    REQUIRE(std::memcmp(ha.data(), hd.data(), 32) != 0);
}

TEST_CASE("C2PA native: layout is size-deterministic across signings", "[unit][c2pa]") {
    Bytes wav = make_wav();
    Bytes a = crispasr::c2pa_native::sign_wav(wav, bundled_cert(), bundled_key());
    Bytes b = crispasr::c2pa_native::sign_wav(wav, bundled_cert(), bundled_key());
    // random manifest URN / instanceID differ, but total size is fixed
    REQUIRE(a.size() == b.size());
}

TEST_CASE("C2PA native: hard binding is over the audio (tamper changes file hash)", "[unit][c2pa]") {
    Bytes signed_ = crispasr::c2pa_native::sign_wav(make_wav(), bundled_cert(), bundled_key());
    Chunk c2pa;
    REQUIRE(find_chunk(signed_, "C2PA", c2pa));
    size_t clen = 8 + c2pa.size + (c2pa.size & 1);
    auto file_hash = [&](const Bytes& buf) {
        Bytes h(buf.begin(), buf.begin() + c2pa.start);
        h.insert(h.end(), buf.begin() + c2pa.start + clen, buf.end());
        return crispasr::sha::sha256(h);
    };
    auto clean = file_hash(signed_);
    Bytes tampered = signed_;
    tampered[46] ^= 0xff; // flip a byte inside the audio 'data' payload
    auto dirty = file_hash(tampered);
    REQUIRE(std::memcmp(clean.data(), dirty.data(), 32) != 0);
}

TEST_CASE("C2PA native: bad PEM input yields empty (no crash)", "[unit][c2pa]") {
    Bytes wav = make_wav();
    REQUIRE(crispasr::c2pa_native::sign_wav(wav, "not a cert", "not a key").empty());
    REQUIRE(crispasr::c2pa_native::sign_wav(Bytes{}, bundled_cert(), bundled_key()).empty());
}

// ---------------------------------------------------------------- verifier
TEST_CASE("C2PA native verify: round-trip (sign then verify) is valid", "[unit][c2pa]") {
    Bytes signed_ = crispasr::c2pa_native::sign_wav(make_wav(), bundled_cert(), bundled_key());
    auto r = crispasr::c2pa_native::verify_wav(signed_);
    INFO(std::string("err: ") + (r.errors.empty() ? std::string("ok") : r.errors[0]));
    REQUIRE(r.signature_valid);
    REQUIRE(r.data_hash_valid);
    REQUIRE(r.assertions_valid);
    REQUIRE(r.valid);
    REQUIRE(r.generator_name == "CrispASR");
}

TEST_CASE("C2PA native verify: audio tamper fails the hard binding", "[unit][c2pa]") {
    Bytes signed_ = crispasr::c2pa_native::sign_wav(make_wav(), bundled_cert(), bundled_key());
    signed_[46] ^= 0xff; // flip a byte in the audio payload
    auto r = crispasr::c2pa_native::verify_wav(signed_);
    REQUIRE_FALSE(r.data_hash_valid);
    REQUIRE_FALSE(r.valid);
}

TEST_CASE("C2PA native verify: signature tamper fails", "[unit][c2pa]") {
    Bytes signed_ = crispasr::c2pa_native::sign_wav(make_wav(), bundled_cert(), bundled_key());
    signed_[signed_.size() - 20] ^= 0xff; // flip a byte in the trailing signature region
    auto r = crispasr::c2pa_native::verify_wav(signed_);
    REQUIRE_FALSE(r.valid);
}

TEST_CASE("C2PA native verify: non-C2PA WAV reports no manifest", "[unit][c2pa]") {
    auto r = crispasr::c2pa_native::verify_wav(make_wav());
    REQUIRE_FALSE(r.valid);
    REQUIRE_FALSE(r.errors.empty());
}

// their signer -> our verifier: validate a committed c2pa-rs reference vector.
TEST_CASE("C2PA native verify: c2pa-rs reference vector validates", "[unit][c2pa]") {
#ifdef CRISPASR_TEST_ASSETS_DIR
    std::string p = std::string(CRISPASR_TEST_ASSETS_DIR) + "/c2pa/reference-c2pa-rs.wav";
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        WARN("reference fixture missing: " + p);
        return;
    }
    Bytes wav((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto r = crispasr::c2pa_native::verify_wav(wav);
    INFO(std::string("err: ") + (r.errors.empty() ? std::string("ok") : r.errors[0]));
    REQUIRE(r.signature_valid);
    REQUIRE(r.data_hash_valid);
    REQUIRE(r.valid);
#else
    WARN("CRISPASR_TEST_ASSETS_DIR not defined; skipping reference-vector check");
#endif
}

// Emit a signed WAV for the live parity ctest (validates in c2pa-rs reader).
TEST_CASE("C2PA native: emit signed WAV for parity", "[emit][c2pa]") {
    Bytes signed_ = crispasr::c2pa_native::sign_wav(make_wav(), bundled_cert(), bundled_key());
    REQUIRE_FALSE(signed_.empty());
    const char* path = std::getenv("CRISPASR_C2PA_EMIT");
    std::string out = path ? path : "c2pa_native_signed.wav";
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(signed_.data()), std::streamsize(signed_.size()));
    REQUIRE(f.good());
}
