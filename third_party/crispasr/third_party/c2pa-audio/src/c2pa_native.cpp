// c2pa_audio_native.cpp — implementation of c2pa_audio_native.h.
//
// Compiled at C++17. All C++14/17-only constructs (generic lambdas,
// multi-statement lambda return deduction, micro-ecc) live in this TU so the
// C++11 crispasr-lib headers that include c2pa_audio_native.h stay clean.
// See the header for the design overview.
#include "c2pa_native.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>

#include "sha256.h"

extern "C" {
#include "uECC.h"
}

namespace crispasr {
namespace c2pa_native {

// ------------------------------------------------------------------ utilities
inline void put(Bytes& b, std::initializer_list<uint8_t> v) {
    b.insert(b.end(), v);
}
inline void put(Bytes& b, const Bytes& v) {
    b.insert(b.end(), v.begin(), v.end());
}
inline void put(Bytes& b, const uint8_t* p, size_t n) {
    b.insert(b.end(), p, p + n);
}
inline void put_str(Bytes& b, const std::string& s) {
    b.insert(b.end(), s.begin(), s.end());
}

// ------------------------------------------------------------- canonical CBOR
namespace cbor {
inline Bytes head(uint8_t major, uint64_t n) {
    Bytes o;
    uint8_t m = uint8_t(major << 5);
    if (n < 24) {
        o = {uint8_t(m | n)};
    } else if (n < 0x100) {
        o = {uint8_t(m | 24), uint8_t(n)};
    } else if (n < 0x10000) {
        o = {uint8_t(m | 25), uint8_t(n >> 8), uint8_t(n)};
    } else if (n < 0x100000000ULL) {
        o = {uint8_t(m | 26), uint8_t(n >> 24), uint8_t(n >> 16), uint8_t(n >> 8), uint8_t(n)};
    } else {
        o = {uint8_t(m | 27),  uint8_t(n >> 56), uint8_t(n >> 48), uint8_t(n >> 40), uint8_t(n >> 32),
             uint8_t(n >> 24), uint8_t(n >> 16), uint8_t(n >> 8),  uint8_t(n)};
    }
    return o;
}
inline Bytes uint_(uint64_t n) {
    return head(0, n);
}
inline Bytes nint(int64_t n) {
    return head(1, uint64_t(-1 - n));
} // n < 0
inline Bytes tstr(const std::string& s) {
    Bytes o = head(3, s.size());
    put_str(o, s);
    return o;
}
inline Bytes bstr(const Bytes& v) {
    Bytes o = head(2, v.size());
    put(o, v);
    return o;
}
inline Bytes null_() {
    return {0xf6};
}
inline Bytes arr(const std::vector<Bytes>& items) {
    Bytes o = head(4, items.size());
    for (auto& it : items)
        put(o, it);
    return o;
}
inline Bytes tag(uint64_t t, const Bytes& v) {
    Bytes o = head(6, t);
    put(o, v);
    return o;
}
// map with pre-encoded (key, value) pairs, sorted canonically (length then bytes)
inline Bytes map_raw(std::vector<std::pair<Bytes, Bytes>> kv) {
    std::sort(kv.begin(), kv.end(), [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size())
            return a.first.size() < b.first.size();
        return std::lexicographical_compare(a.first.begin(), a.first.end(), b.first.begin(), b.first.end());
    });
    Bytes o = head(5, kv.size());
    for (auto& p : kv) {
        put(o, p.first);
        put(o, p.second);
    }
    return o;
}
// convenience: string-keyed map
inline Bytes map(std::vector<std::pair<std::string, Bytes>> kv) {
    std::vector<std::pair<Bytes, Bytes>> raw;
    for (auto& p : kv)
        raw.emplace_back(tstr(p.first), p.second);
    return map_raw(std::move(raw));
}
} // namespace cbor

// ------------------------------------------------------------------ JUMBF box
inline const uint8_t* uuid_suffix() {
    static const uint8_t s[12] = {0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};
    return s;
}
inline Bytes box_type(const char ascii[5]) { // 4 ASCII chars + 12-byte suffix
    Bytes t;
    t.insert(t.end(), ascii, ascii + 4);
    put(t, uuid_suffix(), 12);
    return t;
}
inline void u32be(Bytes& b, uint32_t n) {
    put(b, {uint8_t(n >> 24), uint8_t(n >> 16), uint8_t(n >> 8), uint8_t(n)});
}
inline Bytes box(const char type4[5], const Bytes& payload) {
    Bytes o;
    u32be(o, uint32_t(8 + payload.size()));
    o.insert(o.end(), type4, type4 + 4);
    put(o, payload);
    return o;
}
// jumd description box: [16-byte type-uuid][toggles=0x03][label\0]
inline Bytes jumd(const Bytes& type_uuid, const std::string& label) {
    Bytes p;
    put(p, type_uuid);
    p.push_back(0x03);
    put_str(p, label);
    p.push_back(0x00);
    return box("jumd", p);
}
// jumb superbox: jumd(desc) + content boxes
inline Bytes jumb(const char type_ascii[5], const std::string& label, const std::vector<Bytes>& content) {
    Bytes p = jumd(box_type(type_ascii), label);
    for (auto& c : content)
        put(p, c);
    return box("jumb", p);
}
inline Bytes cbor_box(const Bytes& cbor_bytes) {
    return box("cbor", cbor_bytes);
}

// ---------------------------------------------------------------- PEM parsing
inline int b64val(char c) {
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}
inline Bytes b64decode(const std::string& s) {
    Bytes out;
    int buf = 0, bits = 0;
    for (char c : s) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t')
            continue;
        int v = b64val(c);
        if (v < 0)
            continue;
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t(buf >> bits));
        }
    }
    return out;
}
// extract the base64 body between BEGIN/END <kind> and return decoded DER
inline Bytes pem_to_der(const std::string& pem, const std::string& kind) {
    std::string begin = "-----BEGIN " + kind + "-----";
    std::string end = "-----END " + kind + "-----";
    auto b = pem.find(begin);
    if (b == std::string::npos)
        return {};
    b += begin.size();
    auto e = pem.find(end, b);
    if (e == std::string::npos)
        return {};
    return b64decode(pem.substr(b, e - b));
}
// Extract the 32-byte P-256 private scalar from a PKCS#8 or SEC1 EC key DER.
// Both contain the ECPrivateKey sequence: 02 01 01 (version=1) 04 20 <32 bytes>.
inline bool extract_p256_scalar(const Bytes& der, uint8_t out[32]) {
    static const uint8_t marker[5] = {0x02, 0x01, 0x01, 0x04, 0x20};
    for (size_t i = 0; i + 5 + 32 <= der.size(); i++) {
        if (std::memcmp(&der[i], marker, 5) == 0) {
            std::memcpy(out, &der[i + 5], 32);
            return true;
        }
    }
    return false;
}

// -------------------------------------------------------- ES256 (micro-ecc)
struct Sha256HashCtx {
    uECC_HashContext base;
    sha::Sha256 ctx;
};
inline Sha256HashCtx* hc_of(const uECC_HashContext* b) {
    return const_cast<Sha256HashCtx*>(reinterpret_cast<const Sha256HashCtx*>(b));
}
inline void hc_init(const uECC_HashContext* b) {
    hc_of(b)->ctx.init();
}
inline void hc_update(const uECC_HashContext* b, const uint8_t* m, unsigned n) {
    hc_of(b)->ctx.update(m, n);
}
inline void hc_finish(const uECC_HashContext* b, uint8_t* r) {
    hc_of(b)->ctx.final(r);
}

// Sign msg with ES256; returns raw r||s (64 bytes) — exactly COSE format.
inline bool es256_sign(const uint8_t priv[32], const Bytes& msg, uint8_t sig[64]) {
    auto digest = sha::sha256(msg);
    uint8_t tmp[2 * 32 + 64];
    Sha256HashCtx hc;
    hc.base.init_hash = &hc_init;
    hc.base.update_hash = &hc_update;
    hc.base.finish_hash = &hc_finish;
    hc.base.block_size = 64;
    hc.base.result_size = 32;
    hc.base.tmp = tmp;
    return uECC_sign_deterministic(priv, digest.data(), 32, &hc.base, sig, uECC_secp256r1()) == 1;
}

// ------------------------------------------------------------------ UUID v4
inline std::string uuid_v4() {
    std::random_device rd;
    uint8_t b[16];
    for (auto& x : b)
        x = uint8_t(rd());
    b[6] = uint8_t((b[6] & 0x0f) | 0x40); // version 4
    b[8] = uint8_t((b[8] & 0x3f) | 0x80); // variant
    static const char* hx = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            s += '-';
        s += hx[b[i] >> 4];
        s += hx[b[i] & 0xf];
    }
    return s;
}

// ------------------------------------------------------- RIFF chunk locate
struct Chunk {
    size_t start;
    uint32_t size;
};
inline uint32_t rd_u32le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

// ---------------------------------------------------------------- main entry
// Sign a WAV container with a C2PA manifest. certPem/keyPem: PEM strings.
// Returns the signed WAV, or an empty vector on any failure (caller keeps the
// unsigned WAV — still watermarked / metadata-tagged upstream).
namespace {

// Container-agnostic C2PA manifest-store builder (JUMBF/CBOR/COSE/ES256). The
// container adapters (WAV RIFF chunk, MP3 ID3v2 GEOB, ...) reuse this to build
// the identical store; only the embedding + byte-range exclusion differ.
struct ManifestSigner {
    bool ok = false;
    uint8_t priv[32];
    std::string manifest_urn, instance_id, gen_name, gen_version;
    Bytes actions_box, actions_hash, protected_hdr;

    struct Assembled {
        Bytes store, hash_data_box;
    };

    static Bytes assertion_hash(const Bytes& b) {
        auto h = sha::sha256(b.data() + 8, b.size() - 8);
        return Bytes(h.begin(), h.end());
    }

    ManifestSigner(const std::string& cert_pem, const std::string& key_pem, const SignOptions& opts) {
        Bytes cert_der = pem_to_der(cert_pem, "CERTIFICATE");
        if (cert_der.empty())
            return;
        Bytes key_der = pem_to_der(key_pem, "PRIVATE KEY");
        if (key_der.empty())
            key_der = pem_to_der(key_pem, "EC PRIVATE KEY");
        if (key_der.empty() || !extract_p256_scalar(key_der, priv))
            return;
        manifest_urn = "urn:c2pa:" + uuid_v4();
        instance_id = "xmp:iid:" + uuid_v4();
        gen_name = opts.generator_name;
        gen_version = opts.generator_version;
        Bytes actions_cbor = cbor::map({
            {"actions", cbor::arr({cbor::map({
                            {"action", cbor::tstr("c2pa.created")},
                            {"softwareAgent", cbor::tstr(opts.software_agent)},
                            {"digitalSourceType",
                             cbor::tstr("http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia")},
                        })})},
        });
        actions_box = jumb("cbor", "c2pa.actions.v2", {cbor_box(actions_cbor)});
        actions_hash = assertion_hash(actions_box);
        protected_hdr = cbor::map_raw({
            {cbor::uint_(1), cbor::nint(-7)},
            {cbor::uint_(33), cbor::arr({cbor::bstr(cert_der)})},
        });
        ok = true;
    }

    Bytes build_claim(const std::string& hard_label, const Bytes& hard_assn_hash) const {
        return cbor::map({
            {"instanceID", cbor::tstr(instance_id)},
            {"claim_generator_info", cbor::map({
                                         {"name", cbor::tstr(gen_name)},
                                         {"version", cbor::tstr(gen_version)},
                                     })},
            {"signature", cbor::tstr("self#jumbf=/c2pa/" + manifest_urn + "/c2pa.signature")},
            {"created_assertions", cbor::arr({cbor::map({
                                       {"url", cbor::tstr("self#jumbf=c2pa.assertions/" + hard_label)},
                                       {"hash", cbor::bstr(hard_assn_hash)},
                                   })})},
            {"gathered_assertions", cbor::arr({cbor::map({
                                        {"url", cbor::tstr("self#jumbf=c2pa.assertions/c2pa.actions.v2")},
                                        {"hash", cbor::bstr(actions_hash)},
                                    })})},
            {"alg", cbor::tstr("sha256")},
        });
    }

    Bytes build_cose_box(const Bytes& claim_bytes) const {
        Bytes sig_structure =
            cbor::arr({cbor::tstr("Signature1"), cbor::bstr(protected_hdr), cbor::bstr({}), cbor::bstr(claim_bytes)});
        uint8_t sig[64];
        if (!es256_sign(priv, sig_structure, sig))
            return {};
        Bytes cose = cbor::tag(
            18, cbor::arr({cbor::bstr(protected_hdr), cbor::map({}), cbor::null_(), cbor::bstr(Bytes(sig, sig + 64))}));
        return jumb("c2cs", "c2pa.signature", {cbor_box(cose)});
    }

    // Generic: assemble the store around a caller-built hard-binding assertion
    // box (hash.data for WAV/MP3, hash.bmff.v3 for M4A). Returns {store, hardBox}.
    Assembled assemble_with(const Bytes& hard_box, const std::string& hard_label,
                            const Bytes& hard_assn_hash) const {
        Bytes assertions_box = jumb("c2as", "c2pa.assertions", {actions_box, hard_box});
        Bytes claim_bytes = build_claim(hard_label, hard_assn_hash);
        Bytes claim_box = jumb("c2cl", "c2pa.claim.v2", {cbor_box(claim_bytes)});
        Bytes sig_box = build_cose_box(claim_bytes);
        Bytes manifest_box = jumb("c2ma", manifest_urn, {assertions_box, claim_box, sig_box});
        Bytes store = jumb("c2pa", "c2pa", {manifest_box});
        return {std::move(store), Bytes()};
    }

    // WAV/MP3: byte-range data-hash exclusion [excl_start, excl_len).
    Assembled assemble(const std::array<uint8_t, 32>& file_hash, uint64_t excl_start, uint64_t excl_len,
                       const Bytes& hashdata_assn_hash) const {
        Bytes hashdata_cbor = cbor::map({
            {"exclusions", cbor::arr({cbor::map({
                               {"start", cbor::uint_(excl_start)},
                               {"length", cbor::uint_(excl_len)},
                           })})},
            {"name", cbor::tstr("jumbf manifest")},
            {"alg", cbor::tstr("sha256")},
            {"hash", cbor::bstr(Bytes(file_hash.begin(), file_hash.end()))},
            {"pad", cbor::bstr(Bytes(8, 0))},
        });
        Bytes hash_data_box = jumb("cbor", "c2pa.hash.data", {cbor_box(hashdata_cbor)});
        Assembled a = assemble_with(hash_data_box, "c2pa.hash.data", hashdata_assn_hash);
        a.hash_data_box = std::move(hash_data_box);
        return a;
    }

    // M4A/MP4: BMFF v3 assertion — box-path exclusions + the offset-prepend hash.
    static Bytes bmff_v3_assertion(const std::array<uint8_t, 32>& bmff_hash) {
        static const uint8_t C2PA_UUID[16] = {0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c,
                                              0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81};
        Bytes uuid_excl = cbor::map({
            {"xpath", cbor::tstr("/uuid")},
            {"data", cbor::arr({cbor::map({
                         {"offset", cbor::uint_(8)},
                         {"value", cbor::bstr(Bytes(C2PA_UUID, C2PA_UUID + 16))},
                     })})},
        });
        auto simple = [](const char* xp) { return cbor::map({{"xpath", cbor::tstr(xp)}}); };
        return cbor::map({
            {"exclusions", cbor::arr({uuid_excl, simple("/ftyp"), simple("/mfra"), simple("/free"), simple("/skip")})},
            {"alg", cbor::tstr("sha256")},
            {"hash", cbor::bstr(Bytes(bmff_hash.begin(), bmff_hash.end()))},
            {"name", cbor::tstr("jumbf manifest")},
        });
    }
    Assembled assemble_bmff(const std::array<uint8_t, 32>& bmff_hash, const Bytes& hard_assn_hash) const {
        Bytes box = jumb("cbor", "c2pa.hash.bmff.v3", {cbor_box(bmff_v3_assertion(bmff_hash))});
        Assembled a = assemble_with(box, "c2pa.hash.bmff.v3", hard_assn_hash);
        a.hash_data_box = std::move(box);
        return a;
    }
};

// Read a synchsafe 28-bit integer (ID3v2 sizes).
inline uint32_t synchsafe(const uint8_t* p) { return (uint32_t(p[0]) << 21) | (p[1] << 14) | (p[2] << 7) | p[3]; }
inline void put_synchsafe(Bytes& b, uint32_t n) {
    put(b, {uint8_t((n >> 21) & 0x7f), uint8_t((n >> 14) & 0x7f), uint8_t((n >> 7) & 0x7f), uint8_t(n & 0x7f)});
}

}  // namespace

Bytes sign_wav(const Bytes& wav, const std::string& cert_pem, const std::string& key_pem, const SignOptions& opts) {
    if (wav.size() < 12)
        return {};
    ManifestSigner ms(cert_pem, key_pem, opts);
    if (!ms.ok)
        return {};
    const std::array<uint8_t, 32> ZERO{};

    const size_t chunk_start = wav.size(); // append at end
    const uint64_t excl_start = chunk_start;
    // The chunk length (excl_len) is embedded inside the hash.data exclusion; its
    // CBOR int width changes the store size, so iterate to a fixed point.
    uint64_t excl_len = 0;
    for (int i = 0; i < 6; i++) {
        Bytes st = ms.assemble(ZERO, excl_start, excl_len, Bytes(ZERO.begin(), ZERO.end())).store;
        uint64_t n = 8 + st.size() + (st.size() & 1);
        if (n == excl_len)
            break;
        excl_len = n;
    }
    auto with_chunk = [&](const Bytes& store) {
        Bytes out(wav.begin(), wav.begin() + chunk_start);
        put_str(out, "C2PA");
        put(out, {uint8_t(store.size()), uint8_t(store.size() >> 8), uint8_t(store.size() >> 16),
                  uint8_t(store.size() >> 24)});
        put(out, store);
        if (store.size() & 1)
            out.push_back(0);
        uint32_t riff = uint32_t(out.size() - 8); // RIFF size = total - 8
        out[4] = uint8_t(riff);
        out[5] = uint8_t(riff >> 8);
        out[6] = uint8_t(riff >> 16);
        out[7] = uint8_t(riff >> 24);
        return out;
    };
    auto file_hash_excluding = [&](const Bytes& full) {
        Bytes h(full.begin(), full.begin() + excl_start);
        h.insert(h.end(), full.begin() + excl_start + excl_len, full.end());
        return sha::sha256(h);
    };

    Bytes tmp = with_chunk(ms.assemble(ZERO, excl_start, excl_len, Bytes(ZERO.begin(), ZERO.end())).store);
    auto file_hash = file_hash_excluding(tmp);
    auto a3 = ms.assemble(file_hash, excl_start, excl_len, Bytes(ZERO.begin(), ZERO.end()));
    Bytes hashdata_assn_hash = ManifestSigner::assertion_hash(a3.hash_data_box);
    auto a4 = ms.assemble(file_hash, excl_start, excl_len, hashdata_assn_hash);
    if (a4.store.empty())
        return {};
    return with_chunk(a4.store);
}

// Sign an MP3 by embedding the C2PA manifest store in an ID3v2.4 GEOB frame
// (mime "application/c2pa", description "c2pa manifest store"). The hard-binding
// data-hash exclusion covers exactly the GEOB object (the store bytes).
Bytes sign_mp3(const Bytes& mp3, const std::string& cert_pem, const std::string& key_pem, const SignOptions& opts) {
    if (mp3.size() < 4)
        return {};
    ManifestSigner ms(cert_pem, key_pem, opts);
    if (!ms.ok)
        return {};
    const std::array<uint8_t, 32> ZERO{};

    // Preserve any existing ID3v2 tag frames; the audio starts after them.
    Bytes existing_frames;
    size_t audio_start = 0;
    if (mp3.size() >= 10 && std::memcmp(mp3.data(), "ID3", 3) == 0 && mp3[3] == 0x04) {
        uint32_t tagsize = synchsafe(&mp3[6]);
        size_t tag_end = 10 + tagsize;
        if (tag_end <= mp3.size()) {
            audio_start = tag_end;
            // Walk frames, copying complete real frames until padding (a frame
            // id beginning with 0x00). Blindly trimming trailing zeros would eat
            // body bytes and misalign the tag.
            size_t o = 10;
            while (o + 10 <= tag_end) {
                if (mp3[o] == 0) // padding
                    break;
                uint32_t fsz = synchsafe(&mp3[o + 4]);
                size_t frame_total = 10 + fsz;
                if (o + frame_total > tag_end)
                    break;
                existing_frames.insert(existing_frames.end(), mp3.begin() + o, mp3.begin() + o + frame_total);
                o += frame_total;
            }
        }
    }
    const Bytes audio(mp3.begin() + audio_start, mp3.end());

    // GEOB frame body prefix (before the object/store):
    //   [enc=0x00]["application/c2pa"\0][""\0]["c2pa manifest store"\0]
    Bytes geob_prefix;
    geob_prefix.push_back(0x00);
    put_str(geob_prefix, "application/c2pa");
    geob_prefix.push_back(0);
    geob_prefix.push_back(0); // empty filename
    put_str(geob_prefix, "c2pa manifest store");
    geob_prefix.push_back(0);

    // The store's absolute file offset is fixed (independent of store size):
    //   10 (ID3 hdr) + existing_frames + 10 (GEOB hdr) + geob_prefix
    const uint64_t excl_start = 10 + existing_frames.size() + 10 + geob_prefix.size();

    // excl_len == store size; iterate to a fixed point (CBOR width feedback).
    uint64_t excl_len = 0;
    for (int i = 0; i < 6; i++) {
        Bytes st = ms.assemble(ZERO, excl_start, excl_len, Bytes(ZERO.begin(), ZERO.end())).store;
        if (st.size() == excl_len)
            break;
        excl_len = st.size();
    }

    auto assemble_file = [&](const Bytes& store) {
        // GEOB frame = 10-byte header (id + synchsafe size + 2 flag bytes) + body
        Bytes geob_body = geob_prefix;
        put(geob_body, store);
        Bytes geob;
        put_str(geob, "GEOB");
        put_synchsafe(geob, uint32_t(geob_body.size()));
        put(geob, {0, 0}); // flags
        put(geob, geob_body);
        // ID3v2.4 tag = 10-byte header + existing frames + our GEOB frame
        Bytes frames = existing_frames;
        put(frames, geob);
        Bytes out;
        put_str(out, "ID3");
        put(out, {0x04, 0x00, 0x00}); // v2.4.0, no flags
        put_synchsafe(out, uint32_t(frames.size()));
        put(out, frames);
        put(out, audio);
        return out;
    };
    auto file_hash_excluding = [&](const Bytes& full) {
        Bytes h(full.begin(), full.begin() + excl_start);
        h.insert(h.end(), full.begin() + excl_start + excl_len, full.end());
        return sha::sha256(h);
    };

    Bytes tmp = assemble_file(ms.assemble(ZERO, excl_start, excl_len, Bytes(ZERO.begin(), ZERO.end())).store);
    auto file_hash = file_hash_excluding(tmp);
    auto a3 = ms.assemble(file_hash, excl_start, excl_len, Bytes(ZERO.begin(), ZERO.end()));
    Bytes hashdata_assn_hash = ManifestSigner::assertion_hash(a3.hash_data_box);
    auto a4 = ms.assemble(file_hash, excl_start, excl_len, hashdata_assn_hash);
    if (a4.store.empty())
        return {};
    return assemble_file(a4.store);
}

namespace {
inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
// Read a big-endian box size; returns {size, header_len}. size==0 means "to EOF".
inline void bmff_box_size(const Bytes& b, size_t o, uint64_t& size, size_t& hdr) {
    size = be32(&b[o]);
    hdr = 8;
    if (size == 1 && o + 16 <= b.size()) {
        size = 0;
        for (int i = 0; i < 8; i++)
            size = (size << 8) | b[o + 8 + i];
        hdr = 16;
    } else if (size == 0) {
        size = b.size() - o;
    }
}
// Recursively adjust stco (32-bit) / co64 (64-bit) chunk offsets by +delta within
// [start, end) of buffer `out`. Recurses into container boxes on the moov path.
void adjust_chunk_offsets(Bytes& out, size_t start, size_t end, uint64_t delta) {
    size_t o = start;
    while (o + 8 <= end) {
        uint64_t bs;
        size_t hdr;
        bmff_box_size(out, o, bs, hdr);
        if (bs < hdr || o + bs > end)
            break;
        const char* t = reinterpret_cast<const char*>(&out[o + 4]);
        if (std::memcmp(t, "stco", 4) == 0 || std::memcmp(t, "co64", 4) == 0) {
            bool is64 = std::memcmp(t, "co64", 4) == 0;
            size_t p = o + hdr + 4; // skip version+flags
            if (p + 4 <= end) {
                uint32_t count = uint32_t(be32(&out[p]));
                p += 4;
                for (uint32_t i = 0; i < count; i++) {
                    if (is64) {
                        if (p + 8 > end)
                            break;
                        uint64_t v = 0;
                        for (int k = 0; k < 8; k++)
                            v = (v << 8) | out[p + k];
                        v += delta;
                        for (int k = 0; k < 8; k++)
                            out[p + k] = uint8_t(v >> (56 - k * 8));
                        p += 8;
                    } else {
                        if (p + 4 > end)
                            break;
                        uint32_t v = uint32_t(be32(&out[p])) + uint32_t(delta);
                        out[p] = uint8_t(v >> 24);
                        out[p + 1] = uint8_t(v >> 16);
                        out[p + 2] = uint8_t(v >> 8);
                        out[p + 3] = uint8_t(v);
                        p += 4;
                    }
                }
            }
        } else if (std::memcmp(t, "moov", 4) == 0 || std::memcmp(t, "trak", 4) == 0 ||
                   std::memcmp(t, "mdia", 4) == 0 || std::memcmp(t, "minf", 4) == 0 ||
                   std::memcmp(t, "stbl", 4) == 0) {
            adjust_chunk_offsets(out, o + hdr, o + bs, delta);
        }
        o += bs;
    }
}
// SHA-256 over each NON-excluded top-level box: BE64(offset) ++ box bytes.
// Excludes ftyp, free, mfra, skip, and the C2PA uuid box.
std::array<uint8_t, 32> bmff_v3_hash(const Bytes& file) {
    static const uint8_t C2PA_UUID[16] = {0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c,
                                          0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81};
    sha::Sha256 h;
    h.init();
    size_t o = 0;
    while (o + 8 <= file.size()) {
        uint64_t bs;
        size_t hdr;
        bmff_box_size(file, o, bs, hdr);
        if (bs < hdr || o + bs > file.size())
            break;
        const char* t = reinterpret_cast<const char*>(&file[o + 4]);
        bool excl = std::memcmp(t, "ftyp", 4) == 0 || std::memcmp(t, "free", 4) == 0 ||
                    std::memcmp(t, "mfra", 4) == 0 || std::memcmp(t, "skip", 4) == 0;
        if (std::memcmp(t, "uuid", 4) == 0 && o + hdr + 16 <= file.size() &&
            std::memcmp(&file[o + hdr], C2PA_UUID, 16) == 0)
            excl = true;
        if (!excl) {
            uint8_t be[8];
            for (int i = 0; i < 8; i++)
                be[i] = uint8_t(o >> (56 - i * 8));
            h.update(be, 8);
            h.update(&file[o], bs);
        }
        o += bs;
    }
    std::array<uint8_t, 32> out{};
    h.final(out.data());
    return out;
}
} // namespace

// Sign an M4A/MP4 (ISO BMFF) by inserting a C2PA 'uuid' box after 'ftyp' and
// binding with a c2pa.hash.bmff.v3 assertion (offset-prepend hash). Chunk
// offsets (stco/co64) are adjusted for the inserted box.
Bytes sign_m4a(const Bytes& m4a, const std::string& cert_pem, const std::string& key_pem, const SignOptions& opts) {
    if (m4a.size() < 16 || std::memcmp(&m4a[4], "ftyp", 4) != 0)
        return {};
    ManifestSigner ms(cert_pem, key_pem, opts);
    if (!ms.ok)
        return {};
    const std::array<uint8_t, 32> ZERO{};

    uint64_t ftyp_size;
    size_t ftyp_hdr;
    bmff_box_size(m4a, 0, ftyp_size, ftyp_hdr);
    const size_t insert_at = size_t(ftyp_size); // right after ftyp

    static const uint8_t C2PA_UUID[16] = {0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c,
                                          0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81};
    // uuid box = [size][ "uuid"][16 uuid][4 reserved=0]["manifest\0"(9)][8 merkle=0][store]
    auto uuid_box = [&](const Bytes& store) {
        Bytes b;
        uint32_t sz = uint32_t(8 + 16 + 4 + 9 + 8 + store.size());
        u32be(b, sz);
        put_str(b, "uuid");
        put(b, C2PA_UUID, 16);
        put(b, {0, 0, 0, 0}); // reserved
        put_str(b, "manifest");
        b.push_back(0);
        put(b, Bytes(8, 0)); // merkle offset
        put(b, store);
        return b;
    };
    // Build the final file for a given store: ftyp | uuid(store) | rest, with
    // chunk offsets shifted by the inserted uuid box size.
    auto build_file = [&](const Bytes& store) {
        Bytes ub = uuid_box(store);
        const uint64_t delta = ub.size();
        Bytes out(m4a.begin(), m4a.begin() + insert_at);
        put(out, ub);
        size_t rest = out.size();
        out.insert(out.end(), m4a.begin() + insert_at, m4a.end());
        adjust_chunk_offsets(out, rest, out.size(), delta); // fix stco/co64 in the shifted region
        return out;
    };

    // Store size is fixed (all fields fixed-width) -> uuid box size is stable, so
    // the mdat/moov offsets are determined by a placeholder pass; no iteration.
    Bytes placeholder = ms.assemble_bmff(ZERO, Bytes(ZERO.begin(), ZERO.end())).store;
    Bytes skeleton = build_file(placeholder);
    auto bhash = bmff_v3_hash(skeleton);
    auto a3 = ms.assemble_bmff(bhash, Bytes(ZERO.begin(), ZERO.end()));
    Bytes hard_assn_hash = ManifestSigner::assertion_hash(a3.hash_data_box);
    auto a4 = ms.assemble_bmff(bhash, hard_assn_hash);
    if (a4.store.empty() || a4.store.size() != placeholder.size())
        return {};
    return build_file(a4.store);
}

// FLAC uses the exact same container mechanism as MP3 (an ID3v2.4 GEOB manifest
// tag prepended, the fLaC audio preserved after it) — this is what c2pa-rs does
// too. The signer/verifier are byte-identical to the MP3 path.
Bytes sign_flac(const Bytes& flac, const std::string& cert_pem, const std::string& key_pem, const SignOptions& opts) {
    if (flac.size() < 4 || std::memcmp(flac.data(), "fLaC", 4) != 0)
        return {};
    return sign_mp3(flac, cert_pem, key_pem, opts);
}

// ============================================================================
// Verifier — the native twin of bindings/javascript/c2pa-verify.mjs.
// ============================================================================
namespace {

// ---- minimal CBOR decoder -> generic Value ----
struct Value {
    enum Type { UINT, NINT, BSTR, TSTR, ARRAY, MAP, TAG, BOOL, NUL } type = NUL;
    int64_t i = 0;                            // UINT / NINT / BOOL
    Bytes bytes;                              // BSTR
    std::string str;                          // TSTR
    std::vector<Value> arr;                   // ARRAY, or TAG(1 elem)
    std::vector<std::pair<Value, Value>> map; // MAP
    uint64_t tag = 0;                         // TAG

    const Value* get(const std::string& key) const { // map lookup by tstr key
        for (auto& kv : map)
            if (kv.first.type == TSTR && kv.first.str == key)
                return &kv.second;
        return nullptr;
    }
    const Value* get(int64_t key) const { // map lookup by integer key
        for (auto& kv : map)
            if ((kv.first.type == UINT || kv.first.type == NINT) && kv.first.i == key)
                return &kv.second;
        return nullptr;
    }
};

struct CborReader {
    const uint8_t* p;
    size_t n, pos = 0;
    bool ok = true;
    // `len` is attacker-controlled and up to UINT64_MAX (ai==27), so `pos + len`
    // can wrap and slip past a naive `pos + len > n` bound. Subtract instead —
    // pos <= n always holds, so n - pos cannot underflow.
    bool avail(uint64_t need) const {
        return need <= uint64_t(n - pos);
    }
    uint64_t readLen(uint8_t ai) {
        if (ai < 24)
            return ai;
        int nb = ai == 24 ? 1 : ai == 25 ? 2 : ai == 26 ? 4 : ai == 27 ? 8 : 0;
        if (!nb || !avail(uint64_t(nb))) {
            ok = false;
            return 0;
        }
        uint64_t v = 0;
        for (int k = 0; k < nb; k++)
            v = (v << 8) | p[pos++];
        return v;
    }
    Value value(int depth = 0) {
        Value out;
        if (pos >= n || depth > 64) { // bound nesting: a hostile CBOR must not blow the stack
            ok = false;
            return out;
        }
        uint8_t ib = p[pos++];
        uint8_t major = ib >> 5, ai = ib & 0x1f;
        uint64_t len = readLen(ai);
        if (!ok)
            return out;
        switch (major) {
        // `len` can be up to UINT64_MAX; store it in the int64 `i` without
        // signed overflow (UB). Callers only use `i` as an offset/length and
        // reject out-of-range values, so saturating an oversized int is safe.
        case 0:
            out.type = Value::UINT;
            out.i = len > uint64_t(INT64_MAX) ? INT64_MAX : int64_t(len);
            break;
        case 1:
            out.type = Value::NINT;
            out.i = len >= uint64_t(INT64_MAX) ? INT64_MIN : -1 - int64_t(len);
            break;
        case 2:
            if (!avail(len)) {
                ok = false;
                break;
            }
            out.type = Value::BSTR;
            out.bytes.assign(p + pos, p + pos + len);
            pos += len;
            break;
        case 3:
            if (!avail(len)) {
                ok = false;
                break;
            }
            out.type = Value::TSTR;
            out.str.assign(reinterpret_cast<const char*>(p + pos), len);
            pos += len;
            break;
        // every array item / map pair costs at least one byte, so a count
        // exceeding the bytes left is malformed — reject before growing.
        case 4:
            if (!avail(len)) {
                ok = false;
                break;
            }
            out.type = Value::ARRAY;
            for (uint64_t k = 0; k < len && ok; k++)
                out.arr.push_back(value(depth + 1));
            break;
        case 5:
            if (!avail(len)) {
                ok = false;
                break;
            }
            out.type = Value::MAP;
            for (uint64_t k = 0; k < len && ok; k++) {
                Value key = value(depth + 1);
                Value val = value(depth + 1);
                out.map.emplace_back(std::move(key), std::move(val));
            }
            break;
        case 6:
            out.type = Value::TAG;
            out.tag = len;
            out.arr.push_back(value(depth + 1));
            break;
        case 7:
            if (ai == 20) {
                out.type = Value::BOOL;
                out.i = 0;
            } else if (ai == 21) {
                out.type = Value::BOOL;
                out.i = 1;
            } else
                out.type = Value::NUL;
            break;
        default:
            ok = false;
        }
        return out;
    }
};
Value cbor_decode(const Bytes& b, bool& ok) {
    CborReader r{b.data(), b.size()};
    Value v = r.value();
    ok = r.ok;
    return v;
}

// ---- JUMBF walk: label -> {full box, cbor content} ----
struct JumbfBox {
    Bytes box;
    Bytes content;
    bool has_content = false;
};
uint32_t rd32be(const uint8_t* b) {
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
}
// Real C2PA manifests nest ~5 deep and hold a handful of boxes; these caps stop
// a hostile file (deep nesting / huge box count) from stack-overflowing or
// wedging the parser in O(n^2) copies. depth/count are bounded defensively.
void jumbf_walk(const uint8_t* b, size_t len, std::vector<std::pair<std::string, JumbfBox>>& out, int depth = 0) {
    if (depth > 32 || out.size() > 4096)
        return;
    size_t o = 0;
    while (o + 8 <= len) {
        uint32_t sz = rd32be(b + o);
        if (sz < 8 || o + sz > len)
            break;
        if (std::memcmp(b + o + 4, "jumb", 4) == 0) {
            const uint8_t* payload = b + o + 8;
            size_t plen = sz - 8;
            // a valid jumb holds a jumd box (>= 4 bytes for its size field); a
            // runt box (size 8..11) has too little payload to read jsz safely.
            uint32_t jsz = plen >= 4 ? rd32be(payload) : 0;
            const size_t lblStart = 8 + 16 + 1;
            // scan the label bounded by BOTH the jumd size and the payload length
            std::string label;
            if (lblStart <= plen) {
                size_t lblEnd = plen < jsz ? plen : jsz;
                size_t e = lblStart;
                while (e < lblEnd && payload[e] != 0)
                    e++;
                label.assign(reinterpret_cast<const char*>(payload + lblStart), e - lblStart);
            }
            JumbfBox jb;
            jb.box.assign(b + o, b + o + sz);
            size_t po = jsz;
            while (po + 8 <= plen) {
                uint32_t psz = rd32be(payload + po);
                if (psz < 8 || po + psz > plen)
                    break;
                if (std::memcmp(payload + po + 4, "cbor", 4) == 0) {
                    jb.content.assign(payload + po + 8, payload + po + psz);
                    jb.has_content = true;
                }
                po += psz;
            }
            out.emplace_back(label, std::move(jb));
            if (out.size() > 4096)
                return;
            jumbf_walk(payload, plen, out, depth + 1);
        }
        o += sz;
    }
}
const JumbfBox* find_box(const std::vector<std::pair<std::string, JumbfBox>>& boxes, const std::string& label) {
    for (auto& p : boxes)
        if (p.first == label)
            return &p.second;
    return nullptr;
}

// Extract the 64-byte EC public point (x||y, NO 0x04 prefix — uECC form) from an
// X.509 P-256 cert DER: locate the SPKI BIT STRING `03 42 00 04 <64>`.
bool extract_ec_point(const Bytes& der, uint8_t out[64]) {
    for (size_t i = 0; i + 4 + 65 <= der.size(); i++) {
        if (der[i] == 0x03 && der[i + 1] == 0x42 && der[i + 2] == 0x00 && der[i + 3] == 0x04) {
            std::memcpy(out, &der[i + 4], 64); // skip the 0x04 marker
            return true;
        }
    }
    return false;
}

} // namespace

VerifyResult verify_wav(const Bytes& wav) {
    VerifyResult res;
    auto& err = res.errors;

    // locate the C2PA manifest store — RIFF 'C2PA' chunk (WAV) or ID3v2 GEOB
    // frame (MP3). The store bytes are copied into `store`.
    if (wav.size() < 12) {
        err.push_back("input too small");
        return res;
    }
    Bytes store;
    if (std::memcmp(wav.data(), "RIFF", 4) == 0) {
        for (size_t off = 12; off + 8 <= wav.size();) {
            uint32_t sz = uint32_t(wav[off + 4]) | (uint32_t(wav[off + 5]) << 8) | (uint32_t(wav[off + 6]) << 16) |
                          (uint32_t(wav[off + 7]) << 24);
            if (std::memcmp(&wav[off], "C2PA", 4) == 0) {
                if (off + 8 + sz <= wav.size())
                    store.assign(wav.begin() + off + 8, wav.begin() + off + 8 + sz);
                break;
            }
            off += 8 + sz + (sz & 1);
        }
    } else if (wav.size() >= 10 && std::memcmp(wav.data(), "ID3", 3) == 0) {
        uint32_t tagsize = synchsafe(&wav[6]);
        size_t end = std::min(wav.size(), size_t(10) + tagsize);
        for (size_t o = 10; o + 10 <= end;) {
            if (wav[o] == 0)
                break; // padding
            uint32_t fsz = synchsafe(&wav[o + 4]);
            if (o + 10 + fsz > end)
                break;
            if (std::memcmp(&wav[o], "GEOB", 4) == 0) {
                const uint8_t* body = &wav[o + 10];
                size_t p = 1; // skip text-encoding byte
                size_t ms = p;
                while (p < fsz && body[p] != 0)
                    p++;
                std::string mime(reinterpret_cast<const char*>(body + ms), p - ms);
                p++; // nul
                while (p < fsz && body[p] != 0)
                    p++;
                p++; // filename nul
                while (p < fsz && body[p] != 0)
                    p++;
                p++; // description nul
                if (mime == "application/c2pa" && p <= fsz) {
                    store.assign(body + p, body + fsz);
                    break;
                }
            }
            o += 10 + fsz;
        }
    } else if (wav.size() >= 8 && std::memcmp(&wav[4], "ftyp", 4) == 0) {
        // ISO BMFF (M4A/MP4): the store is in a top-level 'uuid' box with the
        // C2PA uuid, after a [16 uuid][4 reserved]["manifest\0"(9)][8 offset]=37
        // byte prefix.
        static const uint8_t C2PA_UUID[16] = {0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c,
                                              0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81};
        size_t o = 0;
        while (o + 8 <= wav.size()) {
            uint64_t bs = rd32be(&wav[o]);
            size_t hdr = 8;
            if (bs == 1 && o + 16 <= wav.size()) {
                bs = 0;
                for (int i = 0; i < 8; i++)
                    bs = (bs << 8) | wav[o + 8 + i];
                hdr = 16;
            } else if (bs == 0) {
                bs = wav.size() - o;
            }
            if (bs < hdr || o + bs > wav.size())
                break;
            if (std::memcmp(&wav[o + 4], "uuid", 4) == 0 && o + hdr + 16 <= wav.size() &&
                std::memcmp(&wav[o + hdr], C2PA_UUID, 16) == 0) {
                size_t sstart = o + hdr + 16 + 4 + 9 + 8; // reserved + "manifest\0" + merkle offset
                if (sstart <= o + bs)
                    store.assign(wav.begin() + sstart, wav.begin() + o + bs);
                break;
            }
            o += bs;
        }
    }
    if (store.empty()) {
        err.push_back("no C2PA manifest store found");
        return res;
    }

    std::vector<std::pair<std::string, JumbfBox>> boxes;
    jumbf_walk(store.data(), store.size(), boxes);
    const JumbfBox *claimBox = find_box(boxes, "c2pa.claim.v2"), *sigBox = find_box(boxes, "c2pa.signature"),
                   *actBox = find_box(boxes, "c2pa.actions.v2");
    const JumbfBox* hashBox = find_box(boxes, "c2pa.hash.data"); // byte-range (WAV/MP3)
    const JumbfBox* bmffBox = find_box(boxes, "c2pa.hash.bmff.v3"); // BMFF (M4A/MP4)
    if (!claimBox || !sigBox || !actBox || (!hashBox && !bmffBox)) {
        err.push_back("missing required JUMBF box");
        return res;
    }

    bool ok = true;
    Value claim = cbor_decode(claimBox->content, ok);
    if (!ok) {
        err.push_back("claim CBOR decode failed");
        return res;
    }
    Value cose = cbor_decode(sigBox->content, ok);
    if (!ok) {
        err.push_back("signature CBOR decode failed");
        return res;
    }
    const Value& coseArr = (cose.type == Value::TAG && !cose.arr.empty()) ? cose.arr[0] : cose;
    if (coseArr.type != Value::ARRAY || coseArr.arr.size() < 4) {
        err.push_back("malformed COSE_Sign1");
        return res;
    }
    const Bytes& protectedBstr = coseArr.arr[0].bytes;
    const Bytes& signature = coseArr.arr[3].bytes;

    // cert from protected header {1:-7, 33:[cert]}
    Value prot = cbor_decode(protectedBstr, ok);
    const Value* chain = ok ? prot.get(int64_t(33)) : nullptr;
    const Bytes* certDer = nullptr;
    if (chain) {
        if (chain->type == Value::ARRAY && !chain->arr.empty())
            certDer = &chain->arr[0].bytes;
        else if (chain->type == Value::BSTR)
            certDer = &chain->bytes;
    }
    if (!certDer) {
        err.push_back("no certificate in COSE protected header");
        return res;
    }

    // verify COSE ES256 (uECC): message = Sig_structure, hash = sha256(message)
    uint8_t pub[64];
    if (!extract_ec_point(*certDer, pub)) {
        err.push_back("cannot extract EC public key from cert");
        return res;
    }
    // Sig_structure = ["Signature1", protected(bstr), external_aad(empty bstr), payload(claim bstr)]
    Bytes sigStruct;
    {
        put(sigStruct, cbor::head(4, 4));
        put(sigStruct, cbor::tstr("Signature1"));
        put(sigStruct, cbor::bstr(protectedBstr));
        put(sigStruct, cbor::bstr(Bytes{}));
        put(sigStruct, cbor::bstr(claimBox->content));
    }
    auto digest = sha::sha256(sigStruct);
    if (signature.size() == 64)
        res.signature_valid = uECC_verify(pub, digest.data(), 32, signature.data(), uECC_secp256r1()) == 1;
    if (!res.signature_valid)
        err.push_back("COSE ES256 signature does not verify");

    // assertion hashedURIs: hash = sha256(box without 8-byte header)
    res.assertions_valid = true;
    for (const char* grp : {"created_assertions", "gathered_assertions"}) {
        const Value* list = claim.get(std::string(grp));
        if (!list || list->type != Value::ARRAY)
            continue;
        for (const Value& a : list->arr) {
            const Value* url = a.get("url");
            const Value* h = a.get("hash");
            if (!url || !h)
                continue;
            std::string label = url->str.substr(url->str.find_last_of('/') + 1);
            const JumbfBox* bx = find_box(boxes, label);
            if (!bx) {
                err.push_back("assertion box not found: " + label);
                res.assertions_valid = false;
                continue;
            }
            auto actual = sha::sha256(bx->box.data() + 8, bx->box.size() - 8);
            if (h->bytes.size() != 32 || std::memcmp(actual.data(), h->bytes.data(), 32) != 0) {
                err.push_back("assertion hash mismatch: " + label);
                res.assertions_valid = false;
            }
        }
    }

    // hard binding
    if (hashBox) {
        // byte-range data hash (WAV/MP3): hash file minus the [start,length) region.
        Value hd = cbor_decode(hashBox->content, ok);
        const Value* excls = ok ? hd.get("exclusions") : nullptr;
        uint64_t exStart = 0, exLen = 0;
        if (excls && excls->type == Value::ARRAY && !excls->arr.empty()) {
            const Value* s = excls->arr[0].get("start");
            const Value* l = excls->arr[0].get("length");
            // start/length are attacker-controlled CBOR ints; a negative (NINT)
            // or huge value would make the iterator arithmetic below allocate a
            // vector past max_size() (std::length_error). Reject anything not a
            // plausible in-file offset.
            if (s && s->i >= 0)
                exStart = uint64_t(s->i);
            if (l && l->i >= 0)
                exLen = uint64_t(l->i);
        }
        const Value* storedHash = ok ? hd.get("hash") : nullptr;
        // subtractive bound so exStart + exLen cannot wrap past wav.size()
        if (storedHash && exStart <= wav.size() && exLen <= wav.size() - exStart) {
            Bytes concat(wav.begin(), wav.begin() + exStart);
            concat.insert(concat.end(), wav.begin() + exStart + exLen, wav.end());
            auto fh = sha::sha256(concat);
            res.data_hash_valid =
                storedHash->bytes.size() == 32 && std::memcmp(fh.data(), storedHash->bytes.data(), 32) == 0;
        }
    } else {
        // BMFF v3 hash (M4A/MP4): SHA-256 over, for each NON-excluded top-level
        // box, BE64(box offset) ++ box bytes. Exclusions come from the assertion
        // (xpath box types; /uuid additionally matched by a data value).
        Value bh = cbor_decode(bmffBox->content, ok);
        const Value* storedHash = ok ? bh.get("hash") : nullptr;
        const Value* excls = ok ? bh.get("exclusions") : nullptr;
        if (storedHash && storedHash->bytes.size() == 32) {
            sha::Sha256 hh;
            hh.init();
            size_t o = 0;
            while (o + 8 <= wav.size()) {
                uint64_t bs = rd32be(&wav[o]);
                size_t hdr = 8;
                if (bs == 1 && o + 16 <= wav.size()) {
                    bs = 0;
                    for (int i = 0; i < 8; i++)
                        bs = (bs << 8) | wav[o + 8 + i];
                    hdr = 16;
                } else if (bs == 0) {
                    bs = wav.size() - o;
                }
                if (bs < hdr || o + bs > wav.size())
                    break;
                std::string btype(reinterpret_cast<const char*>(&wav[o + 4]), 4);
                // is this box excluded?
                bool excluded = false;
                if (excls && excls->type == Value::ARRAY) {
                    for (const Value& ex : excls->arr) {
                        const Value* xp = ex.get("xpath");
                        if (!xp || xp->str != "/" + btype)
                            continue;
                        const Value* dm = ex.get("data"); // optional data matcher
                        bool matches = true;
                        if (dm && dm->type == Value::ARRAY) {
                            for (const Value& d : dm->arr) {
                                const Value* offv = d.get("offset");
                                const Value* valv = d.get("value");
                                if (!offv || !valv)
                                    continue;
                                // offset is attacker-controlled: guard against a
                                // negative/huge value wrapping `o + offset` and
                                // against the compare below overflowing, so the
                                // memcmp stays inside this box.
                                if (offv->i < 0 || uint64_t(offv->i) > bs ||
                                    valv->bytes.size() > bs - uint64_t(offv->i)) {
                                    matches = false;
                                    break;
                                }
                                // an empty value matches anything — and calling
                                // memcmp with a null (empty vector) pointer, even
                                // for length 0, is UB (its args are nonnull).
                                if (valv->bytes.empty())
                                    continue;
                                size_t doff = o + size_t(offv->i);
                                if (std::memcmp(&wav[doff], valv->bytes.data(), valv->bytes.size()) != 0) {
                                    matches = false;
                                    break;
                                }
                            }
                        }
                        if (matches) {
                            excluded = true;
                            break;
                        }
                    }
                }
                if (!excluded) {
                    uint8_t be[8];
                    for (int i = 0; i < 8; i++)
                        be[i] = uint8_t(o >> (56 - i * 8));
                    hh.update(be, 8);
                    hh.update(&wav[o], bs);
                }
                o += bs;
            }
            uint8_t out[32];
            hh.final(out);
            res.data_hash_valid = std::memcmp(out, storedHash->bytes.data(), 32) == 0;
        }
    }
    if (!res.data_hash_valid)
        err.push_back("data hash (hard binding) mismatch");

    // surface generator name
    const Value* gen = claim.get("claim_generator_info");
    if (gen) {
        const Value* name = gen->get("name");
        if (name)
            res.generator_name = name->str;
    }

    res.valid = res.signature_valid && res.data_hash_valid && res.assertions_valid;
    res.trusted = false;
    return res;
}

} // namespace c2pa_native
} // namespace crispasr
