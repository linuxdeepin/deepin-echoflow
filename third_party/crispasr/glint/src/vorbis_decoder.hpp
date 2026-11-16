// glint - Ogg-Vorbis I decoder
// MIT License - Clean-room implementation
//
// Decodes a complete in-memory Ogg-Vorbis I logical stream to interleaved
// float PCM. Implemented from the Vorbis I specification (xiph.org) only;
// no third-party codec source consulted. See PLAN.md "# Vorbis track".
//
// Scope: the three Vorbis header packets (identification / comment / setup),
// codebooks (scalar + VQ lookup types 1/2), floor types 0 (LSP) and 1
// (piecewise linear), residue types 0/1/2 with inverse channel coupling,
// mapping type 0, the inverse MDCT and windowed overlap-add. Single logical
// stream, mono/stereo and beyond (up to the channel count .sf3 needs).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace glint {
namespace vorbis {

// Decode a whole Ogg-Vorbis buffer to interleaved float PCM (+-1.0).
// Returns 0 on success and fills `pcm` (interleaved), `sample_rate`,
// `channels`. Negative on error / not a Vorbis stream.
int decode_ogg(const uint8_t* ogg, size_t len, std::vector<float>& pcm,
               int& sample_rate, int& channels);

// ---- Internals exposed for unit tests -------------------------------------

// A decoded Vorbis codebook (spec §3).
struct Codebook {
    int dimensions = 0;
    int entries = 0;
    std::vector<uint8_t> lengths;     // codeword length per entry (0 = unused)
    // Huffman decode tree: nodes stored as parallel child arrays. A child
    // value >= 0 is an internal node index; a value <= -2 is the leaf for
    // entry (-value - 2); -1 means "no child" (empty subtree).
    std::vector<int32_t> child0;      // bit 0 branch
    std::vector<int32_t> child1;      // bit 1 branch
    // VQ lookup (spec §3.2 / §3.3).
    int lookup_type = 0;
    float minimum_value = 0.f;
    float delta_value = 0.f;
    int value_bits = 0;
    int sequence_p = 0;
    int lookup_values = 0;
    std::vector<uint32_t> multiplicands;  // raw value-list entries
    std::vector<float> vq;                // entries*dimensions decoded vectors
    int single_entry_ = -1;               // >=0: degenerate 1-entry codebook

    bool build_huffman();  // build the decode tree from `lengths`
    void build_vq();       // materialize the VQ value list (lookup 1/2)
    // Decode one entry index from a bit reader (returns -1 on error).
    int decode_scalar(class BitReader& br) const;
    // Decode one VQ vector into out[0..dimensions) (returns -1 on error).
    int decode_vector(class BitReader& br, float* out) const;
};

// Read + build a codebook from the setup header bitstream (spec §3.2.1).
// Returns 0 on success, negative on a malformed codebook.
int read_codebook(class BitReader& br, Codebook& cb);

// Spec helpers, exposed for unit tests.
float float32_unpack(uint32_t x);          // spec §9.2.2
uint32_t lookup1_values(int entries, int dimensions);  // spec §9.2.3

// Floor 0 (LSP) curve synthesis (spec §6.2.3): from `order` LSP coefficients
// and the amplitude/quantization parameters, fill out[0..n) with the linear
// floor magnitude. Exposed for unit testing (no encoder emits floor 0, so
// the math is cross-checked against an independent reference).
void floor0_curve(int order, const float* coef, int amplitude,
                  int amplitude_bits, int amplitude_offset, int rate,
                  int bark_map_size, int n, float* out);

// Parsed identification header (spec §4.2.2).
struct IdHeader {
    int version = -1;
    int channels = 0;
    uint32_t sample_rate = 0;
    int blocksize0 = 0;  // short block
    int blocksize1 = 0;  // long block
    bool valid = false;
};

// Parse a Vorbis identification header packet (must start with 0x01 "vorbis").
IdHeader parse_id_header(const uint8_t* pkt, size_t len);

// Test hook: demux + parse all three headers (incl. the full setup header)
// of an in-memory Ogg-Vorbis buffer. Returns 0 on success. Writes channel
// count, sample rate, and how many setup-header bits were consumed vs.
// available (a correct parse consumes all but the trailing byte padding).
int debug_parse_headers(const uint8_t* ogg, size_t len, int* channels,
                        int* rate, size_t* setup_bits_used,
                        size_t* setup_bits_total);

}  // namespace vorbis
}  // namespace glint
