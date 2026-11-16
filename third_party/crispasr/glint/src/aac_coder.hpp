// glint - AAC-LC quantization, sectioning and noiseless coding
// MIT License - Clean-room implementation

#ifndef GLINT_AAC_CODER_HPP
#define GLINT_AAC_CODER_HPP

#include <cstdint>

#include "aac_coder_types_fwd.hpp"
#include "aac_tns.hpp"

namespace glint {
namespace aac {

constexpr int kMaxSfb = 52;       // >= largest kNumSwbLong (51) and 3 groups x 15 short sfbs
constexpr int kMaxQuant = 8191;   // spectral magnitude cap (book 11 escape)
constexpr int kSfOffset = 100;    // scalefactor offset per ISO

// Band layout of one frame in CODED ORDER. Long-family windows: bands are the
// long sfbs. EIGHT_SHORT: bands are (group, sfb) pairs, group-major, and the
// spectrum is interleaved to match (for each group: for each sfb: that sfb's
// lines of every window in the group, window-major) — so every band is a
// contiguous coded-order range, exactly like the long case.
struct AacBandLayout {
    uint8_t window_sequence;      // AacWindowSequence
    uint8_t num_groups;           // 1 for long-family
    uint8_t group_len[8];         // windows per group (short only)
    uint8_t max_sfb;              // wire max_sfb (per-group sfb count for short)
    int num_bands;                // long: max_sfb; short: num_groups * max_sfb
    int num_lines;                // offset[num_bands]
    uint16_t offset[kMaxSfb + 1]; // coded-order band offsets
    uint8_t group_of_band[kMaxSfb];
    int sect_bits;                // section-length field width: 5 long, 3 short
    int sect_esc;                 // escape value: 31 long, 7 short
};

// Build a layout. For long-family sequences group_len/num_groups are ignored.
void aac_make_layout(int sr_index, int window_sequence, int max_sfb,
                     const uint8_t* group_len, int num_groups,
                     AacBandLayout* layout);

// Interleave a window-major short spectrum (8 x 128) into coded order for
// `layout` (zero-padding the tail above num_lines up to 1024).
void aac_reorder_short(const SpecT* natural, const AacBandLayout& layout,
                       int sr_index, SpecT* coded);

// Minimal MSB-first bit writer into a caller-owned buffer.
// count_only mode tallies bits without touching memory (used for rate fitting).
class AacBitWriter {
public:
    AacBitWriter(uint8_t* buf, int capacity)
        : buf_(buf), cap_(capacity) {}
    explicit AacBitWriter(int)  // count-only
        : buf_(nullptr), cap_(0) {}

    void put(uint32_t value, int nbits) {
        bit_count_ += nbits;
        if (!buf_) return;
        while (nbits > 0) {
            int take = nbits > 24 ? 24 : nbits;
            put_raw((value >> (nbits - take)) & ((1u << take) - 1), take);
            nbits -= take;
        }
    }
    void byte_align() {
        int pad = (8 - (bit_count_ & 7)) & 7;
        if (pad) put(0, pad);
    }
    int bits() const { return bit_count_; }
    int bytes() const { return (bit_count_ + 7) / 8; }
    bool overflowed() const { return overflow_; }

private:
    void put_raw(uint32_t v, int n) {
        cache_ = (cache_ << n) | v;
        cache_bits_ += n;
        while (cache_bits_ >= 8) {
            cache_bits_ -= 8;
            if (byte_pos_ < cap_) {
                buf_[byte_pos_++] = static_cast<uint8_t>((cache_ >> cache_bits_) & 0xFF);
            } else {
                overflow_ = true;
            }
        }
    }
    uint8_t* buf_;
    int cap_;
    int byte_pos_ = 0;
    uint64_t cache_ = 0;
    int cache_bits_ = 0;
    int bit_count_ = 0;
    bool overflow_ = false;
};

// One channel's fitted quantization plan for a frame (bands per `layout`).
// `tns` must be set (by aac_tns_analyze, or active = 0) BEFORE fitting:
// its bits are part of the exact ICS cost the rate search sees.
struct AacChannelPlan {
    int global_gain;            // wire value = sf of the first coded band
    int fit_gain;               // the gain anchor G the rate search settled on
    AacTnsFilter tns;
    uint8_t book[kMaxSfb];      // per-band codebook (0 = zero band)
    uint8_t sf[kMaxSfb];        // effective per-band scalefactor (= G - offset)
    int16_t ix[1024];           // signed quantized coefficients (coded order)
    int ics_bits;               // exact individual_channel_stream cost, excl. ics_info
};

// Quantize with per-band scalefactors; returns max magnitude across bands.
// p34[i] = |spec[i]|^0.75, SpecT precision is plenty against quantization.
int aac_quantize(const P34T* p34, const SpecT* spec, const AacBandLayout& layout,
                 const uint8_t* sf, int16_t* ix);

// Choose per-band books (optimal sectioning DP; sections never cross group
// boundaries), derive the wire global_gain, and compute the exact ICS bit
// cost. Fills plan->book, plan->global_gain and plan->ics_bits.
void aac_section_and_count(const int16_t* ix, const AacBandLayout& layout,
                           AacChannelPlan* plan);

// Fit a channel: search the gain anchor G so the exact ICS cost fits
// budget_bits (and magnitudes fit kMaxQuant). Band b is quantized with
// sf[b] = clamp(G - sf_offsets[b], 0, 255); sf_offsets may be null (all 0).
// Offsets must stay in [-30, 30] so the scalefactor dpcm range (+-60) holds.
// gain_hint < 0 runs a fresh binary search; otherwise a local walk from the
// hint (cheap refits inside the shaping loop). gain_floor is VBR's
// constant-quality floor: never quantize finer than this anchor gain.
void aac_fit_channel(const SpecT* spec, const AacBandLayout& layout,
                     int budget_bits, const int* sf_offsets, int gain_hint,
                     int gain_floor, AacChannelPlan* plan);

// Per-band reconstruction noise sum((spec - dequant)^2) over coded bands.
void aac_band_noise(const AacChannelPlan& plan, const SpecT* spec,
                    const AacBandLayout& layout, double* noise);

// Distortion-controlled fit (normal/best CBR): per-band scalefactors in
// closed form from noise targets mask[b]*k (uniform-quantizer noise model),
// with the single loudness knob k bisected to the bit budget, then one
// measure-and-correct refinement pass. This replaces the iterative
// amplify/de-amplify walk, which measured ~5 dB short of the noise~mask
// allocation across the whole spectrum at 128k. `mask` is in the CODED
// spec domain (same as aac_band_noise); tns must be set in *plan.
// alpha_scale in [0,1] derates the mask-derived tilt (graduated trust:
// post-transient frames pass ~0, steady frames 1).
void aac_fit_channel_masked(const SpecT* spec, const AacBandLayout& layout,
                            const float* mask, int budget_bits,
                            double alpha_scale, AacChannelPlan* plan);

// Emit ics_info / individual_channel_stream. `include_ics_info` follows the
// wire layout: true for SCE (and CPE with common_window=0), false for the two
// ICS of a common_window CPE, whose shared ics_info the caller writes once.
void aac_write_ics_info(AacBitWriter& bw, const AacBandLayout& layout);
void aac_write_ics_body(AacBitWriter& bw, const AacChannelPlan& plan,
                        const AacBandLayout& layout, bool include_ics_info);

}  // namespace aac
}  // namespace glint

#endif  // GLINT_AAC_CODER_HPP
