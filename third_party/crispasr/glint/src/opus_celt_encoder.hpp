// CELT frame encoder — RFC 6716 section 4.3 (encoder side)
// MIT License - Clean-room implementation
//
// Analysis-complete encoder (PLAN § O4): pitch prefilter, transient
// detection + short blocks + anti-collapse, per-band TF resolution,
// dynalloc boosts, spread/tapset analysis, intensity + dual stereo,
// trim analysis, CBR or unconstrained VBR. Every symbol layer
// underneath is byte-exact with libopus (energy, allocator,
// theta/PVQ), so streams are valid by construction; the top-level
// DECISIONS are policy and only affect quality.
//
// Wire spec: the symbol sequence mirrors glint's own conformant decoder
// (opus_celt_decoder.cpp) — silence flag, postfilter bit, transient bit,
// coarse energy (with the intra decision), tf bits, spread, dynalloc,
// trim, allocation, fine energy, band shapes, energy finalise.

#pragma once

#include <cstdint>

#include "opus_analysis.hpp"
#include "opus_ec.hpp"
#include "opus_mdct.hpp"

namespace glint {
namespace opus {

class CeltEncoder {
public:
    void init(int channels);

    // 0 = CBR (default): encode_frame emits exactly nbytes.
    // >0 = unconstrained VBR at ~bitrate_bps: encode_frame treats nbytes
    // as the packet-size CAP and returns the actual size.
    void set_vbr(int bitrate_bps) { vbr_bitrate_ = bitrate_bps; }

    // Encode frame_size (120<<LM, LM 0..3) samples per channel of float
    // PCM (±1.0, interleaved) into at most nbytes. Returns the bytes
    // written (== nbytes in CBR), or a negative error. final_range()
    // afterwards gives the range-coder state for conformance checks.
    int encode_frame(const float* pcm, int frame_size, uint8_t* out,
                     int nbytes);

    uint32_t final_range() const { return final_range_; }

private:
    static constexpr int kOverlap = 120;
    static constexpr int kMaxFrame = 960;

    static constexpr int kCombMaxPeriod = 1024;

    int channels_ = 0;
    uint32_t final_range_ = 0;
    float preemph_mem_[2] = {};
    // FILTERED overlap history per channel (MDCT input head).
    double in_mem_[2][kOverlap] = {};
    // UNFILTERED pre-emphasized input history (prefilter comb source).
    double prefilter_mem_[2][kCombMaxPeriod] = {};
    int prefilter_period_ = 0;
    double prefilter_gain_ = 0;
    int prefilter_tapset_ = 0;
    int consec_transient_ = 0;
    // Spread/tapset analysis state (recursive averages + hysteresis).
    int tonal_average_ = 256;
    int hf_average_ = 0;
    int tapset_decision_ = 0;
    int spread_decision_ = 2;  // SPREAD_NORMAL
    int intensity_ = 0;
    // VBR state: target bps (0 = CBR), stereo-saving estimate from the
    // trim analysis, band-energy running average for temporal VBR.
    int vbr_bitrate_ = 0;
    double stereo_saving_ = 0;
    float spec_avg_ = 0;
    // FFT-phase tonality analyzer (policy inputs for VBR/trim/dynalloc).
    TonalityAnalyzer analysis_;
    float old_ebands_[2 * 21] = {};
    float energy_error_[2 * 21] = {};
    float delayed_intra_ = 1.0f;
    int last_coded_bands_ = 0;
    CeltImdct mdct_;
    double window_[kOverlap];
};

}  // namespace opus
}  // namespace glint
