// glint - AAC-LC decoder (PLAN § D2)
// MIT License - Clean-room from ISO/IEC 14496-3 (AAC-LC, ADTS).
//
// Decodes ADTS AAC-LC (SCE/CPE, all four window sequences, M/S, TNS)
// into interleaved float PCM (±1.0), 1024 samples/ch per frame. Feed
// whole ADTS frames. Spectral + scalefactor Huffman decode trees are
// built at first use from the ENCODER's own code tables (aac_tables.hpp)
// so the two sides cannot drift.

#pragma once

#include <cstdint>

namespace glint {
namespace aac {

struct AacFrameInfo {
    int sample_rate = 0;
    int channels = 0;
    int samples = 0;       // per channel (1024)
    int frame_bytes = 0;   // whole ADTS frame length
};

// Parse just the ADTS header; returns 0 + fills info, or -1 if data does
// not start with a valid ADTS AAC-LC sync.
int aac_frame_info(const uint8_t* data, int len, AacFrameInfo* info);

class AacDecoder {
public:
    void init();

    // Decode one ADTS frame at data[0]. Returns samples per channel
    // (1024) written to pcm (interleaved ±1.0), or a negative error.
    int decode_frame(const uint8_t* data, int len, float* pcm,
                     AacFrameInfo* info = nullptr);

    struct Ics {
        int window_sequence;
        int window_shape;     // 0 = sine, 1 = KBD
        int max_sfb;
        int num_windows;      // 1 (long) or 8 (short)
        int num_window_groups;
        int group_len[8];     // windows per group
        int window_group_of[8];
    };
    struct TnsFilter {
        int order;
        int length;           // sfbs, counted down from num_swb
        int direction;
        double lpc[13];
    };

private:
    int decode_ics(int ch, bool common_window);
    void dequant_band(int ch, int bk, int sf, int i0, int i1);
    void inverse_quant(int ch);
    void deinterleave_short(int ch);
    void apply_tns(int ch, bool inverse);
    void imdct_channel(int ch, float* out);

    // Per-frame decode state.
    int sr_index_ = 0;
    int channels_ = 0;
    Ics ics_[2];
    int book_[2][8 * 52];     // section codebook per (group, sfb)
    int sf_[2][8 * 52];       // scalefactor per (group, sfb)
    double coef_[2][1024];    // spectral coefficients (window-major)
    int max_sfb_[2];
    TnsFilter tns_[2][8][3];  // [ch][window][filter]; long uses window 0
    int tns_nf_[2][8];        // filters per window
    int ms_mask_present_ = 0;
    int is_cpe_ms_ = 0;
    uint8_t ms_used_[8 * 52];

    double overlap_[2][1024]; // IMDCT overlap-add state
    int prev_window_seq_[2];
    int prev_window_shape_[2]; // left window half uses the PREVIOUS shape
    int first_ = 1;
};

}  // namespace aac
}  // namespace glint
