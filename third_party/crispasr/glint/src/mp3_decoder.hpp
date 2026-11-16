// glint - MPEG-1/2 Layer III decoder (PLAN § D1)
// MIT License - Clean-room implementation from ISO 11172-3 / 13818-3.
//
// Feed whole frames (sync + header + side info + main data); the
// decoder keeps the bit reservoir across calls, so the first frame(s)
// of a stream may yield 0 samples until main_data_begin is satisfied.
// Output: interleaved float PCM (±1.0), 1152 samples/ch (MPEG-1) or
// 576 (MPEG-2/2.5 LSF) per frame.
//
// Huffman decode trees are built at first use from the ENCODER's code
// tables (tables.hpp) — the two sides can't drift apart.

#pragma once

#include <cstdint>

namespace glint {
namespace mp3 {

struct Mp3FrameInfo {
    int sample_rate = 0;
    int channels = 0;
    int samples = 0;     // per channel (0 while the reservoir fills)
    int frame_bytes = 0; // whole-frame length incl. header
    int bitrate_kbps = 0;
    int mpeg2 = 0;       // 1 for MPEG-2/2.5 LSF
};

// Parse just enough of a header to know the frame length; returns 0 and
// fills info, or -1 if data does not start with a valid Layer III sync.
int mp3_frame_info(const uint8_t* data, int len, Mp3FrameInfo* info);

class Mp3Decoder {
public:
    void init();

    // Decode ONE frame starting at data[0] (must be a syncword; use
    // mp3_frame_info / mp3_find_sync to locate). Returns samples per
    // channel written to pcm (interleaved, ±1.0), 0 while the
    // reservoir fills, or a negative error. info (optional) receives
    // stream parameters.
    int decode_frame(const uint8_t* data, int len, float* pcm,
                     Mp3FrameInfo* info = nullptr);

private:
    struct GranuleInfo {
        int part2_3_length, big_values, global_gain, scalefac_compress;
        int window_switching, block_type, mixed_block;
        int table_select[3];
        int subblock_gain[3];
        int region0_count, region1_count;
        int preflag, scalefac_scale, count1table_select;
    };

    void requantize(const GranuleInfo& g, int ch, int sr_unified);
    void stereo_process(const GranuleInfo (*gi)[2], int gr, int mode_ext,
                        int mpeg2, int sr_unified, int intensity_scale);
    void antialias(const GranuleInfo& g, int ch);
    void imdct_granule(const GranuleInfo& g, int ch);
    void synth_granule(int ch, int n_gran, float* pcm, int nch);

    // Bit reservoir (main data): up to 511 bytes back + current frame.
    uint8_t store_[4096];
    int store_len_ = 0;

    // Per-granule working state.
    int ix_[2][576];              // decoded integer spectrum
    double xr_[2][576];           // requantized (then stereo/alias/imdct)
    int scalefac_l_[2][23];
    int scalefac_s_[2][13][3];
    int nonzero_[2];              // count of potentially nonzero lines

    double overlap_[2][576];      // IMDCT overlap-add state
    double synth_v_[2][1024];     // polyphase synthesis FIFO
    int synth_off_[2];

    int first_ = 1;
};

}  // namespace mp3
}  // namespace glint
