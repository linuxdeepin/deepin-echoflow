// glint - MDCT (Modified Discrete Cosine Transform)
// MIT License - Clean-room implementation

#ifndef GLINT_MDCT_HPP
#define GLINT_MDCT_HPP

#include <cstdint>

namespace glint {

#if !defined(GLINT_FIXED_POINT) || defined(GLINT_BOTH_PATHS)
// Double-precision MDCT
class MDCT {
public:
    MDCT();
    void process(const double subband[32][18], double mdct_out[32][18]);
    // Read directly from subband_out[32][36] at slot offset, applying
    // frequency inversion inline. Eliminates the sub_gr copy.
    // block_type selects the analysis window: 0 = long (sine), 1 = start,
    // 3 = stop (the hybrid transition windows that keep MDCT time-domain
    // alias cancellation valid across a long<->short switch). Type 2 is the
    // 3x12 short transform — use process_short for that.
    void process_strided(const double subband_out[32][36], int slot_offset,
                         double mdct_out[32][18], int block_type = 0);
    void process_short(const double subband[32][18], double mdct_out[32][3][6]);
    void reset();
private:
    double prev_[32][18];
};

void alias_reduce_d(double mdct_out[32][18]);
#endif

#ifdef GLINT_FIXED_POINT
// Fixed-point (Q24) MDCT
class MDCT_FP {
public:
    MDCT_FP();
    void process(const int32_t subband[32][18], int32_t mdct_out[32][18]);
    // Fused MDCT + alias reduction + Q24->double conversion. block_type 1/3
    // (start/stop transition windows) take a double-precision branch — the
    // quantizer downstream is double anyway and those granules are rare;
    // block_type 0 keeps the byte-stable integer path.
    void process_and_convert(const int32_t subband[32][18], double mdct_flat[576],
                             int block_type = 0);
    // Short-block MDCT (block_type 2): Q24 -> double, then the same
    // 12-point window/cos math as the double path. No alias reduction
    // (ISO). prev_ overlap handled like the long transform.
    void process_short_and_convert(const int32_t subband[32][18],
                                   double mdct_out[32][3][6]);
#ifdef GLINT_MP3_INT
    // No-FPU long-block path (block_type 0 only): fused window+cos via a
    // Q31 wincos table, integer alias reduction, flat Q24 int32 output.
    void process_int(const int32_t subband[32][18], int32_t mdct_flat[576]);
#endif
    void reset();
private:
    int32_t prev_[32][18];
};

void alias_reduce_fp(int32_t mdct_out[32][18]);
#endif

} // namespace glint

#endif
