// SILK sample-rate converter (decode side) — RFC 6716 section 4.2.9
// MIT License - Clean-room implementation

#include "opus_silk_resampler.hpp"

#include <cassert>
#include <cstring>

#include "opus_silk_math.hpp"
#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

namespace {

enum { kFnCopy = 0, kFnUp2Hq = 1, kFnIirFir = 2, kFnDownFir = 3 };

// 10 ms of input at the highest decode-side internal rate (16 kHz).
constexpr int kMaxBatchIn = 16 * 10;

// Wrapping two's-complement arithmetic (the reference ADD32/SUB32/LSHIFT
// are plain C expressions that rely on wrap; made explicit via uint32).
inline int32_t wsub(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) -
                                static_cast<uint32_t>(b));
}
inline int32_t wshl(int32_t a, int shift) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) << shift);
}

// Decoder delay-equalization table. Rows: fs_in {8,12,16} kHz; columns:
// fs_out {8,12,16,24,48} kHz.
constexpr int8_t kDelayDec[3][5] = {
    { 4, 0,  2, 0, 0 },
    { 0, 9,  4, 7, 4 },
    { 0, 3, 12, 7, 7 },
};

inline int rate_id(int khz) {
    switch (khz) {
        case 8:  return 0;
        case 12: return 1;
        case 16: return 2;
        case 24: return 3;
        case 48: return 4;
    }
    return -1;
}

// 2x upsampler: two cascades of three allpass sections (Q10 internally)
// producing the even/odd output phases, with a notch just above Nyquist.
void up2_hq(int32_t* s, int16_t* out, const int16_t* in, int len) {
    for (int k = 0; k < len; k++) {
        int32_t in32 = static_cast<int32_t>(in[k]) << 10;

        // Even phase: three allpasses through s[0..2].
        int32_t y = wsub(in32, s[0]);
        int32_t x = smulwb(y, kResamplerUp2Hq0[0]);
        int32_t out1 = add32_ovflw(s[0], x);
        s[0] = add32_ovflw(in32, x);

        y = wsub(out1, s[1]);
        x = smulwb(y, kResamplerUp2Hq0[1]);
        int32_t out2 = add32_ovflw(s[1], x);
        s[1] = add32_ovflw(out1, x);

        y = wsub(out2, s[2]);
        x = smlawb(y, y, kResamplerUp2Hq0[2]);
        out1 = add32_ovflw(s[2], x);
        s[2] = add32_ovflw(out2, x);

        out[2 * k] = static_cast<int16_t>(sat16(rshift_round(out1, 10)));

        // Odd phase: three allpasses through s[3..5].
        y = wsub(in32, s[3]);
        x = smulwb(y, kResamplerUp2Hq1[0]);
        out1 = add32_ovflw(s[3], x);
        s[3] = add32_ovflw(in32, x);

        y = wsub(out1, s[4]);
        x = smulwb(y, kResamplerUp2Hq1[1]);
        out2 = add32_ovflw(s[4], x);
        s[4] = add32_ovflw(out1, x);

        y = wsub(out2, s[5]);
        x = smlawb(y, y, kResamplerUp2Hq1[2]);
        out1 = add32_ovflw(s[5], x);
        s[5] = add32_ovflw(out2, x);

        out[2 * k + 1] = static_cast<int16_t>(sat16(rshift_round(out1, 10)));
    }
}

// Fractional upsampler: 2x allpass upsample, then interpolate with the
// 12-phase 8-tap FIR (phases stored as mirrored half-filters).
void iir_fir(Resampler& S, int16_t* out, const int16_t* in, int in_len) {
    int16_t buf[2 * kMaxBatchIn + kResamplerOrderFir12];
    std::memcpy(buf, S.s_fir.i16, kResamplerOrderFir12 * sizeof(int16_t));

    const int32_t inc = S.inv_ratio_q16;
    int n;
    for (;;) {
        n = in_len < S.batch_size ? in_len : S.batch_size;

        up2_hq(S.s_iir, &buf[kResamplerOrderFir12], in, n);

        int32_t max_index = wshl(n, 16 + 1);  // +1: 2x upsampled domain
        for (int32_t idx = 0; idx < max_index; idx += inc) {
            int32_t ti = smulwb(idx & 0xFFFF, 12);
            const int16_t* bp = &buf[idx >> 16];
            int32_t r = smulbb(bp[0], kResamplerFracFir12[ti][0]);
            r = smlabb(r, bp[1], kResamplerFracFir12[ti][1]);
            r = smlabb(r, bp[2], kResamplerFracFir12[ti][2]);
            r = smlabb(r, bp[3], kResamplerFracFir12[ti][3]);
            r = smlabb(r, bp[4], kResamplerFracFir12[11 - ti][3]);
            r = smlabb(r, bp[5], kResamplerFracFir12[11 - ti][2]);
            r = smlabb(r, bp[6], kResamplerFracFir12[11 - ti][1]);
            r = smlabb(r, bp[7], kResamplerFracFir12[11 - ti][0]);
            *out++ = static_cast<int16_t>(sat16(rshift_round(r, 15)));
        }

        in += n;
        in_len -= n;
        if (in_len > 0) {
            std::memcpy(buf, &buf[n << 1],
                        kResamplerOrderFir12 * sizeof(int16_t));
        } else {
            break;
        }
    }
    std::memcpy(S.s_fir.i16, &buf[n << 1],
                kResamplerOrderFir12 * sizeof(int16_t));
}

// Second-order AR filter, output in Q8 (anti-alias IIR before the
// down-FIR interpolation).
void ar2(int32_t* s, int32_t* out_q8, const int16_t* in,
         const int16_t* a_q14, int len) {
    for (int k = 0; k < len; k++) {
        int32_t out32 =
            add32_ovflw(s[0], wshl(static_cast<int32_t>(in[k]), 8));
        out_q8[k] = out32;
        out32 = wshl(out32, 2);
        s[0] = smlawb(s[1], out32, a_q14[0]);
        s[1] = smulwb(out32, a_q14[1]);
    }
}

// Polyphase FIR interpolation on the AR2 output. Only the decode-reachable
// filter orders are implemented: 18-tap fractional (2 or 3 phases, ratios
// 2:3 and 3:4) and 24-tap symmetric half-band (ratio 1:2).
int16_t* down_fir_interpol(int16_t* out, const int32_t* buf,
                           const int16_t* fir, int fir_order, int fir_fracs,
                           int32_t max_index, int32_t inc) {
    if (fir_order == kResamplerDownOrderFir0) {
        for (int32_t idx = 0; idx < max_index; idx += inc) {
            const int32_t* bp = &buf[idx >> 16];
            int32_t ind = smulwb(idx & 0xFFFF, fir_fracs);

            const int16_t* p = &fir[kResamplerDownOrderFir0 / 2 * ind];
            int32_t r = smulwb(bp[0], p[0]);
            r = smlawb(r, bp[1], p[1]);
            r = smlawb(r, bp[2], p[2]);
            r = smlawb(r, bp[3], p[3]);
            r = smlawb(r, bp[4], p[4]);
            r = smlawb(r, bp[5], p[5]);
            r = smlawb(r, bp[6], p[6]);
            r = smlawb(r, bp[7], p[7]);
            r = smlawb(r, bp[8], p[8]);
            p = &fir[kResamplerDownOrderFir0 / 2 * (fir_fracs - 1 - ind)];
            r = smlawb(r, bp[17], p[0]);
            r = smlawb(r, bp[16], p[1]);
            r = smlawb(r, bp[15], p[2]);
            r = smlawb(r, bp[14], p[3]);
            r = smlawb(r, bp[13], p[4]);
            r = smlawb(r, bp[12], p[5]);
            r = smlawb(r, bp[11], p[6]);
            r = smlawb(r, bp[10], p[7]);
            r = smlawb(r, bp[9], p[8]);

            *out++ = static_cast<int16_t>(sat16(rshift_round(r, 6)));
        }
    } else {  // kResamplerDownOrderFir1, symmetric
        for (int32_t idx = 0; idx < max_index; idx += inc) {
            const int32_t* bp = &buf[idx >> 16];
            int32_t r = smulwb(add32_ovflw(bp[0], bp[23]), fir[0]);
            for (int i = 1; i < kResamplerDownOrderFir1 / 2; i++)
                r = smlawb(r, add32_ovflw(bp[i], bp[23 - i]), fir[i]);
            *out++ = static_cast<int16_t>(sat16(rshift_round(r, 6)));
        }
    }
    return out;
}

// Fractional downsampler: AR2 (Q8) followed by polyphase FIR.
void down_fir(Resampler& S, int16_t* out, const int16_t* in, int in_len) {
    int32_t buf[kMaxBatchIn + kResamplerDownOrderFir1];
    std::memcpy(buf, S.s_fir.i32, S.fir_order * sizeof(int32_t));

    const int16_t* fir = &S.coefs[2];  // first two entries: AR2 coefs
    const int32_t inc = S.inv_ratio_q16;
    int n;
    for (;;) {
        n = in_len < S.batch_size ? in_len : S.batch_size;

        ar2(S.s_iir, &buf[S.fir_order], in, S.coefs, n);

        int32_t max_index = wshl(n, 16);
        out = down_fir_interpol(out, buf, fir, S.fir_order, S.fir_fracs,
                                max_index, inc);

        in += n;
        in_len -= n;
        // NOTE: `> 1`, not `> 0` — replicated from the reference.
        if (in_len > 1) {
            std::memcpy(buf, &buf[n], S.fir_order * sizeof(int32_t));
        } else {
            break;
        }
    }
    std::memcpy(S.s_fir.i32, &buf[n], S.fir_order * sizeof(int32_t));
}

}  // namespace

int Resampler::init(int in_khz, int out_khz) {
    std::memset(this, 0, sizeof(*this));

    if ((in_khz != 8 && in_khz != 12 && in_khz != 16) ||
        rate_id(out_khz) < 0) {
        return -1;
    }
    input_delay = kDelayDec[rate_id(in_khz)][rate_id(out_khz)];

    fs_in_khz = in_khz;
    fs_out_khz = out_khz;
    batch_size = in_khz * 10;  // 10 ms per inner batch

    // Ratio arithmetic runs on the Hz values, as in the reference.
    const int32_t fs_in = in_khz * 1000;
    const int32_t fs_out = out_khz * 1000;

    int up2x = 0;
    if (out_khz > in_khz) {
        if (out_khz == 2 * in_khz) {
            fn = kFnUp2Hq;  // exact 2x: 8->16, 12->24
        } else {
            fn = kFnIirFir;  // interpolate the 2x-upsampled signal
            up2x = 1;
        }
    } else if (out_khz < in_khz) {
        fn = kFnDownFir;
        if (out_khz * 4 == in_khz * 3) {         // 3:4  (16 -> 12)
            fir_fracs = 3;
            fir_order = kResamplerDownOrderFir0;
            coefs = kResampler34Coefs;
        } else if (out_khz * 3 == in_khz * 2) {  // 2:3  (12 -> 8)
            fir_fracs = 2;
            fir_order = kResamplerDownOrderFir0;
            coefs = kResampler23Coefs;
        } else if (out_khz * 2 == in_khz) {      // 1:2  (16 -> 8)
            fir_fracs = 1;
            fir_order = kResamplerDownOrderFir1;
            coefs = kResampler12Coefs;
        } else {
            return -1;  // 1:3 / 1:4 / 1:6 need fs_in >= 24: encoder-only
        }
    } else {
        fn = kFnCopy;
    }

    // Input/output ratio in Q16 (in the 2x domain for IIR_FIR), rounded up.
    inv_ratio_q16 = wshl(wshl(fs_in, 14 + up2x) / fs_out, 2);
    while (smulww(inv_ratio_q16, fs_out) < wshl(fs_in, up2x)) inv_ratio_q16++;

    return 0;
}

void Resampler::process(int16_t* out, const int16_t* in, int in_len) {
    assert(in_len >= fs_in_khz);          // at least 1 ms of input
    assert(input_delay <= fs_in_khz);

    // The first 1 ms comes through the delay buffer (delay equalization).
    int n = fs_in_khz - input_delay;
    std::memcpy(&delay_buf[input_delay], in, n * sizeof(int16_t));

    switch (fn) {
        case kFnUp2Hq:
            up2_hq(s_iir, out, delay_buf, fs_in_khz);
            up2_hq(s_iir, &out[fs_out_khz], &in[n], in_len - fs_in_khz);
            break;
        case kFnIirFir:
            iir_fir(*this, out, delay_buf, fs_in_khz);
            iir_fir(*this, &out[fs_out_khz], &in[n], in_len - fs_in_khz);
            break;
        case kFnDownFir:
            down_fir(*this, out, delay_buf, fs_in_khz);
            down_fir(*this, &out[fs_out_khz], &in[n], in_len - fs_in_khz);
            break;
        default:  // copy
            std::memcpy(out, delay_buf, fs_in_khz * sizeof(int16_t));
            std::memcpy(&out[fs_out_khz], &in[n],
                        (in_len - fs_in_khz) * sizeof(int16_t));
    }

    std::memcpy(delay_buf, &in[in_len - input_delay],
                input_delay * sizeof(int16_t));
}

}  // namespace silk
}  // namespace opus
}  // namespace glint
