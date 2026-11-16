// CELT band (PVQ shape) encoding — RFC 6716 section 4.3.4
// MIT License - Clean-room implementation

#include "opus_celt_enc_bands.hpp"

#include <cstring>

#include "opus_celt_bands.hpp"  // bitexact_cos/log2tan, celt_lcg_rand
#include "opus_celt_enc_vq.hpp"
#include "opus_celt_rate.hpp"
#include "opus_celt_tables.hpp"

namespace glint {
namespace opus {

namespace {

using celt::kEBands;
using celt::kNbEBands;

constexpr int kQThetaOffset = 4;
constexpr int kQThetaOffsetTwoPhase = 16;
constexpr int kMaxBandBuf = 176;

inline int imin(int a, int b) { return a < b ? a : b; }
inline int imax(int a, int b) { return a > b ? a : b; }

inline int frac_mul16(int a, int b) {
    return (16384 +
            static_cast<int32_t>(static_cast<int16_t>(a)) *
                static_cast<int16_t>(b)) >> 15;
}

// Float Haar butterfly (single-rounded ops only, like the reference).
void haar1f(float* x, int n0, int stride) {
    n0 >>= 1;
    for (int i = 0; i < stride; i++) {
        for (int j = 0; j < n0; j++) {
            float tmp1 = 0.70710678f * x[stride * 2 * j + i];
            float tmp2 = 0.70710678f * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = tmp1 + tmp2;
            x[stride * (2 * j + 1) + i] = tmp1 - tmp2;
        }
    }
}

const int kOrderyTable[] = {
    1, 0,
    3, 0, 2, 1,
    7, 0, 4, 3, 6, 1, 5, 2,
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
};

void deinterleave_hadamard(float* x, int n0, int stride, int hadamard) {
    float tmp[kMaxBandBuf];
    int n = n0 * stride;
    if (hadamard) {
        const int* ordery = kOrderyTable + stride - 2;
        for (int i = 0; i < stride; i++)
            for (int j = 0; j < n0; j++)
                tmp[ordery[i] * n0 + j] = x[j * stride + i];
    } else {
        for (int i = 0; i < stride; i++)
            for (int j = 0; j < n0; j++) tmp[i * n0 + j] = x[j * stride + i];
    }
    std::memcpy(x, tmp, n * sizeof(float));
}

int compute_qn(int n, int b, int offset, int pulse_cap, int stereo) {
    static const int16_t kExp2Table8[8] = { 16384, 17866, 19483, 21247,
                                            23170, 25267, 27554, 30048 };
    int n2 = 2 * n - 1;
    if (stereo && n == 2) n2--;
    int qb = (b + n2 * offset) / n2;
    qb = imin(b - pulse_cap - (4 << kBitRes), qb);
    qb = imin(8 << kBitRes, qb);
    int qn;
    if (qb < (1 << kBitRes >> 1)) {
        qn = 1;
    } else {
        qn = kExp2Table8[qb & 0x7] >> (14 - (qb >> kBitRes));
        qn = ((qn + 1) >> 1) << 1;
    }
    return qn;
}

struct EncBandCtx {
    int i;
    int intensity;
    int spread;
    int tf_change;
    RangeEncoder* enc;
    int32_t remaining_bits;
    const float* band_e;  // [2][kNbEBands]
    int avoid_split_noise;
};

struct SplitCtx {
    int inv;
    int imid;
    int iside;
    int delta;
    int itheta;
    int qalloc;
};

// Choose and ENCODE the mid/side angle for a split.
void compute_theta_enc(EncBandCtx* ctx, SplitCtx* sctx, float* x, float* y,
                       int n, int* b, int B, int B0, int lm, int stereo,
                       int* fill) {
    RangeEncoder& enc = *ctx->enc;
    int i = ctx->i;
    int inv = 0;
    int imid, iside, delta;

    int pulse_cap = celt::kLogN[i] + lm * (1 << kBitRes);
    int offset = (pulse_cap >> 1) -
                 (stereo && n == 2 ? kQThetaOffsetTwoPhase : kQThetaOffset);
    int qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if (stereo && i >= ctx->intensity) qn = 1;

    // theta from the (normalized, orthogonal) mid/side energies.
    int itheta = stereo_itheta(x, y, stereo, n);

    int32_t tell = static_cast<int32_t>(enc.tell_frac());
    if (qn != 1) {
        itheta = (itheta * static_cast<int32_t>(qn) + 8192) >> 14;
        // Mono transient splits: if the bit split would starve one half
        // into noise-fill, snap theta to put ALL energy on one side.
        if (!stereo && ctx->avoid_split_noise && itheta > 0 && itheta < qn) {
            int unq = static_cast<int>(
                (static_cast<uint32_t>(itheta) * 16384u) /
                static_cast<uint32_t>(qn));
            imid = bitexact_cos(static_cast<int16_t>(unq));
            iside = bitexact_cos(static_cast<int16_t>(16384 - unq));
            delta = frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid));
            if (delta > *b)
                itheta = qn;
            else if (delta < -*b)
                itheta = 0;
        }

        if (stereo && n > 2) {
            // Step PDF.
            const int p0 = 3;
            int x0 = qn / 2;
            uint32_t ft = static_cast<uint32_t>(p0 * (x0 + 1) + x0);
            int v = itheta;
            enc.encode(
                static_cast<uint32_t>(v <= x0 ? p0 * v
                                              : (v - 1 - x0) + (x0 + 1) * p0),
                static_cast<uint32_t>(v <= x0 ? p0 * (v + 1)
                                              : (v - x0) + (x0 + 1) * p0),
                ft);
        } else if (B0 > 1 || stereo) {
            enc.enc_uint(static_cast<uint32_t>(itheta),
                         static_cast<uint32_t>(qn + 1));
        } else {
            // Triangular PDF.
            int half = qn >> 1;
            uint32_t ft = static_cast<uint32_t>((half + 1) * (half + 1));
            int fs = itheta <= half ? itheta + 1 : qn + 1 - itheta;
            int fl = itheta <= half
                         ? (itheta * (itheta + 1)) >> 1
                         : static_cast<int>(ft) -
                               (((qn + 1 - itheta) * (qn + 2 - itheta)) >>
                                1);
            enc.encode(static_cast<uint32_t>(fl),
                       static_cast<uint32_t>(fl + fs), ft);
        }
        itheta = static_cast<int>((static_cast<uint32_t>(itheta) * 16384u) /
                                  static_cast<uint32_t>(qn));
        if (stereo) {
            if (itheta == 0)
                intensity_stereo(x, y, ctx->band_e, i, kNbEBands, n);
            else
                stereo_split(x, y, n);
        }
    } else if (stereo) {
        // qn == 1: side is a (possibly inverted) copy of mid.
        inv = itheta > 8192;
        if (inv) {
            for (int j = 0; j < n; j++) y[j] = -y[j];
        }
        intensity_stereo(x, y, ctx->band_e, i, kNbEBands, n);
        if (*b > 2 << kBitRes && ctx->remaining_bits > 2 << kBitRes)
            enc.enc_bit_logp(inv, 2);
        else
            inv = 0;
        itheta = 0;
    }
    int qalloc = static_cast<int>(enc.tell_frac()) - tell;
    *b -= qalloc;

    if (itheta == 0) {
        imid = 32767;
        iside = 0;
        *fill &= (1 << B) - 1;
        delta = -16384;
    } else if (itheta == 16384) {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << B) - 1) << B;
        delta = 16384;
    } else {
        imid = bitexact_cos(static_cast<int16_t>(itheta));
        iside = bitexact_cos(static_cast<int16_t>(16384 - itheta));
        delta = frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid));
    }

    sctx->inv = inv;
    sctx->imid = imid;
    sctx->iside = iside;
    sctx->delta = delta;
    sctx->itheta = itheta;
    sctx->qalloc = qalloc;
}

unsigned quant_band_n1_enc(EncBandCtx* ctx, const float* x, const float* y) {
    const float* c = x;
    for (int ch = 0; ch < (y ? 2 : 1); ch++) {
        if (ctx->remaining_bits >= 1 << kBitRes) {
            ctx->enc->enc_bits(c[0] < 0 ? 1 : 0, 1);
            ctx->remaining_bits -= 1 << kBitRes;
        }
        c = y;
    }
    return 1;
}

unsigned quant_band_enc(EncBandCtx* ctx, float* x, int n, int b, int B,
                        int lm, float gain, int fill);

unsigned quant_partition_enc(EncBandCtx* ctx, float* x, int n, int b, int B,
                             int lm, float gain, int fill) {
    int i = ctx->i;
    unsigned cm = 0;
    int B0 = B;

    const uint8_t* cache =
        celt::kCacheBits + celt::kCacheIndex[(lm + 1) * kNbEBands + i];
    if (lm != -1 && b > cache[cache[0]] + 12 && n > 2) {
        n >>= 1;
        float* y = x + n;
        lm -= 1;
        if (B == 1) fill = (fill & 1) | (fill << 1);
        B = (B + 1) >> 1;

        SplitCtx sctx;
        compute_theta_enc(ctx, &sctx, x, y, n, &b, B, B0, lm, 0, &fill);
        int itheta = sctx.itheta;
        int delta = sctx.delta;
        int qalloc = sctx.qalloc;
        float mid = (1.0f / 32768) * sctx.imid;
        float side = (1.0f / 32768) * sctx.iside;

        if (B0 > 1 && (itheta & 0x3fff)) {
            if (itheta > 8192)
                delta -= delta >> (4 - lm);
            else
                delta = imin(0, delta + (n << kBitRes >> (5 - lm)));
        }
        int mbits = imax(0, imin(b, (b - delta) / 2));
        int sbits = b - mbits;
        ctx->remaining_bits -= qalloc;

        int32_t rebalance = ctx->remaining_bits;
        if (mbits >= sbits) {
            cm = quant_partition_enc(ctx, x, n, mbits, B, lm, gain * mid,
                                     fill);
            rebalance = mbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 0)
                sbits += rebalance - (3 << kBitRes);
            cm |= quant_partition_enc(ctx, y, n, sbits, B, lm, gain * side,
                                      fill >> B)
                  << (B0 >> 1);
        } else {
            cm = quant_partition_enc(ctx, y, n, sbits, B, lm, gain * side,
                                     fill >> B)
                 << (B0 >> 1);
            rebalance = sbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 16384)
                mbits += rebalance - (3 << kBitRes);
            cm |= quant_partition_enc(ctx, x, n, mbits, B, lm, gain * mid,
                                      fill);
        }
    } else {
        // Leaf: one PVQ codeword.
        int q = bits2pulses(i, lm, b);
        int curr_bits = pulses2bits(i, lm, q);
        ctx->remaining_bits -= curr_bits;
        while (ctx->remaining_bits < 0 && q > 0) {
            ctx->remaining_bits += curr_bits;
            q--;
            curr_bits = pulses2bits(i, lm, q);
            ctx->remaining_bits -= curr_bits;
        }

        if (q != 0) {
            cm = alg_quant(x, n, get_pulses(q), ctx->spread, B, *ctx->enc,
                           gain, /*resynth=*/false);
        } else {
            // No pulses: the decoder folds/noise-fills. Without
            // resynthesis the reference leaves the mask at 0 (the
            // encoder-side masks only feed the anti-collapse policy).
            cm = 0;
        }
    }
    return cm;
}

unsigned quant_band_enc(EncBandCtx* ctx, float* x, int n, int b, int B,
                        int lm, float gain, int fill) {
    int n0 = n;
    int n_b = n;
    int B0 = B;
    int time_divide = 0;
    int recombine = 0;
    int tf_change = ctx->tf_change;
    int longblocks = B0 == 1;
    unsigned cm;

    n_b /= B;

    if (n == 1) return quant_band_n1_enc(ctx, x, nullptr);

    if (tf_change > 0) recombine = tf_change;

    for (int k = 0; k < recombine; k++) {
        static const uint8_t kBitInterleave[16] = { 0, 1, 1, 1, 2, 3, 3, 3,
                                                    2, 3, 3, 3, 2, 3, 3, 3 };
        haar1f(x, n >> k, 1 << k);
        fill = kBitInterleave[fill & 0xF] | kBitInterleave[fill >> 4] << 2;
    }
    B >>= recombine;
    n_b <<= recombine;

    while ((n_b & 1) == 0 && tf_change < 0) {
        haar1f(x, n_b, B);
        fill |= fill << B;
        B <<= 1;
        n_b >>= 1;
        time_divide++;
        tf_change++;
    }
    B0 = B;

    if (B0 > 1)
        deinterleave_hadamard(x, n_b >> recombine, B0 << recombine,
                              longblocks);

    cm = quant_partition_enc(ctx, x, n, b, B, lm, gain, fill);

    // Without resynthesis the reference returns the mask untransformed
    // (the undo path — interleave, Haar, mask folding — is resynth-only).
    (void)n0;
    (void)time_divide;
    return cm;
}

unsigned quant_band_stereo_enc(EncBandCtx* ctx, float* x, float* y, int n,
                               int b, int B, int lm, int fill) {
    RangeEncoder& enc = *ctx->enc;
    unsigned cm = 0;

    if (n == 1) return quant_band_n1_enc(ctx, x, y);

    int orig_fill = fill;
    SplitCtx sctx;
    compute_theta_enc(ctx, &sctx, x, y, n, &b, B, B, lm, 1, &fill);
    int itheta = sctx.itheta;
    int delta = sctx.delta;
    int qalloc = sctx.qalloc;
    float side = (1.0f / 32768) * sctx.iside;

    if (n == 2) {
        int sbits = (itheta != 0 && itheta != 16384) ? 1 << kBitRes : 0;
        int mbits = b - sbits;
        int c = itheta > 8192;
        ctx->remaining_bits -= qalloc + sbits;

        float* x2 = c ? y : x;
        float* y2 = c ? x : y;
        if (sbits) {
            // The side is orthogonal to the mid: one sign suffices.
            int sign = x2[0] * y2[1] - x2[1] * y2[0] < 0 ? 1 : 0;
            enc.enc_bits(static_cast<uint32_t>(sign), 1);
        }
        cm = quant_band_enc(ctx, x2, n, mbits, B, lm, 1.0f, orig_fill);
    } else {
        int mbits = imax(0, imin(b, (b - delta) / 2));
        int sbits = b - mbits;
        ctx->remaining_bits -= qalloc;

        int32_t rebalance = ctx->remaining_bits;
        if (mbits >= sbits) {
            cm = quant_band_enc(ctx, x, n, mbits, B, lm, 1.0f, fill);
            rebalance = mbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 0)
                sbits += rebalance - (3 << kBitRes);
            cm |= quant_band_enc(ctx, y, n, sbits, B, lm, side, fill >> B);
        } else {
            cm = quant_band_enc(ctx, y, n, sbits, B, lm, side, fill >> B);
            rebalance = sbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 16384)
                mbits += rebalance - (3 << kBitRes);
            cm |= quant_band_enc(ctx, x, n, mbits, B, lm, 1.0f, fill);
        }
    }
    return cm;
}

}  // namespace

void quant_all_bands_enc(int start, int end, float* X_, float* Y_,
                         uint8_t* collapse_masks, const float* band_e,
                         const int* pulses, int short_blocks, int spread,
                         int dual_stereo, int intensity, const int* tf_res,
                         int32_t total_bits, int32_t balance,
                         RangeEncoder& enc, int lm, int coded_bands,
                         uint32_t* seed) {
    const int C = Y_ != nullptr ? 2 : 1;
    const int M = 1 << lm;
    int B = short_blocks ? M : 1;

    EncBandCtx ctx;
    ctx.enc = &enc;
    ctx.intensity = intensity;
    ctx.spread = spread;
    ctx.band_e = band_e;
    // No noise injection on the first transient band's split.
    ctx.avoid_split_noise = B > 1;

    for (int i = start; i < end; i++) {
        ctx.i = i;
        float* X = X_ + M * kEBands[i];
        float* Y = Y_ != nullptr ? Y_ + M * kEBands[i] : nullptr;
        int N = M * kEBands[i + 1] - M * kEBands[i];
        int32_t tell = static_cast<int32_t>(enc.tell_frac());

        if (i != start) balance -= tell;
        int32_t remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        int b;
        if (i <= coded_bands - 1) {
            int32_t curr_balance = balance / imin(3, coded_bands - i);
            b = imax(0, imin(16383,
                             imin(static_cast<int>(remaining_bits + 1),
                                  pulses[i] +
                                      static_cast<int>(curr_balance))));
        } else {
            b = 0;
        }

        ctx.tf_change = tf_res[i];

        // resynth == 0: no folding source is ever established, so the
        // conservative estimate is always all-blocks-alive.
        unsigned x_cm = (1u << B) - 1;
        unsigned y_cm = x_cm;

        if (dual_stereo && i == intensity) dual_stereo = 0;

        if (dual_stereo) {
            x_cm = quant_band_enc(&ctx, X, N, b / 2, B, lm, 1.0f,
                                  static_cast<int>(x_cm));
            y_cm = quant_band_enc(&ctx, Y, N, b / 2, B, lm, 1.0f,
                                  static_cast<int>(y_cm));
        } else {
            if (Y != nullptr) {
                x_cm = quant_band_stereo_enc(&ctx, X, Y, N, b, B, lm,
                                             static_cast<int>(x_cm | y_cm));
            } else {
                x_cm = quant_band_enc(&ctx, X, N, b, B, lm, 1.0f,
                                      static_cast<int>(x_cm | y_cm));
            }
            y_cm = x_cm;
        }
        collapse_masks[i * C + 0] = static_cast<uint8_t>(x_cm);
        collapse_masks[i * C + C - 1] = static_cast<uint8_t>(y_cm);
        balance += pulses[i] + tell;

        ctx.avoid_split_noise = 0;
    }
    (void)seed;
}

}  // namespace opus
}  // namespace glint
