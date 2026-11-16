// CELT band (PVQ shape) decoding — RFC 6716 section 4.3.4
// MIT License - Clean-room implementation
//
// Decode-only port of the reference band machinery (encode branches and the
// encoder's theta-RDO dropped; resynth is unconditional in a decoder).
// Float-build macro semantics reduced to plain double arithmetic; every
// integer expression matches the reference exactly (wire-coupled).

#include "opus_celt_bands.hpp"

#include <cmath>
#include <cstring>

#include "opus_celt_energy.hpp"
#include "opus_celt_rate.hpp"
#include "opus_celt_tables.hpp"
#include "opus_cwrs.hpp"

namespace glint {
namespace opus {

namespace {

using celt::kEBands;
using celt::kNbEBands;

constexpr int kSpreadNone = 0;
constexpr int kSpreadAggressive = 3;
constexpr int kQThetaOffset = 4;
constexpr int kQThetaOffsetTwoPhase = 16;
constexpr int kMaxBandBuf = 176;  // largest band N (22 << 3)

inline int imin(int a, int b) { return a < b ? a : b; }
inline int imax(int a, int b) { return a > b ? a : b; }

// (16384 + a*b) >> 15 on 16-bit operands — the reference FRAC_MUL16.
inline int frac_mul16(int a, int b) {
    return (16384 +
            static_cast<int32_t>(static_cast<int16_t>(a)) *
                static_cast<int16_t>(b)) >> 15;
}

// Exact floor(sqrt(v)) for uint32.
uint32_t isqrt32(uint32_t v) {
    uint32_t g = 0;
    int bshift = (ec::ilog(v) - 1) >> 1;
    uint32_t b = 1u << bshift;
    do {
        uint32_t t = ((g << 1) + b) << bshift;
        if (t <= v) {
            g += b;
            v -= t;
        }
        b >>= 1;
        bshift--;
    } while (bshift >= 0);
    return g;
}

}  // namespace

int16_t bitexact_cos(int16_t x) {
    int32_t tmp = (4096 + static_cast<int32_t>(x) * x) >> 13;
    int16_t x2 = static_cast<int16_t>(tmp);
    x2 = static_cast<int16_t>(
        (32767 - x2) +
        frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2))));
    return static_cast<int16_t>(1 + x2);
}

int bitexact_log2tan(int isin, int icos) {
    int lc = ec::ilog(static_cast<uint32_t>(icos));
    int ls = ec::ilog(static_cast<uint32_t>(isin));
    icos <<= 15 - lc;
    isin <<= 15 - ls;
    return (ls - lc) * (1 << 11) +
           frac_mul16(isin, frac_mul16(isin, -2597) + 7932) -
           frac_mul16(icos, frac_mul16(icos, -2597) + 7932);
}

void haar1(double* x, int n0, int stride) {
    n0 >>= 1;
    for (int i = 0; i < stride; i++) {
        for (int j = 0; j < n0; j++) {
            double tmp1 = 0.70710678 * x[stride * 2 * j + i];
            double tmp2 = 0.70710678 * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = tmp1 + tmp2;
            x[stride * (2 * j + 1) + i] = tmp1 - tmp2;
        }
    }
}

namespace {

// ---------------------------------------------------------------------------
// VQ layer (reference vq.c, decode side)
// ---------------------------------------------------------------------------

void exp_rotation1(double* x, int len, int stride, double c, double s) {
    double ms = -s;
    double* p = x;
    for (int i = 0; i < len - stride; i++) {
        double x1 = p[0];
        double x2 = p[stride];
        p[stride] = c * x2 + s * x1;
        *p++ = c * x1 + ms * x2;
    }
    p = &x[len - 2 * stride - 1];
    for (int i = len - 2 * stride - 1; i >= 0; i--) {
        double x1 = p[0];
        double x2 = p[stride];
        p[stride] = c * x2 + s * x1;
        *p-- = c * x1 + ms * x2;
    }
}

void exp_rotation(double* x, int len, int dir, int stride, int k,
                  int spread) {
    static const int kSpreadFactor[3] = { 15, 10, 5 };
    if (2 * k >= len || spread == kSpreadNone) return;
    int factor = kSpreadFactor[spread - 1];
    double gain =
        static_cast<double>(len) / static_cast<double>(len + factor * k);
    double theta = 0.5 * gain * gain;
    double c = std::cos(0.5 * M_PI * theta);
    double s = std::cos(0.5 * M_PI * (1.0 - theta));  // sin(theta)

    int stride2 = 0;
    if (len >= 8 * stride) {
        stride2 = 1;
        while ((stride2 * stride2 + stride2) * stride + (stride >> 2) < len)
            stride2++;
    }
    len /= stride;
    for (int i = 0; i < stride; i++) {
        if (dir < 0) {
            if (stride2) exp_rotation1(x + i * len, len, stride2, s, c);
            exp_rotation1(x + i * len, len, 1, c, s);
        } else {
            exp_rotation1(x + i * len, len, 1, c, -s);
            if (stride2) exp_rotation1(x + i * len, len, stride2, s, -c);
        }
    }
}

unsigned extract_collapse_mask(const int* iy, int n, int b) {
    if (b <= 1) return 1;
    int n0 = n / b;
    unsigned mask = 0;
    for (int i = 0; i < b; i++) {
        unsigned tmp = 0;
        for (int j = 0; j < n0; j++)
            tmp |= static_cast<unsigned>(iy[i * n0 + j] != 0);
        mask |= (tmp != 0 ? 1u : 0u) << i;
    }
    return mask;
}

void renormalise_vector(double* x, int n, double gain) {
    double e = 1e-15;
    for (int i = 0; i < n; i++) e += x[i] * x[i];
    double g = gain / std::sqrt(e);
    for (int i = 0; i < n; i++) x[i] *= g;
}

unsigned alg_unquant(double* x, int n, int k, int spread, int b,
                     RangeDecoder& dec, double gain) {
    int iy[kMaxBandBuf];
    int32_t ryy = decode_pulses(iy, n, k, dec);
    double g = gain / std::sqrt(static_cast<double>(ryy));
    for (int i = 0; i < n; i++) x[i] = g * iy[i];
    exp_rotation(x, n, -1, b, k, spread);
    return extract_collapse_mask(iy, n, b);
}

// ---------------------------------------------------------------------------
// Band helpers
// ---------------------------------------------------------------------------

void stereo_merge(double* x, double* y, double mid, int n) {
    double xp = 0, side = 0;
    for (int j = 0; j < n; j++) {
        xp += y[j] * x[j];
        side += y[j] * y[j];
    }
    xp *= mid;
    double el = mid * mid + side - 2 * xp;
    double er = mid * mid + side + 2 * xp;
    if (er < 6e-4 || el < 6e-4) {
        std::memcpy(y, x, n * sizeof(double));
        return;
    }
    double lgain = 1.0 / std::sqrt(el);
    double rgain = 1.0 / std::sqrt(er);
    for (int j = 0; j < n; j++) {
        double l = mid * x[j];
        double r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

const int kOrderyTable[] = {
    1, 0,                                          // stride 2
    3, 0, 2, 1,                                    // stride 4
    7, 0, 4, 3, 6, 1, 5, 2,                        // stride 8
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,  // stride 16
};

void deinterleave_hadamard(double* x, int n0, int stride, int hadamard) {
    double tmp[kMaxBandBuf];
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
    std::memcpy(x, tmp, n * sizeof(double));
}

void interleave_hadamard(double* x, int n0, int stride, int hadamard) {
    double tmp[kMaxBandBuf];
    int n = n0 * stride;
    if (hadamard) {
        const int* ordery = kOrderyTable + stride - 2;
        for (int i = 0; i < stride; i++)
            for (int j = 0; j < n0; j++)
                tmp[j * stride + i] = x[ordery[i] * n0 + j];
    } else {
        for (int i = 0; i < stride; i++)
            for (int j = 0; j < n0; j++) tmp[j * stride + i] = x[i * n0 + j];
    }
    std::memcpy(x, tmp, n * sizeof(double));
}

int compute_qn(int n, int b, int offset, int pulse_cap, int stereo) {
    static const int16_t kExp2Table8[8] = { 16384, 17866, 19483, 21247,
                                            23170, 25267, 27554, 30048 };
    int n2 = 2 * n - 1;
    if (stereo && n == 2) n2--;
    // qb: bits available for theta, capped so the side keeps >=1 pulse.
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

struct BandCtx {
    int i;                    // current band
    int intensity;
    int spread;
    int tf_change;
    RangeDecoder* dec;
    int32_t remaining_bits;
    uint32_t seed;
    int disable_inv;
};

struct SplitCtx {
    int inv;
    int imid;
    int iside;
    int delta;
    int itheta;
    int qalloc;
};

// Decode the mid/side angle for a split (RFC 4.3.4.1). Chooses the theta
// PDF by context: step for stereo N>2, uniform for transients/stereo,
// triangular otherwise. Consumed bits (qalloc) are charged to *b.
void compute_theta(BandCtx* ctx, SplitCtx* sctx, int n, int* b, int B,
                   int B0, int lm, int stereo, int* fill) {
    RangeDecoder& dec = *ctx->dec;
    int i = ctx->i;
    int itheta = 0;
    int inv = 0;
    int imid, iside, delta;

    // Theta resolution scales with band size and budget.
    int pulse_cap = celt::kLogN[i] + lm * (1 << kBitRes);
    int offset = (pulse_cap >> 1) -
                 (stereo && n == 2 ? kQThetaOffsetTwoPhase : kQThetaOffset);
    int qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if (stereo && i >= ctx->intensity) qn = 1;
    int32_t tell = static_cast<int32_t>(dec.tell_frac());

    if (qn != 1) {
        if (stereo && n > 2) {
            // Step PDF: probability p0 per value up to qn/2, then 1 each.
            const int p0 = 3;
            int x0 = qn / 2;
            uint32_t ft = static_cast<uint32_t>(p0 * (x0 + 1) + x0);
            int fs = static_cast<int>(dec.decode(ft));
            int x = fs < (x0 + 1) * p0 ? fs / p0 : x0 + 1 + (fs - (x0 + 1) * p0);
            dec.dec_update(
                static_cast<uint32_t>(x <= x0 ? p0 * x
                                              : (x - 1 - x0) + (x0 + 1) * p0),
                static_cast<uint32_t>(x <= x0 ? p0 * (x + 1)
                                              : (x - x0) + (x0 + 1) * p0),
                ft);
            itheta = x;
        } else if (B0 > 1 || stereo) {
            // Uniform PDF.
            itheta = static_cast<int>(
                dec.dec_uint(static_cast<uint32_t>(qn + 1)));
        } else {
            // Triangular PDF.
            int half = qn >> 1;
            uint32_t ft = static_cast<uint32_t>((half + 1) * (half + 1));
            int fm = static_cast<int>(dec.decode(ft));
            int fs, fl;
            if (fm < ((half * (half + 1)) >> 1)) {
                itheta = static_cast<int>(
                             (isqrt32(8 * static_cast<uint32_t>(fm) + 1) -
                              1)) >> 1;
                fs = itheta + 1;
                fl = (itheta * (itheta + 1)) >> 1;
            } else {
                itheta =
                    (2 * (qn + 1) -
                     static_cast<int>(isqrt32(
                         8 * static_cast<uint32_t>(static_cast<int>(ft) -
                                                   fm - 1) +
                         1))) >> 1;
                fs = qn + 1 - itheta;
                fl = static_cast<int>(ft) -
                     (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
            }
            dec.dec_update(static_cast<uint32_t>(fl),
                           static_cast<uint32_t>(fl + fs), ft);
        }
        itheta = static_cast<int>(
            (static_cast<uint32_t>(itheta) * 16384u) /
            static_cast<uint32_t>(qn));
    } else if (stereo) {
        // qn==1: only a possible inversion flag survives.
        if (*b > 2 << kBitRes && ctx->remaining_bits > 2 << kBitRes)
            inv = dec.dec_bit_logp(2);
        else
            inv = 0;
        if (ctx->disable_inv) inv = 0;
        itheta = 0;
    }
    int qalloc = static_cast<int>(dec.tell_frac()) - tell;
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
        // Mid/side split minimizing squared error for this angle.
        delta = frac_mul16((n - 1) << 7, bitexact_log2tan(iside, imid));
    }

    sctx->inv = inv;
    sctx->imid = imid;
    sctx->iside = iside;
    sctx->delta = delta;
    sctx->itheta = itheta;
    sctx->qalloc = qalloc;
}

unsigned quant_band_n1(BandCtx* ctx, double* x, double* y,
                       double* lowband_out) {
    RangeDecoder& dec = *ctx->dec;
    int stereo = y != nullptr;
    double* c = x;
    for (int ch = 0; ch < 1 + stereo; ch++) {
        int sign = 0;
        if (ctx->remaining_bits >= 1 << kBitRes) {
            sign = static_cast<int>(dec.dec_bits(1));
            ctx->remaining_bits -= 1 << kBitRes;
        }
        c[0] = sign ? -1.0 : 1.0;
        c = y;
    }
    if (lowband_out) lowband_out[0] = x[0];
    return 1;
}

unsigned quant_band(BandCtx* ctx, double* x, int n, int b, int B,
                    double* lowband, int lm, double* lowband_out,
                    double gain, double* lowband_scratch, int fill);

// Decode one mono partition, recursively splitting while the budget
// exceeds what the PVQ cache can spend on a single codeword.
unsigned quant_partition(BandCtx* ctx, double* x, int n, int b, int B,
                         double* lowband, int lm, double gain, int fill) {
    RangeDecoder& dec = *ctx->dec;
    int i = ctx->i;
    unsigned cm = 0;
    int B0 = B;

    const uint8_t* cache =
        celt::kCacheBits + celt::kCacheIndex[(lm + 1) * kNbEBands + i];
    if (lm != -1 && b > cache[cache[0]] + 12 && n > 2) {
        n >>= 1;
        double* y = x + n;
        lm -= 1;
        if (B == 1) fill = (fill & 1) | (fill << 1);
        B = (B + 1) >> 1;

        SplitCtx sctx;
        compute_theta(ctx, &sctx, n, &b, B, B0, lm, 0, &fill);
        int imid = sctx.imid;
        int iside = sctx.iside;
        int delta = sctx.delta;
        int itheta = sctx.itheta;
        int qalloc = sctx.qalloc;
        double mid = (1.0 / 32768) * imid;
        double side = (1.0 / 32768) * iside;

        // Favor low-energy halves of transient frames (pre-echo/forward
        // masking approximations).
        if (B0 > 1 && (itheta & 0x3fff)) {
            if (itheta > 8192)
                delta -= delta >> (4 - lm);
            else
                delta = imin(0, delta + (n << kBitRes >> (5 - lm)));
        }
        int mbits = imax(0, imin(b, (b - delta) / 2));
        int sbits = b - mbits;
        ctx->remaining_bits -= qalloc;

        double* next_lowband2 = lowband ? lowband + n : nullptr;

        int32_t rebalance = ctx->remaining_bits;
        if (mbits >= sbits) {
            cm = quant_partition(ctx, x, n, mbits, B, lowband, lm,
                                 gain * mid, fill);
            rebalance = mbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 0)
                sbits += rebalance - (3 << kBitRes);
            cm |= quant_partition(ctx, y, n, sbits, B, next_lowband2, lm,
                                  gain * side, fill >> B)
                  << (B0 >> 1);
        } else {
            cm = quant_partition(ctx, y, n, sbits, B, next_lowband2, lm,
                                 gain * side, fill >> B)
                 << (B0 >> 1);
            rebalance = sbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 16384)
                mbits += rebalance - (3 << kBitRes);
            cm |= quant_partition(ctx, x, n, mbits, B, lowband, lm,
                                  gain * mid, fill);
        }
    } else {
        // Leaf: one PVQ codeword (or a noise/fold fill if no pulses fit).
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
            cm = alg_unquant(x, n, get_pulses(q), ctx->spread, B, dec, gain);
        } else {
            unsigned cm_mask = static_cast<unsigned>((1ul << B) - 1);
            fill &= static_cast<int>(cm_mask);
            if (!fill) {
                std::memset(x, 0, n * sizeof(double));
            } else {
                if (lowband == nullptr) {
                    // Pure noise fill.
                    for (int j = 0; j < n; j++) {
                        ctx->seed = celt_lcg_rand(ctx->seed);
                        x[j] = static_cast<double>(
                            static_cast<int32_t>(ctx->seed) >> 20);
                    }
                    cm = cm_mask;
                } else {
                    // Fold the lower band, with ±1/256 dither.
                    for (int j = 0; j < n; j++) {
                        ctx->seed = celt_lcg_rand(ctx->seed);
                        double tmp = (ctx->seed & 0x8000) ? (1.0 / 256)
                                                          : -(1.0 / 256);
                        x[j] = lowband[j] + tmp;
                    }
                    cm = static_cast<unsigned>(fill);
                }
                renormalise_vector(x, n, gain);
            }
        }
    }
    return cm;
}

// Decode one band (mono path): handles TF resolution changes (Haar), the
// Hadamard reordering of transient frames, then delegates to
// quant_partition and undoes the time/frequency reorganization.
unsigned quant_band(BandCtx* ctx, double* x, int n, int b, int B,
                    double* lowband, int lm, double* lowband_out,
                    double gain, double* lowband_scratch, int fill) {
    int n0 = n;
    int n_b = n;
    int B0 = B;
    int time_divide = 0;
    int recombine = 0;
    int tf_change = ctx->tf_change;
    int longblocks = B0 == 1;
    unsigned cm;

    n_b /= B;

    if (n == 1) return quant_band_n1(ctx, x, nullptr, lowband_out);

    if (tf_change > 0) recombine = tf_change;

    if (lowband_scratch && lowband &&
        (recombine || ((n_b & 1) == 0 && tf_change < 0) || B0 > 1)) {
        std::memcpy(lowband_scratch, lowband, n * sizeof(double));
        lowband = lowband_scratch;
    }

    for (int k = 0; k < recombine; k++) {
        static const uint8_t kBitInterleave[16] = { 0, 1, 1, 1, 2, 3, 3, 3,
                                                    2, 3, 3, 3, 2, 3, 3, 3 };
        if (lowband) haar1(lowband, n >> k, 1 << k);
        fill = kBitInterleave[fill & 0xF] |
               kBitInterleave[fill >> 4] << 2;
    }
    B >>= recombine;
    n_b <<= recombine;

    while ((n_b & 1) == 0 && tf_change < 0) {
        if (lowband) haar1(lowband, n_b, B);
        fill |= fill << B;
        B <<= 1;
        n_b >>= 1;
        time_divide++;
        tf_change++;
    }
    B0 = B;
    int n_b0 = n_b;

    if (B0 > 1 && lowband)
        deinterleave_hadamard(lowband, n_b >> recombine, B0 << recombine,
                              longblocks);

    cm = quant_partition(ctx, x, n, b, B, lowband, lm, gain, fill);

    // Resynthesis: undo the time/frequency reorganization on the output.
    if (B0 > 1)
        interleave_hadamard(x, n_b >> recombine, B0 << recombine,
                            longblocks);
    n_b = n_b0;
    B = B0;
    for (int k = 0; k < time_divide; k++) {
        B >>= 1;
        n_b <<= 1;
        cm |= cm >> B;
        haar1(x, n_b, B);
    }
    for (int k = 0; k < recombine; k++) {
        static const uint8_t kBitDeinterleave[16] = {
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F,
            0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF
        };
        cm = kBitDeinterleave[cm];
        haar1(x, n0 >> k, 1 << k);
    }
    B <<= recombine;

    // Scale for later folding: restore sqrt(N) so folded copies sit at the
    // right level relative to a unit-norm band.
    if (lowband_out) {
        double scale = std::sqrt(static_cast<double>(n0));
        for (int j = 0; j < n0; j++) lowband_out[j] = scale * x[j];
    }
    cm &= (1u << B) - 1;
    return cm;
}

// Decode one stereo band: theta joins the channels; N==2 uses the
// orthogonal-side trick (one sign bit).
unsigned quant_band_stereo(BandCtx* ctx, double* x, double* y, int n, int b,
                           int B, double* lowband, int lm,
                           double* lowband_out, double* lowband_scratch,
                           int fill) {
    RangeDecoder& dec = *ctx->dec;
    unsigned cm = 0;

    if (n == 1) return quant_band_n1(ctx, x, y, lowband_out);

    int orig_fill = fill;
    SplitCtx sctx;
    compute_theta(ctx, &sctx, n, &b, B, B, lm, 1, &fill);
    int inv = sctx.inv;
    int imid = sctx.imid;
    int iside = sctx.iside;
    int delta = sctx.delta;
    int itheta = sctx.itheta;
    int qalloc = sctx.qalloc;
    double mid = (1.0 / 32768) * imid;
    double side = (1.0 / 32768) * iside;

    if (n == 2) {
        int sbits = (itheta != 0 && itheta != 16384) ? 1 << kBitRes : 0;
        int mbits = b - sbits;
        int c = itheta > 8192;
        ctx->remaining_bits -= qalloc + sbits;

        double* x2 = c ? y : x;
        double* y2 = c ? x : y;
        int sign = 0;
        if (sbits) sign = static_cast<int>(dec.dec_bits(1));
        sign = 1 - 2 * sign;
        // orig_fill: the side folds even when itheta cleared fill's lows.
        cm = quant_band(ctx, x2, n, mbits, B, lowband, lm, lowband_out, 1.0,
                        lowband_scratch, orig_fill);
        y2[0] = -sign * x2[1];
        y2[1] = sign * x2[0];
        double tmp;
        x[0] = mid * x[0];
        x[1] = mid * x[1];
        y[0] = side * y[0];
        y[1] = side * y[1];
        tmp = x[0];
        x[0] = tmp - y[0];
        y[0] = tmp + y[0];
        tmp = x[1];
        x[1] = tmp - y[1];
        y[1] = tmp + y[1];
    } else {
        int mbits = imax(0, imin(b, (b - delta) / 2));
        int sbits = b - mbits;
        ctx->remaining_bits -= qalloc;

        int32_t rebalance = ctx->remaining_bits;
        if (mbits >= sbits) {
            // Mid gets no gain scaling: its normalized form feeds folding.
            cm = quant_band(ctx, x, n, mbits, B, lowband, lm, lowband_out,
                            1.0, lowband_scratch, fill);
            rebalance = mbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 0)
                sbits += rebalance - (3 << kBitRes);
            // Side never folds (fill's high bits are zero in stereo).
            cm |= quant_band(ctx, y, n, sbits, B, nullptr, lm, nullptr,
                             side, nullptr, fill >> B);
        } else {
            cm = quant_band(ctx, y, n, sbits, B, nullptr, lm, nullptr, side,
                            nullptr, fill >> B);
            rebalance = sbits - (rebalance - ctx->remaining_bits);
            if (rebalance > 3 << kBitRes && itheta != 16384)
                mbits += rebalance - (3 << kBitRes);
            cm |= quant_band(ctx, x, n, mbits, B, lowband, lm, lowband_out,
                             1.0, lowband_scratch, fill);
        }
    }

    if (n != 2) stereo_merge(x, y, mid, n);
    if (inv) {
        for (int j = 0; j < n; j++) y[j] = -y[j];
    }
    return cm;
}

// Hybrid mode starts at band 17 with a single narrow coded band below the
// second one; duplicate folding data so band start+1 can fold.
void special_hybrid_folding(double* norm, double* norm2, int start, int m,
                            int dual_stereo) {
    int n1 = m * (kEBands[start + 1] - kEBands[start]);
    int n2 = m * (kEBands[start + 2] - kEBands[start + 1]);
    std::memmove(&norm[n1], &norm[2 * n1 - n2],
                 (n2 - n1) * sizeof(double));
    if (dual_stereo)
        std::memmove(&norm2[n1], &norm2[2 * n1 - n2],
                     (n2 - n1) * sizeof(double));
}

}  // namespace

void quant_all_bands_dec(int start, int end, double* X_, double* Y_,
                         uint8_t* collapse_masks, const int* pulses,
                         int short_blocks, int spread, int dual_stereo,
                         int intensity, const int* tf_res,
                         int32_t total_bits, int32_t balance,
                         RangeDecoder& dec, int lm, int coded_bands,
                         uint32_t* seed, int disable_inv) {
    const int C = Y_ != nullptr ? 2 : 1;
    const int M = 1 << lm;
    int B = short_blocks ? M : 1;
    const int norm_offset = M * kEBands[start];
    int lowband_offset = 0;
    int update_lowband = 1;

    // Folding source: normalized output of previously decoded bands
    // (per channel in dual stereo). The last band never feeds folding.
    static thread_local double norm_buf[2 * 8 * 78];
    double* norm = norm_buf;
    double* norm2 = norm + M * kEBands[kNbEBands - 1] - norm_offset;
    static thread_local double scratch_buf[8 * 22];  // largest band
    double* lowband_scratch = scratch_buf;

    BandCtx ctx;
    ctx.dec = &dec;
    ctx.intensity = intensity;
    ctx.spread = spread;
    ctx.seed = *seed;
    ctx.disable_inv = disable_inv;

    for (int i = start; i < end; i++) {
        ctx.i = i;
        int last = (i == end - 1);
        double* X = X_ + M * kEBands[i];
        double* Y = Y_ != nullptr ? Y_ + M * kEBands[i] : nullptr;
        int N = M * kEBands[i + 1] - M * kEBands[i];
        int32_t tell = static_cast<int32_t>(dec.tell_frac());

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

        if ((M * kEBands[i] - N >= M * kEBands[start] || i == start + 1) &&
            (update_lowband || lowband_offset == 0))
            lowband_offset = i;
        if (i == start + 1)
            special_hybrid_folding(norm, norm2, start, M, dual_stereo);

        ctx.tf_change = tf_res[i];
        if (last) lowband_scratch = nullptr;

        // Conservative collapse estimate for the folding source region.
        unsigned x_cm, y_cm;
        int effective_lowband = -1;
        if (lowband_offset != 0 &&
            (spread != kSpreadAggressive || B > 1 || ctx.tf_change < 0)) {
            // Never fold a band onto itself.
            effective_lowband =
                imax(0, M * kEBands[lowband_offset] - norm_offset - N);
            int fold_start = lowband_offset;
            while (M * kEBands[--fold_start] >
                   effective_lowband + norm_offset) {
            }
            int fold_end = lowband_offset - 1;
            while (++fold_end < i &&
                   M * kEBands[fold_end] <
                       effective_lowband + norm_offset + N) {
            }
            x_cm = y_cm = 0;
            int fold_i = fold_start;
            do {
                x_cm |= collapse_masks[fold_i * C + 0];
                y_cm |= collapse_masks[fold_i * C + C - 1];
            } while (++fold_i < fold_end);
        } else {
            // LCG noise fill: (almost) always non-zero blocks.
            x_cm = y_cm = (1u << B) - 1;
        }

        if (dual_stereo && i == intensity) {
            // Switch to intensity: collapse the two folding histories.
            dual_stereo = 0;
            for (int j = 0; j < M * kEBands[i] - norm_offset; j++)
                norm[j] = 0.5 * (norm[j] + norm2[j]);
        }
        if (dual_stereo) {
            x_cm = quant_band(
                &ctx, X, N, b / 2, B,
                effective_lowband != -1 ? norm + effective_lowband : nullptr,
                lm, last ? nullptr : norm + M * kEBands[i] - norm_offset,
                1.0, lowband_scratch, static_cast<int>(x_cm));
            y_cm = quant_band(
                &ctx, Y, N, b / 2, B,
                effective_lowband != -1 ? norm2 + effective_lowband
                                        : nullptr,
                lm, last ? nullptr : norm2 + M * kEBands[i] - norm_offset,
                1.0, lowband_scratch, static_cast<int>(y_cm));
        } else {
            if (Y != nullptr) {
                x_cm = quant_band_stereo(
                    &ctx, X, Y, N, b, B,
                    effective_lowband != -1 ? norm + effective_lowband
                                            : nullptr,
                    lm,
                    last ? nullptr : norm + M * kEBands[i] - norm_offset,
                    lowband_scratch, static_cast<int>(x_cm | y_cm));
            } else {
                x_cm = quant_band(
                    &ctx, X, N, b, B,
                    effective_lowband != -1 ? norm + effective_lowband
                                            : nullptr,
                    lm,
                    last ? nullptr : norm + M * kEBands[i] - norm_offset,
                    1.0, lowband_scratch, static_cast<int>(x_cm | y_cm));
            }
            y_cm = x_cm;
        }
        collapse_masks[i * C + 0] = static_cast<uint8_t>(x_cm);
        collapse_masks[i * C + C - 1] = static_cast<uint8_t>(y_cm);
        balance += pulses[i] + tell;

        // Folding position tracks only while depth >= 1 bit/sample.
        update_lowband = b > (N << kBitRes);
    }
    *seed = ctx.seed;
}

void anti_collapse(double* X_, const uint8_t* collapse_masks, int lm,
                   int channels, int size, int start, int end,
                   const double* log_e, const double* prev1_log_e,
                   const double* prev2_log_e, const int* pulses,
                   uint32_t seed) {
    for (int i = start; i < end; i++) {
        int n0 = kEBands[i + 1] - kEBands[i];
        // Depth the band was coded at, in bits/sample.
        int depth = static_cast<int>(
                        (1u + static_cast<unsigned>(pulses[i])) /
                        static_cast<unsigned>(n0)) >> lm;
        double thresh = 0.5 * std::exp2(-0.125 * depth);
        double sqrt_1 = 1.0 / std::sqrt(static_cast<double>(n0 << lm));

        for (int c = 0; c < channels; c++) {
            double prev1 = prev1_log_e[c * kNbEBands + i];
            double prev2 = prev2_log_e[c * kNbEBands + i];
            if (channels == 1) {
                prev1 = prev1 > prev1_log_e[kNbEBands + i]
                            ? prev1
                            : prev1_log_e[kNbEBands + i];
                prev2 = prev2 > prev2_log_e[kNbEBands + i]
                            ? prev2
                            : prev2_log_e[kNbEBands + i];
            }
            double ediff =
                log_e[c * kNbEBands + i] - (prev1 < prev2 ? prev1 : prev2);
            if (ediff < 0) ediff = 0;
            // Short blocks carry less energy per block than long ones.
            double r = 2.0 * std::exp2(-ediff);
            if (lm == 3) r *= 1.41421356;
            r = r < thresh ? r : thresh;
            r = r * sqrt_1;

            double* x = X_ + c * size + (kEBands[i] << lm);
            int renormalize = 0;
            for (int k = 0; k < 1 << lm; k++) {
                if (!(collapse_masks[i * channels + c] & (1u << k))) {
                    // This block collapsed to zero: refill with noise.
                    for (int j = 0; j < n0; j++) {
                        seed = celt_lcg_rand(seed);
                        x[(j << lm) + k] = (seed & 0x8000) ? r : -r;
                    }
                    renormalize = 1;
                }
            }
            if (renormalize) renormalise_vector(x, n0 << lm, 1.0);
        }
    }
}

}  // namespace opus
}  // namespace glint

