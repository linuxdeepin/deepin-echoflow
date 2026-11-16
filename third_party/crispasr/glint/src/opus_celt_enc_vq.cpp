// CELT band (PVQ shape) encoding — VQ layer, RFC 6716 section 4.3.4
// MIT License - Clean-room implementation
//
// See the header for the float32 wire-criticality story. Notation used in
// the comments below: "unfused" = the product is rounded to float32 first
// and then added (two roundings, what the reference build's vectorized loop
// bodies do), "fused" = std::fmaf single rounding (what its scalar tails
// do). The split point between the two is n & ~3 in every loop the
// reference build vectorizes.

#include "opus_celt_enc_vq.hpp"

#include <cmath>

#include "opus_cwrs.hpp"

namespace glint {
namespace opus {

namespace {

constexpr int kSpreadNone = 0;
constexpr float kPi = 3.141592653f;     // reference celt/mathops.h PI
constexpr float kEpsilon = 1e-15f;      // reference float-build EPSILON

// (float)cos of a half-turn fraction: celt_cos_norm(x) = cos((pi/2)*x).
// The pi/2 factor is the float product 0.5f*kPi; the argument product is
// float; the cosine itself runs in double and is rounded back to float.
inline float celt_cos_norm(float x) {
    return static_cast<float>(std::cos(static_cast<double>((0.5f * kPi) * x)));
}

// One rotation pass. Element recurrence (both loops):
//   p[stride] = c*x2 + s*x1,  p[0] = c*x1 - s*x2
// with the SECOND product rounded separately and the first fused (that is
// how the reference build compiles `MAC16_16(MULT16_16(c,x2), s, x1)`, in
// both its scalar and its vectorized form).
void exp_rotation1(float* x, int len, int stride, float c, float s) {
    float ms = -s;
    float* p = x;
    for (int i = 0; i < len - stride; i++) {
        float x1 = p[0];
        float x2 = p[stride];
        float t1 = s * x1;
        p[stride] = std::fmaf(c, x2, t1);
        float t2 = ms * x2;
        *p++ = std::fmaf(c, x1, t2);
    }
    p = &x[len - 2 * stride - 1];
    for (int i = len - 2 * stride - 1; i >= 0; i--) {
        float x1 = p[0];
        float x2 = p[stride];
        float t1 = s * x1;
        p[stride] = std::fmaf(c, x2, t1);
        float t2 = ms * x2;
        *p-- = std::fmaf(c, x1, t2);
    }
}

// Mix the residual pulses down to the unit sphere: x = gain/sqrt(ryy) * iy.
// Only single-rounded operations (sqrt/div/mul), so the values are
// codegen-independent. (float)sqrt(float) is exact double-rounding-safe.
void normalise_residual(const int* iy, float* x, int n, float ryy,
                        float gain) {
    float g = (1.f / std::sqrt(ryy)) * gain;
    for (int i = 0; i < n; i++) x[i] = g * static_cast<float>(iy[i]);
}

// Which of the b interleaved blocks got at least one pulse. Mirrors the
// reference's do/while loops (they touch iy[0] even when n < b).
unsigned extract_collapse_mask(const int* iy, int n, int b) {
    if (b <= 1) return 1;
    int n0 = n / b;
    unsigned mask = 0;
    int i = 0;
    do {
        unsigned tmp = 0;
        int j = 0;
        do {
            tmp |= static_cast<unsigned>(iy[i * n0 + j] != 0);
        } while (++j < n0);
        mask |= (tmp != 0 ? 1u : 0u) << i;
    } while (++i < b);
    return mask;
}

// celt_inner_prod exactly as the reference arm64 float build computes it
// (celt/arm/pitch_neon_intr.c celt_inner_prod_neon): four fused lanes over
// blocks of four (the 8-wide source loop is two such steps), reduced as
// (lane0+lane2) + (lane1+lane3), then a fused scalar tail.
float inner_prod(const float* x, const float* y, int n) {
    float lane0 = 0.f, lane1 = 0.f, lane2 = 0.f, lane3 = 0.f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        lane0 = std::fmaf(x[i + 0], y[i + 0], lane0);
        lane1 = std::fmaf(x[i + 1], y[i + 1], lane1);
        lane2 = std::fmaf(x[i + 2], y[i + 2], lane2);
        lane3 = std::fmaf(x[i + 3], y[i + 3], lane3);
    }
    float lo = lane0 + lane2;
    float hi = lane1 + lane3;
    float xy = lo + hi;
    for (; i < n; i++) xy = std::fmaf(x[i], y[i], xy);
    return xy;
}

// Reference celt/mathops.h fast_atan2f, float build. The (e + coef*e2)
// terms are fused (scalar codegen), everything else single-rounded.
float fast_atan2f(float y, float x) {
    constexpr float kA = 0.43157974f;
    constexpr float kB = 0.67848403f;
    constexpr float kC = 0.08595542f;
    constexpr float kE = kPi / 2;
    float x2 = x * x;
    float y2 = y * y;
    // For very small values the answer doesn't matter — avoid 0/0.
    if (x2 + y2 < 1e-18f) return 0;
    if (x2 < y2) {
        float den = std::fmaf(x2, kB, y2) * std::fmaf(x2, kC, y2);
        float num = std::fmaf(x2, kA, y2);
        float p = x * y;
        float q = (p * num) / den;
        return (y < 0 ? -kE : kE) - q;
    } else {
        float den = std::fmaf(y2, kB, x2) * std::fmaf(y2, kC, x2);
        float num = std::fmaf(y2, kA, x2);
        float p = x * y;
        float q = (p * num) / den;
        return q + (y < 0 ? -kE : kE) - (p < 0 ? -kE : kE);
    }
}

}  // namespace

void exp_rotation(float* x, int len, int dir, int stride, int k, int spread) {
    static const int kSpreadFactor[3] = { 15, 10, 5 };
    if (2 * k >= len || spread == kSpreadNone) return;
    int factor = kSpreadFactor[spread - 1];
    float gain = static_cast<float>(len) /
                 static_cast<float>(len + factor * k);
    float g2 = gain * gain;
    float theta = 0.5f * g2;
    float c = celt_cos_norm(theta);
    float s = celt_cos_norm(1.0f - theta);  // sin(theta)

    int stride2 = 0;
    if (len >= 8 * stride) {
        // sqrt(len/stride) with rounding: increment while
        // (stride2+0.5)^2 < len/stride.
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

float op_pvq_search(float* x, int* iy, int k, int n) {
    float y[kVqMaxBandSize];
    int signx[kVqMaxBandSize];

    // Strip the signs; the search runs on |x| and the argmax is then
    // guaranteed a nonnegative correlation.
    for (int j = 0; j < n; j++) {
        signx[j] = x[j] < 0;
        x[j] = std::fabs(x[j]);
        iy[j] = 0;
        y[j] = 0;
    }

    float xy = 0;  // running correlation <x, y>
    float yy = 0;  // running energy <y, y> (exact integers)
    int pulses_left = k;

    // Pre-search: project onto the pyramid when there are many pulses.
    if (k > (n >> 1)) {
        float sum = 0;
        for (int j = 0; j < n; j++) sum = sum + x[j];
        // Degenerate input (near-silent, huge, or NaN): replace with a
        // single pulse at 0. "64 is an approximation of infinity here."
        if (!(sum > kEpsilon && sum < 64)) {
            x[0] = 1.0f;
            for (int j = 1; j < n; j++) x[j] = 0;
            sum = 1.0f;
        }
        // k + e with e < 1 guarantees no more than k pulses.
        float kf = static_cast<float>(k) + 0.8f;
        float rcp = kf * (1.f / sum);
        // The reference build vectorizes [0, n & ~3) (unfused accumulator
        // updates) and contracts the scalar remainder to fma.
        int nv = n & ~3;
        int j = 0;
        for (; j < nv; j++) {
            float t = rcp * x[j];
            iy[j] = static_cast<int>(std::floor(t));
            y[j] = static_cast<float>(iy[j]);
            float yp = y[j] * y[j];
            yy = yy + yp;
            float xp = x[j] * y[j];
            xy = xy + xp;
            // y is kept doubled so the per-pulse update below is an add.
            y[j] = y[j] + y[j];
            pulses_left -= iy[j];
        }
        for (; j < n; j++) {
            float t = rcp * x[j];
            iy[j] = static_cast<int>(std::floor(t));
            y[j] = static_cast<float>(iy[j]);
            yy = std::fmaf(y[j], y[j], yy);
            xy = std::fmaf(x[j], y[j], xy);
            y[j] = y[j] + y[j];
            pulses_left -= iy[j];
        }
    }

    // Should never happen except on silence: dump the excess into bin 0.
    if (pulses_left > n + 3) {
        float tmp = static_cast<float>(pulses_left);
        yy = std::fmaf(tmp, tmp, yy);
        yy = std::fmaf(tmp, y[0], yy);
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    for (int i = 0; i < pulses_left; i++) {
        int best_id = 0;
        // The +1 from (y_j + 1)^2 is position-independent — add it once.
        yy = yy + 1;
        // Position 0 seeds the running best (all ops single-rounded; the
        // argmax loop is deliberately fusion-free in the reference too).
        float rxy = xy + x[0];
        float ryy = yy + y[0];
        rxy = rxy * rxy;
        float best_den = ryy;
        float best_num = rxy;
        for (int j = 1; j < n; j++) {
            rxy = xy + x[j];
            ryy = yy + y[j];
            // Maximize Rxy^2/Ryy, compared cross-multiplied (no division).
            rxy = rxy * rxy;
            if (best_den * rxy > ryy * best_num) {
                best_den = ryy;
                best_num = rxy;
                best_id = j;
            }
        }
        xy = xy + x[best_id];
        yy = yy + y[best_id];  // y is doubled: adds the cross term 2*y_j
        y[best_id] += 2;
        iy[best_id]++;
    }

    // Put the signs back (branchless form, same result as ?:).
    for (int j = 0; j < n; j++) iy[j] = (iy[j] ^ -signx[j]) + signx[j];
    return yy;
}

unsigned alg_quant(float* x, int n, int k, int spread, int b,
                   RangeEncoder& enc, float gain, bool resynth) {
    int iy[kVqMaxBandSize];
    exp_rotation(x, n, 1, b, k, spread);
    float yy = op_pvq_search(x, iy, k, n);
    encode_pulses(iy, n, k, enc);
    if (resynth) {
        normalise_residual(iy, x, n, yy, gain);
        exp_rotation(x, n, -1, b, k, spread);
    }
    return extract_collapse_mask(iy, n, b);
}

void renormalise_vector(float* x, int n, float gain) {
    float e = kEpsilon + inner_prod(x, x, n);
    float g = (1.f / std::sqrt(e)) * gain;
    for (int i = 0; i < n; i++) x[i] = g * x[i];
}

void stereo_split(float* x, float* y, int n) {
    for (int j = 0; j < n; j++) {
        float l = 0.70710678f * x[j];
        float r = 0.70710678f * y[j];
        x[j] = l + r;
        y[j] = r - l;
    }
}

void intensity_stereo(float* x, const float* y, const float* band_e,
                      int band, int nb_bands, int n) {
    float left = band_e[band];
    float right = band_e[band + nb_bands];
    // norm = EPSILON + sqrt(EPSILON + left^2 + right^2); both squares fuse
    // (verified against the reference build's inlined codegen in bands.o).
    float t = std::fmaf(left, left, kEpsilon);
    t = std::fmaf(right, right, t);
    float norm = kEpsilon + std::sqrt(t);
    float a1 = left / norm;
    float a2 = right / norm;
    for (int j = 0; j < n; j++) {
        float l = x[j];
        float r = y[j];
        // a1*l fused over the separately rounded a2*r (reference codegen).
        float u = a2 * r;
        x[j] = std::fmaf(a1, l, u);
        // Side is not encoded, no need to compute it.
    }
}

int stereo_itheta(const float* x, const float* y, int stereo, int n) {
    float emid = kEpsilon;
    float eside = kEpsilon;
    if (stereo) {
        // Reference build: vectorized unfused body over [0, n & ~3),
        // fma scalar tail.
        int nv = n & ~3;
        int i = 0;
        for (; i < nv; i++) {
            float m = x[i] + y[i];
            float s = x[i] - y[i];
            float mp = m * m;
            emid = emid + mp;
            float sp = s * s;
            eside = eside + sp;
        }
        for (; i < n; i++) {
            float m = x[i] + y[i];
            float s = x[i] - y[i];
            emid = std::fmaf(m, m, emid);
            eside = std::fmaf(s, s, eside);
        }
    } else {
        emid = emid + inner_prod(x, x, n);
        eside = eside + inner_prod(y, y, n);
    }
    float mid = std::sqrt(emid);
    float side = std::sqrt(eside);
    // 0.63662 = 2/pi; the scale-and-bias is one fused rounding, then floor.
    return static_cast<int>(std::floor(
        std::fmaf(fast_atan2f(side, mid), 16384 * 0.63662f, .5f)));
}

}  // namespace opus
}  // namespace glint
