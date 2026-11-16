// SILK NLSF decoding and NLSF-to-LPC conversion — RFC 6716 section 4.2.7.5
// MIT License - Clean-room implementation

#include "opus_silk_nlsf.hpp"

#include "opus_silk_indices.hpp"  // nlsf_unpack
#include "opus_silk_math.hpp"
#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

namespace {

// Quantizer dead-zone adjust: 0.1 in Q10.
constexpr int32_t kNlsfQuantLevelAdjQ10 = 102;

// (a * b) >> q, rounded, through the full 64-bit product.
inline int32_t mul32_frac_q(int32_t a, int32_t b, int q) {
    return static_cast<int32_t>(
        rshift_round64(static_cast<int64_t>(a) * b, q));
}

// Stage-2 residual dequantizer. Runs BACKWARDS (coef order-1 .. 0): each
// step predicts from the coefficient above it (pred_q8 backward predictor),
// applies the ±0.1 dead-zone level adjust to the index, and scales by the
// codebook's quantization step.
void nlsf_residual_dequant(int16_t* x_q10, const int8_t* indices,
                           const uint8_t* pred_q8, int quant_step_q16,
                           int order) {
    int32_t out_q10 = 0;
    for (int i = order - 1; i >= 0; i--) {
        int32_t pred_q10 = smulbb(out_q10, pred_q8[i]) >> 8;
        out_q10 = indices[i] << 10;
        if (out_q10 > 0) {
            out_q10 -= kNlsfQuantLevelAdjQ10;
        } else if (out_q10 < 0) {
            out_q10 += kNlsfQuantLevelAdjQ10;
        }
        out_q10 = smlawb(pred_q10, out_q10, quant_step_q16);
        x_q10[i] = static_cast<int16_t>(out_q10);
    }
}

// Enforce the codebook's minimum NLSF spacing (delta_min[order+1] entries:
// gap below coef 0, between neighbors, and above coef order-1). Up to 20
// minimum-distance repairs (move the worst-violating pair apart around its
// center frequency, clamped so the outer gaps stay feasible); if violations
// persist, fall back to sort + forward/backward clamps.
void nlsf_stabilize(int16_t* nlsf_q15, const int16_t* delta_min_q15,
                    int order) {
    constexpr int kMaxLoops = 20;
    int loops;
    for (loops = 0; loops < kMaxLoops; loops++) {
        // Most-negative slack over all order+1 gaps.
        int32_t min_diff = nlsf_q15[0] - delta_min_q15[0];
        int worst = 0;
        for (int i = 1; i <= order - 1; i++) {
            int32_t diff = nlsf_q15[i] - (nlsf_q15[i - 1] + delta_min_q15[i]);
            if (diff < min_diff) {
                min_diff = diff;
                worst = i;
            }
        }
        int32_t diff = (1 << 15) - (nlsf_q15[order - 1] + delta_min_q15[order]);
        if (diff < min_diff) {
            min_diff = diff;
            worst = order;
        }
        if (min_diff >= 0) return;

        if (worst == 0) {
            nlsf_q15[0] = delta_min_q15[0];
        } else if (worst == order) {
            nlsf_q15[order - 1] =
                static_cast<int16_t>((1 << 15) - delta_min_q15[order]);
        } else {
            // Feasible range for the pair's center frequency given the
            // minimum gaps accumulated from each end.
            int32_t min_center = 0;
            for (int k = 0; k < worst; k++) min_center += delta_min_q15[k];
            min_center += delta_min_q15[worst] >> 1;
            int32_t max_center = 1 << 15;
            for (int k = order; k > worst; k--) max_center -= delta_min_q15[k];
            max_center -= delta_min_q15[worst] >> 1;

            int32_t center = rshift_round(
                static_cast<int32_t>(nlsf_q15[worst - 1]) + nlsf_q15[worst], 1);
            if (center < min_center) center = min_center;
            if (center > max_center) center = max_center;
            int16_t center_q15 = static_cast<int16_t>(center);
            nlsf_q15[worst - 1] =
                static_cast<int16_t>(center_q15 - (delta_min_q15[worst] >> 1));
            nlsf_q15[worst] =
                static_cast<int16_t>(nlsf_q15[worst - 1] + delta_min_q15[worst]);
        }
    }

    if (loops == kMaxLoops) {
        // Fallback: insertion sort, then clamp gaps forward from the low
        // rail and backward from the high rail.
        for (int i = 1; i < order; i++) {
            int16_t value = nlsf_q15[i];
            int j = i - 1;
            for (; j >= 0 && value < nlsf_q15[j]; j--)
                nlsf_q15[j + 1] = nlsf_q15[j];
            nlsf_q15[j + 1] = value;
        }
        if (nlsf_q15[0] < delta_min_q15[0]) nlsf_q15[0] = delta_min_q15[0];
        for (int i = 1; i < order; i++) {
            int32_t floor_i = add_sat16(nlsf_q15[i - 1], delta_min_q15[i]);
            if (nlsf_q15[i] < floor_i)
                nlsf_q15[i] = static_cast<int16_t>(floor_i);
        }
        int32_t hi = (1 << 15) - delta_min_q15[order];
        if (nlsf_q15[order - 1] > hi)
            nlsf_q15[order - 1] = static_cast<int16_t>(hi);
        for (int i = order - 2; i >= 0; i--) {
            int32_t ceil_i = nlsf_q15[i + 1] - delta_min_q15[i + 1];
            if (nlsf_q15[i] > ceil_i)
                nlsf_q15[i] = static_cast<int16_t>(ceil_i);
        }
    }
}

// P/Q polynomial from the interleaved 2*cos(LSF) values (QA = Q16):
// out(z) = prod (1 - 2 cos(w_k) z^-1 + z^-2), built by convolution with
// 64-bit rounded products.
void nlsf2a_find_poly(int32_t* out, const int32_t* clsf_qa, int dd) {
    out[0] = 1 << 16;
    out[1] = -clsf_qa[0];
    for (int k = 1; k < dd; k++) {
        int32_t ftmp = clsf_qa[2 * k];
        out[k + 1] = (out[k - 1] << 1) -
                     mul32_frac_q(ftmp, out[k], 16);
        for (int n = k; n > 1; n--)
            out[n] += out[n - 2] - mul32_frac_q(ftmp, out[n - 1], 16);
        out[1] -= ftmp;
    }
}

// Fit Q(q_in) int32 coefficients into int16 at Q(q_out): up to 10 rounds of
// bandwidth expansion sized so the largest coefficient lands at the int16
// rail, then saturate as a last resort (writing the clipped values back so
// the caller's int32 copy stays consistent).
void lpc_fit(int16_t* a_qout, int32_t* a_qin, int q_out, int q_in, int d) {
    int i;
    int idx = 0;
    for (i = 0; i < 10; i++) {
        int32_t maxabs = 0;
        for (int k = 0; k < d; k++) {
            int32_t absval = a_qin[k] > 0 ? a_qin[k] : -a_qin[k];
            if (absval > maxabs) {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = rshift_round(maxabs, q_in - q_out);
        if (maxabs <= 32767) break;

        // Chirp such that coefficient idx shrinks to ~int16 max
        // (0.999 in Q16 minus the normalized overshoot).
        if (maxabs > 163838) maxabs = 163838;  // (2^31-1 >> 14) + 2^15-1
        int32_t chirp_q16 =
            65470 - ((maxabs - 32767) << 14) / ((maxabs * (idx + 1)) >> 2);
        bwexpander_32(a_qin, d, chirp_q16);
    }

    if (i == 10) {
        for (int k = 0; k < d; k++) {
            a_qout[k] = static_cast<int16_t>(
                sat16(rshift_round(a_qin[k], q_in - q_out)));
            a_qin[k] = static_cast<int32_t>(a_qout[k]) << (q_in - q_out);
        }
    } else {
        for (int k = 0; k < d; k++)
            a_qout[k] =
                static_cast<int16_t>(rshift_round(a_qin[k], q_in - q_out));
    }
}

}  // namespace

void nlsf_decode(int16_t* nlsf_q15, const int8_t* indices,
                 const NlsfCodebook& cb) {
    uint8_t pred_q8[kMaxLpcOrder];
    int16_t ec_ix[kMaxLpcOrder];
    int16_t res_q10[kMaxLpcOrder];

    // ec_ix is unused here (it drives the index DECODE, already done), but
    // unpack computes both from the same selector bytes.
    nlsf_unpack(ec_ix, pred_q8, cb, indices[0]);
    nlsf_residual_dequant(res_q10, &indices[1], pred_q8, cb.quantStepSizeQ16,
                          cb.order);

    const uint8_t* cb1 = &cb.cb1NlsfQ8[indices[0] * cb.order];
    const int16_t* wght_q9 = &cb.cb1WghtQ9[indices[0] * cb.order];
    for (int i = 0; i < cb.order; i++) {
        // Residual weighted by the inverse codebook weight
        // (Q10<<14 / Q9 = Q15), plus the stage-1 vector (Q8<<7 = Q15).
        int32_t v = (static_cast<int32_t>(res_q10[i]) << 14) / wght_q9[i] +
                    (static_cast<int32_t>(cb1[i]) << 7);
        nlsf_q15[i] =
            static_cast<int16_t>(v < 0 ? 0 : (v > 32767 ? 32767 : v));
    }

    nlsf_stabilize(nlsf_q15, cb.deltaMinQ15, cb.order);
}

void nlsf2a(int16_t* a_q12, const int16_t* nlsf_q15, int order) {
    // Interleave order chosen by the reference for numerical accuracy of
    // the polynomial convolution — part of the bit-exact contract.
    static constexpr uint8_t kOrdering16[16] = {0, 15, 8,  7, 4, 11, 12, 3,
                                                2, 13, 10, 5, 6, 9,  14, 1};
    static constexpr uint8_t kOrdering10[10] = {0, 9, 6, 3, 4, 5, 8, 1, 2, 7};
    const uint8_t* ordering = order == 16 ? kOrdering16 : kOrdering10;

    // NLSF -> 2*cos(pi*f), piecewise-linear over the 128-entry Q12 table,
    // rounded into Q16.
    int32_t cos_lsf_qa[kMaxLpcOrder];
    for (int k = 0; k < order; k++) {
        int32_t f_int = nlsf_q15[k] >> 8;   // table cell, 0..127
        int32_t f_frac = nlsf_q15[k] - (f_int << 8);
        int32_t cos_val = kLsfCosTabFixQ12[f_int];
        int32_t delta = kLsfCosTabFixQ12[f_int + 1] - cos_val;
        cos_lsf_qa[ordering[k]] =
            rshift_round((cos_val << 8) + delta * f_frac, 20 - 16);
    }

    int dd = order >> 1;
    int32_t p[kMaxLpcOrder / 2 + 1];
    int32_t q[kMaxLpcOrder / 2 + 1];
    nlsf2a_find_poly(p, &cos_lsf_qa[0], dd);  // even roots
    nlsf2a_find_poly(q, &cos_lsf_qa[1], dd);  // odd roots

    // A(z) from the symmetric/antisymmetric halves, Q17.
    int32_t a32_qa1[kMaxLpcOrder];
    for (int k = 0; k < dd; k++) {
        int32_t ptmp = p[k + 1] + p[k];
        int32_t qtmp = q[k + 1] - q[k];
        a32_qa1[k] = -qtmp - ptmp;
        a32_qa1[order - k - 1] = qtmp - ptmp;
    }

    lpc_fit(a_q12, a32_qa1, 12, 17, order);

    // If still (too close to) unstable, chirp progressively harder on the
    // int32 copy and re-round; up to 16 rounds (the last ones flatten the
    // filter completely).
    for (int i = 0;
         lpc_inverse_pred_gain(a_q12, order) == 0 && i < 16; i++) {
        bwexpander_32(a32_qa1, order, 65536 - (2 << i));
        for (int k = 0; k < order; k++)
            a_q12[k] = static_cast<int16_t>(rshift_round(a32_qa1[k], 17 - 12));
    }
}

void bwexpander_32(int32_t* ar, int order, int32_t chirp_q16) {
    int32_t chirp_minus_one_q16 = chirp_q16 - 65536;
    for (int i = 0; i < order - 1; i++) {
        ar[i] = smulww(chirp_q16, ar[i]);
        chirp_q16 += rshift_round(chirp_q16 * chirp_minus_one_q16, 16);
    }
    ar[order - 1] = smulww(chirp_q16, ar[order - 1]);
}

int32_t lpc_inverse_pred_gain(const int16_t* a_q12, int order) {
    // Filter poles at/inside the unit circle iff every reflection
    // coefficient of the downdated filter has |rc| < 1; the decoder also
    // rejects "legal" filters whose prediction gain exceeds 1e4.
    constexpr int kQA = 24;
    constexpr int32_t kALimitQA = 16773022;        // 0.99975 in Q24
    constexpr int32_t kInvMaxPredGainQ30 = 107374;  // 1/1e4 in Q30

    int32_t a_qa[kMaxLpcOrder];
    int32_t dc_resp = 0;
    for (int k = 0; k < order; k++) {
        dc_resp += a_q12[k];
        a_qa[k] = static_cast<int32_t>(a_q12[k]) << (kQA - 12);
    }
    // DC gain >= 1 is unstable without any recursion.
    if (dc_resp >= 4096) return 0;

    int32_t inv_gain_q30 = 1 << 30;
    for (int k = order - 1; k >= 0; k--) {
        if (a_qa[k] > kALimitQA || a_qa[k] < -kALimitQA) return 0;

        int32_t rc_q31 = -(a_qa[k] << (31 - kQA));
        // 1 - rc^2, range (0, 1] in Q30.
        int32_t rc_mult1_q30 = (1 << 30) - smmul(rc_q31, rc_q31);
        inv_gain_q30 = smmul(inv_gain_q30, rc_mult1_q30) << 2;
        if (inv_gain_q30 < kInvMaxPredGainQ30) return 0;
        if (k == 0) break;

        // Downdate: a'_n = (a_n - rc*a_{k-1-n}) / (1 - rc^2), computed at
        // a per-step Q chosen from the magnitude of 1 - rc^2.
        int mult2q = 32 - clz32(static_cast<uint32_t>(
                              rc_mult1_q30 > 0 ? rc_mult1_q30 : -rc_mult1_q30));
        int32_t rc_mult2 = inverse32_varq(rc_mult1_q30, mult2q + 30);
        for (int n = 0; n < (k + 1) >> 1; n++) {
            int32_t tmp1 = a_qa[n];
            int32_t tmp2 = a_qa[k - n - 1];
            int64_t t64 = rshift_round64(
                static_cast<int64_t>(
                    sub_sat32(tmp1, mul32_frac_q(tmp2, rc_q31, 31))) *
                    rc_mult2,
                mult2q);
            if (t64 > INT32_MAX || t64 < INT32_MIN) return 0;
            a_qa[n] = static_cast<int32_t>(t64);
            t64 = rshift_round64(
                static_cast<int64_t>(
                    sub_sat32(tmp2, mul32_frac_q(tmp1, rc_q31, 31))) *
                    rc_mult2,
                mult2q);
            if (t64 > INT32_MAX || t64 < INT32_MIN) return 0;
            a_qa[k - n - 1] = static_cast<int32_t>(t64);
        }
    }
    return inv_gain_q30;
}

}  // namespace silk
}  // namespace opus
}  // namespace glint
