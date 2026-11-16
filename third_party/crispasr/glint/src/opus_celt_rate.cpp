// CELT implicit bit allocation — RFC 6716 section 4.3.3
// MIT License - Clean-room implementation

#include "opus_celt_rate.hpp"

#include "opus_celt_energy.hpp"  // kMaxFineBits
#include "opus_celt_tables.hpp"

namespace glint {
namespace opus {

namespace {

using celt::kEBands;
using celt::kNbEBands;

constexpr int kAllocSteps = 6;   // 1/64th interpolation between quality rows
constexpr int kFineOffset = 21;

// log2(n+1) in 1/8 bits, for the intensity-position uint's cost.
constexpr uint8_t kLog2FracTable[24] = {
    0,  8,  13, 16, 19, 21, 23, 24, 26, 27, 28, 29,
    30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
};

inline int imin(int a, int b) { return a < b ? a : b; }
inline int imax(int a, int b) { return a > b ? a : b; }

}  // namespace

int bits2pulses(int band, int lm, int bits) {
    lm++;
    const uint8_t* cache =
        celt::kCacheBits + celt::kCacheIndex[lm * kNbEBands + band];
    int lo = 0;
    int hi = cache[0];
    bits--;
    for (int i = 0; i < kLogMaxPseudo; i++) {
        int mid = (lo + hi + 1) >> 1;
        if (static_cast<int>(cache[mid]) >= bits)
            hi = mid;
        else
            lo = mid;
    }
    // Pick the closer of the bracketing entries (lo maps to -1 bits at 0).
    if (bits - (lo == 0 ? -1 : static_cast<int>(cache[lo])) <=
        static_cast<int>(cache[hi]) - bits)
        return lo;
    return hi;
}

int pulses2bits(int band, int lm, int pulses) {
    lm++;
    const uint8_t* cache =
        celt::kCacheBits + celt::kCacheIndex[lm * kNbEBands + band];
    return pulses == 0 ? 0 : cache[pulses] + 1;
}

void init_caps(int* cap, int lm, int channels) {
    for (int i = 0; i < kNbEBands; i++) {
        int n = (kEBands[i + 1] - kEBands[i]) << lm;
        cap[i] = (celt::kCacheCaps[kNbEBands * (2 * lm + channels - 1) + i] +
                  64) *
                 channels * n >> 2;
    }
}

namespace {

// Second stage: interpolate between two allocation rows to land on the
// budget, decode band skips, split each band into fine-energy vs PVQ bits.
int interp_bits2pulses(int start, int end, int skip_start, const int* bits1,
                       const int* bits2, const int* thresh, const int* cap,
                       int32_t total, int32_t* out_balance, int skip_rsv,
                       int* intensity, int intensity_rsv, int* dual_stereo,
                       int dual_stereo_rsv, int* bits, int* ebits,
                       int* fine_priority, int C, int lm,
                       RangeEncoder* enc, RangeDecoder* dec, int prev,
                       int signal_bandwidth) {
    const int alloc_floor = C << kBitRes;
    const int stereo = C > 1 ? 1 : 0;
    const int log_m = lm << kBitRes;
    int32_t psum;
    int done;

    // Bisect the interpolation fraction (1/64ths above row lo).
    int lo = 0;
    int hi = 1 << kAllocSteps;
    for (int i = 0; i < kAllocSteps; i++) {
        int mid = (lo + hi) >> 1;
        psum = 0;
        done = 0;
        for (int j = end; j-- > start;) {
            int tmp = bits1[j] +
                      static_cast<int>(
                          (static_cast<int32_t>(mid) * bits2[j]) >>
                          kAllocSteps);
            if (tmp >= thresh[j] || done) {
                done = 1;
                psum += imin(tmp, cap[j]);
            } else if (tmp >= alloc_floor) {
                psum += alloc_floor;
            }
        }
        if (psum > total)
            hi = mid;
        else
            lo = mid;
    }
    psum = 0;
    done = 0;
    for (int j = end; j-- > start;) {
        int tmp = bits1[j] +
                  static_cast<int>(
                      (static_cast<int32_t>(lo) * bits2[j]) >> kAllocSteps);
        if (tmp < thresh[j] && !done) {
            tmp = tmp >= alloc_floor ? alloc_floor : 0;
        } else {
            done = 1;
        }
        tmp = imin(tmp, cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Band-skip decisions, from the top band down. A skipped band's bits
    // (minus the flag) are folded back into the pool; the intensity
    // reservation shrinks as the coded range shrinks.
    int coded_bands;
    for (coded_bands = end;; coded_bands--) {
        int j = coded_bands - 1;
        if (j <= skip_start) {
            total += skip_rsv;  // no skip flag gets coded after all
            break;
        }
        int32_t left = total - psum;
        int32_t percoeff =
            left / (kEBands[coded_bands] - kEBands[start]);
        left -= (kEBands[coded_bands] - kEBands[start]) * percoeff;
        int rem = imax(static_cast<int>(left - (kEBands[j] - kEBands[start])),
                       0);
        int band_width = kEBands[coded_bands] - kEBands[j];
        int band_bits =
            static_cast<int>(bits[j] + percoeff * band_width + rem);
        if (band_bits >= imax(thresh[j], alloc_floor + (1 << kBitRes))) {
            if (enc) {
                // Stop-skipping policy: keep bands with enough depth (with
                // hysteresis via prev) inside the signalled bandwidth.
                int depth_threshold;
                if (coded_bands > 17)
                    depth_threshold = j < prev ? 7 : 9;
                else
                    depth_threshold = 0;
                if (coded_bands <= start + 2 ||
                    (band_bits >
                         ((depth_threshold * band_width << lm << kBitRes) >>
                          4) &&
                     j <= signal_bandwidth)) {
                    enc->enc_bit_logp(1, 1);
                    break;
                }
                enc->enc_bit_logp(0, 1);
            } else if (dec->dec_bit_logp(1)) {
                break;
            }
            psum += 1 << kBitRes;
            band_bits -= 1 << kBitRes;
        }
        psum -= bits[j] + intensity_rsv;
        if (intensity_rsv > 0)
            intensity_rsv = kLog2FracTable[j - start];
        psum += intensity_rsv;
        if (band_bits >= alloc_floor) {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
    }

    if (intensity_rsv > 0) {
        if (enc) {
            *intensity = imin(*intensity, coded_bands);
            enc->enc_uint(static_cast<uint32_t>(*intensity - start),
                          static_cast<uint32_t>(coded_bands + 1 - start));
        } else {
            *intensity = start + static_cast<int>(dec->dec_uint(
                                     coded_bands + 1 - start));
        }
    } else {
        *intensity = 0;
    }
    if (*intensity <= start) {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if (dual_stereo_rsv > 0) {
        if (enc)
            enc->enc_bit_logp(*dual_stereo, 1);
        else
            *dual_stereo = dec->dec_bit_logp(1);
    } else {
        *dual_stereo = 0;
    }

    // Spread what's left uniformly per coefficient, remainder to the lowest
    // bands one coefficient's worth at a time.
    int32_t left = total - psum;
    int32_t percoeff = left / (kEBands[coded_bands] - kEBands[start]);
    left -= (kEBands[coded_bands] - kEBands[start]) * percoeff;
    for (int j = start; j < coded_bands; j++)
        bits[j] += static_cast<int>(percoeff) *
                   (kEBands[j + 1] - kEBands[j]);
    for (int j = start; j < coded_bands; j++) {
        int tmp = imin(static_cast<int>(left), kEBands[j + 1] - kEBands[j]);
        bits[j] += tmp;
        left -= tmp;
    }

    // Split each band's budget into fine-energy and PVQ bits.
    int32_t balance = 0;
    int j;
    for (j = start; j < coded_bands; j++) {
        int n0 = kEBands[j + 1] - kEBands[j];
        int n = n0 << lm;
        int32_t bit = static_cast<int32_t>(bits[j]) + balance;
        int32_t excess;

        if (n > 1) {
            excess = imax(static_cast<int>(bit - cap[j]), 0);
            bits[j] = static_cast<int>(bit - excess);

            // Stereo above the intensity point codes an extra theta DoF.
            int den = C * n +
                      ((C == 2 && n > 2 && !*dual_stereo && j < *intensity)
                           ? 1
                           : 0);
            int nclogn = den * (celt::kLogN[j] + log_m);
            int offset = (nclogn >> 1) - den * kFineOffset;
            if (n == 2) offset += den << kBitRes >> 2;
            // Bias the second and third fine bits.
            if (bits[j] + offset < (den * 2) << kBitRes)
                offset += nclogn >> 2;
            else if (bits[j] + offset < (den * 3) << kBitRes)
                offset += nclogn >> 3;

            ebits[j] = imax(0, bits[j] + offset + (den << (kBitRes - 1)));
            ebits[j] = static_cast<int>(
                static_cast<uint32_t>(ebits[j]) /
                static_cast<uint32_t>(den)) >> kBitRes;
            if (C * ebits[j] > (bits[j] >> kBitRes))
                ebits[j] = bits[j] >> stereo >> kBitRes;
            ebits[j] = imin(ebits[j], kMaxFineBits);
            fine_priority[j] =
                ebits[j] * (den << kBitRes) >= bits[j] + offset;
            bits[j] -= C * ebits[j] << kBitRes;
        } else {
            // One coefficient: all fine energy except a sign bit.
            excess = imax(0, static_cast<int>(bit) - (C << kBitRes));
            bits[j] = static_cast<int>(bit - excess);
            ebits[j] = 0;
            fine_priority[j] = 1;
        }

        // Over-cap bits can't help PVQ; convert to fine energy here.
        if (excess > 0) {
            int extra_fine = imin(static_cast<int>(excess >> (stereo + kBitRes)),
                                  kMaxFineBits - ebits[j]);
            ebits[j] += extra_fine;
            int32_t extra_bits = static_cast<int32_t>(extra_fine) * C
                                 << kBitRes;
            fine_priority[j] = extra_bits >= excess - balance;
            excess -= extra_bits;
        }
        balance = excess;
    }
    *out_balance = balance;

    // Skipped bands: whatever they kept becomes fine energy.
    for (; j < end; j++) {
        ebits[j] = bits[j] >> stereo >> kBitRes;
        bits[j] = 0;
        fine_priority[j] = ebits[j] < 1;
    }
    return coded_bands;
}

}  // namespace

namespace {
int compute_allocation_impl(int start, int end, const int* offsets,
                            const int* cap, int alloc_trim, int* intensity,
                            int* dual_stereo, int32_t total,
                            int32_t* balance, int* pulses, int* ebits,
                            int* fine_priority, int channels, int lm,
                            RangeEncoder* enc, RangeDecoder* dec, int prev,
                            int signal_bandwidth) {
    const int C = channels;
    int bits1[kNbEBands], bits2[kNbEBands];
    int thresh[kNbEBands], trim_offset[kNbEBands];

    total = imax(static_cast<int>(total), 0);
    int skip_start = start;
    // One bit to end manual band skipping, if we can afford it.
    int skip_rsv = total >= 1 << kBitRes ? 1 << kBitRes : 0;
    total -= skip_rsv;
    // Intensity position and dual-stereo flag reservations.
    int intensity_rsv = 0, dual_stereo_rsv = 0;
    if (C == 2) {
        intensity_rsv = kLog2FracTable[end - start];
        if (intensity_rsv > total) {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = total >= 1 << kBitRes ? 1 << kBitRes : 0;
            total -= dual_stereo_rsv;
        }
    }

    for (int j = start; j < end; j++) {
        // Below this, a band certainly gets no PVQ bits.
        thresh[j] = imax(C << kBitRes,
                         (3 * (kEBands[j + 1] - kEBands[j]) << lm
                          << kBitRes) >> 4);
        // Allocation-curve tilt from alloc_trim (5 = neutral).
        trim_offset[j] = C * (kEBands[j + 1] - kEBands[j]) *
                         (alloc_trim - 5 - lm) * (end - j - 1) *
                         (1 << (lm + kBitRes)) >> 6;
        // Single-coefficient bands live mostly off their coarse energy.
        if ((kEBands[j + 1] - kEBands[j]) << lm == 1)
            trim_offset[j] -= C << kBitRes;
    }

    // Bisect the quality rows of the static allocation table.
    int lo = 1;
    int hi = celt::kNbAllocVectors - 1;
    do {
        int done = 0;
        int psum = 0;
        int mid = (lo + hi) >> 1;
        for (int j = end; j-- > start;) {
            int n = kEBands[j + 1] - kEBands[j];
            int bitsj = C * n * celt::kBandAllocation[mid * kNbEBands + j]
                        << lm >> 2;
            if (bitsj > 0) bitsj = imax(0, bitsj + trim_offset[j]);
            bitsj += offsets[j];
            if (bitsj >= thresh[j] || done) {
                done = 1;
                psum += imin(bitsj, cap[j]);
            } else if (bitsj >= C << kBitRes) {
                psum += C << kBitRes;
            }
        }
        if (psum > total)
            hi = mid - 1;
        else
            lo = mid + 1;
    } while (lo <= hi);
    hi = lo--;

    for (int j = start; j < end; j++) {
        int n = kEBands[j + 1] - kEBands[j];
        int bits1j = C * n * celt::kBandAllocation[lo * kNbEBands + j]
                     << lm >> 2;
        int bits2j = hi >= celt::kNbAllocVectors
                         ? cap[j]
                         : C * n * celt::kBandAllocation[hi * kNbEBands + j]
                               << lm >> 2;
        if (bits1j > 0) bits1j = imax(0, bits1j + trim_offset[j]);
        if (bits2j > 0) bits2j = imax(0, bits2j + trim_offset[j]);
        if (lo > 0) bits1j += offsets[j];
        bits2j += offsets[j];
        if (offsets[j] > 0) skip_start = j;  // dynalloc-boosted: never skip
        bits2j = imax(0, bits2j - bits1j);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    return interp_bits2pulses(start, end, skip_start, bits1, bits2, thresh,
                              cap, total, balance, skip_rsv, intensity,
                              intensity_rsv, dual_stereo, dual_stereo_rsv,
                              pulses, ebits, fine_priority, C, lm, enc, dec,
                              prev, signal_bandwidth);
}
}  // namespace

int compute_allocation_dec(int start, int end, const int* offsets,
                           const int* cap, int alloc_trim, int* intensity,
                           int* dual_stereo, int32_t total,
                           int32_t* balance, int* pulses, int* ebits,
                           int* fine_priority, int channels, int lm,
                           RangeDecoder& dec) {
    return compute_allocation_impl(start, end, offsets, cap, alloc_trim,
                                   intensity, dual_stereo, total, balance,
                                   pulses, ebits, fine_priority, channels,
                                   lm, nullptr, &dec, 0, 0);
}

int compute_allocation_enc(int start, int end, const int* offsets,
                           const int* cap, int alloc_trim, int* intensity,
                           int* dual_stereo, int32_t total,
                           int32_t* balance, int* pulses, int* ebits,
                           int* fine_priority, int channels, int lm,
                           RangeEncoder& enc, int prev,
                           int signal_bandwidth) {
    return compute_allocation_impl(start, end, offsets, cap, alloc_trim,
                                   intensity, dual_stereo, total, balance,
                                   pulses, ebits, fine_priority, channels,
                                   lm, &enc, nullptr, prev,
                                   signal_bandwidth);
}

}  // namespace opus
}  // namespace glint
