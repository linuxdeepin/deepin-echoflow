// glint - AAC storage types (shared by the AAC module headers)
// MIT License - Clean-room implementation

#ifndef GLINT_AAC_CODER_TYPES_FWD_HPP
#define GLINT_AAC_CODER_TYPES_FWD_HPP

#include <cstdint>

namespace glint {
namespace aac {

// Storage/arithmetic profiles:
//
// GLINT_AAC_INT (no-FPU hot path; implies the small-buffers footprint):
//   spectra are int32 in Q4 (value = spec_true * 16), PCM blocks int16.
//   The MDCT, quantizer, M/S decision and TNS filtering run in integer
//   arithmetic; only per-frame scalars (rate control, TNS Levinson) and the
//   -q normal/best psy tiers use floating point (soft-float on FPU-less
//   parts — fine at per-frame rates, fatal at per-coefficient rates).
//
// GLINT_SMALL_BUFFERS (RAM diet, FPU assumed): float spectra, int16 PCM,
//   double arithmetic everywhere.
//
// Desktop default: double storage and arithmetic.
#if defined(GLINT_AAC_INT)
using SpecT = int32_t;
using PcmT = int16_t;
constexpr int kSpecFracBits = 3;  // spec int = spec_true * 2^kSpecFracBits
#elif defined(GLINT_SMALL_BUFFERS)
using SpecT = float;
using PcmT = int16_t;
constexpr int kSpecFracBits = 0;
#else
using SpecT = double;
using PcmT = double;
constexpr int kSpecFracBits = 0;
#endif

// p34 cache element for the rate loop. Linear |x|^0.75 for float paths;
// log2(|x|^0.75) in Q16 for the integer path (quantization becomes one
// add in log domain + one exp2 LUT lookup per coefficient).
#if defined(GLINT_AAC_INT)
using P34T = int32_t;
#else
using P34T = SpecT;
#endif

}  // namespace aac
}  // namespace glint

#endif  // GLINT_AAC_CODER_TYPES_FWD_HPP
