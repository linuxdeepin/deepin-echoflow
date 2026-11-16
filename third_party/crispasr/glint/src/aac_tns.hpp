// glint - AAC temporal noise shaping (TNS), long windows
// MIT License - Clean-room implementation
//
// Open-loop LPC across frequency: the encoder FIR-filters the spectrum with
// A(z) and transmits the quantized REFLECTION coefficients; the decoder's
// all-pole 1/A(z) synthesis shapes quantization noise to follow the signal's
// temporal envelope inside the window (speech pitch pulses, transient tails).

#ifndef GLINT_AAC_TNS_HPP
#define GLINT_AAC_TNS_HPP

#include <cstdint>

#include "aac_coder_types_fwd.hpp"

namespace glint {
namespace aac {

constexpr int kTnsMaxOrder = 12;  // LC profile, long windows

struct AacTnsFilter {
    uint8_t active;               // 0 = no tns_data
    uint8_t length;               // sfbs covered, counted down from max_sfb
    uint8_t order;
    int8_t coef_idx[kTnsMaxOrder];  // 4-bit signed indices (arcsin rule)
};

struct AacBandLayout;

// Analyze the coded-domain spectrum and, when the LPC prediction gain
// clears the threshold, quantize the reflection coefficients and FIR-filter
// `spec` in place over the TNS region. Long-family layouts only.
void aac_tns_analyze(SpecT* spec, const AacBandLayout& layout, int sr_index,
                     AacTnsFilter* f);

}  // namespace aac
}  // namespace glint

#endif  // GLINT_AAC_TNS_HPP
