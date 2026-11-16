// glint - AAC psychoacoustic band masks
// MIT License - Clean-room implementation
//
// Metric-aligned masking model, the same construction that worked for the
// MP3 nmr_outer_loop (and mirrors tests/measure_audio.py's NMR metric):
// per-sfb source energies spread with the Schroeder spreading function
// across Bark distance, -14 dB offset, floored at an ATH curve calibrated
// ~96 dB below the frame's loudest band. Do not "improve" this model
// independently of the metric — the MP3 lesson is that shaping against
// masks the metric does not see just burns bits.

#ifndef GLINT_AAC_PSY_HPP
#define GLINT_AAC_PSY_HPP

#include "aac_coder_types_fwd.hpp"

namespace glint {
namespace aac {

// mask[b] for the coded long-window bands [0, max_sfb) of one channel.
// spec: 1024 MDCT lines (source, pre-quantization). emax_ref calibrates the
// ATH floor (~96 dB below it); pass a RUNNING maximum of band energies, not
// the current frame's — per-frame calibration gives quiet frames absurdly
// low floors and the shaping loop burns its budget on phantom violations
// (the metric calibrates against the file-global maximum). Returns the
// frame's own max band energy so the caller can update the running max.
//
// tonal: per-masker offsets -(6 + 12*alpha_j) dB from band spectral flatness
// (tonal maskers mask less) instead of the metric's flat -14 dB — the same
// construction the MP3 nmr_outer_loop uses at low rates.
double aac_compute_masks(const SpecT* spec, int sr_index, int max_sfb,
                         double emax_ref, double* mask, bool tonal);

struct AacBandLayout;

// Short-frame masks: per (group, sfb) band of a kSeqShort layout, spreading
// only WITHIN each group (the model has no temporal masking across groups —
// the MP3 short-mask lesson: don't let it trade across windows). Flat -14 dB
// offset (no tonality: short windows are too coarse to estimate flatness).
// emax_ref/return: running max in the SHORT band-energy domain — keep it
// separate from the long-frame running max (different window scaling).
double aac_compute_masks_short(const SpecT* spec, const AacBandLayout& layout,
                               int sr_index, double emax_ref, double* mask);

}  // namespace aac
}  // namespace glint

#endif  // GLINT_AAC_PSY_HPP
