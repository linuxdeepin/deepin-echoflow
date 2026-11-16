// CELT band (PVQ shape) ENCODING — RFC 6716 section 4.3.4
// MIT License - Clean-room implementation
//
// Encoder-side twin of opus_celt_bands (which stays decode-only): the same
// integer skeleton — theta resolution/PDFs, bit splits via the bit-exact
// trig, rebalancing, TF Haar/Hadamard reorganization, collapse masks — but
// WRITING symbols, choosing theta via stereo_itheta, and quantizing shapes
// with alg_quant. All spectral arrays are float32: encoder decisions feed
// the wire, so they must round exactly like the reference float build
// (see the fma-pinning notes in opus_celt_enc_vq.cpp).
//
// No resynthesis paths at all (glint encodes without theta-RDO, mirroring
// the reference with resynth == 0): no folding buffers, no noise fills,
// no stereo merge — those exist only on the decode side.

#pragma once

#include <cstdint>

#include "opus_ec.hpp"

namespace glint {
namespace opus {

// Encode all bands' shapes from the normalized spectra X_ (and Y_ for
// stereo). band_e = per-channel linear band energies (float[2*21], the
// intensity_stereo weights). Mirrors the reference quant_all_bands with
// encode=1, resynth=0, theta_rdo=0.
void quant_all_bands_enc(int start, int end, float* X_, float* Y_,
                         uint8_t* collapse_masks, const float* band_e,
                         const int* pulses, int short_blocks, int spread,
                         int dual_stereo, int intensity, const int* tf_res,
                         int32_t total_bits, int32_t balance,
                         RangeEncoder& enc, int lm, int coded_bands,
                         uint32_t* seed);

}  // namespace opus
}  // namespace glint
