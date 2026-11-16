// CELT band (PVQ shape) ENCODING — vector-quantization layer, RFC 6716
// section 4.3.4 (reference celt/vq.c + the stereo helpers of celt/bands.c)
// MIT License - Clean-room implementation
//
// Encoder twin of the VQ machinery in opus_celt_bands.cpp (which is
// decode-only and keeps its helpers file-local). The decoder could relax the
// reference's float arithmetic to double because its floats only shape PCM
// output; the ENCODER cannot: op_pvq_search's greedy argmax decisions pick
// the CODED pulse vector and stereo_itheta's atan2 picks the CODED theta
// index, so float32 semantics are wire-critical. Everything here is float,
// and every multiply-add is pinned EXPLICITLY (std::fmaf where the reference
// build fuses, separate mul/add statements where it does not) so the values
// do not depend on the compiler's contraction or vectorization choices.
// The fused/unfused split was read out of the reference float build's
// disassembly (libopus 1.5.2, Apple clang -O2, arm64):
//  - auto-vectorized loop bodies round each product separately and keep the
//    reduction adds in element order, covering indices [0, n & ~3);
//  - scalar loop tails contract mul+add chains to fma;
//  - celt_inner_prod is the NEON intrinsics version: four fused lanes over
//    blocks of four, (l0+l2)+(l1+l3) reduction, fma scalar tail.
// tools/crosscheck_opus_enc_vq.py verifies all of it against libopus
// byte-for-byte (encoded streams + tells + collapse masks + itheta).
//
// Preconditions everywhere: 2 <= n <= kVqMaxBandSize, 1 <= k <= kCwrsMaxK,
// V(n,k) < 2^32 (guaranteed by CELT's allocation caps).

#pragma once

#include "opus_ec.hpp"

namespace glint {
namespace opus {

// Largest band size any standard CELT mode produces (22 << 3).
constexpr int kVqMaxBandSize = 176;

// Spreading rotation (vq.c exp_rotation), float32. dir=+1 before the pulse
// search (encode), dir=-1 on the resynthesis/decode side. No-op when
// 2*k >= len or spread == 0.
void exp_rotation(float* x, int len, int dir, int stride, int k, int spread);

// Greedy PVQ search (vq.c op_pvq_search_c): projection pre-search when
// k > n/2, then one pulse at a time maximizing Rxy^2/Ryy. x is clobbered
// (rotated input -> absolute values); iy receives the signed pulse vector
// with sum(|iy|) == k. Returns yy = sum(iy^2) (exact integer in float).
float op_pvq_search(float* x, int* iy, int k, int n);

// Encode one band's shape: exp_rotation, op_pvq_search, encode_pulses.
// With resynth, x is overwritten with the decoder-identical reconstruction
// (unit-normalized residual times gain, inverse-rotated); without, x is
// left clobbered. Returns the collapse mask (bit per interleaved block).
unsigned alg_quant(float* x, int n, int k, int spread, int b,
                   RangeEncoder& enc, float gain, bool resynth);

// Scale x to unit norm times gain (vq.c renormalise_vector; the copy in
// opus_celt_bands.cpp is file-local and double — this is the encoder's
// float version, using the reference build's NEON inner-product order).
void renormalise_vector(float* x, int n, float gain);

// In-place mid/side transform with 1/sqrt(2) scaling (bands.c stereo_split;
// the encode-side dual of stereo_merge).
void stereo_split(float* x, float* y, int n);

// Replace x with the intensity-stereo downmix a1*x + a2*y, where a1/a2 are
// the normalized band amplitudes: left = band_e[band], right =
// band_e[band + nb_bands] (bandE layout is [channel][band]).
void intensity_stereo(float* x, const float* y, const float* band_e,
                      int band, int nb_bands, int n);

// Stereo angle index in Q14 (0..16384): itheta =
// floor(.5 + 16384*(2/pi)*atan2(side, mid)) from float channel energies.
// stereo != 0 measures mid/side of (x,y); stereo == 0 measures x vs y as
// already-split channels (vq.c stereo_itheta).
int stereo_itheta(const float* x, const float* y, int stereo, int n);

}  // namespace opus
}  // namespace glint
