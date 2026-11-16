// CELT pitch estimation + comb filter — shared by the decoder's PLC and
// the encoder's prefilter (RFC 6716 sections 4.3.7.1 / 5.1)
// MIT License - Clean-room implementation
//
// Bodies are verbatim copies of the decoder's PLC-gate-verified ports of
// celt/pitch.c + celt/celt_lpc.c float paths (the decoder keeps its own
// file-local copies; keep them in sync), plus remove_doubling, which only
// the encoder needs (sub-multiple pitch disambiguation).

#pragma once

namespace glint {
namespace opus {
namespace pitch {

void celt_autocorr(const double* x, double* ac, const double* window,
                   int overlap, int lag, int n);
void celt_lpc(double* lpc, const double* ac, int p);
void pitch_downsample(const double* const* x, double* x_lp, int len, int C);
void pitch_search(const double* x_lp, const double* y, int len,
                  int max_pitch, int* pitch);
// Checks T0's sub-multiples for the true period; returns the pitch gain
// and refines *t0 (reference remove_doubling).
double remove_doubling(double* x, int maxperiod, int minperiod, int n,
                       int* t0, int prev_period, double prev_gain);

}  // namespace pitch

// The 5-tap comb filter (decoder: postfilter; encoder: prefilter with
// negated gains), cross-faded over `overlap` with the squared window.
void comb_filter_shared(double* y, double* x, int t0, int t1, int n,
                        double g0, double g1, int tapset0, int tapset1,
                        const double* window, int overlap);

}  // namespace opus
}  // namespace glint
