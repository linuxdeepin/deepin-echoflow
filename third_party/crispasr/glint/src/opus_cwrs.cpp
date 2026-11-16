// Opus/CELT PVQ codeword enumeration — RFC 6716 section 4.3.4.4
// MIT License - Clean-room implementation

#include "opus_cwrs.hpp"

namespace glint {
namespace opus {

namespace {

inline int iabs(int v) { return v < 0 ? -v : v; }

// In-place row step of the shared recurrence
//   f(n,k) = f(n-1,k) + f(n,k-1) + f(n-1,k-1):
// u[] holds one row (fixed n, ascending k); base is the new row's entry in
// the column just below u[0]. Unsigned wraparound in intermediates is fine —
// every value actually consumed fits 32 bits by the allocation caps.
void row_next(uint32_t* u, int len, uint32_t base) {
    for (int j = 1; j < len; j++) {
        uint32_t up = u[j] + u[j - 1] + base;
        u[j - 1] = base;
        base = up;
    }
    u[len - 1] = base;
}

// Inverse step: f(n-1,k) = f(n,k) - f(n,k-1) - f(n-1,k-1).
void row_prev(uint32_t* u, int len, uint32_t base) {
    for (int j = 1; j < len; j++) {
        uint32_t down = u[j] - u[j - 1] - base;
        u[j - 1] = base;
        base = down;
    }
    u[len - 1] = base;
}

// Fill u[0..k+1] with U(n, 0..k+1) and return V(n,k). Row n=2 is the
// closed form U(2,k) = 2k-1; higher rows step up via the recurrence
// (the k=1 column is constant 1, so it serves as the step base).
uint32_t fill_row(int n, int k, uint32_t* u) {
    u[0] = 0;
    u[1] = 1;
    for (int j = 2; j <= k + 1; j++) u[j] = 2u * static_cast<uint32_t>(j) - 1;
    for (int row = 2; row < n; row++) row_next(u + 1, k + 1, 1);
    return u[k] + u[k + 1];
}

}  // namespace

void encode_pulses(const int* y, int n, int k, RangeEncoder& enc) {
    uint32_t u[kCwrsMaxK + 2];
    // Enumerate from the LAST dimension up: start on row n=2 and step the
    // row up once per additional leading dimension. At each dimension the
    // index accumulates U(dims,kk) (codewords with fewer trailing pulses)
    // plus U(dims,kk+1) when this coefficient is negative (negative half
    // of the codebook comes second).
    u[0] = 0;
    for (int j = 1; j <= k + 1; j++) u[j] = 2u * static_cast<uint32_t>(j) - 1;
    uint32_t index = y[n - 1] < 0 ? 1u : 0u;
    int kk = iabs(y[n - 1]);
    int j = n - 2;
    index += u[kk];
    kk += iabs(y[j]);
    if (y[j] < 0) index += u[kk + 1];
    while (j-- > 0) {
        row_next(u, k + 2, 0);
        index += u[kk];
        kk += iabs(y[j]);
        if (y[j] < 0) index += u[kk + 1];
    }
    enc.enc_uint(index, u[kk] + u[kk + 1]);
}

int32_t decode_pulses(int* y, int n, int k, RangeDecoder& dec) {
    uint32_t u[kCwrsMaxK + 2];
    uint32_t index = dec.dec_uint(fill_row(n, k, u));
    int32_t yy = 0;
    for (int j = 0; j < n; j++) {
        // Negative-first-coefficient codewords occupy [U(n,k+1), V(n,k)).
        uint32_t p = u[k + 1];
        int s = -(index >= p);
        if (s) index -= p;
        // The number of pulses in this dimension is k - k', where k' is the
        // largest value with U(n,k') <= index.
        int k0 = k;
        p = u[k];
        while (p > index) p = u[--k];
        index -= p;
        int val = (k0 - k + s) ^ s;
        y[j] = val;
        yy += val * val;
        row_prev(u, k + 2, 0);
    }
    return yy;
}

}  // namespace opus
}  // namespace glint
