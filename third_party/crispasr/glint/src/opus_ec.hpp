// Opus range coder (entropy coder) — RFC 6716 section 4.1
// MIT License - Clean-room implementation
//
// The single most load-bearing component of the Opus track (PLAN.md § O0):
// every SILK and CELT symbol flows through this coder, and CELT's implicit
// bit allocation requires the encoder and decoder to agree on tell()/
// tell_frac() to 1/8-bit precision at every step.
//
// Conventions (from the RFC):
//  - Range-coded symbols consume the buffer from the FRONT; "raw" bits
//    (enc_bits/dec_bits) consume it from the BACK. The two streams may share
//    the final byte.
//  - The decoder tracks `val` as the distance from the TOP of the current
//    range (a complemented value), which is why initialization reads
//    127 - (b0 >> 1) and normalization inserts complemented bytes.
//  - rng is kept normalized in (2^23, 2^31]; each renormalization shifts in
//    one byte and adds 8 to nbits_total.
//  - tell() = nbits_total - ilog(rng) counts TOTAL bits used (range + raw),
//    and both sides start at tell() == 1 after init.
//
// Everything here is plain integer arithmetic — safe for GLINT_MODE=fixed.

#pragma once

#include <cstddef>
#include <cstdint>

namespace glint {
namespace opus {

namespace ec {
constexpr unsigned kSymBits = 8;                       // bits per stream byte
constexpr unsigned kCodeBits = 32;                     // coder precision
constexpr uint32_t kSymMax = (1u << kSymBits) - 1;     // 0xFF
constexpr uint32_t kCodeTop = 1u << (kCodeBits - 1);   // 2^31
constexpr uint32_t kCodeBot = kCodeTop >> kSymBits;    // 2^23
constexpr unsigned kCodeShift = kCodeBits - kSymBits - 1;          // 23
constexpr unsigned kCodeExtra = (kCodeBits - 2) % kSymBits + 1;    // 7
constexpr unsigned kWindowSize = 32;   // raw-bit accumulator width
constexpr unsigned kUintBits = 8;      // range-coded MSBs in enc_uint splits
constexpr unsigned kBitRes = 3;        // tell_frac resolution: 1/8 bit

// Number of bits in v (position of highest set bit + 1); ilog(0) == 0.
int ilog(uint32_t v);

// Shared tell()/tell_frac() computation.
uint32_t tell_frac_impl(uint32_t nbits_total, uint32_t rng);
}  // namespace ec

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------
class RangeEncoder {
public:
    // The buffer is the whole Opus frame; its size is fixed up front (the
    // container carries the length, so unused middle bytes are zero-padded
    // by done()).
    void init(uint8_t* buf, uint32_t size);

    // Encode a symbol with cumulative frequency interval [fl, fh) out of ft.
    // Requires 0 <= fl < fh <= ft, ft <= 2^16 in practice (rng/ft must stay
    // large enough; Opus never exceeds 16-bit ft).
    void encode(uint32_t fl, uint32_t fh, uint32_t ft);
    // Same with ft == 1<<bits (saves the division).
    void encode_bin(uint32_t fl, uint32_t fh, unsigned bits);
    // One binary symbol where P(bit == 1) == 2^-logp.
    void enc_bit_logp(int bit, unsigned logp);
    // Symbol from an "inverse CDF" table: icdf[s] = ft - fh(s) with implicit
    // ft = 1<<ftb; the table is decreasing and ends in 0.
    void enc_icdf(int s, const uint8_t* icdf, unsigned ftb);
    // Uniformly distributed integer fl in [0, ft), ft >= 2. Values needing
    // more than kUintBits bits are split: range-coded MSBs + raw LSBs.
    void enc_uint(uint32_t fl, uint32_t ft);
    // Raw bits, written from the END of the buffer. 0 < bits < 25.
    void enc_bits(uint32_t fl, unsigned bits);

    // Finalize: flush enough range bits that any decoder resolves to the
    // encoded sequence, zero the unused middle, merge the raw-bit tail
    // (possibly sharing the last byte with the range stream).
    void done();

    // Total bits used so far (range + raw), rounded up to whole bits.
    uint32_t tell() const { return nbits_total_ - ec::ilog(rng_); }
    // Same in 1/8-bit units (what CELT's allocator uses).
    uint32_t tell_frac() const { return ec::tell_frac_impl(nbits_total_, rng_); }

    // Nonzero if the buffer overflowed at any point.
    int error() const { return error_; }
    // Bytes written at the front (valid after done(); back bytes fill the
    // remainder of the buffer).
    uint32_t range_bytes() const { return offs_; }
    // Range width — the Opus conformance "final range" value (must equal
    // the decoder's after the same symbols).
    uint32_t range() const { return rng_; }
    // The buffer passed to init(). CELT's two-pass coarse-energy encoder
    // snapshots the encoder state, and saves/restores the front bytes the
    // discarded pass wrote (the reference's ec_get_buffer usage).
    uint8_t* buffer() const { return buf_; }
    // Shrink the buffer to `size` bytes (CBR/VBR final sizing): moves any
    // raw-bit tail bytes up against the new end. Requires
    // offs + end_offs <= size.
    void shrink(uint32_t size);
    // Rewrite the first `nbits` (<= 8) bits already emitted (Opus patches
    // mode/flag bits after rate decisions). Sets error() if more than a
    // renormalization's worth of data was already produced.
    void patch_initial_bits(unsigned value, unsigned nbits);

private:
    int write_byte(unsigned value);
    int write_byte_at_end(unsigned value);
    void carry_out(int c);
    void normalize();

    uint8_t* buf_ = nullptr;
    uint32_t storage_ = 0;
    uint32_t offs_ = 0;        // front bytes emitted
    uint32_t end_offs_ = 0;    // back bytes emitted (raw bits)
    uint32_t end_window_ = 0;  // pending raw bits
    int nend_bits_ = 0;
    uint32_t nbits_total_ = 0;
    uint32_t val_ = 0;   // low end of the current range (un-output bits)
    uint32_t rng_ = 0;   // size of the current range
    uint32_t ext_ = 0;   // count of buffered 0xFF bytes awaiting carry
    int rem_ = -1;       // buffered byte awaiting possible carry; -1 = none
    int error_ = 0;
};

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------
class RangeDecoder {
public:
    void init(const uint8_t* buf, uint32_t size);

    // Return fs in [fl, fh) of the next symbol given total ft; the caller
    // looks up which symbol owns fs, then MUST call dec_update(fl, fh, ft).
    unsigned decode(uint32_t ft);
    unsigned decode_bin(unsigned bits);  // ft == 1<<bits
    void dec_update(uint32_t fl, uint32_t fh, uint32_t ft);

    int dec_bit_logp(unsigned logp);
    int dec_icdf(const uint8_t* icdf, unsigned ftb);
    uint32_t dec_uint(uint32_t ft);
    uint32_t dec_bits(unsigned bits);

    uint32_t tell() const { return nbits_total_ - ec::ilog(rng_); }
    uint32_t tell_frac() const { return ec::tell_frac_impl(nbits_total_, rng_); }

    // Set when a decoded value was out of range (corrupt stream); decoding
    // may continue (all outputs stay in valid ranges) per RFC robustness
    // requirements.
    int error() const { return error_; }
    // Frame size in bytes (CELT computes bit budgets as storage * 8).
    uint32_t storage_bytes() const { return storage_; }
    // Current range width; CELT reuses it as the next frame's noise seed.
    uint32_t range() const { return rng_; }
    // Pretend `bits` total bits have been consumed (CELT silence frames
    // mark the whole frame as read).
    void set_tell(uint32_t bits) { nbits_total_ += bits - tell(); }
    // Cut trailing bytes off the buffer (Opus transition redundancy is
    // stored at the end). Only valid before any raw bits were read.
    void shrink(uint32_t bytes) { storage_ -= bytes; }

private:
    int read_byte();
    int read_byte_from_end();
    void normalize();

    const uint8_t* buf_ = nullptr;
    uint32_t storage_ = 0;
    uint32_t offs_ = 0;
    uint32_t end_offs_ = 0;
    uint32_t end_window_ = 0;
    int nend_bits_ = 0;
    uint32_t nbits_total_ = 0;
    uint32_t val_ = 0;   // distance from the TOP of the range (complemented)
    uint32_t rng_ = 0;
    uint32_t ext_ = 0;   // saved rng/ft divisor between decode()/dec_update()
    int rem_ = -1;       // last byte read from the front
    int error_ = 0;
};

}  // namespace opus
}  // namespace glint
