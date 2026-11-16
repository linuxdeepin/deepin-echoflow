// Opus range coder — RFC 6716 section 4.1
// MIT License - Clean-room implementation

#include "opus_ec.hpp"

#include "intmath.hpp"

#include <cstring>

namespace glint {
namespace opus {

namespace ec {

int ilog(uint32_t v) {
    return v ? 32 - intmath::clz32(v) : 0;
}

// tell() in 1/8-bit units. Refines ilog(rng) by kBitRes bits via repeated
// squaring of the top 16 bits of rng (Q15): each squaring doubles the log,
// so the top bit of the square is the next fractional bit of log2(rng).
uint32_t tell_frac_impl(uint32_t nbits_total, uint32_t rng) {
    uint32_t nbits = nbits_total << kBitRes;
    int l = ilog(rng);
    uint32_t r = rng >> (l - 16);
    for (unsigned i = kBitRes; i-- > 0;) {
        r = (r * r) >> 15;
        int b = static_cast<int>(r >> 16);
        l = (l << 1) | b;
        r >>= b;
    }
    return nbits - static_cast<uint32_t>(l);
}

}  // namespace ec

using namespace ec;

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

void RangeEncoder::init(uint8_t* buf, uint32_t size) {
    buf_ = buf;
    storage_ = size;
    offs_ = 0;
    end_offs_ = 0;
    end_window_ = 0;
    nend_bits_ = 0;
    // Matches the decoder's post-init state so both sides report tell()==1.
    nbits_total_ = kCodeBits + 1;
    val_ = 0;
    rng_ = kCodeTop;
    ext_ = 0;
    rem_ = -1;
    error_ = 0;
}

int RangeEncoder::write_byte(unsigned value) {
    if (offs_ + end_offs_ >= storage_) return -1;
    buf_[offs_++] = static_cast<uint8_t>(value);
    return 0;
}

int RangeEncoder::write_byte_at_end(unsigned value) {
    if (offs_ + end_offs_ >= storage_) return -1;
    buf_[storage_ - ++end_offs_] = static_cast<uint8_t>(value);
    return 0;
}

// Output one byte-sized chunk c (9 bits: possible carry in bit 8). A chunk of
// exactly 0xFF cannot be emitted yet — a later carry could ripple through it —
// so it is counted in ext_ and materialized (as 0xFF or 0x00 + carry into
// rem_) once a non-0xFF chunk arrives.
void RangeEncoder::carry_out(int c) {
    if (c != static_cast<int>(kSymMax)) {
        int carry = c >> kSymBits;
        if (rem_ >= 0) error_ |= write_byte(static_cast<unsigned>(rem_ + carry));
        if (ext_ > 0) {
            unsigned sym = (kSymMax + static_cast<unsigned>(carry)) & kSymMax;
            do {
                error_ |= write_byte(sym);
            } while (--ext_ > 0);
        }
        rem_ = c & static_cast<int>(kSymMax);
    } else {
        ext_++;
    }
}

void RangeEncoder::normalize() {
    while (rng_ <= kCodeBot) {
        carry_out(static_cast<int>(val_ >> kCodeShift));
        val_ = (val_ << kSymBits) & (kCodeTop - 1);
        rng_ <<= kSymBits;
        nbits_total_ += kSymBits;
    }
}

void RangeEncoder::encode(uint32_t fl, uint32_t fh, uint32_t ft) {
    uint32_t r = rng_ / ft;
    if (fl > 0) {
        val_ += rng_ - r * (ft - fl);
        rng_ = r * (fh - fl);
    } else {
        rng_ -= r * (ft - fh);
    }
    normalize();
}

void RangeEncoder::encode_bin(uint32_t fl, uint32_t fh, unsigned bits) {
    uint32_t r = rng_ >> bits;
    if (fl > 0) {
        val_ += rng_ - r * ((1u << bits) - fl);
        rng_ = r * (fh - fl);
    } else {
        rng_ -= r * ((1u << bits) - fh);
    }
    normalize();
}

void RangeEncoder::enc_bit_logp(int bit, unsigned logp) {
    uint32_t r = rng_;
    uint32_t l = val_;
    uint32_t s = r >> logp;  // bit==1 owns the top 2^-logp of the range
    r -= s;
    if (bit) val_ = l + r;
    rng_ = bit ? s : r;
    normalize();
}

void RangeEncoder::enc_icdf(int s, const uint8_t* icdf, unsigned ftb) {
    uint32_t r = rng_ >> ftb;
    if (s > 0) {
        val_ += rng_ - r * icdf[s - 1];
        rng_ = r * static_cast<uint32_t>(icdf[s - 1] - icdf[s]);
    } else {
        rng_ -= r * icdf[s];
    }
    normalize();
}

void RangeEncoder::enc_uint(uint32_t fl, uint32_t ft) {
    ft--;  // encode fl in [0, ft] from here on
    int ftb = ilog(ft);
    if (ftb > static_cast<int>(kUintBits)) {
        ftb -= kUintBits;
        uint32_t ft_hi = (ft >> ftb) + 1;
        uint32_t fl_hi = fl >> ftb;
        encode(fl_hi, fl_hi + 1, ft_hi);
        enc_bits(fl & ((1u << ftb) - 1), static_cast<unsigned>(ftb));
    } else {
        encode(fl, fl + 1, ft + 1);
    }
}

void RangeEncoder::enc_bits(uint32_t fl, unsigned bits) {
    uint32_t window = end_window_;
    int used = nend_bits_;
    if (used + static_cast<int>(bits) > static_cast<int>(kWindowSize)) {
        do {
            error_ |= write_byte_at_end(window & kSymMax);
            window >>= kSymBits;
            used -= static_cast<int>(kSymBits);
        } while (used >= static_cast<int>(kSymBits));
    }
    window |= fl << used;
    used += static_cast<int>(bits);
    end_window_ = window;
    nend_bits_ = used;
    nbits_total_ += bits;
}

void RangeEncoder::shrink(uint32_t size) {
    // Raw bits live at the buffer end; carry them to the new end.
    std::memmove(buf_ + size - end_offs_, buf_ + storage_ - end_offs_,
                 end_offs_);
    storage_ = size;
}

void RangeEncoder::patch_initial_bits(unsigned value, unsigned nbits) {
    unsigned shift = kSymBits - nbits;
    unsigned mask = ((1u << nbits) - 1) << shift;
    if (offs_ > 0) {
        // First byte already finalized.
        buf_[0] = static_cast<uint8_t>((buf_[0] & ~mask) | (value << shift));
    } else if (rem_ >= 0) {
        // First byte still awaiting carry propagation.
        rem_ = static_cast<int>((static_cast<unsigned>(rem_) & ~mask) |
                                (value << shift));
    } else if (rng_ <= (kCodeTop >> nbits)) {
        // Renormalization has not run yet; patch inside val.
        val_ = (val_ & ~(static_cast<uint32_t>(mask) << kCodeShift)) |
               (static_cast<uint32_t>(value) << (kCodeShift + shift));
    } else {
        // Fewer than nbits encoded so far.
        error_ = -1;
    }
}

void RangeEncoder::done() {
    // Find the shortest bit string that, zero-extended, still lands inside
    // [val, val + rng): round val up at each precision until the rounded
    // value plus its trailing-ones mask stays below the interval's top.
    int l = static_cast<int>(kCodeBits) - ilog(rng_);
    uint32_t msk = (kCodeTop - 1) >> l;
    uint32_t end = (val_ + msk) & ~msk;
    if ((end | msk) >= val_ + rng_) {
        // One more bit of precision is needed.
        l++;
        msk >>= 1;
        end = (val_ + msk) & ~msk;
    }
    while (l > 0) {
        carry_out(static_cast<int>(end >> kCodeShift));
        end = (end << kSymBits) & (kCodeTop - 1);
        l -= static_cast<int>(kSymBits);
    }
    // Flush any buffered carry chain.
    if (rem_ >= 0 || ext_ > 0) carry_out(0);
    // Flush whole bytes of pending raw bits to the back.
    uint32_t window = end_window_;
    int used = nend_bits_;
    while (used >= static_cast<int>(kSymBits)) {
        error_ |= write_byte_at_end(window & kSymMax);
        window >>= kSymBits;
        used -= static_cast<int>(kSymBits);
    }
    if (!error_) {
        // The frame is fixed-size: zero everything between the two streams.
        std::memset(buf_ + offs_, 0, storage_ - end_offs_ - offs_);
        if (used > 0) {
            // Leftover raw bits share the last byte (possibly the same byte
            // the range stream ended in). -l is how many low bits of that
            // byte the range stream left free.
            if (end_offs_ >= storage_) {
                error_ = -1;
            } else {
                l = -l;
                if (offs_ + end_offs_ >= storage_ && l < used) {
                    // Sharing byte with the range stream and not enough room.
                    window &= (1u << l) - 1;
                    error_ = -1;
                }
                buf_[storage_ - end_offs_ - 1] |=
                    static_cast<uint8_t>(window);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

void RangeDecoder::init(const uint8_t* buf, uint32_t size) {
    buf_ = buf;
    storage_ = size;
    end_offs_ = 0;
    end_window_ = 0;
    nend_bits_ = 0;
    // kCodeExtra bits of the first byte enter val below; normalize() adds 8
    // per byte after that. This constant makes tell()==1 after init, in step
    // with the encoder.
    nbits_total_ = kCodeBits + 1 -
                   ((kCodeBits - kCodeExtra) / kSymBits) * kSymBits;
    offs_ = 0;
    rng_ = 1u << kCodeExtra;
    rem_ = read_byte();
    val_ = rng_ - 1 -
           static_cast<uint32_t>(rem_ >> (kSymBits - kCodeExtra));
    error_ = 0;
    normalize();
}

int RangeDecoder::read_byte() {
    return offs_ < storage_ ? buf_[offs_++] : 0;
}

int RangeDecoder::read_byte_from_end() {
    return end_offs_ < storage_ ? buf_[storage_ - ++end_offs_] : 0;
}

void RangeDecoder::normalize() {
    while (rng_ <= kCodeBot) {
        nbits_total_ += kSymBits;
        rng_ <<= kSymBits;
        int sym = rem_;
        rem_ = read_byte();
        // Straddle the byte boundary: the coder consumes kCodeExtra bits of
        // one byte and tops up from the next. val is complemented (~sym).
        sym = ((sym << kSymBits) | rem_) >> (kSymBits - kCodeExtra);
        val_ = ((val_ << kSymBits) +
                (kSymMax & ~static_cast<uint32_t>(sym))) &
               (kCodeTop - 1);
    }
}

unsigned RangeDecoder::decode(uint32_t ft) {
    ext_ = rng_ / ft;
    uint32_t s = val_ / ext_;
    uint32_t cap = s + 1 < ft ? s + 1 : ft;
    return ft - cap;
}

unsigned RangeDecoder::decode_bin(unsigned bits) {
    ext_ = rng_ >> bits;
    uint32_t ft = 1u << bits;
    uint32_t s = val_ / ext_;
    uint32_t cap = s + 1 < ft ? s + 1 : ft;
    return ft - cap;
}

void RangeDecoder::dec_update(uint32_t fl, uint32_t fh, uint32_t ft) {
    uint32_t s = ext_ * (ft - fh);
    val_ -= s;
    rng_ = fl > 0 ? ext_ * (fh - fl) : rng_ - s;
    normalize();
}

int RangeDecoder::dec_bit_logp(unsigned logp) {
    uint32_t r = rng_;
    uint32_t d = val_;
    uint32_t s = r >> logp;
    int ret = d < s;  // val is complemented: top of range = small val
    if (!ret) val_ = d - s;
    rng_ = ret ? s : r - s;
    normalize();
    return ret;
}

int RangeDecoder::dec_icdf(const uint8_t* icdf, unsigned ftb) {
    uint32_t s = rng_;
    uint32_t d = val_;
    uint32_t r = s >> ftb;
    int ret = -1;
    uint32_t t;
    do {
        t = s;
        s = r * icdf[++ret];
    } while (d < s);
    val_ = d - s;
    rng_ = t - s;
    normalize();
    return ret;
}

uint32_t RangeDecoder::dec_uint(uint32_t ft) {
    ft--;
    int ftb = ilog(ft);
    if (ftb > static_cast<int>(kUintBits)) {
        ftb -= kUintBits;
        uint32_t ft_hi = (ft >> ftb) + 1;
        uint32_t s = decode(ft_hi);
        dec_update(s, s + 1, ft_hi);
        uint32_t t = (s << ftb) | dec_bits(static_cast<unsigned>(ftb));
        if (t <= ft) return t;
        error_ = 1;
        return ft;
    }
    ft++;
    uint32_t s = decode(ft);
    dec_update(s, s + 1, ft);
    return s;
}

uint32_t RangeDecoder::dec_bits(unsigned bits) {
    uint32_t window = end_window_;
    int available = nend_bits_;
    if (available < static_cast<int>(bits)) {
        do {
            window |= static_cast<uint32_t>(read_byte_from_end())
                      << available;
            available += static_cast<int>(kSymBits);
        } while (available <= static_cast<int>(kWindowSize - kSymBits));
    }
    uint32_t ret = window & ((1u << bits) - 1);
    window >>= bits;
    available -= static_cast<int>(bits);
    end_window_ = window;
    nend_bits_ = available;
    nbits_total_ += bits;
    return ret;
}

}  // namespace opus
}  // namespace glint
