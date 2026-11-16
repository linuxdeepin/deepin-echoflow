// Opus/CELT Laplace symbol coder — RFC 6716 section 4.3.2.1
// MIT License - Clean-room implementation

#include "opus_laplace.hpp"

namespace glint {
namespace opus {

namespace {
constexpr unsigned kTotal = 32768;  // 15-bit frequency total
constexpr unsigned kMinP = 1;       // floor probability per tail value
constexpr int kNMin = 16;           // guaranteed-representable deltas/side

// Frequency of |value|==1 given fs0: the probability mass left after the
// zero symbol and the 2*kNMin guaranteed tail values, scaled by the decay.
inline unsigned freq1(unsigned fs0, int decay) {
    unsigned rest = kTotal - kMinP * (2 * kNMin) - fs0;
    return (rest * static_cast<uint32_t>(16384 - decay)) >> 15;
}
}  // namespace

int laplace_encode(RangeEncoder& enc, int value, unsigned fs0, int decay) {
    unsigned fl = 0;
    unsigned fs = fs0;
    if (value) {
        int s = -(value < 0);
        int mag = (value + s) ^ s;
        fl = fs0;
        fs = freq1(fs0, decay);
        // Walk down the geometrically decaying part of the model. Each
        // magnitude covers 2*fs (both signs) plus the two floor slots.
        int i = 1;
        for (; fs > 0 && i < mag; i++) {
            fs *= 2;
            fl += fs + 2 * kMinP;
            fs = (fs * static_cast<uint32_t>(decay)) >> 15;
        }
        if (!fs) {
            // Flat tail: kMinP per value, signs interleaved. Clamp to what
            // is representable in the remaining freq space.
            int max_extra = static_cast<int>((kTotal - fl + kMinP - 1) / kMinP);
            max_extra = (max_extra - s) >> 1;
            int extra = mag - i < max_extra - 1 ? mag - i : max_extra - 1;
            fl += (2 * extra + 1 + s) * kMinP;
            fs = kMinP < kTotal - fl ? kMinP : kTotal - fl;
            value = (i + extra + s) ^ s;
        } else {
            fs += kMinP;
            fl += fs & ~static_cast<unsigned>(s);
        }
    }
    enc.encode_bin(fl, fl + fs, 15);
    return value;
}

int laplace_decode(RangeDecoder& dec, unsigned fs0, int decay) {
    int value = 0;
    unsigned fl = 0;
    unsigned fs = fs0;
    unsigned fm = dec.decode_bin(15);
    if (fm >= fs0) {
        value++;
        fl = fs0;
        fs = freq1(fs0, decay) + kMinP;
        // Walk the decaying region until fm falls inside [fl, fl+2*fs).
        while (fs > kMinP && fm >= fl + 2 * fs) {
            fs *= 2;
            fl += fs;
            fs = ((fs - 2 * kMinP) * static_cast<uint32_t>(decay)) >> 15;
            fs += kMinP;
            value++;
        }
        // Flat tail: constant-width slots.
        if (fs <= kMinP) {
            int extra = static_cast<int>((fm - fl) >> 1);
            value += extra;
            fl += 2 * static_cast<unsigned>(extra) * kMinP;
        }
        if (fm < fl + fs)
            value = -value;
        else
            fl += fs;
    }
    unsigned fh = fl + fs < kTotal ? fl + fs : kTotal;
    dec.dec_update(fl, fh, kTotal);
    return value;
}

}  // namespace opus
}  // namespace glint
