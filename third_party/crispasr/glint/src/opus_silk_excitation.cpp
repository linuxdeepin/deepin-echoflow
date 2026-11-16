// SILK excitation decoding — RFC 6716 section 4.2.7.8
// MIT License - Clean-room implementation

#include "opus_silk_excitation.hpp"

#include <cstring>

#include "opus_silk_math.hpp"
#include "opus_silk_tables.hpp"

namespace glint {
namespace opus {
namespace silk {

namespace {

constexpr int kMaxNbShellBlocks = 20;  // 320 / 16

// One split of the shell tree: child1 pulse count from the row for the
// parent count, child2 is the remainder.
inline void decode_split(int16_t* child1, int16_t* child2,
                         RangeDecoder& dec, int p,
                         const uint8_t* shell_table) {
    if (p > 0) {
        child1[0] = static_cast<int16_t>(
            dec.dec_icdf(&shell_table[kShellCodeTableOffsets[p]], 8));
        child2[0] = static_cast<int16_t>(p - child1[0]);
    } else {
        child1[0] = 0;
        child2[0] = 0;
    }
}

// Distribute `total` pulses over one 16-sample block by binary splits.
void shell_decoder(int16_t* pulses0, RangeDecoder& dec, int total) {
    int16_t pulses3[2], pulses2[4], pulses1[8];

    decode_split(&pulses3[0], &pulses3[1], dec, total, kShellCodeTable3);

    decode_split(&pulses2[0], &pulses2[1], dec, pulses3[0],
                 kShellCodeTable2);
    decode_split(&pulses1[0], &pulses1[1], dec, pulses2[0],
                 kShellCodeTable1);
    decode_split(&pulses0[0], &pulses0[1], dec, pulses1[0],
                 kShellCodeTable0);
    decode_split(&pulses0[2], &pulses0[3], dec, pulses1[1],
                 kShellCodeTable0);
    decode_split(&pulses1[2], &pulses1[3], dec, pulses2[1],
                 kShellCodeTable1);
    decode_split(&pulses0[4], &pulses0[5], dec, pulses1[2],
                 kShellCodeTable0);
    decode_split(&pulses0[6], &pulses0[7], dec, pulses1[3],
                 kShellCodeTable0);

    decode_split(&pulses2[2], &pulses2[3], dec, pulses3[1],
                 kShellCodeTable2);
    decode_split(&pulses1[4], &pulses1[5], dec, pulses2[2],
                 kShellCodeTable1);
    decode_split(&pulses0[8], &pulses0[9], dec, pulses1[4],
                 kShellCodeTable0);
    decode_split(&pulses0[10], &pulses0[11], dec, pulses1[5],
                 kShellCodeTable0);
    decode_split(&pulses1[6], &pulses1[7], dec, pulses2[3],
                 kShellCodeTable1);
    decode_split(&pulses0[12], &pulses0[13], dec, pulses1[6],
                 kShellCodeTable0);
    decode_split(&pulses0[14], &pulses0[15], dec, pulses1[7],
                 kShellCodeTable0);
}

// Signs for nonzero samples; PDF picked by 7*(2*type+offset) + min(p,6).
void decode_signs(RangeDecoder& dec, int16_t* pulses, int length,
                  int signal_type, int quant_offset_type,
                  const int* sum_pulses) {
    uint8_t icdf[2];
    icdf[1] = 0;
    const uint8_t* icdf_ptr =
        &kSignIcdf[7 * (quant_offset_type + (signal_type << 1))];
    int16_t* q = pulses;
    int nblocks = (length + kShellCodecFrameLength / 2) >> 4;
    for (int i = 0; i < nblocks; i++) {
        int p = sum_pulses[i];
        if (p > 0) {
            int idx = p & 0x1F;
            icdf[0] = icdf_ptr[idx < 6 ? idx : 6];
            for (int j = 0; j < kShellCodecFrameLength; j++) {
                if (q[j] > 0) {
                    // Map {0,1} -> {-1,+1}.
                    q[j] = static_cast<int16_t>(
                        q[j] * (2 * dec.dec_icdf(icdf, 8) - 1));
                }
            }
        }
        q += kShellCodecFrameLength;
    }
}

}  // namespace

void decode_pulses(RangeDecoder& dec, int16_t* pulses, int signal_type,
                   int quant_offset_type, int frame_length) {
    int sum_pulses[kMaxNbShellBlocks];
    int n_lshifts[kMaxNbShellBlocks];

    int rate_level =
        dec.dec_icdf(kRateLevelsIcdf[signal_type >> 1], 8);

    int iter = frame_length >> 4;
    if (iter * kShellCodecFrameLength < frame_length) iter++;  // 10ms@12k

    // Per-block total pulse counts; >16 chains LSB extension levels (at 10
    // levels the table row shifts by one to forbid further extension).
    const uint8_t* cdf = kPulsesPerBlockIcdf[rate_level];
    for (int i = 0; i < iter; i++) {
        n_lshifts[i] = 0;
        sum_pulses[i] = dec.dec_icdf(cdf, 8);
        while (sum_pulses[i] == kMaxPulses + 1) {
            n_lshifts[i]++;
            sum_pulses[i] = dec.dec_icdf(
                kPulsesPerBlockIcdf[kNRateLevels - 1] +
                    (n_lshifts[i] == 10 ? 1 : 0),
                8);
        }
    }

    for (int i = 0; i < iter; i++) {
        if (sum_pulses[i] > 0)
            shell_decoder(&pulses[i * kShellCodecFrameLength], dec,
                          sum_pulses[i]);
        else
            std::memset(&pulses[i * kShellCodecFrameLength], 0,
                        kShellCodecFrameLength * sizeof(int16_t));
    }

    // LSB refinement for extended blocks.
    for (int i = 0; i < iter; i++) {
        if (n_lshifts[i] > 0) {
            int nls = n_lshifts[i];
            int16_t* p = &pulses[i * kShellCodecFrameLength];
            for (int k = 0; k < kShellCodecFrameLength; k++) {
                int abs_q = p[k];
                for (int j = 0; j < nls; j++) {
                    abs_q <<= 1;
                    abs_q += dec.dec_icdf(kLsbIcdf, 8);
                }
                p[k] = static_cast<int16_t>(abs_q);
            }
            // Flag nonzero pulse count for the sign pass.
            sum_pulses[i] |= nls << 5;
        }
    }

    decode_signs(dec, pulses, frame_length, signal_type, quant_offset_type,
                 sum_pulses);
}

}  // namespace silk
}  // namespace opus
}  // namespace glint
