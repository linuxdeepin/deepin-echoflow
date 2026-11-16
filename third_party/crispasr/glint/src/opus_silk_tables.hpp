// glint - Opus SILK decoder tables (GENERATED FILE — do not edit)
// Regenerate with: python tools/gen_opus_silk_tables.py
//
// Extracted from libopus 1.5.2 (BSD): silk/tables_*.c,
// silk/pitch_est_tables.c, silk/table_LSF_cos.c, silk/resampler_rom.{c,h}.
// The generator cross-checks parsed element counts against the dimensions
// declared at the definition site AND the extern declarations in
// silk/tables.h, validates every iCDF row (uint8, non-increasing, final 0),
// the shell-code row tiling, the LTP codebook sizes/pointer order, and the
// NLSF codebook shape invariants before emitting.
//
// iCDF convention (RFC 6716 §4.1.3.3): ec_dec_icdf with ft = 1<<8; entry i
// is 256 minus the cumulative frequency up to and including symbol i, so
// rows are non-increasing and end at 0. All SILK decode calls use ftb = 8.
//
// Encoder-only tables deliberately NOT emitted (verified referenced only
// from encoder sources): silk_pulses_per_block_BITS_Q5,
// silk_rate_levels_BITS_Q5, silk_LTP_gain_BITS_Q5_{0,1,2} (+ptr table),
// silk_LTP_gain_vq_{0,1,2}_gain / silk_LTP_vq_gain_ptrs_Q7,
// silk_Lag_range_stage3{,_10_ms}, silk_nb_cbk_searchs_stage3.

#ifndef GLINT_OPUS_SILK_TABLES_HPP
#define GLINT_OPUS_SILK_TABLES_HPP

#include <cstdint>

namespace glint {
namespace opus {
namespace silk {

// Structural constants (silk/define.h, pitch_est_defines.h,
// resampler_rom.h); the generator cross-checks them against the
// table shapes below.
constexpr int kNLevelsQGain = 64;  // gain quantization levels
constexpr int kNRateLevels = 10;  // rate levels for pulse coding
constexpr int kMaxPulses = 16;  // max pulses per shell block
constexpr int kShellCodecFrameLength = 16;  // samples per shell block
constexpr int kNbLtpCbks = 3;  // LTP codebooks (periodicity index)
constexpr int kLtpOrder = 5;  // LTP filter taps
constexpr int kMaxLpcOrder = 16;  // LPC order (WB; NB/MB use 10)
constexpr int kNlsfQuantMaxAmplitude = 4;  // NLSF residual iCDF half-width
constexpr int kPitchEstMinLagMs = 2;  // min pitch lag
constexpr int kPitchEstMaxLagMs = 18;  // max pitch lag
constexpr int kPeMaxNbSubfr = 4;  // subframes per 20 ms frame
constexpr int kStereoQuantTabSize = 16;  // stereo predictor levels
constexpr int kTransitionIntNum = 5;  // transition LP interp points
constexpr int kTransitionNb = 3;  // transition LP FIR taps
constexpr int kTransitionNa = 2;  // transition LP IIR taps
constexpr int kLsfCosTabSize = 128;  // LSF cosine table size
constexpr int kResamplerDownOrderFir0 = 18;
constexpr int kResamplerDownOrderFir1 = 24;
constexpr int kResamplerDownOrderFir2 = 36;
constexpr int kResamplerOrderFir12 = 8;

// ---- gains (silk/tables_gain.c) ------------------------------------------

// Independent (MSB) gain index iCDFs, [signalType] (0 inactive,
// 1 unvoiced, 2 voiced); the 3 LSBs are coded with kUniform8Icdf.
inline constexpr uint8_t kGainIcdf[3][8] = {
    { 224, 112, 44, 15, 3, 2, 1, 0 },
    { 254, 237, 192, 132, 70, 23, 4, 0 },
    { 255, 252, 226, 155, 61, 11, 2, 0 },
};

// Delta gain index iCDF (MIN_DELTA_GAIN_QUANT -4 .. MAX 36).
inline constexpr uint8_t kDeltaGainIcdf[41] = {
    250, 245, 234, 203, 71, 50, 42, 38, 35, 33, 31, 29, 28, 27, 26,
    25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};

// ---- pitch lag / contour (silk/tables_pitch_lag.c, pitch_est_tables.c) ---

// High part of the absolute pitch lag, 32 symbols of fs_kHz/2 lags each.
inline constexpr uint8_t kPitchLagIcdf[32] = {
    253, 250, 244, 233, 212, 182, 150, 131, 120, 110, 98, 85, 72, 60, 49,
    40, 32, 25, 19, 15, 13, 11, 9, 8, 7, 6, 5, 4, 3, 2,
    1, 0,
};

// Relative-lag delta iCDF; symbol 0 escapes to absolute coding.
inline constexpr uint8_t kPitchDeltaIcdf[21] = {
    210, 208, 206, 203, 199, 193, 183, 168, 142, 104, 74, 52, 37, 27, 20,
    14, 10, 6, 4, 2, 0,
};

// Pitch contour index, 20 ms MB/WB (34 entries).
inline constexpr uint8_t kPitchContourIcdf[34] = {
    223, 201, 183, 167, 152, 138, 124, 111, 98, 88, 79, 70, 62, 56, 50,
    44, 39, 35, 31, 27, 24, 21, 18, 16, 14, 12, 10, 8, 6, 4,
    3, 2, 1, 0,
};

// Pitch contour index, 20 ms NB.
inline constexpr uint8_t kPitchContourNbIcdf[11] = {
    188, 176, 155, 138, 119, 97, 67, 43, 26, 10, 0,
};

// Pitch contour index, 10 ms MB/WB.
inline constexpr uint8_t kPitchContour10MsIcdf[12] = {
    165, 119, 80, 61, 47, 35, 27, 20, 14, 9, 4, 0,
};

// Pitch contour index, 10 ms NB.
inline constexpr uint8_t kPitchContour10MsNbIcdf[3] = {
    113, 63, 0,
};

// Per-subframe lag offsets for the 20 ms contour codebook, stage 2
// (decode_pitch.c uses the first PE_NB_CBKS_STAGE2 = 3 columns for NB).
inline constexpr int8_t kCbLagsStage2[4][11] = {
    { 0, 2, -1, -1, -1, 0, 0, 1, 1, 0, 1 },
    { 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
    { 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 },
    { 0, -1, 2, 1, 0, 1, 1, 0, 0, -1, -1 },
};

// Per-subframe lag offsets, 20 ms stage 3 (MB/WB, 34 contours).
inline constexpr int8_t kCbLagsStage3[4][34] = {
    {
        0, 0, 1, -1, 0, 1, -1, 0, -1, 1, -2, 2, -2, -2, 2,
        -3, 2, 3, -3, -4, 3, -4, 4, 4, -5, 5, -6, -5, 6, -7,
        6, 5, 8, -9,
    },
    {
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1,
        -1, 0, 1, -1, -1, 1, -1, 2, 1, -1, 2, -2, -2, 2, -2,
        2, 2, 3, -3,
    },
    {
        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, -1,
        1, 0, 0, 2, 1, -1, 2, -1, -1, 2, -1, 2, 2, -1, 3,
        -2, -2, -2, 3,
    },
    {
        0, 1, 0, 0, 1, 0, 1, -1, 2, -1, 2, -1, 2, 3, -2,
        3, -2, -2, 4, 4, -3, 5, -3, -4, 6, -4, 6, 5, -5, 8,
        -6, -5, -7, 9,
    },
};

// Lag offsets, 10 ms stage 2.
inline constexpr int8_t kCbLagsStage2_10ms[2][3] = {
    { 0, 1, 0 },
    { 0, 0, 1 },
};

// Lag offsets, 10 ms stage 3.
inline constexpr int8_t kCbLagsStage3_10ms[2][12] = {
    { 0, 0, 1, -1, 1, -1, 2, -2, 2, -2, 3, -3 },
    { 0, 1, 0, 1, -1, 2, -1, 2, -2, 3, -2, 3 },
};

// ---- excitation: rate levels, pulse counts, shell coder, LSBs, signs ------
// (silk/tables_pulses_per_block.c)

// Max pulses per shell block for rate levels (encoder-side cap; kept
// for the future O5 encoder — tiny).
inline constexpr uint8_t kMaxPulsesTable[4] = {
    8, 10, 12, 16,
};

// Pulse-count iCDF per rate level; rows 9 (with LSB escape at symbol
// 17) and 10 are the two escape distributions.
inline constexpr uint8_t kPulsesPerBlockIcdf[10][18] = {
    {
        125, 51, 26, 18, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
        2, 1, 0,
    },
    {
        198, 105, 45, 22, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
        2, 1, 0,
    },
    {
        213, 162, 116, 83, 59, 43, 32, 24, 18, 15, 12, 9, 7, 6, 5,
        3, 2, 0,
    },
    {
        239, 187, 116, 59, 28, 16, 11, 10, 9, 8, 7, 6, 5, 4, 3,
        2, 1, 0,
    },
    {
        250, 229, 188, 135, 86, 51, 30, 19, 13, 10, 8, 6, 5, 4, 3,
        2, 1, 0,
    },
    {
        249, 235, 213, 185, 156, 128, 103, 83, 66, 53, 42, 33, 26, 21, 17,
        13, 10, 0,
    },
    {
        254, 249, 235, 206, 164, 118, 77, 46, 27, 16, 10, 7, 5, 4, 3,
        2, 1, 0,
    },
    {
        255, 253, 249, 239, 220, 191, 156, 119, 85, 57, 37, 23, 15, 10, 6,
        4, 2, 0,
    },
    {
        255, 253, 251, 246, 237, 223, 203, 179, 152, 124, 98, 75, 55, 40, 29,
        21, 15, 0,
    },
    {
        255, 254, 253, 247, 220, 162, 106, 67, 42, 28, 18, 12, 9, 6, 4,
        3, 2, 0,
    },
};

// Rate level iCDF, [signalType >> 1] (unvoiced/voiced).
inline constexpr uint8_t kRateLevelsIcdf[2][9] = {
    { 241, 190, 178, 132, 87, 74, 41, 14, 0 },
    { 223, 193, 157, 140, 106, 57, 39, 18, 0 },
};

// Shell-coder split iCDFs, table 0: row for k pulses (k=1..16)
// has k+1 entries and starts at kShellCodeTableOffsets[k].
inline constexpr uint8_t kShellCodeTable0[152] = {
    128, 0, 214, 42, 0, 235, 128, 21, 0, 244, 184, 72, 11, 0, 248,
    214, 128, 42, 7, 0, 248, 225, 170, 80, 25, 5, 0, 251, 236, 198,
    126, 54, 18, 3, 0, 250, 238, 211, 159, 82, 35, 15, 5, 0, 250,
    231, 203, 168, 128, 88, 53, 25, 6, 0, 252, 238, 216, 185, 148, 108,
    71, 40, 18, 4, 0, 253, 243, 225, 199, 166, 128, 90, 57, 31, 13,
    3, 0, 254, 246, 233, 212, 183, 147, 109, 73, 44, 23, 10, 2, 0,
    255, 250, 240, 223, 198, 166, 128, 90, 58, 33, 16, 6, 1, 0, 255,
    251, 244, 231, 210, 181, 146, 110, 75, 46, 25, 12, 5, 1, 0, 255,
    253, 248, 238, 221, 196, 164, 128, 92, 60, 35, 18, 8, 3, 1, 0,
    255, 253, 249, 242, 229, 208, 180, 146, 110, 76, 48, 27, 14, 7, 3,
    1, 0,
};

// Shell-coder split iCDFs, table 1: row for k pulses (k=1..16)
// has k+1 entries and starts at kShellCodeTableOffsets[k].
inline constexpr uint8_t kShellCodeTable1[152] = {
    129, 0, 207, 50, 0, 236, 129, 20, 0, 245, 185, 72, 10, 0, 249,
    213, 129, 42, 6, 0, 250, 226, 169, 87, 27, 4, 0, 251, 233, 194,
    130, 62, 20, 4, 0, 250, 236, 207, 160, 99, 47, 17, 3, 0, 255,
    240, 217, 182, 131, 81, 41, 11, 1, 0, 255, 254, 233, 201, 159, 107,
    61, 20, 2, 1, 0, 255, 249, 233, 206, 170, 128, 86, 50, 23, 7,
    1, 0, 255, 250, 238, 217, 186, 148, 108, 70, 39, 18, 6, 1, 0,
    255, 252, 243, 226, 200, 166, 128, 90, 56, 30, 13, 4, 1, 0, 255,
    252, 245, 231, 209, 180, 146, 110, 76, 47, 25, 11, 4, 1, 0, 255,
    253, 248, 237, 219, 194, 163, 128, 93, 62, 37, 19, 8, 3, 1, 0,
    255, 254, 250, 241, 226, 205, 177, 145, 111, 79, 51, 30, 15, 6, 2,
    1, 0,
};

// Shell-coder split iCDFs, table 2: row for k pulses (k=1..16)
// has k+1 entries and starts at kShellCodeTableOffsets[k].
inline constexpr uint8_t kShellCodeTable2[152] = {
    129, 0, 203, 54, 0, 234, 129, 23, 0, 245, 184, 73, 10, 0, 250,
    215, 129, 41, 5, 0, 252, 232, 173, 86, 24, 3, 0, 253, 240, 200,
    129, 56, 15, 2, 0, 253, 244, 217, 164, 94, 38, 10, 1, 0, 253,
    245, 226, 189, 132, 71, 27, 7, 1, 0, 253, 246, 231, 203, 159, 105,
    56, 23, 6, 1, 0, 255, 248, 235, 213, 179, 133, 85, 47, 19, 5,
    1, 0, 255, 254, 243, 221, 194, 159, 117, 70, 37, 12, 2, 1, 0,
    255, 254, 248, 234, 208, 171, 128, 85, 48, 22, 8, 2, 1, 0, 255,
    254, 250, 240, 220, 189, 149, 107, 67, 36, 16, 6, 2, 1, 0, 255,
    254, 251, 243, 227, 201, 166, 128, 90, 55, 29, 13, 5, 2, 1, 0,
    255, 254, 252, 246, 234, 213, 183, 147, 109, 73, 43, 22, 10, 4, 2,
    1, 0,
};

// Shell-coder split iCDFs, table 3: row for k pulses (k=1..16)
// has k+1 entries and starts at kShellCodeTableOffsets[k].
inline constexpr uint8_t kShellCodeTable3[152] = {
    130, 0, 200, 58, 0, 231, 130, 26, 0, 244, 184, 76, 12, 0, 249,
    214, 130, 43, 6, 0, 252, 232, 173, 87, 24, 3, 0, 253, 241, 203,
    131, 56, 14, 2, 0, 254, 246, 221, 167, 94, 35, 8, 1, 0, 254,
    249, 232, 193, 130, 65, 23, 5, 1, 0, 255, 251, 239, 211, 162, 99,
    45, 15, 4, 1, 0, 255, 251, 243, 223, 186, 131, 74, 33, 11, 3,
    1, 0, 255, 252, 245, 230, 202, 158, 105, 57, 24, 8, 2, 1, 0,
    255, 253, 247, 235, 214, 179, 132, 84, 44, 19, 7, 2, 1, 0, 255,
    254, 250, 240, 223, 196, 159, 112, 69, 36, 15, 6, 2, 1, 0, 255,
    254, 253, 245, 231, 209, 176, 136, 93, 55, 27, 11, 3, 2, 1, 0,
    255, 254, 253, 252, 239, 221, 194, 158, 117, 76, 42, 18, 4, 3, 2,
    1, 0,
};

// Row offsets into the shell code tables.
inline constexpr uint8_t kShellCodeTableOffsets[17] = {
    0, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104,
    119, 135,
};

// Excitation LSB iCDF (used when nLshifts > 0).
inline constexpr uint8_t kLsbIcdf[2] = {
    120, 0,
};

// NOT an iCDF: 6 blocks of 7 single Q8 zero-probabilities indexed
// 7*(quantOffsetType + (signalType << 1)) + min(sum_pulses, 6); the
// decoder builds the 2-entry iCDF { p, 0 } per shell block (code_signs.c).
inline constexpr uint8_t kSignIcdf[42] = {
    254, 49, 67, 77, 82, 93, 99, 198, 11, 18, 24, 31, 36, 45, 255,
    46, 66, 78, 87, 94, 104, 208, 14, 21, 32, 42, 51, 66, 255, 94,
    104, 109, 112, 115, 118, 248, 53, 69, 80, 88, 95, 102,
};

// ---- LTP (silk/tables_LTP.c) ----------------------------------------------

// Periodicity index (selects the LTP codebook).
inline constexpr uint8_t kLtpPerIndexIcdf[3] = {
    179, 99, 0,
};

// LTP gain index iCDF, codebook 0 (8 vectors).
inline constexpr uint8_t kLtpGainIcdf0[8] = {
    71, 56, 43, 30, 21, 12, 6, 0,
};

// LTP gain index iCDF, codebook 1 (16 vectors).
inline constexpr uint8_t kLtpGainIcdf1[16] = {
    199, 165, 144, 124, 109, 96, 84, 71, 61, 51, 42, 32, 23, 15, 8,
    0,
};

// LTP gain index iCDF, codebook 2 (32 vectors).
inline constexpr uint8_t kLtpGainIcdf2[32] = {
    241, 225, 211, 199, 187, 175, 164, 153, 142, 132, 123, 114, 105, 96, 88,
    80, 72, 64, 57, 50, 44, 38, 33, 29, 24, 20, 16, 12, 9, 5,
    2, 0,
};

// LTP filter codebook 0, Q7, kLtpOrder taps per row.
inline constexpr int8_t kLtpGainVq0[8][5] = {
    { 4, 6, 24, 7, 5 },
    { 0, 0, 2, 0, 0 },
    { 12, 28, 41, 13, -4 },
    { -9, 15, 42, 25, 14 },
    { 1, -2, 62, 41, -9 },
    { -10, 37, 65, -4, 3 },
    { -6, 4, 66, 7, -8 },
    { 16, 14, 38, -3, 33 },
};

// LTP filter codebook 1, Q7.
inline constexpr int8_t kLtpGainVq1[16][5] = {
    { 13, 22, 39, 23, 12 },
    { -1, 36, 64, 27, -6 },
    { -7, 10, 55, 43, 17 },
    { 1, 1, 8, 1, 1 },
    { 6, -11, 74, 53, -9 },
    { -12, 55, 76, -12, 8 },
    { -3, 3, 93, 27, -4 },
    { 26, 39, 59, 3, -8 },
    { 2, 0, 77, 11, 9 },
    { -8, 22, 44, -6, 7 },
    { 40, 9, 26, 3, 9 },
    { -7, 20, 101, -7, 4 },
    { 3, -8, 42, 26, 0 },
    { -15, 33, 68, 2, 23 },
    { -2, 55, 46, -2, 15 },
    { 3, -1, 21, 16, 41 },
};

// LTP filter codebook 2, Q7.
inline constexpr int8_t kLtpGainVq2[32][5] = {
    { -6, 27, 61, 39, 5 },
    { -11, 42, 88, 4, 1 },
    { -2, 60, 65, 6, -4 },
    { -1, -5, 73, 56, 1 },
    { -9, 19, 94, 29, -9 },
    { 0, 12, 99, 6, 4 },
    { 8, -19, 102, 46, -13 },
    { 3, 2, 13, 3, 2 },
    { 9, -21, 84, 72, -18 },
    { -11, 46, 104, -22, 8 },
    { 18, 38, 48, 23, 0 },
    { -16, 70, 83, -21, 11 },
    { 5, -11, 117, 22, -8 },
    { -6, 23, 117, -12, 3 },
    { 3, -8, 95, 28, 4 },
    { -10, 15, 77, 60, -15 },
    { -1, 4, 124, 2, -4 },
    { 3, 38, 84, 24, -25 },
    { 2, 13, 42, 13, 31 },
    { 21, -4, 56, 46, -1 },
    { -1, 35, 79, -13, 19 },
    { -7, 65, 88, -9, -14 },
    { 20, 4, 81, 49, -29 },
    { 20, 0, 75, 3, -17 },
    { 5, -9, 44, 92, -8 },
    { 1, -3, 22, 69, 31 },
    { -6, 95, 41, -12, 5 },
    { 39, 67, 16, -4, 1 },
    { 0, -6, 120, 55, -36 },
    { -13, 44, 122, 4, -24 },
    { 81, 5, 11, 3, 7 },
    { 2, 0, 9, 10, 88 },
};

// Codebook sizes {8, 16, 32}.
inline constexpr int8_t kLtpVqSizes[3] = {
    8, 16, 32,
};

// Pointer tables mirroring silk_LTP_gain_iCDF_ptrs / silk_LTP_vq_ptrs_Q7
// (initializer order verified against the source).
inline constexpr const uint8_t* kLtpGainIcdfPtrs[kNbLtpCbks] = {
    kLtpGainIcdf0, kLtpGainIcdf1, kLtpGainIcdf2,
};
inline constexpr const int8_t* kLtpVqPtrsQ7[kNbLtpCbks] = {
    &kLtpGainVq0[0][0], &kLtpGainVq1[0][0], &kLtpGainVq2[0][0],
};

// LTP scaling parameter iCDF (silk/tables_other.c).
inline constexpr uint8_t kLtpScaleIcdf[3] = {
    128, 64, 0,
};

// LTP scale values, Q14.
inline constexpr int16_t kLtpScalesTableQ14[3] = {
    15565, 12288, 8192,
};

// ---- NLSF codebooks (silk/tables_NLSF_CB_NB_MB.c, tables_NLSF_CB_WB.c) ---
//
// Layout mirrors silk_NLSF_CB_struct: CB1 stage-1 vectors (nVectors x
// order, Q8) with per-coefficient weights (Q9); CB1_iCDF holds TWO iCDFs
// of nVectors entries, selected by signalType >> 1; ec_sel packs two
// 4-bit entries per byte -> per-coefficient residual iCDF row (of length
// 2*kNlsfQuantMaxAmplitude + 1 within ec_iCDF) and predictor choice from
// pred_Q8; deltaMin_Q15 (order+1) feeds NLSF stabilization. ec_Rates_Q5
// is encoder-only but kept to mirror the struct (72 B per codebook).

// NB/MB stage-1 codebook, 32 vectors x order 10.
inline constexpr uint8_t kNlsfCb1NbMbQ8[320] = {
    12, 35, 60, 83, 108, 132, 157, 180, 206, 228, 15, 32, 55, 77, 101,
    125, 151, 175, 201, 225, 19, 42, 66, 89, 114, 137, 162, 184, 209, 230,
    12, 25, 50, 72, 97, 120, 147, 172, 200, 223, 26, 44, 69, 90, 114,
    135, 159, 180, 205, 225, 13, 22, 53, 80, 106, 130, 156, 180, 205, 228,
    15, 25, 44, 64, 90, 115, 142, 168, 196, 222, 19, 24, 62, 82, 100,
    120, 145, 168, 190, 214, 22, 31, 50, 79, 103, 120, 151, 170, 203, 227,
    21, 29, 45, 65, 106, 124, 150, 171, 196, 224, 30, 49, 75, 97, 121,
    142, 165, 186, 209, 229, 19, 25, 52, 70, 93, 116, 143, 166, 192, 219,
    26, 34, 62, 75, 97, 118, 145, 167, 194, 217, 25, 33, 56, 70, 91,
    113, 143, 165, 196, 223, 21, 34, 51, 72, 97, 117, 145, 171, 196, 222,
    20, 29, 50, 67, 90, 117, 144, 168, 197, 221, 22, 31, 48, 66, 95,
    117, 146, 168, 196, 222, 24, 33, 51, 77, 116, 134, 158, 180, 200, 224,
    21, 28, 70, 87, 106, 124, 149, 170, 194, 217, 26, 33, 53, 64, 83,
    117, 152, 173, 204, 225, 27, 34, 65, 95, 108, 129, 155, 174, 210, 225,
    20, 26, 72, 99, 113, 131, 154, 176, 200, 219, 34, 43, 61, 78, 93,
    114, 155, 177, 205, 229, 23, 29, 54, 97, 124, 138, 163, 179, 209, 229,
    30, 38, 56, 89, 118, 129, 158, 178, 200, 231, 21, 29, 49, 63, 85,
    111, 142, 163, 193, 222, 27, 48, 77, 103, 133, 158, 179, 196, 215, 232,
    29, 47, 74, 99, 124, 151, 176, 198, 220, 237, 33, 42, 61, 76, 93,
    121, 155, 174, 207, 225, 29, 53, 87, 112, 136, 154, 170, 188, 208, 227,
    24, 30, 52, 84, 131, 150, 166, 186, 203, 229, 37, 48, 64, 84, 104,
    118, 156, 177, 201, 230,
};

// NB/MB stage-1 weights.
inline constexpr int16_t kNlsfCb1WghtQ9[320] = {
    2897, 2314, 2314, 2314, 2287, 2287, 2314, 2300, 2327, 2287,
    2888, 2580, 2394, 2367, 2314, 2274, 2274, 2274, 2274, 2194,
    2487, 2340, 2340, 2314, 2314, 2314, 2340, 2340, 2367, 2354,
    3216, 2766, 2340, 2340, 2314, 2274, 2221, 2207, 2261, 2194,
    2460, 2474, 2367, 2394, 2394, 2394, 2394, 2367, 2407, 2314,
    3479, 3056, 2127, 2207, 2274, 2274, 2274, 2287, 2314, 2261,
    3282, 3141, 2580, 2394, 2247, 2221, 2207, 2194, 2194, 2114,
    4096, 3845, 2221, 2620, 2620, 2407, 2314, 2394, 2367, 2074,
    3178, 3244, 2367, 2221, 2553, 2434, 2340, 2314, 2167, 2221,
    3338, 3488, 2726, 2194, 2261, 2460, 2354, 2367, 2207, 2101,
    2354, 2420, 2327, 2367, 2394, 2420, 2420, 2420, 2460, 2367,
    3779, 3629, 2434, 2527, 2367, 2274, 2274, 2300, 2207, 2048,
    3254, 3225, 2713, 2846, 2447, 2327, 2300, 2300, 2274, 2127,
    3263, 3300, 2753, 2806, 2447, 2261, 2261, 2247, 2127, 2101,
    2873, 2981, 2633, 2367, 2407, 2354, 2194, 2247, 2247, 2114,
    3225, 3197, 2633, 2580, 2274, 2181, 2247, 2221, 2221, 2141,
    3178, 3310, 2740, 2407, 2274, 2274, 2274, 2287, 2194, 2114,
    3141, 3272, 2460, 2061, 2287, 2500, 2367, 2487, 2434, 2181,
    3507, 3282, 2314, 2700, 2647, 2474, 2367, 2394, 2340, 2127,
    3423, 3535, 3038, 3056, 2300, 1950, 2221, 2274, 2274, 2274,
    3404, 3366, 2087, 2687, 2873, 2354, 2420, 2274, 2474, 2540,
    3760, 3488, 1950, 2660, 2897, 2527, 2394, 2367, 2460, 2261,
    3028, 3272, 2740, 2888, 2740, 2154, 2127, 2287, 2234, 2247,
    3695, 3657, 2025, 1969, 2660, 2700, 2580, 2500, 2327, 2367,
    3207, 3413, 2354, 2074, 2888, 2888, 2340, 2487, 2247, 2167,
    3338, 3366, 2846, 2780, 2327, 2154, 2274, 2287, 2114, 2061,
    2327, 2300, 2181, 2167, 2181, 2367, 2633, 2700, 2700, 2553,
    2407, 2434, 2221, 2261, 2221, 2221, 2340, 2420, 2607, 2700,
    3038, 3244, 2806, 2888, 2474, 2074, 2300, 2314, 2354, 2380,
    2221, 2154, 2127, 2287, 2500, 2793, 2793, 2620, 2580, 2367,
    3676, 3713, 2234, 1838, 2181, 2753, 2726, 2673, 2513, 2207,
    2793, 3160, 2726, 2553, 2846, 2513, 2181, 2394, 2221, 2181,
};

// NB/MB stage-1 index iCDFs (2 x 32).
inline constexpr uint8_t kNlsfCb1IcdfNbMb[64] = {
    212, 178, 148, 129, 108, 96, 85, 82, 79, 77, 61, 59, 57, 56, 51,
    49, 48, 45, 42, 41, 40, 38, 36, 34, 31, 30, 21, 12, 10, 3,
    1, 0, 255, 245, 244, 236, 233, 225, 217, 203, 190, 176, 175, 161, 149,
    136, 125, 114, 102, 91, 81, 71, 60, 52, 43, 35, 28, 20, 19, 18,
    12, 11, 5, 0,
};

// NB/MB ec_sel, 32 vectors x order/2 bytes.
inline constexpr uint8_t kNlsfCb2SelectNbMb[160] = {
    16, 0, 0, 0, 0, 99, 66, 36, 36, 34, 36, 34, 34, 34, 34,
    83, 69, 36, 52, 34, 116, 102, 70, 68, 68, 176, 102, 68, 68, 34,
    65, 85, 68, 84, 36, 116, 141, 152, 139, 170, 132, 187, 184, 216, 137,
    132, 249, 168, 185, 139, 104, 102, 100, 68, 68, 178, 218, 185, 185, 170,
    244, 216, 187, 187, 170, 244, 187, 187, 219, 138, 103, 155, 184, 185, 137,
    116, 183, 155, 152, 136, 132, 217, 184, 184, 170, 164, 217, 171, 155, 139,
    244, 169, 184, 185, 170, 164, 216, 223, 218, 138, 214, 143, 188, 218, 168,
    244, 141, 136, 155, 170, 168, 138, 220, 219, 139, 164, 219, 202, 216, 137,
    168, 186, 246, 185, 139, 116, 185, 219, 185, 138, 100, 100, 134, 100, 102,
    34, 68, 68, 100, 68, 168, 203, 221, 218, 168, 167, 154, 136, 104, 70,
    164, 246, 171, 137, 139, 137, 155, 218, 219, 139,
};

// NB/MB residual iCDFs, 8 rows of 9.
inline constexpr uint8_t kNlsfCb2IcdfNbMb[72] = {
    255, 254, 253, 238, 14, 3, 2, 1, 0, 255, 254, 252, 218, 35, 3,
    2, 1, 0, 255, 254, 250, 208, 59, 4, 2, 1, 0, 255, 254, 246,
    194, 71, 10, 2, 1, 0, 255, 252, 236, 183, 82, 8, 2, 1, 0,
    255, 252, 235, 180, 90, 17, 2, 1, 0, 255, 248, 224, 171, 97, 30,
    4, 1, 0, 255, 254, 236, 173, 95, 37, 7, 1, 0,
};

// NB/MB residual rates, Q5 (encoder-only).
inline constexpr uint8_t kNlsfCb2BitsNbMbQ5[72] = {
    255, 255, 255, 131, 6, 145, 255, 255, 255, 255, 255, 236, 93, 15, 96,
    255, 255, 255, 255, 255, 194, 83, 25, 71, 221, 255, 255, 255, 255, 162,
    73, 34, 66, 162, 255, 255, 255, 210, 126, 73, 43, 57, 173, 255, 255,
    255, 201, 125, 71, 48, 58, 130, 255, 255, 255, 166, 110, 73, 57, 62,
    104, 210, 255, 255, 251, 123, 65, 55, 68, 100, 171, 255,
};

// NB/MB backwards predictors, 2 x (order-1).
inline constexpr uint8_t kNlsfPredNbMbQ8[18] = {
    179, 138, 140, 148, 151, 149, 153, 151, 163, 116, 67, 82, 59, 92, 72,
    100, 89, 92,
};

// NB/MB minimum NLSF spacing, order+1.
inline constexpr int16_t kNlsfDeltaMinNbMbQ15[11] = {
    250, 3, 6, 3, 3, 3, 4, 3, 3, 3,
    461,
};

// WB stage-1 codebook, 32 vectors x order 16.
inline constexpr uint8_t kNlsfCb1WbQ8[512] = {
    7, 23, 38, 54, 69, 85, 100, 116, 131, 147, 162, 178, 193, 208, 223,
    239, 13, 25, 41, 55, 69, 83, 98, 112, 127, 142, 157, 171, 187, 203,
    220, 236, 15, 21, 34, 51, 61, 78, 92, 106, 126, 136, 152, 167, 185,
    205, 225, 240, 10, 21, 36, 50, 63, 79, 95, 110, 126, 141, 157, 173,
    189, 205, 221, 237, 17, 20, 37, 51, 59, 78, 89, 107, 123, 134, 150,
    164, 184, 205, 224, 240, 10, 15, 32, 51, 67, 81, 96, 112, 129, 142,
    158, 173, 189, 204, 220, 236, 8, 21, 37, 51, 65, 79, 98, 113, 126,
    138, 155, 168, 179, 192, 209, 218, 12, 15, 34, 55, 63, 78, 87, 108,
    118, 131, 148, 167, 185, 203, 219, 236, 16, 19, 32, 36, 56, 79, 91,
    108, 118, 136, 154, 171, 186, 204, 220, 237, 11, 28, 43, 58, 74, 89,
    105, 120, 135, 150, 165, 180, 196, 211, 226, 241, 6, 16, 33, 46, 60,
    75, 92, 107, 123, 137, 156, 169, 185, 199, 214, 225, 11, 19, 30, 44,
    57, 74, 89, 105, 121, 135, 152, 169, 186, 202, 218, 234, 12, 19, 29,
    46, 57, 71, 88, 100, 120, 132, 148, 165, 182, 199, 216, 233, 17, 23,
    35, 46, 56, 77, 92, 106, 123, 134, 152, 167, 185, 204, 222, 237, 14,
    17, 45, 53, 63, 75, 89, 107, 115, 132, 151, 171, 188, 206, 221, 240,
    9, 16, 29, 40, 56, 71, 88, 103, 119, 137, 154, 171, 189, 205, 222,
    237, 16, 19, 36, 48, 57, 76, 87, 105, 118, 132, 150, 167, 185, 202,
    218, 236, 12, 17, 29, 54, 71, 81, 94, 104, 126, 136, 149, 164, 182,
    201, 221, 237, 15, 28, 47, 62, 79, 97, 115, 129, 142, 155, 168, 180,
    194, 208, 223, 238, 8, 14, 30, 45, 62, 78, 94, 111, 127, 143, 159,
    175, 192, 207, 223, 239, 17, 30, 49, 62, 79, 92, 107, 119, 132, 145,
    160, 174, 190, 204, 220, 235, 14, 19, 36, 45, 61, 76, 91, 108, 121,
    138, 154, 172, 189, 205, 222, 238, 12, 18, 31, 45, 60, 76, 91, 107,
    123, 138, 154, 171, 187, 204, 221, 236, 13, 17, 31, 43, 53, 70, 83,
    103, 114, 131, 149, 167, 185, 203, 220, 237, 17, 22, 35, 42, 58, 78,
    93, 110, 125, 139, 155, 170, 188, 206, 224, 240, 8, 15, 34, 50, 67,
    83, 99, 115, 131, 146, 162, 178, 193, 209, 224, 239, 13, 16, 41, 66,
    73, 86, 95, 111, 128, 137, 150, 163, 183, 206, 225, 241, 17, 25, 37,
    52, 63, 75, 92, 102, 119, 132, 144, 160, 175, 191, 212, 231, 19, 31,
    49, 65, 83, 100, 117, 133, 147, 161, 174, 187, 200, 213, 227, 242, 18,
    31, 52, 68, 88, 103, 117, 126, 138, 149, 163, 177, 192, 207, 223, 239,
    16, 29, 47, 61, 76, 90, 106, 119, 133, 147, 161, 176, 193, 209, 224,
    240, 15, 21, 35, 50, 61, 73, 86, 97, 110, 119, 129, 141, 175, 198,
    218, 237,
};

// WB stage-1 weights.
inline constexpr int16_t kNlsfCb1WbWghtQ9[512] = {
    3657, 2925, 2925, 2925, 2925, 2925, 2925, 2925, 2925, 2925,
    2925, 2925, 2963, 2963, 2925, 2846, 3216, 3085, 2972, 3056,
    3056, 3010, 3010, 3010, 2963, 2963, 3010, 2972, 2888, 2846,
    2846, 2726, 3920, 4014, 2981, 3207, 3207, 2934, 3056, 2846,
    3122, 3244, 2925, 2846, 2620, 2553, 2780, 2925, 3516, 3197,
    3010, 3103, 3019, 2888, 2925, 2925, 2925, 2925, 2888, 2888,
    2888, 2888, 2888, 2753, 5054, 5054, 2934, 3573, 3385, 3056,
    3085, 2793, 3160, 3160, 2972, 2846, 2513, 2540, 2753, 2888,
    4428, 4149, 2700, 2753, 2972, 3010, 2925, 2846, 2981, 3019,
    2925, 2925, 2925, 2925, 2888, 2726, 3620, 3019, 2972, 3056,
    3056, 2873, 2806, 3056, 3216, 3047, 2981, 3291, 3291, 2981,
    3310, 2991, 5227, 5014, 2540, 3338, 3526, 3385, 3197, 3094,
    3376, 2981, 2700, 2647, 2687, 2793, 2846, 2673, 5081, 5174,
    4615, 4428, 2460, 2897, 3047, 3207, 3169, 2687, 2740, 2888,
    2846, 2793, 2846, 2700, 3122, 2888, 2963, 2925, 2925, 2925,
    2925, 2963, 2963, 2963, 2963, 2925, 2925, 2963, 2963, 2963,
    4202, 3207, 2981, 3103, 3010, 2888, 2888, 2925, 2972, 2873,
    2916, 3019, 2972, 3010, 3197, 2873, 3760, 3760, 3244, 3103,
    2981, 2888, 2925, 2888, 2972, 2934, 2793, 2793, 2846, 2888,
    2888, 2660, 3854, 4014, 3207, 3122, 3244, 2934, 3047, 2963,
    2963, 3085, 2846, 2793, 2793, 2793, 2793, 2580, 3845, 4080,
    3357, 3516, 3094, 2740, 3010, 2934, 3122, 3085, 2846, 2846,
    2647, 2647, 2846, 2806, 5147, 4894, 3225, 3845, 3441, 3169,
    2897, 3413, 3451, 2700, 2580, 2673, 2740, 2846, 2806, 2753,
    4109, 3789, 3291, 3160, 2925, 2888, 2888, 2925, 2793, 2740,
    2793, 2740, 2793, 2846, 2888, 2806, 5081, 5054, 3047, 3545,
    3244, 3056, 3085, 2944, 3103, 2897, 2740, 2740, 2740, 2846,
    2793, 2620, 4309, 4309, 2860, 2527, 3207, 3376, 3376, 3075,
    3075, 3376, 3056, 2846, 2647, 2580, 2726, 2753, 3056, 2916,
    2806, 2888, 2740, 2687, 2897, 3103, 3150, 3150, 3216, 3169,
    3056, 3010, 2963, 2846, 4375, 3882, 2925, 2888, 2846, 2888,
    2846, 2846, 2888, 2888, 2888, 2846, 2888, 2925, 2888, 2846,
    2981, 2916, 2916, 2981, 2981, 3056, 3122, 3216, 3150, 3056,
    3010, 2972, 2972, 2972, 2925, 2740, 4229, 4149, 3310, 3347,
    2925, 2963, 2888, 2981, 2981, 2846, 2793, 2740, 2846, 2846,
    2846, 2793, 4080, 4014, 3103, 3010, 2925, 2925, 2925, 2888,
    2925, 2925, 2846, 2846, 2846, 2793, 2888, 2780, 4615, 4575,
    3169, 3441, 3207, 2981, 2897, 3038, 3122, 2740, 2687, 2687,
    2687, 2740, 2793, 2700, 4149, 4269, 3789, 3657, 2726, 2780,
    2888, 2888, 3010, 2972, 2925, 2846, 2687, 2687, 2793, 2888,
    4215, 3554, 2753, 2846, 2846, 2888, 2888, 2888, 2925, 2925,
    2888, 2925, 2925, 2925, 2963, 2888, 5174, 4921, 2261, 3432,
    3789, 3479, 3347, 2846, 3310, 3479, 3150, 2897, 2460, 2487,
    2753, 2925, 3451, 3685, 3122, 3197, 3357, 3047, 3207, 3207,
    2981, 3216, 3085, 2925, 2925, 2687, 2540, 2434, 2981, 3010,
    2793, 2793, 2740, 2793, 2846, 2972, 3056, 3103, 3150, 3150,
    3150, 3103, 3010, 3010, 2944, 2873, 2687, 2726, 2780, 3010,
    3432, 3545, 3357, 3244, 3056, 3010, 2963, 2925, 2888, 2846,
    3019, 2944, 2897, 3010, 3010, 2972, 3019, 3103, 3056, 3056,
    3010, 2888, 2846, 2925, 2925, 2888, 3920, 3967, 3010, 3197,
    3357, 3216, 3291, 3291, 3479, 3704, 3441, 2726, 2181, 2460,
    2580, 2607,
};

// WB stage-1 index iCDFs (2 x 32).
inline constexpr uint8_t kNlsfCb1IcdfWb[64] = {
    225, 204, 201, 184, 183, 175, 158, 154, 153, 135, 119, 115, 113, 110, 109,
    99, 98, 95, 79, 68, 52, 50, 48, 45, 43, 32, 31, 27, 18, 10,
    3, 0, 255, 251, 235, 230, 212, 201, 196, 182, 167, 166, 163, 151, 138,
    124, 110, 104, 90, 78, 76, 70, 69, 57, 45, 34, 24, 21, 11, 6,
    5, 4, 3, 0,
};

// WB ec_sel, 32 vectors x order/2 bytes.
inline constexpr uint8_t kNlsfCb2SelectWb[256] = {
    0, 0, 0, 0, 0, 0, 0, 1, 100, 102, 102, 68, 68, 36, 34,
    96, 164, 107, 158, 185, 180, 185, 139, 102, 64, 66, 36, 34, 34, 0,
    1, 32, 208, 139, 141, 191, 152, 185, 155, 104, 96, 171, 104, 166, 102,
    102, 102, 132, 1, 0, 0, 0, 0, 16, 16, 0, 80, 109, 78, 107,
    185, 139, 103, 101, 208, 212, 141, 139, 173, 153, 123, 103, 36, 0, 0,
    0, 0, 0, 0, 1, 48, 0, 0, 0, 0, 0, 0, 32, 68, 135,
    123, 119, 119, 103, 69, 98, 68, 103, 120, 118, 118, 102, 71, 98, 134,
    136, 157, 184, 182, 153, 139, 134, 208, 168, 248, 75, 189, 143, 121, 107,
    32, 49, 34, 34, 34, 0, 17, 2, 210, 235, 139, 123, 185, 137, 105,
    134, 98, 135, 104, 182, 100, 183, 171, 134, 100, 70, 68, 70, 66, 66,
    34, 131, 64, 166, 102, 68, 36, 2, 1, 0, 134, 166, 102, 68, 34,
    34, 66, 132, 212, 246, 158, 139, 107, 107, 87, 102, 100, 219, 125, 122,
    137, 118, 103, 132, 114, 135, 137, 105, 171, 106, 50, 34, 164, 214, 141,
    143, 185, 151, 121, 103, 192, 34, 0, 0, 0, 0, 0, 1, 208, 109,
    74, 187, 134, 249, 159, 137, 102, 110, 154, 118, 87, 101, 119, 101, 0,
    2, 0, 36, 36, 66, 68, 35, 96, 164, 102, 100, 36, 0, 2, 33,
    167, 138, 174, 102, 100, 84, 2, 2, 100, 107, 120, 119, 36, 197, 24,
    0,
};

// WB residual iCDFs, 8 rows of 9.
inline constexpr uint8_t kNlsfCb2IcdfWb[72] = {
    255, 254, 253, 244, 12, 3, 2, 1, 0, 255, 254, 252, 224, 38, 3,
    2, 1, 0, 255, 254, 251, 209, 57, 4, 2, 1, 0, 255, 254, 244,
    195, 69, 4, 2, 1, 0, 255, 251, 232, 184, 84, 7, 2, 1, 0,
    255, 254, 240, 186, 86, 14, 2, 1, 0, 255, 254, 239, 178, 91, 30,
    5, 1, 0, 255, 248, 227, 177, 100, 19, 2, 1, 0,
};

// WB residual rates, Q5 (encoder-only).
inline constexpr uint8_t kNlsfCb2BitsWbQ5[72] = {
    255, 255, 255, 156, 4, 154, 255, 255, 255, 255, 255, 227, 102, 15, 92,
    255, 255, 255, 255, 255, 213, 83, 24, 72, 236, 255, 255, 255, 255, 150,
    76, 33, 63, 214, 255, 255, 255, 190, 121, 77, 43, 55, 185, 255, 255,
    255, 245, 137, 71, 43, 59, 139, 255, 255, 255, 255, 131, 66, 50, 66,
    107, 194, 255, 255, 166, 116, 76, 55, 53, 125, 255, 255,
};

// WB backwards predictors, 2 x (order-1).
inline constexpr uint8_t kNlsfPredWbQ8[30] = {
    175, 148, 160, 176, 178, 173, 174, 164, 177, 174, 196, 182, 198, 192, 182,
    68, 62, 66, 60, 72, 117, 85, 90, 118, 136, 151, 142, 160, 142, 155,
};

// WB minimum NLSF spacing, order+1.
inline constexpr int16_t kNlsfDeltaMinWbQ15[17] = {
    100, 3, 40, 3, 3, 3, 5, 14, 14, 10,
    11, 3, 8, 9, 7, 3, 347,
};

// Mirror of silk_NLSF_CB_struct (silk/structs.h; field order verified).
struct NlsfCodebook {
    int16_t nVectors;
    int16_t order;
    int16_t quantStepSizeQ16;
    int16_t invQuantStepSizeQ6;
    const uint8_t* cb1NlsfQ8;
    const int16_t* cb1WghtQ9;
    const uint8_t* cb1Icdf;
    const uint8_t* predQ8;
    const uint8_t* ecSel;
    const uint8_t* ecIcdf;
    const uint8_t* ecRatesQ5;
    const int16_t* deltaMinQ15;
};
inline constexpr NlsfCodebook kNlsfCbNbMb = {
    32, 10, 11796, 356,
    kNlsfCb1NbMbQ8, kNlsfCb1WghtQ9, kNlsfCb1IcdfNbMb, kNlsfPredNbMbQ8,
    kNlsfCb2SelectNbMb, kNlsfCb2IcdfNbMb, kNlsfCb2BitsNbMbQ5, kNlsfDeltaMinNbMbQ15,
};
inline constexpr NlsfCodebook kNlsfCbWb = {
    32, 16, 9830, 427,
    kNlsfCb1WbQ8, kNlsfCb1WbWghtQ9, kNlsfCb1IcdfWb, kNlsfPredWbQ8,
    kNlsfCb2SelectWb, kNlsfCb2IcdfWb, kNlsfCb2BitsWbQ5, kNlsfDeltaMinWbQ15,
};

// NLSF interpolation factor for 20 ms frames (silk/tables_other.c).
inline constexpr uint8_t kNlsfInterpolationFactorIcdf[5] = {
    243, 221, 192, 181, 0,
};

// NLSF residual extension (values beyond +-4).
inline constexpr uint8_t kNlsfExtIcdf[7] = {
    100, 40, 16, 7, 3, 1, 0,
};

// Piecewise-linear cosine table for NLSF -> LPC (silk/table_LSF_cos.c),
// Q12, kLsfCosTabSize + 1 entries over [0, pi].
inline constexpr int16_t kLsfCosTabFixQ12[129] = {
    8192, 8190, 8182, 8170, 8152, 8130, 8104, 8072, 8034, 7994,
    7946, 7896, 7840, 7778, 7714, 7644, 7568, 7490, 7406, 7318,
    7226, 7128, 7026, 6922, 6812, 6698, 6580, 6458, 6332, 6204,
    6070, 5934, 5792, 5648, 5502, 5352, 5198, 5040, 4880, 4718,
    4552, 4382, 4212, 4038, 3862, 3684, 3502, 3320, 3136, 2948,
    2760, 2570, 2378, 2186, 1990, 1794, 1598, 1400, 1202, 1002,
    802, 602, 402, 202, 0, -202, -402, -602, -802, -1002,
    -1202, -1400, -1598, -1794, -1990, -2186, -2378, -2570, -2760, -2948,
    -3136, -3320, -3502, -3684, -3862, -4038, -4212, -4382, -4552, -4718,
    -4880, -5040, -5198, -5352, -5502, -5648, -5792, -5934, -6070, -6204,
    -6332, -6458, -6580, -6698, -6812, -6922, -7026, -7128, -7226, -7318,
    -7406, -7490, -7568, -7644, -7714, -7778, -7840, -7896, -7946, -7994,
    -8034, -8072, -8104, -8130, -8152, -8170, -8182, -8190, -8192,
};

// ---- stereo, frame type, misc (silk/tables_other.c) -----------------------

// Stereo predictor quantization levels.
inline constexpr int16_t kStereoPredQuantQ13[16] = {
    -13732, -10050, -8266, -7526, -6500, -5000, -2950, -820, 820, 2950,
    5000, 6500, 7526, 8266, 10050, 13732,
};

// Joint iCDF for the two predictor MSB indices.
inline constexpr uint8_t kStereoPredJointIcdf[25] = {
    249, 247, 246, 245, 244, 234, 210, 202, 201, 200, 197, 174, 82, 59, 56,
    55, 54, 46, 22, 12, 11, 10, 9, 7, 0,
};

// P(side channel not coded).
inline constexpr uint8_t kStereoOnlyCodeMidIcdf[2] = {
    64, 0,
};

// LBRR flags, 2 frames per packet (40 ms).
inline constexpr uint8_t kLbrrFlags2Icdf[3] = {
    203, 150, 0,
};

// LBRR flags, 3 frames per packet (60 ms).
inline constexpr uint8_t kLbrrFlags3Icdf[7] = {
    215, 195, 166, 125, 110, 82, 0,
};

// Mirror of silk_LBRR_flags_iCDF_ptr, indexed nFramesPerPacket - 2.
inline constexpr const uint8_t* kLbrrFlagsIcdfPtr[2] = {
    kLbrrFlags2Icdf, kLbrrFlags3Icdf,
};

// Frame type when VAD active: (signalType - 1) * 2 + quantOffsetType.
inline constexpr uint8_t kTypeOffsetVadIcdf[4] = {
    232, 158, 10, 0,
};

// Frame type when VAD inactive.
inline constexpr uint8_t kTypeOffsetNoVadIcdf[2] = {
    230, 0,
};

// Excitation offsets, [signalType >> 1][quantOffsetType], Q10.
inline constexpr int16_t kQuantizationOffsetsQ10[2][2] = {
    { 100, 240 },
    { 32, 100 },
};

// Uniform iCDFs (stereo predictor LSBs, pitch lag LSBs, LCG seed, ...).
inline constexpr uint8_t kUniform3Icdf[3] = {
    171, 85, 0,
};

inline constexpr uint8_t kUniform4Icdf[4] = {
    192, 128, 64, 0,
};

inline constexpr uint8_t kUniform5Icdf[5] = {
    205, 154, 102, 51, 0,
};

inline constexpr uint8_t kUniform6Icdf[6] = {
    213, 171, 128, 85, 43, 0,
};

inline constexpr uint8_t kUniform8Icdf[8] = {
    224, 192, 160, 128, 96, 64, 32, 0,
};

// Bandwidth-switching transition LP filter, FIR interpolation points
// (silk/LP_variable_cutoff.c), Q28.
inline constexpr int32_t kTransitionLpBQ28[5][3] = {
    { 250767114, 501534038, 250767114 },
    { 209867381, 419732057, 209867381 },
    { 170987846, 341967853, 170987846 },
    { 131531482, 263046905, 131531482 },
    { 89306658, 178584282, 89306658 },
};

// Transition LP filter, IIR interpolation points.
inline constexpr int32_t kTransitionLpAQ28[5][2] = {
    { 506393414, 239854379 },
    { 411067935, 169683996 },
    { 306733530, 116694253 },
    { 185807084, 77959395 },
    { 35497197, 57401098 },
};

// ---- resampler ROM (silk/resampler_rom.{h,c}) ------------------------------
// COEFS layout: 2 IIR coefs then the FIR half-phases (order/2 per phase).

// Allpass coefficients for the 2x downsampler, Q16 (resampler_down2).
inline constexpr int16_t kResamplerDown2Coef0 = 9872;
inline constexpr int16_t kResamplerDown2Coef1 = -25727;

// Allpass coefficients for the high-quality 2x upsampler, Q16.
inline constexpr int16_t kResamplerUp2Hq0[3] = {
    1746, 14986, -26453,
};

inline constexpr int16_t kResamplerUp2Hq1[3] = {
    6854, 25769, -9994,
};

// Fractional downsampler 3/4 (e.g. 16 -> 12 kHz).
inline constexpr int16_t kResampler34Coefs[29] = {
    -20694, -13867, -49, 64, 17, -157, 353, -496, 163, 11047,
    22205, -39, 6, 91, -170, 186, 23, -896, 6336, 19928,
    -19, -36, 102, -89, -24, 328, -951, 2568, 15909,
};

// Fractional downsampler 2/3.
inline constexpr int16_t kResampler23Coefs[20] = {
    -14457, -14019, 64, 128, -122, 36, 310, -768, 584, 9267,
    17733, 12, 128, 18, -142, 288, -117, -865, 4123, 14459,
};

// Downsampler 1/2.
inline constexpr int16_t kResampler12Coefs[14] = {
    616, -14323, -10, 39, 58, -46, -84, 120, 184, -315,
    -541, 1284, 5380, 9024,
};

// Downsampler 1/3.
inline constexpr int16_t kResampler13Coefs[20] = {
    16102, -15162, -13, 0, 20, 26, 5, -31, -43, -4,
    65, 90, 7, -157, -248, -44, 593, 1583, 2612, 3271,
};

// Downsampler 1/4.
inline constexpr int16_t kResampler14Coefs[20] = {
    22500, -15099, 3, -14, -20, -15, 2, 25, 37, 25,
    -16, -71, -107, -79, 50, 292, 623, 982, 1288, 1464,
};

// Downsampler 1/6.
inline constexpr int16_t kResampler16Coefs[20] = {
    27540, -15257, 17, 12, 8, 1, -10, -22, -30, -32,
    -22, 3, 44, 100, 168, 243, 317, 381, 429, 455,
};

// Low-quality 2/3 downsampler (down2_3).
inline constexpr int16_t kResampler23CoefsLq[6] = {
    -2797, -6507, 4697, 10739, 1567, 8276,
};

// Interpolating FIR half-phases for fractions 1/24, 3/24, ..., 23/24
// (resampler_private_IIR_FIR upsampling).
inline constexpr int16_t kResamplerFracFir12[12][4] = {
    { 189, -600, 617, 30567 },
    { 117, -159, -1070, 29704 },
    { 52, 221, -2392, 28276 },
    { -4, 529, -3350, 26341 },
    { -48, 758, -3956, 23973 },
    { -80, 905, -4235, 21254 },
    { -99, 972, -4222, 18278 },
    { -107, 967, -3957, 15143 },
    { -103, 896, -3487, 11950 },
    { -91, 773, -2865, 8798 },
    { -71, 611, -2143, 5784 },
    { -46, 425, -1375, 2996 },
};

}  // namespace silk
}  // namespace opus
}  // namespace glint

#endif  // GLINT_OPUS_SILK_TABLES_HPP
