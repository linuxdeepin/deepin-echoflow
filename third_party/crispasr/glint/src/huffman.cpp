// glint - Huffman encoding implementation
// MIT License - Clean-room implementation

#include "huffman.hpp"
#include <cstdlib>
#include <algorithm>

namespace glint {

namespace {

// Per-table pair-cost LUT: cost[t][(ax<<4)|ay] holds exactly what
// tables::huff_code_length(t, ax, ay) returns for |x|,|y| <= 15 — code length
// + sign bits, + linbits for ESC tables at the ax/ay == 15 boundary (invalid
// entries hold the same 100 penalty). huff_code_length re-fetches the
// HuffTable descriptor through a 34-way switch and re-derives sign/ESC
// handling per pair; this collapses the per-pair cost in the bit-counting hot
// loop to one byte load. Only pair tables 0-31 are represented (32/33 are the
// quad count1 tables, handled by kCount1Cost below).
struct PairCostLUT {
    uint8_t cost[32][256];
    PairCostLUT() {
        for (int t = 0; t < 32; t++)
            for (int ax = 0; ax < 16; ax++)
                for (int ay = 0; ay < 16; ay++)
                    cost[t][(ax << 4) | ay] =
                        static_cast<uint8_t>(tables::huff_code_length(t, ax, ay));
    }
};
const PairCostLUT kPairCost;

// ESC tables clamp values >= 15 into the boundary entry (which already
// includes linbits), so the LUT stays exact; non-ESC tables return the same
// 100 penalty huff_code_length produces for out-of-range values.
inline int pair_cost(const uint8_t* lut, bool esc, int x, int y) {
    unsigned ax = static_cast<unsigned>(x < 0 ? -x : x);
    unsigned ay = static_cast<unsigned>(y < 0 ? -y : y);
    if ((ax | ay) > 15u) {
        if (!esc) return 100;
        if (ax > 15u) ax = 15u;
        if (ay > 15u) ay = 15u;
    }
    return lut[(ax << 4) | ay];
}

// count1 quad cost: code length + one sign bit per nonzero value, indexed by
// the 4-bit nonzero mask. Same values tables::count1_code_length computes.
struct Count1CostLUT {
    uint8_t cost[2][16];
    Count1CostLUT() {
        for (int idx = 0; idx < 16; idx++) {
            cost[0][idx] = static_cast<uint8_t>(tables::count1_code_length(
                32, (idx >> 3) & 1, (idx >> 2) & 1, (idx >> 1) & 1, idx & 1));
            cost[1][idx] = static_cast<uint8_t>(tables::count1_code_length(
                33, (idx >> 3) & 1, (idx >> 2) & 1, (idx >> 1) & 1, idx & 1));
        }
    }
};
const Count1CostLUT kCount1Cost;

inline int count1_mask(const int16_t* q) {
    return ((q[0] != 0) << 3) | ((q[1] != 0) << 2) |
           ((q[2] != 0) << 1) | (q[3] != 0);
}

// Exact bit count for one region under one candidate table (LUT-driven; the
// pair_cost ESC clamp keeps this exact for values >= 15 as well).
inline int count_region_bits(const int16_t* ix, int start, int end, int t) {
    const uint8_t* lut = kPairCost.cost[t];
    const bool esc = t >= 16;
    int total = 0;
    for (int i = start; i < end; i += 2) {
        int y = (i + 1 < end) ? ix[i + 1] : 0;
        total += pair_cost(lut, esc, ix[i], y);
    }
    return total;
}

// Candidate tables whose encodable range covers max_val. Tables in the same
// range group differ only by code-length distribution shape; for ESC ranges
// the two linbits families (16-23 vs 24-31) both get their smallest
// sufficient member. Every group contains the old choose_huff_table pick, so
// the first-minimum reduction can never do worse than the heuristic.
inline int table_candidates(int max_val, int cand[3]) {
    if (max_val <= 1)  { cand[0] = 1;  cand[1] = 2;  cand[2] = 3;  return 3; }
    if (max_val <= 2)  { cand[0] = 2;  cand[1] = 3;  return 2; }
    if (max_val <= 3)  { cand[0] = 5;  cand[1] = 6;  return 2; }
    if (max_val <= 5)  { cand[0] = 7;  cand[1] = 8;  cand[2] = 9;  return 3; }
    if (max_val <= 7)  { cand[0] = 10; cand[1] = 11; cand[2] = 12; return 3; }
    if (max_val <= 15) { cand[0] = 13; cand[1] = 15; return 2; }
    int bits_needed = 0;
    int tmp = max_val - 15;
    while (tmp > 0) { bits_needed++; tmp >>= 1; }
    int nc = 0;
    for (int t = 16; t < 24; ++t)
        if (tables::kLinbits[t - 16] >= bits_needed) { cand[nc++] = t; break; }
    for (int t = 24; t < 32; ++t)
        if (tables::kLinbits[t - 16] >= bits_needed) { cand[nc++] = t; break; }
    if (nc == 0) { cand[0] = 24; nc = 1; }  // unreachable (13 linbits cover int16)
    return nc;
}

} // namespace

// Find the best Huffman table for a region: identify the candidate tables
// whose encodable range covers the region's max value, count actual bits for
// each, and keep the cheapest (first minimum wins, so selection is
// deterministic). Tables in the same range group differ by code-length
// distribution shape (e.g. 7 vs 8 vs 9), which the old max-value-only lookup
// ignored; for ESC ranges the two linbits families (16-23 vs 24-31) are
// compared the same way. Fewer Huffman bits at the same coefficients lets the
// gain search land on a finer global_gain within the same budget.
int select_best_table(const int16_t* ix, int start, int end) {
    if (start >= end) return 0;
    int max_val = 0;
    for (int i = start; i < end; i++) {
        int v = std::abs(ix[i]);
        if (v > max_val) max_val = v;
    }
    if (max_val == 0) return 0;

    int cand[3];
    int nc = table_candidates(max_val, cand);
    int best = cand[0];
    int best_bits = count_region_bits(ix, start, end, cand[0]);
    for (int c = 1; c < nc; c++) {
        int bits = count_region_bits(ix, start, end, cand[c]);
        if (bits < best_bits) { best_bits = bits; best = cand[c]; }
    }
    return best;
}

// Fused region determination + table selection + bit count for the long-block
// quantization hot path. One pass per region accumulates every candidate
// table's total simultaneously; the running minimum across candidates is a
// lower bound of the final count, so the bit_limit early exit stays exact
// (any return over the limit is only compared against the limit — the caller
// never uses partial regions for encoding).
int huffman_select_and_count(const int16_t* ix, int sr_index, int rzero,
                             int count1_start, int bit_limit,
                             HuffRegions* out) {
    HuffRegions r{};
    r.big_values = count1_start / 2;
    r.count1 = (rzero - count1_start) / 4;
    if (r.count1 < 0) r.count1 = 0;
    r.rzero = rzero;

    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    int big_values_end = count1_start;
    const bool use_limit = bit_limit >= 0;
    int total = 0;

    if (big_values_end == 0) {
        r.region0_count = 0;
        r.region1_count = 0;
    } else {
        // Region boundary logic identical to
        // huffman_determine_regions_from_bounds.
        int max_band = 0;
        for (int b = 0; b < 22; b++) {
            if (sfb[b] >= big_values_end) { max_band = b; break; }
            if (b == 21) max_band = 22;
        }
        if (max_band <= 1) {
            r.region0_count = max_band;
            r.region1_count = 0;
        } else if (max_band <= 10) {
            r.region0_count = (max_band + 1) / 2;
            r.region1_count = max_band - r.region0_count - 1;
        } else {
            r.region0_count = 7;
            r.region1_count = max_band > 8 ? std::min(max_band - 8, 14) - 1 : 0;
        }
        if (r.region0_count > 15) r.region0_count = 15;
        if (r.region1_count > 7) r.region1_count = 7;

        int region0_end = sfb[r.region0_count + 1];
        if (region0_end > big_values_end) region0_end = big_values_end;
        int region1_end = sfb[r.region0_count + 1 + r.region1_count + 1];
        if (region1_end > big_values_end) region1_end = big_values_end;

        const int region_start[3] = { 0, region0_end, region1_end };
        const int region_end[3] = { region0_end, region1_end, big_values_end };
        for (int reg = 0; reg < 3; reg++) {
            int start = region_start[reg], end = region_end[reg];
            if (start >= end) { r.table_select[reg] = 0; continue; }
            int max_val = 0;
            for (int i = start; i < end; i++) {
                int v = std::abs(ix[i]);
                if (v > max_val) max_val = v;
            }
            if (max_val == 0) { r.table_select[reg] = 0; continue; }

            int cand[3];
            int nc = table_candidates(max_val, cand);
            const uint8_t* lut[3];
            bool esc[3];
            int tot[3] = { 0, 0, 0 };
            for (int c = 0; c < nc; c++) {
                lut[c] = kPairCost.cost[cand[c]];
                esc[c] = cand[c] >= 16;
            }
            for (int i = start; i < end; i += 2) {
                int x = ix[i];
                int y = (i + 1 < end) ? ix[i + 1] : 0;
                int m = tot[0] += pair_cost(lut[0], esc[0], x, y);
                for (int c = 1; c < nc; c++) {
                    tot[c] += pair_cost(lut[c], esc[c], x, y);
                    if (tot[c] < m) m = tot[c];
                }
                if (use_limit && total + m > bit_limit) {
                    if (out) *out = r;
                    return total + m;
                }
            }
            int best = 0;
            for (int c = 1; c < nc; c++)
                if (tot[c] < tot[best]) best = c;
            r.table_select[reg] = cand[best];
            total += tot[best];
        }
    }

    // count1: accumulate both tables' totals in one pass; min(A,B) is a lower
    // bound of the final choice, keeping the early exit exact.
    int bits_a = 0, bits_b = 0;
    for (int i = count1_start; i + 3 < rzero; i += 4) {
        int m = count1_mask(ix + i);
        bits_a += kCount1Cost.cost[0][m];
        bits_b += kCount1Cost.cost[1][m];
        if (use_limit && total + std::min(bits_a, bits_b) > bit_limit) {
            if (out) *out = r;
            return total + std::min(bits_a, bits_b);
        }
    }
    r.count1table = (bits_b < bits_a) ? 1 : 0;
    total += std::min(bits_a, bits_b);

    if (out) *out = r;
    return total;
}

static int find_count1_start(const int16_t* ix, int rzero) {
    int count1_start = (rzero + 3) & ~3;
    if (count1_start > rzero) count1_start = rzero;

    while (count1_start >= 4) {
        bool all_small = true;
        for (int i = count1_start - 4; i < count1_start; i++) {
            if (i < 576 && std::abs(ix[i]) > 1) {
                all_small = false;
                break;
            }
        }
        if (!all_small) break;
        count1_start -= 4;
    }

    if (count1_start & 1) count1_start++;
    return count1_start;
}

HuffRegions huffman_determine_regions_from_bounds(const int16_t* ix, int sr_index,
                                                  int rzero,
                                                  int count1_start) {
    HuffRegions r{};

    r.big_values = count1_start / 2;
    r.count1 = (rzero - count1_start) / 4;
    if (r.count1 < 0) r.count1 = 0;
    r.rzero = rzero;

    // Determine region boundaries using scalefactor bands
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    int big_values_end = count1_start;

    // Region 0/1/2 boundaries are at scalefactor band edges
    // region0_count and region1_count specify the number of sfb bands in each region
    // region0 spans sfb bands 0..region0_count
    // region1 spans region0_count+1..region0_count+1+region1_count
    // region2 spans the rest up to big_values

    if (big_values_end == 0) {
        r.region0_count = 0;
        r.region1_count = 0;
        r.table_select[0] = 0;
        r.table_select[1] = 0;
        r.table_select[2] = 0;
    } else {
        // Find which sfb band big_values_end falls in
        int max_band = 0;
        for (int b = 0; b < 22; b++) {
            if (sfb[b] >= big_values_end) { max_band = b; break; }
            if (b == 21) max_band = 22;
        }

        // Divide into up to 3 regions
        // A common split: region0=first 8 bands, region1=next 3 bands, region2=rest
        if (max_band <= 1) {
            r.region0_count = max_band;
            r.region1_count = 0;
        } else if (max_band <= 10) {
            // Put approximately half in region0, rest in region1
            r.region0_count = (max_band + 1) / 2;
            r.region1_count = max_band - r.region0_count - 1;
        } else {
            r.region0_count = 7;  // 8 bands in region 0
            r.region1_count = max_band > 8 ? std::min(max_band - 8, 14) - 1 : 0;
        }

        // Clamp to valid range
        if (r.region0_count > 15) r.region0_count = 15;
        if (r.region1_count > 7) r.region1_count = 7;

        // Compute actual region boundaries in spectrum indices
        int region0_end = sfb[r.region0_count + 1];
        if (region0_end > big_values_end) region0_end = big_values_end;

        int region1_end = sfb[r.region0_count + 1 + r.region1_count + 1];
        if (region1_end > big_values_end) region1_end = big_values_end;

        // Select tables for each region
        r.table_select[0] = select_best_table(ix, 0, region0_end);
        r.table_select[1] = select_best_table(ix, region0_end, region1_end);
        r.table_select[2] = select_best_table(ix, region1_end, big_values_end);
    }

    // Choose count1 table (A=32 vs B=33): try both, pick smaller
    int bits_a = 0, bits_b = 0;
    for (int i = count1_start; i + 3 < rzero; i += 4) {
        int m = count1_mask(ix + i);
        bits_a += kCount1Cost.cost[0][m];
        bits_b += kCount1Cost.cost[1][m];
    }
    r.count1table = (bits_b < bits_a) ? 1 : 0;

    return r;
}

// Optimal region0_count/region1_count for a finished long-block granule
// (PLAN 9.7). The gain search uses the fixed heuristic split (searching
// boundaries inside it is too hot); this runs ONCE on the final ix and
// re-derives the cheapest (r0c, r1c) split by exhaustive enumeration over
// per-table prefix bit costs at scalefactor-band granularity. The heuristic
// split is inside the search space, so the result never costs more bits.
// Returns the number of big-values bits saved (0 if the heuristic won) and
// updates r's counts and table selects in place.
int huffman_optimize_regions(const int16_t* ix, int sr_index, HuffRegions* r) {
    int bve = r->big_values * 2;
    if (bve <= 0) return 0;
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);

    // Number of sfb bands covering [0, bve); the last band may be partial.
    int nb = 22;
    for (int b = 1; b <= 22; b++) {
        if (sfb[b] >= bve) { nb = b; break; }
    }

    // Band maxima (band b's slice is [sfb[b], min(sfb[b+1], bve)) ).
    int band_max[23];
    for (int b = 0; b < nb; b++) {
        int start = sfb[b];
        int end = std::min(sfb[b + 1], bve);
        int bmax = 0;
        for (int i = start; i < end; i++) {
            int v = std::abs(ix[i]);
            if (v > bmax) bmax = v;
        }
        band_max[b] = bmax;
    }
    // Prefix costs are memoized PER TABLE on first use: the split search
    // only ever queries the few tables that are candidates for some range
    // maximum (plus the best-cost pruning skips many ranges outright), so
    // computing all 32 rows up front made the polish ~30% of the -q speed
    // profile for nothing.
    int pref[32][23];
    bool have[32] = {};
    auto ensure = [&](int t) {
        if (have[t]) return;
        have[t] = true;
        pref[t][0] = 0;
        for (int b = 0; b < nb; b++) {
            int start = sfb[b];
            int end = std::min(sfb[b + 1], bve);
            pref[t][b + 1] = pref[t][b] + count_region_bits(ix, start, end, t);
        }
    };
    // Suffix/range maxima via a running scan per query would be O(1) with a
    // prefix-max from the left and right; ranges are arbitrary, so build a
    // small sparse table alternative: nb <= 22, just recompute per range.
    auto range_max = [&](int a, int b) {
        int m = 0;
        for (int k = a; k < b; k++)
            if (band_max[k] > m) m = band_max[k];
        return m;
    };
    // Cheapest table for bands [a, b): candidates from the range max, cost
    // from the prefix sums. Returns cost and sets *tsel.
    auto region_cost = [&](int a, int b, int* tsel) {
        if (a >= b) { *tsel = 0; return 0; }
        int m = range_max(a, b);
        if (m == 0) { *tsel = 0; return 0; }
        int cand[3];
        int nc = table_candidates(m, cand);
        ensure(cand[0]);
        int best_t = cand[0];
        int best_c = pref[cand[0]][b] - pref[cand[0]][a];
        for (int c = 1; c < nc; c++) {
            ensure(cand[c]);
            int cost = pref[cand[c]][b] - pref[cand[c]][a];
            if (cost < best_c) { best_c = cost; best_t = cand[c]; }
        }
        *tsel = best_t;
        return best_c;
    };

    // Current cost under the existing split (same boundary clamping as
    // huffman_count_bits).
    int cur_r0end_band = std::min(r->region0_count + 1, nb);
    int cur_r1end_band = std::min(r->region0_count + 1 + r->region1_count + 1, nb);
    int dummy;
    int cur_cost = region_cost(0, cur_r0end_band, &dummy) +
                   region_cost(cur_r0end_band, cur_r1end_band, &dummy) +
                   region_cost(cur_r1end_band, nb, &dummy);

    // Exhaustive split search: region0 = bands [0, i), region1 = [i, j),
    // region2 = [j, nb). Side-info fields: r0c = i-1 (4 bits), r1c = j-i-1
    // (3 bits); sfb index r0c+r1c+2 = j must stay <= 22.
    int best_cost = cur_cost;
    int best_i = -1, best_j = -1;
    int best_t[3] = { 0, 0, 0 };
    // Suffix region [j, nb) costs are shared by every i — memoize them.
    int suf_cost[24], suf_t[24];
    bool suf_have[24] = {};
    auto suffix_cost = [&](int j, int* tsel) {
        if (!suf_have[j]) {
            suf_have[j] = true;
            suf_cost[j] = region_cost(j, nb, &suf_t[j]);
        }
        *tsel = suf_t[j];
        return suf_cost[j];
    };
    for (int i = 1; i <= std::min(16, nb); i++) {
        int t0;
        int c0 = region_cost(0, i, &t0);
        if (c0 >= best_cost) continue;
        int jmax = std::min({ i + 8, 22, nb });
        for (int j = i + 1; j <= jmax; j++) {
            // j < nb with region2 empty is the same split as j == nb;
            // dedupe happens naturally since costs are equal.
            int t1, t2;
            int c1 = region_cost(i, j, &t1);
            int c2 = suffix_cost(std::min(j, nb), &t2);
            int total = c0 + c1 + c2;
            if (total < best_cost) {
                best_cost = total;
                best_i = i;
                best_j = j;
                best_t[0] = t0; best_t[1] = t1; best_t[2] = t2;
            }
        }
        // Degenerate split: region0 covers everything (region1/2 empty).
        if (i >= nb && i <= 16) {
            if (c0 < best_cost) {
                best_cost = c0;
                best_i = i;
                best_j = std::min(i + 1, 22);
                best_t[0] = t0; best_t[1] = 0; best_t[2] = 0;
            }
            break;
        }
    }
    if (best_i < 0) return 0;

    r->region0_count = best_i - 1;
    r->region1_count = best_j - best_i - 1;
    r->table_select[0] = best_t[0];
    r->table_select[1] = best_t[1];
    r->table_select[2] = best_t[2];
    return cur_cost - best_cost;
}

HuffRegions huffman_determine_regions(const int16_t* ix, int sr_index) {
    int rzero = 576;
    while (rzero > 0 && ix[rzero - 1] == 0) rzero--;
    return huffman_determine_regions_from_bounds(ix, sr_index, rzero,
                                                 find_count1_start(ix, rzero));
}

HuffRegions huffman_determine_regions_short_from_bounds(const int16_t* ix,
                                                        int sr_index,
                                                        int rzero,
                                                        int count1_start,
                                                        int block_type) {
    HuffRegions r{};

    r.big_values = count1_start / 2;
    r.count1 = (rzero - count1_start) / 4;
    if (r.count1 < 0) r.count1 = 0;
    r.rzero = rzero;

    // With window_switching_flag=1 the decoder hardwires the region split:
    // region0_end = 36 for SHORT granules (block_type 2) at every rate, but
    // 54 for LSF START/STOP granules (block_type 1/3 at MPEG-2 rates) — see
    // e.g. ffmpeg's init_short_region(). region1_end = big_values*2. We set
    // region0_count so that sfb_long[region0_count+1] == boundary, which
    // makes huffman_count_bits/huffman_encode compute the same boundaries.
    // MPEG-1: sfb_long[8]=36 (r0c 7). MPEG-2: sfb_long_m2[6]=36 (r0c 5),
    // sfb_long_m2[8]=54 (r0c 7).
    const int boundary = (block_type != 2 && sr_index >= 3) ? 54 : 36;
    const int* sfb_long = tables::get_sfb_long_by_unified(sr_index);
    int r0c = 7;  // default for MPEG-1
    for (int b = 0; b < 21; b++) {
        if (sfb_long[b + 1] >= boundary) {
            r0c = b;
            break;
        }
    }
    r.region0_count = r0c;
    r.region1_count = 0;
    r.window_switching = 1;

    int region0_end = boundary;
    if (region0_end > count1_start) region0_end = count1_start;

    r.table_select[0] = select_best_table(ix, 0, region0_end);
    r.table_select[1] = select_best_table(ix, region0_end, count1_start);
    r.table_select[2] = 0;  // not used for short blocks

    // Choose count1 table
    int bits_a = 0, bits_b = 0;
    for (int i = count1_start; i + 3 < rzero; i += 4) {
        int m = count1_mask(ix + i);
        bits_a += kCount1Cost.cost[0][m];
        bits_b += kCount1Cost.cost[1][m];
    }
    r.count1table = (bits_b < bits_a) ? 1 : 0;

    return r;
}

HuffRegions huffman_determine_regions_short(const int16_t* ix, int sr_index,
                                            int block_type) {
    int rzero = 576;
    while (rzero > 0 && ix[rzero - 1] == 0) rzero--;
    return huffman_determine_regions_short_from_bounds(
        ix, sr_index, rzero, find_count1_start(ix, rzero), block_type);
}

int huffman_count_bits_limited(const int16_t* ix, const HuffRegions& regions,
                               int sr_index, int bit_limit) {
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    int big_values_end = regions.big_values * 2;
    int total = 0;
    const bool use_limit = bit_limit >= 0;

    // Big values region
    if (big_values_end > 0) {
        int region0_end = sfb[regions.region0_count + 1];
        if (region0_end > big_values_end) region0_end = big_values_end;

        int region1_end = regions.window_switching
            ? big_values_end
            : sfb[regions.region0_count + 1 + regions.region1_count + 1];
        if (region1_end > big_values_end) region1_end = big_values_end;

        const int region_start[3] = { 0, region0_end, region1_end };
        const int region_end[3] = { region0_end, region1_end, big_values_end };
        for (int r = 0; r < 3; r++) {
            int end = region_end[r];
            int t = regions.table_select[r];
            const uint8_t* lut = kPairCost.cost[t];
            const bool esc = t >= 16;
            for (int i = region_start[r]; i < end; i += 2) {
                int y = (i + 1 < end) ? ix[i + 1] : 0;
                total += pair_cost(lut, esc, ix[i], y);
                if (use_limit && total > bit_limit) return total;
            }
        }
    }

    // Count1 region
    int count1_start = big_values_end;
    int count1_end = count1_start + regions.count1 * 4;
    const uint8_t* c1 = kCount1Cost.cost[regions.count1table ? 1 : 0];
    for (int i = count1_start; i + 3 < count1_end && i + 3 < 576; i += 4) {
        total += c1[count1_mask(ix + i)];
        if (use_limit && total > bit_limit) return total;
    }

    return total;
}

int huffman_count_bits(const int16_t* ix, const HuffRegions& regions, int sr_index) {
    return huffman_count_bits_limited(ix, regions, sr_index, -1);
}

// Get the Huffman code for a pair of values, and write to bitstream
static void encode_pair(int table_id, int x, int y, BitstreamWriter& bs) {
    tables::HuffTable ht = tables::get_huff_table(table_id);
    if (table_id == 0) return;

    int ax = std::abs(x);
    int ay = std::abs(y);
    int linbits = ht.linbits;
    int ext_x = 0, ext_y = 0;
    int ext_x_bits = 0, ext_y_bits = 0;

    if (linbits > 0 && ax >= 15) {
        ext_x = ax - 15;
        ext_x_bits = linbits;
        ax = 15;
    }
    if (linbits > 0 && ay >= 15) {
        ext_y = ay - 15;
        ext_y_bits = linbits;
        ay = 15;
    }

    // Look up the code from the original ISO standard tables
    int idx = ax * ht.xlen + ay;
    int code_len = ht.hlen[idx];
    uint32_t code = 0;
    switch (table_id) {
    case 1:  code = tables::ht1_code[idx]; break;
    case 2:  code = tables::ht2_code[idx]; break;
    case 3:  code = tables::ht3_code[idx]; break;
    case 5:  code = tables::ht5_code[idx]; break;
    case 6:  code = tables::ht6_code[idx]; break;
    case 7:  code = tables::ht7_code[idx]; break;
    case 8:  code = tables::ht8_code[idx]; break;
    case 9:  code = tables::ht9_code[idx]; break;
    case 10: code = tables::ht10_code[idx]; break;
    case 11: code = tables::ht11_code[idx]; break;
    case 12: code = tables::ht12_code[idx]; break;
    case 13: code = tables::ht13_code[idx]; break;
    case 15: code = tables::ht15_code[idx]; break;
    case 16: case 17: case 18: case 19:
    case 20: case 21: case 22: case 23:
        code = tables::ht16_code[idx]; break;
    case 24: case 25: case 26: case 27:
    case 28: case 29: case 30: case 31:
        code = tables::ht24_code[idx]; break;
    default: return;
    }

    bs.write_bits(code, code_len);

    // Write linbits for x if needed
    if (ext_x_bits > 0) {
        bs.write_bits(ext_x, ext_x_bits);
    }
    // Sign bit for x
    if (x != 0) {
        bs.write_bits(x < 0 ? 1 : 0, 1);
    }
    // Write linbits for y if needed
    if (ext_y_bits > 0) {
        bs.write_bits(ext_y, ext_y_bits);
    }
    // Sign bit for y
    if (y != 0) {
        bs.write_bits(y < 0 ? 1 : 0, 1);
    }
}

static void encode_count1(int table_id, int v, int w, int x, int y,
                           BitstreamWriter& bs) {
    int idx = ((v != 0) ? 8 : 0) | ((w != 0) ? 4 : 0) |
              ((x != 0) ? 2 : 0) | ((y != 0) ? 1 : 0);

    const uint8_t* hlen = (table_id == 32) ? tables::ht32_len : tables::ht33_len;

    if (table_id == 33) {
        // Table B: 4-bit codes from ISO table (codes are 15-idx, not idx)
        bs.write_bits(tables::ht33_code[idx], 4);
    } else {
        // Table A (32): use original code table
        bs.write_bits(tables::ht32_code[idx], hlen[idx]);
    }

    // Sign bits for nonzero values
    if (v != 0) bs.write_bits(v < 0 ? 1 : 0, 1);
    if (w != 0) bs.write_bits(w < 0 ? 1 : 0, 1);
    if (x != 0) bs.write_bits(x < 0 ? 1 : 0, 1);
    if (y != 0) bs.write_bits(y < 0 ? 1 : 0, 1);
}

void huffman_encode(const int16_t* ix, const HuffRegions& regions,
                    int sr_index, BitstreamWriter& bs) {
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    int big_values_end = regions.big_values * 2;

    if (big_values_end > 0) {
        int region0_end = sfb[regions.region0_count + 1];
        if (region0_end > big_values_end) region0_end = big_values_end;

        int region1_end = regions.window_switching
            ? big_values_end
            : sfb[regions.region0_count + 1 + regions.region1_count + 1];
        if (region1_end > big_values_end) region1_end = big_values_end;

        // Region 0
        for (int i = 0; i < region0_end; i += 2) {
            int y = (i + 1 < region0_end) ? ix[i + 1] : 0;
            encode_pair(regions.table_select[0], ix[i], y, bs);
        }
        // Region 1
        for (int i = region0_end; i < region1_end; i += 2) {
            int y = (i + 1 < region1_end) ? ix[i + 1] : 0;
            encode_pair(regions.table_select[1], ix[i], y, bs);
        }
        // Region 2
        for (int i = region1_end; i < big_values_end; i += 2) {
            int y = (i + 1 < big_values_end) ? ix[i + 1] : 0;
            encode_pair(regions.table_select[2], ix[i], y, bs);
        }
    }

    // Count1 region
    int count1_start = big_values_end;
    int count1_end = count1_start + regions.count1 * 4;
    int ct = regions.count1table ? 33 : 32;
    for (int i = count1_start; i + 3 < count1_end && i + 3 < 576; i += 4) {
        encode_count1(ct, ix[i], ix[i+1], ix[i+2], ix[i+3], bs);
    }
}

} // namespace glint
