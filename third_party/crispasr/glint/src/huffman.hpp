// glint - Huffman encoding
// MIT License - Clean-room implementation

#ifndef GLINT_HUFFMAN_HPP
#define GLINT_HUFFMAN_HPP

#include <cstdint>
#include "bitstream.hpp"
#include "tables.hpp"

namespace glint {

// Region info for Huffman encoding
struct HuffRegions {
    int big_values;      // number of pairs in big-values region (0..288)
    int count1;          // number of quads in count1 region
    int rzero;           // index where trailing zeros begin
    int region0_count;   // scalefactor bands in region 0
    int region1_count;   // scalefactor bands in region 1
    int table_select[3]; // Huffman table for each big-values sub-region
    int count1table;     // 0 = table A (32), 1 = table B (33)
    // Window-switching layout (block_type != 0): the decoder hardwires
    // region0_end = 36 and region1_end = big_values*2 (there is no region2
    // and no region count fields in that side-info format). Encode/count
    // MUST use the same boundaries or every coefficient past the long-sfb
    // boundary is silently dropped while the decoder keeps reading.
    int window_switching;
};

// Count bits needed to encode the quantized spectrum (dry run)
// ix: quantized spectrum (576 values, unsigned magnitudes)
// signs: sign flags (true = negative) for each value
// regions: region info
// Returns total bits needed
int huffman_count_bits(const int16_t* ix, const HuffRegions& regions,
                       int sr_index);

// Count bits with early exit. Returns a value greater than bit_limit as soon as
// the partial count exceeds bit_limit; otherwise returns the exact count.
int huffman_count_bits_limited(const int16_t* ix, const HuffRegions& regions,
                               int sr_index, int bit_limit);

// Encode the quantized spectrum into the bitstream
// ix: quantized spectrum (576 values, signed)
// regions: region info
// bs: bitstream writer to output to
void huffman_encode(const int16_t* ix, const HuffRegions& regions,
                    int sr_index, BitstreamWriter& bs);

// Determine regions from quantized spectrum
// ix: quantized spectrum (576 values, unsigned magnitudes)
// sr_index: sample rate index (for scalefactor band boundaries)
// Returns filled HuffRegions struct
HuffRegions huffman_determine_regions(const int16_t* ix, int sr_index);

// Determine regions for a window-switching granule (block types 1/2/3).
// These have no region counts in the side info; the decoder hardwires
// region0_end (36 for short blocks; 36 MPEG-1 / 54 LSF for start/stop) and
// only two big-value sub-regions, so they need a different region layout
// than long blocks. The gain search and the final encode MUST use the same
// layout, otherwise the stored part2_3_length will not match the bits
// actually written. block_type distinguishes short (2) from start/stop.
HuffRegions huffman_determine_regions_short(const int16_t* ix, int sr_index,
                                            int block_type = 2);

// Determine regions when the caller already knows the nonzero/count1
// boundaries from quantization. count1_start is the first index of the count1
// region and rzero is the first trailing-zero index.
HuffRegions huffman_determine_regions_from_bounds(const int16_t* ix, int sr_index,
                                                  int rzero, int count1_start);
HuffRegions huffman_determine_regions_short_from_bounds(const int16_t* ix,
                                                        int sr_index,
                                                        int rzero,
                                                        int count1_start,
                                                        int block_type = 2);

// Optimal region0/region1 split for a FINISHED long-block granule: searches
// all valid (region0_count, region1_count) pairs over per-table prefix bit
// costs and rewrites r's counts/table selects. Returns big-values bits
// saved (>= 0; the existing split is in the search space). Run once per
// granule after the gain search — too hot to use inside it.
int huffman_optimize_regions(const int16_t* ix, int sr_index, HuffRegions* r);

// Select the best Huffman table for a region
// Returns table_id that minimizes bit count
int select_best_table(const int16_t* ix, int start, int end);

// Fused long-block region/table selection + bit count for the quantization
// hot path: determines regions, picks each sub-region's cheapest candidate
// table and the count1 table, and returns the total bit count — all in one
// pass over the spectrum. bit_limit >= 0 enables early exit (returns a value
// greater than bit_limit; region/table fields in *out are then partial and
// must not be used for encoding). Equivalent to
// huffman_determine_regions_from_bounds + huffman_count_bits_limited.
int huffman_select_and_count(const int16_t* ix, int sr_index, int rzero,
                             int count1_start, int bit_limit,
                             HuffRegions* out);

} // namespace glint

#endif // GLINT_HUFFMAN_HPP
