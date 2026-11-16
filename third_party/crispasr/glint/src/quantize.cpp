// glint - Quantization loop (double-precision input)
// MIT License - Clean-room implementation

#include "quantize.hpp"
#include "intmath.hpp"
#include "psycho.hpp"
#include "tables.hpp"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <vector>
#ifndef GLINT_NO_THREADS
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#endif
#include <functional>

namespace glint {

// ---------------------------------------------------------------------------
// Thread pool for the per-granule scale-factor search.
//
// The factor search in quantize_granule evaluates nf independent candidates
// (each a full quantize_base + granule_mse). quantize_base is pure, so the
// candidates can run on a pool and be reduced in deterministic index order.
// That makes the output BYTE-IDENTICAL regardless of thread count — the
// reduction picks the lowest-index minimum exactly as the old sequential
// loop did. Default thread count is 1 (the pool is never created; the loop
// runs inline), so the library stays deterministic and dependency-free unless
// a caller explicitly opts in via glint_config.threads / CLI -j.
// ---------------------------------------------------------------------------
// Spin-wait pool. The per-granule tasks are tiny (tens of microseconds), so a
// condition-variable wake/sleep per dispatch (thousands of dispatches/sec)
// costs more than the work it parallelizes. Workers instead spin on an atomic
// generation counter (yielding after a spin budget), giving sub-microsecond
// dispatch latency. This burns idle cores during the sequential per-frame work
// between dispatches, which is the right trade for a batch encoder.
#ifndef GLINT_NO_THREADS
class ThreadPool {
public:
    explicit ThreadPool(int n) {
        for (int i = 0; i < n; i++)
            workers_.emplace_back([this] { worker(); });
    }
    ~ThreadPool() {
        stop_.store(true, std::memory_order_release);
        for (auto& t : workers_) t.join();
    }
    int size() const { return static_cast<int>(workers_.size()); }

    // Run fn(i) for i in [0, n). Blocks until all complete. The calling
    // thread participates, so total parallelism is workers + 1.
    void parallel_for(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        fn_ = &fn;
        n_ = n;
        next_.store(0, std::memory_order_relaxed);
        active_.store(static_cast<int>(workers_.size()) + 1,
                      std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);  // publish the job
        run_indices(&fn, n);                            // calling thread works
        active_.fetch_sub(1, std::memory_order_acq_rel);
        // Wait for workers to drain.
        int spins = 0;
        while (active_.load(std::memory_order_acquire) != 0) {
            if (++spins > 2048) std::this_thread::yield();
        }
    }

private:
    void run_indices(const std::function<void(int)>* fn, int n) {
        for (;;) {
            int i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            (*fn)(i);
        }
    }
    void worker() {
        uint64_t seen = 0;
        int spins = 0;
        for (;;) {
            uint64_t g = gen_.load(std::memory_order_acquire);
            if (g == seen) {
                if (stop_.load(std::memory_order_acquire)) return;
                if (++spins > 2048) std::this_thread::yield();
                continue;
            }
            seen = g;
            spins = 0;
            const std::function<void(int)>* fn = fn_;
            int n = n_;
            run_indices(fn, n);
            active_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};
    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0;
    std::atomic<int> next_{0};
    std::atomic<int> active_{0};
    std::atomic<uint64_t> gen_{0};
};

static int g_quant_threads = 1;
static ThreadPool* g_pool = nullptr;

void quantize_set_threads(int n) {
    if (n < 1) n = 1;
    if (n == g_quant_threads) return;
    g_quant_threads = n;
    delete g_pool;
    g_pool = nullptr;
    if (n > 1) g_pool = new ThreadPool(n - 1);  // + the calling thread
}

// Run fn(i) for i in [0,n): on the pool when threads > 1, else inline. Inline
// runs in ascending index order — identical to the old sequential loop.
static void quant_parallel_for(int n, const std::function<void(int)>& fn) {
    if (g_pool && n > 1) {
        g_pool->parallel_for(n, fn);
    } else {
        for (int i = 0; i < n; i++) fn(i);
    }
}
#else
// Bare-metal builds (GLINT_NO_THREADS): no pool, always inline in order.
static void quant_parallel_for(int n, const std::function<void(int)>& fn) {
    for (int i = 0; i < n; i++) fn(i);
}
void quantize_set_threads(int) {}
#endif  // GLINT_NO_THREADS

static double gain_table[256];
static double sf_table[2][16];
#ifndef GLINT_SMALL_BUFFERS
// cbrt_lut[a] = a * cbrt(a), for a = |ix| in [0, 8191] (ix is clamped to 8191
// in the quantizer). Used in granule_mse to avoid a std::cbrt per nonzero
// coefficient. Holds exactly the same double the inline expression produced,
// so the MSE — and the scale-factor it selects — is bit-identical.
static double cbrt_lut[8192];
#else
// Small-footprint a^(4/3): mantissa cube-root table (65 floats) + the three
// 2^(k/3) constants. Relative error ~1e-5 — indistinguishable in the MSE
// ranking (measured metrics-identical on the battery) — for 64 KB saved.
static float cbrt_mant[65];
static inline double pow43_small(int a) {
    if (a <= 0) return 0.0;
    int n = 31 - glint::intmath::clz32(static_cast<unsigned>(a));
    double m = static_cast<double>(a) / static_cast<double>(1 << n);  // [1,2)
    double f = (m - 1.0) * 64.0;
    int i = static_cast<int>(f);
    double c = cbrt_mant[i] + (f - i) * (cbrt_mant[i + 1] - cbrt_mant[i]);
    static const double k3[3] = { 1.0, 1.2599210498948732, 1.5874010519681994 };
    double cr = std::ldexp(k3[n % 3], n / 3) * c;   // cbrt(a)
    return static_cast<double>(a) * cr;             // a^(4/3)
}
#endif
static bool tables_init = false;

static void init_quant_tables() {
    if (tables_init) return;
    for (int g = 0; g < 256; g++)
        gain_table[g] = std::pow(2.0, -3.0 * (g - 210.0) / 16.0);
#ifndef GLINT_SMALL_BUFFERS
    for (int a = 0; a < 8192; a++)
        cbrt_lut[a] = static_cast<double>(a) * std::cbrt(static_cast<double>(a));
#else
    for (int i = 0; i <= 64; i++)
        cbrt_mant[i] = static_cast<float>(std::cbrt(1.0 + i / 64.0));
#endif
    for (int sf = 0; sf < 16; sf++) {
        // Encoder uses positive exponent to compensate decoder's negative:
        // Decoder: 2^(-0.5*(1+sfs)*sf), Encoder: 2^(+0.75*0.5*(1+sfs)*sf)
        sf_table[0][sf] = std::pow(2.0, 0.75 * sf * 0.5);   // sfs=0
        sf_table[1][sf] = std::pow(2.0, 0.75 * sf * 1.0);   // sfs=1
    }
    tables_init = true;
}

// x^0.75 for the quantizer. Inputs are raw MDCT coefficients — almost all
// FRACTIONAL (~1e-6..0.3). The old integer-grid table interpolation
// (pow34_table[int(x)] + frac*...) degenerated to pow34(x)=x on (0,1) — a
// 33% error at x=0.2 — so ix was linear in x instead of x^0.75 and the
// decoder's exact ix^(4/3) expansion warped the entire spectrum by x^(4/3):
// content 20 dB below the granule peak decoded 6.7 dB too quiet. That single
// curve error capped whole-file SNR at ~15 dB independent of bitrate.
static double fast_pow34(double x) {
    if (x <= 0.0) return 0.0;
    return std::pow(x, 0.75);
}

// Window and short-sfb-band index of each reordered short-spectrum
// coefficient (block_type 2: wire order is [sfb band][window][line]);
// lazily built per sample rate. Band 12 is the untransmitted tail (the
// short analog of sfb21 — no scalefactor).
struct ShortWindowMap { uint8_t win[576]; uint8_t band[576]; };
static const ShortWindowMap* get_short_window_map(int sr_index) {
#ifdef GLINT_SMALL_BUFFERS
    // Single-slot cache: one sample rate per encoder in embedded use;
    // rebuilding on a rate switch is a one-time cost.
    static ShortWindowMap maps[1];
    static int slot_sr = -1;
    if (sr_index < 0 || sr_index > 5) sr_index = 0;
    bool need = (slot_sr != sr_index);
    int slot = 0;
    slot_sr = sr_index;
    if (need) {
#else
    static ShortWindowMap maps[6];
    static bool init[6] = {};
    if (sr_index < 0 || sr_index > 5) sr_index = 0;
    int slot = sr_index;
    if (!init[sr_index]) {
#endif
        const int* sfb = tables::get_sfb_short_by_unified(sr_index);
        int idx = 0;
        for (int b = 0; b < 13; b++) {
            int len = sfb[b + 1] - sfb[b];
            for (int w = 0; w < 3; w++)
                for (int j = 0; j < len && idx < 576; j++) {
                    maps[slot].win[idx] = static_cast<uint8_t>(w);
                    maps[slot].band[idx] = static_cast<uint8_t>(b);
                    idx++;
                }
        }
        while (idx < 576) {
            maps[slot].win[idx] = 2;
            maps[slot].band[idx] = 12;
            idx++;
        }
#ifndef GLINT_SMALL_BUFFERS
        init[sr_index] = true;
#endif
    }
    return &maps[slot];
}

// Cache for pre-computed per-coefficient values (constant across binary search)
struct QuantCache {
    double pow34_sf[576]; // pow34(|xr|) * sf_scale - precomputed
    int8_t sign[576];     // +1 or -1
};

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

static void fill_quant_cache(QuantCache& cache, const double* mdct_in,
                              const int scalefac[21], int scalefac_scale,
                              int preflag, int sr_index,
                              int block_type = 0,
                              const int* subblock_gain = nullptr,
                              const int (*scalefac_s)[3] = nullptr) {
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    // Per-window quantizer boost for short transforms: the decoder scales
    // window w by 2^(-2*subblock_gain[w]), so the encoder pre-boosts by
    // (2^(2*sbg))^(3/4) = 2^(1.5*sbg) — quiet windows quantize finer under
    // the same global_gain at no side-info cost.
    double sbg_boost[3] = { 1.0, 1.0, 1.0 };
    const uint8_t* wmap = nullptr;
    const uint8_t* bmap = nullptr;
    if (block_type == 2 && subblock_gain) {
        for (int w = 0; w < 3; w++)
            sbg_boost[w] = std::pow(2.0, 1.5 * subblock_gain[w]);
        const ShortWindowMap* m = get_short_window_map(sr_index);
        wmap = m->win;
        bmap = m->band;
    }
    int band = 0;
    for (int i = 0; i < 576; i++) {
        while (band < 21 && i >= sfb[band + 1]) band++;
        // Bins at/above sfb[21] are the standard's "sfb21" region: no
        // scalefactor is transmitted, so the decoder applies none — the
        // encoder must not either. (This used to read scalefac[21], out of
        // bounds — aliasing scalefac_compress in GranuleInfo and silently
        // boosting everything above ~9 kHz by up to 29x with no decoder-side
        // compensation.)
        int sf = 0;
        if (wmap) {
            // Short transform: per-(band,window) scalefactor; band 12 (the
            // short sfb21 analog) carries none.
            if (scalefac_s && bmap[i] < 12) sf = scalefac_s[bmap[i]][wmap[i]];
        } else if (band < 21 && i < sfb[21]) {
            sf = scalefac[band];
            if (preflag) sf += tables::preemphasis[band];
        }
        double sfs = (sf > 0 && sf < 16) ? sf_table[scalefac_scale][sf] : 1.0;
        if (wmap) sfs *= sbg_boost[wmap[i]];
        cache.pow34_sf[i] = fast_pow34(std::fabs(mdct_in[i])) * sfs;
        cache.sign[i] = (mdct_in[i] < 0.0) ? -1 : 1;
    }
}

// ws_block_type: 0 = long region layout; 1/2/3 = window-switching layout for
// that block type (the LSF start/stop region boundary differs from short's).
static int quantize_and_count(const double* mdct_in, int16_t* ix,
                               int global_gain, const int scalefac[21],
                               int scalefac_scale, int preflag,
                               int sr_index,
                               HuffRegions* out_regions = nullptr,
                               const QuantCache* cache = nullptr,
                               int ws_block_type = 0,
                               int bit_limit = -1) {
    init_quant_tables();
    double base_step = gain_table[global_gain];

    // Fused quantize + region detection in a single pass
    int rzero = 0;          // last nonzero index + 1

    if (cache) {
        for (int i = 0; i < 576; i++) {
            double qval_d = cache->pow34_sf[i] * base_step + 0.4054;
            int qval = (qval_d >= 8191.0) ? 8191 : static_cast<int>(qval_d);
            ix[i] = cache->sign[i] * qval;
        }
        // Scan backward for rzero (avoids branch in hot loop)
        rzero = 576;
        while (rzero > 0 && ix[rzero - 1] == 0) rzero--;
    } else {
        const int* sfb = tables::get_sfb_long_by_unified(sr_index);
        int band = 0;
        for (int i = 0; i < 576; i++) {
            while (band < 21 && i >= sfb[band + 1]) band++;
            // sfb21 region (>= sfb[21]): no scalefactor (see fill_quant_cache)
            int sf = 0;
            if (band < 21 && i < sfb[21]) {
                sf = scalefac[band];
                if (preflag) sf += tables::preemphasis[band];
            }
            double sf_scale = (sf > 0 && sf < 16) ? sf_table[scalefac_scale][sf] : 1.0;
            double abs_xr = std::fabs(mdct_in[i]);
            double pow34_val = fast_pow34(abs_xr);
            double qval_d = pow34_val * base_step * sf_scale + 0.4054;
            int qval = (qval_d >= 8191.0) ? 8191 : static_cast<int>(qval_d);
            ix[i] = (mdct_in[i] < 0.0) ? -qval : qval;
        }
        rzero = 576;
        while (rzero > 0 && ix[rzero - 1] == 0) rzero--;
    }

    // Window-switching granules use a different region layout; the gain
    // search must count bits with the same layout the encoder will use.
    int count1_start = find_count1_start(ix, rzero);
    if (!ws_block_type) {
        // Fused region/table selection + count: one pass picks each region's
        // cheapest candidate table and returns the total.
        return huffman_select_and_count(ix, sr_index, rzero, count1_start,
                                        bit_limit, out_regions);
    }
    HuffRegions regions = huffman_determine_regions_short_from_bounds(
        ix, sr_index, rzero, count1_start, ws_block_type);
    if (out_regions) *out_regions = regions;
    return bit_limit >= 0 ? huffman_count_bits_limited(ix, regions, sr_index,
                                                       bit_limit)
                          : huffman_count_bits(ix, regions, sr_index);
}

static int encode_scalefac_compress(int slen1, int slen2) {
    static const int table[16][2] = {
        {0,0},{0,1},{0,2},{0,3},{3,0},{1,1},{1,2},{1,3},
        {2,1},{2,2},{2,3},{3,1},{3,2},{3,3},{4,2},{4,3}
    };
    for (int i = 0; i < 16; i++)
        if (table[i][0] == slen1 && table[i][1] == slen2) return i;
    return 0;
}

// MPEG-2/2.5 scalefac_compress encoding (9 bits), ISO 13818-3 mapping for
// normal (non-intensity) long blocks, first range:
//   sfc < 400: slen[0]=(sfc>>4)/5, slen[1]=(sfc>>4)%5,
//              slen[2]=(sfc&15)>>2, slen[3]=sfc&3; band groups [6,5,5,5]
// so sfc = ((slen0*5 + slen1) << 4) | (slen2 << 2) | slen3, valid for
// slen0/1 <= 4 and slen2/3 <= 3 (max sfc 399 < 400 by construction).
// (The previous custom mapping — sfc = slen0*36 + slen1*6 + slen2 — matched
// the encoder's own emission code but NOT the standard, so every real
// decoder derived different slens and misparsed the whole granule: MPEG-2
// output measured -10 dB SNR. See PLAN.md item 7.)
static int encode_scalefac_compress_m2(int slen0, int slen1, int slen2, int slen3) {
    if (slen0 <= 4 && slen1 <= 4 && slen2 <= 3 && slen3 <= 3) {
        return ((slen0 * 5 + slen1) << 4) | (slen2 << 2) | slen3;
    }
    return -1;  // not representable in this range
}

// Compute headroom-based scalefactors using Vorbis/Opus/FLAC psychoacoustic
// masking model. Assigns SF inversely proportional to masking headroom:
// loud bands (high SMR) get sf=0 (noise masked by signal), bands near
// masking threshold get higher SF (need precision), bands below threshold
// get sf=0 (inaudible, save bits).
static bool compute_headroom_scalefactors(int scalefac[21],
                                           const double band_energy[21],
                                           int /* active_bands */) {
    // Compute per-band masking threshold via simplified spreading function
    double band_mask[21];
    for (int band = 0; band < 21; band++) {
        band_mask[band] = 0;
        for (int other = 0; other < 21; other++) {
            if (other == band) continue;  // exclude self-masking
            double spread;
            if (other > band)
                spread = std::pow(10.0, -0.8 * (other - band));  // 8 dB/band upward
            else
                spread = std::pow(10.0, -0.4 * (band - other));  // 4 dB/band downward
            band_mask[band] += band_energy[other] * spread;
        }
        // ATH floor, scaled to MDCT energy domain (MDCT divides by 288)
        double ath_linear = std::pow(10.0, (tables::ath_cb[std::min(band, 24)] - 96.0) / 10.0);
        ath_linear /= (288.0 * 288.0);
        // Use 0.1 (-10 dB) masking offset for inter-band masking
        band_mask[band] = std::max(band_mask[band] * 0.1, ath_linear);
    }

    // SMR (signal-to-mask ratio) per band, assign SF
    // Bands with energy well above their mask have lots of headroom and
    // can tolerate coarse quantization. Bands near the mask need precision.
    bool any = false;
    for (int band = 0; band < 21; band++) {
        double smr = band_energy[band] / band_mask[band];
        double headroom_db = 10.0 * std::log10(std::max(smr, 1e-10));
        // Map headroom to SF with wider range for better differentiation:
        // headroom <= 0 dB  -> sf = 0 (below mask, inaudible)
        // headroom  0-30 dB -> sf = 7..0 (linear mapping)
        // headroom > 30 dB  -> sf = 0 (far above mask, self-masked)
        int sf;
        if (headroom_db <= 0.0)
            sf = 0;
        else if (headroom_db > 30.0)
            sf = 0;
        else
            sf = static_cast<int>((30.0 - headroom_db) * 7.0 / 30.0);
        scalefac[band] = sf;
        if (sf > 0) any = true;
    }
    return any;
}

// Per-band source energy for the granule_mse envelope penalty. Depends only
// on the unscaled input, so it is identical for every candidate in the scale
// search — compute it once per granule and share it. Loop order matches the
// old in-loop accumulation exactly (ascending i within each band), so the
// sums are bit-identical.
static void compute_src_band(const double* mdct_in, int sr_index,
                             double src_band[21]) {
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    for (int b = 0; b < 21; b++) {
        src_band[b] = 0.0;
        int start = sfb[b];
        int end = (b < 20) ? sfb[b + 1] : 576;
        for (int i = start; i < end; i++)
            src_band[b] += mdct_in[i] * mdct_in[i];
    }
}

// Decoder-reconstruction MSE for a quantized granule vs the original MDCT.
// Used to pick the per-granule input scale ("factor") that best preserves the
// spectrum through the quantizer dead-zone.
//
// This used to add an envelope-retention penalty (log-loss on bands whose
// decoded energy fell below 90% of the source, HF-weighted) to fight the
// "dull" rolloff collapse. That collapse was a symptom of the pow34 curve
// bug; with the curve fixed the penalty measured as a no-op on music and
// -0.05..-0.13 dB SNR on speech at 256k/128k, so it was removed. Pure
// reconstruction MSE is the objective now.
static double granule_mse(const GranuleInfo& gi, const double* mdct_in,
                          int sr_index) {
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    double decoder_gain = std::pow(2.0, 0.25 * (gi.global_gain - 210));
    // Short transforms: the decoder scales window w by 2^(-2*subblock_gain[w])
    // and applies the per-(band,window) short scalefactor.
    double sbg_fac[3] = { 1.0, 1.0, 1.0 };
    double sfd_s[13][3];
    const uint8_t* wmap = nullptr;
    const uint8_t* bmap = nullptr;
    if (gi.block_type == 2) {
        for (int w = 0; w < 3; w++) {
            sbg_fac[w] = std::pow(2.0, -2.0 * gi.subblock_gain[w]);
            for (int b = 0; b < 12; b++)
                sfd_s[b][w] = std::pow(2.0, -0.5 * gi.scalefac_s[b][w] *
                                              (1 + gi.scalefac_scale));
            sfd_s[12][w] = 1.0;
        }
        const ShortWindowMap* m = get_short_window_map(sr_index);
        wmap = m->win;
        bmap = m->band;
    }
    double noise = 0.0;
    // Per-band iteration so the expensive std::pow(2.0, ...) sf_d term is
    // computed once per band instead of once per coefficient. "Band" 21 is
    // the sfb21 region [sfb[21], 576): no scalefactor is transmitted there,
    // so the decoder reconstructs with sf_d = 1 — mirror that exactly.
    for (int b = 0; b < 22; b++) {
        int start = sfb[b];
        int end = (b < 21) ? sfb[b + 1] : 576;
        if (start >= end) continue;
        double sf_d = 1.0;
        if (b < 21) {
            int sf = gi.scalefac[b];
            if (gi.preflag) sf += tables::preemphasis[b];
            sf_d = std::pow(2.0, -0.5 * sf * (1 + gi.scalefac_scale));
        }
        for (int i = start; i < end; i++) {
            double xr_hat = 0.0;
            if (gi.ix[i] != 0) {
                // a^(4/3) == a * a^(1/3); precomputed in cbrt_lut[a]. ix is
                // always in [0,8191], but guard the index defensively.
                int a = std::abs(static_cast<int>(gi.ix[i]));
#ifndef GLINT_SMALL_BUFFERS
                double a43 = (a < 8192) ? cbrt_lut[a]
                                        : static_cast<double>(a) * std::cbrt(static_cast<double>(a));
#else
                double a43 = pow43_small(a);
#endif
                double g = wmap ? decoder_gain * sbg_fac[wmap[i]] *
                                      sfd_s[bmap[i]][wmap[i]]
                                : decoder_gain * sf_d;
                xr_hat = std::copysign(a43 * g, mdct_in[i]);
            }
            double err = mdct_in[i] - xr_hat;
            noise += err * err;
        }
    }
    return noise;
}

// Base quantizer: gain search to the bit budget + energy-based scalefactors.
// Operates on already-scaled input; the public quantize_granule wraps this in
// the per-granule scale search. gain_floor > 0 keeps the search from going
// finer than that gain (VBR constant-quality target).
static GranuleInfo quantize_base(const double* mdct_in, int available_bits,
                                 int sr_index, int block_type,
                                 int gain_floor = 0);
static void gain_search_with_scalefacs(GranuleInfo& info, const double* mdct_in,
                                       int available_bits, int sr_index,
                                       int block_type, int gain_floor = 0);
static bool encode_scalefac_fields(GranuleInfo& info, int sr_index);

// Per-band decoder-reconstruction noise vs the original MDCT (same
// reconstruction expression as granule_mse).
static void compute_band_noise(const GranuleInfo& gi, const double* mdct_in,
                               int sr_index, double noise_band[21]) {
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    double decoder_gain = std::pow(2.0, 0.25 * (gi.global_gain - 210));
    double sbg_fac[3] = { 1.0, 1.0, 1.0 };
    double sfd_s[13][3];
    const uint8_t* wmap = nullptr;
    const uint8_t* bmap = nullptr;
    if (gi.block_type == 2) {
        for (int w = 0; w < 3; w++) {
            sbg_fac[w] = std::pow(2.0, -2.0 * gi.subblock_gain[w]);
            for (int b = 0; b < 12; b++)
                sfd_s[b][w] = std::pow(2.0, -0.5 * gi.scalefac_s[b][w] *
                                              (1 + gi.scalefac_scale));
            sfd_s[12][w] = 1.0;
        }
        const ShortWindowMap* m = get_short_window_map(sr_index);
        wmap = m->win;
        bmap = m->band;
    }
    for (int b = 0; b < 21; b++) noise_band[b] = 0.0;
    // Same band/sf_d handling as granule_mse (sfb21 region reconstructs with
    // sf_d = 1); its noise is accounted to band 20 for scoring purposes.
    for (int b = 0; b < 22; b++) {
        int start = sfb[b];
        int end = (b < 21) ? sfb[b + 1] : 576;
        if (start >= end) continue;
        double sf_d = 1.0;
        if (b < 21) {
            int sf = gi.scalefac[b];
            if (gi.preflag) sf += tables::preemphasis[b];
            sf_d = std::pow(2.0, -0.5 * sf * (1 + gi.scalefac_scale));
        }
        double acc = 0.0;
        for (int i = start; i < end; i++) {
            double xr_hat = 0.0;
            if (gi.ix[i] != 0) {
                int a = std::abs(static_cast<int>(gi.ix[i]));
#ifndef GLINT_SMALL_BUFFERS
                double a43 = (a < 8192) ? cbrt_lut[a]
                                        : static_cast<double>(a) * std::cbrt(static_cast<double>(a));
#else
                double a43 = pow43_small(a);
#endif
                double g = wmap ? decoder_gain * sbg_fac[wmap[i]] *
                                      sfd_s[bmap[i]][wmap[i]]
                                : decoder_gain * sf_d;
                xr_hat = std::copysign(a43 * g, mdct_in[i]);
            }
            double err = mdct_in[i] - xr_hat;
            acc += err * err;
        }
        noise_band[(b < 21) ? b : 20] += acc;
    }
}

// Per-band masking thresholds for the outer loop, from the per-band source
// energies. Schroeder spreading evaluated in true Bark distance between
// scalefactor-band centers (precomputed per sample rate), a -14 dB
// tonality-agnostic offset, and an ATH floor — deliberately the same model
// as the NMR metric in tests/measure_audio.py, so the allocation optimizes
// what we measure. (The PsychoModel in psycho.cpp remains unusable as an
// allocation target: 1.5-3 dB/Bark slopes with self-masking excluded let a
// loud low band "mask" the whole spectrum.)
struct BandMaskModel {
    double spread[21][21];  // energy spreading gains between band centers
    double ath[21];         // ATH floor in MDCT-energy domain
};

static const BandMaskModel* get_mask_model(int sr_index) {
#ifdef GLINT_SMALL_BUFFERS
    static BandMaskModel models[1];
    static int slot_sr = -1;
    if (sr_index < 0 || sr_index > 5) sr_index = 0;
    bool need = (slot_sr != sr_index);
    int slot = 0;
    slot_sr = sr_index;
    if (need) {
        BandMaskModel& m = models[slot];
#else
    static BandMaskModel models[6];
    static bool init[6] = {};
    if (sr_index < 0 || sr_index > 5) sr_index = 0;
    int slot = sr_index;
    if (!init[sr_index]) {
        BandMaskModel& m = models[sr_index];
#endif
        static const int rates[6] = { 44100, 48000, 32000, 22050, 24000, 16000 };
        double srate = rates[sr_index];
        const int* sfb = tables::get_sfb_long_by_unified(sr_index);
        double z[21];
        for (int b = 0; b < 21; b++) {
            double fc = 0.5 * (sfb[b] + sfb[b + 1]) * (srate / 2.0) / 576.0;
            z[b] = 13.0 * std::atan(0.00076 * fc) +
                   3.5 * std::atan((fc / 7500.0) * (fc / 7500.0));
            // ATH floor in the MDCT-energy domain (MDCT divides by 288)
            double ath_db = tables::ath_cb[std::min(b, 24)];
            m.ath[b] = std::pow(10.0, (ath_db - 96.0) / 10.0) / (288.0 * 288.0);
        }
        for (int b = 0; b < 21; b++)
            for (int j = 0; j < 21; j++) {
                double dz = z[b] - z[j];
                double s_db = 15.81 + 7.5 * (dz + 0.474) -
                              17.5 * std::sqrt(1.0 + (dz + 0.474) * (dz + 0.474));
                m.spread[b][j] = std::pow(10.0, s_db / 10.0);
            }
#ifndef GLINT_SMALL_BUFFERS
        init[sr_index] = true;
#endif
    }
    return &models[slot];
}

static void compute_band_tonality(const double* mdct_in, int sr_index,
                                  double alpha[21]) {
    const int* sfb = tables::get_sfb_long_by_unified(sr_index);
    for (int b = 0; b < 21; b++) {
        int start = sfb[b], end = sfb[b + 1];
        int n = end - start;
        double am = 0.0, lg = 0.0;
        for (int i = start; i < end; i++) {
            double e = mdct_in[i] * mdct_in[i] + 1e-30;
            am += e;
            lg += std::log(e);
        }
        am /= n;
        double gm = std::exp(lg / n);
        double sfm_db = 10.0 * std::log10(gm / am + 1e-30);
        double a = sfm_db / -20.0;
        alpha[b] = a > 1.0 ? 1.0 : a;
    }
}

static void compute_band_masks(const double* src_band, int sr_index,
                               double mask_band[21],
                               const double* alpha = nullptr) {
    const BandMaskModel* m = get_mask_model(sr_index);
    static const double kOffset = std::pow(10.0, -14.0 / 10.0);
    double off[21];
    if (alpha)
        for (int j = 0; j < 21; j++)
            off[j] = std::pow(10.0, -(6.0 + 12.0 * alpha[j]) / 10.0);
    for (int b = 0; b < 21; b++) {
        double acc = 0.0;
        if (alpha) {
            for (int j = 0; j < 21; j++)
                acc += src_band[j] * m->spread[b][j] * off[j];
            mask_band[b] = std::max(acc, m->ath[b]);
        } else {
            for (int j = 0; j < 21; j++)
                acc += src_band[j] * m->spread[b][j];
            mask_band[b] = std::max(acc * kOffset, m->ath[b]);
        }
    }
}

// Short-block mask model: Schroeder-Bark spreading across the 13 short
// sfbs of one window (temporal masking between the 3 windows is ignored).
struct ShortMaskModel { double spread[13][13]; double ath[13]; };
static const ShortMaskModel* get_short_mask_model(int sr_index) {
#ifdef GLINT_SMALL_BUFFERS
    static ShortMaskModel models[1];
    static int slot_sr = -1;
    if (sr_index < 0 || sr_index > 5) sr_index = 0;
    bool need = (slot_sr != sr_index);
    int slot = 0;
    slot_sr = sr_index;
    if (need) {
        ShortMaskModel& m = models[slot];
#else
    static ShortMaskModel models[6];
    static bool init[6] = {};
    if (sr_index < 0 || sr_index > 5) sr_index = 0;
    int slot = sr_index;
    if (!init[sr_index]) {
        ShortMaskModel& m = models[sr_index];
#endif
        static const int rates[6] = { 44100, 48000, 32000, 22050, 24000, 16000 };
        double srate = rates[sr_index];
        const int* sfb = tables::get_sfb_short_by_unified(sr_index);
        double z[13];
        for (int b = 0; b < 13; b++) {
            double fc = 0.5 * (sfb[b] + sfb[b + 1]) * (srate / 2.0) / 192.0;
            z[b] = 13.0 * std::atan(0.00076 * fc) +
                   3.5 * std::atan((fc / 7500.0) * (fc / 7500.0));
            double ath_db = tables::ath_cb[std::min(b * 2, 24)];
            m.ath[b] = std::pow(10.0, (ath_db - 96.0) / 10.0) / (288.0 * 288.0);
        }
        for (int b = 0; b < 13; b++)
            for (int j = 0; j < 13; j++) {
                double dz = z[b] - z[j];
                double s_db = 15.81 + 7.5 * (dz + 0.474) -
                              17.5 * std::sqrt(1.0 + (dz + 0.474) * (dz + 0.474));
                m.spread[b][j] = std::pow(10.0, s_db / 10.0);
            }
#ifndef GLINT_SMALL_BUFFERS
        init[sr_index] = true;
#endif
    }
    return &models[slot];
}

// Per-(band,window) noise of a short granule's reconstruction vs the
// (reordered) source, and the matching masks from per-cell source energy.
static void compute_cell_noise(const GranuleInfo& gi, const double* mdct_in,
                               int sr_index, double noise[13][3]) {
    const ShortWindowMap* map = get_short_window_map(sr_index);
    double decoder_gain = std::pow(2.0, 0.25 * (gi.global_gain - 210));
    double sbg_fac[3];
    double sfd_s[13][3];
    for (int w = 0; w < 3; w++) {
        sbg_fac[w] = std::pow(2.0, -2.0 * gi.subblock_gain[w]);
        for (int b = 0; b < 12; b++)
            sfd_s[b][w] = std::pow(2.0, -0.5 * gi.scalefac_s[b][w] *
                                          (1 + gi.scalefac_scale));
        sfd_s[12][w] = 1.0;
    }
    for (int b = 0; b < 13; b++)
        for (int w = 0; w < 3; w++) noise[b][w] = 0.0;
    for (int i = 0; i < 576; i++) {
        int b = map->band[i], w = map->win[i];
        double xr_hat = 0.0;
        if (gi.ix[i] != 0) {
            int a = std::abs(static_cast<int>(gi.ix[i]));
#ifndef GLINT_SMALL_BUFFERS
            double a43 = (a < 8192) ? cbrt_lut[a]
                                    : static_cast<double>(a) * std::cbrt(static_cast<double>(a));
#else
            double a43 = pow43_small(a);
#endif
            xr_hat = std::copysign(a43 * decoder_gain * sbg_fac[w] * sfd_s[b][w],
                                   mdct_in[i]);
        }
        double err = mdct_in[i] - xr_hat;
        noise[b][w] += err * err;
    }
}

static void compute_cell_masks(const double* mdct_in, int sr_index,
                               double mask[13][3]) {
    const ShortWindowMap* map = get_short_window_map(sr_index);
    const ShortMaskModel* m = get_short_mask_model(sr_index);
    static const double kOffset = std::pow(10.0, -14.0 / 10.0);
    double e[13][3] = {};
    for (int i = 0; i < 576; i++)
        e[map->band[i]][map->win[i]] += mdct_in[i] * mdct_in[i];
    for (int w = 0; w < 3; w++)
        for (int b = 0; b < 13; b++) {
            double acc = 0.0;
            for (int j = 0; j < 13; j++)
                acc += e[j][w] * m->spread[b][j];
            mask[b][w] = std::max(acc * kOffset, m->ath[b]);
        }
}

static double shape_target();
static double noise_guard();

// Short-granule shaping: same idea as nmr_outer_loop, over the 13x3
// (band,window) cells; band 12 counts toward the objective but has no
// scalefactor to amplify.
static GranuleInfo nmr_outer_loop_short(const GranuleInfo& start, double factor,
                                        const double* mdct_in,
                                        int available_bits, int sr_index,
                                        int max_iters, int gain_floor,
                                        bool vbr_shaping) {
    double mask[13][3];
    compute_cell_masks(mdct_in, sr_index, mask);

    double noise0[13][3];
    compute_cell_noise(start, mdct_in, sr_index, noise0);
    const double target = shape_target();
    double j_best = 0.0, total0 = 0.0, worst0 = 0.0;
    for (int b = 0; b < 13; b++)
        for (int w = 0; w < 3; w++) {
            if (mask[b][w] > 0.0) {
                double r = noise0[b][w] / mask[b][w];
                j_best += r;
                if (r > worst0) worst0 = r;
            }
            total0 += noise0[b][w];
        }
    if (worst0 <= target) return start;

    double scaled[576];
    for (int i = 0; i < 576; i++) scaled[i] = mdct_in[i] * factor;

    // Shaping budget = what the unshaped winner actually spent, NOT the
    // frame budget: re-searching under the full budget converts the rate
    // controller's underspend (its reservoir-fill signal) into scalefactor
    // spending, pinning the anchor floor coarse (see nmr_outer_loop for
    // the measured slack sweep).
    int shape_bits = vbr_shaping
        ? start.part2_3_length + start.part2_3_length / 4
        : start.part2_3_length +
              (available_bits - start.part2_3_length) / 2;
    if (shape_bits > available_bits) shape_bits = available_bits;

    // Same 1.25x guard as the long loop: a looser short-loop guard (2.0)
    // measured WORSE castanets-128k NMR (16.2 vs 15.0) — the short mask
    // model (no temporal masking) misleads when allowed to trade harder.
    const double kNoiseGuard = noise_guard();

    GranuleInfo best = start;
    GranuleInfo cur = start;
    double cur_noise[13][3];
    std::memcpy(cur_noise, noise0, sizeof(cur_noise));

    int stall = 0;
    for (int iter = 0; iter < max_iters; iter++) {
        double worst = 0.0;
        for (int b = 0; b < 13; b++)
            for (int w = 0; w < 3; w++) {
                if (mask[b][w] <= 0.0) continue;
                double r = cur_noise[b][w] / mask[b][w];
                if (r > worst) worst = r;
            }
        if (worst <= target) break;
        double thresh = std::max(target, worst * 0.25);

        GranuleInfo cand = cur;
        bool amplified = false;
        for (int b = 0; b < 12; b++)
            for (int w = 0; w < 3; w++) {
                if (mask[b][w] <= 0.0) continue;
                int cap = (b < 6) ? 15 : 7;
                if (cur_noise[b][w] / mask[b][w] >= thresh &&
                    cand.scalefac_s[b][w] < cap) {
                    cand.scalefac_s[b][w]++;
                    amplified = true;
                }
            }
        if (!amplified) break;
        if (!encode_scalefac_fields(cand, sr_index)) break;
        gain_search_with_scalefacs(cand, scaled, shape_bits, sr_index,
                                   /*block_type=*/2, gain_floor);

        double cand_noise[13][3];
        compute_cell_noise(cand, mdct_in, sr_index, cand_noise);
        double j_cand = 0.0, cand_total = 0.0;
        for (int b = 0; b < 13; b++)
            for (int w = 0; w < 3; w++) {
                if (mask[b][w] > 0.0) j_cand += cand_noise[b][w] / mask[b][w];
                cand_total += cand_noise[b][w];
            }

        if (j_cand < j_best && cand_total <= total0 * kNoiseGuard) {
            j_best = j_cand;
            best = cand;
            stall = 0;
        } else if (++stall >= 3) {
            break;  // see nmr_outer_loop
        }
        cur = cand;
        std::memcpy(cur_noise, cand_noise, sizeof(cur_noise));
    }
    return best;
}

// Shaping target for the NMR outer loops: bands are amplified until their
// noise-to-mask ratio falls to this value (1.0 = stop at the mask; smaller
// = keep pushing below it, trading raw MSE inside the noise guard). 0.125
// (-9 dB) measured the sweet spot on the 256k battery: speech mean NMR
// -12.1 -> -13.7 at -0.16 dB SNR, quartet -9.2 -> -10.1, electronic -12.3
// -> -12.6 (SNR holds). 0.0625 buys speech -14.2 but seg-SNR starts to pay
// (-0.25); music saturates below 0.25 regardless (the sf caps and the
// noise guard bind, not the target).
static double shape_target() { return 0.125; }

// Total-noise guard for both outer loops: shaping may redistribute noise,
// never grow it past this factor. Loosening to 1.6 buys speech only +0.3 dB
// NMR for -0.6 dB SNR (bad trade); a short-loop-only 2.0 measured WORSE
// castanets NMR (the short mask model has no temporal masking and misleads
// when allowed to trade harder).
static double noise_guard() { return 1.25; }

// Fold the preemphasis pattern into preflag when every HF band's transmitted
// scalefactor covers it: reconstruction is unchanged (the decoder adds pretab
// when preflag is set), but the transmitted values shrink, so slen2 — and
// part2_length — drop and the gain search can afford a finer global_gain.
// MPEG-1 long blocks only (LSF side info has no preflag bit).
static void try_fold_preflag(GranuleInfo& gi) {
    if (gi.preflag || gi.block_type == 2) return;
    bool any = false;
    for (int b = 11; b < 21; b++) {
        if (gi.scalefac[b] < tables::preemphasis[b]) return;
        if (tables::preemphasis[b] > 0 && gi.scalefac[b] > 0) any = true;
    }
    if (!any) return;
    gi.preflag = 1;
    for (int b = 11; b < 21; b++) gi.scalefac[b] -= tables::preemphasis[b];
}

// Outer-loop objective, aligned with the measurement metric: total linear
// noise-to-mask ratio across bands (the metric averages N/M over
// band-frames, so minimizing the per-granule sum minimizes the metric).
static double nmr_objective(const double noise_band[21],
                            const double mask_band[21]) {
    double j = 0.0;
    for (int b = 0; b < 21; b++)
        if (mask_band[b] > 0.0) j += noise_band[b] / mask_band[b];
    return j;
}

// NMR-driven outer loop (LAME-style noise shaping): amplify the scalefactors
// of the worst noise-to-mask bands, then re-run the gain search so
// global_gain coarsens to pay for it — noise moves from audible bands into
// masked ones instead of being added on top. Keeps the best iterate by the
// nmr_objective; returns `start` untouched if nothing improves. MPEG-1 long
// blocks only (the m2 field encoding cannot represent all bands, and short
// granules have a different band structure).
static GranuleInfo nmr_outer_loop(const GranuleInfo& start, double factor,
                                  const double* mdct_in, int available_bits,
                                  int sr_index, int max_iters,
                                  const double* src_band, int gain_floor,
                                  bool vbr_shaping, bool tonal_masks) {
    double mask_band[21];
    if (tonal_masks) {
        double alpha[21];
        compute_band_tonality(mdct_in, sr_index, alpha);
        compute_band_masks(src_band, sr_index, mask_band, alpha);
    } else {
        compute_band_masks(src_band, sr_index, mask_band);
    }

    double noise0[21];
    compute_band_noise(start, mdct_in, sr_index, noise0);
    double j_best = nmr_objective(noise0, mask_band);
    // Already at/below the shaping target everywhere: nothing to shape.
    const double target = shape_target();
    double worst0 = 0.0;
    for (int b = 0; b < 21; b++) {
        if (mask_band[b] <= 0.0) continue;
        double r = noise0[b] / mask_band[b];
        if (r > worst0) worst0 = r;
    }
    if (worst0 <= target) return start;

    double total0 = 0.0;
    for (int b = 0; b < 21; b++) total0 += noise0[b];

    double scaled[576];
    for (int i = 0; i < 576; i++) scaled[i] = mdct_in[i] * factor;

    // Shaping is mostly a REDISTRIBUTION: it gets what the unshaped winner
    // spent plus HALF the leftover budget. Handing it the full budget let
    // it consume the easy-granule underspend that the CBR rate controller
    // reads as reservoir fill — the anchor then ratcheted to its coarse
    // clamp and stereo -q best (both channels shaped, so no unshaped side
    // channel kept the reservoir fed) lost 5 dB SNR / 3 dB NMR. Measured
    // (speech 256k best): slack/2 gives stereo 35.9/-10.8 and joint
    // 37.7/-13.5; slack*3/4 re-triggers the ratchet (stereo 33.3/-8.3);
    // zero slack starves joint shaping (-11.9).
    int shape_bits = vbr_shaping
        ? start.part2_3_length + start.part2_3_length / 4
        : start.part2_3_length +
              (available_bits - start.part2_3_length) / 2;
    if (shape_bits > available_bits) shape_bits = available_bits;

    // Loose total-noise guard: shaping means redistributing noise, and a
    // genuinely perceptual trade may raise raw MSE — but runaway trades that
    // tank SNR for the last sliver of NMR are rejected.
    const double kNoiseGuard = noise_guard();

    // MPEG-1 slen field limits: bands 0-10 max 15, bands 11-20 max 7.
    static const int kSfMax[21] = { 15,15,15,15,15,15,15,15,15,15,15,
                                    7,7,7,7,7,7,7,7,7,7 };

    GranuleInfo best = start;
    GranuleInfo cur = start;
    double cur_noise[21];
    std::memcpy(cur_noise, noise0, sizeof(cur_noise));

    int stall = 0;
    for (int iter = 0; iter < max_iters; iter++) {
        // Amplify the bands driving the objective: everything within 6 dB of
        // the worst band's ratio, provided it is actually over the mask.
        double worst = 0.0;
        for (int b = 0; b < 21; b++) {
            if (mask_band[b] <= 0.0) continue;
            double r = cur_noise[b] / mask_band[b];
            if (r > worst) worst = r;
        }
        if (worst <= target) break;
        double thresh = std::max(target, worst * 0.25);

        GranuleInfo cand = cur;
        bool amplified = false;
        bool capped_need = false;
        for (int b = 0; b < 21; b++) {
            if (mask_band[b] <= 0.0) continue;
            double r = cur_noise[b] / mask_band[b];
            if (r >= thresh) {
                if (cand.scalefac[b] < kSfMax[b]) {
                    cand.scalefac[b]++;
                    amplified = true;
                } else if (r > 4.0) {
                    // only escalate for bands clearly (>6 dB) over the mask
                    capped_need = true;
                }
            }
        }
        // Escalate to scalefac_scale=1 when shaping demand hits the sf caps:
        // the step doubles to 3 dB, extending the range (values halve,
        // rounded up, so the achieved shaping never decreases).
        if (!amplified && capped_need && cand.scalefac_scale == 0) {
            cand.scalefac_scale = 1;
            for (int b = 0; b < 21; b++)
                cand.scalefac[b] = (cand.scalefac[b] + 1) / 2;
            amplified = true;
        }
        if (!amplified) break;
        // MPEG-1 only: in the LSF sfc<400 range a decoder derives preflag=0
        // (preflag=1 only comes from the sfc>=500 range, never emitted), so
        // folding would silently desync encoder and decoder.
        if (sr_index < 3) try_fold_preflag(cand);
        if (!encode_scalefac_fields(cand, sr_index)) break;
        gain_search_with_scalefacs(cand, scaled, shape_bits, sr_index,
                                   /*block_type=*/0, gain_floor);

        double cand_noise[21];
        compute_band_noise(cand, mdct_in, sr_index, cand_noise);
        double j_cand = nmr_objective(cand_noise, mask_band);
        double cand_total = 0.0;
        for (int b = 0; b < 21; b++) cand_total += cand_noise[b];

        if (j_cand < j_best && cand_total <= total0 * kNoiseGuard) {
            j_best = j_cand;
            best = cand;
            stall = 0;
        } else if (++stall >= 3) {
            // Three consecutive iterations without an accepted improvement:
            // the walk has left the productive region. Measured metrics-
            // neutral on the whole battery; saves the tail of the
            // (8|20)-iteration budget.
            break;
        }
        cur = cand;
        std::memcpy(cur_noise, cand_noise, sizeof(cur_noise));
    }
    return best;
}

// Per-granule scale search (normal/best tiers).
//
// The quantizer is is = int(|xr|^0.75 * step + 0.4054); coefficients whose
// scaled magnitude is below the ~0.6 dead-zone round to zero. With the pow34
// curve fixed, reconstruction is level-exact at f=1.0; small boosts >1 only
// serve to rescue dead-zone coefficients (at a matching level cost the MSE
// weighs), so the grids sit tightly above 1.0. The old grids (1.3..4.2, and
// the fixed 288/194 "gain correction" before them) existed to compensate the
// broken pow34 curve — post-fix they are pure level errors. -q speed skips
// the search entirely: f=1.0 wins almost always, and the tier's contract is
// throughput.
// Final-granule region polish (PLAN 9.7): rerun the region0/region1 split
// search on the finished coefficients — the gain search counts with the
// fixed heuristic split (searching inside it is too hot), so a few bits per
// granule are usually left on the table. Saved bits shrink part2_3_length,
// feeding the CBR reservoir / smaller VBR frames. Long blocks only
// (window-switching granules have no region fields).
static void polish_regions(GranuleInfo& info, int sr_index) {
    if (info.block_type != 0) return;
    int saved = huffman_optimize_regions(info.ix, sr_index, &info.regions);
    info.part2_3_length -= saved;
}

#ifdef GLINT_MP3_INT

namespace {

// Integer quantize + Huffman count for the -q speed no-FPU path. p34log[i]
// holds log2(|x|^0.75) in Q16 (INT32_MIN for zero lines); the gain step is
// exact in Q16: log2(2^((210-g)*3/16)) = 12288*(210-g). Same 0.4054 dead
// zone and 8191 cap as the double quantizer.
int quantize_and_count_int(const int32_t* p34log, const int8_t* sign,
                           int16_t* ix, int global_gain, int sr_index,
                           HuffRegions* out_regions, int bit_limit) {
    const int32_t lstep = 12288 * (210 - global_gain);
    for (int i = 0; i < 576; i++) {
        int q = 0;
        if (p34log[i] != INT32_MIN) {
            q = intmath::exp2_quant(static_cast<int64_t>(p34log[i]) + lstep,
                                    26573, 8191);  // 0.4054 * 65536
        }
        ix[i] = static_cast<int16_t>(sign[i] < 0 ? -q : q);
    }
    int rzero = 576;
    while (rzero > 0 && ix[rzero - 1] == 0) rzero--;
    int count1_start = find_count1_start(ix, rzero);
    return huffman_select_and_count(ix, sr_index, rzero, count1_start,
                                    bit_limit, out_regions);
}

}  // namespace

GranuleInfo quantize_granule_int_speed(const int32_t* mdct_q24,
                                       int available_bits, int sr_index,
                                       int gain_floor) {
    init_quant_tables();
    GranuleInfo info{};
    std::memset(&info, 0, sizeof(info));
    info.block_type = 0;
    info.scalefac_compress = 0;
    info.part2_length = 0;

    int32_t p34log[576];
    int8_t sign[576];
    int32_t peaklog = INT32_MIN;
    for (int i = 0; i < 576; i++) {
        int32_t v = mdct_q24[i];
        sign[i] = v < 0 ? -1 : 1;
        uint32_t a = static_cast<uint32_t>(v < 0 ? -v : v);
        if (a == 0) {
            p34log[i] = INT32_MIN;
        } else {
            // log2(|x_true|) = log2(a) - 24;  p34log = 0.75 * that (Q16)
            int64_t l = static_cast<int64_t>(intmath::ilog2_q16(a)) - (24 << 16);
            int32_t p = static_cast<int32_t>((l * 3) >> 2);
            p34log[i] = p;
            if (p > peaklog) peaklog = p;
        }
    }

    // Gain bounds, mirroring gain_search_with_scalefacs: min gain keeps the
    // peak below the 8191 clamp, max gain is where the peak quantizes to ~1.
    // Integer arithmetic on the Q16 logs; 12288 = the exact Q16 gain step.
    int min_gain = 0, max_gain = 255;
    if (peaklog != INT32_MIN) {
        const int32_t l8190 = intmath::ilog2_q16(8190);
        // peaklog + 12288*(210-g) < l8190  ->  g > 210 - (l8190-peaklog)/12288
        int gmin = 210 - static_cast<int>((static_cast<int64_t>(l8190) - peaklog) / 12288);
        if (gmin > min_gain) min_gain = gmin;
        if (min_gain < 0) min_gain = 0;
        if (min_gain > 255) min_gain = 255;
        // peak ~1:  peaklog + 12288*(210-g) < log2(0.6) (~ -47720 in Q16)
        int gmax = 210 - static_cast<int>((-47720LL - peaklog) / 12288) + 2;
        if (gmax < max_gain && gmax > min_gain) max_gain = gmax;
    }
    if (gain_floor > min_gain) min_gain = gain_floor;
    if (max_gain < min_gain) max_gain = min_gain;

    int target_bits = available_bits;
    if (target_bits < 0) target_bits = 0;

    int lo = min_gain, hi = max_gain, best_gain = max_gain;
    int best_bits = -1;
    HuffRegions best_regions{};
    int16_t best_ix[576];
    for (int iter = 0; iter < 8 && lo <= hi; iter++) {
        int gain = (lo + hi) / 2;
        HuffRegions regs;
        int bits = quantize_and_count_int(p34log, sign, info.ix, gain,
                                          sr_index, &regs, target_bits);
        if (bits <= target_bits) {
            hi = gain - 1;
            best_gain = gain;
            best_bits = bits;
            best_regions = regs;
            std::memcpy(best_ix, info.ix, sizeof(best_ix));
        } else {
            lo = gain + 1;
        }
    }
    info.global_gain = best_gain;

    int huff_bits;
    if (best_bits >= 0) {
        std::memcpy(info.ix, best_ix, sizeof(best_ix));
        info.regions = best_regions;
        huff_bits = best_bits;
    } else {
        huff_bits = quantize_and_count_int(p34log, sign, info.ix, best_gain,
                                           sr_index, &info.regions, -1);
    }
    info.part2_3_length = info.part2_length + huff_bits;

    // Budget guarantee (12-bit part2_3_length field), same as the double path.
    {
        int limit = available_bits;
        if (limit > 4095) limit = 4095;
        while (info.part2_3_length > limit && info.global_gain < 255) {
            info.global_gain++;
            huff_bits = quantize_and_count_int(p34log, sign, info.ix,
                                               info.global_gain, sr_index,
                                               &info.regions, -1);
            info.part2_3_length = info.part2_length + huff_bits;
        }
    }
    info.rc_gain = info.global_gain;
    return info;
}

#endif  // GLINT_MP3_INT

GranuleInfo quantize_granule(const double* mdct_in, int available_bits,
                              int sr_index, int quality_mode, int block_type,
                              int gain_floor, bool allow_psy,
                              bool vbr_shaping, bool tonal_masks) {
    init_quant_tables();

    if (quality_mode <= 0) {
        // No region polish at -q speed: the tier's contract is throughput,
        // and the polish measured ~30% of the whole speed-tier profile for
        // a bit-savings win the tier's users didn't ask for.
        return quantize_base(mdct_in, available_bits, sr_index, block_type,
                             gain_floor);
    }

    static const double kNormal[] = { 1.0, 1.04, 1.09, 1.15, 1.22, 1.30 };
    static const double kBest[]   = {
        1.0, 1.02, 1.05, 1.08, 1.11, 1.15, 1.19, 1.24, 1.30, 1.38, 1.48, 1.60 };
    const double* factors; int nf;
    if (quality_mode >= 2) { factors = kBest;   nf = 12; }
    else                   { factors = kNormal; nf = 6; }

    // Per-band source energy is factor-independent; compute it once (used by
    // the NMR outer loop when enabled).
    double src_band[21];
    compute_src_band(mdct_in, sr_index, src_band);

    // Evaluate all nf candidate factors (each independent), then reduce in
    // ascending index order — same selection as the old sequential loop
    // (first/lowest-index minimum wins), so the result is byte-identical for
    // any thread count.
    std::vector<GranuleInfo> results(nf);
    std::vector<double> mses(nf);
    quant_parallel_for(nf, [&](int fi) {
        double f = factors[fi];
        double scaled[576];
        for (int i = 0; i < 576; i++) scaled[i] = mdct_in[i] * f;
        results[fi] = quantize_base(scaled, available_bits, sr_index,
                                    block_type, gain_floor);
        mses[fi] = granule_mse(results[fi], mdct_in, sr_index);
    });

    GranuleInfo best_result = results[0];
    double best_mse = mses[0];
    double best_factor = factors[0];
    for (int fi = 1; fi < nf; fi++) {
        if (mses[fi] < best_mse) {
            best_mse = mses[fi];
            best_result = results[fi];
            best_factor = factors[fi];
        }
    }

    if (quality_mode >= 2) {
        // Fine refinement around best_factor: candidates for step -2,-1,1,2
        // (skipping any f < 0.98), kept in that exact order for the reduction.
        double cand[4];
        int nc = 0;
        for (int step = -2; step <= 2; step++) {
            if (step == 0) continue;
            double f = best_factor + step * 0.015;
            if (f < 0.98) continue;
            cand[nc++] = f;
        }
        std::vector<GranuleInfo> rres(nc);
        std::vector<double> rmse(nc);
        quant_parallel_for(nc, [&](int k) {
            double scaled[576];
            for (int i = 0; i < 576; i++) scaled[i] = mdct_in[i] * cand[k];
            rres[k] = quantize_base(scaled, available_bits, sr_index,
                                    block_type, gain_floor);
            rmse[k] = granule_mse(rres[k], mdct_in, sr_index);
        });
        for (int k = 0; k < nc; k++) {
            if (rmse[k] < best_mse) {
                best_mse = rmse[k];
                best_result = rres[k];
                best_factor = cand[k];
            }
        }
    }

    // NMR-driven noise shaping on the winning candidate (MPEG-1 long blocks;
    // see nmr_outer_loop). The masks mirror the measurement metric's model
    // (Schroeder spreading, -14 dB offset, ATH floor); granules already
    // comfortably below the mask skip the loop.
    if (allow_psy && quality_mode >= 1) {
        int max_iters = (quality_mode >= 2) ? 20 : 8;
        if (block_type == 2) {
            // Short shaping works for LSF too (the sf caps match MPEG-1's
            // and encode_scalefac_fields handles the LSF field encoding).
            best_result = nmr_outer_loop_short(best_result, best_factor,
                                               mdct_in, available_bits,
                                               sr_index, max_iters, gain_floor,
                                               vbr_shaping);
        } else if (block_type == 0) {
            // LSF long blocks too: the m2 slen group caps (15/15/7/7 over
            // bands [0-5][6-10][11-15][16-20]) are within kSfMax, and the
            // preflag fold is gated off above.
            best_result = nmr_outer_loop(best_result, best_factor, mdct_in,
                                         available_bits, sr_index, max_iters,
                                         src_band, gain_floor, vbr_shaping,
                                         tonal_masks);
        }
        // Types 1/3 (start/stop) are deliberately NOT shaped: their masks
        // would come from attack-dominated spectra and measured +3 dB NMR
        // on the castanet clip when tried.
    }
    polish_regions(best_result, sr_index);
    return best_result;
}

// Encode the granule's scalefactors into scalefac_compress/part2_length
// (slen selection). Short granules (block_type 2) use scalefac_s: 12 short
// sfbs x 3 windows, slen1 for bands 0-5 and slen2 for 6-11, 18 values per
// slen group. Returns false when the scalefactors cannot be represented.
static bool encode_scalefac_fields(GranuleInfo& info, int sr_index) {
    static const int kSlenTableS[16][2] = {
        {0,0},{0,1},{0,2},{0,3},{3,0},{1,1},{1,2},{1,3},
        {2,1},{2,2},{2,3},{3,1},{3,2},{3,3},{4,2},{4,3}
    };
    if (info.block_type == 2) {
        if (sr_index >= 3) {
            // LSF short (non-mixed, non-intensity), ISO 13818-3 sfc < 400
            // range: the 12x3 scalefactors go out in [band][window] wire
            // order as FOUR groups of 3 bands (9 values each), one slen per
            // group; slen limits 4/4/3/3 like the long-block groups.
            int max_sf[4] = {};
            for (int b = 0; b < 12; b++)
                for (int w = 0; w < 3; w++)
                    max_sf[b / 3] = std::max(max_sf[b / 3],
                                             info.scalefac_s[b][w]);
            if (max_sf[0] > 15 || max_sf[1] > 15 || max_sf[2] > 7 ||
                max_sf[3] > 7)
                return false;
            int sl[4];
            for (int g = 0; g < 4; g++) {
                sl[g] = 0;
                while ((1 << sl[g]) <= max_sf[g]) sl[g]++;
            }
            int sfc = encode_scalefac_compress_m2(sl[0], sl[1], sl[2], sl[3]);
            if (sfc < 0) return false;
            info.scalefac_compress = sfc;
            info.part2_length = (sl[0] + sl[1] + sl[2] + sl[3]) * 9;
            return true;
        }
        int max_sf1 = 0, max_sf2 = 0;
        for (int b = 0; b < 6; b++)
            for (int w = 0; w < 3; w++)
                max_sf1 = std::max(max_sf1, info.scalefac_s[b][w]);
        for (int b = 6; b < 12; b++)
            for (int w = 0; w < 3; w++)
                max_sf2 = std::max(max_sf2, info.scalefac_s[b][w]);
        if (max_sf1 > 15 || max_sf2 > 7) return false;
        int slen1 = 0; while ((1 << slen1) <= max_sf1) slen1++;
        int slen2 = 0; while ((1 << slen2) <= max_sf2) slen2++;
        int best_sfc = -1, best_cost = 1 << 30;
        for (int i = 0; i < 16; i++) {
            if (kSlenTableS[i][0] >= slen1 && kSlenTableS[i][1] >= slen2) {
                int cost = (kSlenTableS[i][0] + kSlenTableS[i][1]) * 18;
                if (cost < best_cost) { best_cost = cost; best_sfc = i; }
            }
        }
        if (best_sfc < 0) return false;
        info.scalefac_compress = best_sfc;
        info.part2_length = best_cost;
        return true;
    }
    bool is_mpeg2 = (sr_index >= 3);
    if (is_mpeg2) {
        // Band groups [6,5,5,5]; slen limits 4/4/3/3 in the sfc<400 range,
        // i.e. sf <= 15 in groups 0-1 and sf <= 7 in groups 2-3.
        int max_sf[4] = {};
        for (int b = 0; b < 6; b++) max_sf[0] = std::max(max_sf[0], info.scalefac[b]);
        for (int b = 6; b < 11; b++) max_sf[1] = std::max(max_sf[1], info.scalefac[b]);
        for (int b = 11; b < 16; b++) max_sf[2] = std::max(max_sf[2], info.scalefac[b]);
        for (int b = 16; b < 21; b++) max_sf[3] = std::max(max_sf[3], info.scalefac[b]);
        if (max_sf[0] > 15 || max_sf[1] > 15 || max_sf[2] > 7 || max_sf[3] > 7)
            return false;
        int sl[4];
        for (int g = 0; g < 4; g++) {
            sl[g] = 0;
            while ((1 << sl[g]) <= max_sf[g]) sl[g]++;
        }
        int sfc = encode_scalefac_compress_m2(sl[0], sl[1], sl[2], sl[3]);
        if (sfc < 0) return false;
        info.scalefac_compress = sfc;
        info.part2_length = sl[0]*6 + sl[1]*5 + sl[2]*5 + sl[3]*5;
    } else {
        int max_sf1 = 0, max_sf2 = 0;
        for (int b = 0; b < 11; b++) max_sf1 = std::max(max_sf1, info.scalefac[b]);
        for (int b = 11; b < 21; b++) max_sf2 = std::max(max_sf2, info.scalefac[b]);
        if (max_sf1 > 15 || max_sf2 > 7) return false;
        int slen1 = 0; while ((1 << slen1) <= max_sf1) slen1++;
        int slen2 = 0; while ((1 << slen2) <= max_sf2) slen2++;
        // The 4-bit scalefac_compress table does NOT contain every
        // (slen1, slen2) pair — e.g. (1,0), (2,0), (4,0), (4,1) don't exist.
        // Pick the cheapest entry that covers both required widths (writing
        // scalefactors with a wider slen is always valid). Returning a wrong
        // sfc here desyncs the whole granule: the decoder reads slen bits
        // per the table while part2_length assumed something else.
        static const int kSlenTable[16][2] = {
            {0,0},{0,1},{0,2},{0,3},{3,0},{1,1},{1,2},{1,3},
            {2,1},{2,2},{2,3},{3,1},{3,2},{3,3},{4,2},{4,3}
        };
        int best_sfc = -1, best_cost = 1 << 30;
        for (int i = 0; i < 16; i++) {
            if (kSlenTable[i][0] >= slen1 && kSlenTable[i][1] >= slen2) {
                int cost = kSlenTable[i][0] * 11 + kSlenTable[i][1] * 10;
                if (cost < best_cost) { best_cost = cost; best_sfc = i; }
            }
        }
        if (best_sfc < 0) return false;
        info.scalefac_compress = best_sfc;
        info.part2_length = best_cost;
    }
    return true;
}

// Gain search with the scalefactors already fixed in info: fills the quant
// cache once, binary-searches global_gain to the bit budget (reusing the last
// accepted iteration's state — see quantize_base history), and runs the
// budget-guarantee loop. Sets ix/global_gain/regions/part2_3_length.
static void gain_search_with_scalefacs(GranuleInfo& info, const double* mdct_in,
                                       int available_bits, int sr_index,
                                       int block_type, int gain_floor) {
    const int ws_layout = (block_type != 0) ? block_type : 0;
    int target_bits = available_bits - info.part2_length;
    if (target_bits < 0) target_bits = 0;

    // The cache is filled once and reused for the final quantize and the
    // budget-guarantee loop below (scalefactors no longer change here).
    QuantCache cache;
    fill_quant_cache(cache, mdct_in, info.scalefac, info.scalefac_scale,
                     info.preflag, sr_index, block_type, info.subblock_gain,
                     info.scalefac_s);

    // Gain bounds are computed from the cache's actual quantizer input —
    // pow34(|xr|) INCLUDING the per-band scalefactor boost. Computing them
    // from pow34(max|xr|) alone (the historical bug) let any band with a
    // nonzero scalefactor (sfs up to 6.17x at sf=7) sail past the 8191 hard
    // clamp at the "minimum" gain: the loudest band then decoded up to
    // ~-21 dB quiet, capping whole-file SNR at ~15 dB regardless of bitrate,
    // piling the error into the loudest (0-1 kHz) bands, and forcing the
    // scale search to prefer f>1 purely to coarsen the gain away from
    // clipping.
    double peak34 = 0.0;
    for (int i = 0; i < 576; i++)
        if (cache.pow34_sf[i] > peak34) peak34 = cache.pow34_sf[i];

    // Minimum gain to prevent clipping (ix > 8191):
    // peak34 * 2^(-3*(g-210)/16) + 0.4054 < 8191
    // → g > 210 - (16/3) * log2(8190 / peak34)
    int min_gain = 0;
    int max_gain = 255;
    if (peak34 > 0.0) {
        double g_min = 210.0 - (16.0 / 3.0) * std::log2(8190.0 / peak34);
        min_gain = static_cast<int>(std::ceil(g_min));
        if (min_gain < 0) min_gain = 0;

        // Tighten the search upper bound: gain where the peak quantizes to ~1.
        // peak34 * gain_table[g] + 0.4054 < 1.0
        // → g > 210 - (16/3)*log2(0.6/peak34)
        double g_max = 210.0 - (16.0 / 3.0) * std::log2(0.6 / peak34);
        int est = static_cast<int>(g_max) + 2;
        if (est < max_gain && est > min_gain) max_gain = est;
    }
    // VBR constant-quality floor: never quantize finer than the target gain.
    if (gain_floor > min_gain) min_gain = gain_floor;
    if (max_gain < min_gain) max_gain = min_gain;

    // When a search iteration accepts a gain (bits <= target_bits), its bit
    // count ran to completion (the early exit only fires when over the limit),
    // so that iteration's ix[]/regions/count are exactly what a fresh
    // quantize at that gain would produce. Snapshot the last accepted state
    // and skip the redundant final quantize+count.
    int lo = min_gain, hi = max_gain, best_gain = max_gain;
    int best_bits = -1;
    HuffRegions best_regions{};
    int16_t best_ix[576];
    for (int iter = 0; iter < 8 && lo <= hi; iter++) {
        int gain = (lo + hi) / 2;
        HuffRegions regs;
        int bits = quantize_and_count(mdct_in, info.ix, gain, info.scalefac,
                                       info.scalefac_scale, info.preflag, sr_index,
                                       &regs, &cache, ws_layout, target_bits);
        if (bits <= target_bits) {
            hi = gain - 1; best_gain = gain;
            best_bits = bits;
            best_regions = regs;
            std::memcpy(best_ix, info.ix, sizeof(best_ix));
        }
        else { lo = gain + 1; }
    }
    info.global_gain = best_gain;

    int huff_bits;
    if (best_bits >= 0) {
        std::memcpy(info.ix, best_ix, sizeof(best_ix));
        info.regions = best_regions;
        huff_bits = best_bits;
    } else {
        // No gain in the search range fit the budget (or the range was empty):
        // quantize at the fallback max_gain the search never evaluated.
        huff_bits = quantize_and_count(mdct_in, info.ix, best_gain,
                                       info.scalefac, info.scalefac_scale,
                                       info.preflag, sr_index, &info.regions,
                                       &cache, ws_layout);
    }
    info.part2_3_length = info.part2_length + huff_bits;

    // Budget guarantee: part2_3_length must fit both the per-granule bit
    // budget and the 12-bit side-info field. The gain search above counts
    // bits with the correct (long/short) region layout, so this rarely
    // triggers, but it is a hard safety net: coarsen the quantization
    // (raise global_gain) until the granule fits. Without it, an overflowing
    // part2_3_length wraps the 12-bit field and desyncs the decoder.
    {
        int limit = available_bits;
        if (limit > 4095) limit = 4095;  // 12-bit part2_3_length field
        while (info.part2_3_length > limit && info.global_gain < 255) {
            info.global_gain++;
            huff_bits = quantize_and_count(mdct_in, info.ix, info.global_gain,
                                           info.scalefac, info.scalefac_scale,
                                           info.preflag, sr_index, &info.regions,
                                           &cache, ws_layout);
            info.part2_3_length = info.part2_length + huff_bits;
        }
    }
}

static GranuleInfo quantize_base(const double* mdct_in, int available_bits,
                                 int sr_index, int block_type,
                                 int gain_floor) {
    GranuleInfo info{};
    std::memset(&info, 0, sizeof(info));
    init_quant_tables();

    info.scalefac_compress = 0;
    info.part2_length = 0;
    info.block_type = block_type;

    // Short transforms: choose per-window subblock_gain from window peaks —
    // the global gain must fit the attack window, so quieter windows get
    // gain relief (each unit is 12 dB) and quantize finer for free.
    if (block_type == 2) {
        const uint8_t* wmap = get_short_window_map(sr_index)->win;
        double e[3] = { 0.0, 0.0, 0.0 };
        for (int i = 0; i < 576; i++)
            e[wmap[i]] += mdct_in[i] * mdct_in[i];
        double emax = std::max(e[0], std::max(e[1], e[2]));
        for (int w = 0; w < 3; w++) {
            int sbg = 0;
            if (emax > 0.0 && e[w] > 0.0 && e[w] < emax) {
                // each sbg unit = 12 dB of gain relief = a factor 16 in energy
                sbg = static_cast<int>(std::log2(emax / e[w]) / 4.0);
                if (sbg > 7) sbg = 7;
                if (sbg < 0) sbg = 0;
            }
            info.subblock_gain[w] = sbg;
        }
    }

    // Assign scalefactors FIRST, then run a single gain search with them.
    {
        const int* sfb = tables::get_sfb_long_by_unified(sr_index);
        double band_energy[21], max_energy = 0.0;
        for (int band = 0; band < 21; band++) {
            double e = 0.0;
            for (int i = sfb[band]; i < sfb[band+1] && i < 576; i++)
                e += mdct_in[i] * mdct_in[i];
            band_energy[band] = e;
            if (e > max_energy) max_energy = e;
        }

        // Count active bands
        int active_bands = 0;
        if (max_energy > 0.0) {
            for (int band = 0; band < 21; band++)
                if (band_energy[band] / max_energy > 0.01) active_bands++;
        }

        static constexpr bool kEnergySeed = false;  // experiment
        if (kEnergySeed && max_energy > 0.0 && active_bands >= 3 && block_type == 0) {
            // Energy-based scalefactor assignment: give a little extra precision
            // to higher-energy bands. Modes 1/2 get their per-granule fidelity
            // from the scale search that wraps this base quantizer.
            bool any = false;
            for (int band = 0; band < 21; band++) {
                double ratio = band_energy[band] / max_energy;
                if (ratio > 0.01) {
                    int sf = static_cast<int>(ratio * 4.0 + 0.5);
                    if (sf > 7) sf = 7;
                    if (sf > 0) { info.scalefac[band] = sf; any = true; }
                }
            }
            if (any && !encode_scalefac_fields(info, sr_index)) {
                // Not representable in the scalefac_compress range: drop the
                // scalefactors rather than desync fields from values.
                std::memset(info.scalefac, 0, sizeof(info.scalefac));
            }
        }
    }

    gain_search_with_scalefacs(info, mdct_in, available_bits, sr_index,
                               block_type, gain_floor);
    // Pre-shaping operating point for the rate controller. The psy loops
    // copy GranuleInfo wholesale and only re-search global_gain, so this
    // survives shaping untouched.
    info.rc_gain = info.global_gain;
    return info;
}

// VBR quality 0-9 maps to target global_gain values (recalibrated after the
// pow34-curve fix: each gain step is ~1.1 dB of quantization noise; the old
// table's 194-230 range dead-zoned entire granules to silence).
// 0 (best): gain ~134 (fine quantization, many bits)
// 9 (worst): gain ~178 (coarse quantization, few bits)
static const int vbr_target_gain[10] = {
    134, 140, 144, 148, 152, 156, 161, 166, 172, 178
};

GranuleInfo quantize_granule_vbr(const double* mdct_in, int available_bits,
                                  int sr_index, int quality_mode,
                                  int vbr_quality, int block_type,
                                  bool allow_psy) {
    // Same path as CBR — including, since PLAN 10.4, the factor search and
    // the NMR outer loops at -q normal/best — with the gain search floored
    // at the constant-quality target gain: "the finest gain >= target that
    // fits". The shaping budget (unshaped spend + slack/2) uses the same
    // discipline as CBR; VBR's slack is the max-rate frame, so frames may
    // grow — judged bytes-honestly in the PLAN entry. The old VBR-only
    // preprocessing (psycho zeroing, headroom scalefactors) stays gone.
    init_quant_tables();
    if (vbr_quality < 0) vbr_quality = 0;
    if (vbr_quality > 9) vbr_quality = 9;
    return quantize_granule(mdct_in, available_bits, sr_index, quality_mode,
                            block_type, vbr_target_gain[vbr_quality],
                            allow_psy, /*vbr_shaping=*/true,
                            /*tonal_masks=*/false);
}

} // namespace glint
