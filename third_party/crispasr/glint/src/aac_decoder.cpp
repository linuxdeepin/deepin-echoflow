// glint - AAC-LC decoder (PLAN § D2)
// MIT License - Clean-room from ISO/IEC 14496-3 (AAC-LC, ADTS).

#include "aac_decoder.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aac_tables.hpp"

namespace glint {
namespace aac {

using namespace aac_tables;

namespace {

// ---------------------------------------------------------------- bits
struct BitReader {
    const uint8_t* data;
    int len;   // bytes
    int pos = 0;
    bool overrun = false;  // set once a read goes past the buffer
    BitReader(const uint8_t* d, int n) : data(d), len(n) {}
    int left() const { return len * 8 - pos; }
    uint32_t get(int n) {
        uint32_t v = 0;
        while (n-- > 0) {
            int byte = pos >> 3, bit = 7 - (pos & 7);
            if (byte >= len) overrun = true;
            v = (v << 1) | (byte < len ? ((data[byte] >> bit) & 1) : 0);
            pos++;
        }
        return v;
    }
    int get1() {
        int byte = pos >> 3, bit = 7 - (pos & 7);
        if (byte >= len) overrun = true;
        int v = byte < len ? ((data[byte] >> bit) & 1) : 0;
        pos++;
        return v;
    }
};

// ------------------------------------------------------------- huffman
struct HuffTree {
    // child[node][bit]: >0 node index, <=0 ~leaf-index. 0 = unset.
    int32_t child[600][2];
    int nodes = 1;
    void reset() {
        nodes = 1;
        child[0][0] = child[0][1] = 0;
    }
    void insert(uint32_t code, int nbits, int leaf) {
        int at = 0;
        for (int i = nbits - 1; i >= 0; i--) {
            int b = (code >> i) & 1;
            if (i == 0) {
                child[at][b] = ~leaf;
                return;
            }
            if (child[at][b] <= 0) {
                child[at][b] = nodes;
                child[nodes][0] = child[nodes][1] = 0;
                nodes++;
            }
            at = child[at][b];
        }
    }
    int decode(BitReader& br) const {
        int at = 0;
        for (int g = 0; g < 24; g++) {
            int32_t nx = child[at][br.get1()];
            if (nx <= 0) return ~nx;
            at = nx;
        }
        return -1;
    }
};

HuffTree g_spec[11];  // spectral codebooks 1..11 -> index 0..10
HuffTree g_scf;       // scalefactor codebook
bool g_built = false;

// ---------------------------------------------------------- transforms
// Fast inverse MDCT (403x over the direct O(N^2) form, matched to it at
// ~2e-13). The decoder's IMDCT and the encoder's forward MDCT are a
// proven perfect-reconstruction pair (see aac_mdct.cpp), so this inverts
// the encoder's exact pipeline: undo the post-twiddle, run the Q=N/4
// complex FFT, undo the pre-twiddle to recover the folded sequence u,
// then unfold via the TDAC symmetry (region B anti-symmetric, region A
// symmetric). N = 2048 (long) or 256 (short); Q = N/4.
struct ImdctPlan {
    int N, H, log2H;
    std::vector<int> brev;               // bit-reversal permutation (H)
    std::vector<double> fw_re, fw_im;    // FFT twiddles e^{+i 2pi j / len}
    std::vector<double> tw_re, tw_im;    // pre/post twiddle per k (H)

    void init(int n) {
        N = n;
        H = N / 4;
        log2H = 0;
        while ((1 << log2H) < H) log2H++;
        brev.resize(H);
        for (int i = 0; i < H; i++) {
            unsigned r = 0;
            for (int b = 0; b < log2H; b++) r = (r << 1) | ((i >> b) & 1);
            brev[i] = static_cast<int>(r);
        }
        fw_re.assign(H, 0);
        fw_im.assign(H, 0);
        for (int j = 0; j < H; j++) {
            fw_re[j] = std::cos(2.0 * M_PI * j / H);
            fw_im[j] = std::sin(2.0 * M_PI * j / H);  // + sign: inverse FFT
        }
        int M = N / 2;
        tw_re.assign(H, 0);
        tw_im.assign(H, 0);
        for (int k = 0; k < H; k++) {
            tw_re[k] = std::cos(M_PI * (k + 0.125) / M);
            tw_im[k] = -std::sin(M_PI * (k + 0.125) / M);
        }
    }

    void run(const double* spec, double* x) const {
        const int M = N / 2, Q = H;
        double re[512], im[512];  // H <= 512
        for (int k = 0; k < H; k++) {
            double A = spec[2 * k] * 0.5;
            double B = -spec[M - 1 - 2 * k] * 0.5;
            double twr = tw_re[k], twi = tw_im[k];
            re[k] = A * twr + B * twi;
            im[k] = -A * twi + B * twr;
        }
        // In-place radix-2 FFT with precomputed twiddles (inverse: +i).
        for (int i = 0; i < H; i++) {
            int r = brev[i];
            if (r > i) {
                std::swap(re[i], re[r]);
                std::swap(im[i], im[r]);
            }
        }
        for (int len = 2; len <= H; len <<= 1) {
            int half = len >> 1, stride = H / len;
            for (int base = 0; base < H; base += len)
                for (int j = 0; j < half; j++) {
                    double wr = fw_re[j * stride], wi = fw_im[j * stride];
                    int a = base + j, b = a + half;
                    double xr = re[b] * wr - im[b] * wi;
                    double xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
        }
        double u[1024];  // M <= 1024
        const double inv = 2.0 / H;
        for (int nn = 0; nn < H; nn++) {
            double R = re[nn] * inv, I = im[nn] * inv;
            double twr = tw_re[nn], twi = tw_im[nn];
            u[2 * nn] = R * twr + I * twi;
            u[M - 1 - 2 * nn] = -R * twi + I * twr;
        }
        for (int nn = 0; nn < Q; nn++) {
            double b = u[Q + nn] * 0.5, a = -u[nn] * 0.5;
            x[nn] = b;
            x[2 * Q - 1 - nn] = -b;
            x[3 * Q + nn] = a;
            x[3 * Q - 1 - nn] = a;
        }
    }
};

ImdctPlan g_imdct_long, g_imdct_short;

void imdct(const double* X, double* x, int N) {
    (N == 2048 ? g_imdct_long : g_imdct_short).run(X, x);
}

void build_trees() {
    if (g_built) return;
    for (int bk = 0; bk < 11; bk++) {
        g_spec[bk].reset();
        int sz = kBookSize[bk];
        for (int idx = 0; idx < sz; idx++) {
            int nb = kSpecBits[bk][idx];
            if (nb == 0) continue;
            g_spec[bk].insert(kSpecCodes[bk][idx], nb, idx);
        }
    }
    g_scf.reset();
    for (int i = 0; i < 121; i++) {
        if (kScfBits[i] == 0) continue;
        g_scf.insert(kScfCodes[i], kScfBits[i], i);
    }
    g_imdct_long.init(2048);
    g_imdct_short.init(256);
    g_built = true;
}

const int kChannelsForConfig[8] = { 0, 1, 2, 3, 4, 5, 6, 8 };


double sl_win(int n) {  // long sine window, n = 0..2047
    return std::sin(M_PI / 2048.0 * ((n < 1024 ? n : 2047 - n) + 0.5));
}
double ss_win(int n) {  // short sine window, n = 0..255
    return std::sin(M_PI / 256.0 * ((n < 128 ? n : 255 - n) + 0.5));
}

// Synthesis window left halves, [shape][n]: shape 0 = sine, 1 = KBD
// (Kaiser-Bessel derived, ISO/IEC 14496-3 4.6.11.3.2 — alpha 4 long,
// alpha 6 short). window_shape selects per frame; real encoders (ffmpeg,
// Apple, fdk) emit KBD on most frames, and synthesizing those with the
// sine window breaks MDCT perfect reconstruction (~17 dB SNR floor).
double g_win_long[2][1024];
double g_win_short[2][128];
bool g_win_built = false;

void kbd_init(double* w, double alpha, int half) {
    // Cumulative Kaiser-Bessel kernel over half+1 points; the I0(pi*alpha)
    // normalization cancels in the ratio. b is the series form of
    // I0(2*sqrt(t)) with t = i*(half-i)*(alpha*pi/half)^2.
    double cum[1024];
    const double a2 = (alpha * M_PI / half) * (alpha * M_PI / half);
    double sum = 0.0;
    for (int i = 0; i < half; i++) {
        const double t = (double)i * (half - i) * a2;
        double b = 1.0;
        for (int j = 50; j > 0; j--) b = b * t / ((double)j * j) + 1.0;
        sum += b;
        cum[i] = sum;
    }
    sum += 1.0;  // kernel term at i == half: I0(0) = 1
    for (int i = 0; i < half; i++) w[i] = std::sqrt(cum[i] / sum);
}

void build_windows() {
    if (g_win_built) return;
    for (int n = 0; n < 1024; n++) g_win_long[0][n] = sl_win(n);
    for (int n = 0; n < 128; n++) g_win_short[0][n] = ss_win(n);
    kbd_init(g_win_long[1], 4.0, 1024);
    kbd_init(g_win_short[1], 6.0, 128);
    g_win_built = true;
}

}  // namespace

int aac_frame_info(const uint8_t* data, int len, AacFrameInfo* info) {
    if (len < 7) return -1;
    if (data[0] != 0xFF || (data[1] & 0xF6) != 0xF0) return -1;  // sync+layer
    int profile = (data[2] >> 6) & 3;  // 1 = AAC-LC (object type 2)
    if (profile != 1) return -1;
    int sr_index = (data[2] >> 2) & 0xF;
    if (sr_index >= kNumSampleRates) return -1;
    int ch_cfg = ((data[2] & 1) << 2) | ((data[3] >> 6) & 3);
    int frame_len = ((data[3] & 3) << 11) | (data[4] << 3) |
                    ((data[5] >> 5) & 7);
    if (frame_len < 7) return -1;
    info->sample_rate = kSampleRates[sr_index];
    info->channels = ch_cfg >= 1 && ch_cfg <= 7 ? kChannelsForConfig[ch_cfg]
                                                : 0;
    info->samples = 1024;
    info->frame_bytes = frame_len;
    return 0;
}

void AacDecoder::init() {
    build_trees();
    build_windows();
    std::memset(overlap_, 0, sizeof(overlap_));
    prev_window_seq_[0] = prev_window_seq_[1] = 0;
    prev_window_shape_[0] = prev_window_shape_[1] = 0;
    first_ = 1;
}

// Parse ics_info into ics_[ch]; returns 0 or negative.
static int parse_ics_info(BitReader& br, AacDecoder::Ics& ics,
                          int sr_index) {
    br.get1();  // ics_reserved_bit
    ics.window_sequence = static_cast<int>(br.get(2));
    ics.window_shape = static_cast<int>(br.get1());  // 0 sine, 1 KBD
    if (ics.window_sequence == 2) {  // EIGHT_SHORT
        ics.max_sfb = static_cast<int>(br.get(4));
        int grouping = static_cast<int>(br.get(7));
        ics.num_windows = 8;
        // scale_factor_grouping: bit i (from MSB, i=0..6) tells whether
        // window i+1 shares window i's group.
        ics.num_window_groups = 1;
        ics.group_len[0] = 1;
        ics.window_group_of[0] = 0;
        for (int w = 1; w < 8; w++) {
            int shares = (grouping >> (6 - (w - 1))) & 1;
            if (shares) {
                ics.group_len[ics.num_window_groups - 1]++;
            } else {
                ics.group_len[ics.num_window_groups] = 1;
                ics.num_window_groups++;
            }
            ics.window_group_of[w] = ics.num_window_groups - 1;
        }
        (void)sr_index;
    } else {
        ics.max_sfb = static_cast<int>(br.get(6));
        if (br.get1()) return -1;  // predictor_data_present unsupported
        ics.num_windows = 1;
        ics.num_window_groups = 1;
        ics.group_len[0] = 1;
        ics.window_group_of[0] = 0;
    }
    return 0;
}

int AacDecoder::decode_ics(int ch, bool /*common*/) {
    // Placeholder: the heavy lifting is inline in decode_frame so it can
    // share the BitReader; kept for header symmetry.
    (void)ch;
    return 0;
}

// PNS decoder-side RNG (lagged xorshift). PNS is decoder-random by
// design: only band ENERGY is transmitted, not samples, so glint's noise
// differs from any other decoder's — foreign PNS content is compared in
// the magnitude-spectrum domain, never sample-for-sample.
static uint32_t g_pns_state = 0x1234567u;
static double pns_rand() {
    g_pns_state ^= g_pns_state << 13;
    g_pns_state ^= g_pns_state >> 17;
    g_pns_state ^= g_pns_state << 5;
    return (g_pns_state >> 8) * (2.0 / 16777216.0) - 1.0;  // [-1,1)
}

// One coded band [i0, i1): normal dequant, PNS noise fill, or leave
// intensity bands at 0 (filled from the other channel later).
void AacDecoder::dequant_band(int ch, int bk, int sf, int i0, int i1) {
    if (bk == 0) {
        for (int i = i0; i < i1; i++) coef_[ch][i] = 0;
    } else if (bk == 13) {  // PNS: random noise normalized to band energy
        double e = 0;
        for (int i = i0; i < i1; i++) {
            double r = pns_rand();
            coef_[ch][i] = r;
            e += r * r;
        }
        int w = i1 - i0;
        double target = std::pow(2.0, 0.25 * (sf - 100));
        double norm = e > 0 ? target * std::sqrt(static_cast<double>(w) /
                                                  e)
                            : 0.0;
        for (int i = i0; i < i1; i++) coef_[ch][i] *= norm;
    } else if (bk == 14 || bk == 15) {
        for (int i = i0; i < i1; i++) coef_[ch][i] = 0;  // filled by IS
    } else {  // normal spectral book
        double gain = std::pow(2.0, 0.25 * (sf - 100));
        for (int i = i0; i < i1; i++) {
            double v = coef_[ch][i];
            double a = std::fabs(v);
            coef_[ch][i] = (v < 0 ? -1 : 1) * a * std::cbrt(a) * gain;
        }
    }
}

// De-quantize: coef holds signed integer ix (coded order); apply
// xhat = sign(ix)*|ix|^(4/3) * 2^(0.25*(sf-100)) per band.
void AacDecoder::inverse_quant(int ch) {
    const Ics& ics = ics_[ch];
    const int sr = sr_index_;
    if (ics.window_sequence == 2) {
        const uint16_t* swb = kSwbOffsetShort[sr];
        int nb = 0, k = 0, wbase = 0;
        for (int g = 0; g < ics.num_window_groups; g++) {
            for (int b = 0; b < ics.max_sfb; b++) {
                int width = (swb[b + 1] - swb[b]) * ics.group_len[g];
                dequant_band(ch, book_[ch][nb], sf_[ch][nb], k, k + width);
                k += width;
                nb++;
            }
            wbase += ics.group_len[g];
        }
        (void)wbase;
    } else {
        const uint16_t* swb = kSwbOffsetLong[sr];
        for (int b = 0; b < ics.max_sfb; b++) {
            int bk = book_[ch][b];
            dequant_band(ch, bk, sf_[ch][b], swb[b], swb[b + 1]);
        }
        // Above max_sfb: zero.
        for (int i = swb[ics.max_sfb]; i < 1024; i++) coef_[ch][i] = 0;
    }
}

// TNS synthesis: invert the encoder's forward FIR y=x+sum aq[j]x[i-j] by
// the all-pole recursion x=y-sum aq[j]x[i-j] over the region, in the
// coded (= natural, long-only) order.
// Short sequences: rebuild coef_[ch] from the coded (group/sfb-interleaved)
// order into window-major order (8 x 128), so TNS and the filterbank can
// address each short window's spectrum contiguously.
void AacDecoder::deinterleave_short(int ch) {
    const Ics& ics = ics_[ch];
    if (ics.window_sequence != 2) return;
    double nat[1024];
    std::memset(nat, 0, sizeof(nat));
    const uint16_t* swb = kSwbOffsetShort[sr_index_];
    int k = 0, wbase = 0;
    for (int g = 0; g < ics.num_window_groups; g++) {
        for (int b = 0; b < ics.max_sfb; b++) {
            for (int w = 0; w < ics.group_len[g]; w++) {
                double* dst = nat + 128 * (wbase + w);
                for (int i = swb[b]; i < swb[b + 1]; i++)
                    dst[i] = coef_[ch][k++];
            }
        }
        wbase += ics.group_len[g];
    }
    std::memcpy(coef_[ch], nat, sizeof(nat));
}

void AacDecoder::apply_tns(int ch, bool /*inverse*/) {
    const Ics& ics = ics_[ch];
    const bool is_short = ics.window_sequence == 2;
    // NOTE: for short sequences coef_ must already be window-major
    // (deinterleave_short runs before this).
    const uint16_t* swb =
        is_short ? kSwbOffsetShort[sr_index_] : kSwbOffsetLong[sr_index_];
    const int num_swb =
        is_short ? kNumSwbShort[sr_index_] : kNumSwbLong[sr_index_];
    // The filterable region is capped by tns_max_bands AND max_sfb
    // (ISO/IEC 14496-3 tns_decode_frame).
    int mmm = is_short ? kTnsMaxBandsShort[sr_index_]
                       : kTnsMaxBandsLong[sr_index_];
    if (mmm > ics.max_sfb) mmm = ics.max_sfb;
    if (mmm > num_swb) mmm = num_swb;
    const int n_win = is_short ? 8 : 1;
    for (int w = 0; w < n_win; w++) {
        // Successive filters stack downward from the top of the spectrum:
        // filter f covers [top - length, top) where top is the previous
        // filter's bottom, starting at num_swb.
        int bottom = num_swb;
        double* spec = coef_[ch] + (is_short ? 128 * w : 0);
        for (int f = 0; f < tns_nf_[ch][w]; f++) {
            const TnsFilter& t = tns_[ch][w][f];
            const int top = bottom;
            bottom = top - t.length;
            if (bottom < 0) bottom = 0;
            if (t.order == 0) continue;
            int start = swb[bottom < mmm ? bottom : mmm];
            int end = swb[top < mmm ? top : mmm];
            if (end <= start) continue;
            // All-pole synthesis x[i] = y[i] - sum lpc[j]*x[i -+ j], run in
            // the coded direction (0 = upward, 1 = downward).
            const int inc = t.direction ? -1 : 1;
            int i = t.direction ? end - 1 : start;
            double hist[13] = { 0 };
            for (int k = 0; k < end - start; k++, i += inc) {
                double y = spec[i];
                int hmax = k < t.order ? k : t.order;
                for (int j = 1; j <= hmax; j++) y -= t.lpc[j] * hist[j - 1];
                for (int j = t.order - 1; j > 0; j--) hist[j] = hist[j - 1];
                hist[0] = y;
                spec[i] = y;
            }
        }
    }
}

void AacDecoder::imdct_channel(int ch, float* out) {
    const Ics& ics = ics_[ch];
    double time[2048];

    // Window shape: the LEFT (rising) half must match the falling edge of
    // the previous frame's window, so it uses the previous frame's shape;
    // the right half uses this frame's shape (ISO/IEC 14496-3 4.6.11.3.2).
    const int cur = ics.window_shape;
    const int prv = prev_window_shape_[ch];

    if (ics.window_sequence != 2) {
        // Long-family: 1024 coefs -> 2048 samples, windowed per sequence.
        double x[2048];
        imdct(coef_[ch], x, 2048);
        int seq = ics.window_sequence;  // 0 long, 1 start, 3 stop
        for (int n = 0; n < 2048; n++) {
            double w;
            if (n < 1024) {
                // Left half (previous frame's shape).
                if (seq == 3) {  // STOP: short-based left
                    if (n < 448) w = 0.0;
                    else if (n < 576) w = g_win_short[prv][n - 448];
                    else w = 1.0;
                } else {
                    w = g_win_long[prv][n];  // long/start left = long
                }
            } else {
                // Right half (this frame's shape).
                if (seq == 1) {  // START: short-based right
                    int m = n - 1024;
                    if (m < 448) w = 1.0;
                    else if (m < 576) w = g_win_short[cur][575 - m];
                    else w = 0.0;
                } else {
                    w = g_win_long[cur][2047 - n];  // long/stop right = long
                }
            }
            time[n] = x[n] * w;
        }
    } else {
        // Eight short (coef_ already window-major via deinterleave_short):
        // IMDCT each window, window, overlap within a 2048 block at
        // 128-sample hops.
        std::memset(time, 0, sizeof(time));
        for (int w = 0; w < 8; w++) {
            double xs[256], ss[256];
            imdct(coef_[ch] + 128 * w, xs, 256);
            // Only the very first short window's rising edge overlaps the
            // previous frame; every other edge is this frame's shape.
            const int lsh = (w == 0) ? prv : cur;
            for (int n = 0; n < 256; n++)
                ss[n] = xs[n] * (n < 128 ? g_win_short[lsh][n]
                                         : g_win_short[cur][255 - n]);
            int base = 448 + 128 * w;
            for (int n = 0; n < 256; n++) time[base + n] += ss[n];
        }
    }

    // Overlap-add: first 1024 with previous overlap; save second 1024.
    // The encoder works in int16 PCM scale; normalize to +-1.0.
    const double kNorm = 1.0 / 32768.0;
    for (int n = 0; n < 1024; n++)
        out[n] = static_cast<float>((time[n] + overlap_[ch][n]) * kNorm);
    for (int n = 0; n < 1024; n++) overlap_[ch][n] = time[1024 + n];
    prev_window_shape_[ch] = cur;
}

int AacDecoder::decode_frame(const uint8_t* data, int len, float* pcm,
                             AacFrameInfo* info_out) {
    AacFrameInfo info;
    if (aac_frame_info(data, len, &info) < 0) return -1;
    if (info.frame_bytes > len) return -1;
    if (info_out) *info_out = info;

    // ADTS: fixed header 28 bits + variable 28 bits; protection_absent in
    // byte1 bit0 (1 = no CRC). Header is 7 bytes then raw_data_block.
    // This decoder supports mono/stereo (SCE/CPE) only; a header
    // claiming more channels (possibly corrupt) must not reach the
    // fixed-width PCM write.
    if (info.channels < 1 || info.channels > 2) return -1;
    int protection_absent = data[1] & 1;
    int hdr = protection_absent ? 7 : 9;
    sr_index_ = (data[2] >> 2) & 0xF;
    channels_ = info.channels;

    BitReader br(data + hdr, info.frame_bytes - hdr);

    int decoded_ch = 0;
    // raw_data_block: id_syn_ele until END (7).
    for (;;) {
        if (br.overrun || br.left() < 3) break;
        int id = static_cast<int>(br.get(3));
        if (id == 7) break;  // ID_END
        if (id == 0 || id == 1) {  // SCE / CPE
            const bool cpe = id == 1;
            const int nch = cpe ? 2 : 1;
            br.get(4);  // element_instance_tag
            bool common = false;
            if (cpe) common = br.get1();

            Ics shared_ics{};
            int ms_nbands = 0;
            if (cpe && common) {
                if (parse_ics_info(br, shared_ics, sr_index_) < 0)
                    return -2;
                ms_mask_present_ = static_cast<int>(br.get(2));
                ics_[0] = shared_ics;
                ics_[1] = shared_ics;
                ms_nbands = shared_ics.window_sequence == 2
                                ? shared_ics.num_window_groups *
                                      shared_ics.max_sfb
                                : shared_ics.max_sfb;
                // ms_used flags come here, BEFORE the channel bodies.
                if (ms_mask_present_ == 1) {
                    for (int b = 0; b < ms_nbands; b++)
                        ms_used_[b] = static_cast<uint8_t>(br.get1());
                } else if (ms_mask_present_ == 2) {
                    for (int b = 0; b < ms_nbands; b++) ms_used_[b] = 1;
                }
            } else {
                ms_mask_present_ = 0;
            }

            for (int c = 0; c < nch; c++) {
                int ch = decoded_ch + c;
                if (ch > 1) return -2;  // stereo max for now
                Ics& ics = ics_[ch];
                // global_gain
                int global_gain = static_cast<int>(br.get(8));
                if (!(cpe && common)) {
                    if (parse_ics_info(br, ics, sr_index_) < 0)
                        return -2;
                }
                max_sfb_[ch] = ics.max_sfb;
                int nbands = ics.window_sequence == 2
                                 ? ics.num_window_groups * ics.max_sfb
                                 : ics.max_sfb;

                // ---- section_data ----
                int sect_bits = ics.window_sequence == 2 ? 3 : 5;
                int sect_esc = ics.window_sequence == 2 ? 7 : 31;
                {
                    int b = 0;
                    // Per group, sections span that group's max_sfb bands.
                    for (int g = 0; g < ics.num_window_groups; g++) {
                        int filled = 0;
                        while (filled < ics.max_sfb) {
                            int cb = static_cast<int>(br.get(4));
                            if (cb == 12) return -2;  // reserved book
                            int len_sect = 0, inc;
                            do {
                                inc = static_cast<int>(br.get(sect_bits));
                                len_sect += inc;
                                if (br.overrun) return -3;
                            } while (inc == sect_esc);
                            if (len_sect == 0 ||
                                filled + len_sect > ics.max_sfb)
                                return -2;  // must advance, must not overrun
                            for (int i = 0; i < len_sect; i++)
                                book_[ch][b + i] = cb;
                            b += len_sect;
                            filled += len_sect;
                        }
                    }
                    (void)nbands;
                }
                // ---- scale_factor_data ----
                // DPCM chain starts at global_gain; every CODED band
                // (book != 0) transmits a delta (the first coded band's
                // delta is 0, so its scalefactor == global_gain).
                {
                    int scale = global_gain;
                    int is_pos = 0;
                    int noise_nrg = global_gain - 90;
                    bool first_noise = true;
                    for (int b = 0; b < nbands; b++) {
                        int bk = book_[ch][b];
                        if (bk == 0) {
                            sf_[ch][b] = 0;
                            continue;
                        }
                        if (bk == 14 || bk == 15) {  // intensity
                            int idx = g_scf.decode(br);
                            if (idx < 0) return -3;
                            is_pos += idx - 60;
                            sf_[ch][b] = is_pos;
                        } else if (bk == 13) {  // noise (PNS)
                            if (first_noise) {
                                first_noise = false;
                                noise_nrg +=
                                    static_cast<int>(br.get(9)) - 256;
                            } else {
                                int idx = g_scf.decode(br);
                                if (idx < 0) return -3;
                                noise_nrg += idx - 60;
                            }
                            sf_[ch][b] = noise_nrg;
                        } else {  // normal
                            int idx = g_scf.decode(br);
                            if (idx < 0) return -3;
                            scale += idx - 60;
                            sf_[ch][b] = scale;
                        }
                    }
                }

                // ---- pulse / tns / gain ----
                int pulse_present = br.get1();
                if (pulse_present) { if(std::getenv("AACDBG"))std::fprintf(stderr,"[PULSE]"); return -2; }
                int tns_present = br.get1();
                std::memset(tns_nf_[ch], 0, sizeof(tns_nf_[ch]));
                if (tns_present) {
                    int n_win = ics.window_sequence == 2 ? 8 : 1;
                    for (int w = 0; w < n_win; w++) {
                        int n_filt = static_cast<int>(
                            br.get(ics.window_sequence == 2 ? 1 : 2));
                        int coef_res = 0;
                        // (per-window: reads only if n_filt>0)
                        if (n_filt) coef_res = br.get1();
                        for (int fi = 0; fi < n_filt; fi++) {
                            int length = static_cast<int>(
                                br.get(ics.window_sequence == 2 ? 4 : 6));
                            int order = static_cast<int>(
                                br.get(ics.window_sequence == 2 ? 3 : 5));
                            // LC profile: order <= 12 (long) / 7 (short);
                            // clamp so a corrupt field can't overflow the
                            // order-13 LPC scratch arrays.
                            int order_cap = ics.window_sequence == 2 ? 7 : 12;
                            if (order > order_cap) order = order_cap;
                            int direction = 0, compress = 0;
                            double lpc[13] = { 1.0 };
                            if (order) {
                                direction = br.get1();
                                compress = br.get1();
                                int res = coef_res + 3;  // 3- or 4-bit coefs
                                int nbits = res - compress;
                                // dequant reflection coefs, refl->LPC.
                                // iqfac depends on coef_res, NOT on compress
                                // (compression only drops the top bit).
                                const double half = 1 << (res - 1);
                                const double iq_p = (half - 0.5) / (M_PI / 2.0);
                                const double iq_n = (half + 0.5) / (M_PI / 2.0);
                                double parc[13];
                                for (int m = 0; m < order; m++) {
                                    int raw = static_cast<int>(
                                        br.get(nbits));
                                    // sign-extend nbits
                                    int sign = 1 << (nbits - 1);
                                    if (raw & sign) raw -= (sign << 1);
                                    parc[m] = std::sin(raw / (raw >= 0 ? iq_p
                                                                       : iq_n));
                                }
                                for (int m = 1; m <= order; m++) {
                                    double k = parc[m - 1];
                                    double tmp[13];
                                    std::memcpy(tmp, lpc, sizeof(tmp));
                                    for (int j = 1; j < m; j++)
                                        lpc[j] = tmp[j] + k * tmp[m - j];
                                    lpc[m] = k;
                                }
                            }
                            // Up to 3 stacked filters per window (apply_tns
                            // walks the regions top-down within a window).
                            if (fi < 3) {
                                TnsFilter& tf = tns_[ch][w][fi];
                                tf.length = length;
                                tf.order = order;
                                tf.direction = direction;
                                std::memcpy(tf.lpc, lpc, sizeof(lpc));
                                tns_nf_[ch][w] = fi + 1;
                            }
                            (void)compress;
                        }
                    }
                }
                int gain_present = br.get1();
                if (gain_present) { if(std::getenv("AACDBG"))std::fprintf(stderr,"[GAIN]"); return -2; }

                // ---- spectral_data ----
                std::memset(coef_[ch], 0, sizeof(coef_[ch]));
                {
                    const uint16_t* swb =
                        ics.window_sequence == 2
                            ? kSwbOffsetShort[sr_index_]
                            : kSwbOffsetLong[sr_index_];
                    int k = 0;  // coded-order line index
                    for (int b = 0; b < nbands; b++) {
                        int bk = book_[ch][b];
                        int width;
                        if (ics.window_sequence == 2) {
                            int g = 0, acc = ics.max_sfb;
                            // band b -> group g, sfb bb
                            int bb = b % ics.max_sfb;
                            g = b / ics.max_sfb;
                            width = (swb[bb + 1] - swb[bb]) *
                                    ics.group_len[g];
                            (void)acc;
                        } else {
                            width = swb[b + 1] - swb[b];
                        }
                        if (bk == 0 || bk >= 12) {
                            k += width;  // zero / reserved / PNS / IS
                            continue;
                        }
                        int dim = kBookDim[bk - 1];
                        int lav = kBookLav[bk - 1];
                        bool sgn = kBookSigned[bk - 1] != 0;
                        int end = k + width;
                        for (int i = k; i < end; i += dim) {
                            if (br.overrun) return -3;
                            int idx = g_spec[bk - 1].decode(br);
                            if (idx < 0) return -3;
                            int vals[4];
                            if (sgn) {
                                int t = idx;
                                for (int j = dim - 1; j >= 0; j--) {
                                    vals[j] = t % (2 * lav + 1) - lav;
                                    t /= (2 * lav + 1);
                                }
                            } else {
                                int t = idx;
                                int base = (bk == 11) ? 17 : (lav + 1);
                                for (int j = dim - 1; j >= 0; j--) {
                                    vals[j] = t % base;
                                    t /= base;
                                }
                                for (int j = 0; j < dim; j++) {
                                    if (vals[j] != 0)
                                        if (br.get1()) vals[j] = -vals[j];
                                }
                                if (bk == 11) {
                                    for (int j = 0; j < dim; j++) {
                                        if (std::abs(vals[j]) == 16) {
                                            int n1 = 0;
                                            while (br.get1()) n1++;
                                            int mant = static_cast<int>(
                                                br.get(n1 + 4));
                                            int a = (1 << (n1 + 4)) + mant;
                                            vals[j] = vals[j] < 0 ? -a : a;
                                        }
                                    }
                                }
                            }
                            for (int j = 0; j < dim && i + j < end; j++)
                                coef_[ch][i + j] = vals[j];
                        }
                        k = end;
                    }
                }
            }

            is_cpe_ms_ = (cpe && common && ms_mask_present_) ? 1 : 0;
            decoded_ch += nch;
        } else if (id == 6) {  // FIL (fill) - skip
            int cnt = static_cast<int>(br.get(4));
            if (cnt == 15) cnt += static_cast<int>(br.get(8)) - 1;
            for (int i = 0; i < cnt; i++) br.get(8);
        } else {
            // LFE(3)/DSE(4)/PCE(5)/CCE(2): not emitted by glint.
            return -2;
        }
        if (decoded_ch >= channels_) {
            // consume until END if more elements, else break
        }
    }

    // ---- reconstruct: inverse-quant, M/S, TNS, filterbank ----
    for (int ch = 0; ch < channels_ && ch < 2; ch++) inverse_quant(ch);

    // Stereo: M/S recombine (skipping intensity bands), then intensity
    // stereo (right band derived from left). Encoder did TNS on L/R then
    // M/S, so decode undoes M/S first; IS runs on the M/S-recombined
    // left channel. Coefficients are coded order, shared layout.
    if (channels_ == 2) {
        const Ics& ics = ics_[0];
        const uint16_t* swb = ics.window_sequence == 2
                                  ? kSwbOffsetShort[sr_index_]
                                  : kSwbOffsetLong[sr_index_];
        int nbands = ics.window_sequence == 2
                         ? ics.num_window_groups * ics.max_sfb
                         : ics.max_sfb;
        int k = 0;
        for (int b = 0; b < nbands; b++) {
            int width;
            if (ics.window_sequence == 2) {
                int g = b / ics.max_sfb, bb = b % ics.max_sfb;
                width = (swb[bb + 1] - swb[bb]) * ics.group_len[g];
            } else {
                width = swb[b + 1] - swb[b];
            }
            int rbook = book_[1][b];
            bool is_band = (rbook == 14 || rbook == 15);
            if (is_cpe_ms_ && ms_used_[b] && !is_band) {
                for (int i = k; i < k + width; i++) {
                    double m = coef_[0][i], s = coef_[1][i];
                    coef_[0][i] = m + s;
                    coef_[1][i] = m - s;
                }
            }
            if (is_band) {
                // right = sign * 2^(0.25*is_pos) * left; sign from the
                // book (15 in-phase, 14 out-of-phase), flipped by M/S.
                int is_pos = sf_[1][b];
                double sign = (rbook == 15) ? 1.0 : -1.0;
                if (is_cpe_ms_ && ms_mask_present_ && ms_used_[b])
                    sign = -sign;
                double scale = sign * std::pow(2.0, 0.25 * is_pos);
                for (int i = k; i < k + width; i++)
                    coef_[1][i] = scale * coef_[0][i];
            }
            k += width;
        }
    }

    for (int ch = 0; ch < channels_ && ch < 2; ch++) {
        deinterleave_short(ch);
        apply_tns(ch, true);
    }

    static thread_local float ch_pcm[2][1024];
    for (int ch = 0; ch < channels_ && ch < 2; ch++)
        imdct_channel(ch, ch_pcm[ch]);

    const int oc = channels_ < 2 ? channels_ : 2;
    for (int n = 0; n < 1024; n++)
        for (int ch = 0; ch < oc; ch++)
            pcm[n * oc + ch] = ch_pcm[ch][n];

    return 1024;
}

}  // namespace aac
}  // namespace glint
