// glint - Ogg-Vorbis I decoder
// MIT License - Clean-room implementation (from the Vorbis I spec, xiph.org).

#include "vorbis_decoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>

#include "vorbis_bits.hpp"
#include "vorbis_imdct.hpp"
#include "vorbis_ogg.hpp"

namespace glint {
namespace vorbis {

// ---------------------------------------------------------------------------
// Identification header (spec §4.2.2)
// ---------------------------------------------------------------------------

IdHeader parse_id_header(const uint8_t* pkt, size_t len) {
    IdHeader h;
    // Common header: packet type byte (0x01) + "vorbis".
    if (len < 7 || pkt[0] != 0x01 || std::memcmp(pkt + 1, "vorbis", 6) != 0)
        return h;
    BitReader br(pkt + 7, len - 7);
    uint32_t version = br.read(32);
    int channels = static_cast<int>(br.read(8));
    uint32_t rate = br.read(32);
    br.read(32);  // bitrate_maximum
    br.read(32);  // bitrate_nominal
    br.read(32);  // bitrate_minimum
    int bs = static_cast<int>(br.read(8));
    int bs0 = 1 << (bs & 0x0F);
    int bs1 = 1 << ((bs >> 4) & 0x0F);
    int framing = br.read_bit();
    if (br.overrun()) return h;
    if (version != 0 || channels < 1 || rate < 1 || framing != 1) return h;
    // Blocksizes: powers of two in [64, 8192], short <= long (spec §4.2.2).
    if (bs0 < 64 || bs0 > 8192 || bs1 < 64 || bs1 > 8192 || bs0 > bs1)
        return h;
    h.version = 0;
    h.channels = channels;
    h.sample_rate = rate;
    h.blocksize0 = bs0;
    h.blocksize1 = bs1;
    h.valid = true;
    return h;
}

// ---------------------------------------------------------------------------
// Spec helpers (§9.2)
// ---------------------------------------------------------------------------

float float32_unpack(uint32_t x) {  // spec §9.2.2
    uint32_t mant = x & 0x1fffffu;
    uint32_t sign = x & 0x80000000u;
    uint32_t expo = (x & 0x7fe00000u) >> 21;
    double m = static_cast<double>(mant);
    if (sign) m = -m;
    return static_cast<float>(std::ldexp(m, static_cast<int>(expo) - 788));
}

uint32_t lookup1_values(int entries, int dimensions) {  // spec §9.2.3
    // Greatest integer r such that r^dimensions <= entries.
    if (dimensions <= 0 || entries < 0) return 0;
    uint32_t r = 0;
    for (;;) {
        // Test (r+1)^dimensions <= entries without overflow.
        double p = 1.0;
        uint32_t rr = r + 1;
        for (int d = 0; d < dimensions; d++) p *= rr;
        if (p > static_cast<double>(entries)) break;
        r = rr;
        if (r > 0x10000000u) break;  // safety
    }
    return r;
}

// ---------------------------------------------------------------------------
// Codebooks (spec §3)
// ---------------------------------------------------------------------------

namespace {
constexpr int32_t kNone = -1;
inline int32_t leaf_of(int entry) { return -(entry + 2); }
inline bool is_leaf(int32_t v) { return v <= -2; }
inline int entry_of(int32_t v) { return -v - 2; }
}  // namespace

bool Codebook::build_huffman() {
    child0.assign(1, kNone);
    child1.assign(1, kNone);
    std::vector<char> full(1, 0);  // per node: subtree has no free leaf slot

    int used = 0, single = -1;
    for (int i = 0; i < entries; i++)
        if (lengths[i] > 0) { used++; single = i; }
    single_entry_ = (used == 1) ? single : -1;
    if (single_entry_ >= 0) {
        // Degenerate single-entry codebook: no bits are read on decode.
        for (int i = 0; i < entries; i++)
            if (lengths[i] > 32) return false;
        return true;
    }

    // A slot value can hold more leaves iff it is empty (kNone) or a
    // non-full internal node.
    auto can_take = [&](int32_t slot) -> bool {
        if (slot == kNone) return true;
        if (is_leaf(slot)) return false;
        return !full[slot];
    };
    // Leftmost placement of a leaf at exactly depth L (child0 preferred).
    std::function<bool(int32_t, int, int, int)> place =
        [&](int32_t node, int depth, int L, int entry) -> bool {
        if (depth + 1 == L) {
            if (child0[node] == kNone)
                child0[node] = leaf_of(entry);
            else if (child1[node] == kNone)
                child1[node] = leaf_of(entry);
            else
                return false;
            full[node] = !can_take(child0[node]) && !can_take(child1[node]);
            return true;
        }
        for (int b = 0; b < 2; b++) {
            int32_t slot = b == 0 ? child0[node] : child1[node];
            if (is_leaf(slot)) continue;
            if (slot == kNone) {
                // Allocate a fresh child (may reallocate the arrays).
                child0.push_back(kNone);
                child1.push_back(kNone);
                full.push_back(0);
                slot = static_cast<int32_t>(child0.size() - 1);
                (b == 0 ? child0[node] : child1[node]) = slot;
            }
            if (!full[slot] && place(slot, depth + 1, L, entry)) {
                full[node] =
                    !can_take(child0[node]) && !can_take(child1[node]);
                return true;
            }
        }
        return false;
    };

    for (int i = 0; i < entries; i++) {
        int L = lengths[i];
        if (L == 0) continue;
        if (L > 32) return false;
        if (!place(0, 0, L, i)) return false;  // over-subscribed tree
    }
    return true;
}

void Codebook::build_vq() {
    vq.clear();
    if (lookup_type != 1 && lookup_type != 2) return;
    vq.assign(static_cast<size_t>(entries) * dimensions, 0.f);
    for (int i = 0; i < entries; i++) {
        double last = 0.0;
        if (lookup_type == 1) {
            uint32_t idx_div = 1;
            for (int j = 0; j < dimensions; j++) {
                uint32_t off =
                    lookup_values ? (static_cast<uint32_t>(i) / idx_div) %
                                        static_cast<uint32_t>(lookup_values)
                                  : 0;
                double v = static_cast<double>(multiplicands[off]) *
                               delta_value + minimum_value + last;
                if (sequence_p) last = v;
                vq[static_cast<size_t>(i) * dimensions + j] =
                    static_cast<float>(v);
                idx_div *= static_cast<uint32_t>(lookup_values);
            }
        } else {  // lookup_type == 2
            for (int j = 0; j < dimensions; j++) {
                size_t off = static_cast<size_t>(i) * dimensions + j;
                double v = static_cast<double>(multiplicands[off]) *
                               delta_value + minimum_value + last;
                if (sequence_p) last = v;
                vq[static_cast<size_t>(i) * dimensions + j] =
                    static_cast<float>(v);
            }
        }
    }
}

int Codebook::decode_scalar(BitReader& br) const {
    if (single_entry_ >= 0) return single_entry_;
    int32_t node = 0;
    if (child0.empty()) return -1;
    for (;;) {
        int bit = br.read_bit();
        if (br.overrun()) return -1;
        int32_t next = bit ? child1[node] : child0[node];
        if (next == kNone) return -1;
        if (is_leaf(next)) return entry_of(next);
        node = next;
    }
}

int Codebook::decode_vector(BitReader& br, float* out) const {
    int e = decode_scalar(br);
    if (e < 0 || e >= entries) return -1;
    if (vq.empty()) return -1;
    for (int j = 0; j < dimensions; j++)
        out[j] = vq[static_cast<size_t>(e) * dimensions + j];
    return e;
}

int read_codebook(BitReader& br, Codebook& cb) {
    if (br.read(24) != 0x564342u) return -1;  // "BCV" sync pattern
    cb.dimensions = static_cast<int>(br.read(16));
    cb.entries = static_cast<int>(br.read(24));
    if (cb.entries < 1 || cb.dimensions < 0 || cb.dimensions > 4096)
        return -1;
    // Bounded allocation: real Vorbis codebooks never approach this; the cap
    // keeps a malformed setup header (ordered coding can claim a huge entry
    // count from a few bits) from forcing an unbounded tree/VQ allocation.
    if (cb.entries > (1 << 20)) return -1;
    cb.lengths.assign(cb.entries, 0);

    int ordered = br.read_bit();
    if (!ordered) {
        int sparse = br.read_bit();
        for (int i = 0; i < cb.entries; i++) {
            if (sparse) {
                if (br.read_bit())
                    cb.lengths[i] = static_cast<uint8_t>(br.read(5) + 1);
                else
                    cb.lengths[i] = 0;
            } else {
                cb.lengths[i] = static_cast<uint8_t>(br.read(5) + 1);
            }
        }
    } else {
        int cur = 0;
        int cur_len = static_cast<int>(br.read(5)) + 1;
        while (cur < cb.entries) {
            int num = static_cast<int>(br.read(ilog(cb.entries - cur)));
            if (cur + num > cb.entries) return -1;
            for (int i = cur; i < cur + num; i++)
                cb.lengths[i] = static_cast<uint8_t>(cur_len);
            cur += num;
            cur_len++;
            if (cur_len > 33) return -1;
        }
    }

    cb.lookup_type = static_cast<int>(br.read(4));
    if (cb.lookup_type == 1 || cb.lookup_type == 2) {
        cb.minimum_value = float32_unpack(br.read(32));
        cb.delta_value = float32_unpack(br.read(32));
        cb.value_bits = static_cast<int>(br.read(4)) + 1;
        cb.sequence_p = br.read_bit();
        if (cb.lookup_type == 1)
            cb.lookup_values =
                static_cast<int>(lookup1_values(cb.entries, cb.dimensions));
        else
            cb.lookup_values = cb.entries * cb.dimensions;
        // Bounded allocation guard: the multiplicands must fit in the packet.
        size_t need_bits =
            static_cast<size_t>(cb.value_bits) * cb.lookup_values;
        if (cb.lookup_values < 0 ||
            need_bits > br.bit_length() - br.bits_read())
            return -1;
        cb.multiplicands.assign(cb.lookup_values, 0);
        for (int i = 0; i < cb.lookup_values; i++)
            cb.multiplicands[i] = br.read(cb.value_bits);
    } else if (cb.lookup_type != 0) {
        return -1;  // reserved
    }

    if (br.overrun()) return -1;
    if (!cb.build_huffman()) return -1;
    cb.build_vq();
    return 0;
}

// ---------------------------------------------------------------------------
// Floor 0 (LSP) curve synthesis (spec §6.2.3)
// ---------------------------------------------------------------------------

void floor0_curve(int order, const float* coef, int amplitude,
                  int amplitude_bits, int amplitude_offset, int rate,
                  int bark_map_size, int n, float* out) {
    auto bark = [](double x) {
        return 13.1 * std::atan(0.00074 * x) +
               2.24 * std::atan(0.0000000185 * x * x) + 0.0001 * x;
    };
    if (n <= 0) return;
    std::vector<int> map(n);
    double denom = bark(0.5 * rate);
    for (int i = 0; i < n; i++) {
        int val = static_cast<int>(std::floor(
            bark((rate / (2.0 * n)) * i) * bark_map_size / denom));
        map[i] = val < bark_map_size - 1 ? val : bark_map_size - 1;
    }
    int i = 0;
    while (i < n) {
        double w = M_PI * map[i] / bark_map_size;
        double cw = std::cos(w);
        double p, q;
        if (order & 1) {
            p = 1.0 - cw * cw;
            q = 0.25;
            for (int j = 0; j < (order - 1) / 2; j++) {
                double t = std::cos(coef[2 * j + 1]) - cw;
                p *= 4.0 * t * t;
            }
            for (int j = 0; j < (order + 1) / 2; j++) {
                double t = std::cos(coef[2 * j]) - cw;
                q *= 4.0 * t * t;
            }
        } else {
            p = (1.0 - cw) * 0.5;
            q = (1.0 + cw) * 0.5;
            for (int j = 0; j < order / 2; j++) {
                double tp = std::cos(coef[2 * j + 1]) - cw;
                double tq = std::cos(coef[2 * j]) - cw;
                p *= 4.0 * tp * tp;
                q *= 4.0 * tq * tq;
            }
        }
        double lin = std::exp(
            0.11512925 *
            (amplitude * amplitude_offset /
                 (((1 << amplitude_bits) - 1) * std::sqrt(p + q)) -
             amplitude_offset));
        int j = i;
        while (j < n && map[j] == map[i]) out[j++] = static_cast<float>(lin);
        i = j;
    }
}

// ---------------------------------------------------------------------------
// Setup header: floors, residues, mappings, modes (spec §4.2.4, §6-§8)
// ---------------------------------------------------------------------------

namespace {

struct Floor1 {
    int partitions = 0;
    std::vector<int> partition_class;        // [partitions]
    std::vector<int> class_dimensions;       // [maxclass+1]
    std::vector<int> class_subclasses;       // [maxclass+1]
    std::vector<int> class_masterbooks;      // [maxclass+1]
    std::vector<std::vector<int>> subclass_books;  // [maxclass+1][1<<subcl]
    int multiplier = 0;
    int rangebits = 0;
    std::vector<int> x_list;                 // coded-order X positions
    int values = 0;
    // Derived: sorted order + low/high neighbours for curve synthesis.
    std::vector<int> sorted_idx;             // indices sorted by x_list value
    std::vector<int> low_neighbour;          // per coded index
    std::vector<int> high_neighbour;
};

struct Floor0 {
    int order = 0;
    int rate = 0;
    int bark_map_size = 0;
    int amplitude_bits = 0;
    int amplitude_offset = 0;
    std::vector<int> books;
};

struct Floor {
    int type = -1;
    Floor1 f1;
    Floor0 f0;
};

struct Residue {
    int type = 0;
    int begin = 0, end = 0, partition_size = 0;
    int classifications = 0, classbook = 0;
    std::vector<int> cascade;                // [classifications]
    std::vector<std::array<int, 8>> books;   // [classifications][8], -1 unused
};

struct Mapping {
    int submaps = 1;
    int coupling_steps = 0;
    std::vector<int> magnitude, angle;       // [coupling_steps]
    std::vector<int> mux;                    // [channels]
    std::vector<int> submap_floor;           // [submaps]
    std::vector<int> submap_residue;         // [submaps]
};

struct Mode {
    int blockflag = 0;
    int windowtype = 0;
    int transformtype = 0;
    int mapping = 0;
};

struct Setup {
    IdHeader id;
    std::vector<Codebook> codebooks;
    std::vector<Floor> floors;
    std::vector<Residue> residues;
    std::vector<Mapping> mappings;
    std::vector<Mode> modes;
};

int parse_floor1(BitReader& br, Floor1& f) {
    f.partitions = static_cast<int>(br.read(5));
    f.partition_class.resize(f.partitions);
    int maxclass = -1;
    for (int i = 0; i < f.partitions; i++) {
        f.partition_class[i] = static_cast<int>(br.read(4));
        if (f.partition_class[i] > maxclass) maxclass = f.partition_class[i];
    }
    f.class_dimensions.assign(maxclass + 1, 0);
    f.class_subclasses.assign(maxclass + 1, 0);
    f.class_masterbooks.assign(maxclass + 1, 0);
    f.subclass_books.assign(maxclass + 1, {});
    for (int j = 0; j <= maxclass; j++) {
        f.class_dimensions[j] = static_cast<int>(br.read(3)) + 1;
        f.class_subclasses[j] = static_cast<int>(br.read(2));
        if (f.class_subclasses[j] > 0)
            f.class_masterbooks[j] = static_cast<int>(br.read(8));
        int n = 1 << f.class_subclasses[j];
        f.subclass_books[j].resize(n);
        for (int k = 0; k < n; k++)
            f.subclass_books[j][k] = static_cast<int>(br.read(8)) - 1;
    }
    f.multiplier = static_cast<int>(br.read(2)) + 1;
    f.rangebits = static_cast<int>(br.read(4));
    f.x_list.clear();
    f.x_list.push_back(0);
    f.x_list.push_back(1 << f.rangebits);
    for (int i = 0; i < f.partitions; i++) {
        int cls = f.partition_class[i];
        for (int j = 0; j < f.class_dimensions[cls]; j++)
            f.x_list.push_back(static_cast<int>(br.read(f.rangebits)));
    }
    f.values = static_cast<int>(f.x_list.size());
    if (f.values > 65) return -1;  // spec cap (2 + 63)
    // Derived neighbours (spec §9.2.5): for each element >= 2, find the
    // closest lower/higher x among earlier elements.
    f.low_neighbour.assign(f.values, 0);
    f.high_neighbour.assign(f.values, 0);
    for (int i = 2; i < f.values; i++) {
        int lo = 0, hi = 0, low_x = -1, high_x = 1 << 30;
        for (int j = 0; j < i; j++) {
            int x = f.x_list[j];
            if (x < f.x_list[i] && x > low_x) { low_x = x; lo = j; }
            if (x > f.x_list[i] && x < high_x) { high_x = x; hi = j; }
        }
        f.low_neighbour[i] = lo;
        f.high_neighbour[i] = hi;
    }
    // sorted_idx: indices ordered by x value (stable).
    f.sorted_idx.resize(f.values);
    for (int i = 0; i < f.values; i++) f.sorted_idx[i] = i;
    std::sort(f.sorted_idx.begin(), f.sorted_idx.end(),
              [&](int a, int b) {
                  if (f.x_list[a] != f.x_list[b])
                      return f.x_list[a] < f.x_list[b];
                  return a < b;
              });
    return br.overrun() ? -1 : 0;
}

int parse_floor0(BitReader& br, Floor0& f) {
    f.order = static_cast<int>(br.read(8));
    f.rate = static_cast<int>(br.read(16));
    f.bark_map_size = static_cast<int>(br.read(16));
    f.amplitude_bits = static_cast<int>(br.read(6));
    f.amplitude_offset = static_cast<int>(br.read(8));
    int nbooks = static_cast<int>(br.read(4)) + 1;
    f.books.resize(nbooks);
    for (int i = 0; i < nbooks; i++)
        f.books[i] = static_cast<int>(br.read(8));
    return br.overrun() ? -1 : 0;
}

int parse_residue(BitReader& br, Residue& r) {
    r.begin = static_cast<int>(br.read(24));
    r.end = static_cast<int>(br.read(24));
    r.partition_size = static_cast<int>(br.read(24)) + 1;
    r.classifications = static_cast<int>(br.read(6)) + 1;
    r.classbook = static_cast<int>(br.read(8));
    r.cascade.assign(r.classifications, 0);
    for (int i = 0; i < r.classifications; i++) {
        int high = 0;
        int low = static_cast<int>(br.read(3));
        int flag = br.read_bit();
        if (flag) high = static_cast<int>(br.read(5));
        r.cascade[i] = low | (high << 3);
    }
    r.books.assign(r.classifications, {});
    for (int i = 0; i < r.classifications; i++)
        for (int k = 0; k < 8; k++)
            r.books[i][k] = (r.cascade[i] & (1 << k))
                                ? static_cast<int>(br.read(8))
                                : -1;
    return br.overrun() ? -1 : 0;
}

int parse_mapping(BitReader& br, Mapping& m, int channels) {
    if (br.read_bit())
        m.submaps = static_cast<int>(br.read(4)) + 1;
    else
        m.submaps = 1;
    if (br.read_bit()) {
        m.coupling_steps = static_cast<int>(br.read(8)) + 1;
        int bits = ilog(channels - 1);
        m.magnitude.resize(m.coupling_steps);
        m.angle.resize(m.coupling_steps);
        for (int i = 0; i < m.coupling_steps; i++) {
            m.magnitude[i] = static_cast<int>(br.read(bits));
            m.angle[i] = static_cast<int>(br.read(bits));
            if (m.magnitude[i] == m.angle[i] ||
                m.magnitude[i] >= channels || m.angle[i] >= channels)
                return -1;
        }
    } else {
        m.coupling_steps = 0;
    }
    if (br.read(2) != 0) return -1;  // reserved
    m.mux.assign(channels, 0);
    if (m.submaps > 1) {
        for (int ch = 0; ch < channels; ch++) {
            m.mux[ch] = static_cast<int>(br.read(4));
            if (m.mux[ch] >= m.submaps) return -1;
        }
    }
    m.submap_floor.resize(m.submaps);
    m.submap_residue.resize(m.submaps);
    for (int i = 0; i < m.submaps; i++) {
        br.read(8);  // unused time-configuration placeholder
        m.submap_floor[i] = static_cast<int>(br.read(8));
        m.submap_residue[i] = static_cast<int>(br.read(8));
    }
    return br.overrun() ? -1 : 0;
}

int parse_setup(BitReader& br, Setup& s, int channels) {
    // Codebooks.
    int cbcount = static_cast<int>(br.read(8)) + 1;
    s.codebooks.resize(cbcount);
    for (int i = 0; i < cbcount; i++)
        if (read_codebook(br, s.codebooks[i]) != 0) return -1;

    // Time-domain transforms: count, each a 16-bit 0 placeholder.
    int tcount = static_cast<int>(br.read(6)) + 1;
    for (int i = 0; i < tcount; i++)
        if (br.read(16) != 0) return -1;

    // Floors.
    int fcount = static_cast<int>(br.read(6)) + 1;
    s.floors.resize(fcount);
    for (int i = 0; i < fcount; i++) {
        int type = static_cast<int>(br.read(16));
        s.floors[i].type = type;
        if (type == 1) {
            if (parse_floor1(br, s.floors[i].f1) != 0) return -1;
        } else if (type == 0) {
            if (parse_floor0(br, s.floors[i].f0) != 0) return -1;
        } else {
            return -1;
        }
    }

    // Residues.
    int rcount = static_cast<int>(br.read(6)) + 1;
    s.residues.resize(rcount);
    for (int i = 0; i < rcount; i++) {
        int type = static_cast<int>(br.read(16));
        if (type > 2) return -1;
        s.residues[i].type = type;
        if (parse_residue(br, s.residues[i]) != 0) return -1;
    }

    // Mappings.
    int mcount = static_cast<int>(br.read(6)) + 1;
    s.mappings.resize(mcount);
    for (int i = 0; i < mcount; i++) {
        if (br.read(16) != 0) return -1;  // mapping type must be 0
        if (parse_mapping(br, s.mappings[i], channels) != 0) return -1;
    }

    // Modes.
    int modecount = static_cast<int>(br.read(6)) + 1;
    s.modes.resize(modecount);
    for (int i = 0; i < modecount; i++) {
        s.modes[i].blockflag = br.read_bit();
        s.modes[i].windowtype = static_cast<int>(br.read(16));
        s.modes[i].transformtype = static_cast<int>(br.read(16));
        s.modes[i].mapping = static_cast<int>(br.read(8));
        if (s.modes[i].windowtype != 0 || s.modes[i].transformtype != 0)
            return -1;
        if (s.modes[i].mapping >= mcount) return -1;
    }

    if (br.read_bit() != 1) return -1;  // framing bit
    if (br.overrun()) return -1;

    // Validate every cross-reference the audio decoder will index blindly:
    // a malformed setup header must be rejected here, not read out of bounds.
    int ncb = static_cast<int>(s.codebooks.size());
    int nfl = static_cast<int>(s.floors.size());
    int nrs = static_cast<int>(s.residues.size());
    for (const Floor& f : s.floors) {
        if (f.type == 1) {
            for (size_t j = 0; j < f.f1.class_masterbooks.size(); j++) {
                if (f.f1.class_subclasses[j] > 0 &&
                    (f.f1.class_masterbooks[j] < 0 ||
                     f.f1.class_masterbooks[j] >= ncb))
                    return -1;
                for (int book : f.f1.subclass_books[j])
                    if (book >= ncb) return -1;
            }
        } else if (f.type == 0) {
            for (int book : f.f0.books)
                if (book < 0 || book >= ncb) return -1;
        }
    }
    for (const Residue& r : s.residues) {
        if (r.classbook < 0 || r.classbook >= ncb) return -1;
        for (const auto& row : r.books)
            for (int book : row)
                if (book >= ncb) return -1;
    }
    for (const Mapping& m : s.mappings) {
        for (int fl : m.submap_floor)
            if (fl < 0 || fl >= nfl) return -1;
        for (int rs : m.submap_residue)
            if (rs < 0 || rs >= nrs) return -1;
    }
    return 0;
}

}  // namespace

int debug_parse_headers(const uint8_t* ogg, size_t len, int* channels,
                        int* rate, size_t* setup_bits_used,
                        size_t* setup_bits_total) {
    std::vector<std::vector<uint8_t>> packets;
    int64_t g = -1;
    if (ogg_demux_first_stream(ogg, len, packets, &g) != 0 ||
        packets.size() < 3)
        return -1;
    IdHeader id = parse_id_header(packets[0].data(), packets[0].size());
    if (!id.valid) return -1;
    const auto& sp = packets[2];
    if (sp.size() < 7 || sp[0] != 0x05 ||
        std::memcmp(sp.data() + 1, "vorbis", 6) != 0)
        return -1;
    BitReader br(sp.data() + 7, sp.size() - 7);
    Setup s;
    s.id = id;
    if (parse_setup(br, s, id.channels) != 0) return -1;
    if (channels) *channels = id.channels;
    if (rate) *rate = static_cast<int>(id.sample_rate);
    if (setup_bits_used) *setup_bits_used = br.bits_read();
    if (setup_bits_total) *setup_bits_total = br.bit_length();
    return 0;
}

// ---------------------------------------------------------------------------
// Top-level Ogg-Vorbis decode (audio path lands in a later slice)
// ---------------------------------------------------------------------------

namespace {

// floor1_inverse_dB_table (spec §10.1): a 256-entry geometric sequence
// table[i] = base * ratio^i (a pure dB->linear map). Reproduced from the
// spec's first two literal values; every entry then follows exactly.
const float* floor1_inverse_db_table() {
    static float t[256];
    static bool once = [] {
        const double base = 1.0649863e-07;   // spec table[0]
        const double next = 1.1341951e-07;   // spec table[1]
        const double ratio = next / base;
        for (int i = 0; i < 256; i++)
            t[i] = static_cast<float>(base * std::pow(ratio, i));
        return true;
    }();
    (void)once;
    return t;
}

// Integer line-height at position x on the segment (x0,y0)-(x1,y1) (§9.2.4).
int render_point(int x0, int y0, int x1, int y1, int x) {
    int dy = y1 - y0;
    int adx = x1 - x0;
    int ady = std::abs(dy);
    int err = ady * (x - x0);
    int off = adx ? err / adx : 0;
    return dy < 0 ? y0 - off : y0 + off;
}

// Render the line (x0,y0)-(x1,y1) into out[x0..x1) as integer heights (§9.2.6).
void render_line(int x0, int y0, int x1, int y1, int* out, int cap) {
    int dy = y1 - y0;
    int adx = x1 - x0;
    if (adx <= 0) return;
    int ady = std::abs(dy);
    int base = dy / adx;
    int sy = (dy < 0) ? base - 1 : base + 1;
    int ad = ady - std::abs(base) * adx;
    int y = y0;
    int err = 0;
    if (x0 >= 0 && x0 < cap) out[x0] = y;
    for (int x = x0 + 1; x < x1; x++) {
        err += ad;
        if (err >= adx) { err -= adx; y += sy; }
        else { y += base; }
        if (x >= 0 && x < cap) out[x] = y;
    }
}

// The floor-1 "range" per multiplier (spec §7.2.3, Y-value dynamic range).
const int kFloor1Range[4] = { 256, 128, 86, 64 };

// Full audio decoder over one logical Vorbis stream.
class AudioDecoder {
public:
    explicit AudioDecoder(Setup& s) : s_(s) {
        n0_ = s.id.blocksize0;
        n1_ = s.id.blocksize1;
        ch_ = s.id.channels;
        lap_.assign(ch_, {});
        lap_n_.assign(ch_, 0);
        started_.assign(ch_, 0);
        imdct0_.init(n0_);
        imdct1_.init(n1_);
    }

    // Decode one audio packet, appending finished interleaved samples.
    // Returns 0 on success (or a skippable packet), negative on a hard error.
    int decode_packet(const uint8_t* pkt, size_t len,
                      std::vector<std::vector<float>>& out /*[ch]*/) {
        BitReader br(pkt, len);
        if (br.read_bit() != 0) return 0;  // not an audio packet; skip
        int mode_bits = ilog(static_cast<int>(s_.modes.size()) - 1);
        int mode_no = static_cast<int>(br.read(mode_bits));
        if (mode_no < 0 || mode_no >= static_cast<int>(s_.modes.size()))
            return -1;
        const Mode& mode = s_.modes[mode_no];
        int blockflag = mode.blockflag;
        int n = blockflag ? n1_ : n0_;
        int half = n / 2;
        int prev_flag = 0, next_flag = 0;
        if (blockflag) {
            prev_flag = br.read_bit();
            next_flag = br.read_bit();
        }
        const Mapping& map = s_.mappings[mode.mapping];

        // Per-channel spectra (residue) buffers, half entries each.
        std::vector<std::vector<float>> spec(ch_,
                                             std::vector<float>(half, 0.f));
        std::vector<std::vector<float>> floor_curve(ch_);
        std::vector<char> no_residue(ch_, 0);

        // 1) Floor decode for every channel.
        for (int c = 0; c < ch_; c++) {
            int fnum = map.submap_floor[map.mux[c]];
            const Floor& fl = s_.floors[fnum];
            floor_curve[c].assign(half, 0.f);
            int used;
            if (fl.type == 1) {
                used = decode_floor1(br, fl.f1, half, floor_curve[c].data());
            } else {  // floor 0 (LSP)
                used = decode_floor0(br, fl.f0, half, floor_curve[c].data());
            }
            if (used < 0) return -1;
            no_residue[c] = used ? 0 : 1;
        }
        if (br.overrun()) return -1;

        // 2) Propagate residue requirement across coupling pairs.
        for (int i = 0; i < map.coupling_steps; i++) {
            if (!no_residue[map.magnitude[i]] || !no_residue[map.angle[i]]) {
                no_residue[map.magnitude[i]] = 0;
                no_residue[map.angle[i]] = 0;
            }
        }

        // 3) Residue decode per submap.
        for (int sm = 0; sm < map.submaps; sm++) {
            std::vector<int> chans;
            for (int c = 0; c < ch_; c++)
                if (map.mux[c] == sm) chans.push_back(c);
            if (chans.empty()) continue;
            std::vector<char> dnd(chans.size());
            std::vector<float*> vecs(chans.size());
            for (size_t j = 0; j < chans.size(); j++) {
                dnd[j] = no_residue[chans[j]];
                vecs[j] = spec[chans[j]].data();
            }
            const Residue& r = s_.residues[map.submap_residue[sm]];
            if (decode_residue(br, r, half, chans.size(), dnd.data(),
                               vecs.data()) != 0)
                return -1;
        }
        if (br.overrun()) return -1;

        // 4) Inverse coupling (last step to first).
        for (int i = map.coupling_steps - 1; i >= 0; i--) {
            float* m = spec[map.magnitude[i]].data();
            float* a = spec[map.angle[i]].data();
            for (int k = 0; k < half; k++) {
                float mv = m[k], av = a[k], newM, newA;
                if (mv > 0) {
                    if (av > 0) { newM = mv; newA = mv - av; }
                    else { newA = mv; newM = mv + av; }
                } else {
                    if (av > 0) { newM = mv; newA = mv + av; }
                    else { newA = mv; newM = mv - av; }
                }
                m[k] = newM;
                a[k] = newA;
            }
        }

        // 5) floor * residue -> spectrum, then iMDCT + window + overlap-add.
        int left_start, left_end, right_start, right_end;
        window_bounds(blockflag, prev_flag, next_flag, n, &left_start,
                      &left_end, &right_start, &right_end);
        std::vector<float> win(n);
        make_window(win.data(), n, left_start, left_end, right_start,
                    right_end);

        for (int c = 0; c < ch_; c++) {
            if (no_residue[c]) {
                for (int k = 0; k < half; k++) spec[c][k] = 0.f;
            } else {
                for (int k = 0; k < half; k++)
                    spec[c][k] *= floor_curve[c][k];
            }
            std::vector<float> time(n);
            (n == n1_ ? imdct1_ : imdct0_).backward(spec[c].data(),
                                                    time.data());
            for (int i = 0; i < n; i++) time[i] *= win[i];
            overlap_add(c, time.data(), n, left_start, right_start, out[c]);
        }
        return 0;
    }

private:
    Setup& s_;
    int n0_ = 0, n1_ = 0, ch_ = 0;
    std::vector<std::vector<float>> lap_;  // carry buffer per channel
    std::vector<int> lap_n_;               // valid length of lap_[c]
    std::vector<char> started_;            // first-block-seen flag per ch
    FastImdct imdct0_, imdct1_;            // short / long block transforms

    // Floor 0 (LSP) decode + curve synthesis (spec §6.2.2/§6.2.3). Returns 1
    // (used) with the linear curve in `curve[0..half)`, 0 (unused), or -1.
    int decode_floor0(BitReader& br, const Floor0& f, int half, float* curve) {
        int amplitude = static_cast<int>(br.read(f.amplitude_bits));
        if (amplitude <= 0) return 0;  // unused this frame
        int nbooks = static_cast<int>(f.books.size());
        int booknumber = static_cast<int>(br.read(ilog(nbooks)));
        if (booknumber >= nbooks) return -1;
        int bk = f.books[booknumber];
        if (bk < 0 || bk >= static_cast<int>(s_.codebooks.size())) return -1;
        const Codebook& cb = s_.codebooks[bk];
        if (cb.dimensions <= 0 || cb.vq.empty()) return -1;
        std::vector<float> coef;
        std::vector<float> vec(cb.dimensions);
        double last = 0.0;
        while (static_cast<int>(coef.size()) < f.order) {
            if (cb.decode_vector(br, vec.data()) < 0) return -1;
            for (int k = 0; k < cb.dimensions; k++)
                coef.push_back(static_cast<float>(vec[k] + last));
            last = coef.back();
        }
        if (br.overrun()) return -1;
        floor0_curve(f.order, coef.data(), amplitude, f.amplitude_bits,
                     f.amplitude_offset, f.rate, f.bark_map_size, half,
                     curve);
        return 1;
    }

    // Floor 1 decode + curve synthesis. Returns 1 (used) with the linear
    // curve in `curve[0..half)`, 0 (unused), or -1 (error).
    int decode_floor1(BitReader& br, const Floor1& f, int half, float* curve) {
        if (br.read_bit() == 0) return 0;  // unused this frame
        int range = kFloor1Range[f.multiplier - 1];
        int ybits = ilog(range - 1);
        std::vector<int> Y(f.values, 0);
        Y[0] = static_cast<int>(br.read(ybits));
        Y[1] = static_cast<int>(br.read(ybits));
        int off = 2;
        for (int i = 0; i < f.partitions; i++) {
            int cls = f.partition_class[i];
            int cdim = f.class_dimensions[cls];
            int cbits = f.class_subclasses[cls];
            int csub = (1 << cbits) - 1;
            int cval = 0;
            if (cbits > 0) {
                cval = s_.codebooks[f.class_masterbooks[cls]].decode_scalar(br);
                if (cval < 0) return -1;
            }
            for (int j = 0; j < cdim; j++) {
                int book = f.subclass_books[cls][cval & csub];
                cval >>= cbits;
                if (book >= 0) {
                    Y[off + j] = s_.codebooks[book].decode_scalar(br);
                    if (Y[off + j] < 0) return -1;
                } else {
                    Y[off + j] = 0;
                }
            }
            off += cdim;
        }
        if (br.overrun()) return -1;

        // Amplitude prediction (spec §7.2.4 step 1).
        std::vector<int> finalY(f.values);
        std::vector<char> step2(f.values, 0);
        step2[0] = step2[1] = 1;
        finalY[0] = Y[0];
        finalY[1] = Y[1];
        for (int i = 2; i < f.values; i++) {
            int lo = f.low_neighbour[i], hi = f.high_neighbour[i];
            int pred = render_point(f.x_list[lo], finalY[lo], f.x_list[hi],
                                    finalY[hi], f.x_list[i]);
            int val = Y[i];
            int highroom = range - pred;
            int lowroom = pred;
            int room = (highroom < lowroom ? highroom : lowroom) * 2;
            if (val == 0) {
                step2[i] = 0;
                finalY[i] = pred;
            } else {
                step2[lo] = 1;
                step2[hi] = 1;
                step2[i] = 1;
                if (val >= room) {
                    if (highroom > lowroom)
                        finalY[i] = val - lowroom + pred;
                    else
                        finalY[i] = pred - val + highroom - 1;
                } else {
                    if (val & 1)
                        finalY[i] = pred - (val + 1) / 2;
                    else
                        finalY[i] = pred + val / 2;
                }
            }
            if (finalY[i] < 0) finalY[i] = 0;
            if (finalY[i] >= range) finalY[i] = range - 1;
        }

        // Curve rendering (spec §7.2.4 step 2).
        const float* db = floor1_inverse_db_table();
        std::vector<int> heights(half, 0);
        int hx = 0;
        int lx = 0;
        int ly = finalY[f.sorted_idx[0]] * f.multiplier;
        for (int j = 1; j < f.values; j++) {
            int idx = f.sorted_idx[j];
            if (!step2[idx]) continue;
            int hy = finalY[idx] * f.multiplier;
            hx = f.x_list[idx];
            if (hx > half) hx = half;
            if (lx < hx) render_line(lx, ly, hx, hy, heights.data(), half);
            lx = hx;
            ly = hy;
            if (lx >= half) break;
        }
        for (int x = lx; x < half; x++) heights[x] = ly;  // flat tail
        for (int x = 0; x < half; x++) {
            int v = heights[x];
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            curve[x] = db[v];
        }
        return 1;
    }

    // Residue decode (spec §8.6). Writes accumulated values into vecs[j][k].
    int decode_residue(BitReader& br, const Residue& r, int half, int nch,
                       char* dnd, float** vecs) {
        // For type 2, all channels are interleaved into one working vector.
        if (r.type == 2) {
            bool all_dnd = true;
            for (int j = 0; j < nch; j++)
                if (!dnd[j]) all_dnd = false;
            if (all_dnd) return 0;
            int total = half * nch;
            std::vector<float> combined(total, 0.f);
            float* one = combined.data();
            char zero = 0;
            if (residue_pass(br, r, total, 1, &zero, &one) != 0) return -1;
            for (int i = 0; i < total; i++)
                vecs[i % nch][i / nch] += combined[i];
            return 0;
        }
        return residue_pass(br, r, half, nch, dnd, vecs);
    }

    // The shared type-0/1 classification+partition loop over `size` values.
    int residue_pass(BitReader& br, const Residue& r, int size, int nch,
                     char* dnd, float** vecs) {
        int begin = r.begin, end = r.end;
        if (end > size) end = size;
        if (begin > end) begin = end;
        int n_to_read = end - begin;
        if (n_to_read <= 0) return 0;
        const Codebook& classbook = s_.codebooks[r.classbook];
        int cpw = classbook.dimensions;  // classwords per codeword
        if (cpw <= 0) return -1;
        int parts = n_to_read / r.partition_size;
        std::vector<std::vector<int>> classif(
            nch, std::vector<int>(parts, 0));

        for (int pass = 0; pass < 8; pass++) {
            int pc = 0;
            while (pc < parts) {
                if (pass == 0) {
                    for (int j = 0; j < nch; j++) {
                        if (dnd[j]) continue;
                        int temp = classbook.decode_scalar(br);
                        if (temp < 0) return -1;
                        for (int i = cpw - 1; i >= 0; i--) {
                            if (pc + i < parts)
                                classif[j][pc + i] = temp % r.classifications;
                            temp /= r.classifications;
                        }
                    }
                }
                for (int i = 0; i < cpw && pc < parts; i++) {
                    for (int j = 0; j < nch; j++) {
                        if (dnd[j]) continue;
                        int vqclass = classif[j][pc];
                        int vqbook = r.books[vqclass][pass];
                        if (vqbook >= 0) {
                            if (decode_partition(
                                    br, s_.codebooks[vqbook], r.type,
                                    vecs[j], begin + pc * r.partition_size,
                                    r.partition_size) != 0)
                                return -1;
                        }
                    }
                    pc++;
                }
            }
        }
        return 0;
    }

    // Decode one residue partition of `psize` values into out[offset..].
    int decode_partition(BitReader& br, const Codebook& cb, int type,
                         float* out, int offset, int psize) {
        int dim = cb.dimensions;
        if (dim <= 0) return -1;
        if (type == 0) {
            int step = psize / dim;
            std::vector<float> v(dim);
            for (int s = 0; s < step; s++) {
                if (cb.decode_vector(br, v.data()) < 0) return -1;
                for (int k = 0; k < dim; k++)
                    out[offset + s + k * step] += v[k];
            }
        } else {  // type 1 and 2 (combined vector) are concatenated
            std::vector<float> v(dim);
            int i = 0;
            while (i < psize) {
                if (cb.decode_vector(br, v.data()) < 0) return -1;
                for (int k = 0; k < dim; k++) out[offset + i + k] += v[k];
                i += dim;
            }
        }
        return 0;
    }

    // Window boundary indices for a block (spec §4.3.1).
    void window_bounds(int blockflag, int prev_flag, int next_flag, int n,
                       int* ls, int* le, int* rs, int* re) {
        if (!blockflag) {
            *ls = 0; *le = n / 2; *rs = n / 2; *re = n;
            return;
        }
        if (prev_flag) { *ls = 0; *le = n / 2; }
        else { *ls = n / 4 - n0_ / 4; *le = n / 4 + n0_ / 4; }
        if (next_flag) { *rs = n / 2; *re = n; }
        else { *rs = 3 * n / 4 - n0_ / 4; *re = 3 * n / 4 + n0_ / 4; }
    }

    void make_window(float* w, int n, int ls, int le, int rs, int re) {
        const double H = 1.5707963267948966;  // pi/2
        for (int i = 0; i < ls; i++) w[i] = 0.f;
        int ln = le - ls;
        for (int i = ls; i < le; i++) {
            double t = (i - ls + 0.5) / ln;
            double s = std::sin(H * t);
            w[i] = static_cast<float>(std::sin(H * s * s));
        }
        for (int i = le; i < rs; i++) w[i] = 1.f;
        int rn = re - rs;
        for (int i = rs; i < re; i++) {
            double t = (re - i - 0.5) / rn;  // mirror of the left ramp
            double s = std::sin(H * t);
            w[i] = static_cast<float>(std::sin(H * s * s));
        }
        for (int i = re; i < n; i++) w[i] = 0.f;
    }

    // Windowed overlap-add (spec §4.3.8). `lap_[c]` is the carry buffer whose
    // index 0 is the current finalized boundary (this block's/previous block's
    // right-ramp start). The new block's samples land at carry index
    // (i - left_start): its left ramp aligns with the carry's overlap region.
    // Samples up to this block's right-ramp start then become final.
    void overlap_add(int c, const float* time, int n, int left_start,
                     int right_start, std::vector<float>& dst) {
        std::vector<float>& lap = lap_[c];
        if (!started_[c]) {
            // First block: prime the carry with [right_start, n); output none.
            started_[c] = true;
            lap.assign(time + right_start, time + n);
            lap_n_[c] = static_cast<int>(lap.size());
            return;
        }
        int span = n - left_start;
        if (static_cast<int>(lap.size()) < span) lap.resize(span, 0.f);
        for (int i = left_start; i < n; i++) lap[i - left_start] += time[i];
        int fin = right_start - left_start;
        for (int i = 0; i < fin; i++) dst.push_back(lap[i]);
        std::vector<float> tail(lap.begin() + fin, lap.end());
        lap.swap(tail);
        lap_n_[c] = static_cast<int>(lap.size());
    }
};

}  // namespace

int decode_ogg(const uint8_t* ogg, size_t len, std::vector<float>& pcm,
               int& sample_rate, int& channels) {
    pcm.clear();
    sample_rate = 0;
    channels = 0;
    if (!ogg || len < 27) return -1;

    std::vector<std::vector<uint8_t>> packets;
    int64_t last_granule = -1;
    if (ogg_demux_first_stream(ogg, len, packets, &last_granule) != 0)
        return -1;
    if (packets.size() < 3) return -1;  // need the 3 header packets

    IdHeader id = parse_id_header(packets[0].data(), packets[0].size());
    if (!id.valid) return -1;
    if (packets[1].size() < 7 || packets[1][0] != 0x03 ||
        std::memcmp(packets[1].data() + 1, "vorbis", 6) != 0)
        return -1;
    const auto& sp = packets[2];
    if (sp.size() < 7 || sp[0] != 0x05 ||
        std::memcmp(sp.data() + 1, "vorbis", 6) != 0)
        return -1;

    Setup setup;
    setup.id = id;
    BitReader sbr(sp.data() + 7, sp.size() - 7);
    if (parse_setup(sbr, setup, id.channels) != 0) return -1;

    sample_rate = static_cast<int>(id.sample_rate);
    channels = id.channels;

    AudioDecoder dec(setup);
    std::vector<std::vector<float>> per_ch(id.channels);
    for (size_t p = 3; p < packets.size(); p++) {
        if (dec.decode_packet(packets[p].data(), packets[p].size(),
                              per_ch) != 0)
            return -1;
    }

    // Interleave. Trim to the stream's declared length (last granulepos).
    size_t frames = per_ch.empty() ? 0 : per_ch[0].size();
    for (int c = 1; c < id.channels; c++)
        frames = std::min(frames, per_ch[c].size());
    if (last_granule >= 0 && static_cast<size_t>(last_granule) < frames)
        frames = static_cast<size_t>(last_granule);
    pcm.resize(frames * id.channels);
    for (size_t i = 0; i < frames; i++)
        for (int c = 0; c < id.channels; c++)
            pcm[i * id.channels + c] = per_ch[c][i];
    return 0;
}

}  // namespace vorbis
}  // namespace glint
