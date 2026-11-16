// src/core/powerset.h — the pyannote-seg-3.0 powerset class layout, once.
//
// The segmentation head does not emit one probability per speaker. It emits one
// probability per SUBSET of the (at most 3) speakers that can be locally
// active, which is what lets a single softmax express overlapped speech. The
// subsets are enumerated the way pyannote builds them — itertools.combinations
// over increasing subset size — giving 7 classes:
//
//   0 = {}            silence
//   1 = {0}           spk0 alone
//   2 = {1}           spk1 alone
//   3 = {2}           spk2 alone
//   4 = {0,1}         spk0 and spk1 talking over each other
//   5 = {0,2}
//   6 = {1,2}
//
// ⚠ This ordering used to be written out by hand in more than one place, and
// one copy had 3 and 4 swapped — so every frame of the third local speaker was
// scored as the first two overlapping. It cost ~15 DER points and was invisible
// to any single-speaker test. That is why the layout lives here now and why
// nothing downstream should hard-code a class number: derive it.

#pragma once

#include <cstdint>

namespace core_powerset {

constexpr int kSpeakers = 3;
constexpr int kClasses = 7; // 1 + 3 + 3

// Bitmask of the speakers active in class `c` (bit s => speaker s).
constexpr uint8_t members(int c) {
    return c == 0 ? 0x0 : c == 1 ? 0x1 : c == 2 ? 0x2 : c == 3 ? 0x4 : c == 4 ? 0x3 : c == 5 ? 0x5 : 0x6;
}

// Inverse of members(): the class index for a speaker bitmask.
constexpr int class_of(uint8_t mask) {
    return mask == 0x0   ? 0
           : mask == 0x1 ? 1
           : mask == 0x2 ? 2
           : mask == 0x4 ? 3
           : mask == 0x3 ? 4
           : mask == 0x5 ? 5
           : mask == 0x6 ? 6
                         : -1; // all three at once is not representable
}

// True when speaker `s` is one of the speakers class `c` covers. This is the
// predicate an activity mask wants: an overlap class counts for BOTH of the
// speakers it names, not just the first.
constexpr bool covers(int c, int s) {
    return (members(c) & (uint8_t)(1u << s)) != 0;
}

// Relabel: if local speaker s is really speaker perm[s], class `c` becomes
// this class. Used to reconcile the arbitrary local speaker numbering of two
// forward passes that overlap in time.
constexpr int permute_class(int c, const int perm[kSpeakers]) {
    uint8_t out = 0;
    for (int s = 0; s < kSpeakers; s++)
        if (covers(c, s))
            out = (uint8_t)(out | (1u << perm[s]));
    return class_of(out);
}

// The 6 permutations of 3 speakers, identity first so that ties keep the
// existing numbering rather than shuffling it for nothing.
constexpr int kPerms[6][kSpeakers] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

} // namespace core_powerset
