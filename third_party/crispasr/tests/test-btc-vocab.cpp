// test-btc-vocab.cpp — BTC chord vocabulary + positional encoding.
//
// Hermetic: no model, no network, no GPU. These cover the parts of the chord
// path where a mistake is INVISIBLE to every numeric check we have. The
// per-stage diff harness compares logits, so a wrong entry in the name table
// or the maj/min collapse passes it at cos 1.000000 and still ships the wrong
// chord to the user. Nothing but a table test catches that.
//
// The positional-encoding tests exist for the same reason: sin/cos halves that
// are interleaved rather than concatenated produce a plausible-looking encoding
// that quietly degrades accuracy. BTC_BLUEPRINT.md lists it as one of the ten
// details that are not the obvious default.

#include "btc_chord_vocab.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <algorithm>
#include <set>
#include <string>
#include <vector>

TEST_CASE("btc vocab: 170-class names are unique and well-formed", "[unit][btc][vocab]") {
    std::set<std::string> seen;
    for (int i = 0; i < 170; i++) {
        const std::string n = btc_vocab::voca_name(i);
        INFO("index " << i << " -> '" << n << "'");
        REQUIRE(!n.empty());
        // Every label must be distinct: a duplicate means two different chords
        // print identically and are indistinguishable downstream.
        REQUIRE(seen.insert(n).second);
    }
    REQUIRE(seen.size() == 170);
}

TEST_CASE("btc vocab: 170-class index layout is root*14 + quality", "[unit][btc][vocab]") {
    // Spot-check the corners of the layout rather than restating the formula.
    REQUIRE(btc_vocab::voca_name(0) == "C:min");   // root 0, quality 0
    REQUIRE(btc_vocab::voca_name(1) == "C");       // quality 1 = maj renders bare
    REQUIRE(btc_vocab::voca_name(2) == "C:dim");   // root 0, quality 2
    REQUIRE(btc_vocab::voca_name(14) == "C#:min"); // root 1, quality 0
    REQUIRE(btc_vocab::voca_name(15) == "C#");
    REQUIRE(btc_vocab::voca_name(167) == "B:sus4"); // root 11, quality 13
    REQUIRE(btc_vocab::voca_name(168) == "X");      // unknown
    REQUIRE(btc_vocab::voca_name(169) == "N");      // no chord

    // Quality "maj" must never be spelled out -- it renders as the bare root.
    // (":maj6" and ":maj7" are different qualities and ARE legitimate, so this
    // has to compare the whole suffix, not search for a substring.)
    for (int i = 0; i < 168; i++) {
        const std::string n = btc_vocab::voca_name(i);
        INFO("index " << i << " -> '" << n << "'");
        REQUIRE(!(n.size() > 4 && n.compare(n.size() - 4, 4, ":maj") == 0));
    }
    for (int root = 0; root < 12; root++)
        REQUIRE(btc_vocab::voca_name(root * 14 + 1) == std::string(btc_vocab::roots()[root]));
}

TEST_CASE("btc vocab: 25-class names alternate major/minor and end with N", "[unit][btc][vocab]") {
    std::set<std::string> seen;
    for (int i = 0; i < 25; i++) {
        const std::string n = btc_vocab::maj_min_name(i);
        INFO("index " << i << " -> '" << n << "'");
        REQUIRE(!n.empty());
        REQUIRE(seen.insert(n).second);
    }
    REQUIRE(btc_vocab::maj_min_name(0) == "C");
    REQUIRE(btc_vocab::maj_min_name(1) == "C:min");
    REQUIRE(btc_vocab::maj_min_name(2) == "C#");
    REQUIRE(btc_vocab::maj_min_name(23) == "B:min");
    REQUIRE(btc_vocab::maj_min_name(24) == "N");
    // Anything past the table is N, not a crash or an empty string.
    REQUIRE(btc_vocab::maj_min_name(25) == "N");
    REQUIRE(btc_vocab::maj_min_name(1000) == "N");
}

TEST_CASE("btc vocab: maj/min collapse lands in range and keeps the root", "[unit][btc][vocab]") {
    for (int i = 0; i < 170; i++) {
        const int m = btc_vocab::voca_to_maj_min(i);
        INFO("170-class " << i << " ('" << btc_vocab::voca_name(i) << "') -> 25-class " << m << " ('"
                          << btc_vocab::maj_min_name(m) << "')");
        REQUIRE(m >= 0);
        REQUIRE(m <= 24);

        if (i >= 168) {
            REQUIRE(m == 24); // X and N both collapse to N
            continue;
        }
        // The ROOT must survive the collapse: only the quality is discarded.
        const int root = i / 14;
        REQUIRE(m / 2 == root);
        const std::string collapsed = btc_vocab::maj_min_name(m);
        REQUIRE(collapsed.rfind(btc_vocab::roots()[root], 0) == 0);
    }
}

TEST_CASE("btc vocab: minor-third qualities collapse to :min, the rest to major", "[unit][btc][vocab]") {
    // Names the convention explicitly so a change to kIsMinor has to be
    // deliberate. Qualities carrying a minor third:
    const std::vector<std::string> minor_q = {"min", "dim", "min6", "min7", "minmaj7", "dim7", "hdim7"};
    // ...and those that do not. aug/sus2/sus4 have no third at all, so their
    // placement is mir_eval convention rather than music theory.
    const std::vector<std::string> major_q = {"maj", "aug", "maj6", "maj7", "7", "sus2", "sus4"};
    REQUIRE(minor_q.size() + major_q.size() == 14);

    for (int q = 0; q < 14; q++) {
        const std::string name = btc_vocab::qualities()[q];
        const bool is_minor = btc_vocab::quality_is_minor()[q];
        const bool listed_minor = std::find(minor_q.begin(), minor_q.end(), name) != minor_q.end();
        const bool listed_major = std::find(major_q.begin(), major_q.end(), name) != major_q.end();
        INFO("quality " << q << " '" << name << "' is_minor=" << is_minor);
        REQUIRE((listed_minor || listed_major));
        REQUIRE(is_minor == listed_minor);

        // And the collapse must agree, checked through the real function on C.
        const int m = btc_vocab::voca_to_maj_min(q); // root 0 = C
        REQUIRE(btc_vocab::maj_min_name(m) == (is_minor ? "C:min" : "C"));
    }
}

TEST_CASE("btc vocab: every 25-class label is reachable from the 170-class set", "[unit][btc][vocab]") {
    // If a 25-class label were unreachable, CRISPASR_BTC_MAJ_MIN=1 could never
    // emit it -- a whole chord would silently disappear from the output.
    std::set<int> reachable;
    for (int i = 0; i < 170; i++)
        reachable.insert(btc_vocab::voca_to_maj_min(i));
    REQUIRE(reachable.size() == 25);
    for (int m = 0; m < 25; m++)
        REQUIRE(reachable.count(m) == 1);
}

// ---------------------------------------------------------------------------
// Pipeline geometry. These pin the numbers MEASURED against the reference
// implementation on its own 257 s test clip, so a regression is caught here
// rather than as a few percent of chord accuracy nobody attributes to the
// front end.
// ---------------------------------------------------------------------------

TEST_CASE("btc geometry: chunked frame count matches the reference pipeline", "[unit][btc][geometry]") {
    const int SR = 22050, HOP = 2048;
    const int chunk = (int)(10.0 * SR); // inst_len = 10 s

    // The upstream test clip: 257.201474 s at 22050 Hz. librosa's own chunked
    // audio_file_to_features emits 2778 frames for it; a CONTINUOUS transform
    // of the same signal emits 2770. Both numbers were measured, not derived.
    const int n = (int)(257.201474 * SR);
    REQUIRE(btc_vocab::chunked_n_frames(n, HOP, chunk) == 2778);
    REQUIRE(btc_vocab::centred_n_frames(n, HOP) == 2770);

    // 2778 decomposes exactly as 25 full chunks x 108 frames + a 78-frame
    // remainder. (The 8-frame gap versus the continuous transform has no tidy
    // closed form -- it falls out of per-chunk centring -- so the decomposition
    // is asserted instead of a formula for the difference.)
    const int full = n / chunk;
    REQUIRE(full == 25);
    REQUIRE(btc_vocab::centred_n_frames(chunk, HOP) == 108);
    REQUIRE(btc_vocab::centred_n_frames(n - full * chunk, HOP) == 78);
    REQUIRE(full * 108 + 78 == 2778);

    // Exactly one chunk of audio gives exactly `timestep` frames -- that
    // equality is what makes feature_per_second = inst_len / timestep hold.
    REQUIRE(btc_vocab::chunked_n_frames(chunk, HOP, chunk) == 108);

    // chunk <= 0 degrades to the continuous transform rather than looping.
    REQUIRE(btc_vocab::chunked_n_frames(n, HOP, 0) == btc_vocab::centred_n_frames(n, HOP));
    REQUIRE(btc_vocab::chunked_n_frames(0, HOP, chunk) == 0);
}

TEST_CASE("btc geometry: frame duration comes from inst_len/timestep", "[unit][btc][geometry]") {
    // 10/108 = 0.0925926, NOT 2048/22050 = 0.0928798. The 0.31 % difference is
    // 0.79 s of drift over a 4-minute song, and fixing it moved agreement with
    // the torch reference from 86.63 % to 98.56 % (mir_eval tetrads).
    const double fs = btc_vocab::frame_seconds(10.0f, 108, 2048, 22050);
    REQUIRE_THAT(fs, Catch::Matchers::WithinRel(10.0 / 108.0, 1e-12));

    const double wrong = 2048.0 / 22050.0;
    REQUIRE(std::abs(fs - wrong) > 1e-5); // they must NOT be interchangeable
    REQUIRE_THAT(std::abs(fs - wrong) * 257.2 / wrong, Catch::Matchers::WithinAbs(0.79, 0.02));

    // Fallback only when the chunk geometry is unavailable.
    REQUIRE_THAT(btc_vocab::frame_seconds(0.0f, 108, 2048, 22050), Catch::Matchers::WithinRel(wrong, 1e-12));
    REQUIRE_THAT(btc_vocab::frame_seconds(10.0f, 0, 2048, 22050), Catch::Matchers::WithinRel(wrong, 1e-12));
}

TEST_CASE("btc positional encoding: halves are concatenated, not interleaved", "[unit][btc][posenc]") {
    const int T = 16, C = 128, n = C / 2;
    std::vector<float> pe;
    btc_vocab::timing_signal(T, C, pe);
    REQUIRE(pe.size() == (size_t)T * (size_t)C);

    // At t = 0 every argument is 0, so the first half is sin(0) = 0 and the
    // second is cos(0) = 1. Interleaved layout would alternate 0,1,0,1 instead
    // of giving two solid blocks -- this is the discriminating test.
    for (int i = 0; i < n; i++) {
        INFO("channel " << i);
        REQUIRE_THAT(pe[(size_t)i], Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        REQUIRE_THAT(pe[(size_t)n + i], Catch::Matchers::WithinAbs(1.0f, 1e-6f));
    }
}

TEST_CASE("btc positional encoding: sin/cos pair is consistent at every position", "[unit][btc][posenc]") {
    const int T = 32, C = 128, n = C / 2;
    std::vector<float> pe;
    btc_vocab::timing_signal(T, C, pe);

    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++) {
            const float s = pe[(size_t)t * C + i];
            const float c = pe[(size_t)t * C + n + i];
            INFO("t " << t << " channel " << i);
            // Same argument feeds both halves, so sin^2 + cos^2 == 1. This
            // fails if the two halves are built from different frequencies --
            // e.g. an off-by-one in the channel index.
            REQUIRE_THAT((double)(s * s + c * c), Catch::Matchers::WithinAbs(1.0, 1e-5));
            REQUIRE(s >= -1.0f);
            REQUIRE(s <= 1.0f);
            REQUIRE(c >= -1.0f);
            REQUIRE(c <= 1.0f);
        }
}

TEST_CASE("btc positional encoding: frequencies decay geometrically", "[unit][btc][posenc]") {
    // Channel 0 is the fastest (inv = 1.0) and channel n-1 the slowest
    // (inv = 1/10000). Verify via t = 1, where sin(scaled) == sin(inv).
    const int T = 2, C = 128, n = C / 2;
    std::vector<float> pe;
    btc_vocab::timing_signal(T, C, pe);

    REQUIRE_THAT((double)pe[(size_t)C + 0], Catch::Matchers::WithinAbs(std::sin(1.0), 1e-5));
    REQUIRE_THAT((double)pe[(size_t)C + (n - 1)], Catch::Matchers::WithinAbs(std::sin(1.0e-4), 1e-6));

    // Monotone decay of the implied frequency across channels.
    double prev = 2.0;
    for (int i = 0; i < n; i++) {
        const double s = pe[(size_t)C + i]; // sin(inv_i), inv_i in (0, 1]
        REQUIRE(s <= prev + 1e-6);
        prev = s;
    }
}

TEST_CASE("btc positional encoding: distinct positions get distinct encodings", "[unit][btc][posenc]") {
    // A degenerate encoding (all rows equal) would make the transformer
    // position-blind while still producing finite, plausible numbers.
    const int T = 108, C = 128; // T = BTC's real block length
    std::vector<float> pe;
    btc_vocab::timing_signal(T, C, pe);

    for (int t = 1; t < T; t++) {
        double diff = 0.0;
        for (int c = 0; c < C; c++)
            diff += std::abs((double)pe[(size_t)t * C + c] - (double)pe[(size_t)(t - 1) * C + c]);
        INFO("t " << t << " L1 distance from previous row " << diff);
        REQUIRE(diff > 1e-3);
    }
}
