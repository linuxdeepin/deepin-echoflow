// test-lid-probe.cpp — hermetic tests for probe-based LID scoring.
// Fixed strings, no model, no audio.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/lid_probe.h"

using namespace core_lid_probe;

TEST_CASE("lid_probe: utf8 length counts codepoints, not bytes", "[unit][lid]") {
    REQUIRE(utf8_length("hello") == 5);
    REQUIRE(utf8_length("مرحبا") == 5);    // Arabic: 2 bytes/char
    REQUIRE(utf8_length("你好世界") == 4); // CJK: 3 bytes/char
    REQUIRE(utf8_length("") == 0);
    // The whole point: the same visible length must not score differently by
    // script. "hello" is 5 bytes, "مرحبا" is 10 — byte length would double it.
    REQUIRE(std::string("مرحبا").size() == 10);
}

// Helper: is every byte sequence in `s` valid UTF-8? Deliberately strict —
// that is exactly what a downstream text decoder does.
static bool is_valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        size_t n;
        if ((c & 0x80) == 0x00)
            n = 1;
        else if ((c & 0xE0) == 0xC0)
            n = 2;
        else if ((c & 0xF0) == 0xE0)
            n = 3;
        else if ((c & 0xF8) == 0xF0)
            n = 4;
        else
            return false;
        if (i + n > s.size())
            return false;
        for (size_t k = 1; k < n; k++)
            if (((unsigned char)s[i + k] & 0xC0) != 0x80)
                return false;
        i += n;
    }
    return true;
}

TEST_CASE("lid_probe: truncating a probe log never emits invalid UTF-8", "[unit][lid]") {
    // Regression. The probe logged its per-candidate transcript with "%.60s",
    // which cuts at 60 BYTES. For Greek/Arabic/CJK that lands mid-character
    // and puts a severed lead byte on stderr. Not cosmetic: it killed a Kaggle
    // run when Python decoded our stderr (UnicodeDecodeError on 0xce, Greek).
    // Only non-Latin languages could trigger it — which is why local Latin
    // testing never saw it.
    const std::string greek = "Και γι' αυτό, αδελφέ Αμερικανέ, μη ρωτάτε τι μπορεί να κάνει";
    const std::string arabic = "العاصفة شبه الاستوائية جيري تغادر الحافلات المحطة الداخلية";
    const std::string cjk = "そして、私の仲間のアメリカ人よ、あなたの国があなたのために";

    for (const auto& s : {greek, arabic, cjk}) {
        REQUIRE(is_valid_utf8(s)); // the fixtures themselves are sane
        for (size_t n = 1; n <= s.size() + 4; n++) {
            const std::string cut = utf8_prefix(s, n);
            REQUIRE(cut.size() <= s.size());
            REQUIRE(is_valid_utf8(cut));
            REQUIRE(s.compare(0, cut.size(), cut) == 0); // a true prefix
        }
    }
    // ASCII and short strings pass through untouched.
    REQUIRE(utf8_prefix("hello", 60) == "hello");
    REQUIRE(utf8_prefix("", 10).empty());
}

TEST_CASE("lid_probe: no-space languages tokenise by character", "[unit][lid]") {
    REQUIRE(is_no_space_language("zh"));
    REQUIRE(is_no_space_language("ja"));
    REQUIRE(is_no_space_language("ko"));
    REQUIRE_FALSE(is_no_space_language("en"));
    REQUIRE_FALSE(is_no_space_language("ar"));

    // Chinese has no spaces: whitespace tokenisation would see ONE token and
    // report perfect diversity for any string at all.
    REQUIRE(diversity("你好你好", "zh") == 0.5); // 4 chars, 2 distinct
    REQUIRE(diversity("你好世界", "zh") == 1.0);
    REQUIRE(diversity("你好你好", "en") == 1.0); // as English: one token, no repetition seen
}

TEST_CASE("lid_probe: diversity flags repetition", "[unit][lid]") {
    REQUIRE(diversity("", "en") == 0.0);
    REQUIRE(diversity("the quick brown fox", "en") == 1.0);
    REQUIRE(diversity("the the the the", "en") == 0.25);
    REQUIRE(diversity("a b a b", "en") == 0.5);
}

TEST_CASE("lid_probe: a repetitive hallucination loses to a shorter clean decode", "[unit][lid]") {
    // The failure mode this scoring exists for: a wrong-language prompt does
    // not error, it produces long fluent-looking repetition. Length alone
    // would pick it; diversity squared must not.
    const std::string hallucinated = "the the the the the the the the the the the the";
    const std::string clean = "and now the weather";

    const double bad = score(hallucinated, "en", 0.0);
    const double good = score(clean, "en", 0.0);
    REQUIRE(good > bad);
}

TEST_CASE("lid_probe: text-LID agreement boosts, up to 4x", "[unit][lid]") {
    const std::string t = "and now the weather";
    const double none = score(t, "en", 0.0);
    const double half = score(t, "en", 0.5);
    const double full = score(t, "en", 1.0);

    REQUIRE(half > none);
    REQUIRE(full > half);
    REQUIRE(full == Catch::Approx(none * 4.0));

    // Out-of-range agreement is clamped, never allowed to run away.
    REQUIRE(score(t, "en", 5.0) == Catch::Approx(full));
    REQUIRE(score(t, "en", -1.0) == Catch::Approx(none));
}

TEST_CASE("lid_probe: an empty probe scores zero", "[unit][lid]") {
    // A candidate the model refuses to transcribe must never win, no matter
    // how confident a text detector is about the empty string.
    REQUIRE(score("", "en", 1.0) == 0.0);
}

TEST_CASE("lid_probe: a fluent translation CAN outscore the truth", "[unit][lid]") {
    // The scoring's one soft spot, with the measured numbers behind it.
    //
    // Asking Cohere Transcribe for a language it was NOT trained on can produce
    // a clean translation rather than garbage, and a text LID then confirms it
    // at 1.00 — so "the output is fluent language X" is not evidence that the
    // AUDIO is X. Pair that with a source line that is legitimately repetitive
    // (JFK's chiasmus, "ask not what your country can do for you, ask what you
    // can do for your country") and diversity^2 penalises the correct answer.
    //
    // Measured by forcing a 14-language list onto the TWO-language Arabic
    // finetune: its 'fr' probe returned real French at div 0.88 and beat the
    // correct 'en' at div 0.73.
    const double fr = score_from(108, 1.00, 0.88);
    const double en = score_from(108, 1.00, 0.73);
    REQUIRE(fr > en); // 336 vs 228

    // ⚠ But do NOT read that as "the probe degrades at 14 candidates". On the
    // REAL 14-language base model (cohere-transcribe-q4_k, republished with its
    // whitelist) the forced 14-way probe is correct on both clips: its 'fr'
    // probe does not translate, it code-switches ("Et so, my fellow
    // Americans…", agree 0.00, score 57). The earlier failure was an artifact
    // of demanding languages the model does not have.
    REQUIRE(score_from(108, 1.00, 0.73) > score_from(107, 0.00, 0.73)); // jfk: en 228 > fr 57
    REQUIRE(score_from(73, 1.00, 1.00) > score_from(91, 1.00, 0.50));   // arabic: ar 292 > en 91

    // Two-language model, both directions.
    REQUIRE(score_from(82, 1.00, 1.00) > score_from(64, 1.00, 0.79));  // ar 328 > en 158
    REQUIRE(score_from(108, 1.00, 0.73) > score_from(59, 1.00, 0.73)); // en 228 > ar 125
}

TEST_CASE("lid_probe: agreement can overturn a length advantage", "[unit][lid]") {
    // Same script, both plausible: 'de' output is shorter but the text
    // detector confirms it, while the 'en' output is longer and unconfirmed.
    const double de = score("guten morgen liebe leser", "de", 0.99);
    const double en = score("good morning dear readers today we", "en", 0.0);
    REQUIRE(de > en);
}
