// test-script-mismatch.cpp — issue #419: the wrong-script decision table.
//
// Hermetic: pure string classification, no model. canary conditioned on the
// wrong language transliterates instead of failing — Russian audio through
// <|en|> gives "vikingi, otvazhnye voyny" — and with the language correctly
// set the output is Cyrillic on every backend. The warning that turns this
// silent failure into a diagnosable one hinges on this table, and a wrong
// threshold either misses the reporter's case or spams every code-switched
// sentence, so it gets its own test.

#include "core/script_mismatch.h"

#include <catch2/catch_test_macros.hpp>

using core_script::count_scripts;
using core_script::mismatch;

TEST_CASE("script-mismatch fires on the #419 shape: ru target, translit output", "[unit][script-mismatch]") {
    // The reporter's literal example plus enough tail to clear the evidence bar.
    CHECK(mismatch("ru", "vikingi, otvazhnye voyny, mastera korablestroyeniya i torgovli"));
    CHECK(mismatch("uk", "kozaky, vidvazhni voyiny, maistry morskoyi spravy ta torhivli"));
    // en-conditioned half-translation (measured output on the ru fixture).
    CHECK(mismatch("ru", "Nichich, not required, praise, Happy Us I'm Hope's Sweet, Posmotrit maybe Ukratka"));
    CHECK(mismatch("el", "kalimera sas, ti kanete simera to proi stin Athina"));
}

TEST_CASE("script-mismatch stays quiet on correct-script output", "[unit][script-mismatch]") {
    CHECK_FALSE(mismatch("ru", "Ничьих не требуя похвал, счастлив уж я надеждой сладкой, что дева с трепетом любви"));
    CHECK_FALSE(mismatch("el", "καλημέρα σας, τι κάνετε σήμερα το πρωί στην Αθήνα, όλα καλά εδώ"));
    // Latin-script languages are never classified — loanwords make any
    // threshold noisy there.
    CHECK_FALSE(mismatch("en", "the quick brown fox jumps over the lazy dog"));
    CHECK_FALSE(mismatch("de", "викинги отважные воины")); // odd, but not ours to police
}

TEST_CASE("script-mismatch tolerates code-switching and short fragments", "[unit][script-mismatch]") {
    // An English brand inside a Russian sentence must not trip it.
    CHECK_FALSE(mismatch("ru", "я купил новый iPhone вчера в магазине на Тверской улице возле метро"));
    // Below the 20-letter evidence bar nothing fires, either way.
    CHECK_FALSE(mismatch("ru", "da"));
    CHECK_FALSE(mismatch("ru", ""));
    CHECK_FALSE(mismatch("ru", "ok, da, net"));
}

TEST_CASE("script counter handles UTF-8 correctly", "[unit][script-mismatch]") {
    auto c = count_scripts("абв ABC αβγ 123 !?");
    CHECK(c.cyrillic == 3);
    CHECK(c.latin == 3);
    CHECK(c.greek == 3);
}
