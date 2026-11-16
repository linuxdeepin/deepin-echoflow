// core/phoneme_dialect.h — one G2P, several models, and they do NOT agree on
// how to spell a phoneme.
//
// Kokoro's phoneme embedding was trained on the output of misaki, its own G2P,
// which does NOT emit textbook IPA. It uses single-codepoint stand-ins for the
// English diphthongs and affricates and omits length marks entirely:
//
//     misaki   A    I    O    W    Y    ʧ    ʤ    T        ᵊ
//     IPA      eɪ   aɪ   oʊ   aʊ   ɔɪ   tʃ   dʒ   ɾ(flap)  ə(reduced)
//
// Our builtin G2P and espeak both produce the textbook forms, "tuned to match
// espeak-ng output for piper compatibility" (see core/g2p_en.h) — correct for
// piper, wrong for Kokoro. Every one of those symbols IS in Kokoro's vocab, so
// nothing is dropped and nothing errors; the model simply receives a token
// sequence from outside its training distribution and drifts. That is the
// "sounds British / old English" in #316: `ː` is the RP length mark, and
// `spˈiːtʃ` reaches the model as s,p,ˈ,i,ː,t,ʃ where it was trained on
// s,p,ˈ,i,ʧ.
//
// This is deliberately Kokoro-scoped rather than a change to the shared G2P:
// piper wants the espeak spelling and would regress. Gate:
// CRISPASR_KOKORO_MISAKI_IPA=0 restores the raw G2P output for A/B.
//
// Header-only and weight-free — tests/test-kokoro-misaki-ipa.cpp pins it, with
// misaki's own output for the issue's sentence as the expectation.

#pragma once

#include <string>
#include <vector>

namespace core_phoneme {

// Which spelling of IPA a TTS backend was trained on.
//
//   EspeakIpa  textbook / espeak-ng: `tʃ`, `oʊ`, `ɜː`, length marks. What piper
//              expects — core/g2p_en.h's ARPAbet table is explicitly "tuned to
//              match espeak-ng output for piper compatibility". THE DEFAULT:
//              changing what the G2P itself emits would regress piper.
//   Misaki     Kokoro's own G2P: `ʧ`, `O`, `ɜɹ`, and no length marks at all.
//
// New backends add a dialect here rather than teaching g2p_en a second
// spelling. The G2P has one output; the conversion belongs to the consumer
// that needs something else.
enum class Dialect {
    EspeakIpa, // identity — what the G2P already produces
    Misaki,    // Kokoro (English)
    DeVocab,   // Kokoro (German) — vocabulary fixups only, see de_vocab_rules()
    MisakiDe,  // Kokoro (German) — fixups + misaki's tied collapse, see de_rules()
};


// Multi-codepoint sequences first: longest match wins, so `tʃ` is consumed
// before a bare `t`. Order matters here, not just content.
struct Rule {
    const char* from;
    const char* to;
};

inline const std::vector<Rule>& rules() {
    static const std::vector<Rule> table = {
        // Affricates → misaki's single codepoints.
        {"tʃ", "ʧ"},
        {"dʒ", "ʤ"},
        // Diphthongs → misaki's capitals.
        {"eɪ", "A"},
        {"aɪ", "I"},
        {"oʊ", "O"},
        {"əʊ", "O"}, // en-gb spelling of the same vowel
        {"aʊ", "W"},
        {"ɔɪ", "Y"},
        // Rhotic schwa: misaki spells it out as schwa + r.
        {"ɚ", "əɹ"},
        {"ɝ", "ɜɹ"},
        // NURSE. CMUdict's ER1 reaches us as the RP `ɜː` (non-rhotic!), so
        // "world" was wˈɜːld and, once the length mark went, wˈɜld — an r
        // short of misaki's wˈɜɹld. Measured at 56 missing ɹ across a 1508-word
        // corpus, the single largest remaining divergence. Must precede the
        // length-mark rule below, which would otherwise eat the `ː` first.
        {"ɜː", "ɜɹ"},
        // The alveolar flap.
        {"ɾ", "T"},
        // t-glottalization. Our ARPAbet conversion emits a glottal stop before a
        // syllabic nasal ("certainly" -> sˈɜɹʔnli); misaki keeps the /t/ and was
        // never trained on `ʔ` in that position, so restore it.
        {"ʔ", "t"},
        // espeak's r is ɹ in misaki's inventory.
        {"r", "ɹ"},
        // Length marks do not exist in misaki's US output. Drop them LAST so
        // the vowel rules above still see `iː`/`uː` intact if they need to.
        {"ː", ""},
    };
    return table;
}

// Apply the rewrite. Single left-to-right pass with longest-match-first, so a
// replacement can never be re-examined and cascade (`tʃ`→`ʧ` must not then hit
// a `t` rule).
inline std::string to_misaki(const std::string& ipa) {
    std::string out;
    out.reserve(ipa.size());
    const auto& tbl = rules();
    size_t i = 0;
    while (i < ipa.size()) {
        bool hit = false;
        for (const auto& r : tbl) {
            const size_t len = std::char_traits<char>::length(r.from);
            if (len && ipa.compare(i, len, r.from) == 0) {
                out += r.to;
                i += len;
                hit = true;
                break;
            }
        }
        if (!hit)
            out += ipa[i++];
    }
    // ɚ→əɹ can meet an ɹ the G2P already emitted ("parameters" arrives as
    // pɚɹ…), leaving a doubled rhotic that misaki never produces. Collapse it.
    std::string collapsed;
    collapsed.reserve(out.size());
    for (size_t j = 0; j < out.size();) {
        if (out.compare(j, 4, "ɹɹ") == 0) {
            collapsed += "ɹ";
            j += 4;
            continue;
        }
        collapsed += out[j++];
    }
    return collapsed;
}

// ── German ─────────────────────────────────────────────────────────────────
//
// The German Kokoro models (dida-80b/kokoro-deutsch and the kikiri-tts family,
// both "misaki 0.9.4 + espeak-ng") phonemize with
//
//     EspeakBackend(language="de", preserve_punctuation=True,
//                   with_stress=True, tie='^', language_switch='remove-flags')
//
// and then collapse every TIED sequence to a single codepoint before the model
// ever sees it (`misaki/espeak.py`, `EspeakG2P.e2m`). `tie='^'` is what makes
// that possible: espeak marks the affricate in "Zwei" as `t^s` and the
// diphthong as `a^ɪ`, so the map is unambiguous.
//
// Our German dictionary was generated WITHOUT the tie, so we ship `tsvˈaɪ`
// where the model was trained on `ʦvˈI` — every symbol is in the 178-token
// vocabulary, nothing errors, and the model simply gets a sequence it never
// saw. That is the same defect as the English "sounds British" alphabet bug
// (#316), one language over, and it lands on the two most common German
// diphthongs (`ei`/`ai`, `au`) plus the `z`/`tz` affricate.
//
// ⚠ Without ties we cannot tell an affricate from a `t`+`s` across a morpheme
// boundary, so `ts` -> `ʦ` is a blanket rewrite. In German that is right nearly
// always (z, tz, ts are the affricate); the exceptions are compound seams.
// Regenerating the dictionary WITH `--tie` would remove the ambiguity — see
// PLAN.md.
// Symbols the German Kokoro vocabulary does NOT contain. A missing symbol is
// not approximated by `kokoro_phonemes_to_ids` — it is DROPPED, so the sound
// disappears from the utterance entirely. `ʏ` (short ü) is the one that
// matters: it is in every München, Frühstück, fünf, Glück, zurück, and we were
// deleting the vowel out of all of them. dida-80b's own dataset script hit the
// same wall and made the same substitution ("ʏ → y … the duration difference
// is learned from audio"), which is the confirmation that this is the intended
// spelling and not a workaround of ours.
//
// Applied for German whatever else is on: an approximate vowel always beats a
// deleted one.
inline const std::vector<Rule>& de_vocab_rules() {
    static const std::vector<Rule> table = {
        {"ʏ", "y"},
        // The non-syllabic mark on a diphthong's second element. Not in the
        // vocabulary either; dropping it leaves the two vowels, which is what
        // we want when the diphthong is not being collapsed.
        {"\u032f", ""},
    };
    return table;
}

inline const std::vector<Rule>& de_rules() {
    static const std::vector<Rule> table = {
        // Diphthongs → misaki's capitals. Longest first.
        {"a\u0361ɪ", "I"},
        {"aɪ̯", "I"},
        {"aɪ", "I"},
        {"a\u0361ʊ", "W"},
        {"aʊ̯", "W"},
        {"aʊ", "W"},
        {"ɔ\u0361ɪ", "Y"},
        {"ɔʏ̯", "Y"},
        {"ɔʏ", "Y"},
        {"ɔɪ", "Y"},
        {"e\u0361ɪ", "A"},
        {"eɪ", "A"},
        {"o\u0361ʊ", "O"},
        {"oʊ", "O"},
        {"ə\u0361ʊ", "Q"},
        {"əʊ", "Q"},
        // Affricates → misaki's single codepoints.
        {"t\u0361s", "ʦ"},
        {"ts", "ʦ"},
        {"t\u0361ʃ", "ʧ"},
        {"tʃ", "ʧ"},
        {"d\u0361ʒ", "ʤ"},
        {"dʒ", "ʤ"},
        {"d\u0361z", "ʣ"},
        {"dz", "ʣ"},
        {"s\u0361s", "S"},
    };
    return table;
}

// Apply a rule table left to right, longest match first.
inline std::string rewrite(const std::string& ipa, const std::vector<Rule>& tbl) {
    std::string out;
    out.reserve(ipa.size());
    size_t i = 0;
    while (i < ipa.size()) {
        bool hit = false;
        for (const auto& r : tbl) {
            const size_t len = std::char_traits<char>::length(r.from);
            if (len && ipa.compare(i, len, r.from) == 0) {
                out += r.to;
                i += len;
                hit = true;
                break;
            }
        }
        if (!hit)
            out += ipa[i++];
    }
    return out;
}

// The German vocabulary fixups alone — no alphabet collapse.
inline std::string to_de_vocab(const std::string& ipa) {
    return rewrite(ipa, de_vocab_rules());
}

// The vocabulary fixups PLUS misaki's tied-sequence collapse.
inline std::string to_misaki_de(const std::string& ipa) {
    return rewrite(rewrite(ipa, de_rules()), de_vocab_rules());
}

// Convert G2P output into the dialect a backend expects. EspeakIpa is the
// identity, so a backend on the default pays nothing.
inline std::string convert(const std::string& ipa, Dialect d) {
    if (d == Dialect::Misaki)
        return to_misaki(ipa);
    if (d == Dialect::MisakiDe)
        return to_misaki_de(ipa);
    if (d == Dialect::DeVocab)
        return to_de_vocab(ipa);
    return ipa;
}

} // namespace core_phoneme
