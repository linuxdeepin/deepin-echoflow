// src/core/tts_lang.h — language tags + reference-transcript language ID for TTS.
//
// Cross-lingual voice cloning (#304, #329) needs to answer one question before
// synthesis: "is the language the user asked for the same as the language the
// REFERENCE clip is spoken in?" If it differs, CosyVoice3 mirrors upstream's
// frontend_cross_lingual and drops the reference transcript; if it matches, the
// transcript stays and the clone is a plain zero-shot.
//
// That predicate lives downstream of every tensor, so the per-stage diff harness
// cannot see it: get it wrong and the graph still computes cos 1.000000 while
// the user hears a heavily accented clone (or, in the reverse error, loses the
// zero-shot anchoring for no reason). It is exactly the "harness-blind zone"
// that gets its own hermetic test — hence a weight-free header rather than a
// static helper buried in cosyvoice3_tts.cpp. See tests/test-tts-lang.cpp.
//
// The script detector (#304) only ever answered for Hangul / Kana / Han /
// Cyrillic, so every Latin-script pair — en↔de, en↔fr, es↔it, the ones a
// subtitle-dubbing workflow actually asks for — returned "unknown" and silently
// stayed zero-shot. That is the bug behind #329. Two answers here:
//   (a) an EXPLICIT reference-language tag from the caller always wins, and
//   (b) a conservative function-word detector for Latin-script languages, which
//       returns "" whenever the evidence is thin rather than guessing.
//
// Why not the real text-LID (`crispasr_text_detect_language`, CLD3/GlotLID)?
// It needs its own GGUF. Making a voice clone depend on a second model download
// to decide one boolean is the wrong trade; a caller that already has the LID
// model can run it and pass the answer in as the explicit tag, which is exactly
// what path (a) is for.
#pragma once

#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace core_tts_lang {

// Normalize a language tag to a lowercase 2-letter base for comparison
// ("en-US" -> "en", "cmn"/"zho" -> "zh", "jpn" -> "ja", "German" -> "de", …).
// Unknown 3+ letter tags are truncated to their first two characters, which is
// correct for ISO-639-1/2 pairs that share a stem and harmless otherwise (an
// unknown tag only ever compares equal to itself).
inline std::string norm(const std::string& s) {
    std::string t;
    for (char c : s) {
        if (c == '-' || c == '_')
            break;
        t += (char)std::tolower((unsigned char)c);
    }
    if (t.empty() || t == "auto")
        return std::string();
    // ISO-639-2/3 and English names. The English names matter because the
    // session ABI and several CLI adapters spell languages out for LLM prompts
    // (core_lang::iso_to_english), so a caller can legitimately hand us "German".
    static const std::map<std::string, std::string> alias = {
        {"cmn", "zh"},     {"zho", "zh"},      {"chi", "zh"},        {"chinese", "zh"}, {"mandarin", "zh"},
        {"jpn", "ja"},     {"japanese", "ja"}, {"kor", "ko"},        {"korean", "ko"},  {"eng", "en"},
        {"english", "en"}, {"deu", "de"},      {"ger", "de"},        {"german", "de"},  {"fra", "fr"},
        {"fre", "fr"},     {"french", "fr"},   {"spa", "es"},        {"spanish", "es"}, {"ita", "it"},
        {"italian", "it"}, {"por", "pt"},      {"portuguese", "pt"}, {"nld", "nl"},     {"dut", "nl"},
        {"dutch", "nl"},   {"rus", "ru"},      {"russian", "ru"},    {"pol", "pl"},     {"polish", "pl"},
        {"ara", "ar"},     {"arabic", "ar"},   {"hin", "hi"},        {"hindi", "hi"},   {"tur", "tr"},
        {"turkish", "tr"}, {"vie", "vi"},      {"vietnamese", "vi"}, {"ukr", "uk"},     {"ukrainian", "uk"},
    };
    auto it = alias.find(t);
    if (it != alias.end())
        return it->second;
    return t.size() > 2 ? t.substr(0, 2) : t;
}

// Two language tags name the same language (after normalization). An empty tag
// on either side is "unknown", which is NOT a match — callers use this to gate
// a behaviour change and must stay on the safe default when in doubt.
inline bool same_language(const std::string& a, const std::string& b) {
    const std::string na = norm(a), nb = norm(b);
    return !na.empty() && !nb.empty() && na == nb;
}

// The caller asked for `target` and the reference clip is `reference`: does that
// require cross-lingual synthesis? Only when BOTH are known and they differ.
inline bool is_cross_lingual(const std::string& target, const std::string& reference) {
    const std::string t = norm(target), r = norm(reference);
    return !t.empty() && !r.empty() && t != r;
}

// ---------------------------------------------------------------------------
// Script detection — unambiguous, decided by codepoint ranges alone.
// ---------------------------------------------------------------------------

// Decode UTF-8 into codepoints, tolerating malformed input (a bad byte becomes
// its own codepoint rather than derailing the scan).
inline std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp = c;
        int n = 1;
        if (c >= 0xF0) {
            cp = c & 0x07u;
            n = 4;
        } else if (c >= 0xE0) {
            cp = c & 0x0Fu;
            n = 3;
        } else if (c >= 0xC0) {
            cp = c & 0x1Fu;
            n = 2;
        }
        for (int k = 1; k < n && i + (size_t)k < s.size(); k++)
            cp = (cp << 6) | ((unsigned char)s[i + (size_t)k] & 0x3Fu);
        i += (size_t)n;
        out.push_back(cp);
    }
    return out;
}

// Language implied by the writing system, or "" for Latin/unknown script.
// Hangul beats Kana beats Han: Korean and Japanese both mix Han characters in,
// so the script that only ONE language uses is the decisive one. Han-only text
// therefore reads as Chinese — the same convention #304 shipped.
inline std::string script_language(const std::string& text) {
    bool hangul = false, kana = false, han = false, cyr = false, arab = false, deva = false, thai = false, hebr = false,
         grek = false;
    for (uint32_t cp : utf8_codepoints(text)) {
        if (cp >= 0xAC00 && cp <= 0xD7A3)
            hangul = true;
        else if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF))
            kana = true;
        else if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF))
            han = true;
        else if (cp >= 0x0400 && cp <= 0x04FF)
            cyr = true;
        else if ((cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0750 && cp <= 0x077F))
            arab = true;
        else if (cp >= 0x0900 && cp <= 0x097F)
            deva = true;
        else if (cp >= 0x0E00 && cp <= 0x0E7F)
            thai = true;
        else if (cp >= 0x0590 && cp <= 0x05FF)
            hebr = true;
        else if (cp >= 0x0370 && cp <= 0x03FF)
            grek = true;
    }
    if (hangul)
        return "ko";
    if (kana)
        return "ja";
    if (han)
        return "zh";
    if (cyr)
        return "ru";
    if (arab)
        return "ar";
    if (deva)
        return "hi";
    if (thai)
        return "th";
    if (hebr)
        return "he";
    if (grek)
        return "el";
    return std::string();
}

// ---------------------------------------------------------------------------
// Latin-script detection — function words + orthography.
// ---------------------------------------------------------------------------

// Split into lowercase words. ASCII letters are lowercased; bytes >= 0x80 are
// kept verbatim so "für" and "não" survive as single words (their accented
// characters are never word-initial in practice, so ASCII-only lowercasing is
// enough to match the tables below). Everything else is a separator.
inline std::vector<std::string> words_lower(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : text) {
        if (std::isalpha(c) || c >= 0x80) {
            cur += (char)(c < 0x80 ? std::tolower(c) : c);
        } else if (!cur.empty()) {
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

// Function-word table, one row per Latin-script language we can tell apart.
//
// Scoring weights each hit by 1/(number of languages claiming that word), so
// the many words Romance languages share ("la", "de", "que", "un") contribute
// almost nothing while a word only one language uses ("nicht", "the", "não")
// carries a full point. That is what lets es/it/pt be separated at all without
// a model — no hand-tuned per-word weights to get wrong.
inline const std::vector<std::pair<std::string, std::vector<std::string>>>& function_words() {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> table = {
        {"en", {"the",   "and",   "of",    "to",    "is",   "in",   "that", "it",   "you",  "for", "with",
                "was",   "this",  "are",   "have",  "not",  "but",  "they", "from", "what", "his", "her",
                "there", "would", "about", "which", "been", "will", "than", "were", "when"}},
        {"de", {"der",   "die",   "das",  "und",   "ist",    "nicht", "ein",   "eine",  "mit",   "sich", "auch",
                "dem",   "den",   "für",  "ich",   "wir",    "aber",  "oder",  "war",   "wie",   "auf",  "von",
                "eines", "einem", "sind", "haben", "werden", "noch",  "nach",  "über",  "durch", "sehr", "es",
                "sie",   "was",   "wenn", "man",   "hat",    "hier",  "dann",  "schon", "immer", "weil", "doch",
                "mehr",  "alle",  "kann", "muss",  "wird",   "ihnen", "diese", "geht",  "zum",   "zur"}},
        {"fr", {"le",   "la",   "les",  "et",   "est",  "une",   "un",   "des", "du",  "que",  "qui",
                "pour", "dans", "pas",  "avec", "sur",  "nous",  "vous", "ce",  "il",  "elle", "sont",
                "être", "plus", "mais", "ses",  "leur", "cette", "aux",  "ont", "très"}},
        {"es", {"el",  "la",   "los",  "las", "y",    "es",     "un",      "una", "de",   "que",
                "en",  "por",  "para", "con", "no",   "se",     "su",      "del", "pero", "como",
                "más", "está", "son",  "muy", "todo", "cuando", "también", "hay", "esta", "sus"}},
        {"it", {"il",  "la",   "le",     "e",      "che",   "di",     "un",    "una", "per",  "con",
                "non", "sono", "questo", "come",   "anche", "del",    "della", "gli", "nel",  "ma",
                "più", "sua",  "suo",    "essere", "hanno", "quando", "molto", "dei", "alla", "nella"}},
        {"pt", {"o",    "os",  "as",  "e",    "que",    "de",     "um",   "uma",  "para",  "com",
                "não",  "se",  "por", "do",   "da",     "dos",    "das",  "mais", "muito", "são",
                "está", "seu", "sua", "como", "quando", "também", "isso", "ele",  "ela",   "pelo"}},
        {"nl", {"de",   "het",    "een",   "en",   "van",  "is",   "dat", "niet", "met",  "voor",
                "op",   "te",     "zijn",  "maar", "ook",  "aan",  "ze",  "hij",  "er",   "als",
                "naar", "worden", "heeft", "deze", "over", "door", "wij", "dan",  "geen", "nog"}},
        {"pl", {"nie", "się",  "jest",     "na",   "do",    "to",    "że",    "jak",     "ale",    "przez",
                "dla", "oraz", "który",    "była", "były",  "tego",  "przy",  "tylko",   "bardzo", "jego",
                "ich", "być",  "wszystko", "może", "który", "gdzie", "kiedy", "jeszcze", "tak",    "już"}},
    };
    return table;
}

// Orthographic tie-breakers: characters (or character pairs) that only one of
// these languages writes. Worth a full point each, capped so a single exotic
// character can't outvote a whole sentence of function words.
inline std::map<std::string, float> orthographic_hints(const std::string& text) {
    std::map<std::string, float> score;
    int de = 0, es = 0, pt = 0, fr = 0, pl = 0;
    const std::vector<uint32_t> cps = utf8_codepoints(text);
    for (size_t i = 0; i < cps.size(); i++) {
        const uint32_t cp = cps[i];
        switch (cp) {
        case 0x00DF: // ß
            de++;
            break;
        case 0x00F1: // ñ
        case 0x00D1: // Ñ
        case 0x00BF: // ¿
        case 0x00A1: // ¡
            es++;
            break;
        case 0x00E3: // ã
        case 0x00F5: // õ
        case 0x00C3: // Ã
        case 0x00D5: // Õ
            pt++;
            break;
        case 0x0153: // œ
        case 0x00FB: // û
        case 0x00EE: // î
            fr++;
            break;
        case 0x0142: // ł
        case 0x017C: // ż
        case 0x017A: // ź
        case 0x0119: // ę
        case 0x0105: // ą
        case 0x015B: // ś
        case 0x0107: // ć
            pl++;
            break;
        default:
            break;
        }
    }
    auto add = [&score](const char* lang, int n) {
        if (n > 0)
            score[lang] = (float)(n > 2 ? 2 : n);
    };
    add("de", de);
    add("es", es);
    add("pt", pt);
    add("fr", fr);
    add("pl", pl);
    return score;
}

// Best-effort language of Latin-script text, or "" when the evidence is thin.
//
// Deliberately conservative: fewer than `min_words` words, a winner that scores
// below `min_score`, or a winner that fails to beat the runner-up by
// `min_margin` all return "". A wrong answer here silently changes synthesis
// behaviour, so "I don't know" is the correct output far more often than a
// guess would be.
inline std::string latin_language(const std::string& text, size_t min_words = 4, float min_score = 1.0f,
                                  float min_margin = 1.5f) {
    const std::vector<std::string> ws = words_lower(text);
    if (ws.size() < min_words)
        return std::string();

    // How many languages claim each word → the inverse is its weight.
    static const std::map<std::string, int> claims = [] {
        std::map<std::string, int> m;
        for (const auto& row : function_words())
            for (const auto& w : row.second)
                m[w]++;
        return m;
    }();

    std::map<std::string, float> score = orthographic_hints(text);
    for (const auto& row : function_words()) {
        float s = 0.0f;
        for (const auto& w : ws) {
            for (const auto& fw : row.second) {
                if (fw == w) {
                    auto it = claims.find(w);
                    const int n = (it == claims.end() || it->second < 1) ? 1 : it->second;
                    s += 1.0f / (float)n;
                    break;
                }
            }
        }
        if (s > 0.0f)
            score[row.first] += s;
    }
    if (score.empty())
        return std::string();

    std::string best;
    float best_s = 0.0f, second_s = 0.0f;
    for (const auto& kv : score) {
        if (kv.second > best_s) {
            second_s = best_s;
            best_s = kv.second;
            best = kv.first;
        } else if (kv.second > second_s) {
            second_s = kv.second;
        }
    }
    if (best_s < min_score)
        return std::string();
    if (second_s > 0.0f && best_s < second_s * min_margin)
        return std::string();
    return best;
}

// Language of a piece of text: script first (unambiguous), then Latin function
// words. "" = undeterminable — callers must treat that as "keep the default",
// never as "different from the target".
inline std::string detect(const std::string& text) {
    const std::string s = script_language(text);
    if (!s.empty())
        return s;
    return latin_language(text);
}

// Resolve the language of a cloning reference, in descending order of
// authority:
//   1. `explicit_lang`  — the caller said so (CLI --source-lang, server
//      "source_lang", crispasr_session_set_source_language). A human statement
//      about their own recording outranks anything we can infer.
//   2. `bank_lang`      — a baked voice whose bank entry names its language
//      (CosyVoice3's "fleurs-de"). Ground truth for that voice.
//   3. detection over the reference transcript.
// Returns "" when none of the three answers.
inline std::string resolve_reference_language(const std::string& explicit_lang, const std::string& bank_lang,
                                              const std::string& prompt_text) {
    const std::string e = norm(explicit_lang);
    if (!e.empty())
        return e;
    const std::string b = norm(bank_lang);
    if (!b.empty())
        return b;
    return detect(prompt_text);
}

} // namespace core_tts_lang
