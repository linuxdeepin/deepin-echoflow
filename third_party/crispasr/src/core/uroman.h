// uroman.h — lightweight Unicode romanization for CTC forced alignment.
//
// When the aligner vocabulary is Latin/romanized (wav2vec2-based CTC models),
// non-Latin reference text produces all-zero alignments because no tokens match.
// This header provides a minimal romanization that maps Arabic, Cyrillic, Greek,
// and Hebrew scripts to Latin approximations — sufficient for CTC alignment
// (which needs approximate phonetic correspondence, not perfect transliteration).
//
// Based on the uroman lookup tables (MIT license, Ulf Hermjakob).
// Only the most common characters are included; rare characters fall through
// as-is (the aligner ignores unknown characters gracefully).
//
// Usage:
//   std::string romanized = core_uroman::romanize(utf8_text);
//   // feed romanized text to the CTC aligner

#pragma once

#include <cstdint>
#include <string>

namespace core_uroman {

// Returns true if the string contains any non-Latin script characters
// that would benefit from romanization.
inline bool needs_romanization(const std::string& text) {
    size_t i = 0;
    while (i < text.size()) {
        uint8_t b = (uint8_t)text[i];
        uint32_t cp;
        int len;
        if (b < 0x80) {
            i++;
            continue;
        } else if ((b & 0xE0) == 0xC0) {
            cp = b & 0x1F;
            len = 2;
        } else if ((b & 0xF0) == 0xE0) {
            cp = b & 0x0F;
            len = 3;
        } else if ((b & 0xF8) == 0xF0) {
            cp = b & 0x07;
            len = 4;
        } else {
            i++;
            continue;
        }
        for (int j = 1; j < len && i + j < text.size(); j++)
            cp = (cp << 6) | ((uint8_t)text[i + j] & 0x3F);
        // Arabic: U+0600–U+06FF, U+0750–U+077F, U+08A0–U+08FF
        if ((cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0750 && cp <= 0x077F) || (cp >= 0x08A0 && cp <= 0x08FF))
            return true;
        // Cyrillic: U+0400–U+04FF
        if (cp >= 0x0400 && cp <= 0x04FF)
            return true;
        // Devanagari: U+0900–U+097F
        if (cp >= 0x0900 && cp <= 0x097F)
            return true;
        // Greek: U+0370–U+03FF
        if (cp >= 0x0370 && cp <= 0x03FF)
            return true;
        // Hebrew: U+0590–U+05FF
        if (cp >= 0x0590 && cp <= 0x05FF)
            return true;
        i += len;
    }
    return false;
}

// Map a single Unicode codepoint to its romanized string (or empty if no mapping).
inline const char* romanize_cp(uint32_t cp) {
    // Arabic consonants (simplified uroman mapping)
    switch (cp) {
    // Arabic
    case 0x0627:
        return "a"; // alef
    case 0x0628:
        return "b"; // ba
    case 0x062A:
        return "t"; // ta
    case 0x062B:
        return "th"; // tha
    case 0x062C:
        return "j"; // jim
    case 0x062D:
        return "h"; // ha
    case 0x062E:
        return "kh"; // kha
    case 0x062F:
        return "d"; // dal
    case 0x0630:
        return "dh"; // dhal
    case 0x0631:
        return "r"; // ra
    case 0x0632:
        return "z"; // zain
    case 0x0633:
        return "s"; // sin
    case 0x0634:
        return "sh"; // shin
    case 0x0635:
        return "s"; // sad
    case 0x0636:
        return "d"; // dad
    case 0x0637:
        return "t"; // ta (emphatic)
    case 0x0638:
        return "z"; // za (emphatic)
    case 0x0639:
        return ""; // ain (glottal, often dropped)
    case 0x063A:
        return "gh"; // ghain
    case 0x0641:
        return "f"; // fa
    case 0x0642:
        return "q"; // qaf
    case 0x0643:
        return "k"; // kaf
    case 0x0644:
        return "l"; // lam
    case 0x0645:
        return "m"; // mim
    case 0x0646:
        return "n"; // nun
    case 0x0647:
        return "h"; // ha
    case 0x0648:
        return "w"; // waw
    case 0x064A:
        return "y"; // ya
    case 0x0629:
        return "h"; // ta marbuta
    case 0x0621:
        return ""; // hamza
    case 0x0622:
        return "a"; // alef madda
    case 0x0623:
        return "a"; // alef hamza above
    case 0x0625:
        return "i"; // alef hamza below
    case 0x0624:
        return "w"; // waw hamza
    case 0x0626:
        return "y"; // ya hamza
    case 0x0649:
        return "a"; // alef maksura
    // Arabic vowel marks (diacritics) — map to vowels
    case 0x064E:
        return "a"; // fatha
    case 0x064F:
        return "u"; // damma
    case 0x0650:
        return "i"; // kasra
    case 0x0651:
        return ""; // shadda (gemination, skip)
    case 0x0652:
        return ""; // sukun (no vowel, skip)
    case 0x0640:
        return ""; // tatweel (kashida, cosmetic)
    // Cyrillic (Russian)
    case 0x0410:
    case 0x0430:
        return "a";
    case 0x0411:
    case 0x0431:
        return "b";
    case 0x0412:
    case 0x0432:
        return "v";
    case 0x0413:
    case 0x0433:
        return "g";
    case 0x0414:
    case 0x0434:
        return "d";
    case 0x0415:
    case 0x0435:
        return "e";
    case 0x0416:
    case 0x0436:
        return "zh";
    case 0x0417:
    case 0x0437:
        return "z";
    case 0x0418:
    case 0x0438:
        return "i";
    case 0x0419:
    case 0x0439:
        return "y";
    case 0x041A:
    case 0x043A:
        return "k";
    case 0x041B:
    case 0x043B:
        return "l";
    case 0x041C:
    case 0x043C:
        return "m";
    case 0x041D:
    case 0x043D:
        return "n";
    case 0x041E:
    case 0x043E:
        return "o";
    case 0x041F:
    case 0x043F:
        return "p";
    case 0x0420:
    case 0x0440:
        return "r";
    case 0x0421:
    case 0x0441:
        return "s";
    case 0x0422:
    case 0x0442:
        return "t";
    case 0x0423:
    case 0x0443:
        return "u";
    case 0x0424:
    case 0x0444:
        return "f";
    case 0x0425:
    case 0x0445:
        return "kh";
    case 0x0426:
    case 0x0446:
        return "ts";
    case 0x0427:
    case 0x0447:
        return "ch";
    case 0x0428:
    case 0x0448:
        return "sh";
    case 0x0429:
    case 0x0449:
        return "shch";
    case 0x042A:
    case 0x044A:
        return ""; // hard sign
    case 0x042B:
    case 0x044B:
        return "y";
    case 0x042C:
    case 0x044C:
        return ""; // soft sign
    case 0x042D:
    case 0x044D:
        return "e";
    case 0x042E:
    case 0x044E:
        return "yu";
    case 0x042F:
    case 0x044F:
        return "ya";
    case 0x0401:
    case 0x0451:
        return "yo"; // yo
    default:
        return nullptr;
    }
}

// Romanize a UTF-8 string. Characters with no mapping pass through as-is.
// Spaces and ASCII are preserved.
inline std::string romanize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        uint8_t b = (uint8_t)text[i];
        if (b < 0x80) {
            out += (char)b;
            i++;
            continue;
        }
        uint32_t cp;
        int len;
        if ((b & 0xE0) == 0xC0) {
            cp = b & 0x1F;
            len = 2;
        } else if ((b & 0xF0) == 0xE0) {
            cp = b & 0x0F;
            len = 3;
        } else if ((b & 0xF8) == 0xF0) {
            cp = b & 0x07;
            len = 4;
        } else {
            i++;
            continue;
        }
        for (int j = 1; j < len && i + j < text.size(); j++)
            cp = (cp << 6) | ((uint8_t)text[i + j] & 0x3F);

        const char* rom = romanize_cp(cp);
        if (rom) {
            out += rom;
        } else {
            // Pass through unmapped characters (punctuation, other scripts)
            out.append(text, i, len);
        }
        i += len;
    }
    return out;
}

} // namespace core_uroman
