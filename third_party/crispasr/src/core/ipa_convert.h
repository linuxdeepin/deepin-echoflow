// core/ipa_convert.h — convert between IPA conventions.
//
// Different phonemization systems use different IPA conventions:
//   - espeak-ng (en-us): ɚ for -er, ɹ linking, ɾ flapping, ᵻ barred-i
//   - OLaPh (en-us):     ɐ for -er, ɒ for LOT, ɝ for NURSE, ɫ dark-L
//   - open-dict-data:    similar to OLaPh (Wiktionary-sourced)
//
// This header converts OLaPh/Wiktionary IPA to espeak-compatible IPA
// so we can use any IPA dictionary as input to piper models.

#pragma once

#include <string>

namespace ipa_convert {

// Convert OLaPh/Wiktionary US English IPA to espeak-ng en-us IPA.
// Applies systematic character-level substitutions.
inline std::string olaph_to_espeak_en(const std::string& ipa) {
    std::string out;
    out.reserve(ipa.size());
    size_t len = ipa.size();

    for (size_t i = 0; i < len;) {
        unsigned char c = (unsigned char)ipa[i];

        // --- Skip ZWJ (U+200D = E2 80 8D) ---
        if (c == 0xE2 && i + 2 < len && (unsigned char)ipa[i + 1] == 0x80 && (unsigned char)ipa[i + 2] == 0x8D) {
            i += 3;
            continue;
        }

        // --- 2-byte UTF-8 substitutions (C_ __) ---
        if (c >= 0xC0 && c < 0xE0 && i + 1 < len) {
            unsigned char c2 = (unsigned char)ipa[i + 1];
            unsigned int cp = ((c & 0x1F) << 6) | (c2 & 0x3F);

            switch (cp) {
            case 0x0250:           // ɐ → ɚ (near-open central → rhotacized schwa)
                out += "\xC9\x9A"; // ɚ
                i += 2;
                continue;
            case 0x0252:           // ɒ → ɔ (LOT vowel: British → American)
                out += "\xC9\x94"; // ɔ
                i += 2;
                continue;
            case 0x025D:                   // ɝ → ɜː (NURSE vowel: rhotacized → plain + length)
                out += "\xC9\x9C\xCB\x90"; // ɜː
                i += 2;
                continue;
            case 0x026B: // ɫ → l (dark L → plain L)
                out += "l";
                i += 2;
                continue;
            default:
                break;
            }
        }

        // --- Pass through everything else ---
        // Determine UTF-8 byte count
        int cp_len = 1;
        if (c >= 0xF0)
            cp_len = 4;
        else if (c >= 0xE0)
            cp_len = 3;
        else if (c >= 0xC0)
            cp_len = 2;
        if (i + cp_len > len)
            break;
        out.append(ipa, i, cp_len);
        i += cp_len;
    }
    return out;
}

} // namespace ipa_convert
