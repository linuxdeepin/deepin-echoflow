#!/usr/bin/env python3
"""Regenerate src/core/g2p_de_unstressed.h by asking espeak-ng.

Our German dictionary (`espeak_de.tsv`) was built by running espeak-ng over a
word list ONE WORD AT A TIME, so every entry is the citation form. espeak
stresses a word in isolation that it leaves unstressed in a sentence:

    espeak "sie"                       ->  zˈiː
    espeak "sie ging dann nach Hause"  ->  ziː ɡˈɪŋ dan nɑːx hˈaʊzə

which means we emit a primary stress on every article, pronoun, preposition and
auxiliary — and both the German Kokoro and the German piper voices were trained
on espeak's SENTENCE output. This script recovers the difference from espeak
itself rather than guessing at a list.

Method: for each candidate, take the isolated reading and the reading inside TWO
different carrier frames. Keep the word only when both frames agree AND the
in-frame form is exactly the isolated form with its stress marks removed —
so a rhythm artefact of one frame cannot invent an entry, and a word whose
SEGMENTS change (sandhi, not de-stressing) is rejected.

    python tools/gen-g2p-de-unstressed.py > src/core/g2p_de_unstressed.h

Needs espeak-ng on PATH. The candidate list is the German closed class; adding
to it is safe, since espeak rejects anything that does not actually de-stress.
"""

from __future__ import annotations

import subprocess
import sys

# Articles, pronouns, prepositions, conjunctions, auxiliaries, particles.
CANDIDATES = """
der die das den dem des ein eine einen einem einer eines
ich du er sie es wir ihr mich dich sich uns euch mir dir ihm ihn ihnen
mein dein sein ihre ihrer seine seinen seinem meiner deiner
und oder aber denn sondern doch dass ob wenn als wie weil damit
in an auf aus bei mit nach von vor zu zum zur für um durch ohne gegen
über unter neben zwischen hinter seit bis ab je pro
ist sind war waren bin bist seid hat habe haben hast hatte hatten
wird werden wurde wurden kann konnte können muss müssen soll sollen
will wollen mag mögen darf dürfen
nicht kein keine keinen nur auch noch schon mehr sehr so dann da hier dort
man am im vom beim ans ins
""".split()

FRAMES = [
    ("Haus {} Baum", 3, 1),
    ("Der Tisch {} Baum steht", 5, 2),
]


def ipa(text: str) -> str:
    r = subprocess.run(
        ["espeak-ng", "-v", "de", "-q", "--ipa", text],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip().replace("\n", " ")


def derive() -> dict[str, str]:
    out: dict[str, str] = {}
    for w in sorted(set(CANDIDATES)):
        iso = ipa(w)
        if not iso:
            continue
        forms = []
        for frame, n_tokens, index in FRAMES:
            toks = ipa(frame.format(w)).split()
            if len(toks) != n_tokens:
                forms = []
                break
            forms.append(toks[index])
        if len(forms) != len(FRAMES) or len(set(forms)) != 1:
            continue
        got = forms[0]
        if got == iso:
            continue  # espeak does not de-stress it
        if got != iso.replace("ˈ", "").replace("ˌ", ""):
            continue  # segments changed — not a pure stress loss
        out[w] = got
    return out


HEADER = '''// core/g2p_de_unstressed.h — German function words that carry no stress in
// running speech.
//
// GENERATED — do not hand-edit. Regenerate with
// tools/gen-g2p-de-unstressed.py, which asks espeak-ng directly.
//
// Why this file exists. Our German dictionary (`espeak_de.tsv`) was generated
// by running espeak-ng over a word list ONE WORD AT A TIME, so every entry is
// the CITATION form — and espeak stresses a word in isolation that it leaves
// unstressed in a sentence:
//
//     espeak "sie"                       ->  zˈiː
//     espeak "sie ging dann nach Hause"  ->  ziː ɡˈɪŋ dan nɑːx hˈaʊzə
//
// So we were emitting a primary stress on every article, pronoun, preposition
// and auxiliary in the sentence — the German shape of exactly the #316 English
// bug, where a per-word lexicon can only store `the` as `ði` and never `ðə`.
// Both the German Kokoro (dida-80b hui) and the German piper voices are trained
// on espeak's SENTENCE output, so the citation form is a token sequence they
// never saw.
//
// Measured against espeak's sentence output over 8 sentences: token agreement
// 45.9% -> 87.1% with this table applied.
//
// The rule is purely LEXICAL, not positional — espeak reads even a
// sentence-initial "Der" as `dɛɾ`. Every entry below was verified in two
// independent carrier frames and accepted only when the in-frame form is the
// citation form with its stress marks removed and nothing else changed.
//
// Weight-free and header-only.

#pragma once

#include <map>
#include <string>

namespace core_g2p_de_unstressed {

// word (lowercase) -> the unstressed reading espeak gives it in a sentence.
inline const std::map<std::string, std::string>& table() {
    static const std::map<std::string, std::string> t = {'''

FOOTER = '''    };
    return t;
}

// The unstressed reading, or "" when the word is not in the closed class.
inline std::string lookup(const std::string& lower_word) {
    const auto& t = table();
    auto it = t.find(lower_word);
    return it == t.end() ? std::string() : it->second;
}

} // namespace core_g2p_de_unstressed'''


def main() -> int:
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        print("espeak-ng not on PATH", file=sys.stderr)
        return 1
    entries = derive()
    print(HEADER)
    for k in sorted(entries):
        print(f'        {{"{k}", "{entries[k]}"}},')
    print(FOOTER)
    print(f"\n// {len(entries)} entries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
