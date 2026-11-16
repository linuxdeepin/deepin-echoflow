# Vendored: OmniVoice prompt vocabularies

Both files are **byte-identical** copies from
https://github.com/k2-fsa/OmniVoice, under `omnivoice/utils/`.

| file | upstream commit | git blob sha1 |
|---|---|---|
| `lang_map.py` | `c8c0a6257b9b652dc7d239df9d911dbb1f9dabdf` (2026-04-01) | `ffcda10ae5ab93ac45ca12de1147a752585a5ed1` |
| `voice_design.py` | `9c3ab463713d15e94e252f468ac02f55bdba640b` (2026-07-06) | `b369b773e38885395313ddfb0f78cbb7d2b2303f` |

License: Apache-2.0, headers retained in the files; see `THIRD_PARTY_NOTICES.txt`.

## Why they are here

They generate `src/core/omnivoice_lang_table.h` and
`src/core/omnivoice_instruct_table.h`, and neither vocabulary is cosmetic:
OmniVoice carries both the target language and the voice-design instruct as
literal strings in its prompt, so a wrong or unresolved value conditions the
model on tokens it never saw in that slot (SubtitleEdit-13273). The instruct
vocabulary is closed — upstream *rejects* anything outside it — and casing
matters: `Male, British Accent` and `male, british accent` share no token ids.

Vendoring makes `tools/gen-omnivoice-lang-map.py --check` **hermetic**, so CI can
gate the generated header against its source without a network call. Fetching
upstream at check time would have made the gate flaky, and a gate CI cannot run
is a gate that ships wrong.

## Keep it byte-identical

Do not edit these files. The blob sha1s above are what `--check-upstream`
compares against the GitHub API, which is only an exact test while the copies
are unmodified. To take a new upstream revision:

```bash
python tools/gen-omnivoice-lang-map.py     --update-vendored
python tools/gen-omnivoice-instruct-map.py --update-vendored
```

then update the hashes and dates in the table above, and re-read the diff. New
languages and new accents are additions we want; a *changed* id for an existing
language, or a *renamed* instruct item, silently alters (or invalidates) what
callers already send.
