# #329 — target language for cosyvoice3 / qwen3-tts cloning

> *"There is usually an option to select the target language when porting
> languages, but I don't find this option for these engines in Subtitle Edit.
> At the same time, VoxCPM2 seems to be doing a good job without it, which is
> not the case with the other two. Does it exist and how can I configure it?"*

## NOW — active work

- **Done** — branch `feat/329-tts-target-lang`. Four defects fixed + the whole
  knob documented (it was undocumented on every surface, which is most of why
  the reporter could not find it).
- **Done** — live cross-lingual A/B on real cosyvoice3 weights, en → de. See
  "Live A/B" below.

## What was actually wrong

The knob existed; four separate things kept it from working or from being
findable.

### 1. Cross-lingual never engaged for Latin-script references (the real bug)

`cv3_voice_language()` (#304) inferred the reference clip's language from the
writing system: Hangul → ko, Kana → ja, Han → zh, Cyrillic → ru, **everything
else → ""**. And the gate is

```cpp
cross_lingual = !target.empty() && !ref.empty() && target != ref;
```

so a reference whose language is unknown can never be cross-lingual. Every
Latin-script pair — en↔de, en↔fr, es↔it, i.e. exactly what a subtitle-dubbing
workflow asks for — silently stayed zero-shot: the reference transcript stayed
in the LM prompt and the clone came out with the reference's accent. Passing
`-tl de` did nothing and said nothing.

Fix: `src/core/tts_lang.h` — script detection (extended past the original four)
plus a conservative Latin function-word detector, and an **explicit** reference
language that outranks both. The detector weights each function-word hit by
`1/(number of languages claiming that word)`, which is what separates es/it/pt
without hand-tuned per-word weights, and it returns `""` rather than guessing
when the evidence is thin (< 4 words, or the winner fails to beat the runner-up
by 1.5×). Getting this wrong changes synthesis behaviour silently, so declining
is the correct output far more often than a guess.

Not the real text-LID (`crispasr_text_detect_language`, CLD3/GlotLID): it needs
its own GGUF, and making a voice clone depend on a second model download to
decide one boolean is the wrong trade. A caller who has that model can run it
and pass the answer in as the explicit tag.

### 2. No way to say what the reference language is

Even with a detector there are references it cannot call (a three-word
`--ref-text`). New knob on all four surfaces:

| Surface | Spelling |
|---|---|
| CLI | `-sl` / `--source-lang` (cosyvoice3 already declared `CAP_SRC_TGT_LANGUAGE`, so the flag was accepted and silently discarded) |
| Server | `"source_lang"` (alias `"ref_lang"`) on `POST /v1/audio/speech` |
| C ABI | `crispasr_session_set_tts_reference_language()` |
| Python / Rust | `set_tts_reference_language()` |

It is its own session setter rather than reusing `set_source_language` because
for TTS that one already doubles as the *output*-language fallback
(`target ?: source`) across zonos/kokoro/qwen3-tts/moss.

And when a target language is set but the reference language is undeterminable,
the backend now says so and names the flag that resolves it, instead of
silently ignoring the request.

### 3. The session ABI dropped the language entirely (multi-surface trap)

`crispasr_c_api.cpp` reimplements each backend's synthesize inline rather than
calling the CLI adapter (HARD RULE #6). The cosyvoice3 branch never called
`cosyvoice3_tts_set_target_language`, so **bindings, Flutter, Android and every
non-CLI/non-server surface** could not do cross-lingual cosyvoice3 at all — the
#304 wiring only ever existed in the CLI adapter.

### 4. qwen3-tts ignored `-tl` and advertised nothing

The adapter read `params.language` only, and the backend did not declare
`CAP_SRC_TGT_LANGUAGE` — so `--target-lang de` printed
*"--target-lang ignored by this backend"* and was dropped. `-l de` worked, which
is not where a dubbing user looks. Now `-tl` wins over `-l` (matching
cosyvoice3) and the capability bit is declared.

## Why voxcpm2 "just works"

It is language-agnostic — it reads the script of the input text and has no
language knob to miss. cosyvoice3 and qwen3-tts both carry explicit language
conditioning (a reference-transcript decision and a `codec_language_id`
respectively), which is more controllable *and* fails closed when nobody tells
it what to do.

## Files

| File | Change |
|---|---|
| `src/core/tts_lang.h` | **new** — tags, script + Latin LID, `is_cross_lingual`, `resolve_reference_language` |
| `tests/test-tts-lang.cpp` | **new** — 68 assertions, hermetic |
| `src/cosyvoice3_tts.{h,cpp}` | `set_reference_language()`; local helpers → `core_tts_lang`; the "could not determine" message |
| `src/crispasr_c_api.cpp` | session cosyvoice3 language wiring + `crispasr_session_set_tts_reference_language` |
| `include/crispasr_session.h` | ABI declaration |
| `examples/cli/crispasr_backend_cosyvoice3.cpp` | pass `-sl` through |
| `examples/cli/crispasr_backend_qwen3_tts.cpp` | `-tl` > `-l`; `CAP_SRC_TGT_LANGUAGE` |
| `examples/cli/crispasr_server.cpp` | `source_lang` / `ref_lang` on `/v1/audio/speech` |
| `examples/cli/cli.cpp` | `-sl`/`-tl` help text mentions the TTS meaning |
| `python/crispasr/_binding.py`, `crispasr{,-sys}/src/lib.rs` | binding methods |
| `docs/tts.md`, `docs/cli.md`, `docs/server.md` | the knob, per surface and per backend |

## Guard first, then fix

Per the "write the guard BEFORE the fix so you watch it fail" rule, the #329
regression case was run against a counterfactual build with `detect()` reduced
to the old script-only behaviour:

```
tests/test-tts-lang.cpp:129: FAILED:
  REQUIRE( resolve_reference_language("", "", "Das Wetter ist heute schön …") == "de" )
with expansion: "" == "de"
tests/test-tts-lang.cpp:147: FAILED:
  REQUIRE( is_cross_lingual("de", detect(en)) )
with expansion: false
```

2 failed / 9 cases. With the fix restored: 68 assertions, 9 cases, all passed.

## Live A/B — en reference → de output

M1 Metal, `cosyvoice3-llm-q4_k` + `flow-q8_0` + `hift-f16` + `s3tok-f16` +
`campplus-f16`, seed 42, `--voice samples/jfk.wav --ref-text "<JFK transcript>"
--i-have-rights --no-spoken-disclaimer --accept-marking-responsibility`,
text `"Der Bundeskanzler sprach gestern über die wirtschaftliche Lage in
Europa."`

Arm A is `-tl` omitted, which executes **exactly the code path a pre-fix
`-tl de` took** — the flag resolved the reference to "unknown" and was
discarded. Arm B is `-tl de`.

| | speech tokens | duration | whisper LID | speaker cos vs ref |
|---|---:|---:|---|---:|
| **A** — no `-tl` (= pre-fix `-tl de`) | 262 | 10.48 s | de **0.615** | 0.778 |
| **B** — `-tl de` (fixed) | 122 | 4.88 s | de **0.966** | 0.689 |
| un-cloned floor (bank voice, same text) | — | 2.32 s | — | 0.436 |

**The new path executes** (the thing a byte-comparison cannot prove):

```
cosyvoice3_tts: cross-lingual (voice=runtime[en] → target=de): dropping
reference transcript + LM reference speech tokens (flow keeps them for timbre)
```

`voice=runtime[en]` is the fix itself — the Latin-script English reference
transcript is now identified. Pre-fix that read `""` and the branch was
unreachable.

**Content survives.** Both arms round-trip to the input sentence verbatim
(`whisper large-v3-turbo -l auto`), WER 0. The fix does not damage what is said.

**Arm A renders 2.1× the speech tokens for the same sentence** — 10.48 s of
continuously-voiced audio (per-500 ms RMS never drops to silence) for text that
takes 4.88 s, while whisper recovers only the German sentence from it. That is
the reference-transcript leak the cross-lingual path exists to remove, and the
clearest hard number here.

**Cloning survives the transcript drop** — 0.689 vs a 0.436 un-cloned floor, so
dropping the LM's reference speech tokens did not throw the voice away (the flow
still gets them for timbre, as designed).

Read with care:

- Arm A's *higher* 0.778 is confounded: 5.6 s of its audio is extra
  reference-like material, which inflates a similarity score against the
  reference. It is not evidence of a better clone.
- Whisper's LID confidence (0.62 → 0.97) is a **proxy** for accent, not a
  measurement of it. I did not listen to the audio.
- One sentence, one seed, one language pair.

## Open

- Only cosyvoice3 acts on `source_lang`. If another cross-lingual cloning
  backend lands, it reads the same session field.
- Subtitle Edit still has to surface the field; nothing here can add a menu to
  a third-party UI. The server contract it would call is now documented in
  `docs/server.md`.
