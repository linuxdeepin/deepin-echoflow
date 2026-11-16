# Speaker labels in diarization — session-scoped clustering vs. named profiles

CrispASR's diarization assigns a label to every transcribed segment. There
are two very different ways to make those labels meaningful, with very
different privacy and legal profiles. **Read this before enabling the second
one.**

| | Session-scoped clustering | Named voiceprint profiles |
|---|---|---|
| Flag | `--diarize-speakers` | `--enroll-speaker` + `--speaker-db` + `--expect-speakers` + `--speaker-db-consent` |
| Output | `(speaker 0)`, `(speaker 1)`, … | a real name, e.g. `(Mustermann)`, for **claimed** participants only |
| Embeddings | computed per recording, then **discarded** | **persisted to disk** as `.spkr` files |
| Scope | one recording only | a standing database reused across recordings |
| Identifies a named person? | **No** | **Confirms claimed participants** (closed roster; no open 1:N search) |
| Privacy footprint | transient audio processing | **biometric special-category data (GDPR Art. 9)** |

The default and recommended path is **session-scoped clustering**. The named
path exists, is **off by default**, and should only be used deliberately and
with the obligations in the last section understood.

> **Streaming?** Both columns above are **recorded-file (offline)** features.
> For live transcription, a backend that emits a structured per-segment speaker
> label (`moss-diarize`; `vibevoice` from v0.8.24; `granite` in speaker-aware
> `--diarize` mode) surfaces its own per-utterance `(Speaker N)` labels under
> `--stream` — but the cross-recording clustering and named-voiceprint paths
> here do not run in real time. See [`streaming.md`](streaming.md#speaker-diarization-while-streaming).

> **From a binding?** The same native label is on the session ABI from v0.8.24 —
> `crispasr_session_result_segment_speaker(result, i)` in C, `.speaker` on
> Python's `SessionSegment`, `.Speaker` on Go's `TranscribeSegment`, `.speaker`
> on Dart's `SessionSegment`. It is `""` when the backend does not diarize
> natively, and its ordinals are **chunk-local** — `Speaker 1` from one
> `transcribe()` call is not necessarily the same voice as `Speaker 1` from the
> next, because no cross-chunk clustering runs there. Use the `diarize_*`
> helpers in section 1 when you need labels stable across a whole recording.

---

## 1. Session-scoped speaker clustering (recommended)

This makes diarization labels **stable within a single recording** — so the
same physical voice gets the same `(speaker N)` label from start to finish —
**without identifying anyone**. Speaker embeddings are extracted per recording
purely to cluster segments, then thrown away. Nothing is stored, no names are
attached, and there is no database.

```bash
# Easiest: opt-in convenience alias. Enables --diarize and session-scoped
# clustering with the default embedder (auto-downloads ~46 MB TitaNet once).
crispasr -m auto --backend cohere -f meeting.wav --diarize-speakers -ojf

# Best quality for >2 speakers / long meetings — combine with the pyannote
# segmenter so segments also split at speaker-turn boundaries:
crispasr -m auto --backend cohere -f meeting.wav \
    --diarize --diarize-method pyannote --sherpa-segment-model auto \
    --diarize-embedder auto -ojf
```

`--diarize-speakers` is shorthand for `--diarize --diarize-method pyannote
--diarize-embedder auto` (it only fills in fields you didn't set explicitly):
the pyannote segmenter gives proper speaker-turn boundaries and the embedder
clusters them into stable per-recording labels. By default the embedding stage
**estimates the speaker count itself** (`core_spectral::cluster_speakers` —
PCA + GMM/BIC, then spectral clustering), bounded above by
`--diarize-max-speakers` (default `8`) and, when you know it, pinned by
`--diarize-num-speakers`. Passing `--diarize-cluster-threshold` *explicitly*
(default `0.5`; higher = more distinct clusters) switches that stage to the
legacy agglomerative single-linkage clusterer instead — the threshold is
meaningless to the spectral path, so its default value is never applied.
The first run auto-downloads the
pyannote segmentation GGUF (~6 MB) and the TitaNet embedder (~46 MB).

The output uses generic labels:

```
(speaker 0) Welcome everyone, let's get started.
(speaker 1) Thanks. I pulled the numbers for Q3...
(speaker 0) Great, walk us through them.
```

### Putting real names on it — do it yourself, downstream

If you want names in the final transcript, the privacy-clean approach is a
**manual** find-and-replace once you know who is who (e.g. you recognise that
`speaker 1` is the person who presented Q3 numbers):

```bash
sed -e 's/(speaker 0)/Schmidt:/g' -e 's/(speaker 1)/Mustermann:/g' \
    meeting.txt > meeting.named.txt
```

This keeps the *identification* step a human decision made per recording. The
tool never builds or consults a biometric database of people's names, so the
heavyweight obligations in section 2 don't apply.

### Why this path is privacy-clean

Session-scoped clustering is **not** a biometric *identification* system:

- **No enrollment** and **no stored voiceprints** — embeddings live in memory
  for the duration of one run and are discarded.
- **No names** and **no standing database** — labels are anonymous ordinals
  scoped to a single file.
- **No 1:N matching against a roster** of known individuals.

It is ordinary, transient audio processing whose only purpose is diarization
quality. It does not fall under the EU AI Act's *remote biometric
identification* regime (there is no identification system to classify), and it
carries the GDPR footprint of normal audio processing rather than the
special-category biometric apparatus that a named voiceprint database triggers.

---

## 2. Named voiceprint profiles (`--speaker-db`) — deliberate opt-in

> **This is a biometric feature. It persists voiceprints linked to real names.
> Treat it accordingly.** It is **off by default**, not part of the recommended
> diarization path, and deliberately restricted to a **closed-roster
> confirmation**: it can only put names on participants you claim are present
> and who consented at enrollment. There is **no** "who is this voice?" mode.

The flags `--enroll-speaker NAME` (save an embedding) and `--speaker-db DIR`
plus `--expect-speakers "NameA,NameB"` (confirm which claimed participant each
diarized speaker cluster is) let you auto-label recordings of a known,
consenting group — e.g. recurring meeting minutes.

**How identification works (issue #266).** Anonymous diarization and named
identification share one pipeline:

```
ASR segments
  -> diarization (speaker-turn labels per slice)
  -> merge slices
  -> global speaker clustering (anonymous, stable (speaker N) labels)
  -> representative embedding per cluster (centroid; reuses the
     embeddings clustering already computed)
  -> OPTIONAL: match each cluster against the CLAIMED roster only
  -> matched cluster  -> (Alice)
     unmatched cluster -> (speaker N)      # stays anonymous
```

Each cluster is matched **independently**; a successful match renames only
that cluster, and nothing later in the pipeline can overwrite a matched name.
A slice containing several speakers can never be blanket-labeled with one
identity — names attach to global clusters, not to audio slices. Without
`--diarize`, the recording is treated as a single cluster (single-speaker
verification) and, on a match, all segments get the one name.

**Three hard gates**, all enforced before any matching:

1. `--speaker-db-consent` — affirms a lawful basis (GDPR Art. 9) and explicit
   consent from every enrolled person. Without it, `--enroll-speaker` errors
   out and `--speaker-db` is ignored (with a notice pointing you at
   `--diarize-speakers`). Enrollment records the attestation + timestamp in
   the `.spkr` profile (v2 format) as an audit trail.
2. `--expect-speakers "NameA,NameB"` — the **closed roster**. Matching runs
   only against these enrolled profiles; naming nobody is a hard error. An
   open-ended scan of the whole database is unsupported by design (see the
   legal section below). Claimed names with no enrolled profile are warned
   about and skipped.
3. **Recorded files only.** `--speaker-db` and `--enroll-speaker` refuse to
   run in streaming/live mode — real-time identification is deliberately
   unsupported.

```bash
# Enroll a reference clip (writes <db>/Alice.spkr — a stored voiceprint,
# with the consent attestation recorded in the file):
crispasr -f alice-sample.wav --enroll-speaker Alice --speaker-db ./voiceprints \
    --speaker-db-consent

# Later, label a meeting of Alice and Bob (both enrolled, both consented).
# --speaker-db with --diarize implies global clustering (--diarize-embedder
# auto), so this is the full shared pipeline:
crispasr -m auto -f meeting.wav --diarize-speakers \
    --speaker-db ./voiceprints --expect-speakers "Alice,Bob" \
    --speaker-db-consent -ojf
# -> (Alice) ... / (Bob) ... / (speaker 2) ... for any third voice
```

Tune the match with `--speaker-threshold` (default `0.7`; a cluster whose
centroid scores below it stays anonymous). Enroll and identify must use the
same embedder family (TitaNet 192-d by default) so dimensions match; a
mismatched `--diarize-embedder` (e.g. 512-d ECAPA) simply never matches.

### Caveats

- **Over-split clusters can both get the same name.** Global clustering is
  imperfect: if it splits one physical speaker's audio into two clusters
  (e.g. their voice drifts across a long recording), each cluster is matched
  independently against the roster, so **both** may match the same enrolled
  name. This is intentional, not a bug — both clusters genuinely are that
  speaker, and per-cluster matching has no way (or need) to notice they were
  once the same person.
- **Very short segments keep their local diarize label.** Segments shorter
  than about 0.25 s can't be embedded (below TitaNet's reliable floor), so
  they never take part in cluster matching and keep whatever local
  `(speaker N)` label the diarizer assigned. A transcript can therefore
  occasionally mix a matched `(Alice)` with a leftover `(speaker N)` for the
  same physical speaker, on short interjections ("mm-hmm", "yeah").

### Legal & privacy obligations (not legal advice)

Storing voiceprints to identify named people means you are processing
**biometric data for the purpose of uniquely identifying a natural person** —
**special-category data under [GDPR Article 9](https://iapp.org/news/a/biometrics-in-the-eu-navigating-the-gdpr-ai-act)**.
This holds regardless of how the EU AI Act classifies the system. In practice
that means, at minimum:

- **Explicit, freely-given, revocable consent** from every enrolled person
  (an Art. 9(2) basis), obtained before enrollment.
- A clear **retention and deletion** policy and an easy way to honour
  data-subject deletion requests (delete the relevant `.spkr` file).
- **Transparency**: tell people they are being identified.

On the **EU AI Act** (Regulation (EU) 2024/1689): *remote biometric
identification* — identifying people **without their active involvement**
against a reference database ([Art. 3(41)](https://artificialintelligenceact.eu/article/3/),
[Recital 17](https://ai-act-service-desk.ec.europa.eu/en/ai-act/recital-17)) —
is **high-risk** under [Annex III(1)(a)](https://artificialintelligenceact.eu/annex/3/),
with heavyweight provider and deployer obligations. CrispASR therefore does
not implement that kind of system, and the constraints above are what keep
this feature outside it:

- **Closed claimed roster only** (`--expect-speakers` is mandatory): the tool
  confirms which *asserted, actively-enrolled, consenting* participants speak
  in a recording. It cannot answer "who is this unknown voice?" — the
  open-ended 1:N search that characterizes an identification system is not
  implemented, at the CLI or at the C API.
- **Active involvement**: enrollment is a deliberate act by the enrolled
  person (consent + a provided sample), recorded in the profile.
- **Post-processing only**: no real-time/streaming identification path
  exists (cf. the Art. 5(1)(h) prohibition on real-time RBI in public
  spaces).

**Intended purpose** — and the only supported one — is cooperative labeling
of consenting, enrolled participants in recordings they know are being
transcribed (meeting minutes, interview archives, podcast production).
Identifying unknown persons, surveillance, law-enforcement use, or processing
publicly-scraped audio are **out of scope and unsupported**; do not attempt to
repurpose the feature for them. Get legal review before deploying the named
path commercially.

To stay clearly on the safe side, prefer **section 1** and rename manually.

The full AI Act position — including why the constraints above keep this
outside Annex III(1)(a), and what the Act requires of you as deployer — is in
[`eu-ai-act.md`](eu-ai-act.md).

---

## Implementation notes

- Session clustering: `crispasr_remap_speakers_via_embeddings()` in
  `examples/cli/crispasr_diarize_cli.cpp` — per-recording embedding extraction
  + `core_spectral::cluster_speakers()` (`src/core/spectral_diarize.h`, the
  count-estimating spectral clusterer foxnose also uses), falling back to
  `crispasr_agglomerative_cluster()` (`src/crispasr_speaker_cluster.cpp`) only
  when `--diarize-cluster-threshold` was passed explicitly. No persistence.
- Embedder adapters (pluggable): `src/crispasr_speaker_embedder.{h,cpp}`
  (TitaNet-Large 192-d default; IndexTTS-BigVGAN ECAPA-TDNN 512-d).
- Named profiles: `src/speaker_db.{h,cpp}` — the `.spkr` on-disk format
  (v2 adds the consent attestation + enrollment timestamp) and cosine
  matching. `speaker_db_retain()` narrows a loaded db to the claimed roster
  before any match. Enroll and identify must use the **same** embedder so
  dimensions match.
- Cluster identification: `crispasr_identify_speaker_clusters()` /
  `crispasr_identify_single_speaker()` in
  `examples/cli/crispasr_diarize_cli.cpp`, driven from
  `crispasr_apply_global_speaker_stages()` in `examples/cli/crispasr_run.cpp`
  (one post-merge stage shared by the sequential and parallel output paths).
- C API: `crispasr_speaker_db_open(dir, expected_names_csv, consent_attested)`
  and `crispasr_speaker_db_enroll2(..., consent_attested)` are the only entry
  points; the pre-#266 ungated `_load`/`_enroll` symbols refuse at runtime.

See [`cli.md`](cli.md#diarization) for the full diarization flag reference.

## Splitting a segment that spans two speakers (#395)

Diarization **labels** the segments you hand it. That means the label
resolution can never be finer than your own segment grid: a segment that
straddles a speaker change is awarded to whoever holds the majority of it, and
the other speaker's words are silently absorbed. It is not a clustering
failure — FoxNose finds the boundary, the caller's grid just cannot express it.

You cannot fix this by sending a finer grid, either. FoxNose skips any span
shorter than `kMinSegmentSeconds = 0.4 s`
(`src/core/foxnose_pipeline.h`), so a per-word grid starves the embedder and
the clusterer under-counts speakers. Coarse segments cluster well and label
coarsely; fine segments cluster badly. The way out is the third option: label
coarse, then **split on the turns the method derived from the audio**.

FoxNose produces those turns anyway — they are what the labelling reads. Ask
for them and you get the audio's own boundaries, independent of your grid:

| Surface | Entry point |
|---|---|
| C++ | `crispasr_diarize_segments(..., std::vector<CrispasrDiarizeTurn>* out_turns)` |
| C ABI | `crispasr_diarize_segments_turns_abi(...)` (0.8.30+) |
| Rust | `crispasr::diarize_segments_with_turns(...) -> Result<Vec<DiarizeTurn>, String>` |
| Go | `whisper.DiarizeSegmentsWithTurns(...) ([]DiarizeTurn, error)` |
| Python | `diarize_segments_with_turns(...) -> (bool, list[DiarizeTurn])` |
| Dart | `diarizeSegments(..., outTurns: turns)` |
| Java/JNA | `Lib.crispasr_diarize_segments_turns_abi(...)` |

All surfaces are additive: the segments come back labelled exactly as they would
without the turns, and every other method reports zero turns rather than an
error. Turn timestamps are on the **same absolute timeline as your segments**
(the ABI adds `slice_t0_cs` back on the way out), so they compare directly.

With word timings, the recipe is: merge words into utterance-length spans,
then cut each merged run wherever the turn id changes rather than only at a
gap threshold, re-merging within those bounds so every piece still clears the
0.4 s floor — a piece that cannot reach it joins the neighbour it overlaps
most instead of being dropped.

The C ABI is a **new symbol**, not a signature change, so existing callers are
untouched (the same append-only convention as `crispasr_diarize_opts_abi`).
Its turn buffer is caller-allocated: pass `out_n_turns` to learn the count,
`out_turns` + `n_turns_cap` to receive them, and expect `2` when the buffer
was short — the segments are still labelled, `*out_n_turns` holds the required
capacity, and a retry costs a second full pass because the ABI keeps no state
between calls. The Rust, Go, Python, and Dart wrappers size the buffer from
the audio length
(one slot per 0.5 s, above FoxNose's 0.6 s embedding hop) and retry once on
`2`, so their callers never see any of that.

Java exposes both calls at its low-level JNA surface. JavaScript and Ruby do
not currently expose standalone diarization; their unused native declarations
are not public binding APIs.

## pyannote segmentation: chunked inference and the powerset layout (#326)

Two changes landed together here, one a speed fix and one a correctness fix
that was found while measuring the first.

### Chunked parallel inference

`pyannote_seg_run` used to push the entire file through as ONE sequence. The
segmentation net is SincNet → 4 stacked bidirectional LSTMs → classifier, and
an LSTM recurrence is inherently sequential, so the dominant stage ran on
exactly two threads (one per direction) no matter what `-t` said. At
16.875 ms per frame a 48-minute recording is 171k timesteps, ~70% of
segmentation time, with every core past the second one idle.

Audio is now cut into fixed 60 s chunks that are inferred concurrently, each
with 5 s of real audio spliced on either side and then trimmed. That context
absorbs the two edge effects — the k=5 convolutions' zero padding and the LSTM
starting from a zero hidden state.

Measured on a 2888 s file (M1, 8 cores), whole-file vs chunked:

| `-t` | whole-file | chunked |
|---|---|---|
| 1 | ~50 s | 47.0 s |
| 2 | ~50 s | 25.2 s |
| 4 | ~50 s | 18.5 s |
| 8 | 49.8–56.1 s | 18.1 s |

Two properties worth relying on:

* **Output does not depend on `-t`.** Chunking is decided by audio length
  alone; the thread count only picks how many chunks are in flight. `-t 1`,
  `2`, `4` and `8` produce byte-identical posteriors on a 2888 s file.
* **Accuracy does not regress.** On the VoxConverse dev shard (8 files, DER vs
  human labels) every chunked setting beat the single scan:
  whole-file 33.37%, 60/ctx2 32.88%, **60/ctx5 30.67%**, 60/ctx10 31.87%,
  120/ctx5 31.08%, 120/ctx10 29.88%.

  ⚠ Read those as a RELATIVE A/B of the segmenter only, never as pipeline
  quality. They score the raw powerset posteriors with the local speaker
  tracks taken as global identity — no embedder, no clustering. The shard
  averages 4.5 speakers per file and the segmentation head models at most 3
  LOCALLY, so ~30% is that harness's floor, not the product's. Scored end to
  end on the same files and the same scorer, what actually ships is:

  | path | mean DER |
  |---|---|
  | raw posteriors, no clustering (the A/B harness above) | 33.37% |
  | `--diarize-method pyannote --diarize-embedder auto` | **7.81%** |
  | `--diarize-method foxnose` (#324) | **7.32%** |

  The two shipped paths are now within half a point of each other. The
  pyannote+TitaNet figure was **15.74%** until `a719c89d`: it over-clustered,
  pinning to the `--diarize-max-speakers 8` cap on 4 of the 8 files (esrit 8 vs
  5 real, mesob 8 vs 4, nnqfq 8 vs 5, fsaal 8 vs 7), because single-linkage at a
  fixed 0.5 threshold never reached the threshold and the cap became the answer.
  Routing it through the same `core_spectral::cluster_speakers` that foxnose
  uses — which ESTIMATES the count — closed most of the gap.
  (The 7.32% here is foxnose labelling whisper-tiny's ASR segments; #324's
  3.18% scored foxnose's own turns directly, without ASR segmentation as a
  ceiling. Different measurement, not a regression.)

  Both rows are reproducible from a clean corpus extraction with
  `tools/der_voxconverse.py` (`--prepare` pulls the 8 dev files and their human
  labels straight out of the HuggingFace parquet shards). Re-measured that way
  on 2026-08-03: mean 7.81%, per-file counts 5/6/3/4/5/4/3/3 against ground
  truth 5/7/4/4/5/4/5/2 — the counts no longer pin to the cap, but the
  estimator now UNDER-counts on 3 of 8. That is the remaining accuracy item
  (root `PLAN.md` NOW §2).

This also moves toward pyannote's own design rather than away from it:
upstream infers on a sliding 10 s window, and one continuous 48-minute scan
was the outlier.

Tunable with `CRISPASR_PYANNOTE_CHUNK_S` (0 restores the single scan) and
`CRISPASR_PYANNOTE_CHUNK_CONTEXT_S`.

Because each chunk numbers its local speakers arbitrarily, chunks are stitched
by choosing, per seam, the relabelling of {spk0, spk1, spk2} that best matches
the previous chunk on the frames they both cover. That pass is sequential but
pure arithmetic over a few seconds of frames, so it costs nothing next to the
forward passes.

### The powerset class layout was wrong

The segmentation head emits one probability per *subset* of the ≤3 locally
active speakers, enumerated by increasing subset size:

```
0 = {}        1 = {spk0}      2 = {spk1}      3 = {spk2}
              4 = {spk0,spk1} 5 = {spk0,spk2} 6 = {spk1,spk2}
```

`SPK_MASK` had **3 and 4 swapped** — it read class 3 as "spk0+spk1" and class 4
as "spk2 alone". Every frame in which the third local speaker was talking
alone was therefore credited to the first two speakers as if they were talking
over each other, and real overlap was credited to a speaker who was silent.

The ground truth gives the layout away, because the implied overlap fraction
is only plausible under one reading of the table:

| file | GT overlap | as `{4,5,6}` (correct) | as `{3,5,6}` (old) |
|---|---|---|---|
| fsaal | 0.42% | 0.17% | 62.65% |
| jyirt | 0.09% | 0.00% | 28.32% |
| mesob | 28.84% | 34.05% | 0.00% |
| nnqfq | 14.40% | 10.14% | 48.40% |

Fixing it moved mean DER on that shard from **48.21% to 33.37%**.

The bug survived because every existing test used at most two speakers, where
the wrong table and the right one agree. The layout now lives in one place —
`src/core/powerset.h` — with `tests/test-powerset.cpp` guarding the
singleton/pair split, bijectivity, and permutation closure. Nothing downstream
should hard-code a class number; derive it.
