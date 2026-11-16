# Guitar tablature in CrispASR — model scoping + integration spec (§GT1)

Scoping pass for a `--tab` task surface, answering a caller who wants CrispASR
to ship an **emission scorer** — a model that emits per-note or per-frame
`(string, fret)` scores — while their Viterbi/DP decoder applies the hard
constraints (one note per string, fret range, capo) and picks the path.

Every number below is from a primary source (paper PDF or repo), fetched
2026-07-20 and adversarially verified: 104 claims extracted, 25 verified by
independent 3-vote refutation, **5 killed**. Claims that died are recorded in
§8 rather than quietly dropped, because two of them were the *obvious* readings
and one was mine.

**Bottom line up front.**

| arm | verdict |
|---|---|
| **Audio → tab** | ✅ Adopt the contract as proposed. TabCNN *is already* an emission scorer. |
| **Symbolic → tab** | ⛔ Do NOT ship a DadaGP-derived model — licensing. And the split fits badly. |
| **Classical DP + learned emissions** | 🎯 The interesting move. An empty slot nobody has filled. |

---

## 0. RESOLUTIONS (2026-07-20) — both blockers cleared, audio arm is GO

The two gates in §5 were reading tasks. Both are now answered from primary
sources, and the answers are better than the spec assumed.

### R1 ✅ Licensing is clean end-to-end — the audio arm survives

Every corpus in the audio chain is **CC BY 4.0**, open access, no NonCommercial
and no ShareAlike:

| artifact | licence | note |
|---|---|---|
| **GuitarSet** v1.1.0 (Zenodo 3371780) | Creative Commons Attribution 4.0 International | training corpus for every audio candidate |
| **EGSet12** (Zenodo 11406378) | Creative Commons Attribution 4.0 International | ⭐ **also ships the trained model** |
| **Guitar-TECHS** (arXiv 2501.03720) | CC-BY-4.0 | dataset |
| **GOAT** (arXiv 2509.22655) | CC BY 4.0 | dataset, 5.9 h direct-input audio |

The EGSet12 record contains not just audio + tablature annotations but **a
trained machine-learning model** — i.e. the GuitarProFX-augmented TabCNN of
§1.3, the exact variant §1 recommends shipping, under CC BY 4.0. **That is the
shippable artifact, and its provenance is clean.** Attribution is required
(registry `license` field + model card), nothing more.

⚠ **Code licences differ from weight licences here — mind the gap.**

- FretNet (`cwitkowitz/guitar-transcription-continuous`): **MIT**, code only, no
  weights.
- TabCNN (`andywiggins/tab-cnn`): **no licence file found** — treat as
  all-rights-reserved until verified. Code only; it tells you to train yourself.

So the correct build path is the repo's existing **clean-room protocol** (the
Transcoda precedent: permissive weights, non-permissive reference code): take
the *weights* from the CC BY 4.0 Zenodo record, implement the graph in ggml from
the **ISMIR paper** (§3.3–3.5 fully specify it), and use the MIT FretNet repo —
not the unlicensed TabCNN repo — as the readable reference. Do not transcribe
unlicensed source.

### R2 ✅ Nothing in the 2024–26 wave supersedes TabCNN

- **TART** (arXiv 2510.02597, Oct 2025) — the presumed direct competitor.
  **It is not a competitor on this axis.** No GuitarSet 6-fold player-wise CV
  (it uses plain 80/20 splits), **no tablature P/R/F1 and no TDR at all** (only
  note-level P50/R50/F50 for its audio→MIDI stage), and **no comparison against
  TabCNN or FretNet**. Its final tablature stage is a *rule-based generator*
  validated on "synthetic MIDI example".
  Crucially, its string/fret stage **is the Fretting-Transformer** — the same
  autoregressive T5 already in §2.4 — trained on "SynthTab and DadaGP", so it
  **inherits the §2.3 licensing blocker**. Tab accuracy **42.1 %** on the DadaGP
  acoustic subset. No code, no weights, no licence.
  → TART is a *pipeline paper*, and it independently reproduces this spec's
  central finding: even the newest "comprehensive audio-to-tab tool" delegates
  string/fret to an autoregressive symbolic model, and scores poorly.
- **Guitar-TECHS** and **GOAT** are **datasets, not models**. Neither benchmarks
  against TabCNN/FretNet on GuitarSet 6-fold.

**So TabCNN (GuitarProFX-augmented) stands as the audio-arm recommendation, and
§6's comparability warning gets sharper, not weaker: TART cannot be tabled
against TabCNN either.**

### R3 🎯 A clean-room corpus without DadaGP now exists

Open question §5.3 asked whether training could avoid DadaGP. The answer arrived
with R2: **GuitarSet + EGSet12 + Guitar-TECHS + GOAT are all CC BY 4.0**. That
is a genuine multi-source, commercially-usable corpus — which did not exist when
the DadaGP-dependent symbolic models were trained. It does not make the symbolic
*architecture* fit the contract any better (§4.2 stands), but it removes the
licensing reason to abandon the symbolic arm permanently, and it is exactly the
material a purpose-trained sequence labeler (§4.3 option a) would need.

### What remains open

Only §5.3's second half and the §7.4 metric question. Nothing blocks starting
the audio arm.

---

## 1. Audio → tab: the contract is a perfect fit

### 1.1 TabCNN is the emission scorer, not something to be adapted into one

`TabCNN` (Wiggins & Kim, ISMIR 2019) — ~0.8 M parameters. 192×9 CQT frame
window → 3 conv layers → dense-126 reshaped to **6×21**, with a softmax applied
to each of the 6 rows: "six probability mass functions, which represent the
probability of each fret class for each string" (§3.4). The 21 classes are
"open, closed, or any one of the 19 frets" (§3.3). Loss is six summed
independent categorical cross-entropies (§3.5).

That means, verbatim from the architecture:

- no inter-string coupling term,
- no CRF,
- no temporal model over outputs,
- **no decoding step of any kind.**

The published metrics are computed on a plain argmax of that layer. A
constrained Viterbi consuming the same layer is therefore a strict improvement
over what the paper reports, not a lossy adaptation of it.

- Paper: <https://archives.ismir.net/ismir2019/paper/000033.pdf>
- Code: <https://github.com/andywiggins/tab-cnn>

### 1.2 The GuitarSet numbers

GuitarSet 6-fold **player-wise** CV, 472,560 test frames (hold out one of the
6 guitarists, train on the remaining 5):

| metric | P | R | F |
|---|---|---|---|
| tablature | 0.809 ± 0.029 | 0.696 ± 0.061 | **0.748 ± 0.047** |
| multipitch | 0.900 ± 0.016 | 0.764 ± 0.043 | **0.826 ± 0.025** |

TDR ("Tablature Disambiguation Rate", the paper's own term) **0.899 ± 0.033**.
Deep Salience multipitch F = 0.646 for reference. All unsmoothed — these are
raw-emitter numbers.

### 1.3 ⚠ The finding that actually decides the product: GuitarSet overstates

Zero-shot on **EGSet12** (real electric guitar performances, DAFx-24, peer
reviewed — Pedroza, Abreu, Corey, Roman):

| | GuitarSet | EGSet12 zero-shot |
|---|---|---|
| tablature F1 | 0.748 | **0.447** |
| multipitch F1 | 0.826 | 0.638 |
| TDR | 0.899 | 0.695 |

**The fix is data, not architecture.** Re-rendering the training audio with real
electric guitar tones and effects (GuitarProFX) raises EGSet12 tablature F1 to
**0.585** and TDR to **0.819**, with architecture, optimizer, LR, batch size and
validation data all held constant.

So the variant to ship is the **GuitarProFX-augmented TabCNN**, not the vanilla
one — and its weights are public.

- <https://arxiv.org/html/2405.14679> · <https://dafx.de/paper-archive/2024/papers/DAFx24_paper_99.pdf>
- EGSet12: <https://zenodo.org/records/11406378>

Honest limits on that result (§7): EGSet12 is 12 tracks / 379.8 s / **one**
guitarist / one signal chain, per-metric σ 0.06–0.11. Only the tablature-F1 gain
clears p<0.05; the headline TDR gain is 0.1>p>0.05. And GuitarProFX adds data
*volume* as well as timbre diversity, so "real tones" is confounded with "more
data". The direction is solid; don't quote the magnitude as precise.

### 1.4 FretNet does not supersede it on the headline metric

FretNet (Cwitkowitz et al., ICASSP 2023), like-for-like — FretNet's authors
re-implemented and re-trained TabCNN under their own identical six-fold
player-wise protocol, so this *is* a fair comparison:

| | TabCNN | FretNet |
|---|---|---|
| tablature F1 | 0.717 | **0.727** |
| frame multipitch F1 | **0.820** | 0.818 |
| string-dependent **note** F1 | 0.430 | **0.506** |

FretNet's real advance is at **note level** (driven by an explicit onset head
TabCNN lacks) and at tight continuous-pitch tolerances — not on the frame-level
tab metric. Pick FretNet if note-level onsets matter; pick TabCNN if frame-level
tab emissions feeding a DP are the deliverable.

⚠ **FretNet's tablature head is NOT a per-string softmax.** A decoder must not
assume the TabCNN output shape. Code:
<https://github.com/cwitkowitz/guitar-transcription-continuous>

---

## 2. Symbolic → tab: weaker evidence, and a live legal problem

### 2.1 DadaGP is a dataset, not a benchmarked model

DadaGP (Sarmento et al., ISMIR 2021) contributes a corpus + tokenizer. Its only
evaluation counts grammar errors (repeated once-only tokens) across training
epochs — which the authors themselves call "simple" and "limited". There is:

- no playability metric,
- no fingering-cost or hand-position metric,
- **no comparison against any baseline**, neural or classical.

It therefore provides **zero evidence** that neural models beat classical
fingering search. Any claim otherwise does not come from this paper.

### 2.2 The tokenization is lossy where it matters commercially

- key signature unreliable — **93.7 % auto-assigned C Major**
- MIDI-style velocity / dynamics unsupported
- vocals and lyrics dropped
- rare tunings dropped (the encoder cannot represent them)
- time signature inferred from wait-token sums — **cannot disambiguate 3/4 from 6/8**
- `measure:repeat` frequently misinterpreted → disproportionate repetitions

### 2.3 ⛔ LICENSING — the blocker

DadaGP is **not openly redistributable**, and shipping derived weights
commercially is a live risk:

- Zenodo access is **request-gated, "FOR RESEARCH PURPOSES"**
- **no license covers the 26,181 scraped GuitarPro files**
- the MIT license on the `dadaGP` GitHub covers only the **encoder/decoder code**
- the CC BY-4.0 on the arXiv listing is arXiv's **manuscript** license, not a
  corpus license
- the authors **explicitly leave commercial fair use unresolved**

This is the same shape as the BTC chord-weights problem (PLAN.md, "The chord
problem is DATA provenance, not code") — except BTC's weights were at least
under a nameable license (CC-BY-NC-SA) that `--accept-license` could gate.
Here there is **no license to gate on**. An `--accept-license` tag cannot
launder an unlicensed scrape.

### 2.4 What the symbolic SOTA actually is

| model | size | result | weights |
|---|---|---|---|
| **MIDI-to-Tab** (ISMIR 2024) | BART-style, 384/6/6/1536, ~12–25 M | string agreement **73.58 %** vs Guitar Pro 8 62.27 / MuseScore 62.51 / TuxGuitar 55.42 | ❌ none |
| **Fretting-Transformer** (ICMC 2025) | ¼-width/½-depth T5, d_model=128, d_ff=1024, 3 enc-dec layers, 4 heads | tab accuracy 72.19 % post-processing | ❌ none |

Both are comfortably in ggml/GGUF size range. **Neither releases weights**, and
both depend on DadaGP. The only weight release found is a third-party
reimplementation (`github.com/Sidmaz666/open-fret`) which is not the authors'
weights and inherits the same provenance risk.

MIDI-to-Tab wins a 15-guitarist playability study but **does not reach human
ground truth** (6.04 vs 7.45).

⚠ Fretting-Transformer's **100.00 % post-processing pitch accuracy is achieved
BY CONSTRUCTION** — the fallback forces a viable string/fret pair for the
tuning. Do not quote it as model capability. Neighbor search contributes
+0.04 pp (72.15 → 72.19); overlap correction does essentially all the work.

---

## 3. The classical baseline has an empty slot shaped like our contract

The canonical HMM guitar-fingering decoder has **no learned emission scorer at
all**. Hidden states are left-hand forms `(string, fret, finger)` whose pitch is
a *deterministic function of the state*, so emission probabilities are
degenerate **0/1** and **all** playability knowledge lives in hand-designed
transition probabilities.

That is an empty slot that learned per-`(string, fret)` emissions drop into
**without changing the DP at all**.

And the classical line is essentially unbenchmarked: parameters were
hand-designed rather than learned because aligned score+tablature data did not
exist in 2016, and evaluation is qualitative on three monophonic example
phrases — no dataset, no quantitative playability metric, no ground-truth
comparison.

**Consequence:** there is no rigorous head-to-head anywhere in the literature
establishing that neural models beat classical search on measured playability.
The 2016 data-scarcity objection no longer holds, and nobody in the surveyed
literature has run the obvious experiment.

---

## 4. Verdict on the integration contract

### 4.1 Audio arm — adopt as proposed ✅

TabCNN and FretNet natively emit per-frame per-`(string, fret)` scores with
decoding held strictly outside the network, and both papers explicitly leave
smoothing/decoding to a downstream step. The caller's Viterbi consuming those
emissions is a strict improvement over the published argmax.

```
CrispASR                                   caller
────────────────────────────────────────   ──────────────────────────────
audio ──► CQT ──► TabCNN ──► [T, 6, 21]    ──► Viterbi/DP over the same
                             log-probs          matrix, applying:
          (no decoding, no smoothing)             • one note per string
                                                  • fret range / capo
                                                  • hand-span transitions
                                                ──► List<Fretting>
```

Emissions are **log-probabilities**, not probabilities — a DP sums costs, and
handing over pre-softmax logits or raw probabilities invites the caller to take
logs of zeros. Ship `log_softmax` output.

### 4.2 Symbolic arm — the split is lossy and awkward, not impossible ⚠

Both strong symbolic systems are **autoregressive token decoders**:
Fretting-Transformer emits `TAB<<<#,#>>>` tokens (string *and* fret in one
token) interleaved with `TIME_SHIFT`, and a T5 decoder's logits are conditioned
on its own previously emitted TAB tokens. That violates the conditional
independence a Viterbi/DP needs — constraint enforcement becomes constrained
*sampling* during generation, or a post-hoc repair of a commitment the model
already made.

But — and this is why the verdict is "awkward" not "fatal" — **symbolic SOTA
already depends on an external rule-based constraint pass to be playable at
all.** Fretting-Transformer's raw output reaches only **97.23 %** pitch
accuracy: it emits string/fret pairs that *do not reproduce the target pitch*.
External overlap correction lifts tab accuracy 68.56 % → 72.19 %.

So the field's own practice puts hard constraints outside the model. The
disagreement is only about whether what crosses the boundary is a **score
matrix** (our contract) or a **committed token sequence needing repair** (their
practice).

### 4.3 Recommended shape

- **Audio arm:** ship TabCNN (GuitarProFX-augmented) as a `--tab` emission
  scorer. Clean fit, public weights, ~0.8 M params.
- **Symbolic arm:** do **not** ship a DadaGP-derived model. Either
  (a) keep the caller's existing DP and fill its empty emission slot with a
  small purpose-trained sequence labeler (BiLSTM-CRF or tiny transformer over
  note columns — this *does* fit the contract, unlike an autoregressive
  decoder), or (b) leave it heuristic. Option (a) is also the unrun experiment
  from §3, so it is a contribution rather than a port.

---

## 5. Blockers — ✅ both cleared, see §0

1. ~~**GuitarSet's license is UNVERIFIED**~~ ✅ **RESOLVED (R1)** — CC BY 4.0,
   open access, no NC/SA. So are EGSet12, Guitar-TECHS and GOAT. The audio arm
   survives, and EGSet12 ships the recommended trained model under CC BY 4.0.
   Residual care: TabCNN's *code* repo has no licence file — use the clean-room
   path (paper + MIT FretNet as reference), not that source.
2. ~~**The 2024–26 wave is unverified**~~ ✅ **RESOLVED (R2)** — TART does not
   benchmark on GuitarSet 6-fold, reports no tablature F1 or TDR, never compares
   to TabCNN/FretNet, and delegates string/fret to the DadaGP-trained
   Fretting-Transformer. Guitar-TECHS and GOAT are datasets. Nothing supersedes
   TabCNN. Still unread: "A Machine Learning Approach for MIDI to Guitar
   Tablature Conversion" (arXiv 2510.10619) — symbolic arm only, so it cannot
   change the audio recommendation.
3. **Can a clean-room corpus avoid DadaGP entirely?** ✅ **Materially yes (R3)** —
   GuitarSet + EGSet12 + Guitar-TECHS + GOAT are all CC BY 4.0. What is still
   open is the *sufficiency* question: is that enough data to train a usable
   emission scorer.

### R4 SynthTab — the METHOD is the prize, the DATASET is unusable

SynthTab (Zang, Zhong, Cwitkowitz & Duan, ICASSP 2024, arXiv 2309.09085,
synthtab.dev) is **both** a synthesis pipeline and a dataset. It renders
symbolic tablature into audio that "faithfully adheres to the original
fingerings and a subset of techniques", so the fingering ground truth is exact
*by construction* rather than annotated. Scale is enormous next to GuitarSet's
~3 hours: **13,113 hours across 20,715 tracks and 23 timbral profiles.**

Reported result: TabCNNx4 pre-trained on SynthTab then fine-tuned on GuitarSet
reaches **tablature F1 80.1 %, multi-pitch F1 84.5 %**, and the authors report
that pre-training "significantly mitigates issues with overfitting" across
datasets.

⚠️ Do not read 80.1 % as a pure pre-training delta against TabCNN's 0.748:
**TabCNNx4 is a 4x-widened variant**, not the model in §1.1, and the paper does
not establish that its protocol matches the 6-fold player-wise CV of §1.2. It
is evidence that synthetic pre-training helps, not a like-for-like number.

**Licensing — TWO stacked encumbrances, and no source addresses either.**

1. It is *derived from DadaGP* ("we used a portion of DadaGP, which contains
   26,181 songs"), so it inherits §2.3 in full.
2. It is rendered with **commercial Ample Sound plugins** (acoustic L/T/M/SJ,
   electric SH/LP/TC/VC/PF/SC/E). Whether 13,000 hours of their rendered output
   may be redistributed as a corpus is governed by that EULA, not by us.

The paper states no licence, no redistribution terms and no copyright position;
synthtab.dev likewise states none and simply offers "Download Full SynthTab
Set". Silence is not a grant. **SynthTab-derived weights are not shippable.**

**But the pipeline answers the question the dataset cannot.** A renderer that
turns symbolic tab into audio with exact fingering labels is precisely what a
clean-room corpus needs: point it at licensable symbolic input instead of
DadaGP, and at a plugin whose EULA permits it, and it produces training audio
we may actually ship from. That is the strongest lead for the audio arm beyond
EGSet12 — and it converges with §1.3: SynthTab's timbral diversity and
GuitarProFX's real-tone re-rendering are the same finding (timbre variety, not
architecture, is what generalises), reached independently, with Cwitkowitz on
both papers.

---

## 6. ⚠ Comparability — do not build a league table

This is the single biggest hazard in this literature.

- ✅ **TabCNN vs FretNet IS comparable** — FretNet's authors re-implemented and
  re-trained TabCNN under their own identical protocol.
- ❌ **TabCNN vs Burlet & Hindle is NOT** (multipitch F 0.71 raw / 0.77
  HMM-smoothed) — different dataset, and Wiggins & Kim say so explicitly.
- ❌ **Pre-2019 tablature comparisons do not exist** — TabCNN *introduced*
  ptab/rtab/ftab/TDR. "there are no prior approaches to directly compare these
  metrics to" (§4.4).
- ❌ The Deep Salience 0.646 baseline was **not re-run** by Wiggins & Kim; it is
  quoted from prior work by the GuitarSet authors.
- ❌❌ **Symbolic and audio share NO metric.** MIDI-to-Tab reports string-assignment
  agreement on 9 held-out pieces; Fretting-Transformer reports tab accuracy on
  GuitarToday/Leduc/DadaGP; neither runs GuitarSet 6-fold CV, and MIDI-to-Tab
  *consumes GuitarSet's training split as fine-tuning data*. **Never put
  symbolic and audio numbers in the same table.**

---

## 7. Acceptance gates if we build this

Following the CREPE/BTC precedent in PLAN.md — a converter parity tool, then a
per-stage diff, then a real-music acceptance run:

1. **Converter parity** — `tools/tabcnn_torch_parity.py`, cos vs the reference
   on the CQT front end *and* the 6×21 output. Note the CQT lesson from BTC:
   cosine and peak-bin match are **scale-invariant**, so assert on the median
   per-bin magnitude ratio too (`core/cqt.h` shipped a 152× scale bug that
   correlation could not see).
2. **Per-stage diff** — `crispasr-diff tabcnn` against a dumped reference,
   registered in `crispasr_diff_main.cpp`. Per the voxcpm2-vae finding: a
   reference dumper with **no C++ consumer is dead code that looks like
   coverage** — wire both halves or neither.
3. **Real-music acceptance** — GuitarSet 6-fold is the training protocol, so it
   is not an honest acceptance set on its own. Report **EGSet12 zero-shot**
   alongside it; that is the number that predicts field behaviour.
4. **Decoder regression metric** — open question. The audio family uses
   frame-level TDR / tablature F1, the symbolic family uses string-assignment
   agreement, and they do not compose. Pick one and state it, rather than
   reporting whichever flatters.

---

## 8. What was refuted (recorded, not hidden)

Five claims died under 3-vote adversarial verification. Two matter here:

- **"DadaGP's baked-in `instrument:note:string:fret` token format makes the
  emission-scorer split fatally adverse"** — refuted **0-3**. This was the
  session's own initial reading and it was too strong. It does not generalize:
  DadaGP/GTR-CTRL decoder-only LMs use a different vocabulary from the two
  strongest symbolic systems.
- **"MIDI-to-Tab's top-2 beam search validates the emission-scorer split"** —
  also refuted **0-3**.

Both *extremes* lost, which is why §4.2 lands on "lossy and awkward". Also
refuted: that neural beats A*/commercial tools across all three symbolic test
sets (1-2), and that above-12th-fret failure is unrepairable by a downstream
decoder (0-3).

---

## 9. Packaging, if it goes ahead

Mirrors the `--pitch` / `--chords` / `--beats` task-shaped precedent
(`docs/contributing.md` §7 — a task surface, not a `transcribe()` overload):

- `CAP_TAB` bit; both capability-name tables in `crispasr_backend.cpp`
- `examples/cli/crispasr_tab_cli.{h,cpp}` early dispatcher, called from
  `crispasr_run_backend()` **and** from `cli.cpp` before any transcribe backend
- redirect shim `crispasr_backend_tabcnn.cpp` so `--list-backends` knows it
- **both** detect passes — `crispasr_backend.cpp` *and*
  `crispasr_detect_backend_from_gguf()` in `src/crispasr_c_api.cpp`
- session C ABI `crispasr_session_tab*`: a run call returning a count, an `n_*`
  accessor, and a **flat all-float view** (a mixed int/float struct read through
  a float view misreads the int lanes)
- registry entry with a `license` field — and see §2.3: if the corpus provenance
  cannot be named, there is nothing to gate on and it should not ship
- `python tools/gen-feature-matrix.py` (never hand-edit the matrix)
- `python tools/check-backend-wiring.py --crispasr ./build/bin/crispasr`

Emission-scorer contract at the ABI: `[T, 6, 21]` **log-probabilities**, plus
the frame hop in seconds so the caller can align to its own grid. The caller
owns the DP.
