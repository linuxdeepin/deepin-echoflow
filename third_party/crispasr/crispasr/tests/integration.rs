//! Integration tests for the CrispASR Rust wrapper.
//!
//! Requires:
//!   - whisper-tiny model at CRISPASR_MODEL env var (or ../models/ggml-tiny.en.bin)
//!   - parakeet model at PARAKEET_MODEL env var (optional, skipped if absent)
//!   - GGUF chat model at CRISPASR_CHAT_MODEL env var (optional, skipped if
//!     absent; CRISPASR_CHAT_TEST_MODEL, which the C++ chat suite reads, is
//!     honoured as a fallback so one variable covers both suites)
//!   - jfk.wav at ../samples/jfk.wav

use std::path::Path;

fn jfk_pcm() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../samples/jfk.wav");
    let mut reader = hound::WavReader::open(path).expect("failed to open jfk.wav");
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn whisper_model() -> String {
    std::env::var("CRISPASR_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/ggml-tiny.en.bin").to_string()
    })
}

fn parakeet_model() -> Option<String> {
    let p = std::env::var("PARAKEET_MODEL").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_cohere/parakeet-tdt-0.6b-v3.gguf"
        )
        .to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn omni_ctc_model() -> Option<String> {
    let p = std::env::var("OMNI_CTC_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/omniasr-ctc.gguf").to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn canary_ctc_model() -> Option<String> {
    let p = std::env::var("CANARY_CTC_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/canary-ctc.gguf").to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn wav2vec2_model() -> Option<String> {
    let p = std::env::var("WAV2VEC2_MODEL").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/wav2vec2-ctc.gguf").to_string()
    });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

/// Backend-agnostic sanity for an exposed CTC grid: correctly shaped, finite,
/// and carrying real per-frame acoustic structure (the argmax varies across
/// the clip but isn't noise every frame). Makes no assumption about which id
/// is the CTC blank, so it holds for Omni (blank 0), canary-ctc, and wav2vec2.
fn assert_real_ctc_grid(lg: &crispasr::CtcLogits) {
    assert!(lg.n_vocab > 0 && lg.n_frames > 0);
    assert_eq!(lg.data.len(), lg.n_vocab * lg.n_frames);
    assert!(
        lg.data.iter().all(|x| x.is_finite()),
        "logits must be finite"
    );

    let v = lg.n_vocab;
    let argmax: Vec<usize> = (0..lg.n_frames)
        .map(|t| {
            let frame = &lg.data[t * v..(t + 1) * v];
            (0..v)
                .max_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap())
                .unwrap()
        })
        .collect();
    let transitions = (1..lg.n_frames)
        .filter(|&t| argmax[t] != argmax[t - 1])
        .count();
    assert!(
        transitions > 0,
        "degenerate grid: constant argmax across all {} frames",
        lg.n_frames
    );
    assert!(
        transitions < lg.n_frames,
        "argmax changes every frame ({transitions}/{}): suspect noise, not a real decode",
        lg.n_frames
    );
}

// ---- CrispASR (whisper-only) tests ----

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_load_and_transcribe() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: whisper model not found at {model_path}");
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).expect("load whisper-tiny");
    let pcm = jfk_pcm();
    let segs = model.transcribe_pcm(&pcm).expect("transcribe");
    assert!(!segs.is_empty(), "should produce segments");
    let full = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    assert!(
        full.contains("fellow americans"),
        "text should mention 'fellow americans': {full}"
    );
    assert!(
        full.contains("country"),
        "text should mention 'country': {full}"
    );
}

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_timestamps_valid() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).unwrap();
    let segs = model.transcribe_pcm(&jfk_pcm()).unwrap();
    for seg in &segs {
        assert!(seg.start >= 0.0, "start >= 0");
        assert!(
            seg.end > seg.start,
            "end > start: {} vs {}",
            seg.end,
            seg.start
        );
        assert!(seg.end < 15.0, "end < 15s (audio is ~11s)");
    }
}

#[test]
#[ignore = "CrispASR (whisper-direct) API crashes in Rust — use Session API instead"]
fn whisper_empty_audio() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let model = crispasr::CrispASR::new(&model_path).unwrap();
    let silence = vec![0.0f32; 16000]; // 1s silence
    let segs = model.transcribe_pcm(&silence).unwrap();
    // Should not crash; may produce empty or whitespace-only segments
    let _ = segs;
}

// ---- Session (unified, any backend) tests ----

#[test]
fn session_whisper_auto_detect() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        return;
    }
    let sess = crispasr::Session::open(&model_path).expect("session open whisper");
    assert_eq!(sess.backend(), "whisper");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());
    let full = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    assert!(full.contains("country"));
}

#[test]
fn session_whisper_no_speech_prob() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: whisper model not found at {model_path}");
        return;
    }
    let sess = crispasr::Session::open(&model_path).expect("session open whisper");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());

    // Every whisper segment carries a real no-speech probability in [0, 1] —
    // not the -1.0 "no data" sentinel other backends leave. JFK is clean
    // speech, so the values should also sit well below the 0.6 suspect
    // threshold, confirming it is the true posterior and not a placeholder.
    for s in &segs {
        assert!(
            (0.0..=1.0).contains(&s.no_speech_prob),
            "no_speech_prob {} out of [0,1] for segment {:?}",
            s.no_speech_prob,
            s.text
        );
        assert!(
            s.no_speech_prob < 0.6,
            "unexpected high no_speech_prob {} on clean speech {:?}",
            s.no_speech_prob,
            s.text
        );
    }
}

#[test]
fn session_whisper_detected_language() {
    let model_path = whisper_model();
    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: whisper model not found at {model_path}");
        return;
    }
    let sess = crispasr::Session::open(&model_path).expect("session open whisper");
    // Whisper's in-decode acoustic language detection surfaces on the
    // exception-safe session (JFK is English).
    sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert_eq!(sess.detected_language(), "en");
}

#[test]
fn session_available_backends() {
    let backends = crispasr::Session::available_backends();
    assert!(backends.contains(&"whisper".to_string()));
    assert!(backends.contains(&"parakeet".to_string()));
}

#[test]
fn session_parakeet_word_timestamps() {
    let model_path = match parakeet_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: parakeet model not found");
            return;
        }
    };
    let sess = crispasr::Session::open(&model_path).expect("session open parakeet");
    assert_eq!(sess.backend(), "parakeet");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty());

    // Parakeet should produce word-level timestamps
    let words = &segs[0].words;
    assert!(!words.is_empty(), "parakeet should produce words");
    for w in words {
        assert!(w.start >= 0.0);
        assert!(w.end >= w.start);
        assert!(!w.text.is_empty());
    }

    // Monotonicity
    let mut prev_end = 0.0f64;
    for w in words {
        assert!(
            w.start >= prev_end - 0.02,
            "word '{}' starts at {} before prev end {}",
            w.text,
            w.start,
            prev_end
        );
        prev_end = w.end;
    }
}

#[test]
fn session_omni_ctc_logits() {
    let model_path = match omni_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: omni CTC model not found (set OMNI_CTC_MODEL)");
            return;
        }
    };
    // Auto-detect doesn't recognise every Omni GGUF on this pinned release;
    // the generic "omniasr" backend routes all CTC/LLM variants.
    let sess = crispasr::Session::open_with_backend(&model_path, "omniasr", 4)
        .expect("session open omniasr");

    // The 300M CTC model has a ~5 s positional-encoding limit (per its HF
    // card), so decode only the first ~4 s of the ~11 s clip.
    let pcm: Vec<f32> = jfk_pcm().into_iter().take(16_000 * 4).collect();

    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");

    // Accessor contract: a dense [n_vocab × n_frames] grid, correctly shaped
    // and finite.
    let lg = logits.expect("CTC backend should return Some(CtcLogits)");
    assert!(lg.n_vocab > 0 && lg.n_frames > 0);
    assert_eq!(lg.data.len(), lg.n_vocab * lg.n_frames);
    assert!(
        lg.data.iter().all(|x| x.is_finite()),
        "logits must be finite"
    );

    // Greedy CTC over the exposed logits (argmax per frame, collapse repeats,
    // drop blank id 0) must yield a non-degenerate token stream — evidence the
    // grid is the real decode input, not zeros/garbage.
    let v = lg.n_vocab;
    let mut prev: i32 = -1;
    let mut n_tokens = 0usize;
    for t in 0..lg.n_frames {
        let frame = &lg.data[t * v..(t + 1) * v];
        let best = (0..v)
            .max_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap())
            .unwrap() as i32;
        if best != 0 && best != prev {
            n_tokens += 1;
        }
        prev = best;
    }
    assert!(
        n_tokens > 0 && n_tokens < lg.n_frames,
        "degenerate greedy decode: {n_tokens} tokens over {} frames",
        lg.n_frames
    );

    // Capturing logits must not perturb the transcript.
    let plain = sess.transcribe(&pcm).expect("transcribe");
    let ptext = plain
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(ptext, text, "logits capture changed the transcript");
}

#[test]
fn session_omni_ctc_vocab() {
    let model_path = match omni_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: omni CTC model not found (set OMNI_CTC_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "omniasr", 4)
        .expect("session open omniasr");

    // Accessor contract: a non-empty vocab of raw SentencePiece pieces.
    let vocab = sess.ctc_vocab().expect("CTC backend should expose a vocab");
    assert!(
        vocab.len() > 1000,
        "unexpectedly small vocab: {}",
        vocab.len()
    );
    // Real pieces carry a word-boundary marker. The v2 Omni CTC vocab is built
    // verbatim from vocab.json and uses a literal ASCII space; v1 (SentencePiece)
    // uses U+2581 (▁). Accept either so the accessor test isn't tied to one
    // tokenizer flavour.
    assert!(
        vocab.iter().any(|p| p.contains('\u{2581}') || p == " "),
        "no word-boundary token (U+2581 piece or literal space) — not a real vocab"
    );

    // End-to-end: a greedy CTC decode over the exposed logits, detokenized via
    // the exposed vocab, must reproduce the backend's built-in transcript. This
    // proves the vocab indexing aligns with the logits argmax (same id space).
    let pcm: Vec<f32> = jfk_pcm().into_iter().take(16_000 * 4).collect();
    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");
    let lg = logits.expect("CTC backend should return Some(CtcLogits)");
    assert_eq!(lg.n_vocab, vocab.len(), "logit vocab dim != vocab len");

    // Greedy CTC: argmax per frame, collapse repeats, drop blank (id 0);
    // detokenize SentencePiece pieces with U+2581 → space, then trim.
    let v = lg.n_vocab;
    let mut prev: i32 = -1;
    let mut decoded = String::new();
    for t in 0..lg.n_frames {
        let frame = &lg.data[t * v..(t + 1) * v];
        let best = (0..v)
            .max_by(|&a, &b| frame[a].partial_cmp(&frame[b]).unwrap())
            .unwrap() as i32;
        if best != 0 && best != prev {
            decoded.push_str(&vocab[best as usize].replace('\u{2581}', " "));
        }
        prev = best;
    }
    let decoded = decoded.trim();
    assert_eq!(
        decoded, text,
        "vocab-detokenized greedy decode != built-in transcript"
    );
}

#[test]
fn session_canary_ctc_logits() {
    let model_path = match canary_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: canary-ctc model not found (set CANARY_CTC_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "canary-ctc", 4)
        .expect("session open canary-ctc");
    let pcm = jfk_pcm();

    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");

    // canary_ctc_compute_logits returns per-frame log-probabilities; the grid
    // sanity is normalization-agnostic (argmax only).
    let lg = logits.expect("canary-ctc should return Some(CtcLogits)");
    assert_real_ctc_grid(&lg);

    // Capturing logits must not perturb the transcript.
    let plain = sess.transcribe(&pcm).expect("transcribe");
    let ptext = plain
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(ptext, text, "logits capture changed the transcript");
}

#[test]
fn session_wav2vec2_ctc_logits() {
    let model_path = match wav2vec2_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: wav2vec2 model not found (set WAV2VEC2_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "wav2vec2", 4)
        .expect("session open wav2vec2");
    let pcm = jfk_pcm();

    let (segs, logits) = sess
        .transcribe_with_logits(&pcm)
        .expect("transcribe_with_logits");
    let text = segs
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(!text.trim().is_empty(), "expected a transcript");

    // wav2vec2_compute_logits returns raw pre-softmax logits.
    let lg = logits.expect("wav2vec2 should return Some(CtcLogits)");
    assert_real_ctc_grid(&lg);

    // Capturing logits must not perturb the transcript.
    let plain = sess.transcribe(&pcm).expect("transcribe");
    let ptext = plain
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(ptext, text, "logits capture changed the transcript");
}

// Shared vocab-accessor contract for a CTC backend (PR #259 made ctc_vocab
// comprehensive across omni-ctc / canary-ctc / wav2vec2 / data2vec, but only
// omni-ctc had a vocab test). Asserts: a Some, non-empty vocab of valid C
// strings, and that its length lines up with the exposed logit grid — equal
// when blank is an in-vocab id (wav2vec2 <pad>), or one less when blank is a
// separate appended index (canary-ctc blank_id). No tokenizer-flavour
// assumptions, so it stays green across backends.
fn assert_ctc_vocab_contract(sess: &crispasr::Session, pcm: &[f32]) {
    let vocab = sess
        .ctc_vocab()
        .expect("CTC backend should expose Some(vocab)");
    assert!(
        vocab.len() > 1,
        "unexpectedly small CTC vocab: {}",
        vocab.len()
    );
    // token_text must always yield a valid (possibly empty) string, never panic
    // — including an out-of-range id, which the accessor guards to "".
    assert!(
        vocab.iter().any(|p| !p.is_empty()),
        "every vocab piece was empty — accessor returned no token strings"
    );

    let (_segs, logits) = sess
        .transcribe_with_logits(pcm)
        .expect("transcribe_with_logits");
    let lg = logits.expect("CTC backend should return Some(CtcLogits)");
    // The logit grid is either the vocab (blank in-vocab) or vocab + blank.
    assert!(
        lg.n_vocab == vocab.len() || lg.n_vocab == vocab.len() + 1,
        "logit dim {} inconsistent with vocab len {} (expected == or +1 for blank)",
        lg.n_vocab,
        vocab.len()
    );
}

#[test]
fn session_canary_ctc_vocab() {
    let model_path = match canary_ctc_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: canary-ctc model not found (set CANARY_CTC_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "canary-ctc", 4)
        .expect("session open canary-ctc");
    // canary appends the blank as a separate index (blank_id), so the exposed
    // vocab is one shorter than the logit grid — covered by the shared contract.
    assert_ctc_vocab_contract(&sess, &jfk_pcm());
}

#[test]
fn session_wav2vec2_ctc_vocab() {
    let model_path = match wav2vec2_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: wav2vec2 model not found (set WAV2VEC2_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, "wav2vec2", 4)
        .expect("session open wav2vec2");
    assert_ctc_vocab_contract(&sess, &jfk_pcm());
}

#[test]
fn session_ctc_backend_no_speech_sentinel() {
    // The no_speech_prob / detected_language sentinels must hold on a real
    // NON-whisper session: CTC backends never populate the <|nospeech|>
    // posterior, so every segment must carry the -1.0 "no data" sentinel (not a
    // bogus in-[0,1] value), and detected_language must not crash — it falls
    // back to the source-language hint or "unknown". Prefer canary-ctc, else
    // wav2vec2; skip if neither model is present.
    let (model_path, backend) = match (canary_ctc_model(), wav2vec2_model()) {
        (Some(p), _) => (p, "canary-ctc"),
        (None, Some(p)) => (p, "wav2vec2"),
        (None, None) => {
            eprintln!("SKIP: no CTC model found (set CANARY_CTC_MODEL or WAV2VEC2_MODEL)");
            return;
        }
    };
    let sess = crispasr::Session::open_with_backend(&model_path, backend, 4)
        .expect("session open CTC backend");
    let segs = sess.transcribe(&jfk_pcm()).expect("transcribe");
    assert!(!segs.is_empty(), "expected a transcript");
    for s in &segs {
        assert_eq!(
            s.no_speech_prob, -1.0,
            "non-whisper backend must leave the -1.0 no_speech_prob sentinel, got {}",
            s.no_speech_prob
        );
    }
    // Fallback path: never a whisper acoustic code here; a non-empty string
    // (source hint or "unknown"), never a panic.
    let lang = sess.detected_language();
    assert!(
        !lang.is_empty(),
        "detected_language fallback must be non-empty"
    );
}

// ---- Registry + cache ----

#[test]
fn registry_lookup_parakeet() {
    let entry = crispasr::registry_lookup("parakeet").expect("registry call");
    if let Some(e) = entry {
        assert!(!e.filename.is_empty());
        assert!(!e.url.is_empty());
    }
}

#[test]
fn registry_default_bundle_omnivoice() {
    let bundle = crispasr::registry_default_bundle("omnivoice")
        .expect("bundle call")
        .expect("omnivoice bundle");
    assert_eq!(bundle.backend, "omnivoice");
    assert_eq!(bundle.artifacts.len(), 2);
    assert_eq!(
        bundle.artifacts[0].kind,
        crispasr::RegistryArtifactKind::Primary
    );
    assert_eq!(bundle.artifacts[0].filename, "omnivoice-f16.gguf");
    assert_eq!(
        bundle.artifacts[1].kind,
        crispasr::RegistryArtifactKind::Companion
    );
    assert_eq!(bundle.artifacts[1].filename, "omnivoice-tokenizer-f16.gguf");
}

#[test]
fn cache_dir_exists() {
    let dir = crispasr::cache_dir(None).expect("cache_dir");
    if let Some(d) = dir {
        assert!(!d.is_empty());
    }
}

// ---- C-ABI parity: new types from bindings-parity milestone ----

#[test]
fn lcs_dedup_empty_inputs() {
    assert_eq!(crispasr::lcs_dedup_prefix_count(&[], &[], 1), 0);
    assert_eq!(crispasr::lcs_dedup_prefix_count(&[1, 2, 3], &[], 1), 0);
    assert_eq!(crispasr::lcs_dedup_prefix_count(&[], &[1, 2, 3], 1), 0);
}

#[test]
fn lcs_dedup_overlap() {
    // prev ends with [3, 4, 5], curr starts with [4, 5, 6] -> drop 2 leading
    let prev = vec![1, 2, 3, 4, 5];
    let curr = vec![4, 5, 6, 7];
    let drop = crispasr::lcs_dedup_prefix_count(&prev, &curr, 1);
    assert!(drop >= 0, "should return non-negative");
}

#[test]
fn titanet_cosine_sim_identical() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    let sim = crispasr::titanet_cosine_sim(&a, &b);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "identical vectors should have sim ~1.0, got {sim}"
    );
}

#[test]
fn titanet_cosine_sim_orthogonal() {
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let sim = crispasr::titanet_cosine_sim(&a, &b);
    assert!(
        sim.abs() < 1e-5,
        "orthogonal vectors should have sim ~0, got {sim}"
    );
}

#[test]
fn kokoro_lang_helpers() {
    assert!(crispasr::kokoro_lang_is_german("de"));
    assert!(crispasr::kokoro_lang_is_german("de-DE"));
    assert!(crispasr::kokoro_lang_is_german("de_AT"));
    // NOT "deu": the C predicate is documented as "de" followed by '\0', '-'
    // or '_' (src/kokoro.h), CrispASR's language surface is ISO 639-1, and
    // nothing in the tree maps 639-3 down to it. This case asserted "deu" for
    // as long as the suite could not run — see the rpath fix in
    // crispasr/build.rs — so it never failed out loud.
    assert!(!crispasr::kokoro_lang_is_german("deu"));
    assert!(!crispasr::kokoro_lang_is_german("en"));
    // "en" always has a native Kokoro voice
    assert!(crispasr::kokoro_lang_has_native_voice("en"));
}

#[test]
fn speaker_db_missing_dir() {
    // Loading from a non-existent directory should return an error
    let result = crispasr::SpeakerDB::load("/nonexistent/speaker_db_dir_12345");
    assert!(result.is_err());
}

#[test]
fn vad_segments_null_model() {
    // Passing a nonsense model path should return an error
    let pcm = vec![0.0f32; 16000];
    let result = crispasr::vad_segments(
        "/nonexistent/vad.gguf",
        &pcm,
        16000,
        0.5,
        250,
        100,
        1,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn vad_slices_null_model() {
    let pcm = vec![0.0f32; 16000];
    let result = crispasr::vad_slices(
        "/nonexistent/vad.gguf",
        &pcm,
        16000,
        0.5,
        250,
        100,
        30,
        30.0,
        1,
    );
    // -3, not 0. A model that cannot be loaded must not look like "this audio
    // contains no speech" — crispasr_compute_vad_slices has carried an
    // out_load_failed flag for exactly this reason and the C ABI wrapper was
    // discarding it, so every binding read a broken install as silence.
    let msg = result.unwrap_err();
    assert!(
        msg.contains("could not be loaded"),
        "a missing VAD model must be reported as a load failure, got: {msg}"
    );
}

#[test]
fn vad_slices_real_model_is_not_a_load_failure() {
    // The positive control for the case above: with a model that DOES load, the
    // same call must succeed. Without this the assertion above passes for a
    // wrapper that simply errors on everything.
    let Some(model) = std::env::var("CRISPASR_VAD_MODEL")
        .ok()
        .filter(|p| !p.is_empty() && Path::new(p).exists())
    else {
        eprintln!("SKIP: set CRISPASR_VAD_MODEL to a VAD model");
        return;
    };
    let pcm = vec![0.0f32; 16000];
    let result = crispasr::vad_slices(&model, &pcm, 16000, 0.5, 250, 100, 30, 30.0, 1);
    assert!(result.is_ok(), "a loadable VAD model must not error: {result:?}");
}

// -------------------------------------------------------------------------
// Diarization (#332) — model-free, so it always runs.
//
// Regression: CrispasrDiarizeOptsAbi was 24 bytes short of the C layout
// after #324 appended the FoxNose fields, so every call made the C side
// read past the Rust allocation. These calls crossing the ABI at the full
// 48-byte layout (plus the layout test in crispasr-sys) pin the fix.
// -------------------------------------------------------------------------

#[test]
fn diarize_vad_turns_model_free() {
    let pcm = vec![0.01f32; 16000 * 4]; // 4 s of quiet mono PCM
    let mut segs = vec![
        crispasr::DiarizeSegment::new(0.0, 1.0),
        crispasr::DiarizeSegment::new(2.0, 3.0), // 1 s gap > 600 ms turn gap
    ];
    let opts = crispasr::DiarizeOptions::default();
    crispasr::diarize_segments(&mut segs, &pcm, None, false, &opts)
        .expect("vad_turns diarize failed");
    assert_ne!(
        segs[0].speaker, segs[1].speaker,
        "VadTurns must alternate speakers across a >600 ms gap"
    );
}

#[test]
fn diarize_foxnose_missing_model_errors() {
    let pcm = vec![0.01f32; 16000];
    let mut segs = vec![crispasr::DiarizeSegment::new(0.0, 1.0)];
    let mut opts = crispasr::DiarizeOptions::default();
    opts.method = crispasr::DiarizeMethod::FoxNose;
    opts.foxnose_embedder_path = Some("/nonexistent/wespeaker.gguf".to_string());
    let err = crispasr::diarize_segments(&mut segs, &pcm, None, false, &opts)
        .expect_err("foxnose with a missing embedder must fail, not crash");
    assert!(err.contains("load failed"), "unexpected error: {err}");
}

/// #395: the turn-forwarding wrapper labels exactly like the plain one, and a
/// method that derives no turns says so with an empty Vec rather than an error.
#[test]
fn diarize_with_turns_matches_plain_and_yields_no_turns_for_vad() {
    let pcm = vec![0.01f32; 16000 * 4];
    let mk = || {
        vec![
            crispasr::DiarizeSegment::new(0.0, 1.0),
            crispasr::DiarizeSegment::new(2.0, 3.0),
        ]
    };
    let opts = crispasr::DiarizeOptions::default(); // VadTurns

    let mut plain = mk();
    crispasr::diarize_segments(&mut plain, &pcm, None, false, &opts)
        .expect("vad_turns diarize failed");

    let mut with_turns = mk();
    let turns = crispasr::diarize_segments_with_turns(&mut with_turns, &pcm, None, false, &opts)
        .expect("vad_turns diarize_with_turns failed");

    assert!(
        turns.is_empty(),
        "only FoxNose derives turns; VadTurns must report none"
    );
    for (a, b) in plain.iter().zip(with_turns.iter()) {
        assert_eq!(a.speaker, b.speaker, "asking for turns changed the labels");
    }
}

/// The error path must stay an error on the turns entry point too — a model
/// load failure is not "zero turns".
#[test]
fn diarize_with_turns_foxnose_missing_model_errors() {
    let pcm = vec![0.01f32; 16000];
    let mut segs = vec![crispasr::DiarizeSegment::new(0.0, 1.0)];
    let mut opts = crispasr::DiarizeOptions::default();
    opts.method = crispasr::DiarizeMethod::FoxNose;
    opts.foxnose_embedder_path = Some("/nonexistent/wespeaker.gguf".to_string());
    let err = crispasr::diarize_segments_with_turns(&mut segs, &pcm, None, false, &opts)
        .expect_err("foxnose with a missing embedder must fail, not crash");
    assert!(err.contains("load failed"), "unexpected error: {err}");
}

/// Live: the case the issue is about — real turns, on the caller's own
/// absolute timeline, finer than the caller's segment grid. Opt-in via
/// CRISPASR_TEST_FOXNOSE_WAV + CRISPASR_TEST_FOXNOSE_EMBEDDER (16 kHz 16-bit
/// mono WAV); skipped when either is unset.
#[test]
fn diarize_with_turns_foxnose_live() {
    let (Ok(wav), Ok(embedder)) = (
        std::env::var("CRISPASR_TEST_FOXNOSE_WAV"),
        std::env::var("CRISPASR_TEST_FOXNOSE_EMBEDDER"),
    ) else {
        eprintln!("skipping: set CRISPASR_TEST_FOXNOSE_WAV + CRISPASR_TEST_FOXNOSE_EMBEDDER");
        return;
    };
    let Some(pcm) = read_wav_mono_16k(&wav) else {
        panic!("could not read {wav} as 16 kHz 16-bit PCM WAV");
    };

    // A deliberately coarse 3 s grid, offset 10 s into a longer recording —
    // both halves of what the ABI has to get right.
    const SLICE_T0: f64 = 10.0;
    let total = pcm.len() as f64 / 16_000.0;
    let mut segs: Vec<crispasr::DiarizeSegment> = (0..)
        .map(|i| i as f64 * 3.0)
        .take_while(|t| t + 3.0 <= total)
        .map(|t| crispasr::DiarizeSegment::new(SLICE_T0 + t, SLICE_T0 + t + 3.0))
        .collect();
    assert!(segs.len() >= 2, "fixture too short for a 3 s grid");

    let mut opts = crispasr::DiarizeOptions::default();
    opts.method = crispasr::DiarizeMethod::FoxNose;
    opts.foxnose_embedder_path = Some(embedder);
    opts.slice_t0 = SLICE_T0;
    opts.num_speakers = 2;

    let turns = crispasr::diarize_segments_with_turns(&mut segs, &pcm, None, false, &opts)
        .expect("foxnose diarize_with_turns failed");

    assert!(!turns.is_empty(), "FoxNose must derive turns");
    let mut prev_end = f64::NEG_INFINITY;
    for t in &turns {
        assert!(t.t1 > t.t0, "empty turn {t:?}");
        assert!(t.speaker >= 0, "a turn is always labelled: {t:?}");
        assert!(t.t0 >= prev_end - 1e-9, "turns out of order at {t:?}");
        // The wrapper must hand back the CALLER's timeline, not the buffer's.
        assert!(
            t.t0 >= SLICE_T0 - 1e-9 && t.t1 <= SLICE_T0 + total + 0.02,
            "turn {t:?} outside [{SLICE_T0}, {}]",
            SLICE_T0 + total
        );
        prev_end = t.t1;
    }

    // At least one caller segment covers two speakers — the whole reason the
    // turns had to be forwarded in the first place.
    let straddling = segs
        .iter()
        .filter(|g| {
            let mut spk: Vec<i32> = turns
                .iter()
                .filter(|t| t.t1.min(g.t1) - t.t0.max(g.t0) > 0.2)
                .map(|t| t.speaker)
                .collect();
            spk.sort_unstable();
            spk.dedup();
            spk.len() > 1
        })
        .count();
    assert!(
        straddling > 0,
        "fixture no longer exercises a segment spanning two speakers"
    );
}

/// Minimal 16-bit PCM WAV reader for the live diarize test. Returns None
/// unless the file is 16 kHz 16-bit (mono or stereo, downmixed).
fn read_wav_mono_16k(path: &str) -> Option<Vec<f32>> {
    let raw = std::fs::read(path).ok()?;
    if raw.len() < 44 || &raw[0..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
        return None;
    }
    let u16at = |o: usize| u16::from_le_bytes([raw[o], raw[o + 1]]);
    let u32at = |o: usize| u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);

    let (mut channels, mut bits, mut rate) = (0usize, 0usize, 0u32);
    let mut pos = 12usize;
    while pos + 8 <= raw.len() {
        let id = &raw[pos..pos + 4];
        let sz = u32at(pos + 4) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= raw.len() {
            channels = u16at(body + 2) as usize;
            rate = u32at(body + 4);
            bits = u16at(body + 14) as usize;
        } else if id == b"data" {
            if bits != 16 || channels == 0 || rate != 16_000 {
                return None;
            }
            let end = (body + sz).min(raw.len());
            let frames = (end - body) / 2 / channels;
            let mut out = Vec::with_capacity(frames);
            for f in 0..frames {
                let mut acc = 0i32;
                for c in 0..channels {
                    let o = body + (f * channels + c) * 2;
                    acc += i16::from_le_bytes([raw[o], raw[o + 1]]) as i32;
                }
                out.push(acc as f32 / channels as f32 / 32768.0);
            }
            return Some(out);
        }
        pos = body + sz + (sz & 1);
    }
    None
}

// =========================================================================
// Chat / LLM (crispasr_chat.h)
// =========================================================================

use std::cell::Cell;
use std::panic::AssertUnwindSafe;

use crispasr::{ChatGenerateOptions, ChatMessage, ChatOptions, ChatSession};

/// Context window the chat tests open with. Well above any prompt here and
/// far below a modern model's train context, whose KV cache is big enough
/// to matter when several of these tests run at once.
const CHAT_N_CTX: i32 = 2048;

fn chat_model() -> Option<String> {
    let p = std::env::var("CRISPASR_CHAT_MODEL")
        .or_else(|_| std::env::var("CRISPASR_CHAT_TEST_MODEL"))
        .unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../models/chat.gguf").to_string()
        });
    if Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

fn open_chat(model_path: &str) -> ChatSession {
    let opts = ChatOptions {
        n_ctx: Some(CHAT_N_CTX),
        n_batch: Some(256),
        n_ubatch: Some(256),
        ..Default::default()
    };
    ChatSession::open_with_options(model_path, &opts).expect("chat session open")
}

/// Greedy decoding, so every case below is reproducible run to run.
fn greedy(max_tokens: i32) -> ChatGenerateOptions {
    ChatGenerateOptions {
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..Default::default()
    }
}

fn one_turn(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage::user(text)]
}

/// A prompt whose reply is outside the tokeniser's character vocabulary, so
/// the model spells it with byte-fallback tokens: the C side then delivers
/// one chunk per BYTE of each character rather than one per character.
const MULTIBYTE_PROMPT: &str = "Reply with exactly this and nothing else: 🪿🫏🪼";

/// True when `s` holds a character the streamed path could split — a
/// multi-byte one that is not itself a replacement character.
fn has_splittable_char(s: &str) -> bool {
    s.chars().any(|c| c.len_utf8() > 1 && c != '\u{fffd}')
}

#[test]
fn chat_open_reports_context_and_template() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    assert!(chat.n_ctx() > 0, "context window must be positive");
    // The open params are plumbed through, not merely accepted.
    assert_eq!(chat.n_ctx(), CHAT_N_CTX);
    assert!(
        !chat.template_name().is_empty(),
        "session must resolve a chat template"
    );
    assert!(
        ChatSession::ai_disclosure_text().len() > 20,
        "AI disclosure wording must be present"
    );
}

#[test]
fn chat_generate_returns_text() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let out = chat
        .generate_with_options(
            &one_turn("Name one colour. Answer with one word."),
            &greedy(16),
        )
        .expect("generate");
    assert!(!out.trim().is_empty(), "generate returned no text");
}

#[test]
fn chat_stream_matches_one_shot() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn("List three fruits, separated by commas.");
    let params = greedy(32);

    let one_shot = chat
        .generate_with_options(&msgs, &params)
        .expect("one-shot");
    chat.reset().expect("reset");

    let mut streamed = String::new();
    let mut chunks = 0usize;
    chat.generate_stream_with_options(&msgs, &params, |chunk| {
        chunks += 1;
        streamed.push_str(chunk);
    })
    .expect("stream");

    assert!(chunks > 1, "expected several chunks, got {chunks}");
    assert_eq!(streamed, one_shot, "streamed chunks must rebuild the reply");
}

#[test]
fn chat_stream_rebuilds_characters_split_across_chunks() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn(MULTIBYTE_PROMPT);
    let params = greedy(32);

    let one_shot = chat
        .generate_with_options(&msgs, &params)
        .expect("one-shot");
    if !has_splittable_char(&one_shot) {
        eprintln!("SKIP: this model answered {one_shot:?}, with nothing to split");
        return;
    }
    chat.reset().expect("reset");

    let mut streamed = String::new();
    chat.generate_stream_with_options(&msgs, &params, |chunk| streamed.push_str(chunk))
        .expect("stream");

    assert_eq!(
        streamed.as_bytes(),
        one_shot.as_bytes(),
        "a character split across chunks must survive: streamed {streamed:?} vs one-shot {one_shot:?}"
    );
}

/// A token budget that runs out part-way through a character leaves bytes
/// that can never be completed. They must still reach the caller, the same
/// way the one-shot path renders them.
#[test]
fn chat_stream_delivers_a_character_the_token_budget_cut_in_half() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn(MULTIBYTE_PROMPT);
    // Two tokens is under one byte-fallback character's worth for a model
    // that has to spell the reply out byte by byte.
    let params = greedy(2);

    let one_shot = chat
        .generate_with_options(&msgs, &params)
        .expect("one-shot");
    if !one_shot.contains('\u{fffd}') {
        eprintln!("SKIP: this model stopped on a character boundary, at {one_shot:?}");
        return;
    }
    chat.reset().expect("reset");

    let mut streamed = String::new();
    chat.generate_stream_with_options(&msgs, &params, |chunk| streamed.push_str(chunk))
        .expect("stream");

    assert_eq!(
        streamed.as_bytes(),
        one_shot.as_bytes(),
        "the half character must not be dropped: streamed {streamed:?} vs one-shot {one_shot:?}"
    );
}

#[test]
fn chat_count_tokens_is_positive_and_monotone() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);

    let short = one_turn("Hello.");
    let long = vec![
        ChatMessage::user("Hello."),
        ChatMessage::assistant("Hi there, how can I help?"),
        ChatMessage::user("Tell me about the tides, at length and in detail."),
    ];

    let n_short = chat.count_tokens(&short).expect("count short");
    let n_long = chat.count_tokens(&long).expect("count long");
    assert!(n_short > 0, "a rendered prompt has tokens: {n_short}");
    assert!(
        n_long > n_short,
        "more conversation must cost more tokens: {n_long} vs {n_short}"
    );
    assert!(
        n_long < chat.n_ctx(),
        "test prompts must fit the context window"
    );

    // A pure query: counting neither prefills nor extends the history.
    assert_eq!(chat.count_tokens(&short).expect("count again"), n_short);
}

#[test]
fn chat_memory_estimate_covers_the_weights_and_scales_with_context() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let file_size = std::fs::metadata(&model_path)
        .expect("stat chat model")
        .len() as usize;
    assert!(file_size > 0);

    let at = |n_ctx: i32| {
        let opts = ChatOptions {
            n_ctx: Some(n_ctx),
            ..Default::default()
        };
        ChatSession::memory_estimate(&model_path, &opts).expect("memory estimate")
    };

    // Default options leave n_ctx unset, so the model's own trained context
    // sizes the KV term.
    let at_default = ChatSession::memory_estimate(&model_path, &ChatOptions::default())
        .expect("memory estimate at the model's own context");
    assert!(
        at_default > file_size,
        "the estimate must cover the weights on disk: {at_default} vs {file_size}"
    );

    let at_1k = at(1024);
    let at_2k = at(2048);
    let at_4k = at(4096);
    assert!(at_1k > file_size, "{at_1k} vs {file_size}");
    assert!(at_2k > at_1k, "{at_2k} vs {at_1k}");
    assert!(at_4k > at_2k, "{at_4k} vs {at_2k}");

    // The KV term is linear in n_ctx, so doubling the context doubles the
    // amount by which the estimate grows. A load path that returned before
    // reading the context / layer / embedding metadata would leave every
    // difference at zero and still report success.
    assert_eq!(
        at_4k - at_2k,
        2 * (at_2k - at_1k),
        "the KV term is not linear in n_ctx: {at_1k} / {at_2k} / {at_4k}"
    );

    // Everything outside the KV term is context-independent, so back it out
    // and the remainder still has to cover the weights on disk.
    let kv_per_1k = at_2k - at_1k;
    assert!(
        at_1k - kv_per_1k > file_size,
        "the context-independent part must cover the weights: {} vs {file_size}",
        at_1k - kv_per_1k
    );

    // A path that names no model is an error, not a zero estimate reported
    // as success.
    let missing = ChatSession::memory_estimate(
        "/nonexistent/crispasr-memory-estimate.gguf",
        &ChatOptions::default(),
    );
    assert!(missing.is_err(), "a missing model must not estimate");
}

#[test]
fn chat_abort_stops_stream_and_session_is_reusable() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn("Count from one to forty, one number per line.");
    let params = greedy(96);

    // Baseline: what an uninterrupted run of the same request delivers.
    let mut baseline = 0usize;
    chat.generate_stream_with_options(&msgs, &params, |_| baseline += 1)
        .expect("baseline stream");
    assert!(
        baseline > 10,
        "baseline should stream freely, got {baseline}"
    );
    chat.reset().expect("reset");

    let seen = Cell::new(0usize);
    let err = chat
        .with_abort_callback(
            || seen.get() < 3,
            |c| c.generate_stream_with_options(&msgs, &params, |_| seen.set(seen.get() + 1)),
        )
        .expect_err("an aborted generation must not report success");
    assert!(err.is_aborted(), "expected an abort, got: {err}");
    assert!(
        seen.get() >= 3 && seen.get() < baseline,
        "abort must stop the stream early: {} chunks vs {baseline}",
        seen.get()
    );

    // An abort flushes the session, so no reset is needed before reuse.
    let after = chat
        .generate_with_options(&one_turn("Say hello."), &greedy(16))
        .expect("session must be reusable after an abort");
    assert!(!after.trim().is_empty());
}

#[test]
fn chat_reopen_after_drop() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let first = open_chat(&model_path);
    let n_ctx = first.n_ctx();
    drop(first);

    let second = open_chat(&model_path);
    assert_eq!(second.n_ctx(), n_ctx);
    let out = second
        .generate_with_options(&one_turn("Say hello."), &greedy(16))
        .expect("generate after reopen");
    assert!(!out.trim().is_empty());
}

/// A panic in either callback is caught at the FFI boundary and resumed
/// after the native call returns — it never unwinds through C, which would
/// abort the process rather than fail this test.
#[test]
fn chat_callback_panic_does_not_unwind_through_c() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn("Say hello.");

    let token_panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        chat.generate_stream_with_options(&msgs, &greedy(16), |_| panic!("token callback exploded"))
    }))
    .expect_err("the token callback's panic must reach the caller");
    assert_eq!(
        token_panic.downcast_ref::<&str>().copied(),
        Some("token callback exploded")
    );

    let abort_panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        chat.with_abort_callback(
            || panic!("abort hook exploded"),
            |c| c.generate_with_options(&msgs, &greedy(16)),
        )
    }))
    .expect_err("the abort hook's panic must reach the caller");
    assert_eq!(
        abort_panic.downcast_ref::<&str>().copied(),
        Some("abort hook exploded")
    );

    // Both callbacks were deregistered on the way out, so the session works.
    let out = chat
        .generate_with_options(&msgs, &greedy(16))
        .expect("session usable after a callback panic");
    assert!(!out.trim().is_empty());
}

/// A panicking token callback cancels the generation through a registered
/// abort hook, and the caller's own predicate is not consulted again once it
/// has — the behaviour the Go and Python bindings also hold to.
#[test]
fn chat_token_panic_cancels_through_the_abort_hook() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn("Count from one to forty, one number per line.");

    let token_panicked = Cell::new(false);
    let calls = Cell::new(0usize);
    let calls_after_panic = Cell::new(0usize);

    let panic_val = std::panic::catch_unwind(AssertUnwindSafe(|| {
        chat.with_abort_callback(
            || {
                calls.set(calls.get() + 1);
                if token_panicked.get() {
                    calls_after_panic.set(calls_after_panic.get() + 1);
                }
                true
            },
            |c| {
                c.generate_stream_with_options(&msgs, &greedy(96), |_| {
                    token_panicked.set(true);
                    panic!("token callback exploded");
                })
            },
        )
    }))
    .expect_err("the token callback's panic must reach the caller");
    assert_eq!(
        panic_val.downcast_ref::<&str>().copied(),
        Some("token callback exploded")
    );

    // Positive control: the hook really was registered and consulted, so the
    // count below is zero for the right reason.
    assert!(calls.get() > 0, "the abort hook was never consulted");
    assert!(token_panicked.get(), "the token callback never ran");
    assert_eq!(
        calls_after_panic.get(),
        0,
        "the predicate must not be asked again once the token callback failed"
    );

    // The cancellation flushed the session, as any other abort does.
    let out = chat
        .generate_with_options(&one_turn("Say hello."), &greedy(16))
        .expect("session usable after a cancelled generation");
    assert!(!out.trim().is_empty());
}

/// A nested `with_abort_callback` scope puts the enclosing hook back when it
/// ends. Leaving the session with no hook instead would silently disarm the
/// outer cancellation for the rest of the outer body.
#[test]
fn chat_nested_abort_scope_restores_the_outer_hook() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn("Count from one to forty, one number per line.");

    let outer_calls = Cell::new(0usize);
    let inner_calls = Cell::new(0usize);

    let outcome = chat.with_abort_callback(
        || {
            outer_calls.set(outer_calls.get() + 1);
            outer_calls.get() < 3
        },
        |c| {
            // An inner scope with a hook of its own, which runs a complete
            // generation and then ends.
            c.with_abort_callback(
                || {
                    inner_calls.set(inner_calls.get() + 1);
                    true
                },
                |inner| inner.generate_with_options(&one_turn("Say hello."), &greedy(8)),
            )
            .expect("the inner scope's own generation must complete");
            // The outer hook has to be live again here.
            c.generate_with_options(&msgs, &greedy(64))
        },
    );

    assert!(inner_calls.get() > 0, "the inner hook was never consulted");
    assert!(
        outer_calls.get() >= 3,
        "the outer hook was not restored: consulted {} times after the inner scope ended",
        outer_calls.get()
    );
    let err = outcome.expect_err("the restored outer hook must abort the second generation");
    assert!(err.is_aborted(), "expected an abort, got: {err}");
}

/// Pins which way round the abort predicate reads: `true` lets the
/// generation run, `false` aborts it — the polarity of the C callback it
/// binds. A silently inverted predicate would turn every cancel into a hang,
/// and both halves of this case would still "pass" one at a time.
#[test]
fn chat_abort_polarity_true_continues() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn("Name one colour. Answer with one word.");

    // true = keep going: the generation completes normally.
    let kept = chat
        .with_abort_callback(|| true, |c| c.generate_with_options(&msgs, &greedy(16)))
        .expect("a callback that always says continue must let the run finish");
    assert!(!kept.trim().is_empty());

    // false = abort: stopped before it produces anything.
    let mut chunks = 0usize;
    let err = chat
        .with_abort_callback(
            || false,
            |c| c.generate_stream_with_options(&msgs, &greedy(16), |_| chunks += 1),
        )
        .expect_err("a callback that always says stop must abort the run");
    assert!(err.is_aborted(), "expected an abort, got: {err}");
    assert_eq!(chunks, 0, "an immediate abort emits no tokens");
}

/// A prompt whose greedy reply is fixed and made of short, distinct pieces, so
/// a stop substring can be placed inside it and the truncated text pinned
/// exactly. The reply this model gives is "1\n2\n3\n4\n5\n6\n7\n8\n".
const COUNTING_PROMPT: &str =
    "Count from 1 to 8. Write only the numbers, one per line, nothing else.";

/// What `COUNTING_PROMPT` yields once generation stops on "4" — the text the
/// caller receives, with the match itself cut off. The Python and Go chat
/// suites assert this same string for the same prompt, stop list and sampler
/// settings: three separate marshallings of one C feature, agreeing byte for
/// byte.
const COUNTING_STOPPED_AT_FOUR: &str = "1\n2\n3\n";

/// The full greedy reply the literals above describe.
///
/// Those literals are one MODEL's output, not a property of the stop feature,
/// while the gate accepts any small chat GGUF. On smollm2-360m-instruct the same
/// prompt answers "1 2 3 4 " with SPACES, so the stop cases failed on the
/// separator while every behavioural assertion beside them passed — a red for a
/// reason unrelated to the code under test, which is worse than no test because
/// it teaches you to ignore the suite.
///
/// The cross-binding oracle is worth keeping (Python, Go, Java and Dart pin the
/// same strings, so four marshallings of one C feature are held byte-identical),
/// so rather than weaken the assertions, check the precondition they encode:
/// confirm this IS the pinned model, and skip with a reason if not.
const COUNTING_BASELINE_REPLY: &str = "1\n2\n3\n4\n5\n6\n7\n8\n";

/// True when `chat` is the model the pinned literals describe. Called only
/// by the cases that assert an exact string, so the model-independent ones
/// (empty stop list, prefill-only) still run on any gate model. Prints why when
/// it is not, so a skipped case reads as "different model" rather than silence.
fn is_pinned_stop_baseline(chat: &ChatSession) -> bool {
    let msgs = one_turn(COUNTING_PROMPT);
    let baseline = match chat.generate_with_options(&msgs, &greedy(64)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIP: baseline generate failed: {e}");
            return false;
        }
    };
    let _ = chat.reset();
    if baseline != COUNTING_BASELINE_REPLY {
        eprintln!(
            "SKIP: stop-sequence literals are pinned to the model whose greedy reply is {COUNTING_BASELINE_REPLY:?} \
             (e.g. gemma-3-1b-it-Q4_K_M); this model replies {baseline:?}. Behaviour is covered \
             model-independently by tests/test-chat-ggml.cpp."
        );
        return false;
    }
    true
}

/// `greedy`, plus stop sequences.
fn greedy_with_stop(max_tokens: i32, stop: &[&str]) -> ChatGenerateOptions {
    ChatGenerateOptions {
        stop: stop.iter().map(|s| (*s).to_string()).collect(),
        ..greedy(max_tokens)
    }
}

#[test]
fn chat_stop_sequence_truncates_before_the_match() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    if !is_pinned_stop_baseline(&chat) {
        return;
    }
    let msgs = one_turn(COUNTING_PROMPT);

    let full = chat
        .generate_with_options(&msgs, &greedy(64))
        .expect("unstopped generate");
    // Without this the case is vacuous: a reply that never reaches the stop
    // string would pass whether or not stop sequences work at all.
    assert!(
        full.contains('5'),
        "the unstopped reply must contain the stop string, got {full:?}"
    );
    chat.reset().expect("reset");

    let stopped = chat
        .generate_with_options(&msgs, &greedy_with_stop(64, &["5"]))
        .expect("stopped generate");
    assert!(
        !stopped.contains('5'),
        "the matched text must not reach the caller, got {stopped:?}"
    );
    assert!(
        full.starts_with(&stopped),
        "the stopped reply must be a prefix of the unstopped one: {stopped:?} vs {full:?}"
    );
    assert_eq!(stopped, "1\n2\n3\n4\n");
}

#[test]
fn chat_stop_sequences_stop_at_the_earliest_match() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    if !is_pinned_stop_baseline(&chat) {
        return;
    }
    let msgs = one_turn(COUNTING_PROMPT);

    let full = chat
        .generate_with_options(&msgs, &greedy(64))
        .expect("unstopped generate");
    assert!(
        full.contains('4') && full.contains('7'),
        "the unstopped reply must contain both stop strings, got {full:?}"
    );

    // Two sequences, in both orders. "4" is generated before "7", so "4" wins
    // either way: the earliest match in the output decides, not the position
    // in the array. Order-independence is also what says the whole array was
    // marshalled, rather than only its first element.
    for stop in [["7", "4"], ["4", "7"]] {
        chat.reset().expect("reset");
        let stopped = chat
            .generate_with_options(&msgs, &greedy_with_stop(64, &stop))
            .expect("stopped generate");
        assert_eq!(stopped, COUNTING_STOPPED_AT_FOUR, "stop list {stop:?}");
        assert!(
            !stopped.contains('4') && !stopped.contains('7'),
            "neither match may reach the caller, got {stopped:?}"
        );
    }
}

#[test]
fn chat_empty_stop_list_is_the_same_as_no_stop_list() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn(COUNTING_PROMPT);

    let none = chat
        .generate_with_options(&msgs, &greedy(64))
        .expect("generate with no stop list");
    chat.reset().expect("reset");
    let empty = chat
        .generate_with_options(&msgs, &greedy_with_stop(64, &[]))
        .expect("generate with an empty stop list");

    assert_eq!(empty, none, "an empty stop list must not truncate");
    // The reply is long enough that a stop list WOULD have truncated it, so
    // the equality above is not two empty strings agreeing.
    assert!(
        none.contains('4'),
        "the reply must be long enough for a stop list to bite, got {none:?}"
    );
}

#[test]
fn chat_prefill_only_suppresses_generation() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    let msgs = one_turn(COUNTING_PROMPT);
    let prefill = ChatGenerateOptions {
        prefill_only: true,
        ..greedy(64)
    };

    let out = chat
        .generate_with_options(&msgs, &prefill)
        .expect("prefill-only generate must succeed, not fail");
    assert_eq!(out, "", "prefill_only must produce no assistant text");

    chat.reset().expect("reset");
    let mut chunks = 0usize;
    chat.generate_stream_with_options(&msgs, &prefill, |_| chunks += 1)
        .expect("prefill-only stream must succeed");
    assert_eq!(chunks, 0, "prefill_only must emit no token chunks");

    // Positive control: the same messages and sampler settings, minus the
    // flag, do produce text — so the two emptinesses above are the flag's
    // doing and not a prompt that generates nothing.
    chat.reset().expect("reset");
    let generated = chat
        .generate_with_options(&msgs, &greedy(64))
        .expect("control generate");
    assert!(!generated.is_empty(), "control produced nothing");
}

#[test]
fn chat_stream_delivers_the_chunk_the_one_shot_path_truncates() {
    let model_path = match chat_model() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: chat model not found (set CRISPASR_CHAT_MODEL)");
            return;
        }
    };
    let chat = open_chat(&model_path);
    if !is_pinned_stop_baseline(&chat) {
        return;
    }
    let msgs = one_turn(COUNTING_PROMPT);
    let params = greedy_with_stop(64, &["7", "4"]);

    let one_shot = chat
        .generate_with_options(&msgs, &params)
        .expect("one-shot");
    chat.reset().expect("reset");

    let mut streamed = String::new();
    chat.generate_stream_with_options(&msgs, &params, |chunk| streamed.push_str(chunk))
        .expect("stream");

    // The C side hands each piece to the callback before it scans for a stop
    // match, so the streamed text carries the matched piece that the one-shot
    // return value has cut off. The two paths are equal without a stop list
    // and differ by exactly that piece with one.
    assert_eq!(one_shot, COUNTING_STOPPED_AT_FOUR);
    assert_eq!(streamed, "1\n2\n3\n4");
    assert!(
        streamed.starts_with(&one_shot),
        "the streamed text must extend the one-shot text, got {streamed:?}"
    );
}

// ---- Source separation (#359) ----
//
// Reported as "there is no clear way to utilize models like htdemucs from
// libraries". The C ABI has always had a five-function separation surface and
// Python and Dart bound it; Rust and Go did not, so the only separation-shaped
// verb a Rust caller could find was speech_to_speech — which is not what these
// models do, and which answers with "returned no audio ... (S2S may be
// unsupported)" and a 0 sample rate. Exactly the report.
//
// Gate: SEPARATE_MODEL=/path/to/mel-band-roformer-or-htdemucs.gguf

fn separate_model() -> Option<String> {
    let p = std::env::var("SEPARATE_MODEL").unwrap_or_default();
    if !p.is_empty() && Path::new(&p).exists() {
        Some(p)
    } else {
        None
    }
}

#[test]
fn separate_returns_named_stems_at_the_models_own_rate() {
    let Some(model) = separate_model() else {
        eprintln!("SKIP: set SEPARATE_MODEL to a separation GGUF");
        return;
    };
    let s = crispasr::Session::open(&model).expect("open separation model");

    // The rate accessor is the thing a caller needs BEFORE they can feed it
    // anything, and its absence is why the reporter saw 0.
    let sr = s.separate_sample_rate();
    assert!(sr > 0, "separate_sample_rate must be known once loaded, got {sr}");

    // 2 s of interleaved stereo at the model's own rate. Not silence: a quiet
    // tone, so a backend that returns its input unchanged is still exercised.
    let n_frames = (sr as usize) * 2;
    let mut pcm = Vec::with_capacity(n_frames * 2);
    for i in 0..n_frames {
        let t = i as f32 / sr as f32;
        let v = (t * 220.0 * std::f32::consts::TAU).sin() * 0.2;
        pcm.push(v);
        pcm.push(v);
    }

    let stems = s.separate(&pcm).expect("separate should produce stems");
    assert!(!stems.is_empty(), "expected at least one stem");
    for st in &stems {
        assert!(!st.name.is_empty(), "every stem is named");
        assert_eq!(
            st.pcm.len() % 2,
            0,
            "stem {} must be interleaved stereo",
            st.name
        );
        assert!(!st.pcm.is_empty(), "stem {} came back empty", st.name);
        assert!(
            st.pcm.iter().all(|v| v.is_finite()),
            "stem {} has non-finite samples",
            st.name
        );
    }
    eprintln!(
        "separate: {} stems at {} Hz: {:?}",
        stems.len(),
        sr,
        stems.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn separate_rejects_input_that_is_not_stereo() {
    let Some(model) = separate_model() else {
        eprintln!("SKIP: set SEPARATE_MODEL to a separation GGUF");
        return;
    };
    let s = crispasr::Session::open(&model).expect("open separation model");
    // The C API counts PER-CHANNEL frames; handing it fewer than one frame is
    // a caller error, not something to pass to the model.
    assert!(s.separate(&[]).is_err());
}
