// crispasr_output.h — output formatting shared across non-whisper backends.
//
// These writers consume std::vector<crispasr_segment> (the common result
// type) rather than whisper_context, so any backend can drive them.
//
// The whisper code path in cli.cpp continues to use its own writers
// (output_txt, output_srt, etc. defined there) because they have features
// like token-level WTS karaoke output and JSON metadata that are
// whisper-specific for now. A later refactor will unify the two.

#pragma once

#include "crispasr_backend.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

// Convert a centisecond timestamp to "HH:MM:SS.mmm" (VTT) or
// "HH:MM:SS,mmm" (SRT, when comma=true). Mirrors common-whisper's
// to_timestamp() but avoids a dependency on that library here.
std::string crispasr_to_timestamp(int64_t cs, bool comma = false);

// Derive an output path from an audio input path by stripping a known
// audio extension and appending the given extension (including the dot).
// "audio.wav" + ".srt" -> "audio.srt".
std::string crispasr_make_out_path(const std::string& audio, const std::string& ext);

// ---------------------------------------------------------------------------
// Display segments: what actually gets written to stdout and output files.
// Built from the crispasr_segment vector by splitting long segments on word
// boundaries when max_len > 0, or emitting one segment per word when
// max_len == 1.
// ---------------------------------------------------------------------------

struct crispasr_disp_segment {
    int64_t t0, t1; // centiseconds, absolute
    std::string text;
    std::string speaker; // empty if none
};

// Build display segments from backend segments according to max_len.
//   max_len = 0 -> one display segment per input segment (no splitting)
//   max_len = 1 -> one display segment per word (requires words populated)
//   max_len > 1 -> split at word boundaries when accumulated text would
//                  exceed max_len characters
// split_on_punct: additionally split at sentence-ending punctuation (. ! ?)
//   This creates natural subtitle lines even when segments are long.
//   Works with and without word-level timestamps.
std::vector<crispasr_disp_segment> crispasr_make_disp_segments(const std::vector<crispasr_segment>& segments,
                                                               int max_len, bool split_on_punct = false);

// Issue #356: the cue stream the writers emit must be non-decreasing in start
// time. Note WHICH start time — crispasr_make_disp_segments reads the WORD
// timestamps of any segment that carries words and only falls back to
// seg.t0/seg.t1 for a text-only one, so a list whose segment spans look
// monotone can still write backward-jumping cues. This walks the same values
// the writers will and returns the index of the first segment that starts
// before the previous emitted position, or -1 when the list is in order.
// `prev_cs` / `cur_cs` (optional) receive the two offending timestamps.
//
// Diagnostic only: it reports, it never reorders.
int crispasr_first_backward_segment(const std::vector<crispasr_segment>& segments, int64_t* prev_cs = nullptr,
                                    int64_t* cur_cs = nullptr);

// Print one warning per process if `segments` is out of order. #356 reached a
// user as a subtitle-rendering complaint because nothing between the producer
// (the #89 gap-fill second pass) and the writers ever checked this — every
// stage that could have was either opt-in or documented as "never reorders".
// `where` names the producing stage. CRISPASR_ORDER_WARN=0 silences it.
void crispasr_warn_if_segments_backward(const std::vector<crispasr_segment>& segments, const char* where);

// ---------------------------------------------------------------------------
// Writers. All take a full file path; callers are expected to choose the
// path via crispasr_make_out_path().
// ---------------------------------------------------------------------------

bool crispasr_write_txt(const std::string& path, const std::vector<crispasr_disp_segment>& segs);

bool crispasr_write_srt(const std::string& path, const std::vector<crispasr_disp_segment>& segs);

bool crispasr_write_vtt(const std::string& path, const std::vector<crispasr_disp_segment>& segs);

bool crispasr_write_csv(const std::string& path, const std::vector<crispasr_disp_segment>& segs);

// Optional LID (language identification) result for JSON output
struct crispasr_lid_info {
    std::string lang_code;    // detected language (e.g. "en")
    float confidence = -1.0f; // [0,1] or -1 if not available
    std::string source;       // "whisper", "ecapa", "silero", etc.
};

bool crispasr_write_json(const std::string& path, const std::vector<crispasr_segment>& segs,
                         const std::string& backend_name, const std::string& model_path, const std::string& language,
                         bool full, const crispasr_lid_info* lid = nullptr);

bool crispasr_write_ctc_logits_json(const std::string& path, const crispasr_ctc_logits& logits,
                                    const std::string& backend_name);

bool crispasr_write_lrc(const std::string& path, const std::vector<crispasr_disp_segment>& segs);

// Print segments to stdout. If show_timestamps is true, each line is
// "[t0 --> t1] text"; otherwise the transcript is printed as one blob per
// segment separated by spaces.
void crispasr_print_stdout(const std::vector<crispasr_disp_segment>& segs, bool show_timestamps);

// Print per-token alternatives (--alt mode). Shows each token with its
// confidence and top-N alternative candidates, inspired by antirez/voxtral.c.
void crispasr_print_alternatives(const std::vector<crispasr_segment>& segs, int n_alt);

// Print each segment's transcript with an inline per-token confidence
// annotation (`--print-confidence`). One line per segment: `word[95%]
// word[88%] ...`. Segments with no token-level info print their plain text.
void crispasr_print_confidence(const std::vector<crispasr_segment>& segs);

// ---------------------------------------------------------------------------
// String-based formatters (for HTTP server responses, in-memory use).
// These mirror the file-based writers above but return std::string.
// ---------------------------------------------------------------------------

// Concatenate all segment texts into a single string separated by spaces.
std::string crispasr_segments_to_text(const std::vector<crispasr_segment>& segs);

// Format segments as SRT subtitle string.
std::string crispasr_segments_to_srt(const std::vector<crispasr_segment>& segs, int max_len = 0);

// Format segments as WebVTT subtitle string.
std::string crispasr_segments_to_vtt(const std::vector<crispasr_segment>& segs, int max_len = 0);

// Minimal JSON escape (RFC 8259). Shared so the server doesn't duplicate it.
std::string crispasr_json_escape(const std::string& s);

// OpenAI-compatible JSON: {"text": "..."}
std::string crispasr_segments_to_openai_json(const std::vector<crispasr_segment>& segs);

// OpenAI-compatible verbose JSON with segments, word timestamps, duration,
// language, task. Matches the OpenAI /v1/audio/transcriptions verbose_json
// response format.
std::string crispasr_segments_to_openai_verbose_json(const std::vector<crispasr_segment>& segs, double duration_s,
                                                     const std::string& language, const std::string& task,
                                                     float temperature);

// Diarized JSON: OpenAI-compatible verbose JSON extended with speaker labels.
// Speaker strings like "(speaker 0) " are normalised to single letters "A", "B", …
// Matches the diarized_json schema requested in issue #206.
std::string crispasr_segments_to_diarized_json(const std::vector<crispasr_segment>& segs, double duration_s,
                                               const std::string& language, const std::string& task, float temperature);

// CrispASR native JSON (the format returned by /inference).
std::string crispasr_segments_to_native_json(const std::vector<crispasr_segment>& segs, const std::string& backend_name,
                                             double duration_s);

std::string crispasr_ctc_logits_to_json(const crispasr_ctc_logits& logits);

// Remove punctuation from a segment in-place: from seg.text, each
// seg.words[i].text, and each seg.tokens[i].text. Called by the
// dispatch layer when --no-punctuation is set and the backend didn't
// strip punctuation natively. Targets ASCII punctuation plus a small
// set of common Unicode marks the LLM backends emit (smart quotes, em
// dash, ellipsis). Idempotent — running it twice is a no-op.
void crispasr_strip_punctuation(crispasr_segment& seg);
