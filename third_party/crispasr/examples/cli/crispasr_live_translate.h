// crispasr_live_translate.h — sentence-incremental commit policy for the
// live transcribe + translate mode (`--live-translate`, or `--translate-model`
// on a streaming run).
//
// The streaming loop in crispasr_run.cpp produces a growing PARTIAL hypothesis
// for the open utterance every `--stream-step`, and a FINAL when trailing
// silence closes it. Translating only finals makes the translation wait for a
// pause — tens of seconds behind on a lecture. Translating every partial
// re-translates text that is still changing and the output flickers.
//
// The policy here sits between the two: a sentence is COMMITTED as soon as
// the recogniser has moved past it and two consecutive partials agree on it
// (LocalAgreement-2). A committed sentence is immutable — it is translated
// exactly once and never revised, not even by the final. The still-open tail
// after the last committed sentence is the only part that may be re-translated
// as a draft.
//
// Everything in this header is pure (no I/O, no model, no clock) so the policy
// is unit-testable; crispasr_live_translate_sink.h owns the translator thread
// and the terminal.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace crispasr {

struct lt_word {
    std::string raw;   // token as the recogniser wrote it, punctuation included
    std::string norm;  // case-folded, edge punctuation stripped — the alignment key
    bool glue = false; // true: joins to the previous token without a space (CJK)
    // Stream time (s) by which this word had NOT yet been heard: the audio
    // position of the last partial that did not contain it. Its audio ends
    // after this. Only maintained for open words; see lt_committer::decode_from.
    double not_before = 0.0;
    // Recogniser word timing (stream seconds), when it supplied any: -1 = none.
    double t_mid = -1.0;
    double t_end = -1.0;
};

// A word with its recogniser timestamps, in stream seconds.
struct lt_timed_word {
    std::string text;
    double t0 = 0.0;
    double t1 = 0.0;
};

namespace lt_detail {

// Byte length of the UTF-8 sequence starting at `c` (1 for stray bytes).
inline size_t u8_len(unsigned char c) {
    if (c < 0x80)
        return 1;
    if ((c >> 5) == 0x6)
        return 2;
    if ((c >> 4) == 0xE)
        return 3;
    if ((c >> 3) == 0x1E)
        return 4;
    return 1;
}

// CJK / fullwidth sentence terminators: 。 ！ ？
inline bool is_cjk_terminator(const std::string& s, size_t i) {
    if (i + 2 >= s.size())
        return false;
    const unsigned char a = s[i], b = s[i + 1], c = s[i + 2];
    return (a == 0xE3 && b == 0x80 && c == 0x82) || (a == 0xEF && b == 0xBC && (c == 0x81 || c == 0x9F));
}

// Edge punctuation the normaliser strips. Covers ASCII plus the typographic
// quotes/dashes/ellipsis German and English recognisers emit.
inline size_t edge_punct_len(const std::string& s, size_t i) {
    const unsigned char c = s[i];
    if (c < 0x80)
        return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) ||
                       (c >= 0x7B && c <= 0x7E)
                   ? 1
                   : 0;
    if (is_cjk_terminator(s, i))
        return 3;
    if (c == 0xE2 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x80) {
        const unsigned char d = s[i + 2];
        // ‐‑‒–— ‘’‚‛ “”„‟ … (U+2010..U+2015, U+2018..U+201F, U+2026)
        if ((d >= 0x90 && d <= 0x95) || (d >= 0x98 && d <= 0x9F) || d == 0xA6)
            return 3;
    }
    if (c == 0xC2 && i + 1 < s.size() && ((unsigned char)s[i + 1] == 0xAB || (unsigned char)s[i + 1] == 0xBB))
        return 2; // « »
    return 0;
}

inline std::string normalise(const std::string& raw) {
    size_t b = 0, e = raw.size();
    for (;;) {
        const size_t n = b < e ? edge_punct_len(raw, b) : 0;
        if (!n)
            break;
        b += n;
    }
    // Trailing punctuation: walk back over whole UTF-8 sequences.
    while (e > b) {
        size_t s = e - 1;
        while (s > b && ((unsigned char)raw[s] & 0xC0) == 0x80)
            --s;
        if (edge_punct_len(raw, s) != e - s || e - s == 0)
            break;
        e = s;
    }
    std::string out;
    out.reserve(e - b);
    for (size_t i = b; i < e; ++i) {
        const unsigned char c = raw[i];
        if (c >= 'A' && c <= 'Z') {
            out += (char)(c + 32);
        } else if (c == 0xC3 && i + 1 < e) {
            // Latin-1 supplement: À..Þ (C3 80..9E, minus × = 97) fold to à..þ.
            unsigned char d = raw[i + 1];
            if (d >= 0x80 && d <= 0x9E && d != 0x97)
                d += 0x20;
            out += (char)c;
            out += (char)d;
            ++i;
        } else {
            out += (char)c;
        }
    }
    return out;
}

inline bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::char_traits<char>::length(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

} // namespace lt_detail

// Split recogniser text into tokens. Whitespace separates tokens; a CJK
// sentence terminator additionally closes one, so scripts written without
// spaces still yield sentence-sized units instead of one giant "word".
inline std::vector<lt_word> lt_split_words(const std::string& text) {
    std::vector<lt_word> out;
    std::string cur;
    bool glue_next = false;
    auto flush = [&](bool closed_by_cjk) {
        if (cur.empty())
            return;
        lt_word w;
        w.raw = cur;
        w.norm = lt_detail::normalise(cur);
        w.glue = glue_next;
        out.push_back(std::move(w));
        cur.clear();
        glue_next = closed_by_cjk;
    };
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            flush(false);
            glue_next = false;
            ++i;
            continue;
        }
        if (lt_detail::is_cjk_terminator(text, i)) {
            cur.append(text, i, 3);
            i += 3;
            flush(true);
            continue;
        }
        const size_t n = lt_detail::u8_len(c);
        cur.append(text, i, n);
        i += n;
    }
    flush(false);
    return out;
}

inline std::string lt_join(const std::vector<lt_word>& w, size_t b, size_t e) {
    std::string out;
    for (size_t i = b; i < e && i < w.size(); ++i) {
        if (!out.empty() && !w[i].glue)
            out += ' ';
        out += w[i].raw;
    }
    return out;
}

// Does token `i` close a sentence? Deliberately conservative about '.': a
// false split costs translation quality (the translator sees half a
// sentence), a missed one only costs a little latency.
inline bool lt_ends_sentence(const std::vector<lt_word>& w, size_t i) {
    if (i >= w.size())
        return false;
    std::string t = w[i].raw;
    // Closing quotes / brackets after the terminator: `sagte er."`
    for (;;) {
        if (t.empty())
            return false;
        const char c = t.back();
        if (c == '"' || c == '\'' || c == ')' || c == ']') {
            t.pop_back();
            continue;
        }
        if (lt_detail::ends_with(t, "\xE2\x80\x9C") || lt_detail::ends_with(t, "\xE2\x80\x9D") ||
            lt_detail::ends_with(t, "\xC2\xBB") || lt_detail::ends_with(t, "\xC2\xAB")) {
            t.resize(t.size() - (((unsigned char)t[t.size() - 2] == 0xC2) ? 2 : 3));
            continue;
        }
        break;
    }
    if (lt_detail::ends_with(t, "\xE3\x80\x82") || lt_detail::ends_with(t, "\xEF\xBC\x81") ||
        lt_detail::ends_with(t, "\xEF\xBC\x9F"))
        return true;
    const char last = t.back();
    if (last == '!' || last == '?')
        return true;
    if (last != '.')
        return false;
    // "..." / "…" is a hesitation, not an end.
    if (lt_detail::ends_with(t, ".."))
        return false;
    const std::string& n = w[i].norm;
    if (n.empty())
        return false;
    // "3." / "21." is a German ordinal or an enumeration ("am 3. Oktober"),
    // not an end. A longer number is a year or a quantity and can close a
    // sentence like any other word ("… im Jahr 1990.").
    bool all_digits = true;
    for (unsigned char c : n)
        if (!(c >= '0' && c <= '9'))
            all_digits = false;
    if (all_digits && n.size() <= 2)
        return false;
    // Single letter: initials, "z. B.", "u. a.", "d. h.", "e. g.".
    if (n.size() == 1 || (n.size() == 2 && ((unsigned char)n[0] & 0xE0) == 0xC0))
        return false;
    // Inner dot: "z.B.", "e.g.", "U.S.".
    if (n.find('.') != std::string::npos)
        return false;
    static const char* const kAbbrev[] = {
        "dr",  "prof", "nr",  "bzw", "ca",  "usw", "evtl", "ggf",    "ggfs", "inkl", "vgl",  "sog", "st",  "mr",
        "mrs", "ms",   "vs",  "jr",  "sr",  "inc", "ltd",  "mio",    "mrd",  "tel",  "str",  "abs", "bsp", "zzgl",
        "fig", "gen",  "gov", "rev", "hon", "mt",  "etc",  "approx", "no",   "vol",  "dipl", "ing", "hr",  "fr",
    };
    for (const char* a : kAbbrev)
        if (n == a)
            return false;
    return true;
}

struct lt_sentence {
    int id = 0;
    std::string text;
    // False for a piece that was committed without reaching a sentence end:
    // a forced commit of an over-long run, or the unterminated remainder of
    // an utterance. Such a piece is settled TEXT, but not a translation unit.
    bool complete = true;
};

struct lt_update {
    std::vector<lt_sentence> committed; // newly committed, in order
    std::string tail;                   // text still open after the last commit
    bool tail_changed = false;
};

struct lt_commit_options {
    // A tail this long with no sentence end is force-committed at a clause
    // boundary, so a recogniser that emits no punctuation (or a speaker who
    // never finishes a sentence) cannot hold the translation back forever.
    int force_commit_words = 30;
    // Words kept back from a forced commit: the newest words of a hypothesis
    // are the ones the next partial is most likely to revise.
    int force_keep_back = 4;
    // The same, while the caller reports pressure (set_pressure): the open
    // region has grown past what it can afford to re-decode every step.
    int pressure_commit_words = 10;
    // Partials in a row that may fail to align before we resync on the best
    // available guess rather than stalling.
    int max_align_misses = 3;
};

// Tracks one stream. Feed every partial and every final; get back the
// sentences that became committed and the current open tail.
class lt_committer {
public:
    explicit lt_committer(const lt_commit_options& o = lt_commit_options()) : opt_(o) {}

    // `t_audio` is the stream time (seconds of audio received) this partial
    // was decoded at. It only feeds decode_from(); callers that do not use
    // that may leave it at 0.
    //
    // `timed`, when the recogniser has word timestamps, lists the words of
    // this hypothesis with their times (it may cover only the END of `text`).
    // With it the committed boundary is a point in time instead of a guess:
    // "everything before 40.52 s is committed" needs no committed words in
    // view to line up against, so the caller can start decoding right at the
    // boundary rather than a second and a half before it.
    lt_update on_partial(int64_t utterance_id, const std::string& text, double t_audio = 0.0,
                         const std::vector<lt_timed_word>* timed = nullptr) {
        sync_utterance(utterance_id);
        lt_update up;
        std::vector<lt_word> words = lt_split_words(text);
        size_t first_timed = words.size();
        if (timed) {
            // Walk both lists from the end for as long as they agree.
            size_t i = words.size(), j = timed->size();
            while (i > 0 && j > 0 && words[i - 1].norm == lt_detail::normalise((*timed)[j - 1].text)) {
                words[i - 1].t_mid = 0.5 * ((*timed)[j - 1].t0 + (*timed)[j - 1].t1);
                words[i - 1].t_end = (*timed)[j - 1].t1;
                --i;
                --j;
            }
            first_timed = i;
        }
        size_t p = 0;
        bool resumed = false;
        // The boundary by the clock. A word belongs to the open text when its
        // MIDDLE lies after the boundary: the committed sentence's last word
        // ends at the boundary, so its middle is before it, and the next
        // word's is after — with half a word of room for timestamp jitter
        // either way. Usable when the timed words reach back to the boundary.
        // Either the first timed word is still committed text, or every word
        // is timed (then a hypothesis that starts after the boundary — the
        // speaker paused there — is simply all open).
        if (committed_until_ >= 0.0 && first_timed < words.size() &&
            (first_timed == 0 || words[first_timed].t_mid <= committed_until_)) {
            p = first_timed;
            while (p < words.size() && words[p].t_mid <= committed_until_)
                ++p;
            // A forced commit can end mid-phrase, with no pause after its
            // last word. Decoding from just before the boundary then hears
            // the tail of that word again and may time it a little late
            // ("…auf den" | "Den starken Export"). Same word, right at the
            // boundary: it is the committed one.
            if (p < words.size() && !committed_.empty() && words[p].norm == committed_.back() &&
                words[p].t_mid - committed_until_ < 0.35)
                ++p;
            resumed = true;
            align_misses_ = 0;
        }
        if (!resumed && !resume_point(words, /*is_final=*/false, p)) {
            // The partial does not line up with what we already committed:
            // the recogniser rewrote the boundary, or the caller decoded from
            // decode_from() and that started too late to show it. Committed
            // text is immutable, so wait for a partial that does line up —
            // and withdraw decode_from() so the next one is decoded from the
            // start of the speech region, where the boundary certainly is.
            ++align_misses_total_;
            last_text_.clear();
            decode_from_ = -1.0;
            decode_from_exact_ = false;
            committed_until_ = -1.0;
            up.tail = last_tail_;
            return up;
        }
        std::vector<lt_word> open(words.begin() + (std::ptrdiff_t)p, words.end());

        // LocalAgreement-2: how far does this partial agree with the last one?
        size_t agree = 0;
        while (agree < open.size() && agree < prev_open_.size() && open[agree].norm == prev_open_[agree].norm)
            ++agree;
        // A word keeps the bound of the word that stood at its position
        // before (a hypothesis grows at its end, so position ≈ time even when
        // the word itself was revised); a word at a new position cannot have
        // been in the audio the previous partial covered.
        for (size_t i = 0; i < open.size(); ++i)
            open[i].not_before = i < prev_open_.size() ? prev_open_[i].not_before : last_partial_t_;
        last_partial_t_ = t_audio;

        // Commit through the last sentence end that (a) the recogniser has
        // moved past, and (b) the previous partial also wrote, terminator
        // included. (b) is what rejects the period a recogniser puts at the
        // cut end of every partial: "Ich gehe." → "Ich gehe nach Hause."
        size_t cut = 0; // open[0..cut) gets committed
        // The terminator has to have been written before, by the previous
        // partial or the one before it. A recogniser that cannot decide
        // between "vorstellen. Er kommt" and "vorstellen, er kommt" flips
        // between them on alternate partials; asking for two in a row then
        // never commits (seen: a sentence held back 5 s until the NEXT one
        // ended). Two of the last three is the same evidence without the
        // deadlock, and either reading is a fine place to cut.
        for (size_t i = 0; i + 1 < open.size() && i < agree; ++i) {
            if (!lt_ends_sentence(open, i))
                continue;
            const bool prev1 = open[i].raw == prev_open_[i].raw;
            const bool prev2 =
                i < prev_open2_.size() && open[i].raw == prev_open2_[i].raw && open[i].norm == prev_open2_[i].norm;
            if (prev1 || prev2)
                cut = i + 1;
        }
        size_t done = 0;
        for (size_t i = 0; i < cut; ++i) {
            if (lt_ends_sentence(open, i)) {
                commit(open, done, i + 1, up);
                done = i + 1;
            }
        }
        // Forced commit of an over-long unterminated run.
        const int force_words = pressure_ ? opt_.pressure_commit_words : opt_.force_commit_words;
        if ((int)(open.size() - done) > force_words) {
            const size_t stable_end = agree > (size_t)opt_.force_keep_back ? agree - (size_t)opt_.force_keep_back : 0;
            if (stable_end > done + (size_t)force_words / 2) {
                size_t fc = stable_end;
                for (size_t i = stable_end; i > done + (size_t)force_words / 2; --i) {
                    const std::string& r = open[i - 1].raw;
                    if (!r.empty() && (r.back() == ',' || r.back() == ';' || r.back() == ':')) {
                        fc = i;
                        break;
                    }
                }
                commit(open, done, fc, up);
                done = fc;
            }
        }
        // The partial before this one, re-based onto the same starting point.
        if (prev_open_.size() >= done)
            prev_open2_.assign(prev_open_.begin() + (std::ptrdiff_t)done, prev_open_.end());
        else
            prev_open2_.clear();
        prev_open_.assign(open.begin() + (std::ptrdiff_t)done, open.end());
        last_text_ = text;
        up.tail = lt_join(prev_open_, 0, prev_open_.size());
        up.tail_changed = up.tail != last_tail_;
        last_tail_ = up.tail;
        return up;
    }

    // The speaker paused (the caller's VAD saw the speech end a moment ago)
    // and the last partial already covered all of it. If the open text ends
    // on a sentence terminator, commit it now: waiting for the next word
    // would only measure the length of the pause, and waiting for the final
    // costs the whole --stream-final-on-silence-ms. Text that does not end a
    // sentence stays open — a pause mid-sentence is just a breath.
    lt_update on_pause(int64_t utterance_id) {
        lt_update up;
        if (utterance_id != utterance_id_ || prev_open_.empty() || !lt_ends_sentence(prev_open_, prev_open_.size() - 1))
            return up;
        const std::vector<lt_word> open = prev_open_;
        size_t done = 0;
        for (size_t i = 0; i < open.size(); ++i) {
            if (lt_ends_sentence(open, i)) {
                commit(open, done, i + 1, up);
                done = i + 1;
            }
        }
        prev_open_.clear();
        prev_open2_.clear();
        up.tail_changed = !last_tail_.empty();
        last_tail_.clear();
        return up;
    }

    // The utterance closed: commit everything that is still open. Sentences
    // committed from partials stay as they are — the final only supplies the
    // remainder.
    lt_update on_final(int64_t utterance_id, const std::string& text) {
        sync_utterance(utterance_id);
        lt_update up;
        // The final is the very hypothesis we resolved last (the live loop
        // closes an utterance on its last partial): what is open is already
        // known. Re-deriving it from the text would need the committed words
        // in view, and a caller decoding from decode_from() no longer shows
        // them — the remainder was silently dropped that way.
        if (!last_text_.empty() && text == last_text_) {
            size_t done = 0;
            for (size_t i = 0; i < prev_open_.size(); ++i) {
                if (lt_ends_sentence(prev_open_, i)) {
                    commit(prev_open_, done, i + 1, up);
                    done = i + 1;
                }
            }
            if (done < prev_open_.size())
                commit(prev_open_, done, prev_open_.size(), up);
            up.tail_changed = !last_tail_.empty();
            reset_utterance();
            return up;
        }
        const std::vector<lt_word> words = lt_split_words(text);
        size_t p = 0;
        if (!resume_point(words, /*is_final=*/true, p)) {
            // The final does not line up with the committed text and there
            // is no later hypothesis to wait for. What the last hypothesis
            // that DID line up left open is real, unshown text: commit that,
            // rather than guess at this one and either repeat committed
            // sentences or drop the remainder (which is what happened).
            const std::vector<lt_word> open = prev_open_;
            size_t done = 0;
            for (size_t i = 0; i < open.size(); ++i) {
                if (lt_ends_sentence(open, i)) {
                    commit(open, done, i + 1, up);
                    done = i + 1;
                }
            }
            if (done < open.size())
                commit(open, done, open.size(), up);
            up.tail_changed = !last_tail_.empty();
            reset_utterance();
            return up;
        }
        size_t done = p;
        for (size_t i = p; i < words.size(); ++i) {
            if (lt_ends_sentence(words, i)) {
                commit(words, done, i + 1, up);
                done = i + 1;
            }
        }
        if (done < words.size())
            commit(words, done, words.size(), up);
        up.tail_changed = !last_tail_.empty();
        reset_utterance();
        return up;
    }

    int sentences_committed() const { return next_id_; }

    // Stream time (s) from which the open utterance still needs decoding, or
    // < 0 for "from its start". Audio before this belongs to committed
    // sentences, so a caller may stop re-decoding it on every step — the
    // dominant cost of a long utterance. This is an ESTIMATE from when each
    // word first showed up (no word timestamps needed, so it works for every
    // recogniser), deliberately early. Decode from a couple of seconds before
    // it, so the end of the committed text is in view for resume_point() to
    // find; if it is not, on_partial() withdraws the estimate (see there).
    double decode_from() const { return decode_from_; }

    // True when decode_from() is a recogniser word timestamp rather than the
    // estimate. The caller then needs no committed words in view and can
    // start a fraction of a second before it.
    bool decode_from_exact() const { return decode_from_ >= 0.0 && decode_from_exact_; }

    // The caller is re-decoding more open audio per step than it can afford
    // (or than the recogniser handles well — small models drop the end of a
    // long clip). While set, an unterminated run is force-committed much
    // earlier, which moves decode_from() forward. Clause-sized pieces translate
    // worse than whole sentences; that is the price of staying live.
    void set_pressure(bool on) { pressure_ = on; }

    // Partials that did not line up with the committed text. Each one is a
    // step whose hypothesis was thrown away.
    int align_misses() const { return align_misses_total_; }

private:
    void sync_utterance(int64_t id) {
        if (id != utterance_id_) {
            reset_utterance();
            utterance_id_ = id;
        }
    }

    void reset_utterance() {
        committed_.clear();
        prev_open_.clear();
        prev_open2_.clear();
        last_tail_.clear();
        last_text_.clear();
        n_committed_ = 0;
        align_misses_ = 0;
        decode_from_ = -1.0;
        decode_from_exact_ = false;
        committed_until_ = -1.0;
        last_partial_t_ = 0.0;
    }

    void commit(const std::vector<lt_word>& w, size_t b, size_t e, lt_update& up) {
        if (e <= b)
            return;
        lt_sentence s;
        s.id = next_id_++;
        s.text = lt_join(w, b, e);
        s.complete = lt_ends_sentence(w, e - 1);
        up.committed.push_back(std::move(s));
        for (size_t i = b; i < e; ++i)
            committed_.push_back(w[i].norm);
        n_committed_ += e - b;
        // The sentence's last word was not in the audio up to not_before, so
        // the sentence ends after it. Half a second of slack: a recogniser
        // can hold a word back until it has heard a little of what follows.
        decode_from_ = w[e - 1].not_before - 0.5;
        // With a real timestamp the boundary is exact.
        committed_until_ = w[e - 1].t_end;
        decode_from_exact_ = committed_until_ >= 0.0;
        if (decode_from_exact_)
            decode_from_ = committed_until_;
        if (committed_.size() > kKeep)
            committed_.erase(committed_.begin(), committed_.end() - (std::ptrdiff_t)kKeep);
    }

    // Index in `words` where not-yet-committed text begins.
    //
    // Aligns the last few committed words (the key) against the hypothesis
    // and returns the position right after where the key ends. It has to be a
    // real alignment, not a position-by-position compare: a recogniser
    // re-renders the same audio differently from one partial to the next
    // ("12%" → "zwölf Prozent" — one token becomes two), and a compare that
    // cannot absorb the inserted token lands one word early, which re-commits
    // the committed sentence's last word as a "sentence" of its own.
    //
    // Overlap alignment: the key may start before the hypothesis does (the
    // rolling window evicted its beginning — free), must end inside it, and
    // may differ from it by substitutions and gaps (each -1; a match is +2).
    bool resume_point(const std::vector<lt_word>& words, bool is_final, size_t& p_out) {
        if (committed_.empty()) {
            p_out = 0;
            return true;
        }
        const size_t m = committed_.size() < kKey ? committed_.size() : kKey;
        const size_t n = words.size();
        const std::string* key = committed_.data() + (committed_.size() - m);
        // d[i][j]: best score aligning key[0..i) so that it ends at words[j).
        std::vector<int> prev(n + 1, 0), cur(n + 1, 0);
        for (size_t i = 1; i <= m; ++i) {
            cur[0] = 0; // key prefix cut off by the window start
            for (size_t j = 1; j <= n; ++j) {
                const int diag = prev[j - 1] + (key[i - 1] == words[j - 1].norm ? 2 : -1);
                const int up = prev[j] - 1;      // key word missing from the hypothesis
                const int left = cur[j - 1] - 1; // extra word in the hypothesis
                cur[j] = diag > up ? (diag > left ? diag : left) : (up > left ? up : left);
            }
            prev.swap(cur);
        }
        size_t best_p = 0;
        int best_score = 0;
        size_t best_dist = (size_t)-1;
        for (size_t p = 1; p <= n; ++p) {
            const size_t dist = p > n_committed_ ? p - n_committed_ : n_committed_ - p;
            if (prev[p] > best_score || (prev[p] == best_score && best_score > 0 && dist < best_dist)) {
                best_score = prev[p];
                best_p = p;
                best_dist = dist;
            }
        }
        // Three net matching words when the key is long enough to ask for
        // that; every word when it is not.
        const int need = m >= 4 ? 5 : 2 * (int)m;
        if (best_score >= need) {
            align_misses_ = 0;
            p_out = best_p;
            return true;
        }
        if (is_final)
            return false;
        if (++align_misses_ < opt_.max_align_misses)
            return false;
        // Resync: take the best guess. No overlap at all means the committed
        // text has left the recogniser's window, so everything here is new.
        align_misses_ = 0;
        p_out = best_score > 0 ? best_p : 0;
        return true;
    }

    static constexpr size_t kKeep = 96; // committed words remembered for alignment
    static constexpr size_t kKey = 8;   // of which this many form the match key

    lt_commit_options opt_;
    int64_t utterance_id_ = -1;
    std::vector<std::string> committed_; // norms, current utterance, last kKeep
    size_t n_committed_ = 0;             // words committed in this utterance
    std::vector<lt_word> prev_open_;     // previous partial's open words
    std::vector<lt_word> prev_open2_;    // the one before that, on the same base
    std::string last_tail_;
    std::string last_text_; // the last partial that resolved (lined up)
    int align_misses_ = 0;
    int align_misses_total_ = 0;
    bool pressure_ = false;
    double decode_from_ = -1.0;
    bool decode_from_exact_ = false;
    double committed_until_ = -1.0; // end time of the last committed word, if timed
    double last_partial_t_ = 0.0;   // stream time of the previous partial
    int next_id_ = 0;
};

} // namespace crispasr
