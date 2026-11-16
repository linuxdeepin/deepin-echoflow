package whisper

// In-package because its subject, chatErrorFrom, is unexported and — by design
// — unreachable through the public API: every C path fills the error struct
// today, so nothing a caller can do produces the fallback case below. No cgo
// here: a test file may not import "C", which is why chatError keeps its
// classification in this cgo-free half.

import (
	"errors"
	"testing"
	"unicode/utf8"
)

// Test_chatErrorFrom_FallsBackToTheReturnedCode pins the defence in
// chatErrorFrom: a C path that returned CRISPASR_CHAT_ERR_ABORTED without
// filling err must still be reported as a cancel, the way the Rust and Python
// bindings report it.
func Test_chatErrorFrom_FallsBackToTheReturnedCode(t *testing.T) {
	err := chatErrorFrom("", 0, "fallback text", chatErrAborted)
	if !errors.Is(err, ErrChatAborted) {
		t.Fatalf("a zero err.code must fall back to the returned code, got %v", err)
	}
	if want := "fallback text: " + ErrChatAborted.Error(); err.Error() != want {
		t.Fatalf("message %q, want %q", err.Error(), want)
	}

	// The struct wins when it is filled in: the returned code is a fallback,
	// not an override.
	if err := chatErrorFrom("from C", chatErrAborted, "fallback text", 0); !errors.Is(err, ErrChatAborted) {
		t.Fatalf("a filled err.code must still be honoured, got %v", err)
	}

	// Any other code is a fault, not a cancel — from either source.
	if err := chatErrorFrom("from C", 0, "fallback text", 7); errors.Is(err, ErrChatAborted) {
		t.Fatal("a returned code that is not the abort code must not read as a cancel")
	}
	if err := chatErrorFrom("from C", 7, "fallback text", chatErrAborted); errors.Is(err, ErrChatAborted) {
		t.Fatal("a filled non-abort code must not be overridden by the returned code")
	}
}

// feed drives chatCall's assembler with the given chunks and returns what the
// caller's callback saw, one entry per call, plus whatever flush produced.
func feed(chunks ...string) []string {
	var got []string
	call := &chatCall{onToken: func(s string) { got = append(got, s) }}
	for _, c := range chunks {
		call.deliverBytes([]byte(c))
	}
	call.flush()
	return got
}

func joined(got []string) string {
	out := ""
	for _, s := range got {
		out += s
	}
	return out
}

// Test_chatCall_ChunksAreWholeCharacters is the regression guard on the defect
// the byte-fallback path had: the C side hands over the raw bytes of one
// detokenised piece, so a character the tokeniser does not hold arrives one
// BYTE per call. Handing each of those to the caller gives it a string that is
// not valid UTF-8, which a JSON or SSE consumer replaces or rejects — so the
// unfinished tail is held back until the bytes that finish it arrive. This is
// the same contract the Rust, Python and Java bindings state.
func Test_chatCall_ChunksAreWholeCharacters(t *testing.T) {
	// One byte at a time is what a byte-fallback token stream looks like.
	got := feed("\xf0", "\x9f\xaa", "\xbf")
	if len(got) != 1 || got[0] != "\U0001fabf" {
		t.Fatalf("a split character must be delivered once and whole, got %q", got)
	}

	// A complete chunk passes straight through, and the ordinary path is not
	// buffered a chunk behind.
	if got := feed("hello"); len(got) != 1 || got[0] != "hello" {
		t.Fatalf("a whole chunk must pass straight through, got %q", got)
	}

	// Nothing before the unfinished tail is held back with it.
	got = feed("ab\xe2\x82", "\xacc")
	if len(got) != 2 || got[0] != "ab" || got[1] != "€c" {
		t.Fatalf("only the unfinished tail may be held back, got %q", got)
	}

	// Every chunk a caller sees is valid on its own.
	for _, s := range feed("\xf0", "\x9f", "\xaa", "\xbf", "!") {
		if !utf8.ValidString(s) {
			t.Fatalf("chunk % x is not valid UTF-8", []byte(s))
		}
	}
}

// Test_chatCall_MalformedBytesAreReplacedNotBuffered pins the other half of
// the rule: bytes no continuation could ever complete must not be held, or a
// stream that ends in one would stall until flush. The unit replaced is the
// Unicode "maximal subpart", which is what Rust's from_utf8, Python's
// incremental decoder and Java's CharsetDecoder all substitute — one
// replacement character for the whole truncated sequence, not one per byte.
func Test_chatCall_MalformedBytesAreReplacedNotBuffered(t *testing.T) {
	cases := []struct {
		name   string
		chunks []string
		want   string
	}{
		// A lone continuation byte, and a lead byte above the U+10FFFF ceiling.
		{"a stray byte", []string{"a\xffb"}, "a�b"},
		{"a continuation with nothing to continue", []string{"a\x80b"}, "a�b"},
		// Unfinished when it arrives, invalid once the next character does.
		{"a truncated sequence then a character", []string{"\xe2\x82", "x"}, "�x"},
		// Two bytes of one character, so ONE replacement, not two.
		{"maximal subpart", []string{"\xe2\x82x"}, "�x"},
		{"maximal subpart, four-byte lead", []string{"\xf0\x9f\xaax"}, "�x"},
		// The narrowed first-continuation ranges: an overlong three-byte form,
		// a surrogate, and a code point past U+10FFFF each stop at one byte.
		{"overlong", []string{"\xe0\x80\x80"}, "���"},
		{"surrogate", []string{"\xed\xa0\x80"}, "���"},
		{"past the ceiling", []string{"\xf4\x90\x80\x80"}, "����"},
		// A generation that simply stopped mid-character: the leftover is
		// handed over by flush rather than dropped.
		{"a tail at end of stream", []string{"ok\xf0\x9f"}, "ok�"},
	}
	for _, tc := range cases {
		got := feed(tc.chunks...)
		if joined(got) != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, joined(got), tc.want)
		}
		for _, s := range got {
			if s == "" {
				t.Errorf("%s: an empty chunk was delivered", tc.name)
			}
		}
	}
}

// Test_chatCall_EmptyChunkKeepsTheBuffer checks an empty callback neither
// delivers nor discards: C can hand over a zero-length piece, and the bytes
// waiting for their continuation must survive it.
func Test_chatCall_EmptyChunkKeepsTheBuffer(t *testing.T) {
	var got []string
	call := &chatCall{onToken: func(s string) { got = append(got, s) }}
	call.deliverBytes([]byte("\xf0\x9f"))
	call.deliverBytes(nil)
	if len(got) != 0 {
		t.Fatalf("nothing is complete yet, got %q", got)
	}
	call.deliverBytes([]byte("\xaa\xbf"))
	if len(got) != 1 || got[0] != "\U0001fabf" {
		t.Fatalf("the buffer must survive an empty chunk, got %q", got)
	}
	call.flush()
	if len(got) != 1 {
		t.Fatalf("flush must deliver nothing when nothing is pending, got %q", got)
	}
}
