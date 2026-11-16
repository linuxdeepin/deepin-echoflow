package whisper

// Chat / LLM surface for the Go binding: the crispasr_chat_* C ABI declared in
// include/crispasr_chat.h. Text in, text out — separate from the ASR session
// above, and usable on its own.
//
// EU AI Act Art. 50(1): a product that puts this in front of a natural person
// owes them a "you are talking to an AI" notice. ChatAIDisclosureText returns
// the canonical wording; see the header for the full note.

/*
// LDFLAGS for libcrispasr + all conditionally-built sub-libs are set in
// whisper.go (the canonical cgo block). Don't re-list here to avoid
// `ld: warning: ignoring duplicate libraries: '-lcrispasr'`.
#include <crispasr_chat.h>
#include <stdint.h>
#include <stdlib.h>

// The callbacks cross back into Go through these two exported functions. The
// `user` word is a runtime/cgo.Handle, never a Go pointer, so it is carried as
// uintptr_t and only cast to void* at the ABI edge.
// The chunk is const on the way in; the declaration drops the qualifier to
// match the signature cgo generates for the exported Go function.
extern void crispasrGoChatToken(char* utf8_chunk, uintptr_t user);
extern bool crispasrGoChatAbort(uintptr_t user);

static void crispasr_go_chat_token_cb(const char* utf8_chunk, void* user) {
    crispasrGoChatToken((char*)utf8_chunk, (uintptr_t)user);
}

static bool crispasr_go_chat_abort_cb(void* user) {
    return crispasrGoChatAbort((uintptr_t)user);
}

static int32_t crispasr_go_chat_generate_stream(crispasr_chat_session_t s,
                                                const crispasr_chat_message* messages, size_t n_messages,
                                                const crispasr_chat_generate_params* params, uintptr_t user,
                                                crispasr_chat_error* err) {
    return crispasr_chat_generate_stream(s, messages, n_messages, params,
                                         crispasr_go_chat_token_cb, (void*)user, err);
}

static void crispasr_go_chat_set_abort(crispasr_chat_session_t s, uintptr_t user) {
    crispasr_chat_set_abort_callback(s, crispasr_go_chat_abort_cb, (void*)user);
}

static void crispasr_go_chat_clear_abort(crispasr_chat_session_t s) {
    crispasr_chat_set_abort_callback(s, NULL, NULL);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime/cgo"
	"strings"
	"sync"
	"unicode/utf8"
	"unsafe"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

// ChatMessage is one turn of a conversation. Role is one of "system", "user",
// "assistant" or "tool", matching the OpenAI chat schema; the chat template
// translates those names into whatever the model expects.
type ChatMessage struct {
	Role    string
	Content string
}

// ChatOpenParams are the per-session, model-level options of
// crispasr_chat_open. It carries the C struct the ABI's own defaults function
// fills and is read and written through the accessors below — the same shape
// as Params on the ASR side. Setting one option therefore leaves every other
// one at the ABI default, and the zero value behaves as DefaultChatOpenParams:
// the first accessor call seeds it from the ABI.
type ChatOpenParams struct {
	c      C.crispasr_chat_open_params
	seeded bool

	// chatTemplate stays on the Go side. The C field is a borrowed pointer the
	// ABI copies out of, so the string is put into C memory for the length of
	// one call rather than being owned by this struct.
	chatTemplate string
}

// ChatGenerateParams are the per-call sampler options of the two generate
// entry points, with the same construction rule as ChatOpenParams: it carries
// the C struct the ABI's defaults function fills, is read and written through
// the accessors below, and its zero value behaves as
// DefaultChatGenerateParams.
type ChatGenerateParams struct {
	c      C.crispasr_chat_generate_params
	seeded bool

	// stop stays on the Go side for the same reason as ChatOpenParams's
	// template: the C fields are a borrowed array of borrowed pointers.
	stop []string
}

// ChatSession is an open chat model plus its KV cache. One call at a time: the
// C session serialises its own context, and this wrapper serialises the calls
// that register callbacks on it.
type ChatSession struct {
	handle C.crispasr_chat_session_t

	mu sync.Mutex // one native call at a time

	abortMu        sync.Mutex // guards shouldContinue, which a caller may set from any goroutine
	shouldContinue func() bool
}

///////////////////////////////////////////////////////////////////////////////
// ERRORS

// ErrChatAborted reports that a registered abort predicate stopped the
// generation rather than the model faulting. Test for it with errors.Is: it is
// the one error code the C header promises as stable, so a caller running its
// own cancellation can tell a cancel from a decode fault.
var ErrChatAborted = errors.New("crispasr_chat: generation aborted")

// chatErrAborted is the one error code crispasr_chat.h promises as a contract.
const chatErrAborted = int32(C.CRISPASR_CHAT_ERR_ABORTED)

// chatError turns a filled-in C error struct into a Go error.
//
// codeHint is the value the entry point itself returned, used when err.code is
// zero. The one-shot path signals failure by returning NULL, so there err is
// the only carrier and the hint is 0; the streaming and reset paths also
// return the code. Every C path fills err today, so the fallback is defence
// against a future one that does not rather than a live fix.
func chatError(fallback string, err *C.crispasr_chat_error, codeHint int32) error {
	return chatErrorFrom(C.GoString(&err.message[0]), int32(err.code), fallback, codeHint)
}

// chatErrorFrom is the cgo-free half of chatError: it picks the code to
// classify on — the struct's, falling back to the one the entry point returned
// — and wraps ErrChatAborted when that is the documented abort code.
func chatErrorFrom(msg string, code int32, fallback string, codeHint int32) error {
	if msg == "" {
		msg = fallback
	}
	if code == 0 {
		code = codeHint
	}
	if code == chatErrAborted {
		return fmt.Errorf("%s: %w", msg, ErrChatAborted)
	}
	return errors.New(msg)
}

// chatCString copies s into C memory, rejecting an interior NUL. C reads a NUL
// as the end of the string, so passing one through would silently drop the
// rest of a model path, message, stop sequence or chat template — for a path,
// that means opening a different file from the one named. Rust's CString::new
// rejects the same input.
func chatCString(s, field string) (*C.char, error) {
	if strings.IndexByte(s, 0) >= 0 {
		return nil, fmt.Errorf("crispasr_chat: %s contains an interior NUL byte, which C cannot carry", field)
	}
	return C.CString(s), nil
}

///////////////////////////////////////////////////////////////////////////////
// CALLBACK PLUMBING

// chatCall is the state of one native generate call. It reaches the callbacks
// through a runtime/cgo.Handle — the C `user` word is that handle's uintptr,
// never a Go pointer, which the cgo pointer-passing rules forbid and the
// garbage collector would be free to move.
// pending holds the tail of a character the C side delivered in pieces. A
// model that falls back to byte tokens emits one chunk per BYTE, so a chunk
// can end part-way through a character; handing each chunk to the caller on
// its own would give it a string that is not valid UTF-8.
type chatCall struct {
	onToken        func(string)
	shouldContinue func() bool

	pending  []byte
	panicked bool
	panicVal any
}

// capture records the first panic a callback raised. Later callbacks are then
// skipped and, if an abort predicate is registered, answer "do not continue".
func (c *chatCall) capture(r any) {
	if !c.panicked {
		c.panicked = true
		c.panicVal = r
	}
}

// deliver hands one chunk to the caller's token callback, containing any panic.
// Empty text is not delivered.
func (c *chatCall) deliver(chunk string) {
	if c.panicked || c.onToken == nil || chunk == "" {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			c.capture(r)
		}
	}()
	c.onToken(chunk)
}

// deliverBytes hands over everything in chunk that completes a character,
// keeping any trailing unfinished sequence for the chunk after it.
func (c *chatCall) deliverBytes(chunk []byte) {
	c.pending = append(c.pending, chunk...)
	var out strings.Builder
	i := 0
	for i < len(c.pending) {
		size, whole := utf8Unit(c.pending[i:])
		if size == 0 {
			break // unfinished: the bytes that finish it arrive next
		}
		if whole {
			out.Write(c.pending[i : i+size])
		} else {
			out.WriteRune(utf8.RuneError)
		}
		i += size
	}
	c.pending = append(c.pending[:0], c.pending[i:]...)
	c.deliver(out.String())
}

// flush hands over whatever is still buffered when the generation ends. Those
// bytes are a character the generation stopped in the middle of, so they are
// malformed on their own — hand them over with replacement rather than drop
// output silently.
func (c *chatCall) flush() {
	if len(c.pending) == 0 {
		return
	}
	tail := strings.ToValidUTF8(string(c.pending), string(utf8.RuneError))
	c.pending = c.pending[:0]
	c.deliver(tail)
}

// utf8Unit measures the one unit at the start of b, following the Unicode
// "maximal subpart" rule the Rust, Python and Java bindings' decoders use:
//
//	size > 0, whole      one whole character of size bytes
//	size > 0, not whole  size bytes no continuation could ever complete,
//	                     which become a single replacement character
//	size == 0            an unfinished prefix; wait for more bytes
func utf8Unit(b []byte) (size int, whole bool) {
	c := b[0]
	switch {
	case c < utf8.RuneSelf:
		return 1, true
	case c < 0xC2 || c > 0xF4:
		// A continuation byte with nothing to continue, an overlong two-byte
		// lead, or a lead above the U+10FFFF ceiling.
		return 1, false
	}
	// The first continuation byte's accepted range is narrowed for the leads
	// that would otherwise admit an overlong form, a surrogate, or a code
	// point past U+10FFFF; every later byte takes the full range.
	need, lo, hi := 4, byte(0x80), byte(0xBF)
	switch {
	case c < 0xE0:
		need = 2
	case c == 0xE0:
		need, lo = 3, 0xA0
	case c == 0xED:
		need, hi = 3, 0x9F
	case c < 0xF0:
		need = 3
	case c == 0xF0:
		lo = 0x90
	case c == 0xF4:
		hi = 0x8F
	}
	for i := 1; i < need; i++ {
		if i >= len(b) {
			return 0, false
		}
		if b[i] < lo || b[i] > hi {
			return i, false
		}
		lo, hi = 0x80, 0xBF
	}
	return need, true
}

// keepGoing answers the C callback's "may I continue?" question in the C
// callback's own polarity: true continues, false aborts. A panicking predicate
// aborts the generation, as does a panic already captured from the token
// callback.
func (c *chatCall) keepGoing() (keep bool) {
	if c.panicked {
		return false
	}
	if c.shouldContinue == nil {
		return true
	}
	defer func() {
		if r := recover(); r != nil {
			c.capture(r)
			keep = false
		}
	}()
	return c.shouldContinue()
}

//export crispasrGoChatToken
func crispasrGoChatToken(chunk *C.char, user C.uintptr_t) {
	cgo.Handle(user).Value().(*chatCall).deliverBytes([]byte(C.GoString(chunk)))
}

//export crispasrGoChatAbort
func crispasrGoChatAbort(user C.uintptr_t) C.bool {
	// No polarity change here: the C callback returns true to CONTINUE, and so
	// does the Go predicate.
	return C.bool(cgo.Handle(user).Value().(*chatCall).keepGoing())
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// DefaultChatOpenParams returns the open params the C ABI itself defaults to.
func DefaultChatOpenParams() ChatOpenParams {
	var p ChatOpenParams
	p.ensure()
	return p
}

// DefaultChatGenerateParams returns the generate params the C ABI itself
// defaults to.
func DefaultChatGenerateParams() ChatGenerateParams {
	var p ChatGenerateParams
	p.ensure()
	return p
}

// ensure seeds a value that has not been through DefaultChatOpenParams from
// the ABI defaults, so the zero value is never what reaches C.
func (p *ChatOpenParams) ensure() {
	if !p.seeded {
		C.crispasr_chat_open_params_default(&p.c)
		p.seeded = true
	}
}

// NThreads reports the generation thread count.
func (p *ChatOpenParams) NThreads() int { p.ensure(); return int(p.c.n_threads) }

// SetNThreads sets the generation thread count.
func (p *ChatOpenParams) SetNThreads(n int) { p.ensure(); p.c.n_threads = C.int32_t(n) }

// NThreadsBatch reports the batch / prefill thread count.
func (p *ChatOpenParams) NThreadsBatch() int { p.ensure(); return int(p.c.n_threads_batch) }

// SetNThreadsBatch sets the batch / prefill thread count.
func (p *ChatOpenParams) SetNThreadsBatch(n int) { p.ensure(); p.c.n_threads_batch = C.int32_t(n) }

// NCtx reports the requested context window in tokens; 0 = the model's own.
func (p *ChatOpenParams) NCtx() int { p.ensure(); return int(p.c.n_ctx) }

// SetNCtx sets the context window in tokens; 0 takes the model's own.
func (p *ChatOpenParams) SetNCtx(n int) { p.ensure(); p.c.n_ctx = C.int32_t(n) }

// NBatch reports the logical batch size.
func (p *ChatOpenParams) NBatch() int { p.ensure(); return int(p.c.n_batch) }

// SetNBatch sets the logical batch size.
func (p *ChatOpenParams) SetNBatch(n int) { p.ensure(); p.c.n_batch = C.int32_t(n) }

// NUBatch reports the physical micro-batch size.
func (p *ChatOpenParams) NUBatch() int { p.ensure(); return int(p.c.n_ubatch) }

// SetNUBatch sets the physical micro-batch size.
func (p *ChatOpenParams) SetNUBatch(n int) { p.ensure(); p.c.n_ubatch = C.int32_t(n) }

// NGPULayers reports how many layers are offloaded; -1 = all, 0 = CPU only.
func (p *ChatOpenParams) NGPULayers() int { p.ensure(); return int(p.c.n_gpu_layers) }

// SetNGPULayers sets how many layers to offload; -1 = all, 0 = CPU only.
func (p *ChatOpenParams) SetNGPULayers(n int) { p.ensure(); p.c.n_gpu_layers = C.int32_t(n) }

// UseMmap reports whether the weights are mapped rather than read.
func (p *ChatOpenParams) UseMmap() bool { p.ensure(); return bool(p.c.use_mmap) }

// SetUseMmap chooses whether to map the weights rather than read them.
func (p *ChatOpenParams) SetUseMmap(v bool) { p.ensure(); p.c.use_mmap = C.bool(v) }

// UseMlock reports whether the weights are locked into RAM.
func (p *ChatOpenParams) UseMlock() bool { p.ensure(); return bool(p.c.use_mlock) }

// SetUseMlock chooses whether to lock the weights into RAM.
func (p *ChatOpenParams) SetUseMlock(v bool) { p.ensure(); p.c.use_mlock = C.bool(v) }

// ChatTemplate reports the template override, empty for none.
func (p *ChatOpenParams) ChatTemplate() string { p.ensure(); return p.chatTemplate }

// SetChatTemplate overrides the template baked into the GGUF. Empty — the
// default — reads tokenizer.chat_template from the model and falls back to
// "chatml" if the model has none.
func (p *ChatOpenParams) SetChatTemplate(tmpl string) { p.ensure(); p.chatTemplate = tmpl }

// ensure seeds a value that has not been through DefaultChatGenerateParams
// from the ABI defaults, so the zero value is never what reaches C.
func (p *ChatGenerateParams) ensure() {
	if !p.seeded {
		C.crispasr_chat_generate_params_default(&p.c)
		p.seeded = true
	}
}

// MaxTokens reports the hard cap on generated tokens.
func (p *ChatGenerateParams) MaxTokens() int { p.ensure(); return int(p.c.max_tokens) }

// SetMaxTokens sets the hard cap on generated tokens. 0 does NOT mean
// "generate nothing": the ABI reads any non-positive value as unset and
// applies its own default of 256. Use SetPrefillOnly to suppress generation.
func (p *ChatGenerateParams) SetMaxTokens(n int) { p.ensure(); p.c.max_tokens = C.int32_t(n) }

// Temperature reports the sampling temperature; 0.0 = greedy.
func (p *ChatGenerateParams) Temperature() float32 { p.ensure(); return float32(p.c.temperature) }

// SetTemperature sets the sampling temperature; 0.0 = greedy.
func (p *ChatGenerateParams) SetTemperature(t float32) { p.ensure(); p.c.temperature = C.float(t) }

// TopK reports the top-k cutoff; 0 = disabled.
func (p *ChatGenerateParams) TopK() int { p.ensure(); return int(p.c.top_k) }

// SetTopK sets the top-k cutoff; 0 disables it.
func (p *ChatGenerateParams) SetTopK(k int) { p.ensure(); p.c.top_k = C.int32_t(k) }

// TopP reports the nucleus cutoff; 1.0 = disabled.
func (p *ChatGenerateParams) TopP() float32 { p.ensure(); return float32(p.c.top_p) }

// SetTopP sets the nucleus cutoff; 1.0 disables it.
func (p *ChatGenerateParams) SetTopP(v float32) { p.ensure(); p.c.top_p = C.float(v) }

// MinP reports the minimum-probability cutoff; 0.0 = disabled.
func (p *ChatGenerateParams) MinP() float32 { p.ensure(); return float32(p.c.min_p) }

// SetMinP sets the minimum-probability cutoff; 0.0 disables it.
func (p *ChatGenerateParams) SetMinP(v float32) { p.ensure(); p.c.min_p = C.float(v) }

// RepeatPenalty reports the repetition penalty; 1.0 = disabled.
func (p *ChatGenerateParams) RepeatPenalty() float32 {
	p.ensure()
	return float32(p.c.repeat_penalty)
}

// SetRepeatPenalty sets the repetition penalty; 1.0 disables it.
func (p *ChatGenerateParams) SetRepeatPenalty(v float32) {
	p.ensure()
	p.c.repeat_penalty = C.float(v)
}

// RepeatLastN reports the repetition window; -1 = ctx size, 0 = disabled.
func (p *ChatGenerateParams) RepeatLastN() int { p.ensure(); return int(p.c.repeat_last_n) }

// SetRepeatLastN sets the repetition window; -1 = ctx size, 0 disables it.
func (p *ChatGenerateParams) SetRepeatLastN(n int) { p.ensure(); p.c.repeat_last_n = C.int32_t(n) }

// Seed reports the sampler seed; 0xFFFFFFFF = random.
func (p *ChatGenerateParams) Seed() uint32 { p.ensure(); return uint32(p.c.seed) }

// SetSeed sets the sampler seed; 0xFFFFFFFF draws a random one.
func (p *ChatGenerateParams) SetSeed(s uint32) { p.ensure(); p.c.seed = C.uint32_t(s) }

// Stop reports a copy of the stop sequences.
func (p *ChatGenerateParams) Stop() []string {
	p.ensure()
	return append([]string(nil), p.stop...)
}

// SetStop sets the stop sequences, replacing any already set; calling it with
// no arguments clears them. Generation halts the first time any of these
// substrings appears in the accumulated output, which is truncated before the
// match.
func (p *ChatGenerateParams) SetStop(stop ...string) {
	p.ensure()
	p.stop = append([]string(nil), stop...)
}

// PrefillOnly reports whether assistant generation is suppressed.
func (p *ChatGenerateParams) PrefillOnly() bool { p.ensure(); return bool(p.c.prefill_only) }

// SetPrefillOnly asks for the prompt to be prefilled with assistant generation
// suppressed — useful for measuring prompt cost.
func (p *ChatGenerateParams) SetPrefillOnly(v bool) { p.ensure(); p.c.prefill_only = C.bool(v) }

// ChatAIDisclosureText returns the canonical "you are talking to an AI"
// wording (EU AI Act Art. 50(1)). Show it visibly at or before the first turn
// of any conversational product built on this binding.
func ChatAIDisclosureText() string {
	return C.GoString(C.crispasr_chat_ai_disclosure_text())
}

// ChatOpen opens a chat session over a GGUF chat model on disk. Pass nil for
// params to take the ABI defaults.
func ChatOpen(modelPath string, params *ChatOpenParams) (*ChatSession, error) {
	cpath, err := chatCString(modelPath, "modelPath")
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(cpath))

	cparams, freeParams, err := chatOpenParamsToC(params)
	defer freeParams()
	if err != nil {
		return nil, err
	}

	var cerr C.crispasr_chat_error
	h := C.crispasr_chat_open(cpath, &cparams, &cerr)
	if h == nil {
		return nil, chatError("crispasr_chat_open: failed to open "+modelPath, &cerr, 0)
	}
	return &ChatSession{handle: h}, nil
}

// ChatMemoryEstimate returns a conservative working set in bytes (weights +
// KV cache + activations) for a GGUF chat model on disk, reading its metadata
// but never its tensor data — a pre-flight guard for low-memory devices.
// params matters mostly for NCtx, which sizes the KV term linearly; pass nil
// for the ABI defaults, which leave NCtx at the model's own trained context.
//
// The number is deliberately high, not approximate. The KV term bills both the
// K and the V cache at the full attention width n_embd, but a grouped-query
// model gives each layer a K/V width that is a fraction of that: on Gemma 3 1B
// the KV term comes out 4.50x llama.cpp's real cache (117.00 MiB against
// 26.00 MiB at NCtx 1024), which is 1.33x on the whole estimate at NCtx 4096.
// Over-reporting is the safe direction for a "will this fit?" guard: it can
// turn away a model that would just have fitted, and never admits one that
// would not.
//
// A model that could not be read comes back as an error, not as a zero
// estimate.
func ChatMemoryEstimate(modelPath string, params *ChatOpenParams) (uint64, error) {
	cpath, err := chatCString(modelPath, "modelPath")
	if err != nil {
		return 0, err
	}
	defer C.free(unsafe.Pointer(cpath))

	cparams, freeParams, err := chatOpenParamsToC(params)
	defer freeParams()
	if err != nil {
		return 0, err
	}

	var cerr C.crispasr_chat_error
	n := C.crispasr_chat_memory_estimate(cpath, &cparams, &cerr)
	if n == 0 {
		return 0, chatError("crispasr_chat_memory_estimate: could not estimate "+modelPath, &cerr, 0)
	}
	return uint64(n), nil
}

// Close frees the session and its KV cache. Safe to call more than once.
//
// Safe from another goroutine while a generation runs, and it WAITS for that
// generation rather than cutting it off. Two things cover the handle between
// them: s.mu below, for the window between reading s.handle and entering C,
// which C cannot see into; and crispasr_chat_close itself, which counts the
// calls already inside the session and waits for them. A generation holds the
// session for as long as it decodes, so cancel with SetAbortCallback first if
// the length of the wait matters.
func (s *ChatSession) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle != nil {
		C.crispasr_chat_close(s.handle)
		s.handle = nil
	}
}

// Reset clears the KV cache so the next generation re-prefills from scratch.
// Call it when starting a new conversation in a reused session. An abort does
// this for you — a cancelled session needs no Reset before its next use.
func (s *ChatSession) Reset() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return errors.New("crispasr_chat_reset: session is closed")
	}
	var cerr C.crispasr_chat_error
	if rc := C.crispasr_chat_reset(s.handle, &cerr); rc != 0 {
		return chatError("crispasr_chat_reset failed", &cerr, int32(rc))
	}
	return nil
}

// NCtx returns the session's context window in tokens.
func (s *ChatSession) NCtx() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return 0
	}
	return int(C.crispasr_chat_n_ctx(s.handle))
}

// TemplateName returns the name of the chat template the session resolved
// against, e.g. "chatml", "llama3", "gemma".
func (s *ChatSession) TemplateName() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return ""
	}
	return C.GoString(C.crispasr_chat_template_name(s.handle))
}

// SetAbortCallback registers a predicate that can cancel a generation.
//
// Polarity: shouldContinue returns TRUE to LET THE GENERATION CONTINUE and
// false to abort it. That is the polarity of the C callback it is handed to,
// crispasr_chat_abort_callback, and of the ASR side's EncoderBeginCallback;
// this binding forwards the answer unchanged. Passing nil clears the predicate.
//
// The predicate is handed to the C session for the length of each Generate or
// GenerateStream call and taken back when the call returns, so it is only ever
// live on the goroutine running the generation. Register it before starting
// one and have it read your own flag — atomically, since it runs on the
// generating thread. It must be cheap and non-blocking: on the CPU backend it
// is called from inside a running compute graph, many times per batch. It must
// not call back into the same session, which deadlocks on the session lock.
//
// A panic from the predicate is captured, stops the generation, and is
// re-raised on the goroutine that called Generate / GenerateStream once the
// native call has returned. It never unwinds through C.
func (s *ChatSession) SetAbortCallback(shouldContinue func() bool) {
	s.abortMu.Lock()
	defer s.abortMu.Unlock()
	s.shouldContinue = shouldContinue
}

func (s *ChatSession) abortCallback() func() bool {
	s.abortMu.Lock()
	defer s.abortMu.Unlock()
	return s.shouldContinue
}

// CountTokens returns the number of tokens the model's own tokenizer produces
// for messages once the session's chat template has been applied — the prompt
// length a FRESH session prefills, so it compares straight against NCtx.
//
// The count covers the whole prompt: the template's control tokens, the
// leading BOS and the trailing generation prompt. An empty messages slice
// counts the template's own opening, which is whatever that template emits for
// no messages — template-dependent, and possibly nothing at all: several chat
// templates write only from inside their loop over the messages, and those
// return 0. Do not read a positive overhead into it.
//
// It is a pure query: it neither touches the KV cache nor extends the
// history. For a
// session part-way through a conversation it is an upper bound, since that
// session re-decodes only the suffix its history does not already hold.
func (s *ChatSession) CountTokens(messages []ChatMessage) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return 0, errors.New("crispasr_chat_count_tokens: session is closed")
	}

	cmsgs, freeMsgs, err := chatMessagesToC(messages)
	defer freeMsgs()
	if err != nil {
		return 0, err
	}

	var cerr C.crispasr_chat_error
	n := C.crispasr_chat_count_tokens(s.handle, chatMessagesPtr(cmsgs), C.size_t(len(messages)), &cerr)
	if n < 0 {
		// A negative return is the failure sentinel, not an error code, so
		// there is no hint to fall back on — cerr is the only carrier.
		return 0, chatError("crispasr_chat_count_tokens failed", &cerr, 0)
	}
	return int(n), nil
}

// Generate applies the chat template to messages, prefills and runs generation
// to MaxTokens or a stop sequence, and returns the assistant's reply.
//
// Pass the WHOLE conversation on every call, not just the new turn: the
// session compares the templated prompt against the tokens it already holds
// and decodes only what is new. Passing just the latest turn is not wrong, it
// simply shares no prefix and re-prefills from scratch.
//
// On cancellation the error wraps ErrChatAborted and the session is left as if
// freshly opened — no Reset needed.
func (s *ChatSession) Generate(messages []ChatMessage, params *ChatGenerateParams) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return "", errors.New("crispasr_chat_generate: session is closed")
	}

	cmsgs, freeMsgs, err := chatMessagesToC(messages)
	defer freeMsgs()
	if err != nil {
		return "", err
	}
	cparams, freeParams, err := chatGenerateParamsToC(params)
	defer freeParams()
	if err != nil {
		return "", err
	}

	call := &chatCall{shouldContinue: s.abortCallback()}
	_, done := s.begin(call)
	defer done()

	var cerr C.crispasr_chat_error
	out := C.crispasr_chat_generate(s.handle, chatMessagesPtr(cmsgs), C.size_t(len(messages)), &cparams, &cerr)

	// A callback panic outranks whatever the native call reported, and is
	// re-raised only now that no C frame is left on the stack.
	call.repanic()

	if out == nil {
		return "", chatError("crispasr_chat_generate failed", &cerr, 0)
	}
	defer C.crispasr_chat_string_free(out)
	return C.GoString(out), nil
}

// GenerateStream is Generate with the reply delivered chunk by chunk as it is
// decoded. Concatenating the chunks gives the same text Generate returns for
// the same messages and params, except when a stop sequence ends the
// generation: the C side hands each piece to the callback before it scans for
// a match, so the chunk the match lands in has already been delivered, while
// Generate's return value is truncated before the match. With SetStop in play
// the streamed text is therefore Generate's text plus that last chunk, and a
// caller who wants the truncated form has to cut it back themselves. onToken
// may be nil.
//
// Every chunk is whole characters, and therefore valid UTF-8. A model that
// spells a character the tokeniser does not hold emits it one byte per token,
// so those bytes are buffered here and delivered when the character is
// complete: such a run of tokens produces ONE call rather than one call per
// token, and the number of calls is therefore at most the number of tokens,
// not equal to it. If the generation stops part-way through a character the
// leftover bytes are delivered as replacement characters once the call
// finishes, rather than dropped.
//
// onToken runs on the generating thread while the session lock is held, so it
// must not call back into the same session. A panic from it is captured,
// suppresses the remaining chunks, and is re-raised on the calling goroutine
// after the native call returns; it never unwinds through C. If an abort
// predicate is registered the generation is also stopped at the next
// opportunity: from then on the predicate is no longer consulted and the
// answer is "stop". With no predicate registered there is no way to ask the
// ABI to stop, so the call runs to completion first.
func (s *ChatSession) GenerateStream(messages []ChatMessage, params *ChatGenerateParams, onToken func(string)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return errors.New("crispasr_chat_generate_stream: session is closed")
	}

	cmsgs, freeMsgs, err := chatMessagesToC(messages)
	defer freeMsgs()
	if err != nil {
		return err
	}
	cparams, freeParams, err := chatGenerateParamsToC(params)
	defer freeParams()
	if err != nil {
		return err
	}

	call := &chatCall{onToken: onToken, shouldContinue: s.abortCallback()}
	handle, done := s.begin(call)
	defer done()

	var cerr C.crispasr_chat_error
	rc := C.crispasr_go_chat_generate_stream(s.handle, chatMessagesPtr(cmsgs), C.size_t(len(messages)),
		&cparams, C.uintptr_t(handle), &cerr)

	// Whatever is still buffered belongs to the caller, aborted run or not.
	call.flush()
	call.repanic()

	if rc != 0 {
		return chatError("crispasr_chat_generate_stream failed", &cerr, int32(rc))
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// begin takes a cgo.Handle on the call state — the only thing that crosses
// into C as the `user` word — registers the abort trampoline if there is a
// predicate to carry, and returns the cleanup to defer.
func (s *ChatSession) begin(call *chatCall) (cgo.Handle, func()) {
	handle := cgo.NewHandle(call)
	if call.shouldContinue == nil {
		return handle, handle.Delete
	}
	C.crispasr_go_chat_set_abort(s.handle, C.uintptr_t(handle))
	return handle, func() {
		C.crispasr_go_chat_clear_abort(s.handle)
		handle.Delete()
	}
}

// repanic re-raises a captured callback panic on the calling goroutine, now
// that the native call has returned and no C frame is left on the stack.
func (c *chatCall) repanic() {
	if c.panicked {
		panic(c.panicVal)
	}
}

// chatMessagesToC copies the messages into C memory. The returned slice holds
// no Go pointers, so it is legal to hand its backing array to C. The cleanup
// is always safe to call, error or not.
func chatMessagesToC(messages []ChatMessage) ([]C.crispasr_chat_message, func(), error) {
	if len(messages) == 0 {
		return nil, func() {}, nil
	}
	out := make([]C.crispasr_chat_message, len(messages))
	owned := make([]*C.char, 0, 2*len(messages))
	free := func() {
		for _, p := range owned {
			C.free(unsafe.Pointer(p))
		}
	}
	for i, m := range messages {
		role, err := chatCString(m.Role, fmt.Sprintf("messages[%d].Role", i))
		if err != nil {
			free()
			return nil, func() {}, err
		}
		owned = append(owned, role)
		content, err := chatCString(m.Content, fmt.Sprintf("messages[%d].Content", i))
		if err != nil {
			free()
			return nil, func() {}, err
		}
		owned = append(owned, content)
		out[i].role = role
		out[i].content = content
	}
	return out, free, nil
}

func chatMessagesPtr(cmsgs []C.crispasr_chat_message) *C.crispasr_chat_message {
	if len(cmsgs) == 0 {
		return nil
	}
	return &cmsgs[0]
}

// chatGenerateParamsToC takes the C struct the caller has been mutating and
// attaches the stop sequences to it. The stop-sequence array is C memory: it
// lives inside a struct that crosses into C, which may hold no Go pointers.
// The cleanup is always safe to call, error or not.
func chatGenerateParamsToC(params *ChatGenerateParams) (C.crispasr_chat_generate_params, func(), error) {
	if params == nil {
		params = &ChatGenerateParams{}
	}
	params.ensure()
	out := params.c

	if len(params.stop) == 0 {
		return out, func() {}, nil
	}
	mem := C.malloc(C.size_t(len(params.stop)) * C.size_t(unsafe.Sizeof((*C.char)(nil))))
	arr := unsafe.Slice((**C.char)(mem), len(params.stop))
	free := func() {
		for _, p := range arr {
			C.free(unsafe.Pointer(p))
		}
		C.free(mem)
	}
	for i := range arr {
		arr[i] = nil // C.free ignores NULL, so a partial fill stays freeable
	}
	for i, s := range params.stop {
		p, err := chatCString(s, fmt.Sprintf("stop[%d]", i))
		if err != nil {
			free()
			return out, func() {}, err
		}
		arr[i] = p
	}
	out.stop = (**C.char)(mem)
	out.n_stop = C.size_t(len(params.stop))
	return out, free, nil
}

// chatOpenParamsToC takes the C struct the caller has been mutating and
// attaches the chat-template override to it. The cleanup is always safe to
// call, error or not.
func chatOpenParamsToC(params *ChatOpenParams) (C.crispasr_chat_open_params, func(), error) {
	if params == nil {
		params = &ChatOpenParams{}
	}
	params.ensure()
	out := params.c

	if params.chatTemplate == "" {
		return out, func() {}, nil
	}
	tmpl, err := chatCString(params.chatTemplate, "ChatTemplate")
	if err != nil {
		return out, func() {}, err
	}
	out.chat_template = tmpl
	// The ABI copies the string, so the caller may free it as soon as the
	// call it was passed to has returned.
	return out, func() { C.free(unsafe.Pointer(tmpl)) }, nil
}
