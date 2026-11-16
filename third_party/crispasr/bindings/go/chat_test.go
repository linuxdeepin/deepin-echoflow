package whisper_test

import (
	"errors"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"

	// Packages
	whisper "github.com/CrispStrobe/CrispASR/bindings/go"
	assert "github.com/stretchr/testify/assert"
)

// The chat cases are gated on CRISPASR_CHAT_TEST_MODEL — a path to a small
// GGUF chat model — the same env var the Catch2 chat suite uses. Without it
// they skip, so a checkout with no model on disk stays green.
const chatModelEnv = "CRISPASR_CHAT_TEST_MODEL"

// chatSmokeMessages is the prompt every generating case runs. Terse on
// purpose: the cases assert on lengths and reproducibility, not on content.
var chatSmokeMessages = []whisper.ChatMessage{
	{Role: "system", Content: "You are a terse assistant. Answer in one word."},
	{Role: "user", Content: "Say hello."},
}

// chatVerboseMessages asks for an answer long enough to be interrupted
// part-way. The terse prompt above stops after a handful of tokens, which
// leaves nothing for an abort to cut short.
var chatVerboseMessages = []whisper.ChatMessage{
	{Role: "user", Content: "Count from one to fifty in words, one number per line."},
}

// chatMultibyteMessages asks for a reply outside the tokeniser's character
// vocabulary, so the model spells it with byte-fallback tokens: the C side
// then delivers one chunk per BYTE of each character, not one per character.
var chatMultibyteMessages = []whisper.ChatMessage{
	{Role: "user", Content: "Reply with exactly this and nothing else: \U0001fabf\U0001facf\U0001fabc"},
}

// hasSplittableChar reports whether s holds a character the streamed path
// could split — a multi-byte one that is not itself a replacement character.
func hasSplittableChar(s string) bool {
	return utf8.ValidString(s) && utf8.RuneCountInString(s) != len(s)
}

// chatModelPath returns the gated model's path, or skips the case.
func chatModelPath(t *testing.T) string {
	t.Helper()
	modelPath := os.Getenv(chatModelEnv)
	if modelPath == "" {
		t.Skip("Skipping test, " + chatModelEnv + " not set")
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, chat model not found:", modelPath)
	}
	return modelPath
}

// openChatSession opens the gated model with a small context window, or skips.
func openChatSession(t *testing.T) *whisper.ChatSession {
	t.Helper()
	modelPath := chatModelPath(t)

	params := whisper.DefaultChatOpenParams()
	params.SetNCtx(1024)
	params.SetNGPULayers(-1)

	sess, err := whisper.ChatOpen(modelPath, &params)
	if err != nil {
		t.Fatalf("ChatOpen(%s): %v", modelPath, err)
	}
	t.Cleanup(sess.Close)
	return sess
}

// greedyChatParams generates deterministically, so the same prompt always
// yields the same text and the one-shot and streamed paths are comparable.
func greedyChatParams(maxTokens int) *whisper.ChatGenerateParams {
	params := whisper.DefaultChatGenerateParams()
	params.SetMaxTokens(maxTokens)
	params.SetTemperature(0.0)
	params.SetSeed(1)
	return &params
}

// Test_Chat_DisclosureText needs no model: the AI-disclosure wording is a
// static string in the library.
func Test_Chat_DisclosureText(t *testing.T) {
	assert := assert.New(t)
	text := whisper.ChatAIDisclosureText()
	assert.NotEmpty(text)
	assert.Greater(len(text), 16, "expected a sentence, not a token")
}

// Test_Chat_ParamsKeepABIDefaults is the regression guard on a partial fill:
// setting one option must leave every other one at the value
// crispasr_chat_open_params_default / crispasr_chat_generate_params_default
// wrote, not at Go's zero value. The accessors read the very C struct that
// crosses into C, so the check is exact and needs no model.
//
// The generate half is the one that bites hardest: a zeroed temperature is
// greedy decoding, silently replacing the ABI's 0.8. On the open side a zeroed
// n_batch clamps prefill to one token per batch and a zeroed n_gpu_layers
// moves the whole model onto the CPU.
func Test_Chat_ParamsKeepABIDefaults(t *testing.T) {
	assert := assert.New(t)

	genDefaults := whisper.DefaultChatGenerateParams()
	openDefaults := whisper.DefaultChatOpenParams()

	// The defaults must differ from Go's zero value, or every equality below
	// would also hold for a binding that ignored the ABI entirely.
	assert.NotZero(genDefaults.Temperature())
	assert.NotZero(genDefaults.MaxTokens())
	assert.NotZero(genDefaults.TopK())
	assert.NotZero(genDefaults.TopP())
	assert.NotZero(genDefaults.MinP())
	assert.NotZero(genDefaults.RepeatPenalty())
	assert.NotZero(genDefaults.RepeatLastN())
	assert.NotZero(openDefaults.NThreads())
	assert.NotZero(openDefaults.NThreadsBatch())
	assert.NotZero(openDefaults.NBatch())
	assert.NotZero(openDefaults.NUBatch())
	assert.NotZero(openDefaults.NGPULayers())
	assert.True(openDefaults.UseMmap())

	assertGenerateDefaultsHeld := func(p *whisper.ChatGenerateParams, what string) {
		assert.Equal(genDefaults.Temperature(), p.Temperature(), what+" changed temperature")
		assert.Equal(genDefaults.TopK(), p.TopK(), what+" changed top_k")
		assert.Equal(genDefaults.TopP(), p.TopP(), what+" changed top_p")
		assert.Equal(genDefaults.MinP(), p.MinP(), what+" changed min_p")
		assert.Equal(genDefaults.RepeatPenalty(), p.RepeatPenalty(), what+" changed repeat_penalty")
		assert.Equal(genDefaults.RepeatLastN(), p.RepeatLastN(), what+" changed repeat_last_n")
		assert.Equal(genDefaults.Seed(), p.Seed(), what+" changed seed")
		assert.Equal(genDefaults.PrefillOnly(), p.PrefillOnly(), what+" changed prefill_only")
	}

	gen := whisper.DefaultChatGenerateParams()
	gen.SetMaxTokens(64)
	assert.Equal(64, gen.MaxTokens())
	assertGenerateDefaultsHeld(&gen, "SetMaxTokens")

	// A value that never went through DefaultChatGenerateParams falls into the
	// same hole unless the zero value seeds itself from the ABI.
	var bareGen whisper.ChatGenerateParams
	bareGen.SetMaxTokens(64)
	assert.Equal(64, bareGen.MaxTokens())
	assertGenerateDefaultsHeld(&bareGen, "SetMaxTokens on the zero value")

	assertOpenDefaultsHeld := func(p *whisper.ChatOpenParams, what string) {
		assert.Equal(openDefaults.NThreads(), p.NThreads(), what+" changed n_threads")
		assert.Equal(openDefaults.NThreadsBatch(), p.NThreadsBatch(), what+" changed n_threads_batch")
		assert.Equal(openDefaults.NBatch(), p.NBatch(), what+" changed n_batch")
		assert.Equal(openDefaults.NUBatch(), p.NUBatch(), what+" changed n_ubatch")
		assert.Equal(openDefaults.NGPULayers(), p.NGPULayers(), what+" changed n_gpu_layers")
		assert.Equal(openDefaults.UseMmap(), p.UseMmap(), what+" changed use_mmap")
		assert.Equal(openDefaults.UseMlock(), p.UseMlock(), what+" changed use_mlock")
		assert.Equal(openDefaults.ChatTemplate(), p.ChatTemplate(), what+" changed chat_template")
	}

	open := whisper.DefaultChatOpenParams()
	open.SetNCtx(4096)
	assert.Equal(4096, open.NCtx())
	assertOpenDefaultsHeld(&open, "SetNCtx")

	var bareOpen whisper.ChatOpenParams
	bareOpen.SetNCtx(4096)
	assert.Equal(4096, bareOpen.NCtx())
	assertOpenDefaultsHeld(&bareOpen, "SetNCtx on the zero value")
}

// Test_Chat_BareOpenParamsReachC is the C-side half of the case above: a
// ChatOpenParams that never went through DefaultChatOpenParams still carries
// the one option that was set all the way into the session.
func Test_Chat_BareOpenParamsReachC(t *testing.T) {
	assert := assert.New(t)
	modelPath := chatModelPath(t)

	var open whisper.ChatOpenParams
	open.SetNCtx(1024)

	sess, err := whisper.ChatOpen(modelPath, &open)
	assert.NoError(err)
	if err != nil {
		return
	}
	defer sess.Close()
	assert.Equal(1024, sess.NCtx(), "the set field did not reach the session")
}

// Test_Chat_BareGenerateParamsReachC is the generate-side counterpart: a
// ChatGenerateParams nobody touched must decode exactly as nil does, both
// being the ABI defaults. The comparison is deterministic — the default seed
// is 0, a fixed seed, not a random one — and it is sensitive because the
// failure mode is greedy decoding, which sampling at the ABI's temperature
// does not reproduce.
func Test_Chat_BareGenerateParamsReachC(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)

	withNil, err := sess.Generate(chatVerboseMessages, nil)
	assert.NoError(err)
	assert.NotEmpty(withNil)
	assert.NoError(sess.Reset())

	withBare, err := sess.Generate(chatVerboseMessages, &whisper.ChatGenerateParams{})
	assert.NoError(err)
	assert.Equal(withNil, withBare, "an untouched params value did not decode as the ABI defaults")
}

// Test_Chat_MemoryEstimate checks the pre-flight estimate covers the weights
// on disk and that its KV term scales with the context window.
func Test_Chat_MemoryEstimate(t *testing.T) {
	assert := assert.New(t)
	modelPath := chatModelPath(t)

	info, err := os.Stat(modelPath)
	assert.NoError(err)
	fileSize := uint64(info.Size())
	assert.Greater(fileSize, uint64(0))

	at := func(nCtx int) uint64 {
		var params *whisper.ChatOpenParams
		if nCtx > 0 {
			params = &whisper.ChatOpenParams{}
			params.SetNCtx(nCtx)
		}
		n, err := whisper.ChatMemoryEstimate(modelPath, params)
		assert.NoError(err)
		return n
	}

	// nil params: the model's own trained context sizes the KV term.
	assert.Greater(at(0), fileSize)

	at1k, at2k, at4k := at(1024), at(2048), at(4096)
	assert.Greater(at1k, fileSize)
	assert.Greater(at2k, at1k)
	assert.Greater(at4k, at2k)

	// The KV term is linear in NCtx, so doubling the context doubles the
	// amount by which the estimate grows. A load path that returned before
	// reading the context / layer / embedding metadata would leave every
	// difference at zero and still report success.
	assert.Equal(at4k-at2k, 2*(at2k-at1k),
		"KV term not linear in NCtx: %d / %d / %d", at1k, at2k, at4k)

	// Everything outside the KV term is context-independent, so back it out
	// and the remainder still has to cover the weights on disk.
	assert.Greater(at1k-(at2k-at1k), fileSize)
}

// Test_Chat_MemoryEstimateRejectsAnUnreadableModel pins that the C side's
// "0 with err filled" failure signal becomes a Go error rather than an
// estimate of nothing.
func Test_Chat_MemoryEstimateRejectsAnUnreadableModel(t *testing.T) {
	assert := assert.New(t)
	n, err := whisper.ChatMemoryEstimate("/nonexistent/crispasr-memory-estimate.gguf", nil)
	assert.Error(err)
	assert.Equal(uint64(0), n)
}

// Test_Chat_InteriorNulIsRejected checks a NUL inside a string bound for C is
// refused rather than silently truncating the value at the NUL, which is where
// C stops reading it.
func Test_Chat_InteriorNulIsRejected(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)

	_, err := sess.CountTokens([]whisper.ChatMessage{{Role: "user", Content: "before\x00after"}})
	assert.Error(err)
	assert.Contains(err.Error(), "messages[0].Content")

	_, err = sess.CountTokens([]whisper.ChatMessage{
		{Role: "user", Content: "fine"},
		{Role: "us\x00er", Content: "hi"},
	})
	assert.Error(err)
	assert.Contains(err.Error(), "messages[1].Role")

	params := greedyChatParams(8)
	params.SetStop("ok", "sto\x00p")
	_, err = sess.Generate(chatSmokeMessages, params)
	assert.Error(err)
	assert.Contains(err.Error(), "stop[1]")

	// ChatTemplate is refused before the model is touched.
	open := whisper.DefaultChatOpenParams()
	open.SetChatTemplate("{{ bos_token }}\x00{{ messages }}")
	_, err = whisper.ChatOpen(os.Getenv(chatModelEnv), &open)
	assert.Error(err)
	assert.Contains(err.Error(), "ChatTemplate")

	// The model path too, on both entry points that take one: truncating it
	// at the NUL would open a different file from the one named — or, worse
	// for a guard, estimate one.
	badPath := os.Getenv(chatModelEnv) + "\x00.gguf"
	_, err = whisper.ChatOpen(badPath, nil)
	assert.Error(err)
	assert.Contains(err.Error(), "modelPath")

	_, err = whisper.ChatMemoryEstimate(badPath, nil)
	assert.Error(err)
	assert.Contains(err.Error(), "modelPath")

	// None of it disturbed the session.
	n, err := sess.CountTokens(chatSmokeMessages)
	assert.NoError(err)
	assert.Greater(n, 0)
}

// Test_Chat_Open checks the two introspection accessors report something
// usable: a context window and the name of the resolved chat template.
func Test_Chat_Open(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)

	assert.Greater(sess.NCtx(), 0, "expected a context size")
	assert.NotEmpty(sess.TemplateName(), "expected a resolved template name")
}

// Test_Chat_Generate exercises the one-shot path.
func Test_Chat_Generate(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)

	text, err := sess.Generate(chatSmokeMessages, greedyChatParams(16))
	assert.NoError(err)
	assert.NotEmpty(strings.TrimSpace(text), "expected generated text")
	t.Logf("one-shot reply: %q", text)
}

// Test_Chat_GenerateStream checks the streamed chunks concatenate to exactly
// the one-shot output for the same messages and params — the regression guard
// against the two paths drifting apart.
func Test_Chat_GenerateStream(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)
	params := greedyChatParams(16)

	oneShot, err := sess.Generate(chatSmokeMessages, params)
	assert.NoError(err)
	assert.NotEmpty(oneShot)

	assert.NoError(sess.Reset())

	var streamed strings.Builder
	nChunks := 0
	err = sess.GenerateStream(chatSmokeMessages, params, func(chunk string) {
		nChunks++
		streamed.WriteString(chunk)
	})
	assert.NoError(err)
	assert.Greater(nChunks, 0, "on_token never fired")
	assert.Equal(oneShot, streamed.String(), "streamed text != one-shot text")
}

// Test_Chat_GenerateStreamSplitMultibyte checks the same equality for a reply
// the model spells one byte at a time, where a chunk can end part-way through
// a character — and that no caller ever sees one of those bare bytes. The
// concatenation alone is not enough: a Go string carries arbitrary bytes, so
// it holds whatever C delivered whether or not each piece was a character.
func Test_Chat_GenerateStreamSplitMultibyte(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)
	params := greedyChatParams(32)

	oneShot, err := sess.Generate(chatMultibyteMessages, params)
	assert.NoError(err)
	if !hasSplittableChar(oneShot) {
		t.Skipf("this model answered %q, with nothing to split", oneShot)
	}

	assert.NoError(sess.Reset())

	var streamed strings.Builder
	nChunks := 0
	err = sess.GenerateStream(chatMultibyteMessages, params, func(chunk string) {
		nChunks++
		assert.True(utf8.ValidString(chunk),
			"chunk % x is not valid UTF-8 on its own", []byte(chunk))
		streamed.WriteString(chunk)
	})
	assert.NoError(err)
	assert.Equal(oneShot, streamed.String(),
		"a character split across chunks must survive")
	// Every delivered chunk holds at least one whole character, so there
	// cannot be more chunks than characters. One chunk per BYTE — what the C
	// side actually hands over here — breaks this.
	assert.LessOrEqual(nChunks, utf8.RuneCountInString(oneShot),
		"the bytes of one character must arrive as one chunk, not several")
}

// Test_Chat_CountTokens checks the count is positive and grows with the
// conversation. An empty conversation still counts the template's own opening.
func Test_Chat_CountTokens(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)

	// Never an error, whatever the template renders for no messages; for
	// this model's template that opening is a real, positive cost.
	empty, err := sess.CountTokens(nil)
	assert.NoError(err)
	assert.GreaterOrEqual(empty, 0)
	assert.Greater(empty, 0, "gemma opens the assistant turn for add_ass")

	// One user turn, not one system turn: some templates (Gemma's among them)
	// have no system role and fold a lone system message away to nothing.
	short, err := sess.CountTokens([]whisper.ChatMessage{{Role: "user", Content: "Say hello."}})
	assert.NoError(err)
	assert.Greater(short, empty)

	long, err := sess.CountTokens(append(chatSmokeMessages,
		whisper.ChatMessage{Role: "assistant", Content: "Hello. " + strings.Repeat("More words here. ", 40)}))
	assert.NoError(err)
	assert.Greater(long, short, "count is not monotone in conversation length")
	assert.Less(long, sess.NCtx(), "test prompt should fit the context window")
}

// Test_Chat_AbortStopsStreamAndSessionIsReusable checks the cancellation
// contract end to end: the predicate stops the stream part-way, the error is
// distinguishable from a decode fault, and the session is immediately reusable
// with NO Reset — an abort flushes it back to its just-opened state.
func Test_Chat_AbortStopsStreamAndSessionIsReusable(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)
	params := greedyChatParams(64)

	full, err := sess.Generate(chatVerboseMessages, params)
	assert.NoError(err)
	assert.NotEmpty(full)
	assert.NoError(sess.Reset())

	// Stop once a few chunks have arrived.
	var seen int32
	sess.SetAbortCallback(func() bool { return atomic.LoadInt32(&seen) < 3 })
	var partial strings.Builder
	err = sess.GenerateStream(chatVerboseMessages, params, func(chunk string) {
		atomic.AddInt32(&seen, 1)
		partial.WriteString(chunk)
	})
	sess.SetAbortCallback(nil)

	assert.Error(err)
	assert.True(errors.Is(err, whisper.ErrChatAborted), "abort must be distinguishable from a fault, got %v", err)
	assert.Greater(int(atomic.LoadInt32(&seen)), 0, "nothing was delivered before the abort")
	assert.Less(len(partial.String()), len(full), "abort did not stop the stream early")
	t.Logf("aborted after %d chunks: %d of %d bytes", atomic.LoadInt32(&seen), len(partial.String()), len(full))

	// No Reset here on purpose: the abort already flushed the KV cache and the
	// history, so the next generation must prefill from scratch and reproduce
	// what the untouched session produced.
	again, err := sess.Generate(chatVerboseMessages, params)
	assert.NoError(err)
	assert.Equal(full, again, "session not reusable after an abort without a reset")
}

// Test_Chat_AbortPolarity pins which way round the Go predicate reads: it
// returns TRUE to CONTINUE, the same way round as the C callback. Both halves
// are needed — either one alone passes under an inverted implementation.
func Test_Chat_AbortPolarity(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)
	params := greedyChatParams(48)

	// true means "keep going": the generation must complete normally.
	var alwaysCalls int32
	sess.SetAbortCallback(func() bool { atomic.AddInt32(&alwaysCalls, 1); return true })
	var completed strings.Builder
	err := sess.GenerateStream(chatVerboseMessages, params, func(chunk string) { completed.WriteString(chunk) })
	sess.SetAbortCallback(nil)

	assert.NoError(err, "a predicate returning true must not abort")
	assert.Greater(int(atomic.LoadInt32(&alwaysCalls)), 0, "the predicate was never consulted")
	assert.NotEmpty(completed.String())

	assert.NoError(sess.Reset())

	// false means "stop now": the generation must be cut short.
	var neverCalls int32
	sess.SetAbortCallback(func() bool { atomic.AddInt32(&neverCalls, 1); return false })
	var stopped strings.Builder
	err = sess.GenerateStream(chatVerboseMessages, params, func(chunk string) { stopped.WriteString(chunk) })
	sess.SetAbortCallback(nil)

	assert.True(errors.Is(err, whisper.ErrChatAborted), "a predicate returning false must abort, got %v", err)
	assert.Greater(int(atomic.LoadInt32(&neverCalls)), 0, "the predicate was never consulted")
	assert.Less(len(stopped.String()), len(completed.String()), "aborting produced as much text as completing")
}

// Test_Chat_TokenCallbackPanic checks a panic from the token callback is
// captured, re-raised on the calling goroutine once the native call has
// returned, and never unwound through C — a panic crossing the C frame kills
// the process, so reaching the recover at all is the evidence.
func Test_Chat_TokenCallbackPanic(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)

	sentinel := errors.New("token callback exploded")
	recovered := func() (r any) {
		defer func() { r = recover() }()
		_ = sess.GenerateStream(chatSmokeMessages, greedyChatParams(8), func(string) { panic(sentinel) })
		return nil
	}()
	assert.Equal(sentinel, recovered, "the callback's panic must reach the caller")

	// The session survived the C call and is still usable.
	text, err := sess.Generate(chatSmokeMessages, greedyChatParams(8))
	assert.NoError(err)
	assert.NotEmpty(strings.TrimSpace(text))
}

// Test_Chat_AbortCallbackPanic is the same for the abort predicate, which
// additionally stops the generation: a predicate that cannot answer is treated
// as an abort.
func Test_Chat_AbortCallbackPanic(t *testing.T) {
	assert := assert.New(t)
	sess := openChatSession(t)

	sentinel := errors.New("abort predicate exploded")
	sess.SetAbortCallback(func() bool { panic(sentinel) })
	recovered := func() (r any) {
		defer func() { r = recover() }()
		_ = sess.GenerateStream(chatSmokeMessages, greedyChatParams(8), nil)
		return nil
	}()
	sess.SetAbortCallback(nil)
	assert.Equal(sentinel, recovered, "the predicate's panic must reach the caller")

	text, err := sess.Generate(chatSmokeMessages, greedyChatParams(8))
	assert.NoError(err)
	assert.NotEmpty(strings.TrimSpace(text))
}

// chatCountingMessages is a prompt whose greedy reply is fixed and made of
// short, distinct pieces, so a stop substring can be placed inside it and the
// truncated text pinned exactly. The reply this model gives is
// "1\n2\n3\n4\n5\n6\n7\n8\n".
var chatCountingMessages = []whisper.ChatMessage{
	{Role: "user", Content: "Count from 1 to 8. Write only the numbers, one per line, nothing else."},
}

// chatCountingStoppedAtFour is what chatCountingMessages yields once
// generation stops on "4" — the text the caller receives, with the match
// itself cut off. The Rust and Python chat suites assert this same string for
// the same prompt, stop list and sampler settings: three separate
// marshallings of one C feature, agreeing byte for byte.
const chatCountingStoppedAtFour = "1\n2\n3\n"

// chatCountingBaselineReply is the full greedy reply the literals above
// describe. openStopChatSession confirms the gated model actually produces it
// before any case asserts a truncation of it.
const chatCountingBaselineReply = "1\n2\n3\n4\n5\n6\n7\n8\n"

// openStopChatSession opens the gated model with the context and batch sizes
// the Rust and Python suites pin, so the text below is comparable across all
// three bindings rather than merely reproducible in this one.
func openStopChatSession(t *testing.T) *whisper.ChatSession {
	t.Helper()
	modelPath := chatModelPath(t)

	params := whisper.DefaultChatOpenParams()
	params.SetNCtx(2048)
	params.SetNBatch(256)
	params.SetNUBatch(256)
	params.SetNGPULayers(-1)

	sess, err := whisper.ChatOpen(modelPath, &params)
	if err != nil {
		t.Fatalf("ChatOpen(%s): %v", modelPath, err)
	}
	t.Cleanup(sess.Close)

	return sess
}

// requirePinnedStopBaseline skips the caller unless the gated model is the one
// the stop-sequence literals describe.
//
// Those literals are one MODEL's greedy reply, not a property of the stop
// feature, while the gate accepts any small chat GGUF. On
// smollm2-360m-instruct this prompt answers "1 2 3 4 " with SPACES, so the
// literal cases failed on the separator while every behavioural assertion
// beside them passed — a red for a reason unrelated to the code under test.
//
// The cross-binding oracle is worth keeping (Rust, Python, Java and Dart pin
// the same strings), so rather than weaken the assertions this checks the
// precondition they encode. Called only by the cases that assert a literal, so
// the model-independent ones (empty stop list, prefill-only) still run
// everywhere.
func requirePinnedStopBaseline(t *testing.T, sess *whisper.ChatSession) {
	t.Helper()
	baseline, err := sess.Generate(chatCountingMessages, greedyChatParams(64))
	if err != nil {
		t.Skipf("Skipping literal stop-sequence assertions, baseline generate failed: %v", err)
	}
	if err := sess.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if baseline != chatCountingBaselineReply {
		t.Skipf("Skipping literal stop-sequence assertions: they are pinned to the model whose greedy reply "+
			"is %q (e.g. gemma-3-1b-it-Q4_K_M); this model replies %q. Behaviour is covered "+
			"model-independently by tests/test-chat-ggml.cpp.", chatCountingBaselineReply, baseline)
	}
}

// stopChatParams is greedyChatParams with stop sequences attached.
func stopChatParams(maxTokens int, stop ...string) *whisper.ChatGenerateParams {
	params := greedyChatParams(maxTokens)
	params.SetStop(stop...)
	return params
}

// Test_Chat_StopSequenceTruncatesBeforeTheMatch covers the one-shot path: the
// reply is cut at the match and the matched text never reaches the caller.
func Test_Chat_StopSequenceTruncatesBeforeTheMatch(t *testing.T) {
	assert := assert.New(t)
	sess := openStopChatSession(t)
	requirePinnedStopBaseline(t, sess)

	full, err := sess.Generate(chatCountingMessages, greedyChatParams(64))
	assert.NoError(err)
	// Without this the case is vacuous: a reply that never reaches the stop
	// string would pass whether or not stop sequences work at all.
	assert.Contains(full, "5", "the unstopped reply must contain the stop string")
	assert.NoError(sess.Reset())

	stopped, err := sess.Generate(chatCountingMessages, stopChatParams(64, "5"))
	assert.NoError(err)
	assert.NotContains(stopped, "5", "the matched text must not reach the caller")
	assert.True(strings.HasPrefix(full, stopped), "%q is not a prefix of %q", stopped, full)
	assert.Equal("1\n2\n3\n4\n", stopped)
}

// Test_Chat_StopSequencesEarliestMatchWins passes more than one sequence, in
// both orders — the case that exercises the array rather than a single
// pointer. "4" is generated before "7", so "4" wins either way: the earliest
// match in the output decides, not the position in the array.
func Test_Chat_StopSequencesEarliestMatchWins(t *testing.T) {
	assert := assert.New(t)
	sess := openStopChatSession(t)
	requirePinnedStopBaseline(t, sess)

	full, err := sess.Generate(chatCountingMessages, greedyChatParams(64))
	assert.NoError(err)
	assert.Contains(full, "4")
	assert.Contains(full, "7")

	for _, stop := range [][]string{{"7", "4"}, {"4", "7"}} {
		assert.NoError(sess.Reset())
		params := stopChatParams(64, stop...)
		assert.Equal(stop, params.Stop(), "SetStop must round-trip the whole array")

		stopped, err := sess.Generate(chatCountingMessages, params)
		assert.NoError(err)
		assert.Equal(chatCountingStoppedAtFour, stopped, "stop list %q", stop)
		assert.NotContains(stopped, "4")
		assert.NotContains(stopped, "7")
	}
}

// Test_Chat_EmptyStopListIsTheSameAsNone pins that SetStop with no arguments
// leaves the ABI's own "no stop sequences" state rather than sending an empty
// array the C side could read past.
func Test_Chat_EmptyStopListIsTheSameAsNone(t *testing.T) {
	assert := assert.New(t)
	sess := openStopChatSession(t)

	none, err := sess.Generate(chatCountingMessages, greedyChatParams(64))
	assert.NoError(err)
	assert.NoError(sess.Reset())

	cleared := stopChatParams(64, "4")
	cleared.SetStop()
	assert.Empty(cleared.Stop(), "SetStop with no arguments must clear the list")

	empty, err := sess.Generate(chatCountingMessages, cleared)
	assert.NoError(err)
	assert.Equal(none, empty, "an empty stop list must not truncate")
	// The reply is long enough that a stop list WOULD have truncated it, so
	// the equality above is not two empty strings agreeing.
	assert.Contains(none, "4")
}

// Test_Chat_PrefillOnlySuppressesGeneration covers the other never-exercised
// generate param: the prompt is prefilled, no assistant text is produced, and
// neither path reports a failure for the empty result.
func Test_Chat_PrefillOnlySuppressesGeneration(t *testing.T) {
	assert := assert.New(t)
	sess := openStopChatSession(t)

	params := greedyChatParams(64)
	params.SetPrefillOnly(true)
	assert.True(params.PrefillOnly())

	out, err := sess.Generate(chatCountingMessages, params)
	assert.NoError(err, "prefill-only must succeed, not fail on the empty string")
	assert.Equal("", out)

	assert.NoError(sess.Reset())
	nChunks := 0
	err = sess.GenerateStream(chatCountingMessages, params, func(string) { nChunks++ })
	assert.NoError(err)
	assert.Equal(0, nChunks, "prefill_only must emit no token chunks")

	// Positive control: the same messages and sampler settings without the
	// flag do produce text, so the two emptinesses above are the flag's doing
	// and not a prompt that generates nothing.
	assert.NoError(sess.Reset())
	control, err := sess.Generate(chatCountingMessages, greedyChatParams(64))
	assert.NoError(err)
	assert.NotEmpty(control)
}

// Test_Chat_StreamDeliversTheChunkTheOneShotPathTruncates pins the one place
// the two paths differ. The C side hands each piece to the callback before it
// scans for a stop match, so the streamed text carries the matched piece that
// the one-shot return value has cut off.
func Test_Chat_StreamDeliversTheChunkTheOneShotPathTruncates(t *testing.T) {
	assert := assert.New(t)
	sess := openStopChatSession(t)
	requirePinnedStopBaseline(t, sess)
	params := stopChatParams(64, "7", "4")

	oneShot, err := sess.Generate(chatCountingMessages, params)
	assert.NoError(err)
	assert.NoError(sess.Reset())

	var streamed strings.Builder
	err = sess.GenerateStream(chatCountingMessages, params, func(chunk string) {
		streamed.WriteString(chunk)
	})
	assert.NoError(err)

	assert.Equal(chatCountingStoppedAtFour, oneShot)
	assert.Equal("1\n2\n3\n4", streamed.String())
	assert.True(strings.HasPrefix(streamed.String(), oneShot),
		"the streamed text must extend the one-shot text")
}
