package io.github.ggerganov.whispercpp.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * End-to-end cases for the chat binding, gated on {@code CRISPASR_CHAT_TEST_MODEL}
 * — an absolute path to a GGUF chat model — the same house convention the other
 * bindings' chat suites use. The whole class self-skips when it is unset, so a
 * fresh checkout has nothing to download.
 *
 * <p>{@code memoryEstimate} is covered here because its figure is a function of
 * the model's own metadata; its failure path, which needs no model, is covered
 * in {@link ChatParamsTest} instead.
 */
@EnabledIfEnvironmentVariable(named = "CRISPASR_CHAT_TEST_MODEL", matches = ".+")
class ChatSessionTest {

    /** Three rare code points Gemma 3 has no whole token for. */
    private static final String RARE = "\uD83E\uDEBF\uD83E\uDECF\uD83E\uDEBC";

    private static final List<ChatMessage> TURNS = Collections.singletonList(
            ChatMessage.user("Reply with exactly: hello world"));

    /** A prompt whose reply runs long enough that cutting it short is visible. */
    private static final List<ChatMessage> LONG_TURNS = Collections.singletonList(
            ChatMessage.user("Count from one to forty in words, one per line."));

    /**
     * A prompt whose greedy reply is fixed and made of short, distinct pieces,
     * so a stop substring can be placed inside it and the truncated text pinned
     * exactly. The reply this model gives is "1\n2\n3\n4\n5\n6\n7\n8\n".
     */
    private static final List<ChatMessage> COUNTING_TURNS = Collections.singletonList(
            ChatMessage.user("Count from 1 to 8. Write only the numbers, one per line, "
                    + "nothing else."));

    /**
     * What {@code COUNTING_TURNS} yields once generation stops on "4" — the
     * text the caller receives, with the match itself cut off. The Rust, Python
     * and Go chat suites assert this same string for the same prompt, stop list
     * and sampler settings: four separate marshallings of one C feature,
     * agreeing byte for byte.
     */
    private static final String COUNTING_STOPPED_AT_FOUR = "1\n2\n3\n";

    /**
     * The full greedy reply the literals above describe. Those literals are one
     * MODEL's output, not a property of the stop feature, while the gate accepts
     * any small chat GGUF. On smollm2-360m-instruct this prompt answers
     * "1 2 3 4 " with SPACES, so the stop cases failed on the separator while
     * every behavioural assertion beside them passed — a red for a reason
     * unrelated to the code under test.
     *
     * The cross-binding oracle is worth keeping (Rust, Python, Go and Dart pin
     * the same strings), so rather than weaken the assertions we check the
     * precondition they encode and skip when it does not hold.
     */
    private static final String COUNTING_BASELINE_REPLY = "1\n2\n3\n4\n5\n6\n7\n8\n";

    /** Whether the gated model is the one COUNTING_STOPPED_AT_FOUR describes. */
    private static boolean pinnedStopBaseline;

    private static String modelPath;
    private static ChatSession session;

    /**
     * A second session on the open params the other bindings' chat suites use,
     * so the literals the counting cases pin are the same numbers those suites
     * pin. Kept apart from {@link #session}, whose smaller window is what the
     * context-size case asserts.
     */
    private static ChatSession countingSession;

    @BeforeAll
    static void openSession() {
        modelPath = System.getenv("CRISPASR_CHAT_TEST_MODEL");
        session = ChatSession.open(modelPath, new ChatOpenParams().nCtx(1024).nThreads(4));
        countingSession = ChatSession.open(modelPath,
                new ChatOpenParams().nCtx(2048).nBatch(256).nUbatch(256));
        String baseline = countingSession.generate(COUNTING_TURNS, greedy(64));
        countingSession.reset();
        pinnedStopBaseline = COUNTING_BASELINE_REPLY.equals(baseline);
        if (!pinnedStopBaseline) {
            System.err.println("stop-sequence literals are pinned to the model whose greedy reply is "
                    + escape(COUNTING_BASELINE_REPLY) + " (e.g. gemma-3-1b-it-Q4_K_M); this model replies "
                    + escape(baseline) + ". Those cases will be skipped; behaviour is covered "
                    + "model-independently by tests/test-chat-ggml.cpp.");
        }
    }

    private static String escape(String s) {
        return s == null ? "null" : "\"" + s.replace("\n", "\\n") + "\"";
    }

    @AfterAll
    static void closeSession() {
        if (session != null) {
            session.close();
        }
        if (countingSession != null) {
            countingSession.close();
        }
    }

    @BeforeEach
    void freshConversation() {
        session.setAbortCallback(null);
        session.reset();
    }

    /** Greedy, so a one-shot and a streamed run of the same prompt agree exactly. */
    private static ChatGenerateParams greedy(int maxTokens) {
        return new ChatGenerateParams().temperature(0f).maxTokens(maxTokens);
    }

    @Test
    void openReportsAContextSizeAndTemplateName() {
        assertTrue(session.nCtx() > 0, "context window must be positive");
        assertEquals(1024, session.nCtx(), "the requested window must be honoured");
        assertFalse(session.templateName().isEmpty(), "the resolved template must be named");
    }

    /** Estimate the model at one context window. */
    private static long estimateAt(int nCtx) {
        return ChatSession.memoryEstimate(modelPath, new ChatOpenParams().nCtx(nCtx));
    }

    /**
     * A model reached through a path outside ASCII opens and estimates.
     *
     * <p>Both entry points marshal the path with {@link ChatNative#cstring},
     * not as a JNA {@code String}: JNA converts a String through its
     * configured native encoding, which on the Java 8 target this binding
     * supports is a legacy code page on Windows, while the C side reads UTF-8.
     * This case traverses that marshalling; the encodings only actually
     * diverge on a Windows host, so a green run here is coverage of the path,
     * not proof of the Windows behaviour.
     */
    @Test
    void aModelUnderANonAsciiPathOpensAndEstimates() throws Exception {
        Path dir = Files.createTempDirectory("crispasr-chat-ümläut-日本語-");
        Path link = dir.resolve("modèle-模型.gguf");
        try {
            Files.createSymbolicLink(link, new File(modelPath).toPath().toAbsolutePath());
            String path = link.toString();

            assertTrue(ChatSession.memoryEstimate(path, null) > new File(modelPath).length(),
                    "estimate through a non-ASCII path");
            ChatSession s = ChatSession.open(path, new ChatOpenParams().nCtx(512));
            try {
                assertTrue(s.nCtx() > 0);
            } finally {
                s.close();
            }
        } finally {
            Files.deleteIfExists(link);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    void memoryEstimateCoversTheWeightsAndScalesWithContext() {
        long fileSize = new File(modelPath).length();
        assertTrue(fileSize > 0, "the model file must be readable");

        // Null params: the model's own trained context sizes the KV term.
        assertTrue(ChatSession.memoryEstimate(modelPath, null) > fileSize,
                "the estimate must cover the weights on disk");

        long at1k = estimateAt(1024);
        long at2k = estimateAt(2048);
        long at4k = estimateAt(4096);
        assertTrue(at1k > fileSize, "at 1024: " + at1k + " vs " + fileSize);
        assertTrue(at2k > at1k, at2k + " vs " + at1k);
        assertTrue(at4k > at2k, at4k + " vs " + at2k);

        // The KV term is linear in nCtx, so doubling the context doubles the
        // amount by which the estimate grows. A load path that returned before
        // reading the context / layer / embedding metadata would leave every
        // difference at zero and still report success.
        assertEquals(2 * (at2k - at1k), at4k - at2k,
                "KV term not linear in nCtx: " + at1k + " / " + at2k + " / " + at4k);

        // Everything outside the KV term is context-independent, so back it out
        // and the remainder still has to cover the weights on disk.
        assertTrue(at1k - (at2k - at1k) > fileSize,
                "the context-independent part must cover the weights");
    }

    @Test
    void generateReturnsText() {
        String out = session.generate(TURNS, greedy(32));
        assertFalse(out.trim().isEmpty(), "expected a reply");
    }

    /**
     * With no stop sequence the two paths agree exactly. The one case where
     * they do not is
     * {@link #theStreamDeliversTheChunkTheOneShotPathTruncates()}.
     */
    @Test
    void streamedChunksConcatenateToTheOneShotOutput() {
        String oneShot = session.generate(TURNS, greedy(48));
        session.reset();

        final StringBuilder streamed = new StringBuilder();
        final List<String> chunks = new ArrayList<String>();
        session.generateStream(TURNS, greedy(48), chunk -> {
            chunks.add(chunk);
            streamed.append(chunk);
        });

        assertFalse(chunks.isEmpty(), "the stream delivered nothing");
        assertEquals(oneShot, streamed.toString(), "streamed text must match the one-shot text");
        assertArrayEqualsBytes(oneShot, streamed.toString());
    }

    @Test
    void countTokensIsPositiveAndMonotone() {
        // An EMPTY message list counts the template's own opening. Never an
        // error, whatever that template renders for no messages; for this
        // model's template the opening is a real, positive cost.
        int empty = session.countTokens(Collections.<ChatMessage>emptyList());
        assertTrue(empty > 0, "an empty conversation still costs the template opening: " + empty);

        int one = session.countTokens(TURNS);
        int two = session.countTokens(Arrays.asList(
                ChatMessage.user("Reply with exactly: hello world"),
                ChatMessage.assistant("hello world"),
                ChatMessage.user("Now say it again, twice.")));
        assertTrue(one > empty, one + " should exceed " + empty);
        assertTrue(two > one, two + " should exceed " + one);
        assertTrue(two < session.nCtx(), "the prompt should fit the window it is compared against");
    }

    /**
     * Polarity, both sides. The predicate returns TRUE to CONTINUE, matching the
     * C header. One half alone would pass under an inverted implementation:
     * always-continue would abort, always-abort would run to completion.
     */
    @Test
    void abortPredicateReturnsTrueToContinue() {
        session.setAbortCallback(() -> true);
        String out = session.generate(TURNS, greedy(16));
        assertFalse(out.trim().isEmpty(), "an always-continue predicate must produce real output");

        session.setAbortCallback(() -> false);
        ChatAbortedException e = assertThrows(ChatAbortedException.class,
                () -> session.generate(TURNS, greedy(16)));
        assertEquals(ChatLib.CRISPASR_CHAT_ERR_ABORTED, e.code());
    }

    /**
     * An abort stops the stream early and leaves the session usable with NO
     * reset: the C side flushes its KV cache and history back to the
     * just-opened state on the way out.
     */
    @Test
    void abortStopsEarlyAndTheSessionStaysUsableWithoutAReset() {
        final ChatGenerateParams params = greedy(120);
        final List<String> full = new ArrayList<String>();
        session.generateStream(LONG_TURNS, params, full::add);
        assertTrue(full.size() > 10, "need a run long enough to cut short: " + full.size());
        session.reset();

        final AtomicInteger delivered = new AtomicInteger();
        final List<String> partial = new ArrayList<String>();
        session.setAbortCallback(() -> delivered.get() < 2);
        assertThrows(ChatAbortedException.class, () -> session.generateStream(LONG_TURNS, params,
                chunk -> {
                    partial.add(chunk);
                    delivered.incrementAndGet();
                }));
        assertTrue(partial.size() < full.size(),
                "aborted run delivered " + partial.size() + " of " + full.size() + " chunks");

        // No reset() here: that is the point.
        session.setAbortCallback(null);
        String after = session.generate(TURNS, greedy(24));
        assertFalse(after.trim().isEmpty(), "the session must be reusable straight after an abort");
    }

    /**
     * An exception from the token listener is captured, cancels the generation,
     * and is re-thrown from the call rather than unwinding through C. From the
     * moment it throws the caller's predicate is not consulted again.
     */
    @Test
    void aListenerExceptionSurfacesAfterTheCallAndSilencesThePredicate() {
        final IllegalStateException boom = new IllegalStateException("boom from the token listener");
        final AtomicInteger predicateCalls = new AtomicInteger();
        final AtomicInteger callsAtThrow = new AtomicInteger(-1);
        final AtomicInteger deliveries = new AtomicInteger();

        session.setAbortCallback(() -> {
            predicateCalls.incrementAndGet();
            return true;
        });

        IllegalStateException thrown = assertThrows(IllegalStateException.class,
                () -> session.generateStream(LONG_TURNS, greedy(120), chunk -> {
                    deliveries.incrementAndGet();
                    callsAtThrow.set(predicateCalls.get());
                    throw boom;
                }));

        assertSame(boom, thrown, "the listener's own exception must come back unwrapped");
        // The listener is offered exactly one chunk. That is the joint effect
        // of two guards — the abort hook answers "stop" from the moment the
        // listener threw, and the delivery path checks the failure slot too —
        // so this count alone attributes the stop to neither. The predicate
        // count below is what pins the abort hook's half; the delivery guard
        // is pinned in CallStateTest, which can offer a chunk after a failure
        // as C never does.
        assertEquals(1, deliveries.get(), "the listener was offered exactly one chunk");
        assertTrue(predicateCalls.get() > 0, "the predicate ran before the failure");
        assertEquals(callsAtThrow.get(), predicateCalls.get(),
                "the predicate was consulted again after the listener failed");
    }

    /** The same, for an exception thrown by the predicate itself. */
    @Test
    void aPredicateExceptionSurfacesAfterTheCall() {
        final IllegalStateException boom = new IllegalStateException("boom from the predicate");
        session.setAbortCallback(() -> {
            throw boom;
        });
        IllegalStateException thrown = assertThrows(IllegalStateException.class,
                () -> session.generate(TURNS, greedy(32)));
        assertSame(boom, thrown);
    }

    /**
     * The UTF-8 case, both sides.
     *
     * <p>Before: the raw C callback is handed the detokenizer's bytes, and this
     * prompt makes Gemma 3 fall back to BYTE tokens, so characters arrive split.
     * Decoding each chunk on its own corrupts them.
     *
     * <p>After: the same observed byte chunks, run through the binding's
     * assembler, come back whole — and the high-level streaming API delivers no
     * replacement characters at all.
     */
    @Test
    void splitCharactersAreReassembled() {
        List<ChatMessage> prompt = Collections.singletonList(ChatMessage.user(
                "Reply with exactly this and nothing else: " + RARE));

        List<byte[]> raw = rawChunkBytes(prompt, 40);
        assertFalse(raw.isEmpty(), "the raw stream delivered nothing");

        int fragments = 0;
        StringBuilder naive = new StringBuilder();
        int totalBytes = 0;
        for (byte[] b : raw) {
            totalBytes += b.length;
            String piece = new String(b, StandardCharsets.UTF_8);
            naive.append(piece);
            if (piece.indexOf('\uFFFD') >= 0) {
                fragments++;
            }
        }
        assertTrue(fragments > 0,
                "expected byte-fallback chunks that split a character; got " + raw.size()
                        + " chunks, " + totalBytes + " bytes, none of them a fragment");

        byte[] all = new byte[totalBytes];
        int at = 0;
        for (byte[] b : raw) {
            System.arraycopy(b, 0, all, at, b.length);
            at += b.length;
        }
        String correct = new String(all, StandardCharsets.UTF_8);
        assertTrue(correct.contains(RARE), "the model did not echo the characters: " + correct);
        assertFalse(naive.toString().equals(correct),
                "per-chunk decoding happened to be lossless, so this case proves nothing");

        Utf8ChunkAssembler assembler = new Utf8ChunkAssembler();
        StringBuilder repaired = new StringBuilder();
        for (byte[] b : raw) {
            repaired.append(assembler.take(b));
        }
        repaired.append(assembler.flush());
        assertEquals(correct, repaired.toString(), "the assembler must recover the whole text");

        // And the high-level API delivers the same, with no replacement chars.
        session.reset();
        final StringBuilder streamed = new StringBuilder();
        session.generateStream(prompt, greedy(40), streamed::append);
        assertTrue(streamed.indexOf("\uFFFD") < 0,
                "the streaming API leaked a replacement character: " + streamed);
        assertTrue(streamed.toString().contains(RARE), streamed.toString());
    }

    // ----- interior NUL through the real call paths -----

    @Test
    void generateRejectsAnInteriorNulNamingTheField() {
        IllegalArgumentException role = assertThrows(IllegalArgumentException.class,
                () -> session.generate(Collections.singletonList(
                        new ChatMessage("us\0er", "hi")), greedy(8)));
        assertTrue(role.getMessage().contains("messages[0].role"), role.getMessage());

        IllegalArgumentException content = assertThrows(IllegalArgumentException.class,
                () -> session.generate(Arrays.asList(
                        ChatMessage.user("fine"),
                        ChatMessage.user("bro\0ken")), greedy(8)));
        assertTrue(content.getMessage().contains("messages[1].content"), content.getMessage());

        IllegalArgumentException stop = assertThrows(IllegalArgumentException.class,
                () -> session.generate(TURNS, greedy(8).stop("ok", "ba\0d")));
        assertTrue(stop.getMessage().contains("stop[1]"), stop.getMessage());

        IllegalArgumentException count = assertThrows(IllegalArgumentException.class,
                () -> session.countTokens(Collections.singletonList(
                        new ChatMessage("user", "x\0y"))));
        assertTrue(count.getMessage().contains("messages[0].content"), count.getMessage());

        // The session survived every rejection: nothing reached C.
        assertFalse(session.generate(TURNS, greedy(16)).trim().isEmpty());
    }

    // ----- stop sequences and prefill_only, against the shared literals -----

    @Test
    void aStopSequenceTruncatesBeforeTheMatch() {
        Assumptions.assumeTrue(pinnedStopBaseline,
                "gated model is not the one the stop-sequence literals pin");
        countingSession.reset();
        String full = countingSession.generate(COUNTING_TURNS, greedy(64));
        // Without this the case is vacuous: a reply that never reached the stop
        // string would pass whether or not stop sequences work at all.
        assertTrue(full.contains("5"), "the unstopped reply must contain the stop string: " + full);

        countingSession.reset();
        String stopped = countingSession.generate(COUNTING_TURNS, greedy(64).stop("5"));
        assertFalse(stopped.contains("5"), "the matched text must not reach the caller: " + stopped);
        assertTrue(full.startsWith(stopped),
                "the stopped reply must be a prefix of the unstopped one: " + stopped);
        assertEquals("1\n2\n3\n4\n", stopped);
    }

    /**
     * Two stop sequences, passed in both orders. "4" is generated before "7",
     * so "4" wins either way: the earliest match in the output decides, not the
     * position in the array. Order-independence is also what says the whole
     * array was marshalled, rather than only its first element.
     */
    @Test
    void stopSequencesStopAtTheEarliestMatchInTheOutput() {
        Assumptions.assumeTrue(pinnedStopBaseline,
                "gated model is not the one the stop-sequence literals pin");
        countingSession.reset();
        String full = countingSession.generate(COUNTING_TURNS, greedy(64));
        assertTrue(full.contains("4") && full.contains("7"),
                "the unstopped reply must contain both stop strings: " + full);

        for (String[] stop : new String[][] { { "7", "4" }, { "4", "7" } }) {
            countingSession.reset();
            String stopped = countingSession.generate(COUNTING_TURNS, greedy(64).stop(stop));
            assertEquals(COUNTING_STOPPED_AT_FOUR, stopped, "stop list " + Arrays.toString(stop));
            assertFalse(stopped.contains("4") || stopped.contains("7"),
                    "neither match may reach the caller: " + stopped);
        }
    }

    @Test
    void anEmptyStopListIsTheSameAsNoStopList() {
        countingSession.reset();
        String none = countingSession.generate(COUNTING_TURNS, greedy(64));
        countingSession.reset();
        // An empty argument list would bind to the GETTER, so the empty stop
        // list has to be spelled as an empty array.
        String empty = countingSession.generate(COUNTING_TURNS, greedy(64).stop(new String[0]));

        assertEquals(none, empty, "an empty stop list must not truncate");
        // The reply is long enough that a stop list WOULD have truncated it, so
        // the equality above is not two empty strings agreeing.
        assertTrue(none.contains("4"),
                "the reply must be long enough for a stop list to bite: " + none);
    }

    @Test
    void prefillOnlySuppressesGeneration() {
        countingSession.reset();
        assertEquals("", countingSession.generate(COUNTING_TURNS, greedy(64).prefillOnly(true)),
                "prefillOnly must produce no assistant text");

        countingSession.reset();
        final AtomicInteger chunks = new AtomicInteger();
        countingSession.generateStream(COUNTING_TURNS, greedy(64).prefillOnly(true),
                chunk -> chunks.incrementAndGet());
        assertEquals(0, chunks.get(), "prefillOnly must emit no token chunks");

        // Positive control: the same messages and sampler settings, minus the
        // flag, do produce text — so the two emptinesses above are the flag's
        // doing and not a prompt that generates nothing.
        countingSession.reset();
        assertFalse(countingSession.generate(COUNTING_TURNS, greedy(64)).isEmpty(),
                "the control produced nothing, so this case proves nothing");
    }

    /**
     * The one place the streamed and one-shot paths disagree. C hands each
     * piece to the callback before it scans for a stop match, so the streamed
     * text carries the matched piece the one-shot return value has cut off.
     */
    @Test
    void theStreamDeliversTheChunkTheOneShotPathTruncates() {
        Assumptions.assumeTrue(pinnedStopBaseline,
                "gated model is not the one the stop-sequence literals pin");
        countingSession.reset();
        String oneShot = countingSession.generate(COUNTING_TURNS, greedy(64).stop("7", "4"));

        countingSession.reset();
        final StringBuilder streamed = new StringBuilder();
        countingSession.generateStream(COUNTING_TURNS, greedy(64).stop("7", "4"), streamed::append);

        assertEquals(COUNTING_STOPPED_AT_FOUR, oneShot);
        assertEquals("1\n2\n3\n4", streamed.toString());
        assertTrue(streamed.toString().startsWith(oneShot),
                "the streamed text must extend the one-shot text: " + streamed);
    }

    @Test
    void usingAClosedSessionFailsCleanly() {
        ChatSession s = ChatSession.open(modelPath, new ChatOpenParams().nCtx(256));
        s.close();
        s.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> s.nCtx());
        assertThrows(IllegalStateException.class, () -> s.generate(TURNS, greedy(4)));
    }

    // ----- helpers -----

    private static void assertArrayEqualsBytes(String expected, String actual) {
        org.junit.jupiter.api.Assertions.assertArrayEquals(
                expected.getBytes(StandardCharsets.UTF_8),
                actual.getBytes(StandardCharsets.UTF_8),
                "streamed and one-shot must agree byte for byte");
    }

    /**
     * Run one generation straight against the C ABI, recording the exact bytes
     * of every {@code crispasr_chat_on_token} callback. The wrapper is
     * deliberately bypassed: the point is to see what C actually delivers.
     */
    private static List<byte[]> rawChunkBytes(List<ChatMessage> messages, int maxTokens) {
        final List<byte[]> chunks = new ArrayList<byte[]>();
        List<Memory> owned = new ArrayList<Memory>();
        ChatLib.ChatError err = new ChatLib.ChatError();

        ChatLib.COpenParams open = new ChatLib.COpenParams();
        ChatLib.INSTANCE.crispasr_chat_open_params_default(open);
        open.n_ctx = 1024;
        Pointer handle = ChatLib.INSTANCE.crispasr_chat_open(
                ChatNative.cstring(modelPath, "modelPath", owned), open, err);
        assertNotNull(handle, "raw crispasr_chat_open failed: " + err.messageString());
        try {
            ChatLib.CMessage[] cmsgs =
                    (ChatLib.CMessage[]) new ChatLib.CMessage().toArray(messages.size());
            for (int i = 0; i < messages.size(); i++) {
                cmsgs[i].role = ChatNative.cstring(messages.get(i).role(), "role", owned);
                cmsgs[i].content = ChatNative.cstring(messages.get(i).content(), "content", owned);
            }
            ChatLib.CGenerateParams gp = new ChatLib.CGenerateParams();
            ChatLib.INSTANCE.crispasr_chat_generate_params_default(gp);
            gp.max_tokens = maxTokens;
            gp.temperature = 0f;

            ChatLib.OnTokenCallback cb = new ChatLib.OnTokenCallback() {
                @Override
                public void invoke(Pointer utf8Chunk, Pointer user) {
                    chunks.add(ChatNative.readBytes(utf8Chunk));
                }
            };
            int rc = ChatLib.INSTANCE.crispasr_chat_generate_stream(handle, cmsgs,
                    new ChatLib.SizeT(messages.size()), gp, cb, Pointer.NULL, err);
            assertEquals(0, rc, "raw crispasr_chat_generate_stream: " + err.messageString());
            owned.clear();
        } finally {
            ChatLib.INSTANCE.crispasr_chat_close(handle);
        }
        return chunks;
    }
}
