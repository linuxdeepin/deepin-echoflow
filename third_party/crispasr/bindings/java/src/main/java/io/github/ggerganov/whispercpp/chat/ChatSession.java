package io.github.ggerganov.whispercpp.chat;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;

import java.util.ArrayList;
import java.util.List;

/**
 * An open chat model plus its KV cache — the wrapper over the
 * {@code crispasr_chat_*} C ABI.
 *
 * <pre>{@code
 * List<ChatMessage> turns = Arrays.asList(
 *         ChatMessage.system("You are terse."),
 *         ChatMessage.user("Name three primes."));
 * try (ChatSession s = ChatSession.open("/models/gemma-3-1b-it-Q4_K_M.gguf")) {
 *     System.out.println(s.generate(turns, new ChatGenerateParams().maxTokens(64)));
 * }
 * }</pre>
 *
 * <p>One call at a time per session: the C side serialises its own context with
 * a mutex and this wrapper serialises the calls that register callbacks on it,
 * so a second concurrent call blocks rather than racing.
 *
 * <p>The KV cache persists across generate calls, and reuse depends on passing
 * the WHOLE conversation every time: the session compares the templated prompt
 * against the tokens it already holds and decodes only what is new. Passing
 * just the latest turn is not wrong — it simply shares no prefix, so every call
 * re-prefills from scratch.
 *
 * <p>EU AI Act Art. 50: this is open-ended synthetic text generation, and
 * nothing here marks it as machine-generated. A product that puts it in front
 * of a natural person owes them a visible "you are talking to an AI" notice —
 * {@link #aiDisclosureText()} is the canonical wording — and owes the output a
 * machine-readable marking that travels with it. See the header and
 * {@code docs/eu-ai-act.md} §6.6.
 */
public final class ChatSession implements AutoCloseable {

    private final Object lock = new Object();

    private Pointer handle;

    /** Read on the generating thread, written by any thread. */
    private volatile ContinuePredicate predicate;

    /**
     * Strong references keeping the JNA callbacks alive for exactly the length
     * of one native call. JNA frees a callback's native trampoline once the
     * Java object becomes unreachable, and a local variable is not proof
     * against that while the call is still running, so the reference lives in a
     * field of this session — which is unambiguously reachable — and is cleared
     * in the call's {@code finally}.
     */
    private ChatLib.OnTokenCallback liveTokenCallback;
    private ChatLib.AbortCallback liveAbortCallback;

    private ChatSession(Pointer handle) {
        this.handle = handle;
    }

    /**
     * Open a session over a GGUF chat model on disk, taking the ABI defaults.
     *
     * @param modelPath path to the GGUF
     * @return the open session
     * @throws ChatException            if the model cannot be loaded
     * @throws IllegalArgumentException if {@code modelPath} holds an interior NUL
     */
    public static ChatSession open(String modelPath) {
        return open(modelPath, null);
    }

    /**
     * Open a session over a GGUF chat model on disk.
     *
     * @param modelPath path to the GGUF
     * @param params    per-session options, or null for the ABI defaults
     * @return the open session
     * @throws ChatException            if the model cannot be loaded
     * @throws IllegalArgumentException if the path or chat template holds an
     *                                  interior NUL
     */
    public static ChatSession open(String modelPath, ChatOpenParams params) {
        if (modelPath == null) {
            throw new IllegalArgumentException("crispasr_chat: modelPath must not be null");
        }
        List<Memory> owned = new ArrayList<Memory>();
        Pointer path = ChatNative.cstring(modelPath, "modelPath", owned);
        ChatLib.COpenParams cp = (params == null ? new ChatOpenParams() : params).toNative(owned);
        ChatLib.ChatError err = new ChatLib.ChatError();
        Pointer h;
        try {
            h = ChatLib.INSTANCE.crispasr_chat_open(path, cp, err);
        } finally {
            owned.clear();
        }
        if (h == null) {
            throw ChatNative.error("crispasr_chat_open: failed to open " + modelPath, err, 0);
        }
        return new ChatSession(h);
    }

    /**
     * Conservative working set in bytes (weights + KV cache + activations) for a
     * GGUF chat model on disk, reading its metadata but never its tensor data —
     * a pre-flight guard for low-memory devices. {@code params} matters mostly
     * for {@code nCtx}, which sizes the KV term linearly; pass null and the
     * model's own trained context is used.
     *
     * <p>The number is deliberately high, not approximate. The KV term bills
     * both the K and the V cache at the full attention width {@code n_embd}, but
     * a grouped-query model gives each layer a K/V width that is a fraction of
     * that: on Gemma 3 1B the KV term comes out 4.50× llama.cpp's real cache
     * (117.00 MiB against 26.00 MiB at {@code nCtx} 1024), which is 1.33× on the
     * whole estimate at {@code nCtx} 4096. Over-reporting is the safe direction
     * for a "will this fit?" guard: it can turn away a model that would just
     * have fitted, and never admits one that would not.
     *
     * @param modelPath path to the GGUF
     * @param params    per-session options, or null for the ABI defaults
     * @return the estimate in bytes
     * @throws ChatException            if the estimate could not be made
     * @throws IllegalArgumentException if the path or chat template holds an
     *                                  interior NUL
     */
    public static long memoryEstimate(String modelPath, ChatOpenParams params) {
        if (modelPath == null) {
            throw new IllegalArgumentException("crispasr_chat: modelPath must not be null");
        }
        List<Memory> owned = new ArrayList<Memory>();
        Pointer path = ChatNative.cstring(modelPath, "modelPath", owned);
        ChatLib.COpenParams cp = (params == null ? new ChatOpenParams() : params).toNative(owned);
        ChatLib.ChatError err = new ChatLib.ChatError();
        long n;
        try {
            n = ChatLib.INSTANCE.crispasr_chat_memory_estimate(path, cp, err).longValue();
        } finally {
            owned.clear();
        }
        if (n == 0) {
            throw ChatNative.error("crispasr_chat_memory_estimate: could not estimate " + modelPath,
                    err, 0);
        }
        return n;
    }

    /**
     * The canonical "you are talking to an AI" wording (EU AI Act Art. 50(1)).
     * Show it visibly at or before the first turn of any conversational product
     * built on this binding.
     *
     * @return the disclosure text
     */
    public static String aiDisclosureText() {
        String s = ChatLib.INSTANCE.crispasr_chat_ai_disclosure_text();
        return s == null ? "" : s;
    }

    /**
     * Free the session and its KV cache. Safe to call more than once.
     *
     * <p>Safe from another thread while a generation runs, and it WAITS for
     * that generation rather than cutting it off. Two things cover the handle
     * between them: the monitor below, for the window between reading
     * {@code handle} and entering C, which C cannot see into; and
     * {@code crispasr_chat_close} itself, which counts the calls already
     * inside the session and waits for them. A generation holds the session
     * for as long as it decodes, so register a {@link ContinuePredicate}
     * first if the length of the wait matters.
     */
    @Override
    public void close() {
        synchronized (lock) {
            if (handle != null) {
                ChatLib.INSTANCE.crispasr_chat_close(handle);
                handle = null;
            }
        }
    }

    /**
     * Clear the KV cache so the next generation re-prefills from scratch. Call
     * it when starting a new conversation in a reused session.
     *
     * <p>An abort does this for you: a session that threw
     * {@link ChatAbortedException} needs no reset before its next use.
     *
     * @throws ChatException         if the C side reports a failure
     * @throws IllegalStateException if the session is closed
     */
    public void reset() {
        synchronized (lock) {
            Pointer h = requireOpen("crispasr_chat_reset");
            ChatLib.ChatError err = new ChatLib.ChatError();
            int rc = ChatLib.INSTANCE.crispasr_chat_reset(h, err);
            if (rc != 0) {
                throw ChatNative.error("crispasr_chat_reset failed", err, rc);
            }
        }
    }

    /**
     * @return the session's context window in tokens
     * @throws IllegalStateException if the session is closed
     */
    public int nCtx() {
        synchronized (lock) {
            return ChatLib.INSTANCE.crispasr_chat_n_ctx(requireOpen("crispasr_chat_n_ctx"));
        }
    }

    /**
     * @return the name of the chat template the session resolved against, e.g.
     *         "chatml", "llama3", "gemma"
     * @throws IllegalStateException if the session is closed
     */
    public String templateName() {
        synchronized (lock) {
            String s = ChatLib.INSTANCE.crispasr_chat_template_name(
                    requireOpen("crispasr_chat_template_name"));
            return s == null ? "" : s;
        }
    }

    /**
     * Register a predicate that can cancel a generation, or clear it with null.
     *
     * <p>Polarity: the predicate returns TRUE to LET THE GENERATION CONTINUE
     * and false to abort it — the polarity of the C callback it is handed to,
     * forwarded unchanged. See {@link ContinuePredicate} for the rest of the
     * contract, including the ban on calling back into the same session.
     *
     * <p>The predicate is handed to the C session for the length of each
     * generate call and taken back when the call returns, so it is only ever
     * live on the thread running the generation. Register it before starting
     * one and have it read your own flag.
     *
     * @param shouldContinue the predicate, or null to clear
     */
    public void setAbortCallback(ContinuePredicate shouldContinue) {
        this.predicate = shouldContinue;
    }

    /**
     * Tokens the model's own tokenizer produces for {@code messages} once the
     * session's chat template has been applied — the prompt length a FRESH
     * session prefills, so it compares straight against {@link #nCtx()}.
     *
     * <p>The count covers the whole prompt: the template's control tokens, the
     * leading BOS and the trailing generation prompt that opens the assistant
     * turn. For a session part-way through a conversation it is an upper
     * bound, since that session re-decodes only the suffix its history does
     * not already hold.
     *
     * <p>An EMPTY message list counts the template's own opening, which is
     * whatever that template emits for no messages — template-dependent, and
     * possibly nothing at all: several chat templates write only from inside
     * their loop over the messages, and those return 0. Do not read a
     * positive overhead into it.
     *
     * <p>A pure query: it neither touches the KV cache nor extends the history,
     * so it can be called freely between generations.
     *
     * @param messages the conversation
     * @return the prompt length in tokens
     * @throws ChatException            if the C side reports a failure
     * @throws IllegalStateException    if the session is closed
     * @throws IllegalArgumentException if a role or content holds an interior NUL
     */
    public int countTokens(List<ChatMessage> messages) {
        synchronized (lock) {
            Pointer h = requireOpen("crispasr_chat_count_tokens");
            List<Memory> owned = new ArrayList<Memory>();
            ChatLib.CMessage[] cmsgs = toNative(messages, owned);
            ChatLib.ChatError err = new ChatLib.ChatError();
            int n;
            try {
                n = ChatLib.INSTANCE.crispasr_chat_count_tokens(h, cmsgs, size(messages), err);
            } finally {
                owned.clear();
            }
            if (n < 0) {
                // The negative return is a failure SENTINEL, not an error code,
                // so it is not offered to the classifier as one: the struct is
                // the only carrier here.
                throw ChatNative.error("crispasr_chat_count_tokens failed", err, 0);
            }
            return n;
        }
    }

    /**
     * Apply the chat template to {@code messages}, prefill, and generate the
     * assistant's reply, taking the ABI's default sampler options.
     *
     * @param messages the WHOLE conversation
     * @return the reply
     * @throws ChatException         if the C side reports a failure
     * @throws ChatAbortedException  if a registered predicate cancelled the run
     * @throws IllegalStateException if the session is closed
     */
    public String generate(List<ChatMessage> messages) {
        return generate(messages, null);
    }

    /**
     * Apply the chat template to {@code messages}, prefill, and generate the
     * assistant's reply.
     *
     * <p>Pass the WHOLE conversation on every call, not just the new turn — see
     * the class javadoc for why.
     *
     * <p>On cancellation this throws {@link ChatAbortedException} and the
     * session is left as if freshly opened; no {@link #reset()} is needed.
     *
     * @param messages the WHOLE conversation
     * @param params   sampler options, or null for the ABI defaults
     * @return the reply
     * @throws ChatException            if the C side reports a failure
     * @throws ChatAbortedException     if a registered predicate cancelled the run
     * @throws IllegalStateException    if the session is closed
     * @throws IllegalArgumentException if a role, content or stop sequence holds
     *                                  an interior NUL
     */
    public String generate(List<ChatMessage> messages, ChatGenerateParams params) {
        synchronized (lock) {
            Pointer h = requireOpen("crispasr_chat_generate");
            List<Memory> owned = new ArrayList<Memory>();
            ChatLib.CMessage[] cmsgs = toNative(messages, owned);
            ChatLib.CGenerateParams cp =
                    (params == null ? new ChatGenerateParams() : params).toNative(owned);

            CallState state = new CallState(null, predicate);
            ChatLib.ChatError err = new ChatLib.ChatError();
            Pointer out;
            try {
                arm(h, state);
                out = ChatLib.INSTANCE.crispasr_chat_generate(h, cmsgs, size(messages), cp, err);
            } finally {
                disarm(h);
                owned.clear();
            }

            String text = null;
            if (out != null) {
                try {
                    text = ChatNative.readString(out);
                } finally {
                    ChatLib.INSTANCE.crispasr_chat_string_free(out);
                }
            }
            // A callback failure outranks whatever the native call reported,
            // and is raised only now that no C frame is left on the stack.
            state.rethrow();
            if (text == null) {
                throw ChatNative.error("crispasr_chat_generate failed", err, 0);
            }
            return text;
        }
    }

    /**
     * {@link #generate(List, ChatGenerateParams)} with the reply delivered
     * chunk by chunk as it is decoded. Concatenating the chunks gives the same
     * text {@code generate} returns for the same messages and params — except
     * when a stop sequence ends the generation. The C side hands each piece to
     * the callback before it scans for a match, so the chunk the match lands in
     * has already been delivered, while the one-shot return value is truncated
     * before the match. With {@link ChatGenerateParams#stop(String...)} in play
     * the streamed text is therefore the one-shot text plus that last chunk,
     * and a caller who wants the truncated form has to cut it back themselves.
     *
     * <p>{@code onToken} runs on the generating thread while the session lock
     * is held, so it must NOT call back into the same session — that deadlocks.
     * An exception from it cancels the generation, suppresses the remaining
     * chunks, and is re-thrown here once the native call has returned; it never
     * unwinds through C. From the moment it throws, a registered abort
     * predicate is no longer consulted and the answer to C is "stop".
     *
     * @param messages the WHOLE conversation
     * @param params   sampler options, or null for the ABI defaults
     * @param onToken  the listener, or null to run the generation with no
     *                 streaming delivery
     * @throws ChatException            if the C side reports a failure
     * @throws ChatAbortedException     if a registered predicate cancelled the run
     * @throws IllegalStateException    if the session is closed
     * @throws IllegalArgumentException if a role, content or stop sequence holds
     *                                  an interior NUL
     */
    public void generateStream(List<ChatMessage> messages, ChatGenerateParams params,
            TokenListener onToken) {
        synchronized (lock) {
            Pointer h = requireOpen("crispasr_chat_generate_stream");
            List<Memory> owned = new ArrayList<Memory>();
            ChatLib.CMessage[] cmsgs = toNative(messages, owned);
            ChatLib.CGenerateParams cp =
                    (params == null ? new ChatGenerateParams() : params).toNative(owned);

            final CallState state = new CallState(onToken, predicate);
            ChatLib.OnTokenCallback tokenCallback = new ChatLib.OnTokenCallback() {
                @Override
                public void invoke(Pointer utf8Chunk, Pointer user) {
                    state.deliver(utf8Chunk);
                }
            };
            ChatLib.ChatError err = new ChatLib.ChatError();
            int rc;
            liveTokenCallback = tokenCallback;
            try {
                arm(h, state);
                rc = ChatLib.INSTANCE.crispasr_chat_generate_stream(h, cmsgs, size(messages), cp,
                        tokenCallback, Pointer.NULL, err);
            } finally {
                disarm(h);
                liveTokenCallback = null;
                owned.clear();
            }

            // The tail of a character the generation stopped inside is handed
            // over rather than dropped, now that no C frame is on the stack.
            state.flush();
            state.rethrow();
            if (rc != 0) {
                throw ChatNative.error("crispasr_chat_generate_stream failed", err, rc);
            }
        }
    }

    // ----- private -----

    /**
     * Register the abort trampoline for this call.
     *
     * <p>It is registered whenever the caller has a predicate OR the call has a
     * token listener. The second case is what makes a listener failure cancel
     * the generation even when the caller registered no predicate of their own:
     * without a hook there is no way to ask the ABI to stop.
     */
    private void arm(Pointer h, final CallState state) {
        if (!state.needsAbortHook()) {
            return;
        }
        ChatLib.AbortCallback cb = new ChatLib.AbortCallback() {
            @Override
            public byte invoke(Pointer user) {
                return state.keepGoing();
            }
        };
        liveAbortCallback = cb;
        ChatLib.INSTANCE.crispasr_chat_set_abort_callback(h, cb, Pointer.NULL);
    }

    private void disarm(Pointer h) {
        if (liveAbortCallback != null) {
            ChatLib.INSTANCE.crispasr_chat_set_abort_callback(h, null, Pointer.NULL);
            liveAbortCallback = null;
        }
    }

    private Pointer requireOpen(String what) {
        if (handle == null) {
            throw new IllegalStateException(what + ": session is closed");
        }
        return handle;
    }

    private static ChatLib.SizeT size(List<ChatMessage> messages) {
        return new ChatLib.SizeT(messages == null ? 0 : messages.size());
    }

    /**
     * Copy the messages into one contiguous native array. Returns null for an
     * empty list, which is the NULL the ABI expects alongside a count of zero.
     */
    private static ChatLib.CMessage[] toNative(List<ChatMessage> messages, List<Memory> owned) {
        if (messages == null || messages.isEmpty()) {
            return null;
        }
        ChatLib.CMessage[] out =
                (ChatLib.CMessage[]) new ChatLib.CMessage().toArray(messages.size());
        for (int i = 0; i < messages.size(); i++) {
            ChatMessage m = messages.get(i);
            if (m == null) {
                throw new IllegalArgumentException("crispasr_chat: messages[" + i + "] is null");
            }
            out[i].role = ChatNative.cstring(m.role(), "messages[" + i + "].role", owned);
            out[i].content = ChatNative.cstring(m.content(), "messages[" + i + "].content", owned);
        }
        return out;
    }

    /**
     * The state of one native generate call: where chunks go, what answers the
     * abort hook, and the slot for an exception a callback raised.
     *
     * <p>No exception may unwind through C, so one is caught here, the
     * generation is stopped, and it is re-thrown once the native call has
     * returned. The catch covers the WHOLE delivery — reading the native bytes
     * and reassembling the characters as well as the listener call itself.
     * Anything thrown outside it escapes into JNA, which prints it, tells the
     * caller nothing, and lets C carry on: chunks would be dropped while the
     * call reported success.
     *
     * <p>Package-private, not private, so a test can drive {@link #deliver} and
     * {@link #flush} with chunk shapes the C side does not itself produce.
     */
    static final class CallState {

        private final TokenListener listener;
        private final ContinuePredicate predicate;
        private final Utf8ChunkAssembler assembler = new Utf8ChunkAssembler();

        private Throwable failure;

        CallState(TokenListener listener, ContinuePredicate predicate) {
            this.listener = listener;
            this.predicate = predicate;
        }

        boolean needsAbortHook() {
            return predicate != null || listener != null;
        }

        /**
         * Hand one native chunk on, as far as it completes whole characters.
         *
         * <p>The abort hook answers "stop" from the moment the listener throws
         * and C asks it before offering another chunk, so the failure check
         * here does not fire during a real generation; it is what makes the
         * rule hold anyway for a C side that offers one.
         *
         * @param utf8Chunk the NUL-terminated bytes C delivered
         */
        void deliver(Pointer utf8Chunk) {
            if (failure != null || listener == null) {
                return;
            }
            try {
                emit(assembler.take(ChatNative.readBytes(utf8Chunk)));
            } catch (Throwable t) {
                capture(t);
            }
        }

        /**
         * Hand over the tail of a character the generation stopped inside.
         *
         * <p>The failure check here is not ruled out the way {@link #deliver}'s
         * is: a chunk that completes one character and opens another leaves a
         * tail, and a listener that throws on the completed part reaches this
         * line.
         */
        void flush() {
            if (failure != null || listener == null) {
                return;
            }
            try {
                emit(assembler.flush());
            } catch (Throwable t) {
                capture(t);
            }
        }

        private void emit(String text) {
            if (!text.isEmpty()) {
                listener.onToken(text);
            }
        }

        /**
         * Answer C's "may I continue?" in C's own polarity: 1 continues, 0
         * aborts.
         */
        byte keepGoing() {
            if (failure != null) {
                // The listener already threw: nothing is reading the output any
                // more, so stop WITHOUT asking the caller's predicate again.
                return 0;
            }
            if (predicate == null) {
                return 1;
            }
            try {
                return (byte) (predicate.shouldContinue() ? 1 : 0);
            } catch (Throwable t) {
                // A predicate that threw cannot be trusted to answer again, and
                // finishing the generation would waste the work anyway.
                capture(t);
                return 0;
            }
        }

        private void capture(Throwable t) {
            if (failure == null) {
                failure = t;
            }
        }

        void rethrow() {
            Throwable t = failure;
            if (t == null) {
                return;
            }
            failure = null;
            if (t instanceof RuntimeException) {
                throw (RuntimeException) t;
            }
            if (t instanceof Error) {
                throw (Error) t;
            }
            throw new IllegalStateException("crispasr_chat: callback threw", t);
        }
    }
}
