package io.github.ggerganov.whispercpp.chat;

/**
 * Asked repeatedly during a generation whether to keep going.
 *
 * <p>Polarity: return {@code true} to LET THE GENERATION CONTINUE and
 * {@code false} to abort it. That is the polarity of the C callback it is
 * handed to, {@code crispasr_chat_abort_callback}, and of the ASR side's
 * encoder-begin callback; this binding forwards the answer unchanged.
 *
 * <p>Called on the generating thread: once before each prompt batch during
 * prefill and once before each sampled token, and on the CPU backend also from
 * inside a running compute graph, so it can fire many times per batch. Keep it
 * cheap and non-blocking, and have it read a flag your own thread sets — a
 * {@code volatile} field or an {@code AtomicBoolean}, since the read happens on
 * another thread.
 *
 * <p>It must NOT call back into the same session: the session mutex is held for
 * the whole generation and re-entering deadlocks.
 *
 * <p>An exception thrown from here aborts the generation, is not allowed to
 * unwind through C, and is re-thrown from the generate call once the native
 * call has returned.
 */
public interface ContinuePredicate {

    /**
     * @return {@code true} to continue generating, {@code false} to abort
     */
    boolean shouldContinue();
}
