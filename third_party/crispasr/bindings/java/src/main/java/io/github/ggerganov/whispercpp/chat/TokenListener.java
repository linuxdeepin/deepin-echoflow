package io.github.ggerganov.whispercpp.chat;

/**
 * Receives the assistant's reply chunk by chunk as it is decoded.
 * Concatenating every chunk gives the same text
 * {@link ChatSession#generate(java.util.List, ChatGenerateParams)} returns for
 * the same messages and params — except when a stop sequence ends the
 * generation. The C side hands each piece to the callback before it scans for
 * a match, so the chunk the match lands in has already been delivered, while
 * the one-shot return value is truncated before the match. With
 * {@link ChatGenerateParams#stop(String...)} in play the streamed text is
 * therefore the one-shot text plus that last chunk, and a caller who wants the
 * truncated form has to cut it back themselves.
 *
 * <p>Runs on the generating thread while the session lock is held, so it must
 * NOT call back into the same session — that deadlocks. It must also be cheap:
 * it sits in the decode loop.
 *
 * <p>A chunk is always whole characters. The C ABI hands out the detokenizer's
 * raw bytes, and a byte-fallback token is one byte of a multi-byte character,
 * so a character can be split across several native callbacks; the binding
 * buffers the incomplete tail and delivers only what completes.
 *
 * <p>An exception thrown from here cancels the generation, is not allowed to
 * unwind through C, and is re-thrown from the {@code generateStream} call once
 * the native call has returned.
 */
public interface TokenListener {

    /**
     * @param chunk one or more whole characters of the reply; never empty
     */
    void onToken(String chunk);
}
