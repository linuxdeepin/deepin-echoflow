package io.github.ggerganov.whispercpp.chat;

/**
 * A registered abort predicate stopped the generation — a cancellation, not a
 * fault. Catch this rather than {@link ChatException} to tell the two apart.
 *
 * <p>The C side flushes the session back to its just-opened state on an abort,
 * so a session that threw this needs no {@link ChatSession#reset()} before its
 * next use.
 */
public final class ChatAbortedException extends ChatException {

    private static final long serialVersionUID = 1L;

    /**
     * @param message the C diagnostic
     * @param code    always {@code CRISPASR_CHAT_ERR_ABORTED}
     */
    public ChatAbortedException(String message, int code) {
        super(message, code);
    }
}
