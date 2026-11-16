package io.github.ggerganov.whispercpp.chat;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;

import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Marshalling helpers shared by the wrapper classes. Package-private: this is
 * where Java strings become the borrowed {@code const char*}s the ABI reads.
 */
final class ChatNative {

    private ChatNative() {
    }

    /**
     * Reject a string C cannot carry whole.
     *
     * <p>C reads a NUL as the end of the string, so passing one through would
     * silently drop the rest of a message, stop sequence or chat template —
     * the caller would get a truncated prompt and no indication why. Rust's
     * {@code CString::new} rejects the same input.
     *
     * @param s     the candidate
     * @param field where it came from, named in the exception
     * @throws IllegalArgumentException if {@code s} holds an interior NUL
     */
    static void requireNoInteriorNul(String s, String field) {
        if (s != null && s.indexOf('\0') >= 0) {
            throw new IllegalArgumentException("crispasr_chat: " + field
                    + " contains an interior NUL byte, which C cannot carry");
        }
    }

    /**
     * Copy {@code s} into native memory as NUL-terminated UTF-8.
     *
     * @param s     the string, checked for an interior NUL first
     * @param field where it came from, named in the exception
     * @param owned collects the allocations; the caller holds this list for the
     *              length of the native call so nothing is freed under C
     * @return a pointer valid until {@code owned} becomes unreachable
     * @throws IllegalArgumentException if {@code s} holds an interior NUL
     */
    static Pointer cstring(String s, String field, List<Memory> owned) {
        requireNoInteriorNul(s, field);
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        Memory m = new Memory(bytes.length + 1L);
        m.write(0, bytes, 0, bytes.length);
        m.setByte(bytes.length, (byte) 0);
        owned.add(m);
        return m;
    }

    /**
     * Read a NUL-terminated UTF-8 string out of native memory.
     *
     * @param p the pointer, may be null
     * @return the decoded string, empty when {@code p} is null
     */
    static String readString(Pointer p) {
        if (p == null) {
            return "";
        }
        byte[] bytes = readBytes(p);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * Read the raw bytes of a NUL-terminated buffer, without decoding them —
     * the token callback needs the bytes, because a chunk can end part-way
     * through a character.
     *
     * @param p the pointer, may be null
     * @return the bytes before the NUL
     */
    static byte[] readBytes(Pointer p) {
        if (p == null) {
            return new byte[0];
        }
        int n = 0;
        while (p.getByte(n) != 0) {
            n++;
        }
        return n == 0 ? new byte[0] : p.getByteArray(0, n);
    }

    /**
     * Turn a filled-in C error struct into the exception to throw.
     *
     * <p>{@code codeHint} is the value the entry point itself returned, used
     * when {@code err.code} is zero. The one-shot path signals failure by
     * returning NULL, so there the struct is the only carrier and the hint is
     * 0; the streaming and reset paths also return the code.
     *
     * <p>Only {@code CRISPASR_CHAT_ERR_ABORTED} is switched on. The header says
     * every other value is a diagnostic aid rather than a contract, so they all
     * become a plain {@link ChatException} carrying the message.
     *
     * @param fallback message to use when C supplied none
     * @param err      the struct the call was given
     * @param codeHint the call's own return value, or 0 when it has none
     * @return the exception to throw
     */
    static ChatException error(String fallback, ChatLib.ChatError err, int codeHint) {
        String msg = err.messageString();
        if (msg.isEmpty()) {
            msg = fallback;
        }
        int code = err.code;
        if (code == 0) {
            code = codeHint;
        }
        if (code == ChatLib.CRISPASR_CHAT_ERR_ABORTED) {
            return new ChatAbortedException(msg, code);
        }
        return new ChatException(msg, code);
    }
}
