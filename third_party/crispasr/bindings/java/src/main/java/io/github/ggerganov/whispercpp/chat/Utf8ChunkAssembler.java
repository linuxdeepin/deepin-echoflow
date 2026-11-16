package io.github.ggerganov.whispercpp.chat;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Reassembles the byte chunks {@code crispasr_chat_on_token} delivers into
 * whole characters.
 *
 * <p>The C side hands the callback the raw bytes of the detokenized piece. A
 * model that falls back to byte tokens emits ONE BYTE per callback, so a
 * multi-byte character arrives split across two to four calls. Decoding each
 * chunk on its own turns every such character into replacement characters
 * irrecoverably, so the trailing incomplete sequence is held back here and
 * delivered once the bytes that finish it arrive.
 *
 * <p>The distinction the decoder draws is the one that matters: a sequence that
 * is merely UNFINISHED is kept for the next chunk, while one that can never
 * become valid however many bytes follow becomes a single replacement character
 * and decoding continues after it. {@link #flush()} covers the case where the
 * generation simply stopped mid-character — those bytes are handed over as
 * replacement characters rather than silently dropped.
 *
 * <p>Not thread-safe; one instance belongs to one generate call.
 */
final class Utf8ChunkAssembler {

    private static final byte[] NO_BYTES = new byte[0];

    private final CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE)
            .onUnmappableCharacter(CodingErrorAction.REPLACE);

    private byte[] pending = NO_BYTES;

    /**
     * Append one native chunk and return everything it completes.
     *
     * @param chunk the raw bytes of one callback
     * @return the whole characters now available, possibly empty
     */
    String take(byte[] chunk) {
        if (chunk == null || chunk.length == 0) {
            return "";
        }
        byte[] buf;
        if (pending.length == 0) {
            buf = chunk;
        } else {
            buf = Arrays.copyOf(pending, pending.length + chunk.length);
            System.arraycopy(chunk, 0, buf, pending.length, chunk.length);
        }
        ByteBuffer in = ByteBuffer.wrap(buf);
        CharBuffer out = CharBuffer.allocate(buf.length + 1);
        // endOfInput = false, so a truncated trailing sequence is left in `in`
        // rather than replaced. Everything genuinely malformed is replaced,
        // because that is the decoder's configured action.
        CoderResult r = decoder.decode(in, out, false);
        if (r.isOverflow()) {
            // Cannot happen: the output buffer holds one char per input byte
            // plus one, and no decoded form of n bytes exceeds n chars.
            throw new IllegalStateException("crispasr_chat: UTF-8 decode overflowed its buffer");
        }
        pending = in.hasRemaining() ? remaining(in) : NO_BYTES;
        out.flip();
        return out.toString();
    }

    /**
     * Deliver whatever is still buffered when the generation ends. Those bytes
     * are a character the generation stopped in the middle of, so they are
     * malformed on their own — hand them over as replacement characters rather
     * than drop output silently.
     *
     * @return the leftover, or empty when nothing was buffered
     */
    String flush() {
        if (pending.length == 0) {
            return "";
        }
        ByteBuffer in = ByteBuffer.wrap(pending);
        CharBuffer out = CharBuffer.allocate(pending.length + 1);
        decoder.decode(in, out, true);
        decoder.flush(out);
        decoder.reset();
        pending = NO_BYTES;
        out.flip();
        return out.toString();
    }

    /** @return true when a chunk boundary fell inside a character */
    boolean hasPending() {
        return pending.length != 0;
    }

    private static byte[] remaining(ByteBuffer in) {
        byte[] rest = new byte[in.remaining()];
        in.get(rest);
        return rest;
    }
}
