package io.github.ggerganov.whispercpp.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

/**
 * The reassembler on its own — no native library, no model, so these run
 * everywhere.
 *
 * <p>They pin the behaviour the C ABI forces on every binding: the token
 * callback is handed the detokenizer's raw bytes, and a byte-fallback token is
 * ONE BYTE of a multi-byte character, so a character arrives split across
 * several callbacks.
 *
 * <p>Every non-ASCII character is written as an escape. The Gradle build sets
 * no source encoding, so it takes the platform default, and the CI job for this
 * binding runs on Windows.
 */
class Utf8ChunkAssemblerTest {

    /** Goose, donkey, jellyfish — the rare code points the model-gated suite asks for. */
    private static final String RARE = "\uD83E\uDEBF\uD83E\uDECF\uD83E\uDEBC";

    /** U+FFFD, what a decoder emits for bytes it cannot make a character of. */
    private static final char REPLACEMENT = '\uFFFD';

    @Test
    void bytewiseDeliveryReassemblesIntact() {
        byte[] all = RARE.getBytes(StandardCharsets.UTF_8);
        assertEquals(12, all.length, "three astral code points are four bytes each");

        Utf8ChunkAssembler a = new Utf8ChunkAssembler();
        StringBuilder assembled = new StringBuilder();
        int emitted = 0;
        for (byte b : all) {
            String out = a.take(new byte[] { b });
            if (!out.isEmpty()) {
                emitted++;
            }
            assembled.append(out);
        }
        assertEquals(RARE, assembled.toString(), "byte-at-a-time delivery must round-trip");
        assertEquals(3, emitted, "one delivery per completed character, not per byte");
        assertFalse(a.hasPending(), "nothing left buffered");
        assertEquals("", a.flush());
    }

    /**
     * The failure the buffering exists to prevent: decoding each chunk on its
     * own turns every split character into replacement characters for good.
     */
    @Test
    void perChunkDecodingWouldCorrupt() {
        byte[] all = RARE.getBytes(StandardCharsets.UTF_8);
        StringBuilder naive = new StringBuilder();
        for (byte b : all) {
            naive.append(new String(new byte[] { b }, StandardCharsets.UTF_8));
        }
        assertNotEquals(RARE, naive.toString());
        assertTrue(naive.indexOf(String.valueOf(REPLACEMENT)) >= 0,
                "naive per-chunk decoding loses the characters");
    }

    @Test
    void multiByteChunksSplitAtArbitraryBoundaries() {
        // "a", e-acute, a CJK ideograph, a goose, "z"
        String text = "a\u00E9\u4E2D\uD83E\uDEBFz";
        byte[] all = text.getBytes(StandardCharsets.UTF_8);
        for (int cut = 1; cut < all.length; cut++) {
            Utf8ChunkAssembler a = new Utf8ChunkAssembler();
            byte[] head = new byte[cut];
            byte[] tail = new byte[all.length - cut];
            System.arraycopy(all, 0, head, 0, cut);
            System.arraycopy(all, cut, tail, 0, tail.length);
            String s = a.take(head) + a.take(tail) + a.flush();
            assertEquals(text, s, "split at byte " + cut);
        }
    }

    /**
     * A sequence that is merely UNFINISHED is held back; one that can never
     * become valid however many bytes follow is replaced on the spot and
     * decoding continues after it.
     */
    @Test
    void invalidBytesAreReplacedButTruncatedOnesAreHeld() {
        Utf8ChunkAssembler a = new Utf8ChunkAssembler();
        assertEquals("A" + REPLACEMENT + "B", a.take(new byte[] { 'A', (byte) 0xC0, 'B' }),
                "a lead byte that cannot start any valid sequence is replaced at once");
        assertFalse(a.hasPending());

        Utf8ChunkAssembler b = new Utf8ChunkAssembler();
        assertEquals("", b.take(new byte[] { (byte) 0xF0, (byte) 0x9F }), "held, not replaced");
        assertTrue(b.hasPending());
        assertEquals("\uD83E\uDEBF", b.take(new byte[] { (byte) 0xAA, (byte) 0xBF }));
    }

    /** A generation that stopped mid-character hands the tail over, not nothing. */
    @Test
    void flushSurfacesATruncatedTailRatherThanDroppingIt() {
        Utf8ChunkAssembler a = new Utf8ChunkAssembler();
        assertEquals("hi", a.take("hi".getBytes(StandardCharsets.UTF_8)));
        assertEquals("", a.take(new byte[] { (byte) 0xF0, (byte) 0x9F }));
        String tail = a.flush();
        assertFalse(tail.isEmpty(), "the buffered bytes must not be dropped silently");
        assertTrue(tail.indexOf(REPLACEMENT) >= 0);
        assertFalse(a.hasPending());
    }

    @Test
    void emptyAndNullChunksAreIgnored() {
        Utf8ChunkAssembler a = new Utf8ChunkAssembler();
        assertEquals("", a.take(null));
        assertEquals("", a.take(new byte[0]));
        assertEquals("ok", a.take("ok".getBytes(StandardCharsets.UTF_8)));
    }
}
