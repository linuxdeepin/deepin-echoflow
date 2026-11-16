package io.github.ggerganov.whispercpp.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The per-call state {@link ChatSession} hangs a streamed generation on, driven
 * by hand: once the token listener has thrown, nothing more may be handed to
 * it, on either the delivery path or the end-of-generation flush.
 *
 * <p>Neither guard can be reached through a real generation. C asks the abort
 * hook before it offers another chunk, so the delivery one never fires there;
 * the flush one needs a chunk that completes one character and opens another,
 * and the detokenizer spells a character it has no token for one byte per
 * callback, so its chunks do not have that shape. Building the state object
 * directly is what makes both observable. No model and no shared library.
 */
class CallStateTest {

    /** A chunk of raw bytes as C would deliver it: NUL-terminated native memory. */
    private static Pointer chunk(int... bytes) {
        Memory m = new Memory(bytes.length + 1L);
        for (int i = 0; i < bytes.length; i++) {
            m.setByte(i, (byte) bytes[i]);
        }
        m.setByte(bytes.length, (byte) 0);
        return m;
    }

    /** 'A', which stands alone, then the first byte of a three-byte character. */
    private static Pointer completeThenPartial() {
        return chunk(0x41, 0xE2);
    }

    @Test
    void nothingIsDeliveredOnceTheListenerHasThrown() {
        List<String> seen = new ArrayList<String>();
        IllegalStateException boom = new IllegalStateException("boom from the token listener");
        ChatSession.CallState state = new ChatSession.CallState(text -> {
            seen.add(text);
            throw boom;
        }, null);

        state.deliver(chunk(0x41));
        assertEquals(Arrays.asList("A"), seen, "the first chunk must reach the listener");

        state.deliver(chunk(0x42));
        assertEquals(Arrays.asList("A"), seen,
                "a chunk offered after the failure must not reach the listener");

        assertSame(boom, assertThrows(IllegalStateException.class, state::rethrow));
    }

    @Test
    void theFlushedTailIsNotDeliveredOnceTheListenerHasThrown() {
        List<String> seen = new ArrayList<String>();
        IllegalStateException boom = new IllegalStateException("boom from the token listener");
        ChatSession.CallState state = new ChatSession.CallState(text -> {
            seen.add(text);
            throw boom;
        }, null);

        state.deliver(completeThenPartial());
        assertEquals(Arrays.asList("A"), seen, "the completed character must reach the listener");

        state.flush();
        assertEquals(Arrays.asList("A"), seen,
                "the held-back tail must not reach a listener that already threw");

        assertSame(boom, assertThrows(IllegalStateException.class, state::rethrow));
    }

    /**
     * The positive control for the case above: the same chunk really does leave
     * a tail, and the flush really does deliver it. Without this, a chunk that
     * had buffered nothing would make the guard look effective when there was
     * nothing for it to hold back.
     */
    @Test
    void theFlushedTailIsDeliveredWhenNothingHasThrown() {
        List<String> seen = new ArrayList<String>();
        ChatSession.CallState state = new ChatSession.CallState(seen::add, null);

        state.deliver(completeThenPartial());
        assertEquals(Arrays.asList("A"), seen, "only the completed character is deliverable yet");

        state.flush();
        assertEquals(Arrays.asList("A", "\uFFFD"), seen,
                "the tail is a character the generation stopped inside, and is handed over");

        state.rethrow();
    }
}
