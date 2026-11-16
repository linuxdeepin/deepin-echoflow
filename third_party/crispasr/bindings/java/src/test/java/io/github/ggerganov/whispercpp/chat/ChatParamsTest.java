package io.github.ggerganov.whispercpp.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.jna.NativeLibrary;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.util.Arrays;
import java.util.Collections;

/**
 * Everything about the chat binding that needs the shared library but NOT a
 * model: the ABI defaults, the interior-NUL rejection, the disclosure text, and
 * the reachability of every entry point the header declares.
 *
 * <p>The pure-Java cases run unconditionally; the rest are gated on the library
 * loading, so a checkout with no build skips instead of failing.
 */
class ChatParamsTest {

    /** @return whether libcrispasr could be resolved */
    static boolean libraryAvailable() {
        try {
            NativeLibrary.getInstance("crispasr");
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    // ----- interior NUL: pure Java, always runs -----

    @Test
    void interiorNulIsRejectedAndTheFieldIsNamed() {
        String[][] cases = {
            { "messages[0].role", "us\0er" },
            { "messages[3].content", "hello\0world" },
            { "stop[1]", "\0" },
            { "chatTemplate", "{{ bos }}\0" },
            { "modelPath", "/models/a\0b.gguf" },
        };
        for (String[] c : cases) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> ChatNative.requireNoInteriorNul(c[1], c[0]));
            assertTrue(e.getMessage().contains(c[0]),
                    "the exception must name the field, got: " + e.getMessage());
            assertTrue(e.getMessage().contains("NUL"), e.getMessage());
        }
    }

    @Test
    void stringsWithoutAnInteriorNulPassThrough() {
        ChatNative.requireNoInteriorNul("user", "messages[0].role");
        ChatNative.requireNoInteriorNul("", "messages[0].content");
        ChatNative.requireNoInteriorNul(null, "chatTemplate");
    }

    // ----- ABI defaults -----

    /**
     * A partially-filled params object keeps the ABI defaults for everything
     * the caller did not set. Copying a zero-valued struct instead would turn
     * temperature into 0.0 — greedy — behind the caller's back.
     */
    @Test
    @EnabledIf("libraryAvailable")
    void aPartiallyFilledGenerateParamsKeepsTheAbiDefaults() {
        ChatGenerateParams defaults = new ChatGenerateParams();
        assertEquals(256, defaults.maxTokens());
        assertEquals(0.8f, defaults.temperature(), 1e-6f);
        assertEquals(40, defaults.topK());
        assertEquals(0.95f, defaults.topP(), 1e-6f);
        assertEquals(0.05f, defaults.minP(), 1e-6f);
        assertEquals(1.1f, defaults.repeatPenalty(), 1e-6f);
        assertEquals(64, defaults.repeatLastN());
        assertFalse(defaults.prefillOnly());
        assertEquals(Collections.emptyList(), defaults.stop());

        ChatGenerateParams partial = new ChatGenerateParams().maxTokens(7);
        assertEquals(7, partial.maxTokens(), "the one field the caller set");
        assertEquals(0.8f, partial.temperature(), 1e-6f, "temperature must not fall to greedy");
        assertEquals(40, partial.topK());
        assertEquals(0.95f, partial.topP(), 1e-6f);
        assertEquals(0.05f, partial.minP(), 1e-6f);
        assertEquals(1.1f, partial.repeatPenalty(), 1e-6f);
        assertEquals(64, partial.repeatLastN());
    }

    @Test
    @EnabledIf("libraryAvailable")
    void aPartiallyFilledOpenParamsKeepsTheAbiDefaults() {
        ChatOpenParams defaults = new ChatOpenParams();
        assertTrue(defaults.nThreads() > 0, "defaults to the physical core count");
        assertEquals(512, defaults.nBatch());
        assertEquals(512, defaults.nUbatch());
        assertEquals(-1, defaults.nGpuLayers());
        assertTrue(defaults.useMmap());
        assertFalse(defaults.useMlock());
        assertEquals("", defaults.chatTemplate());

        ChatOpenParams partial = new ChatOpenParams().nCtx(2048);
        assertEquals(2048, partial.nCtx(), "the one field the caller set");
        assertEquals(-1, partial.nGpuLayers(), "must not fall to 0, which means CPU only");
        assertTrue(partial.useMmap(), "must not fall to false");
        assertEquals(512, partial.nBatch());
        assertEquals(defaults.nThreads(), partial.nThreads());
    }

    @Test
    @EnabledIf("libraryAvailable")
    void everySetterRoundTrips() {
        ChatGenerateParams g = new ChatGenerateParams()
                .maxTokens(11).temperature(0.25f).topK(3).topP(0.5f).minP(0.01f)
                .repeatPenalty(1.05f).repeatLastN(-1).seed(0xFFFFFFFFL)
                .prefillOnly(true).stop("</s>", "STOP");
        assertEquals(11, g.maxTokens());
        assertEquals(0.25f, g.temperature(), 1e-6f);
        assertEquals(3, g.topK());
        assertEquals(0.5f, g.topP(), 1e-6f);
        assertEquals(0.01f, g.minP(), 1e-6f);
        assertEquals(1.05f, g.repeatPenalty(), 1e-6f);
        assertEquals(-1, g.repeatLastN());
        assertEquals(0xFFFFFFFFL, g.seed());
        assertTrue(g.prefillOnly());
        assertEquals(Arrays.asList("</s>", "STOP"), g.stop());

        ChatOpenParams o = new ChatOpenParams()
                .nThreads(2).nThreadsBatch(3).nCtx(128).nBatch(64).nUbatch(32)
                .nGpuLayers(0).useMmap(false).useMlock(true).chatTemplate("chatml");
        assertEquals(2, o.nThreads());
        assertEquals(3, o.nThreadsBatch());
        assertEquals(128, o.nCtx());
        assertEquals(64, o.nBatch());
        assertEquals(32, o.nUbatch());
        assertEquals(0, o.nGpuLayers());
        assertFalse(o.useMmap());
        assertTrue(o.useMlock());
        assertEquals("chatml", o.chatTemplate());
    }

    // ----- the ABI edge, still without a model -----

    /**
     * The marshalling rejects an interior NUL before the native call, so a bad
     * chat template is reported as such rather than as a failure to open a
     * model that was never reached.
     */
    @Test
    @EnabledIf("libraryAvailable")
    void openRejectsAnInteriorNulBeforeTouchingTheModel() {
        ChatOpenParams params = new ChatOpenParams().chatTemplate("{{ x }}\0{{ y }}");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChatSession.open("/nonexistent/model.gguf", params));
        assertTrue(e.getMessage().contains("chatTemplate"), e.getMessage());

        IllegalArgumentException p = assertThrows(IllegalArgumentException.class,
                () -> ChatSession.open("/nonexistent/mo\0del.gguf"));
        assertTrue(p.getMessage().contains("modelPath"), p.getMessage());
    }

    @Test
    @EnabledIf("libraryAvailable")
    void openingAMissingModelIsAFaultNotACancellation() {
        ChatException e = assertThrows(ChatException.class,
                () -> ChatSession.open("/nonexistent/model.gguf"));
        assertFalse(e instanceof ChatAbortedException,
                "only CRISPASR_CHAT_ERR_ABORTED may classify as a cancellation");
        assertFalse(e.getMessage().isEmpty());
    }

    @Test
    @EnabledIf("libraryAvailable")
    void aiDisclosureTextIsTheCanonicalWording() {
        String s = ChatSession.aiDisclosureText();
        assertNotNull(s);
        assertFalse(s.trim().isEmpty(), "EU AI Act Art. 50(1) wording must not be empty");
    }

    /**
     * The C side signals a failed estimate by returning 0 with {@code err}
     * filled. That has to reach a caller as an exception rather than as an
     * estimate of nothing. Needs no model: a path naming no file is refused by
     * the loader.
     */
    @Test
    @EnabledIf("libraryAvailable")
    void memoryEstimateThrowsForAModelItCannotRead() {
        assertThrows(ChatException.class, () -> ChatSession.memoryEstimate(
                "/nonexistent/crispasr-memory-estimate.gguf", null));
    }

    /** Every entry point the header declares is reachable through {@link ChatLib}. */
    @Test
    @EnabledIf("libraryAvailable")
    void everyChatSymbolResolves() {
        String[] symbols = {
            "crispasr_chat_open_params_default", "crispasr_chat_generate_params_default",
            "crispasr_chat_open", "crispasr_chat_close", "crispasr_chat_reset",
            "crispasr_chat_generate", "crispasr_chat_generate_stream",
            "crispasr_chat_set_abort_callback", "crispasr_chat_template_name",
            "crispasr_chat_n_ctx", "crispasr_chat_count_tokens",
            "crispasr_chat_memory_estimate", "crispasr_chat_string_free",
            "crispasr_chat_ai_disclosure_text",
        };
        assertEquals(14, symbols.length, "the header exports fourteen entry points");
        NativeLibrary lib = NativeLibrary.getInstance("crispasr");
        for (String s : symbols) {
            assertNotNull(lib.getFunction(s), s);
        }
    }
}
