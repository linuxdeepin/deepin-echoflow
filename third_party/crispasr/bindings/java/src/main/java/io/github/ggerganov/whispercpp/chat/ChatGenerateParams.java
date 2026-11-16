package io.github.ggerganov.whispercpp.chat;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Per-call sampler options of the two generate entry points.
 *
 * <p>A fresh instance is seeded from
 * {@code crispasr_chat_generate_params_default}, so setting one option leaves
 * every other one at the ABI default rather than at a Java zero — in
 * particular {@code temperature} stays 0.8 instead of becoming 0.0, which
 * would silently turn every generation greedy.
 *
 * <p>The setters are fluent:
 * <pre>{@code
 * ChatGenerateParams p = new ChatGenerateParams().maxTokens(64).temperature(0f);
 * }</pre>
 */
public final class ChatGenerateParams {

    private final ChatLib.CGenerateParams c = new ChatLib.CGenerateParams();
    private List<String> stop = Collections.emptyList();

    /** A params object holding the ABI's own defaults. */
    public ChatGenerateParams() {
        ChatLib.INSTANCE.crispasr_chat_generate_params_default(c);
    }

    /** @return the hard cap on generated tokens */
    public int maxTokens() {
        return c.max_tokens;
    }

    /**
     * Cap the tokens generated.
     *
     * <p>0 does NOT mean "generate nothing": the ABI reads any non-positive
     * value as unset and applies its own default of 256. Use
     * {@link #prefillOnly(boolean)} to suppress generation.
     *
     * @param n the cap
     * @return this
     */
    public ChatGenerateParams maxTokens(int n) {
        c.max_tokens = n;
        return this;
    }

    /** @return the sampling temperature; 0.0 = greedy */
    public float temperature() {
        return c.temperature;
    }

    /** @param t sampling temperature; 0.0 = greedy
     *  @return this */
    public ChatGenerateParams temperature(float t) {
        c.temperature = t;
        return this;
    }

    /** @return the top-k cutoff; 0 = disabled */
    public int topK() {
        return c.top_k;
    }

    /** @param k top-k cutoff; 0 disables it
     *  @return this */
    public ChatGenerateParams topK(int k) {
        c.top_k = k;
        return this;
    }

    /** @return the nucleus cutoff; 1.0 = disabled */
    public float topP() {
        return c.top_p;
    }

    /** @param v nucleus cutoff; 1.0 disables it
     *  @return this */
    public ChatGenerateParams topP(float v) {
        c.top_p = v;
        return this;
    }

    /** @return the minimum-probability cutoff; 0.0 = disabled */
    public float minP() {
        return c.min_p;
    }

    /** @param v minimum-probability cutoff; 0.0 disables it
     *  @return this */
    public ChatGenerateParams minP(float v) {
        c.min_p = v;
        return this;
    }

    /** @return the repetition penalty; 1.0 = disabled */
    public float repeatPenalty() {
        return c.repeat_penalty;
    }

    /** @param v repetition penalty; 1.0 disables it
     *  @return this */
    public ChatGenerateParams repeatPenalty(float v) {
        c.repeat_penalty = v;
        return this;
    }

    /** @return the repetition window; -1 = ctx size, 0 = disabled */
    public int repeatLastN() {
        return c.repeat_last_n;
    }

    /** @param n repetition window; -1 = ctx size, 0 disables it
     *  @return this */
    public ChatGenerateParams repeatLastN(int n) {
        c.repeat_last_n = n;
        return this;
    }

    /**
     * @return the sampler seed as an unsigned 32-bit value; 0xFFFFFFFF = random
     */
    public long seed() {
        return c.seed & 0xFFFFFFFFL;
    }

    /** @param s sampler seed; 0xFFFFFFFF draws a random one
     *  @return this */
    public ChatGenerateParams seed(long s) {
        c.seed = (int) s;
        return this;
    }

    /** @return a copy of the stop sequences */
    public List<String> stop() {
        return new ArrayList<String>(stop);
    }

    /**
     * Halt generation the first time any of these substrings appears in the
     * accumulated output, which is truncated BEFORE the match.
     *
     * <p>To clear the sequences pass an EMPTY ARRAY, not an empty argument
     * list: {@code stop()} written with no arguments binds to the getter
     * above, leaving the sequences untouched.
     *
     * @param stop the sequences
     * @return this
     */
    public ChatGenerateParams stop(String... stop) {
        this.stop = stop == null ? Collections.<String>emptyList()
                : new ArrayList<String>(Arrays.asList(stop));
        return this;
    }

    /** @return whether assistant generation is suppressed */
    public boolean prefillOnly() {
        return c.prefill_only != 0;
    }

    /**
     * Prefill the system / user portion with assistant generation suppressed —
     * useful for measuring prompt cost.
     *
     * @param v true to suppress generation
     * @return this
     */
    public ChatGenerateParams prefillOnly(boolean v) {
        c.prefill_only = (byte) (v ? 1 : 0);
        return this;
    }

    /**
     * Build the C struct for one call, attaching the stop sequences.
     *
     * @param owned collects the allocations the struct borrows
     * @return the struct to pass
     * @throws IllegalArgumentException if a stop sequence holds an interior NUL
     */
    ChatLib.CGenerateParams toNative(List<Memory> owned) {
        if (stop.isEmpty()) {
            c.stop = Pointer.NULL;
            c.n_stop = new ChatLib.SizeT(0);
            return c;
        }
        Memory array = new Memory((long) stop.size() * Native.POINTER_SIZE);
        owned.add(array);
        for (int i = 0; i < stop.size(); i++) {
            array.setPointer((long) i * Native.POINTER_SIZE,
                    ChatNative.cstring(stop.get(i), "stop[" + i + "]", owned));
        }
        c.stop = array;
        c.n_stop = new ChatLib.SizeT(stop.size());
        return c;
    }
}
