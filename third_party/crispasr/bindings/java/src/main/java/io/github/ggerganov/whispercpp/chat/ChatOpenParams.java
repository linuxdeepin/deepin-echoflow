package io.github.ggerganov.whispercpp.chat;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;

import java.util.List;

/**
 * Per-session, model-level options of {@code crispasr_chat_open}.
 *
 * <p>A fresh instance is seeded from {@code crispasr_chat_open_params_default},
 * so setting one option leaves every other one at the ABI default rather than
 * at a Java zero — {@code nGpuLayers} stays -1 (all layers) instead of becoming
 * 0 (CPU only), {@code useMmap} stays true, {@code nBatch} stays 512.
 *
 * <p>{@code embeddings} is deliberately not exposed: the header says "future
 * use; keep false".
 *
 * <p>The setters are fluent, so a call chain reads as the options it sets:
 * <pre>{@code
 * ChatOpenParams p = new ChatOpenParams().nCtx(4096).nThreads(8);
 * }</pre>
 */
public final class ChatOpenParams {

    private final ChatLib.COpenParams c = new ChatLib.COpenParams();
    private String chatTemplate = "";

    /** A params object holding the ABI's own defaults. */
    public ChatOpenParams() {
        ChatLib.INSTANCE.crispasr_chat_open_params_default(c);
    }

    /** @return generation threads */
    public int nThreads() {
        return c.n_threads;
    }

    /** @param n generation threads
     *  @return this */
    public ChatOpenParams nThreads(int n) {
        c.n_threads = n;
        return this;
    }

    /** @return batch / prefill threads */
    public int nThreadsBatch() {
        return c.n_threads_batch;
    }

    /** @param n batch / prefill threads
     *  @return this */
    public ChatOpenParams nThreadsBatch(int n) {
        c.n_threads_batch = n;
        return this;
    }

    /** @return the requested context window in tokens; 0 = the model's own */
    public int nCtx() {
        return c.n_ctx;
    }

    /** @param n context window in tokens; 0 takes the model's own
     *  @return this */
    public ChatOpenParams nCtx(int n) {
        c.n_ctx = n;
        return this;
    }

    /** @return the logical batch size */
    public int nBatch() {
        return c.n_batch;
    }

    /** @param n logical batch size
     *  @return this */
    public ChatOpenParams nBatch(int n) {
        c.n_batch = n;
        return this;
    }

    /** @return the physical micro-batch size */
    public int nUbatch() {
        return c.n_ubatch;
    }

    /** @param n physical micro-batch size
     *  @return this */
    public ChatOpenParams nUbatch(int n) {
        c.n_ubatch = n;
        return this;
    }

    /** @return offloaded layers; -1 = all, 0 = CPU only */
    public int nGpuLayers() {
        return c.n_gpu_layers;
    }

    /** @param n offloaded layers; -1 = all, 0 = CPU only
     *  @return this */
    public ChatOpenParams nGpuLayers(int n) {
        c.n_gpu_layers = n;
        return this;
    }

    /** @return whether the weights are mapped rather than read */
    public boolean useMmap() {
        return c.use_mmap != 0;
    }

    /** @param v map the weights rather than read them
     *  @return this */
    public ChatOpenParams useMmap(boolean v) {
        c.use_mmap = (byte) (v ? 1 : 0);
        return this;
    }

    /** @return whether the weights are locked into RAM */
    public boolean useMlock() {
        return c.use_mlock != 0;
    }

    /** @param v lock the weights into RAM
     *  @return this */
    public ChatOpenParams useMlock(boolean v) {
        c.use_mlock = (byte) (v ? 1 : 0);
        return this;
    }

    /** @return the template override, empty for none */
    public String chatTemplate() {
        return chatTemplate;
    }

    /**
     * Override the template baked into the GGUF. Empty — the default — reads
     * {@code tokenizer.chat_template} from the model and falls back to
     * {@code "chatml"} if the model has none. The ABI copies the string, so
     * nothing has to outlive the open call.
     *
     * @param template the Jinja template, or empty for the model's own
     * @return this
     */
    public ChatOpenParams chatTemplate(String template) {
        this.chatTemplate = template == null ? "" : template;
        return this;
    }

    /**
     * Build the C struct for one call, attaching the template override.
     *
     * @param owned collects the allocations the struct borrows
     * @return the struct to pass
     * @throws IllegalArgumentException if the template holds an interior NUL
     */
    ChatLib.COpenParams toNative(List<Memory> owned) {
        c.chat_template = chatTemplate.isEmpty()
                ? Pointer.NULL
                : ChatNative.cstring(chatTemplate, "chatTemplate", owned);
        return c;
    }
}
