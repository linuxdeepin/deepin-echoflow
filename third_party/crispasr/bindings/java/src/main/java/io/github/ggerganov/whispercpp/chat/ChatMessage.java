package io.github.ggerganov.whispercpp.chat;

/**
 * One turn of a conversation. {@code role} is one of "system", "user",
 * "assistant" or "tool", matching the OpenAI chat schema; the chat template
 * translates those names into whatever the model expects.
 *
 * <p>Immutable, so a list of them can be reused across calls — which is the
 * normal thing to do, since the session wants the WHOLE conversation on every
 * generate call for its KV cache to share a prefix.
 */
public final class ChatMessage {

    private final String role;
    private final String content;

    /**
     * @param role    "system", "user", "assistant" or "tool"
     * @param content the turn's text
     * @throws IllegalArgumentException if either argument is null
     */
    public ChatMessage(String role, String content) {
        if (role == null) {
            throw new IllegalArgumentException("crispasr_chat: role must not be null");
        }
        if (content == null) {
            throw new IllegalArgumentException("crispasr_chat: content must not be null");
        }
        this.role = role;
        this.content = content;
    }

    /** @param content the system instruction
     *  @return a "system" turn */
    public static ChatMessage system(String content) {
        return new ChatMessage("system", content);
    }

    /** @param content what the user said
     *  @return a "user" turn */
    public static ChatMessage user(String content) {
        return new ChatMessage("user", content);
    }

    /** @param content what the assistant said
     *  @return an "assistant" turn */
    public static ChatMessage assistant(String content) {
        return new ChatMessage("assistant", content);
    }

    /** @param content the tool's output
     *  @return a "tool" turn */
    public static ChatMessage tool(String content) {
        return new ChatMessage("tool", content);
    }

    /** @return the role name */
    public String role() {
        return role;
    }

    /** @return the turn's text */
    public String content() {
        return content;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ChatMessage)) {
            return false;
        }
        ChatMessage other = (ChatMessage) o;
        return role.equals(other.role) && content.equals(other.content);
    }

    @Override
    public int hashCode() {
        return 31 * role.hashCode() + content.hashCode();
    }

    @Override
    public String toString() {
        return "ChatMessage[" + role + "]";
    }
}
