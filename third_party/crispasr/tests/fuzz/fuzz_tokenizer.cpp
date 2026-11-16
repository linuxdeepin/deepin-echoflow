// fuzz_tokenizer.cpp — libFuzzer harness over the shared text tokenizers
// core_bpe::tokenize_simple (GPT-2 byte-level BPE) and
// core_wordpiece::Tokenizer::tokenize (BERT WordPiece).
//
// The TEXT fed to a tokenizer is untrusted: it is the user's prompt / reference
// transcript / `--tts` / `--ref-text` string, or a caption pulled from a file.
// The vocab/merges come from the (separately fuzzed, see fuzz_gguf_meta) model
// GGUF, so here we pin small benign vocabs and fuzz only the text — exercising
// the byte→unicode map, the whitespace/punctuation pre-tokenizer, the BPE merge
// loop, and the WordPiece greedy longest-match over arbitrary bytes (invalid
// UTF-8, lone continuation bytes, huge repeats, embedded NULs). Any OOB read,
// unbounded alloc, or infinite loop shows up under ASan/UBSan + libFuzzer.
//
//   cmake -B build-fuzz -DCRISPASR_FUZZ=ON -DCRISPASR_SANITIZE_ADDRESS=ON \
//         -DCRISPASR_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
//   cmake --build build-fuzz --target crispasr-fuzz-tokenizer
//   ./build-fuzz/bin/crispasr-fuzz-tokenizer -max_len=65536 corpus

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "core/bpe.h"
#include "core/wordpiece.h"

// A tiny GPT-2-style vocab: byte-encoded single chars + a couple of merges so
// bpe_one's merge loop is actually taken. Exact contents are irrelevant to the
// safety property — arbitrary text still drives every code path.
static const std::unordered_map<std::string, int32_t>& bpe_vocab() {
    static const std::unordered_map<std::string, int32_t> v = {
        {"h", 0},  {"e", 1},  {"l", 2},    {"o", 3},     {"\xC4\xA0", 4}, // 'Ġ' (byte-encoded space)
        {"he", 5}, {"ll", 6}, {"hell", 7}, {"hello", 8}, {"\xC4\xA0h", 9},
    };
    return v;
}
static const std::unordered_map<std::string, int32_t>& bpe_merges() {
    static const std::unordered_map<std::string, int32_t> m = {
        {"h e", 0}, {"l l", 1}, {"he ll", 2}, {"hell o", 3}, {"\xC4\xA0 h", 4},
    };
    return m;
}

static const core_wordpiece::Tokenizer& wp_tokenizer() {
    static const core_wordpiece::Tokenizer t = [] {
        core_wordpiece::Tokenizer w;
        w.id_to_token = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "the",
                         "##ing", "##s",   "a",     "!",     "?",     ",",     "."};
        w.unk_id = 1;
        w.build_map();
        return w;
    }();
    return t;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Bound the input so a pathological case can't just OOM the fuzzer host;
    // real prompts/transcripts are far below this.
    if (size > 1u * 1024u * 1024u)
        return 0;

    const std::string text(reinterpret_cast<const char*>(data), size);

    // GPT-2 byte-level BPE encode over arbitrary text.
    (void)core_bpe::tokenize_simple(bpe_vocab(), bpe_merges(), text);

    // BERT WordPiece tokenize over arbitrary text.
    (void)wp_tokenizer().tokenize(text);

    return 0;
}
