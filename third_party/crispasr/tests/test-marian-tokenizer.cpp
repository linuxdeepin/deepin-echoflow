// test-marian-tokenizer.cpp — unit tests for core/marian_tokenizer.h and the
// Marian branch of the m2m100 loader. No model files.
//
// The properties pinned here are the ones where "close" tokenization would
// still translate, just worse, and nothing downstream would notice:
//   * the SentencePiece normalizer (charsmap longest-match, whitespace rules);
//   * unigram Viterbi as SentencePiece runs it — an <unk> edge only where no
//     one-character piece exists, runs of unknowns merged;
//   * ids come from vocab.json, NOT from the SentencePiece table: a piece the
//     SP model has and vocab.json lacks is <unk>;
//   * a Marian GGUF with a tokenizer table missing is refused at load.
//
// Parity with the real Opus-MT tokenizers (token ids equal to Hugging Face's)
// is measured with the checkpoints, by tools/marian_parity.py.

#include <catch2/catch_test_macros.hpp>

#include "core/marian_tokenizer.h"
#include "m2m100.h"

#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using core_marian_tok::Normalizer;
using core_marian_tok::Piece;
using core_marian_tok::Tokenizer;
using core_marian_tok::Unigram;

namespace {

const char* const SP = "\xE2\x96\x81"; // U+2581

std::string sp(const std::string& ascii) {
    // '_' in the test strings stands for U+2581.
    std::string out;
    for (char ch : ascii) {
        if (ch == '_')
            out += SP;
        else
            out += ch;
    }
    return out;
}

// A two-key Darts-clone double array, built by hand:
//   "A"  → "b"      "\t" → " "
// Layout: root at 0 with offset 256; the child for byte c sits at 256 ^ c and
// carries (label | has_leaf | offset << 10); its value unit sits at
// child ^ offset with bit 31 set.
std::vector<uint8_t> tiny_charsmap() {
    std::vector<uint32_t> units(322, 0);
    units[0] = 256u << 10;
    units[256 ^ 0x41] = 0x41u | (1u << 8) | (1u << 10); // 'A', value unit at 321 ^ 1 = 320
    units[320] = (1u << 31) | 0u;                       // → pool offset 0 ("b")
    units[256 ^ 0x09] = 0x09u | (1u << 8) | (2u << 10); // '\t', value unit at 265 ^ 2 = 267
    units[267] = (1u << 31) | 2u;                       // → pool offset 2 (" ")
    const char pool[] = {'b', '\0', ' ', '\0'};
    const uint32_t trie_bytes = (uint32_t)(units.size() * 4);
    std::vector<uint8_t> blob(4 + trie_bytes + sizeof(pool));
    std::memcpy(blob.data(), &trie_bytes, 4);
    std::memcpy(blob.data() + 4, units.data(), trie_bytes);
    std::memcpy(blob.data() + 4 + trie_bytes, pool, sizeof(pool));
    return blob;
}

std::vector<std::string> texts(const std::vector<Piece>& pieces) {
    std::vector<std::string> out;
    for (const Piece& p : pieces)
        out.push_back(p.is_unk ? "<unk:" + p.text + ">" : p.text);
    return out;
}

} // namespace

TEST_CASE("marian normalizer: whitespace rules without a charsmap", "[unit][marian]") {
    Normalizer n;
    REQUIRE(n.load_charsmap(nullptr, 0));
    CHECK(n.normalize("Hallo Welt") == sp("_Hallo_Welt"));
    // leading / trailing / repeated spaces are removed, not turned into pieces
    CHECK(n.normalize("   Hallo    Welt   ") == sp("_Hallo_Welt"));
    CHECK(n.normalize("").empty());
    CHECK(n.normalize("    ").empty());
    // UTF-8 passes through untouched
    CHECK(n.normalize("Stra\xC3\x9F" "e") == sp("_Stra\xC3\x9F" "e"));
    // a stray continuation byte becomes U+FFFD, as in SentencePiece
    CHECK(n.normalize("a\x80z") == sp("_a\xEF\xBF\xBDz"));
    // a truncated sequence at the end too
    CHECK(n.normalize("a\xC3") == sp("_a\xEF\xBF\xBD"));

    Normalizer raw;
    raw.add_dummy_prefix = false;
    raw.remove_extra_whitespaces = false;
    CHECK(raw.normalize(" a  b ") == sp("_a__b_"));
}

TEST_CASE("marian normalizer: precompiled charsmap, longest match", "[unit][marian]") {
    Normalizer n;
    const std::vector<uint8_t> blob = tiny_charsmap();
    REQUIRE(n.load_charsmap(blob.data(), blob.size()));
    CHECK(n.normalize("A") == sp("_b"));
    CHECK(n.normalize("xAy") == sp("_xby"));
    // the tab is mapped to a space FIRST, so it obeys the whitespace rules:
    // collapsed in the middle, stripped at both ends
    CHECK(n.normalize("A\tA") == sp("_b_b"));
    CHECK(n.normalize("A \t A") == sp("_b_b"));
    CHECK(n.normalize("\tA\t") == sp("_b"));
    // bytes that are not keys are untouched ('@' = 0x40 shares the value slot
    // index 256 ^ 0x40 = 320 with nothing: its unit has bit 31 set)
    CHECK(n.normalize("@B") == sp("_@B"));
}

TEST_CASE("marian normalizer: a malformed charsmap is refused", "[unit][marian]") {
    Normalizer n;
    const uint8_t three[3] = {1, 2, 3};
    CHECK_FALSE(n.load_charsmap(three, 3));
    // declared trie size larger than the blob
    const uint8_t lying[8] = {0xFF, 0xFF, 0x00, 0x00, 0, 0, 0, 0};
    CHECK_FALSE(n.load_charsmap(lying, 8));
    // not a multiple of the unit size
    const uint8_t odd[8] = {0x03, 0x00, 0x00, 0x00, 0, 0, 0, 0};
    CHECK_FALSE(n.load_charsmap(odd, 8));
}

TEST_CASE("marian unigram: Viterbi, not greedy longest match", "[unit][marian]") {
    Unigram u;
    //                0     1        2      3     4     5      6
    u.load({"<unk>", "</s>", sp("_ab"), "c", sp("_a"), "bc", sp("_")},
           {0.0f, 0.0f, -1.0f, -9.0f, -2.0f, -2.0f, -5.0f}, {2, 3, 1, 1, 1, 1, 1});
    // greedy would take "_ab" + "c" (-10); the best path is "_a" + "bc" (-4)
    CHECK(texts(u.encode(sp("_abc"))) == std::vector<std::string>{sp("_a"), "bc"});
    CHECK(u.encode("").empty());
}

TEST_CASE("marian unigram: unknowns", "[unit][marian]") {
    Unigram u;
    u.load({"<unk>", "</s>", sp("_"), "a", "qu", sp("_q")}, {0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -30.0f},
           {2, 3, 1, 1, 1, 1});
    // a run of characters with no piece is ONE unknown, carrying its surface
    CHECK(texts(u.encode(sp("_xyz"))) == std::vector<std::string>{sp("_"), "<unk:xyz>"});
    CHECK(texts(u.encode(sp("_axa"))) == std::vector<std::string>{sp("_"), "a", "<unk:x>", "a"});
    // 'q' has no one-character piece, so SentencePiece offers an <unk> edge
    // there (min_score - 10 = -40) next to the pieces that start with it
    CHECK(texts(u.encode(sp("_qua"))) == std::vector<std::string>{sp("_"), "qu", "a"});
    CHECK(texts(u.encode(sp("_qa"))) == std::vector<std::string>{sp("_q"), "a"});
    // control pieces are not text: "</s>" in the input is five characters
    const auto pieces = u.encode(sp("_</s>"));
    REQUIRE(pieces.size() == 2);
    CHECK(pieces[1].is_unk);
    CHECK(pieces[1].text == "</s>");
}

TEST_CASE("marian unigram: a user-defined piece always wins", "[unit][marian]") {
    // SentencePiece ignores a user-defined piece's stored score and gives it
    // length * max_score - 0.1 instead: here 2 * -1 - 0.1 = -2.1, which beats
    // "a" + "b" = -4 although the table says -50.
    Unigram u;
    u.load({"<unk>", sp("_"), "a", "b", "ab"}, {0.0f, -1.0f, -2.0f, -2.0f, -50.0f}, {2, 1, 1, 1, 4});
    CHECK(texts(u.encode(sp("_ab"))) == std::vector<std::string>{sp("_"), "ab"});
    // … and as an ordinary piece with that score it loses.
    Unigram v;
    v.load({"<unk>", sp("_"), "a", "b", "ab"}, {0.0f, -1.0f, -2.0f, -2.0f, -50.0f}, {2, 1, 1, 1, 1});
    CHECK(texts(v.encode(sp("_ab"))) == std::vector<std::string>{sp("_"), "a", "b"});
}

TEST_CASE("marian tokenizer: ids are vocab.json ids, not SentencePiece ids", "[unit][marian]") {
    Tokenizer t;
    REQUIRE(t.norm.load_charsmap(nullptr, 0));
    // SentencePiece order …
    t.src.load({"<unk>", "<s>", "</s>", sp("_Haus"), sp("_peoples"), sp("_"), "e"},
               {0.0f, 0.0f, 0.0f, -2.0f, -2.0f, -3.0f, -3.0f}, {2, 3, 3, 1, 1, 1, 1});
    // … and a vocab.json that orders differently, lacks "_peoples", and has a
    // target-side character the source model does not know.
    t.vocab = {{"</s>", 0}, {"<unk>", 1}, {"e", 2}, {sp("_Haus"), 3}, {sp("_"), 4}, {"\xE7\xAA\xAA", 5}, {">>fr<<", 6},
               {"<pad>", 7}};
    t.unk_id = 1;
    t.eos_id = 0;

    CHECK(t.encode("Haus") == std::vector<int32_t>{3, 0});
    CHECK(t.encode("Haus e") == std::vector<int32_t>{3, 4, 2, 0});
    // known to SentencePiece, absent from vocab.json → <unk> (HF does this for
    // "▁peoples" in opus-mt-en-de)
    CHECK(t.encode("peoples") == std::vector<int32_t>{1, 0});
    // unknown to SentencePiece, but its surface is a vocab.json token
    CHECK(t.encode("Haus \xE7\xAA\xAA") == std::vector<int32_t>{3, 4, 5, 0});
    // unknown to both
    CHECK(t.encode("Haus \xE5\xB0\xAD") == std::vector<int32_t>{3, 4, 1, 0});
    // empty input is just </s>
    CHECK(t.encode("") == std::vector<int32_t>{0});
    CHECK(t.encode("   ") == std::vector<int32_t>{0});
    CHECK(t.encode("Haus", /*add_eos=*/false) == std::vector<int32_t>{3});
}

TEST_CASE("marian tokenizer: a leading >>xx<< is one token", "[unit][marian]") {
    Tokenizer t;
    REQUIRE(t.norm.load_charsmap(nullptr, 0));
    t.src.load({"<unk>", sp("_Haus")}, {0.0f, -1.0f}, {2, 1});
    t.vocab = {{"</s>", 0}, {"<unk>", 1}, {sp("_Haus"), 3}, {">>fr<<", 6}};
    CHECK(Tokenizer::language_code_len(">>fr<< Haus") == 6);
    CHECK(Tokenizer::language_code_len("Haus >>fr<<") == 0);
    CHECK(Tokenizer::language_code_len(">>fr") == 0);
    CHECK(t.encode(">>fr<< Haus") == std::vector<int32_t>{6, 3, 0});
    CHECK(t.encode(">>fr<<Haus") == std::vector<int32_t>{6, 3, 0});
    // a code the model does not have is <unk>, not silently dropped
    CHECK(t.encode(">>xx<< Haus") == std::vector<int32_t>{1, 3, 0});
}

TEST_CASE("marian tokenizer: joining pieces", "[unit][marian]") {
    using core_marian_tok::join_pieces;
    CHECK(join_pieces({sp("_Hello"), ",", sp("_world"), "."}) == "Hello, world.");
    CHECK(join_pieces({sp("_3"), ".", sp("_Oktober")}) == "3. Oktober");
    CHECK(join_pieces({sp("_"), sp("_a_")}) == "a");
    CHECK(join_pieces({}).empty());
}

// ── the loader ─────────────────────────────────────────────────────────────

namespace {

std::string write_marian_meta(const char* tag, bool with_hparams, bool with_vocab) {
    const std::string path = std::string("test-marian-") + tag + ".gguf";
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "marian");
    if (with_hparams) {
        for (const char* key : {"marian.d_model", "marian.encoder.n_layers", "marian.encoder.n_heads",
                                "marian.encoder.ffn_dim", "marian.decoder.n_layers", "marian.decoder.n_heads",
                                "marian.decoder.ffn_dim", "marian.max_position_embeddings", "marian.scale_embedding"})
            gguf_set_val_u32(g, key, 8);
        gguf_set_val_u32(g, "marian.vocab_size", 3);
        gguf_set_val_u32(g, "marian.normalize_before", 0);
        gguf_set_val_u32(g, "marian.eos_token_id", 0);
        gguf_set_val_u32(g, "marian.pad_token_id", 2);
        gguf_set_val_u32(g, "marian.unk_token_id", 1);
        gguf_set_val_u32(g, "marian.decoder_start_token_id", 2);
        gguf_set_val_u32(g, "marian.gen.num_beams", 4);
        gguf_set_val_u32(g, "marian.gen.max_length", 512);
        gguf_set_val_u32(g, "marian.gen.early_stopping", 0);
        gguf_set_val_str(g, "marian.activation_function", "swish");
    }
    if (with_vocab) {
        const char* toks[] = {"</s>", "<unk>", "<pad>"};
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks, 3);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), /*only_meta=*/true);
    gguf_free(g);
    REQUIRE(ok);
    return path;
}

m2m100_context* try_load(const std::string& path) {
    m2m100_context_params p = m2m100_context_default_params();
    p.verbosity = 0;
    p.use_gpu = false;
    m2m100_context* ctx = m2m100_init_from_file(path.c_str(), p);
    std::remove(path.c_str());
    return ctx;
}

} // namespace

TEST_CASE("marian loader: a GGUF without hyperparameters is refused", "[unit][marian]") {
    // An m2m100 file falls back to the 418M's numbers for a missing key. A
    // Marian file must not: there is no checkpoint those defaults describe.
    m2m100_context* ctx = try_load(write_marian_meta("nohp", false, false));
    CHECK(ctx == nullptr);
    m2m100_free(ctx);
}

TEST_CASE("marian loader: a GGUF without the SentencePiece tables is refused", "[unit][marian]") {
    // Hyperparameters and vocab.json present, source pieces absent. Loading
    // this and segmenting by some other rule is the failure the converter's
    // strictness exists to prevent; the runtime has to hold the same line.
    m2m100_context* ctx = try_load(write_marian_meta("nospm", true, true));
    CHECK(ctx == nullptr);
    m2m100_free(ctx);

    m2m100_context* ctx2 = try_load(write_marian_meta("novocab", true, false));
    CHECK(ctx2 == nullptr);
    m2m100_free(ctx2);
}

TEST_CASE("marian: runtime accessors tolerate a NULL context", "[unit][marian]") {
    CHECK(m2m100_is_marian(nullptr) == 0);
    CHECK(m2m100_model_beam_size(nullptr) == m2m100_default_beam_size());
    int32_t ids[4];
    CHECK(m2m100_tokenize(nullptr, "x", "de", "en", ids, 4) == -1);
}
