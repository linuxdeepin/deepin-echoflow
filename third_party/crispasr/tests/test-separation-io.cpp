// test-separation-io.cpp — unit tests for the shared source-separation output
// surface (§248): stem output-path naming, --stems selection, and the
// multi-channel WAV writer. Pure string/byte logic; no model, no audio.

#include "core/crispasr_wav_writer.h"
#include "core/separation_io.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

TEST_CASE("stem output path: alongside input, extension stripped", "[unit][separation]") {
    REQUIRE(crispasr_stem_output_path("/music/song.flac", "vocals", "") == "/music/song_vocals.wav");
    REQUIRE(crispasr_stem_output_path("/music/song.wav", "Drums", "") == "/music/song_drums.wav");
    REQUIRE(crispasr_stem_output_path("song.mp3", "other", "") == "song_other.wav");
}

TEST_CASE("stem output path: explicit output dir, trailing slash tolerant", "[unit][separation]") {
    REQUIRE(crispasr_stem_output_path("/in/song.flac", "vocals", "/out") == "/out/song_vocals.wav");
    REQUIRE(crispasr_stem_output_path("/in/song.flac", "vocals", "/out/") == "/out/song_vocals.wav");
    REQUIRE(crispasr_stem_output_path("/in/song.flac", "bass", "rel/dir") == "rel/dir/song_bass.wav");
}

TEST_CASE("stem output path: multi-dot + no-extension + dotfile", "[unit][separation]") {
    REQUIRE(crispasr_stem_output_path("/a/my.song.v2.flac", "vocals", "") == "/a/my.song.v2_vocals.wav");
    REQUIRE(crispasr_stem_output_path("/a/noext", "vocals", "") == "/a/noext_vocals.wav");
    // leading-dot name has no extension to strip
    REQUIRE(crispasr_stem_output_path("/a/.hidden", "vocals", "") == "/a/.hidden_vocals.wav");
}

TEST_CASE("stem output path: custom extension", "[unit][separation]") {
    REQUIRE(crispasr_stem_output_path("/m/song.wav", "vocals", "", "flac") == "/m/song_vocals.flac");
}

TEST_CASE("stem selection: empty/all selects everything", "[unit][separation]") {
    REQUIRE(crispasr_stem_selected("", "vocals"));
    REQUIRE(crispasr_stem_selected("all", "drums"));
    REQUIRE(crispasr_stem_selected("  ", "bass"));
}

TEST_CASE("stem selection: csv is case-insensitive and whitespace-tolerant", "[unit][separation]") {
    REQUIRE(crispasr_stem_selected("vocals,drums", "vocals"));
    REQUIRE(crispasr_stem_selected("vocals, Drums", "drums"));
    REQUIRE(crispasr_stem_selected("VOCALS", "vocals"));
    REQUIRE_FALSE(crispasr_stem_selected("vocals,drums", "bass"));
    REQUIRE_FALSE(crispasr_stem_selected("vocals", "vocal")); // no substring match
}

TEST_CASE("multi-channel WAV: header fields for stereo", "[unit][separation]") {
    // 3 stereo frames -> 6 interleaved samples.
    const float pcm[6] = {0.0f, 0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
    const std::string w = crispasr_make_wav_int16_interleaved(pcm, 3, 2, 44100);

    REQUIRE(w.size() == 44 + 3 * 2 * 2); // header + 3 frames * 2ch * 2 bytes
    REQUIRE(w.compare(0, 4, "RIFF") == 0);
    REQUIRE(w.compare(8, 4, "WAVE") == 0);

    auto u16 = [&](size_t o) { return (uint16_t)((uint8_t)w[o] | ((uint8_t)w[o + 1] << 8)); };
    auto u32 = [&](size_t o) {
        return (uint32_t)((uint8_t)w[o] | ((uint8_t)w[o + 1] << 8) | ((uint8_t)w[o + 2] << 16) |
                          ((uint8_t)w[o + 3] << 24));
    };
    REQUIRE(u16(22) == 2);            // channels
    REQUIRE(u32(24) == 44100u);       // sample rate
    REQUIRE(u16(32) == 4);            // block align = 2ch * 2 bytes
    REQUIRE(u16(34) == 16);           // bits per sample
    REQUIRE(u32(40) == 3u * 2u * 2u); // data size

    // full-scale +1.0 clamps to 32767, -1.0 to -32767; frame 1 L/R.
    auto s16 = [&](size_t o) { return (int16_t)u16(o); };
    REQUIRE(s16(44 + 4) == 32767);  // 3rd sample = 1.0
    REQUIRE(s16(44 + 6) == -32767); // 4th sample = -1.0
}

TEST_CASE("multi-channel WAV: no AI-provenance INFO chunk (stems are user audio)", "[unit][separation]") {
    const float pcm[2] = {0.1f, -0.1f};
    const std::string w = crispasr_make_wav_int16_interleaved(pcm, 1, 2, 44100);
    // The mono AI writer appends a LIST/INFO chunk; the separation writer must not.
    REQUIRE(w.find("LIST") == std::string::npos);
    REQUIRE(w.find("ISFT") == std::string::npos);
}

TEST_CASE("separation view -> wav: out-of-range index is empty", "[unit][separation]") {
    const float ch[2] = {0.2f, -0.2f};
    const float* srcs[1] = {ch};
    const char* names[1] = {"vocals"};
    crispasr_separation_view v;
    v.n_sources = 1;
    v.n_channels = 2;
    v.n_frames = 1;
    v.sample_rate = 44100;
    v.sources = srcs;
    v.source_names = names;

    REQUIRE_FALSE(crispasr_stem_to_wav(v, 0).empty());
    REQUIRE(crispasr_stem_to_wav(v, 1).empty());
    REQUIRE(crispasr_stem_to_wav(v, -1).empty());
}
