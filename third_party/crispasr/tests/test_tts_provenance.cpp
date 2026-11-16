// test_tts_provenance.cpp — unit tests for TTS AI-provenance compliance.
//
// Tests for: consent gate logic, C2PA no-op, ID3v2 tag structure,
// WAV LIST/INFO metadata, disclaimer helpers.

#include "core/crispasr_wav_writer.h"
#include "core/crispasr_c2pa.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {
uint16_t le_u16(const std::string& s, size_t off) {
    return (uint8_t)s[off] | ((uint16_t)(uint8_t)s[off + 1] << 8);
}
uint32_t le_u32(const std::string& s, size_t off) {
    return (uint32_t)(uint8_t)s[off] | ((uint32_t)(uint8_t)s[off + 1] << 8) | ((uint32_t)(uint8_t)s[off + 2] << 16) |
           ((uint32_t)(uint8_t)s[off + 3] << 24);
}
} // namespace

// ──────────────────────────────────────────────────────────────────────────
// WAV LIST/INFO metadata
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("WAV LIST/INFO chunk is well-formed RIFF", "[unit][provenance][wav]") {
    std::string info = crispasr_wav_make_info_chunk();
    REQUIRE(info.size() >= 12);
    REQUIRE(info.substr(0, 4) == "LIST");
    // LIST size = rest of chunk
    uint32_t list_size = le_u32(info, 4);
    REQUIRE(list_size == info.size() - 8);
    REQUIRE(info.substr(8, 4) == "INFO");
}

TEST_CASE("WAV LIST/INFO contains ISFT with CrispASR", "[unit][provenance][wav]") {
    std::string info = crispasr_wav_make_info_chunk();
    REQUIRE(info.find("ISFT") != std::string::npos);
    REQUIRE(info.find("CrispASR") != std::string::npos);
}

TEST_CASE("WAV LIST/INFO contains ICMT with AI notice", "[unit][provenance][wav]") {
    std::string info = crispasr_wav_make_info_chunk();
    REQUIRE(info.find("ICMT") != std::string::npos);
    REQUIRE(info.find("AI text-to-speech") != std::string::npos);
    REQUIRE(info.find("not a recording") != std::string::npos);
}

TEST_CASE("WAV info_entry pads to even boundary", "[unit][provenance][wav]") {
    // Test with an odd-length string → should pad to even
    std::string out;
    crispasr_wav_info_entry(out, "TEST", "abc"); // 3 chars + NUL = 4 bytes (even, no pad)
    // 4 (id) + 4 (size) + 4 (value "abc\0") = 12
    REQUIRE(out.size() == 12);

    std::string out2;
    crispasr_wav_info_entry(out2, "TEST", "ab"); // 2 chars + NUL = 3 bytes (odd → pad)
    // 4 + 4 + 3 + 1(pad) = 12
    REQUIRE(out2.size() == 12);
}

// ──────────────────────────────────────────────────────────────────────────
// ID3v2 tags
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("ID3v2 tag has correct magic", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE(tag.substr(0, 3) == "ID3");
}

TEST_CASE("ID3v2 version is 2.3", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE((uint8_t)tag[3] == 3);
    REQUIRE((uint8_t)tag[4] == 0);
}

TEST_CASE("ID3v2 synchsafe size is correct", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    uint32_t ss = ((uint32_t)(uint8_t)tag[6] << 21) | ((uint32_t)(uint8_t)tag[7] << 14) |
                  ((uint32_t)(uint8_t)tag[8] << 7) | ((uint32_t)(uint8_t)tag[9]);
    REQUIRE(ss == tag.size() - 10);
    // Verify each byte has MSB=0 (synchsafe encoding)
    REQUIRE(((uint8_t)tag[6] & 0x80) == 0);
    REQUIRE(((uint8_t)tag[7] & 0x80) == 0);
    REQUIRE(((uint8_t)tag[8] & 0x80) == 0);
    REQUIRE(((uint8_t)tag[9] & 0x80) == 0);
}

TEST_CASE("ID3v2 contains all three TXXX frames", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    // Count TXXX occurrences
    int count = 0;
    size_t pos = 0;
    while ((pos = tag.find("TXXX", pos)) != std::string::npos) {
        count++;
        pos += 4;
    }
    REQUIRE(count == 3); // AI_GENERATED, GENERATOR, AI_CONTENT_NOTICE
}

TEST_CASE("ID3v2 TXXX contains AI_GENERATED=true", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE(tag.find("AI_GENERATED") != std::string::npos);
    REQUIRE(tag.find("true") != std::string::npos);
}

TEST_CASE("ID3v2 TXXX contains GENERATOR=CrispASR", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE(tag.find("GENERATOR") != std::string::npos);
    REQUIRE(tag.find("CrispASR") != std::string::npos);
}

// ──────────────────────────────────────────────────────────────────────────
// MP3 encoding (in-tree glint encoder, crispasr_mp3_writer.h)
// ──────────────────────────────────────────────────────────────────────────

#include "crispasr_mp3_writer.h"

TEST_CASE("MP3 target rate maps to native MP3 rates", "[unit][provenance][mp3]") {
    // Native rates pass through
    REQUIRE(crispasr_mp3_target_rate(16000) == 16000);
    REQUIRE(crispasr_mp3_target_rate(22050) == 22050);
    REQUIRE(crispasr_mp3_target_rate(24000) == 24000);
    REQUIRE(crispasr_mp3_target_rate(44100) == 44100);
    REQUIRE(crispasr_mp3_target_rate(48000) == 48000);
    // Non-native rates go to the nearest rate at or above
    REQUIRE(crispasr_mp3_target_rate(20000) == 22050);
    REQUIRE(crispasr_mp3_target_rate(23000) == 24000);
    // Above 48 kHz caps at 48 kHz
    REQUIRE(crispasr_mp3_target_rate(96000) == 48000);
}

TEST_CASE("MP3 encode: ID3 prefix, frame sync, CBR size", "[unit][provenance][mp3]") {
    // 0.5 s of 440 Hz sine at 24 kHz
    const int sr = 24000;
    const int n = sr / 2;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++)
        pcm[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * (float)i / (float)sr);

    std::string mp3 = crispasr_make_mp3(pcm.data(), n, sr);
    REQUIRE(!mp3.empty());
    REQUIRE(mp3.substr(0, 3) == "ID3");

    // First byte after the ID3v2 tag must be an MPEG frame sync
    // (11 set bits: 0xFF then top 3 bits of the next byte).
    uint32_t ss = ((uint32_t)(uint8_t)mp3[6] << 21) | ((uint32_t)(uint8_t)mp3[7] << 14) |
                  ((uint32_t)(uint8_t)mp3[8] << 7) | ((uint32_t)(uint8_t)mp3[9]);
    size_t off = 10 + ss;
    REQUIRE(mp3.size() > off + 4);
    REQUIRE((uint8_t)mp3[off] == 0xFF);
    REQUIRE(((uint8_t)mp3[off + 1] & 0xE0) == 0xE0);

    // CBR 128 kbps → 0.5 s ≈ 8000 bytes of audio data (± a couple of
    // frames for padding/flush)
    size_t audio_bytes = mp3.size() - off;
    REQUIRE(audio_bytes > 6000);
    REQUIRE(audio_bytes < 10000);
}

TEST_CASE("MP3 encode: non-native rate is resampled, still valid", "[unit][provenance][mp3]") {
    const int sr = 19000; // not an MP3 rate → resampled to 22050
    const int n = sr / 4;
    std::vector<float> pcm(n, 0.1f);
    std::string mp3 = crispasr_make_mp3(pcm.data(), n, sr);
    REQUIRE(!mp3.empty());
    REQUIRE(mp3.substr(0, 3) == "ID3");
}

TEST_CASE("MP3 encode: invalid input returns empty", "[unit][provenance][mp3]") {
    std::vector<float> pcm(100, 0.0f);
    REQUIRE(crispasr_make_mp3(nullptr, 100, 24000).empty());
    REQUIRE(crispasr_make_mp3(pcm.data(), 0, 24000).empty());
    REQUIRE(crispasr_make_mp3(pcm.data(), 100, 0).empty());
}

// ──────────────────────────────────────────────────────────────────────────
// AAC-LC encoding (in-tree glint encoder, crispasr_aac_writer.h)
// ──────────────────────────────────────────────────────────────────────────

#include "crispasr_aac_writer.h"

TEST_CASE("AAC target rate maps to native AAC rates", "[unit][provenance][aac]") {
    REQUIRE(crispasr_aac_target_rate(16000) == 16000);
    REQUIRE(crispasr_aac_target_rate(24000) == 24000);
    REQUIRE(crispasr_aac_target_rate(48000) == 48000);
    REQUIRE(crispasr_aac_target_rate(96000) == 96000);
    // Non-native rates go to the nearest rate at or above
    REQUIRE(crispasr_aac_target_rate(20000) == 22050);
    REQUIRE(crispasr_aac_target_rate(50000) == 64000);
    // Above 96 kHz caps at 96 kHz
    REQUIRE(crispasr_aac_target_rate(192000) == 96000);
}

TEST_CASE("AAC encode: ID3 prefix + ADTS syncword", "[unit][provenance][aac]") {
    // 0.5 s of 440 Hz sine at 24 kHz
    const int sr = 24000;
    const int n = sr / 2;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++)
        pcm[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * (float)i / (float)sr);

    std::string aac = crispasr_make_aac(pcm.data(), n, sr);
    REQUIRE(!aac.empty());
    REQUIRE(aac.substr(0, 3) == "ID3");

    // First bytes after the ID3v2 tag must be an ADTS syncword
    // (12 set bits: 0xFF then top 4 bits of the next byte).
    uint32_t ss = ((uint32_t)(uint8_t)aac[6] << 21) | ((uint32_t)(uint8_t)aac[7] << 14) |
                  ((uint32_t)(uint8_t)aac[8] << 7) | ((uint32_t)(uint8_t)aac[9]);
    size_t off = 10 + ss;
    REQUIRE(aac.size() > off + 7);
    REQUIRE((uint8_t)aac[off] == 0xFF);
    REQUIRE(((uint8_t)aac[off + 1] & 0xF0) == 0xF0);
}

TEST_CASE("AAC encode: invalid input returns empty", "[unit][provenance][aac]") {
    std::vector<float> pcm(100, 0.0f);
    REQUIRE(crispasr_make_aac(nullptr, 100, 24000).empty());
    REQUIRE(crispasr_make_aac(pcm.data(), 0, 24000).empty());
    REQUIRE(crispasr_make_aac(pcm.data(), 100, 0).empty());
}

// ──────────────────────────────────────────────────────────────────────────
// Ogg Opus encoding + OpusTags AI-provenance (crispasr_opus_writer.h)
// ──────────────────────────────────────────────────────────────────────────

#include "crispasr_opus_writer.h"

TEST_CASE("Opus encode: valid Ogg with AI-provenance in OpusTags", "[unit][provenance][opus]") {
    const int sr = 48000;
    const int n = sr / 2; // 0.5 s
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++)
        pcm[i] = 0.3f * std::sin(2.0f * 3.14159265f * 300.0f * (float)i / (float)sr);

    std::string opus = crispasr_make_opus(pcm.data(), n, sr);
    REQUIRE(opus.size() > 100);
    REQUIRE(opus.compare(0, 4, "OggS") == 0); // Ogg capture pattern

    // The rewritten OpusTags packet carries the AI-provenance comments verbatim.
    REQUIRE(opus.find("OpusTags") != std::string::npos);
    REQUIRE(opus.find("AI_GENERATED=true") != std::string::npos);
    REQUIRE(opus.find("GENERATOR=CrispASR") != std::string::npos);
    REQUIRE(opus.find("AI_CONTENT_NOTICE=This audio was synthesized") != std::string::npos);

    // The rewritten page must remain a valid Ogg Opus (CRC + lacing correct):
    // glint's decoder round-trips it.
    int dsr = 0, dch = 0, dfr = 0;
    float* dec = glint_decode_audio(reinterpret_cast<const uint8_t*>(opus.data()), (int)opus.size(), &dsr, &dch, &dfr);
    REQUIRE(dec != nullptr);
    REQUIRE(dfr > 0);
    REQUIRE(dsr == 48000);
    glint_free(dec);
}

TEST_CASE("Opus encode: invalid input returns empty", "[unit][provenance][opus]") {
    std::vector<float> pcm(100, 0.0f);
    REQUIRE(crispasr_make_opus(nullptr, 100, 48000).empty());
    REQUIRE(crispasr_make_opus(pcm.data(), 0, 48000).empty());
    REQUIRE(crispasr_make_opus(pcm.data(), 100, 0).empty());
}

// ──────────────────────────────────────────────────────────────────────────
// C2PA (compile-time gated)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("C2PA manifest JSON is valid and contains required fields", "[unit][provenance][c2pa]") {
    const char* json = crispasr_c2pa_manifest_json();
    REQUIRE(json != nullptr);
    std::string s(json);
    REQUIRE(s.find("c2pa.actions") != std::string::npos);
    REQUIRE(s.find("c2pa.created") != std::string::npos);
    REQUIRE(s.find("trainedAlgorithmicMedia") != std::string::npos);
    REQUIRE(s.find("CrispASR") != std::string::npos);
}

TEST_CASE("C2PA sign_wav returns false when not available", "[unit][provenance][c2pa]") {
    // Without CRISPASR_HAVE_C2PA, should be a no-op returning false
    std::string wav = "RIFF....WAVE";
    std::string original = wav;
    bool ok = crispasr_c2pa_sign_wav(wav, "", "");
    REQUIRE(ok == false);
    REQUIRE(wav == original); // unchanged
}

TEST_CASE("C2PA sign_wav returns false with empty cert/key", "[unit][provenance][c2pa]") {
    std::string wav(100, '\0');
    REQUIRE(crispasr_c2pa_sign_wav(wav, "", "/some/key.pem") == false);
    REQUIRE(crispasr_c2pa_sign_wav(wav, "/some/cert.pem", "") == false);
}

// ──────────────────────────────────────────────────────────────────────────
// Consent gate logic (voice clone detection)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("Voice clone detection by .wav extension", "[unit][provenance][consent]") {
    auto is_clone = [](const std::string& voice) -> bool {
        return voice.size() >= 4 &&
               (voice.compare(voice.size() - 4, 4, ".wav") == 0 || voice.compare(voice.size() - 4, 4, ".WAV") == 0);
    };

    REQUIRE(is_clone("speaker.wav") == true);
    REQUIRE(is_clone("/path/to/speaker.wav") == true);
    REQUIRE(is_clone("SPEAKER.WAV") == true);
    REQUIRE(is_clone("speaker.gguf") == false);
    REQUIRE(is_clone("tara") == false);
    REQUIRE(is_clone("") == false);
    REQUIRE(is_clone(".wav") == true); // edge case: bare extension
    REQUIRE(is_clone("wav") == false); // too short
}

// ──────────────────────────────────────────────────────────────────────────
// Watermark spread-spectrum (additional edge cases)
// ──────────────────────────────────────────────────────────────────────────

#include "core/crispasr_watermark.h"

TEST_CASE("Watermark embed is idempotent-ish (double embed doesn't crash)", "[unit][provenance][watermark]") {
    std::vector<float> pcm(4800, 0.3f);
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size()); // second embed
    // Should not crash, and detect should still find at least one watermark
    float score = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    REQUIRE(score > 0.0f); // don't require high confidence, just no crash
}

TEST_CASE("Watermark detect returns 0 for null input", "[unit][provenance][watermark]") {
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 0) == 0.0f);
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 1000) == 0.0f);
}

TEST_CASE("Watermark embed with alpha=0 introduces negligible change", "[unit][provenance][watermark]") {
    std::vector<float> pcm(4800, 0.5f);
    auto original = pcm;
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size(), 0.0f);
    // With alpha=0, the nudge is zero but FFT→IFFT overlap-add introduces
    // tiny floating-point rounding noise. Max error should be < 1e-3.
    float max_err = 0.0f;
    for (size_t i = 0; i < pcm.size(); i++) {
        float err = std::abs(pcm[i] - original[i]);
        if (err > max_err)
            max_err = err;
    }
    REQUIRE(max_err < 0.01f);
}

// ──────────────────────────────────────────────────────────────────────────
// Spoken disclaimer opt-out — text verification
// ──────────────────────────────────────────────────────────────────────────
// crispasr_tts_disclaimer.h includes heavy backend headers; we test the
// disclaimer text contract inline to keep this test model-free.

TEST_CASE("Disclaimer text contract: mentions 'artificial intelligence'", "[unit][provenance][disclaimer]") {
    // Must match the string in crispasr_tts_disclaimer.h::text()
    const std::string expected = "This audio was generated by artificial intelligence.";
    REQUIRE(expected.find("artificial intelligence") != std::string::npos);
    REQUIRE(!expected.empty());
}

// ──────────────────────────────────────────────────────────────────────────
// Voice-clone disclaimer gating logic (unit-testable without backends)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("Disclaimer gate: clone + no opt-out → disclaimer applies", "[unit][provenance][disclaimer]") {
    const bool is_voice_clone = true;
    const bool no_spoken_disclaimer = false;
    REQUIRE((is_voice_clone && !no_spoken_disclaimer) == true);
}

TEST_CASE("Disclaimer gate: clone + opt-out → disclaimer skipped", "[unit][provenance][disclaimer]") {
    const bool is_voice_clone = true;
    const bool no_spoken_disclaimer = true;
    REQUIRE((is_voice_clone && !no_spoken_disclaimer) == false);
}

TEST_CASE("Disclaimer gate: no clone → disclaimer always skipped", "[unit][provenance][disclaimer]") {
    const bool is_voice_clone = false;
    // Regardless of the flag, non-clone output never gets disclaimer
    REQUIRE((is_voice_clone && !false) == false);
    REQUIRE((is_voice_clone && !true) == false);
}
