// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "../third_party/crispasr/src/core/audio_encoder_cache.h"
#include "../third_party/crispasr/src/core/qwen3_stream_model.h"
#include "CrispSession.h"

#include <QScopeGuard>
#include <QTest>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

using core_audio_encoder_cache::State;

namespace {
std::vector<float> melInput(int frames, int n_mels = 128)
{
    std::vector<float> result(static_cast<size_t>(frames) * n_mels);
    for (int feature = 0; feature < n_mels; ++feature) {
        for (int frame = 0; frame < frames; ++frame) {
            result[static_cast<size_t>(feature) * frames + frame] =
                0.5f + static_cast<float>(0.2 * std::sin((feature * 17 + frame) * 0.03));
        }
    }
    return result;
}
void fillComputed(State& state)
{
    for (auto& block : state.blocks) block.stem = {1, 2, 3};
    for (auto& region : state.regions) region.output = {4, 5, 6};
}
struct Environment {
    QByteArray key;
    QByteArray value;
    bool present;
    explicit Environment(const char* name)
        : key(name), value(qgetenv(name)), present(qEnvironmentVariableIsSet(name)) {}
    ~Environment() { if (present) qputenv(key, value); else qunsetenv(key); }
};
}

class TestEncoderCache : public QObject {
    Q_OBJECT
private slots:
    void denseAppendInvalidatesAttentionAndPreservesStems();
    void independentWindowsReuseCompleteEncoder();
    void changedNormalizationInvalidatesHistoricalBlocks();
    void partialTailLengthInvalidatesEvenWithIdenticalPadding();
    void shrinkAndResetDropOldState();
    void invalidInputIsRejected();
    void realEncoderParity_data();
    void realEncoderParity();
    void realPcmAndTranscriptParity();
    void realNativeStreamCacheLifecycle();
};

void TestEncoderCache::denseAppendInvalidatesAttentionAndPreservesStems()
{
    State state;
    auto mel = melInput(120, 2);
    state.prepare(mel.data(), 2, 120, 100, 0);
    QCOMPARE(state.blocks.size(), size_t(2));
    QCOMPARE(state.regions.size(), size_t(1));
    fillComputed(state);
    state.prepare(mel.data(), 2, 120, 100, 0);
    QVERIFY(!state.regions[0].output.empty());
    mel = melInput(160, 2);
    state.prepare(mel.data(), 2, 160, 100, 0);
    QVERIFY(!state.blocks[0].stem.empty());
    QVERIFY(state.blocks[1].stem.empty());
    QVERIFY(state.regions[0].output.empty());
    fillComputed(state);
    mel = melInput(201, 2);
    state.prepare(mel.data(), 2, 201, 100, 0);
    QCOMPARE(state.blocks.size(), size_t(3));
    QVERIFY(!state.blocks[0].stem.empty());
    QVERIFY(state.regions[0].output.empty());
}

void TestEncoderCache::independentWindowsReuseCompleteEncoder()
{
    State state;
    auto mel = melInput(850, 2);
    state.prepare(mel.data(), 2, 850, 100, 8);
    fillComputed(state);
    QCOMPARE(state.regions.size(), size_t(2));
    mel = melInput(901, 2);
    state.prepare(mel.data(), 2, 901, 100, 8);
    QVERIFY(!state.regions[0].output.empty());
    QVERIFY(state.regions[1].output.empty());
    for (size_t index = 0; index < 8; ++index) QVERIFY(!state.blocks[index].stem.empty());
    fillComputed(state);
    mel[5] += 0.1f;
    state.prepare(mel.data(), 2, 901, 100, 8);
    QVERIFY(state.regions[0].output.empty());
    QVERIFY(!state.regions[1].output.empty());
    QVERIFY(state.blocks[0].stem.empty());
    QVERIFY(!state.blocks[1].stem.empty());
}

void TestEncoderCache::changedNormalizationInvalidatesHistoricalBlocks()
{
    State state;
    auto mel = melInput(901, 2);
    state.prepare(mel.data(), 2, 901, 100, 8);
    fillComputed(state);
    for (auto& value : mel) value -= 0.1f;
    state.prepare(mel.data(), 2, 901, 100, 8);
    for (const auto& block : state.blocks) QVERIFY(block.stem.empty());
    for (const auto& region : state.regions) QVERIFY(region.output.empty());
}

void TestEncoderCache::partialTailLengthInvalidatesEvenWithIdenticalPadding()
{
    State state;
    std::vector<float> zeros(202, 0.0f);
    state.prepare(zeros.data(), 2, 100, 100, 0);
    fillComputed(state);
    state.prepare(zeros.data(), 2, 101, 100, 0);
    QVERIFY(!state.blocks[0].stem.empty());
    QVERIFY(state.regions[0].output.empty());
    fillComputed(state);
    state.prepare(zeros.data(), 2, 99, 100, 0);
    QCOMPARE(state.blocks.size(), size_t(1));
    QVERIFY(!state.blocks[0].stem.empty());
    QVERIFY(state.regions[0].output.empty());
}

void TestEncoderCache::shrinkAndResetDropOldState()
{
    State state;
    auto mel = melInput(901, 2);
    state.prepare(mel.data(), 2, 901, 100, 8);
    fillComputed(state);
    QVERIFY(state.bytes() > 0);
    mel = melInput(50, 2);
    state.prepare(mel.data(), 2, 50, 100, 8);
    QCOMPARE(state.blocks.size(), size_t(1));
    QCOMPARE(state.regions.size(), size_t(1));
    QVERIFY(state.regions[0].output.empty());
    fillComputed(state);
    state.prepare(mel.data(), 2, 50, 100, 0);
    QVERIFY(state.blocks[0].stem.empty());
    QVERIFY(state.regions[0].output.empty());
    state.clear();
    QVERIFY(state.blocks.empty());
    QVERIFY(state.regions.empty());
    QCOMPARE(state.bytes(), size_t(0));
}

void TestEncoderCache::invalidInputIsRejected()
{
    State state;
    auto mel = melInput(100, 2);
    QVERIFY_EXCEPTION_THROWN(state.prepare(nullptr, 2, 100, 100, 0), std::invalid_argument);
    QVERIFY_EXCEPTION_THROWN(state.prepare(mel.data(), 0, 100, 100, 0), std::invalid_argument);
    QVERIFY_EXCEPTION_THROWN(state.prepare(mel.data(), 2, 0, 100, 0), std::invalid_argument);
    QVERIFY_EXCEPTION_THROWN(state.prepare(mel.data(), 2, 100, 0, 0), std::invalid_argument);
    QVERIFY_EXCEPTION_THROWN(state.prepare(mel.data(), 2, 100, 100, -1), std::invalid_argument);
    QCOMPARE(crisp_audio_encoder_cache_get_stats(nullptr).cached_bytes, size_t(0));
    crisp_audio_encoder_cache_reset(nullptr);
    crisp_audio_encoder_cache_free(nullptr);
}

void TestEncoderCache::realEncoderParity_data()
{
    QTest::addColumn<bool>("windowed");
    QTest::addColumn<bool>("keepPadding");
    QTest::newRow("dense") << false << false;
    QTest::newRow("dense-legacy-padding") << false << true;
    QTest::newRow("windowed") << true << false;
}

void TestEncoderCache::realEncoderParity()
{
    QFETCH(bool, windowed);
    QFETCH(bool, keepPadding);
    const auto model = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (model.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL for encoder tensor parity");
    Environment windowSetting("CRISP_AUDIO_WINDOWED_ATTN");
    Environment paddingSetting("CRISP_AUDIO_KEEP_PAD_FRAMES");
    qputenv("CRISP_AUDIO_WINDOWED_ATTN", windowed ? "1" : "0");
    qputenv("CRISP_AUDIO_KEEP_PAD_FRAMES", keepPadding ? "1" : "0");
    auto params = crisp_audio_params_default();
    params.n_threads = 2;
    params.use_gpu = false;
    params.verbosity = 0;
    params.tensor_prefix = "audio.";
    params.meta_prefix = "qwen3asr.audio.";
    std::unique_ptr<crisp_audio_context, decltype(&crisp_audio_free)> context(
        crisp_audio_init_from_file(model.constData(), &params), &crisp_audio_free);
    QVERIFY(context != nullptr);
    std::unique_ptr<crisp_audio_encoder_cache, decltype(&crisp_audio_encoder_cache_free)> cache(
        crisp_audio_encoder_cache_new(), &crisp_audio_encoder_cache_free);
    QVERIFY(cache != nullptr);
    std::vector<int> lengths = windowed ? std::vector<int>{799, 801, 820, 901, 901, 100}
                                         : std::vector<int>{99, 100, 101, 201, 250, 250, 99};
    bool sawStemReuse = false;
    bool sawRegionReuse = false;
    double baselineMs = 0, cachedMs = 0;
    for (int length : lengths) {
        const auto mel = melInput(length);
        int frames = 0, dim = 0, cachedFrames = 0, cachedDim = 0;
        auto start = std::chrono::steady_clock::now();
        std::unique_ptr<float, decltype(&std::free)> full(crisp_audio_encode(
            context.get(), mel.data(), 128, length, &frames, &dim), &std::free);
        baselineMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        QVERIFY(full != nullptr);
        start = std::chrono::steady_clock::now();
        std::unique_ptr<float, decltype(&std::free)> cached(crisp_audio_encode_cached(
            context.get(), cache.get(), mel.data(), 128, length, &cachedFrames, &cachedDim), &std::free);
        cachedMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        QVERIFY(cached != nullptr);
        QCOMPARE(frames, cachedFrames);
        QCOMPARE(dim, cachedDim);
        double squared = 0, norm = 0, maxError = 0;
        for (size_t index = 0; index < static_cast<size_t>(frames) * dim; ++index) {
            const double delta = full.get()[index] - cached.get()[index];
            squared += delta * delta;
            norm += static_cast<double>(full.get()[index]) * full.get()[index];
            maxError = std::max(maxError, std::abs(delta));
        }
        const double relative = std::sqrt(squared / std::max(norm, 1e-20));
        qInfo("encoder length=%d max_error=%.8f relative_rms=%.8f", length, maxError, relative);
        QVERIFY2(relative < 0.0001, "cached encoder differs from full encoder");
        QVERIFY(maxError < 0.005);
        const auto stats = crisp_audio_encoder_cache_get_stats(cache.get());
        sawStemReuse |= stats.reused_stem_chunks > 0;
        sawRegionReuse |= stats.reused_encoder_regions > 0;
        QCOMPARE(stats.total_chunks, 1 + (length - 1) / 100);
        QCOMPARE(stats.window_chunks, windowed ? 8 : 0);
        QVERIFY(stats.cached_bytes > 0);
        QCOMPARE(stats.reused_stem_chunks + stats.computed_stem_chunks, stats.total_chunks);
        if (windowed && length == 820) {
            QCOMPARE(stats.reused_encoder_regions, 1);
            QCOMPARE(stats.computed_encoder_regions, 1);
        }
    }
    QVERIFY(sawStemReuse);
    QVERIFY(sawRegionReuse);
    qInfo("encoder totals full_ms=%.3f cached_ms=%.3f", baselineMs, cachedMs);
    // Changing attention semantics on the same model must clear old regions.
    qputenv("CRISP_AUDIO_WINDOWED_ATTN", windowed ? "0" : "1");
    const auto switchedMel = melInput(100);
    int switchedFrames = 0, switchedDim = 0, fullFrames = 0, fullDim = 0;
    std::unique_ptr<float, decltype(&std::free)> switched(crisp_audio_encode_cached(
        context.get(), cache.get(), switchedMel.data(), 128, 100, &switchedFrames, &switchedDim), &std::free);
    std::unique_ptr<float, decltype(&std::free)> full(crisp_audio_encode(
        context.get(), switchedMel.data(), 128, 100, &fullFrames, &fullDim), &std::free);
    QVERIFY(switched != nullptr);
    QVERIFY(full != nullptr);
    QCOMPARE(switchedFrames, fullFrames);
    QCOMPARE(switchedDim, fullDim);
    const auto switchedStats = crisp_audio_encoder_cache_get_stats(cache.get());
    QCOMPARE(switchedStats.reused_stem_chunks, 0);
    QCOMPARE(switchedStats.reused_encoder_regions, 0);
    QCOMPARE(switchedStats.window_chunks, windowed ? 0 : 8);
    QVERIFY(std::equal(switched.get(), switched.get() + static_cast<size_t>(switchedFrames) * switchedDim,
                       full.get()));
    // A fresh model cannot consume cached tensors from another context.
    std::unique_ptr<crisp_audio_context, decltype(&crisp_audio_free)> other(
        crisp_audio_init_from_file(model.constData(), &params), &crisp_audio_free);
    QVERIFY(other != nullptr);
    std::unique_ptr<float, decltype(&std::free)> otherResult(crisp_audio_encode_cached(
        other.get(), cache.get(), switchedMel.data(), 128, 100, &switchedFrames, &switchedDim), &std::free);
    QVERIFY(otherResult != nullptr);
    QCOMPARE(crisp_audio_encoder_cache_get_stats(cache.get()).reused_stem_chunks, 0);
    QCOMPARE(crisp_audio_encoder_cache_get_stats(cache.get()).reused_encoder_regions, 0);
    QVERIFY(std::equal(otherResult.get(), otherResult.get() + static_cast<size_t>(switchedFrames) * switchedDim,
                       full.get()));
    crisp_audio_encoder_cache_reset(cache.get());
    QCOMPARE(crisp_audio_encoder_cache_get_stats(cache.get()).cached_bytes, size_t(0));
}

void TestEncoderCache::realPcmAndTranscriptParity()
{
    using namespace core_qwen3_stream_model;
    const auto model = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (model.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL for PCM and transcript cache parity");
    const auto wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    QVERIFY(!wav.isEmpty());
    const auto clip = echoflow::CrispSession::readWavF32(wav.toStdString());
    QVERIFY(clip.size() >= size_t(48000));
    auto params = qwen3_asr_context_default_params();
    params.n_threads = 2;
    params.use_gpu = false;
    params.verbosity = 0;
    std::unique_ptr<qwen3_asr_context, decltype(&qwen3_asr_free)> context(
        qwen3_asr_init_from_file(model.constData(), params), &qwen3_asr_free);
    QVERIFY(context != nullptr);
    std::unique_ptr<crisp_audio_encoder_cache, decltype(&crisp_audio_encoder_cache_free)> cache(
        crisp_audio_encoder_cache_new(), &crisp_audio_encoder_cache_free);
    QVERIFY(cache != nullptr);
    PrefixKvCache prefix;
    bool sawReuse = false;
    for (int count : {8000, 16000, 24000, 32000, 48000}) {
        std::vector<float> audio(clip.begin(), clip.begin() + count);
        bool ok = false;
        const auto reference = generate(context.get(), false, audio, "language Korean<asr_text>", 64, &ok);
        QVERIFY(ok);
        DecodeMetrics metrics;
        const auto cached = generate(context.get(), false, audio, "language Korean<asr_text>", 64,
                                      &ok, &prefix, &metrics, false, cache.get());
        QVERIFY(ok);
        QCOMPARE(cached, reference);
        sawReuse |= metrics.stem_reused > 0;
        QCOMPARE(metrics.encoder_window_chunks, 0);
    }
    QVERIFY(sawReuse);
    // A loud revised input can invalidate all previous normalized Mel blocks.
    auto revised = std::vector<float>(clip.begin(), clip.begin() + 48000);
    for (size_t index = 32000; index < revised.size(); ++index)
        revised[index] = index % 2 ? 0.95f : -0.95f;
    bool ok = false;
    const auto expected = generate(context.get(), false, revised, "language Korean<asr_text>", 64, &ok);
    QVERIFY(ok);
    QCOMPARE(generate(context.get(), false, revised, "language Korean<asr_text>", 64,
        &ok, &prefix, nullptr, false, cache.get()), expected);
    QVERIFY(ok);
}

void TestEncoderCache::realNativeStreamCacheLifecycle()
{
    const auto model = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (model.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL for stream cache lifecycle");
    const auto wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    QVERIFY(!wav.isEmpty());
    const auto clip = echoflow::CrispSession::readWavF32(wav.toStdString());
    QVERIFY(clip.size() >= size_t(48000));
    Environment setting("CRISPASR_QWEN3_STREAM_ENCODER_CACHE");
    echoflow::CrispSession session(model.toStdString(), "qwen3", 2);
    QVERIFY(session.isLoaded());
    session.setLanguage("ko");
    std::vector<std::vector<std::string>> expected;
    for (bool useCache : {false, true}) {
        qputenv("CRISPASR_QWEN3_STREAM_ENCODER_CACHE", useCache ? "1" : "0");
        for (int recording = 0; recording < 2; ++recording) {
            QVERIFY(session.openStream(500, 500, 800));
            const int length = recording == 0 ? 48000 : 24000;
            std::vector<std::string> updates;
            for (int offset = 0; offset < length; offset += 8000) {
                session.feedStream(clip.data() + offset, 8000);
                updates.push_back(session.streamText());
            }
            updates.push_back(session.flushStream());
            if (!useCache) expected.push_back(updates);
            else QCOMPARE(updates, expected[recording]);
            session.closeStream();
            // Discard an unflushed stream. Its cache must not survive into
            // the next recording, even though model weights and prompt KV do.
            QVERIFY(session.openStream(500, 500, 800));
            session.feedStream(clip.data() + 16000, 8000);
            session.closeStream();
        }
    }
}

QTEST_GUILESS_MAIN(TestEncoderCache)
#include "test_encoder_cache.moc"
