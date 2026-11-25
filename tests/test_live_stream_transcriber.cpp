// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LiveStreamTranscriber.h"
#include "AsrHotwords.h"
#include "CrispSession.h"
#include "CrispLiveVoicePipeline.h"
#include "SileroVadBackend.h"

#include <QTest>
#include <QFile>
#include <QTemporaryDir>
#include <QScopeGuard>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace echoflow;

namespace {

struct FakeStreamDecoder {
    std::vector<std::vector<int16_t>> audio;
    std::vector<std::string> finals;
    std::string partial = "preview";
    size_t flushes = 0;
    size_t closes = 0;
    bool open = false;
    bool failOpen = false;

    LiveStreamDecoder callbacks()
    {
        return {
            [this] {
                if (failOpen) throw std::runtime_error("stream open failed");
                if (open) throw std::runtime_error("previous stream not closed");
                open = true;
                audio.emplace_back();
            },
            [this](const int16_t* samples, size_t count) {
                if (!open) throw std::runtime_error("feed on closed stream");
                audio.back().insert(audio.back().end(), samples, samples + count);
                return partial;
            },
            [this] {
                if (!open) throw std::runtime_error("flush on closed stream");
                const size_t index = flushes++;
                return index < finals.size() ? finals[index] : partial;
            },
            [this] {
                if (!open) throw std::runtime_error("double close");
                open = false;
                ++closes;
            }
        };
    }
};

struct PauseAwareDecoder : FakeStreamDecoder {
    bool quiet = false;
    size_t liveFeeds = 0;
    size_t checkpoints = 0;
    std::string checkpointText = "corrected phrase";

    LiveStreamDecoder callbacks()
    {
        auto result = FakeStreamDecoder::callbacks();
        const auto feed = result.feed;
        result.feed = [this, feed](const int16_t* samples, size_t count) {
            if (!quiet) ++liveFeeds;
            return feed(samples, count);
        };
        result.setSilence = [this](bool value) { quiet = value; };
        result.checkpoint = [this] {
            if (!open || !quiet) throw std::runtime_error("checkpoint requires an open paused stream");
            ++checkpoints;
            partial = checkpointText;
            return partial;
        };
        return result;
    }
};

std::vector<int16_t> speech(size_t count)
{
    std::vector<int16_t> audio(count);
    for (size_t i = 0; i < count; ++i) {
        audio[i] = (i % 2 == 0) ? int16_t(3000 + i % 500) : int16_t(-3000 - i % 500);
    }
    return audio;
}

LiveStreamConfig shortSegments()
{
    LiveStreamConfig config;
    config.maxSegmentMs = 1000;
    config.endSilenceMs = 300;
    return config;
}

} // namespace

class TestLiveStreamTranscriber : public QObject {
    Q_OBJECT
private slots:
    void pipelineHandlesBacklogWhenModelAvailable_data();
    void pipelineHandlesBacklogWhenModelAvailable();
    void previewSchedulerAdaptsWithoutLosingLookahead();
    void segmentBoundaryDefersPreviewBeforeFlush();
    void previewSchedulerRefreshesDuringBacklogAndSkipsFinalPreviews();
    void previewDefersLargeRewriteUntilNextDecode();
    void pauseCheckpointBypassesDeferredRewrite();
    void stopSuppressesAnInFlightCheckpoint();
    void batchedQuietTailPreservesLateResumedVoice_data();
    void batchedQuietTailPreservesLateResumedVoice();
    void batchedEndpointPreservesNextOnset_data();
    void batchedEndpointPreservesNextOnset();
    void pauseCheckpointRetractionIsPreserved();
    void skippedCheckpointStillFinalizesAllSpeech();
    void finalVadBatchSkipsCheckpoint();
    void quietTailRefreshesOnlyOnce();
    void longPauseConfirmsCheckpointWithoutDecodingAgain();
    void checkpointRetainsWeakTail_data();
    void checkpointRetainsWeakTail();
    void checkpointIgnoresSteadyNoiseAndDc();
    void resumedSpeechPreservesGapAndRefreshesAgain();
    void stopBeforeShortPauseStillDecodesTail();
    void shortBurstNeverTriggersPauseCheckpoint();
    void earlyEndpointDoesNotCheckpointThenFlush();
    void cancelDropsPendingPause();
    void defaultStreamKeepsSentencePastThreeSeconds();
    void briefPauseKeepsCurrentSentenceOpen();
    void confirmsAtPauseAndKeepsPrefixStable();
    void shortBurstAtPauseIsDiscardedAndRetracted();
    void shortBurstAtStopIsDiscarded();
    void shortBurstDoesNotChangeConfirmedText();
    void validPauseAtHardLimitIsConfirmed();
    void prefersPauseNearSegmentLimit_data();
    void prefersPauseNearSegmentLimit();
    void nearLimitPauseAfterCheckpoint_data();
    void nearLimitPauseAfterCheckpoint();
    void longSpeechHasBoundedStreamsWithoutLosingSamples();
    void partitionsDoNotChangeBoundaries();
    void nonFrameAlignedLimitKeepsEverySample();
    void forcedCutUsesQuietSpanWithoutLosingSamples();
    void forcedCutPreservesShortContinuation_data();
    void forcedCutPreservesShortContinuation();
    void forcedCutKeepsContinuousQuietSpeech();
    void forcedCutKeepsFadingTail();
    void emptyConfirmedSegmentRetriesOnce();
    void emptyUnconfirmedSegmentDoesNotRetry();
    void hotwordsFilterOnlyAFullListEcho();
    void silenceDoesNotDecodeOrAccumulate();
    void idleLeadInIsBounded();
    void finishKeepsShortTailAfterForcedBoundary();
    void shortPauseAfterHardBoundaryKeepsValidTail_data();
    void shortPauseAfterHardBoundaryKeepsValidTail();
    void fullPauseAfterHardBoundaryRejectsNewShortBurst();
    void injectedVadDetectsQuietSpeech();
    void batchedVadReportsShortUtteranceAtFinish();
    void naturalRepetitionsArePreserved();
    void accentedWordsKeepTheirSeparator();
    void cancelDoesNotFlushOrPublish();
    void finalTextAndFinishAreIdempotent();
    void reopenErrorsPropagate();
    void liveStreamConfigKeepsDefaultLimitAndRejectsOverflow();
    void longMinimumSpeechExtendsSegmentLimit_data();
    void longMinimumSpeechExtendsSegmentLimit();
    void invalidConfigIsRejected();
    void realModelKeepsConfirmedTextAcrossStreams();
};

void TestLiveStreamTranscriber::pipelineHandlesBacklogWhenModelAvailable_data()
{
    QTest::addColumn<bool>("denoise");
    QTest::newRow("raw") << false;
    QTest::newRow("rnnoise") << true;
}

void TestLiveStreamTranscriber::pipelineHandlesBacklogWhenModelAvailable()
{
    QFETCH(bool, denoise);
    const auto model = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (model.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL to test capture backpressure");
    const auto wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    QVERIFY(!wav.isEmpty());
    const auto clip = CrispSession::readWavF32(wav.toStdString());
    QVERIFY(!clip.empty());
    QTemporaryDir directory;
    QVERIFY(directory.isValid());
    const auto pcmPath = directory.filePath("capture.pcm");
    QFile audio(pcmPath);
    QVERIFY(audio.open(QIODevice::WriteOnly));
    std::vector<int16_t> pcm(16000 * 12);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = static_cast<int16_t>(clip[i % clip.size()] * 32767);
    const auto bytes = static_cast<qint64>(pcm.size() * sizeof(int16_t));
    QCOMPARE(audio.write(reinterpret_cast<const char*>(pcm.data()), bytes), bytes);
    audio.close();
    QFile program(directory.filePath("pw-record"));
    QVERIFY(program.open(QIODevice::WriteOnly));
    program.write(R"PY(#!/usr/bin/env python3
import signal
import sys
import time
signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
with open(sys.argv[sys.argv.index('--target') + 1], 'rb') as source:
    sys.stdout.buffer.write(source.read())
    sys.stdout.buffer.flush()
while True:
    time.sleep(0.01)
)PY");
    program.close();
    QVERIFY(program.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner));
    const auto previousPath = qgetenv("PATH");
    const auto restorePath = qScopeGuard([previousPath] { qputenv("PATH", previousPath); });
    qputenv("PATH", directory.path().toLocal8Bit() + ':' + previousPath);
    Config config = Config::defaultConfig();
    config.crispModelPath = model.toStdString();
    config.crispThreads = 2;
    config.pwRecord.source = pcmPath.toStdString();
    config.language = "ko";
    config.noiseSuppression = denoise;
    std::atomic<int> levels{0};
    std::vector<std::string> confirmed;
    CrispLiveVoicePipeline pipeline(config);
    pipeline.setAudioLevelCallback([&](const AudioLevelFrame&) { ++levels; });
    pipeline.setConfirmedTextCallback([&](const std::string& text) { confirmed.push_back(text); });
    pipeline.start();
    // The recorder writes 12 seconds immediately, exceeding the old 5 s cap.
    QTRY_COMPARE_WITH_TIMEOUT(levels.load(), 240, 5000);
    const auto result = pipeline.finish();
    QVERIFY(!result.empty());
    QVERIFY(!confirmed.empty());
    for (size_t i = 1; i < confirmed.size(); ++i) {
        QVERIFY(confirmed[i].rfind(confirmed[i - 1], 0) == 0);
    }
    QCOMPARE(result, confirmed.back());
    if (!denoise) {
        // Reuse this pipeline/model, but never its text, stream caches or PCM.
        const auto shortBytes = static_cast<qint64>(std::min<size_t>(clip.size(), 32000) * sizeof(int16_t));
        QVERIFY(audio.open(QIODevice::WriteOnly | QIODevice::Truncate));
        QCOMPARE(audio.write(reinterpret_cast<const char*>(pcm.data()), shortBytes), shortBytes);
        audio.close();
        const int expectedLevels = static_cast<int>(shortBytes / 2 / 800);
        levels = 0;
        confirmed.clear();
        pipeline.start();
        QTRY_COMPARE_WITH_TIMEOUT(levels.load(), expectedLevels, 5000);
        const auto warmResult = pipeline.finish();
        QVERIFY(!warmResult.empty());
        QVERIFY(!confirmed.empty());
        QCOMPARE(warmResult, confirmed.back());
        levels = 0;
        pipeline.start();
        QTRY_COMPARE_WITH_TIMEOUT(levels.load(), expectedLevels, 5000);
        pipeline.cancel();
        levels = 0;
        confirmed.clear();
        pipeline.start();
        QTRY_COMPARE_WITH_TIMEOUT(levels.load(), expectedLevels, 5000);
        const auto afterCancel = pipeline.finish();
        QVERIFY(!afterCancel.empty());
        QCOMPARE(afterCancel, confirmed.back());

        // Also pace the source like real capture: previews must arrive while
        // recording, rather than only when finish() drains the queue.
        QVERIFY(audio.open(QIODevice::WriteOnly | QIODevice::Truncate));
        const qint64 pacedBytes = 16000 * 6 * sizeof(int16_t);
        QCOMPARE(audio.write(reinterpret_cast<const char*>(pcm.data()), pacedBytes), pacedBytes);
        audio.close();
        QVERIFY(program.open(QIODevice::WriteOnly | QIODevice::Truncate));
        program.write(R"PY(#!/usr/bin/env python3
import signal
import sys
import time
signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
with open(sys.argv[sys.argv.index('--target') + 1], 'rb') as source:
    while chunk := source.read(3200):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        time.sleep(0.1)
while True:
    time.sleep(0.01)
)PY");
        program.close();
        levels = 0;
        confirmed.clear();
        std::atomic<int> previews{0};
        std::atomic<long long> firstPreviewMs{-1};
        const auto recordingStarted = std::chrono::steady_clock::now();
        pipeline.setPartialTextCallback([&](const std::string& text) {
            if (text.empty()) return;
            ++previews;
            long long unset = -1;
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - recordingStarted).count();
            firstPreviewMs.compare_exchange_strong(unset, elapsed);
        });
        const auto stopPacedCapture = qScopeGuard([&] {
            pipeline.cancel();
            pipeline.setPartialTextCallback({});
        });
        pipeline.start();
        QTRY_COMPARE_WITH_TIMEOUT(levels.load(), 120, 15000);
        const int beforeStop = previews.load();
        const auto pacedResult = pipeline.finish();
        pipeline.setPartialTextCallback({});
        QVERIFY(beforeStop > 0);
        QVERIFY(!pacedResult.empty());
        QCOMPARE(pacedResult, confirmed.back());
        qInfo("paced 6s capture: previews_before_stop=%d, first_preview_ms=%lld",
              beforeStop, firstPreviewMs.load());
    }
}

void TestLiveStreamTranscriber::segmentBoundaryDefersPreviewBeforeFlush()
{
    FakeStreamDecoder decoder;
    auto callbacks = decoder.callbacks();
    bool deferred = false;
    bool flushedDeferred = false;
    const auto originalOpen = callbacks.open;
    const auto originalFlush = callbacks.flush;
    callbacks.open = [&] { deferred = false; originalOpen(); };
    callbacks.deferPreview = [&] { deferred = true; };
    callbacks.flush = [&] { flushedDeferred = deferred; return originalFlush(); };
    LiveStreamTranscriber stream(shortSegments(), std::move(callbacks));
    auto input = speech(16000);
    stream.append(input.data(), input.size());
    QCOMPARE(decoder.flushes, size_t(1));
    QVERIFY(flushedDeferred);
    // A new segment resumes previews; its silence boundary defers them again.
    stream.append(input.data(), 3200);
    QVERIFY(!deferred);
    std::vector<int16_t> silence(4800, 0);
    stream.append(silence.data(), silence.size());
    QCOMPARE(decoder.flushes, size_t(2));
    QVERIFY(flushedDeferred);
}

void TestLiveStreamTranscriber::previewSchedulerAdaptsWithoutLosingLookahead()
{
    using namespace std::chrono_literals;
    using Clock = LivePreviewScheduler::Clock;
    const auto start = Clock::time_point{};
    LivePreviewScheduler scheduler(500, start);
    QVERIFY(!scheduler.admit(7680, false, false, start + 480ms));
    QVERIFY(scheduler.admit(320, false, false, start + 500ms));
    // An admission without a decode must keep collecting R2T2 lookahead.
    QVERIFY(scheduler.admit(320, false, false, start + 520ms));
    scheduler.decoded(800ms, start + 1320ms);
    QCOMPARE(scheduler.intervalMs(), 1000LL);
    QVERIFY(!scheduler.admit(16000, false, false, start + 1400ms));
    QVERIFY(scheduler.admit(320, false, false, start + 1520ms));
    scheduler.decoded(100ms, start + 1620ms);
    QCOMPARE(scheduler.intervalMs(), 750LL);
    scheduler.beginSegment();
    QVERIFY(!scheduler.admit(320, false, false, start + 5s));
    scheduler.decoded(10s, start + 15s);
    QCOMPARE(scheduler.intervalMs(), 2000LL);
    LivePreviewScheduler slow(4000, start);
    slow.decoded(10s, start + 10s);
    QCOMPARE(slow.intervalMs(), 4000LL);
}

void TestLiveStreamTranscriber::previewSchedulerRefreshesDuringBacklogAndSkipsFinalPreviews()
{
    using namespace std::chrono_literals;
    const auto start = LivePreviewScheduler::Clock::time_point{};
    LivePreviewScheduler scheduler(500, start);
    QVERIFY(!scheduler.admit(16000, true, false, start + 1s));
    QVERIFY(scheduler.admit(8000, true, false, start + 1500ms));
    scheduler.decoded(200ms, start + 1700ms);
    QVERIFY(!scheduler.admit(16000, true, false, start + 3s));
    QVERIFY(scheduler.admit(320, true, false, start + 3200ms));
    scheduler.decoded(200ms, start + 3400ms);
    // Even a large pending tail is consumed only by final segment decodes.
    QVERIFY(!scheduler.admit(160000, false, true, start + 20s));
    QVERIFY(!scheduler.admit(320, true, true, start + 30s));
}

namespace {
struct ScheduledPreviewDecoder : PauseAwareDecoder {
    std::optional<std::string> nextPreview;

    LiveStreamDecoder callbacks()
    {
        auto result = PauseAwareDecoder::callbacks();
        const auto feed = result.feed;
        result.feed = [this, feed](const int16_t* samples, size_t count) {
            feed(samples, count);
            return std::exchange(nextPreview, std::nullopt);
        };
        return result;
    }
};
}

void TestLiveStreamTranscriber::previewDefersLargeRewriteUntilNextDecode()
{
    ScheduledPreviewDecoder decoder;
    decoder.finals = {"今天下午的项目会议改到线上"};
    std::vector<std::string> previews;
    LiveStreamTranscriber stream({}, decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    const auto voicedFrame = speech(320);
    const auto decode = [&](const std::string& text) {
        decoder.nextPreview = text;
        stream.append(voicedFrame.data(), voicedFrame.size());
    };

    const std::string original = "今天下午有个项目会议";
    const std::string tailEdit = "今天下午有个预算评审";
    const std::string extension = tailEdit + "要参加";
    const std::string rewrite = "明天上午要开一个客户需求会议";
    decode(original);
    QVERIFY(!previews.empty());
    QCOMPARE(previews.back(), original);
    decode(tailEdit);
    QCOMPARE(previews.back(), tailEdit);
    decode(rewrite);
    QCOMPARE(previews.back(), tailEdit);
    const auto buffered = speech(6400);
    stream.append(buffered.data(), buffered.size());
    QCOMPARE(previews.back(), tailEdit);
    decode(extension);
    QCOMPARE(previews.back(), extension);

    decode(rewrite);
    QCOMPARE(previews.back(), extension);
    decode(rewrite); // A new decode can confirm unchanged text.
    QCOMPARE(previews.back(), rewrite);
    decode(""); // A real empty hypothesis is not the same as no decode.
    QCOMPARE(previews.back(), rewrite);
    QCOMPARE(stream.finish(), decoder.finals.front());
    QCOMPARE(previews.back(), decoder.finals.front());
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::pauseCheckpointBypassesDeferredRewrite()
{
    ScheduledPreviewDecoder decoder;
    decoder.checkpointText = "会议改到线上";
    std::vector<std::string> previews;
    LiveStreamTranscriber stream({}, decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    const auto input = speech(6400);
    const std::string original = "今天下午有个项目会议";
    decoder.nextPreview = original;
    stream.append(input.data(), input.size());
    decoder.nextPreview = "明天不用到现场开会了";
    stream.append(input.data(), input.size());
    QVERIFY(!previews.empty());
    QCOMPARE(previews.back(), original);

    const std::vector<int16_t> pause(3200, 0);
    stream.append(pause.data(), pause.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(previews.back(), decoder.checkpointText);
    QCOMPARE(stream.finish(), decoder.checkpointText);
    QCOMPARE(decoder.flushes, size_t(0));
}

void TestLiveStreamTranscriber::stopSuppressesAnInFlightCheckpoint()
{
    PauseAwareDecoder decoder;
    const auto preview = decoder.partial;
    auto callbacks = decoder.callbacks();
    auto checkpoint = callbacks.checkpoint;
    bool stopping = false;
    callbacks.suppressPreview = [&] { return stopping; };
    callbacks.checkpoint = [&] {
        stopping = true;
        return checkpoint();
    };
    std::vector<std::string> previews;
    LiveStreamTranscriber stream({}, callbacks,
        [&](const std::string& text) { previews.push_back(text); });
    const auto input = speech(6400);
    stream.append(input.data(), input.size());
    QCOMPARE(previews, std::vector<std::string>{preview});

    const std::vector<int16_t> quiet(3200, 0);
    stream.append(quiet.data(), quiet.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(previews, std::vector<std::string>{preview});
    QCOMPARE(stream.finish(), decoder.checkpointText);
    QCOMPARE(previews.back(), decoder.checkpointText);
    QCOMPARE(decoder.flushes, size_t(0));
}

namespace {
class TailAwareVad : public IVadBackend {
public:
    explicit TailAwareVad(int silenceMs = 500, int onsetAmplitude = 1)
        : silenceMs_(silenceMs), onsetAmplitude_(onsetAmplitude) {}

    std::vector<SpeechInterval> detect(const int16_t* samples, size_t count) override
    {
        std::vector<float> probabilities;
        for (size_t i = 0; i < count; i += 320) {
            probabilities.push_back(std::abs(samples[i]) >= onsetAmplitude_ ? 0.9f : 0.0f);
        }
        return segmentByDualThreshold(probabilities.data(), static_cast<int>(probabilities.size()),
                                      320, count, 0.5f, 0.35f, 100, silenceMs_);
    }
    std::string name() const override { return "tail-aware-test"; }
private:
    int silenceMs_;
    int onsetAmplitude_;
};
}

void TestLiveStreamTranscriber::batchedQuietTailPreservesLateResumedVoice_data()
{
    QTest::addColumn<int>("tailMs");
    QTest::newRow("stop-before-cadence") << 60;
    QTest::newRow("speech-between-analyses") << 100;
}

void TestLiveStreamTranscriber::batchedQuietTailPreservesLateResumedVoice()
{
    QFETCH(int, tailMs);
    PauseAwareDecoder decoder;
    decoder.finals = {"complete with late voice"};
    auto config = LiveStreamConfig{};
    auto endpoint = makeVadStreamEndpoint(std::make_unique<TailAwareVad>(), 100, 8000, 500);
    LiveStreamTranscriber stream(config, decoder.callbacks(), {}, {}, std::move(endpoint));
    const auto input = speech(6400);
    const std::vector<int16_t> pause(3200, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    auto resumed = speech(640); // 40 ms voice inside the next VAD analysis period.
    resumed.resize(static_cast<size_t>(tailMs) * 16, 0);
    stream.append(resumed.data(), resumed.size());
    QCOMPARE(stream.finish(), std::string("complete with late voice"));
    std::vector<int16_t> expected(input);
    expected.insert(expected.end(), pause.begin(), pause.end());
    expected.insert(expected.end(), resumed.begin(), resumed.end());
    QCOMPARE(decoder.audio.front(), expected);
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::batchedEndpointPreservesNextOnset_data()
{
    QTest::addColumn<bool>("checkpoint");
    QTest::addColumn<int>("nextSpeechMs");
    QTest::newRow("checkpoint-reused") << true << 100;
    QTest::newRow("checkpoint-disabled") << false << 100;
    QTest::newRow("endpoint-first-seen-at-stop") << true << 20;
    QTest::newRow("stop-right-after-handoff") << true << 40;
}

void TestLiveStreamTranscriber::batchedEndpointPreservesNextOnset()
{
    QFETCH(bool, checkpoint);
    QFETCH(int, nextSpeechMs);
    PauseAwareDecoder decoder;
    decoder.checkpointText = "first";
    decoder.finals = checkpoint ? std::vector<std::string>{"second"}
                                : std::vector<std::string>{"first", "second"};
    auto callbacks = decoder.callbacks();
    if (!checkpoint) callbacks.checkpoint = {};
    LiveStreamConfig config;
    config.minSpeechMs = nextSpeechMs;
    auto endpoint = makeVadStreamEndpoint(std::make_unique<TailAwareVad>(),
        config.minSpeechMs, config.maxSegmentMs, config.endSilenceMs);
    LiveStreamTranscriber stream(config, std::move(callbacks), {}, {}, std::move(endpoint));
    const std::vector<int16_t> first(440 * 16, 1000);
    const std::vector<int16_t> pause(520 * 16, 0);
    const std::vector<int16_t> second(static_cast<size_t>(nextSpeechMs) * 16, 2000);
    stream.append(first.data(), first.size());
    stream.append(pause.data(), pause.size());
    stream.append(second.data(), second.size());

    QCOMPARE(stream.finish(), std::string("first second"));
    QCOMPARE(decoder.audio.size(), size_t(2));
    QVERIFY(std::none_of(decoder.audio.front().begin(), decoder.audio.front().end(),
                         [](int16_t sample) { return sample == 2000; }));
    QCOMPARE(decoder.audio.back(), second);
    QCOMPARE(decoder.checkpoints, checkpoint ? size_t(1) : size_t(0));
    QCOMPARE(decoder.flushes, checkpoint ? size_t(1) : size_t(2));
}

void TestLiveStreamTranscriber::skippedCheckpointStillFinalizesAllSpeech()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"first and second"};
    auto callbacks = decoder.callbacks();
    size_t skippedCheckpoints = 0;
    callbacks.checkpoint = [&]() -> std::optional<std::string> {
        ++skippedCheckpoints;
        return std::nullopt;
    };
    LiveStreamTranscriber stream({}, std::move(callbacks));
    const auto first = speech(6400);
    const std::vector<int16_t> pause(3840, 0);
    const auto second = speech(960);
    stream.append(first.data(), first.size());
    stream.append(pause.data(), pause.size());
    QVERIFY(skippedCheckpoints > 0);
    QCOMPARE(decoder.flushes, size_t(0));
    stream.append(second.data(), second.size());
    QCOMPARE(stream.finish(), std::string("first and second"));
    QCOMPARE(decoder.flushes, size_t(1));
    std::vector<int16_t> expected(first);
    expected.insert(expected.end(), pause.begin(), pause.end());
    expected.insert(expected.end(), second.begin(), second.end());
    QCOMPARE(decoder.audio.front(), expected);
}

void TestLiveStreamTranscriber::finalVadBatchSkipsCheckpoint()
{
    PauseAwareDecoder decoder;
    decoder.finals = {"final phrase"};
    auto endpoint = makeVadStreamEndpoint(std::make_unique<TailAwareVad>(), 100, 8000, 500);
    LiveStreamTranscriber stream({}, decoder.callbacks(), {}, {}, std::move(endpoint));
    const auto input = speech(440 * 16);
    const std::vector<int16_t> pause(200 * 16, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(stream.finish(), std::string("final phrase"));
    QCOMPARE(decoder.checkpoints, size_t(0));
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::pauseCheckpointRetractionIsPreserved()
{
    PauseAwareDecoder decoder;
    decoder.checkpointText.clear();
    std::vector<std::string> previews;
    LiveStreamTranscriber stream({}, decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    const auto input = speech(6400);
    const std::vector<int16_t> quiet(8000, 0);
    stream.append(input.data(), input.size());
    QVERIFY(!previews.back().empty());
    stream.append(quiet.data(), quiet.size());
    QVERIFY(previews.back().empty());
    QVERIFY(stream.finish().empty());
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(decoder.flushes, size_t(0));
}

void TestLiveStreamTranscriber::quietTailRefreshesOnlyOnce()
{
    PauseAwareDecoder decoder;
    std::vector<std::string> previews;
    LiveStreamTranscriber stream({}, decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    const auto input = speech(6400);
    stream.append(input.data(), input.size());
    const auto beforeQuiet = decoder.liveFeeds;
    const std::vector<int16_t> quiet(1600, 0);
    stream.append(quiet.data(), quiet.size());
    QCOMPARE(decoder.checkpoints, size_t(0));
    stream.append(quiet.data(), quiet.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(previews.back(), std::string("corrected phrase"));
    const auto fedAtCheckpoint = decoder.audio.back().size();
    stream.append(quiet.data(), quiet.size());
    stream.append(quiet.data(), quiet.size());
    QCOMPARE(decoder.liveFeeds, beforeQuiet);
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(decoder.audio.back().size(), fedAtCheckpoint);
    QCOMPARE(stream.confirmedSegments(), size_t(0));
    QCOMPARE(stream.finish(), std::string("corrected phrase"));
    QCOMPARE(decoder.flushes, size_t(0));
}

void TestLiveStreamTranscriber::longPauseConfirmsCheckpointWithoutDecodingAgain()
{
    PauseAwareDecoder decoder;
    LiveStreamTranscriber stream({}, decoder.callbacks());
    const auto input = speech(6400);
    stream.append(input.data(), input.size());
    const std::vector<int16_t> quiet(8000, 0);
    stream.append(quiet.data(), quiet.size());
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(decoder.flushes, size_t(0));
    QCOMPARE(decoder.closes, size_t(1));
    stream.append(quiet.data(), quiet.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(decoder.audio.size(), size_t(1));
    QCOMPARE(stream.finish(), std::string("corrected phrase"));
}

void TestLiveStreamTranscriber::checkpointRetainsWeakTail_data()
{
    QTest::addColumn<bool>("batched");
    QTest::addColumn<int>("tailSamples");
    QTest::addColumn<bool>("endsNaturally");
    QTest::newRow("energy-stop") << false << 640 << false;
    QTest::newRow("energy-partial-frame-at-stop") << false << 137 << false;
    QTest::newRow("energy-natural-end") << false << 640 << true;
    QTest::newRow("batched-partial-frame-at-stop") << true << 137 << false;
    QTest::newRow("batched-natural-end") << true << 640 << true;
}

void TestLiveStreamTranscriber::checkpointRetainsWeakTail()
{
    QFETCH(bool, batched);
    QFETCH(int, tailSamples);
    QFETCH(bool, endsNaturally);
    PauseAwareDecoder decoder;
    decoder.finals = {"phrase with a weak ending"};
    auto endpoint = batched ? makeVadStreamEndpoint(std::make_unique<TailAwareVad>(500, 80),
                                                    100, 8000, 500) : nullptr;
    LiveStreamTranscriber stream({}, decoder.callbacks(), {}, {}, std::move(endpoint));
    const auto input = speech(6400);
    const std::vector<int16_t> pause(3200, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(decoder.checkpoints, size_t(1));

    std::vector<int16_t> tail(tailSamples);
    for (size_t i = 0; i < tail.size(); ++i) tail[i] = i % 2 ? -60 : 60;
    stream.append(tail.data(), tail.size());
    QCOMPARE(decoder.audio.front().size(), input.size() + pause.size());
    QCOMPARE(decoder.flushes, size_t(0));
    if (endsNaturally) {
        const std::vector<int16_t> quiet(4800, 0);
        stream.append(quiet.data(), quiet.size());
        QCOMPARE(stream.confirmedSegments(), size_t(1));
    }

    QCOMPARE(stream.finish(), std::string("phrase with a weak ending"));
    auto expected = input;
    expected.insert(expected.end(), pause.begin(), pause.end());
    expected.insert(expected.end(), tail.begin(), tail.end());
    QCOMPARE(decoder.audio.size(), size_t(1));
    QCOMPARE(decoder.audio.front(), expected);
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::checkpointIgnoresSteadyNoiseAndDc()
{
    PauseAwareDecoder decoder;
    LiveStreamTranscriber stream({}, decoder.callbacks());
    const auto input = speech(6400);
    std::vector<int16_t> noise(3200);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = i % 2 ? 940 : 1060;
    stream.append(input.data(), input.size());
    stream.append(noise.data(), noise.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    stream.append(noise.data(), noise.size());
    const std::vector<int16_t> dc(1600, 1000);
    stream.append(dc.data(), dc.size());
    QCOMPARE(stream.finish(), decoder.checkpointText);
    QCOMPARE(decoder.audio.front().size(), input.size() + noise.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(decoder.flushes, size_t(0));
}

void TestLiveStreamTranscriber::resumedSpeechPreservesGapAndRefreshesAgain()
{
    PauseAwareDecoder decoder;
    LiveStreamTranscriber stream({}, decoder.callbacks());
    const auto input = speech(6400);
    const std::vector<int16_t> pause(4800, 0);
    const std::vector<int16_t> tail(3200, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    const auto beforeResume = decoder.liveFeeds;
    decoder.checkpointText = "corrected complete sentence";
    stream.append(input.data(), input.size());
    QVERIFY(decoder.liveFeeds > beforeResume);
    stream.append(tail.data(), tail.size());
    QCOMPARE(decoder.checkpoints, size_t(2));
    QCOMPARE(stream.finish(), std::string("corrected complete sentence"));
    QCOMPARE(decoder.audio.size(), size_t(1));
    std::vector<int16_t> expected(input);
    expected.insert(expected.end(), pause.begin(), pause.end());
    expected.insert(expected.end(), input.begin(), input.end());
    expected.insert(expected.end(), tail.begin(), tail.end());
    QCOMPARE(decoder.audio.front(), expected);
    QCOMPARE(decoder.flushes, size_t(0));
}

void TestLiveStreamTranscriber::stopBeforeShortPauseStillDecodesTail()
{
    PauseAwareDecoder decoder;
    decoder.finals = {"complete tail"};
    LiveStreamTranscriber stream({}, decoder.callbacks());
    const auto input = speech(6400);
    const std::vector<int16_t> pause(1600, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(stream.finish(), std::string("complete tail"));
    QCOMPARE(decoder.checkpoints, size_t(0));
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::shortBurstNeverTriggersPauseCheckpoint()
{
    PauseAwareDecoder decoder;
    LiveStreamTranscriber stream({}, decoder.callbacks());
    const auto input = speech(640);
    const std::vector<int16_t> pause(9600, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QVERIFY(stream.finish().empty());
    QCOMPARE(decoder.checkpoints, size_t(0));
    QCOMPARE(decoder.flushes, size_t(0));
}

void TestLiveStreamTranscriber::earlyEndpointDoesNotCheckpointThenFlush()
{
    PauseAwareDecoder decoder;
    auto config = LiveStreamConfig{};
    config.endSilenceMs = 100;
    LiveStreamTranscriber stream(config, decoder.callbacks());
    const auto input = speech(6400);
    const std::vector<int16_t> pause(1600, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QCOMPARE(decoder.checkpoints, size_t(0));
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::cancelDropsPendingPause()
{
    PauseAwareDecoder decoder;
    LiveStreamTranscriber stream({}, decoder.callbacks());
    const auto input = speech(6400);
    const std::vector<int16_t> pause(6400, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(decoder.checkpoints, size_t(1));
    stream.cancel();
    QVERIFY(stream.finish().empty());
    QCOMPARE(decoder.flushes, size_t(0));
    QCOMPARE(decoder.closes, size_t(1));
}

void TestLiveStreamTranscriber::defaultStreamKeepsSentencePastThreeSeconds()
{
    FakeStreamDecoder decoder;
    decoder.partial = "early guess";
    decoder.finals = {"corrected complete sentence"};
    std::vector<std::string> previews;
    LiveStreamTranscriber stream(makeLiveStreamConfig(Config{}), decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    const auto input = speech(4 * 16000);
    stream.append(input.data(), input.size());
    QCOMPARE(decoder.flushes, size_t(0));
    QCOMPARE(previews.back(), std::string("early guess"));
    decoder.partial += " with later speech";
    stream.append(input.data(), 3500 * 16);
    QCOMPARE(decoder.audio.front().size(), size_t(7500 * 16));
    QCOMPARE(previews.back(), decoder.partial);
    stream.append(input.data() + 3500 * 16, 500 * 16);
    QCOMPARE(decoder.flushes, size_t(1));
    QCOMPARE(decoder.audio.front().size(), size_t(8 * 16000));
    QCOMPARE(previews.back(), std::string("corrected complete sentence"));
    QCOMPARE(stream.finish(), std::string("corrected complete sentence"));
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::briefPauseKeepsCurrentSentenceOpen()
{
    FakeStreamDecoder decoder;
    LiveStreamTranscriber stream(makeLiveStreamConfig(Config{}), decoder.callbacks());
    const auto input = speech(8000);
    const std::vector<int16_t> pause(400 * 16, 0);
    stream.append(input.data(), input.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(decoder.flushes, size_t(0));
    stream.append(input.data(), input.size());
    stream.finish();
    QCOMPARE(decoder.flushes, size_t(1));
    QCOMPARE(decoder.audio.size(), size_t(1));
}

void TestLiveStreamTranscriber::confirmsAtPauseAndKeepsPrefixStable()
{
    FakeStreamDecoder decoder;
    decoder.partial = "错误预览";
    decoder.finals = {"第一句。", "第二句。"};
    std::vector<std::string> previews;
    std::vector<std::string> confirmed;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); },
        [&](const std::string& text) { confirmed.push_back(text); });
    const auto first = speech(6400);
    stream.append(first.data(), first.size());
    const std::vector<int16_t> pause(4800, 0);
    stream.append(pause.data(), pause.size());
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QCOMPARE(decoder.flushes, size_t(1));
    QVERIFY(!decoder.open);
    QCOMPARE(previews.back(), std::string("第一句。"));
    QCOMPARE(confirmed, (std::vector<std::string>{"第一句。"}));

    decoder.partial = "第";
    stream.append(first.data(), first.size());
    QCOMPARE(previews.back(), std::string("第一句。第"));
    decoder.partial = "修改后的第二句";
    stream.append(first.data(), 320);
    QCOMPARE(previews.back(), std::string("第一句。修改后的第二句"));
    QCOMPARE(stream.finish(), std::string("第一句。第二句。"));
    QCOMPARE(confirmed, (std::vector<std::string>{"第一句。", "第一句。第二句。"}));
    QCOMPARE(decoder.audio.size(), size_t(2));
}

void TestLiveStreamTranscriber::shortBurstAtPauseIsDiscardedAndRetracted()
{
    FakeStreamDecoder decoder;
    decoder.partial = "false preview";
    std::vector<std::string> previews;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    auto burst = speech(320); // 20 ms, below the 100 ms minimum.
    stream.append(burst.data(), burst.size());
    QCOMPARE(previews.back(), std::string("false preview"));
    const std::vector<int16_t> silence(4800, 0);
    stream.append(silence.data(), silence.size());
    QCOMPARE(stream.confirmedSegments(), size_t(0));
    QCOMPARE(decoder.flushes, size_t(0));
    QCOMPARE(decoder.closes, size_t(1));
    QVERIFY(previews.back().empty());
    decoder.partial = "valid speech";
    const auto valid = speech(3200);
    stream.append(valid.data(), valid.size());
    QCOMPARE(stream.finish(), std::string("valid speech"));
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::shortBurstAtStopIsDiscarded()
{
    FakeStreamDecoder decoder;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto burst = speech(320);
    stream.append(burst.data(), burst.size());
    QVERIFY(stream.finish().empty());
    QCOMPARE(decoder.flushes, size_t(0));
    QCOMPARE(stream.confirmedSegments(), size_t(0));
    QCOMPARE(decoder.closes, size_t(1));
}

void TestLiveStreamTranscriber::shortBurstDoesNotChangeConfirmedText()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"已确认。"};
    std::vector<std::string> previews;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    const auto valid = speech(3200);
    const std::vector<int16_t> silence(4800, 0);
    stream.append(valid.data(), valid.size());
    stream.append(silence.data(), silence.size());
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    decoder.partial = "噪声";
    const auto burst = speech(320);
    stream.append(burst.data(), burst.size());
    QCOMPARE(previews.back(), std::string("已确认。噪声"));
    stream.append(silence.data(), silence.size());
    QCOMPARE(previews.back(), std::string("已确认。"));
    QCOMPARE(stream.finish(), std::string("已确认。"));
    QCOMPARE(decoder.flushes, size_t(1));
}

void TestLiveStreamTranscriber::validPauseAtHardLimitIsConfirmed()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"valid speech"};
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    auto input = speech(11200); // 700 ms speech plus 300 ms silence hits 1 s exactly.
    input.insert(input.end(), 4800, 0);
    stream.append(input.data(), input.size());
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QCOMPARE(decoder.flushes, size_t(1));
    QCOMPARE(stream.finish(), std::string("valid speech"));
    QCOMPARE(decoder.closes, size_t(1));
}

void TestLiveStreamTranscriber::prefersPauseNearSegmentLimit_data()
{
    QTest::addColumn<bool>("batched");
    QTest::addColumn<int>("speechMs");
    QTest::addColumn<int>("pauseMs");
    QTest::addColumn<int>("limitMs");
    QTest::addColumn<int>("nextLeadInMs");
    QTest::newRow("energy") << false << 7400 << 200 << 8000 << 0;
    QTest::newRow("batched") << true << 7400 << 200 << 8000 << 0;
    QTest::newRow("batched-analysis-crosses-limit") << true << 7820 << 280 << 8050 << 0;
    QTest::newRow("batched-limit-before-pause-confirmed") << true << 7820 << 280 << 8000 << 100;
}

void TestLiveStreamTranscriber::prefersPauseNearSegmentLimit()
{
    QFETCH(bool, batched);
    QFETCH(int, speechMs);
    QFETCH(int, pauseMs);
    QFETCH(int, limitMs);
    QFETCH(int, nextLeadInMs);
    PauseAwareDecoder decoder;
    decoder.finals = {"first sentence", "second sentence"};
    LiveStreamConfig config;
    config.maxSegmentMs = limitMs;
    auto endpoint = batched ? makeVadStreamEndpoint(std::make_unique<TailAwareVad>(),
                                                    100, limitMs, 500) : nullptr;
    LiveStreamTranscriber stream(config, decoder.callbacks(), {}, {}, std::move(endpoint));
    auto first = speech(static_cast<size_t>(speechMs) * 16);
    const std::vector<int16_t> pause(pauseMs * 16, 0);
    stream.append(first.data(), first.size());
    stream.append(pause.data(), pause.size());
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QCOMPARE(decoder.checkpoints, size_t(0));
    QCOMPARE(decoder.flushes, size_t(1));
    QCOMPARE(decoder.audio.size(), size_t(1));
    const size_t acceptedPause = std::min(pause.size(), size_t(limitMs * 16) - first.size());
    first.insert(first.end(), acceptedPause, 0);
    QCOMPARE(decoder.audio.front(), first);

    auto next = speech(6400);
    for (auto& sample : next) sample /= 2;
    stream.append(next.data(), next.size());
    QCOMPARE(stream.finish(), std::string("first sentence second sentence"));
    QCOMPARE(decoder.audio.size(), size_t(2));
    next.insert(next.begin(), nextLeadInMs * 16, 0);
    QCOMPARE(decoder.audio.back(), next);
}

void TestLiveStreamTranscriber::nearLimitPauseAfterCheckpoint_data()
{
    QTest::addColumn<bool>("weakTail");
    QTest::newRow("reuse-quiet-checkpoint") << false;
    QTest::newRow("wait-for-pause-after-weak-tail") << true;
}

void TestLiveStreamTranscriber::nearLimitPauseAfterCheckpoint()
{
    QFETCH(bool, weakTail);
    PauseAwareDecoder decoder;
    decoder.finals = {"complete ending"};
    LiveStreamTranscriber stream({}, decoder.callbacks());
    auto expected = speech(6900 * 16);
    const std::vector<int16_t> silence(200 * 16, 0);
    stream.append(expected.data(), expected.size());
    stream.append(silence.data(), silence.size());
    expected.insert(expected.end(), silence.begin(), silence.end());
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(stream.confirmedSegments(), size_t(0));

    std::vector<int16_t> tail(320, 0);
    if (weakTail) {
        for (size_t i = 0; i < tail.size(); ++i) tail[i] = i % 2 ? -60 : 60;
        expected.insert(expected.end(), tail.begin(), tail.end());
    }
    stream.append(tail.data(), tail.size());
    stream.append(silence.data(), 80 * 16);
    QCOMPARE(stream.confirmedSegments(), weakTail ? size_t(0) : size_t(1));
    stream.append(silence.data(), 120 * 16);
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QCOMPARE(stream.finish(), weakTail ? decoder.finals.front() : decoder.checkpointText);
    QCOMPARE(decoder.audio.size(), size_t(1));
    QCOMPARE(decoder.audio.front(), expected);
    QCOMPARE(decoder.checkpoints, size_t(1));
    QCOMPARE(decoder.flushes, weakTail ? size_t(1) : size_t(0));
}

void TestLiveStreamTranscriber::longSpeechHasBoundedStreamsWithoutLosingSamples()
{
    FakeStreamDecoder decoder;
    LiveStreamTranscriber stream({}, decoder.callbacks());
    const auto input = speech(16000 * 25 + 137);
    stream.append(input.data(), input.size());
    QCOMPARE(stream.confirmedSegments(), size_t(3));
    stream.finish();
    QCOMPARE(decoder.audio.size(), size_t(4));
    QCOMPARE(decoder.flushes, size_t(4));
    QCOMPARE(decoder.closes, size_t(4));
    std::vector<int16_t> decoded;
    for (const auto& segment : decoder.audio) {
        QVERIFY(segment.size() <= size_t(16000 * 8));
        decoded.insert(decoded.end(), segment.begin(), segment.end());
    }
    QVERIFY(decoded == input);
}

void TestLiveStreamTranscriber::partitionsDoNotChangeBoundaries()
{
    FakeStreamDecoder bulkDecoder;
    FakeStreamDecoder packetDecoder;
    LiveStreamTranscriber bulk(shortSegments(), bulkDecoder.callbacks());
    LiveStreamTranscriber packets(shortSegments(), packetDecoder.callbacks());
    auto input = speech(7000);
    input.insert(input.end(), 6000, 0);
    const auto rest = speech(31000);
    input.insert(input.end(), rest.begin(), rest.end());
    bulk.append(input.data(), input.size());
    const size_t sizes[] = {1, 319, 2048, 13, 640};
    size_t index = 0;
    for (size_t offset = 0; offset < input.size();) {
        const size_t count = std::min(sizes[index++ % 5], input.size() - offset);
        packets.append(input.data() + offset, count);
        offset += count;
    }
    QCOMPARE(bulk.finish(), packets.finish());
    QVERIFY(bulkDecoder.audio == packetDecoder.audio);
    QCOMPARE(bulkDecoder.flushes, packetDecoder.flushes);
}

void TestLiveStreamTranscriber::nonFrameAlignedLimitKeepsEverySample()
{
    FakeStreamDecoder decoder;
    auto config = shortSegments();
    config.maxSegmentMs = 1013;
    LiveStreamTranscriber stream(config, decoder.callbacks());
    const auto input = speech(40000);
    stream.append(input.data(), input.size());
    stream.finish();
    QCOMPARE(decoder.audio.size(), size_t(3));
    std::vector<int16_t> decoded;
    for (size_t i = 0; i < decoder.audio.size(); ++i) {
        const auto& segment = decoder.audio[i];
        QVERIFY(segment.size() <= size_t(1013 * 16));
        if (i < 2) QCOMPARE(segment.size(), size_t(1013 * 16));
        decoded.insert(decoded.end(), segment.begin(), segment.end());
    }
    QVERIFY(decoded == input);
}

void TestLiveStreamTranscriber::forcedCutUsesQuietSpanWithoutLosingSamples()
{
    FakeStreamDecoder decoder;
    auto config = shortSegments();
    config.maxSegmentMs = 2000;
    LiveStreamTranscriber stream(config, decoder.callbacks());
    auto input = speech(2500 * 16);
    for (size_t i = 1840 * 16; i < 1960 * 16; ++i) input[i] = i % 2 ? -200 : 200;

    stream.append(input.data(), 2000 * 16);
    QCOMPARE(decoder.flushes, size_t(1));
    QCOMPARE(decoder.audio.front().size(), size_t(1910 * 16));
    stream.append(input.data() + 2000 * 16, 500 * 16);
    stream.finish();

    QCOMPARE(decoder.audio.size(), size_t(2));
    auto decoded = decoder.audio.front();
    decoded.insert(decoded.end(), decoder.audio.back().begin(), decoder.audio.back().end());
    QVERIFY(decoded == input);
}

void TestLiveStreamTranscriber::forcedCutPreservesShortContinuation_data()
{
    QTest::addColumn<bool>("quietRecovery");
    QTest::newRow("vad-detects-carry") << false;
    QTest::newRow("vad-waits-for-onset") << true;
}

void TestLiveStreamTranscriber::forcedCutPreservesShortContinuation()
{
    QFETCH(bool, quietRecovery);
    FakeStreamDecoder decoder;
    decoder.finals = {"first", "tail"};
    auto config = shortSegments();
    config.maxSegmentMs = 2000;
    LiveStreamTranscriber stream(config, decoder.callbacks());
    auto input = speech(2000 * 16);
    const int16_t quietLevel = quietRecovery ? 30 : 200;
    for (size_t i = 1820 * 16; i < 1940 * 16; ++i) input[i] = i % 2 ? -quietLevel : quietLevel;
    if (quietRecovery) {
        for (size_t i = 1940 * 16; i < input.size(); ++i) input[i] = i % 2 ? -85 : 85;
    }
    stream.append(input.data(), input.size());
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QVERIFY(decoder.audio.front().size() < input.size());

    const std::vector<int16_t> quiet(320, 0);
    const auto tail = speech(960);
    stream.append(quiet.data(), quiet.size());
    stream.append(tail.data(), tail.size());
    QCOMPARE(stream.finish(), std::string("first tail"));
    QCOMPARE(decoder.audio.size(), size_t(2));
    auto decoded = decoder.audio.front();
    decoded.insert(decoded.end(), decoder.audio.back().begin(), decoder.audio.back().end());
    input.insert(input.end(), quiet.begin(), quiet.end());
    input.insert(input.end(), tail.begin(), tail.end());
    QCOMPARE(decoded, input);
}

void TestLiveStreamTranscriber::forcedCutKeepsContinuousQuietSpeech()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"first", "quiet continuation"};
    LiveStreamTranscriber stream({}, decoder.callbacks());
    std::vector<int16_t> input(8560 * 16);
    for (size_t i = 0; i < input.size(); ++i) {
        const int16_t amplitude = i < 7800 * 16 ? 3200 : (i < 7940 * 16 ? 30 : 85);
        input[i] = i % 2 ? -amplitude : amplitude;
    }
    stream.append(input.data(), 8000 * 16);
    QCOMPARE(decoder.flushes, size_t(1));
    QVERIFY(decoder.audio.front().size() < 8000 * 16);

    stream.append(input.data() + 8000 * 16, 560 * 16);
    QCOMPARE(decoder.flushes, size_t(1));
    QCOMPARE(stream.finish(), std::string("first quiet continuation"));
    QCOMPARE(decoder.audio.size(), size_t(2));
    auto decoded = decoder.audio.front();
    decoded.insert(decoded.end(), decoder.audio.back().begin(), decoder.audio.back().end());
    QCOMPARE(decoded, input);
}

void TestLiveStreamTranscriber::forcedCutKeepsFadingTail()
{
    FakeStreamDecoder decoder;
    auto config = shortSegments();
    config.maxSegmentMs = 2000;
    LiveStreamTranscriber stream(config, decoder.callbacks());
    auto input = speech(2000 * 16);
    for (size_t i = 1860 * 16; i < input.size(); ++i) input[i] = i % 2 ? -200 : 200;

    stream.append(input.data(), input.size());
    QCOMPARE(decoder.flushes, size_t(1));
    QVERIFY(decoder.audio.front() == input);
}

void TestLiveStreamTranscriber::emptyConfirmedSegmentRetriesOnce()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"already correct", " \t"};
    auto callbacks = decoder.callbacks();
    const auto input = speech(16000 + 3200);
    std::vector<int16_t> retried;
    size_t retries = 0;
    callbacks.retryEmptySegment = [&](const int16_t* pcm, size_t count) {
        if (decoder.open) throw std::logic_error("offline retry while stream is open");
        ++retries;
        retried.assign(pcm, pcm + count);
        return "recovered";
    };
    LiveStreamTranscriber stream(shortSegments(), callbacks);
    stream.append(input.data(), input.size());
    QCOMPARE(stream.finish(), std::string("already correct recovered"));
    QCOMPARE(stream.finish(), std::string("already correct recovered"));
    QCOMPARE(retries, size_t(1));
    QVERIFY(retried == std::vector<int16_t>(input.begin() + 16000, input.end()));
}

void TestLiveStreamTranscriber::emptyUnconfirmedSegmentDoesNotRetry()
{
    FakeStreamDecoder decoder;
    decoder.finals = {""};
    auto callbacks = decoder.callbacks();
    size_t retries = 0;
    callbacks.retryEmptySegment = [&](const int16_t*, size_t) {
        ++retries;
        return "incorrect recovery";
    };
    LiveStreamTranscriber stream(shortSegments(), callbacks);
    const std::vector<int16_t> silence(3200, 0);
    const auto burst = speech(960);
    stream.append(silence.data(), silence.size());
    stream.append(burst.data(), burst.size());
    QCOMPARE(stream.finish(), std::string());
    QCOMPARE(retries, size_t(0));
}

void TestLiveStreamTranscriber::hotwordsFilterOnlyAFullListEcho()
{
    const std::string name = u8"\u5F20\u4E09";
    const AsrHotwords hotwords(u8"Deepin\uFF0C" + name + u8"\u3001DEEPIN\uFF1BC#; <|im_end|>");
    QCOMPARE(hotwords.prompt(), "Deepin, " + name + ", C#");
    QCOMPARE(hotwords.filterEcho(u8"deepin\uFF0C" + name + u8"\u3001C#\u3002"), std::string());

    const auto normalSpeech = "Ask " + name + " to open the C# project on Deepin.";
    QCOMPARE(hotwords.filterEcho(normalSpeech), normalSpeech);
    QCOMPARE(hotwords.filterEcho("Deepin, " + name), "Deepin, " + name);
    QCOMPARE(AsrHotwords(name + u8"\uFF0C" + name).filterEcho(name + u8"\u3002"), name + u8"\u3002");
    QCOMPARE(AsrHotwords(std::string(513, 'x') + "," + name).prompt(), name);
    const AsrHotwords technicalTerms("S3, Bucket");
    QCOMPARE(technicalTerms.filterEcho("s3, Bucket."), std::string());
    QCOMPARE(technicalTerms.filterEcho("S3Bucket"), std::string("S3Bucket"));
    const AsrHotwords multiWordTerms("New York, City");
    QCOMPARE(multiWordTerms.filterEcho("New, York City"), std::string("New, York City"));
}

void TestLiveStreamTranscriber::silenceDoesNotDecodeOrAccumulate()
{
    FakeStreamDecoder decoder;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const std::vector<int16_t> silence(16000 * 60, 0);
    stream.append(silence.data(), silence.size());
    QCOMPARE(stream.finish(), std::string());
    QVERIFY(decoder.audio.empty());
    QCOMPARE(decoder.flushes, size_t(0));
}

void TestLiveStreamTranscriber::idleLeadInIsBounded()
{
    FakeStreamDecoder decoder;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const std::vector<int16_t> silence(16000 * 3, 0);
    stream.append(silence.data(), silence.size());
    const auto input = speech(6400);
    stream.append(input.data(), input.size());
    stream.finish();
    QCOMPARE(decoder.audio.size(), size_t(1));
    QCOMPARE(decoder.audio.front().size(), size_t(3200 + 6400));
    QVERIFY(std::equal(input.begin(), input.end(), decoder.audio.front().begin() + 3200));
}

void TestLiveStreamTranscriber::finishKeepsShortTailAfterForcedBoundary()
{
    FakeStreamDecoder decoder;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto input = speech(16000 + 137);
    stream.append(input.data(), input.size());
    stream.finish();
    QCOMPARE(decoder.audio.size(), size_t(2));
    QCOMPARE(decoder.audio[0].size(), size_t(16000));
    QCOMPARE(decoder.audio[1].size(), size_t(137));
    QVERIFY(std::equal(input.begin() + 16000, input.end(), decoder.audio[1].begin()));
}

void TestLiveStreamTranscriber::shortPauseAfterHardBoundaryKeepsValidTail_data()
{
    QTest::addColumn<int>("pauseMs");
    for (int ms : {20, 100, 280}) {
        QTest::newRow(qPrintable(QString::number(ms))) << ms;
    }
}

void TestLiveStreamTranscriber::shortPauseAfterHardBoundaryKeepsValidTail()
{
    QFETCH(int, pauseMs);
    FakeStreamDecoder decoder;
    decoder.finals = {"first", "tail"};
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto first = speech(16000);
    stream.append(first.data(), first.size());
    const std::vector<int16_t> pause(pauseMs * 16, 0);
    stream.append(pause.data(), pause.size());
    const auto tail = speech(960); // 60 ms, validated by preceding continuous speech.
    stream.append(tail.data(), tail.size());
    QCOMPARE(stream.finish(), std::string("first tail"));
    QCOMPARE(stream.confirmedSegments(), size_t(2));
    QCOMPARE(decoder.flushes, size_t(2));
    QVERIFY(std::equal(tail.begin(), tail.end(), decoder.audio.back().end() - tail.size()));
}

void TestLiveStreamTranscriber::fullPauseAfterHardBoundaryRejectsNewShortBurst()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"first", "noise"};
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto first = speech(16000);
    stream.append(first.data(), first.size());
    const std::vector<int16_t> pause(4800, 0);
    stream.append(pause.data(), pause.size());
    const auto noise = speech(960);
    stream.append(noise.data(), noise.size());
    QCOMPARE(stream.finish(), std::string("first"));
    QCOMPARE(stream.confirmedSegments(), size_t(1));
    QCOMPARE(decoder.flushes, size_t(1));
}

namespace {
class QuietVad : public IVadBackend {
public:
    std::vector<SpeechInterval> detect(const int16_t* samples, size_t count) override
    {
        std::vector<SpeechInterval> intervals;
        size_t i = 0;
        while (i < count) {
            while (i < count && samples[i] == 0) ++i;
            const auto begin = i;
            while (i < count && samples[i] != 0) ++i;
            if (i > begin) intervals.push_back({begin, i, 1.0f});
        }
        return intervals;
    }
    std::string name() const override { return "quiet-vad"; }
};
}

void TestLiveStreamTranscriber::injectedVadDetectsQuietSpeech()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"quiet speech"};
    auto config = shortSegments();
    auto endpoint = makeVadStreamEndpoint(std::make_unique<QuietVad>(), 100, 1000, 300);
    LiveStreamTranscriber stream(config, decoder.callbacks(), {}, {}, std::move(endpoint));
    const std::vector<int16_t> quiet(3200, 2); // Below energy's speech threshold.
    stream.append(quiet.data(), quiet.size());
    QCOMPARE(stream.finish(), std::string("quiet speech"));
    QCOMPARE(decoder.flushes, size_t(1));
    QVERIFY(decoder.audio.front() == quiet);
}

void TestLiveStreamTranscriber::batchedVadReportsShortUtteranceAtFinish()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"short speech"};
    auto config = shortSegments();
    config.minSpeechMs = 20;
    auto endpoint = makeVadStreamEndpoint(std::make_unique<QuietVad>(), 20, 1000, 300);
    LiveStreamTranscriber stream(config, decoder.callbacks(), {}, {}, std::move(endpoint));
    const std::vector<int16_t> quiet(960, 2);
    stream.append(quiet.data(), quiet.size());
    QVERIFY(decoder.audio.empty()); // 100 ms cadence has not elapsed yet.
    QCOMPARE(stream.finish(), std::string("short speech"));
    QVERIFY(decoder.audio.front() == quiet);
}

void TestLiveStreamTranscriber::naturalRepetitionsArePreserved()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"hello", "hello"};
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto input = speech(16000 * 2);
    stream.append(input.data(), input.size());
    QCOMPARE(stream.finish(), std::string("hello hello"));
    QCOMPARE(decoder.audio.size(), size_t(2));
}

void TestLiveStreamTranscriber::accentedWordsKeepTheirSeparator()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"café", "été"};
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto input = speech(32000);
    stream.append(input.data(), input.size());
    QCOMPARE(stream.finish(), std::string("café été"));
}

void TestLiveStreamTranscriber::cancelDoesNotFlushOrPublish()
{
    FakeStreamDecoder decoder;
    std::vector<std::string> previews;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks(),
        [&](const std::string& text) { previews.push_back(text); });
    const auto input = speech(3200);
    stream.append(input.data(), input.size());
    const size_t published = previews.size();
    stream.cancel();
    stream.cancel();
    stream.append(input.data(), input.size());
    QVERIFY(stream.finish().empty());
    QCOMPARE(decoder.flushes, size_t(0));
    QCOMPARE(decoder.closes, size_t(1));
    QCOMPARE(previews.size(), published);
}

void TestLiveStreamTranscriber::finalTextAndFinishAreIdempotent()
{
    FakeStreamDecoder decoder;
    decoder.finals = {"final correction"};
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto input = speech(3200);
    stream.append(input.data(), input.size());
    QCOMPARE(stream.finish(), std::string("final correction"));
    QCOMPARE(stream.finish(), std::string("final correction"));
    stream.append(input.data(), input.size());
    QCOMPARE(decoder.flushes, size_t(1));
    QCOMPARE(decoder.audio.size(), size_t(1));
}

void TestLiveStreamTranscriber::reopenErrorsPropagate()
{
    FakeStreamDecoder decoder;
    LiveStreamTranscriber stream(shortSegments(), decoder.callbacks());
    const auto input = speech(16000);
    stream.append(input.data(), input.size());
    decoder.failOpen = true;
    QVERIFY_EXCEPTION_THROWN(stream.append(input.data(), 320), std::runtime_error);
    stream.cancel();
}

void TestLiveStreamTranscriber::liveStreamConfigKeepsDefaultLimitAndRejectsOverflow()
{
    Config config;
    QCOMPARE(makeLiveStreamConfig(config).maxSegmentMs, 8000);
    config.streamStepMs = 4000;
    QCOMPARE(makeLiveStreamConfig(config).maxSegmentMs, 8000);
    config.vadMinSpeechMs = std::numeric_limits<int>::max();
    QVERIFY_EXCEPTION_THROWN(makeLiveStreamConfig(config), std::invalid_argument);
}

void TestLiveStreamTranscriber::longMinimumSpeechExtendsSegmentLimit_data()
{
    QTest::addColumn<QString>("backend");
    QTest::newRow("energy") << QStringLiteral("energy");
    QTest::newRow("silero-padding") << QStringLiteral("silero");
}

void TestLiveStreamTranscriber::longMinimumSpeechExtendsSegmentLimit()
{
    QFETCH(QString, backend);
    Config config;
    config.vadBackend = backend.toStdString();
    config.vadMinSpeechMs = 9000;
    const auto streamConfig = makeLiveStreamConfig(config);
    QVERIFY(streamConfig.maxSegmentMs >= streamConfig.minSpeechMs + streamConfig.prePaddingMs);
    QCOMPARE(streamConfig.prePaddingMs, backend == QStringLiteral("silero") ? 1000 : 200);
    FakeStreamDecoder decoder;
    // Use the energy detector to test the duration contract without weights.
    LiveStreamTranscriber stream(streamConfig, decoder.callbacks());
    const auto lead = std::vector<int16_t>(static_cast<size_t>(streamConfig.prePaddingMs) * 16, 0);
    stream.append(lead.data(), lead.size());
    const auto input = speech(27 * 16000);
    stream.append(input.data(), input.size());
    QVERIFY(!stream.finish().empty());
    QVERIFY(decoder.flushes > 0);
    for (const auto& audio : decoder.audio) {
        QVERIFY(audio.size() <= static_cast<size_t>(streamConfig.maxSegmentMs) * 16);
    }
}

void TestLiveStreamTranscriber::invalidConfigIsRejected()
{
    FakeStreamDecoder decoder;
    auto config = shortSegments();
    config.maxSegmentMs = 0;
    QVERIFY_EXCEPTION_THROWN(LiveStreamTranscriber(config, decoder.callbacks()), std::invalid_argument);
    config = shortSegments();
    config.prePaddingMs = config.maxSegmentMs;
    QVERIFY_EXCEPTION_THROWN(LiveStreamTranscriber(config, decoder.callbacks()), std::invalid_argument);
    config = shortSegments();
    config.minSpeechMs = config.maxSegmentMs + 1;
    QVERIFY_EXCEPTION_THROWN(LiveStreamTranscriber(config, decoder.callbacks()), std::invalid_argument);
    config = shortSegments();
    config.minSpeechMs = config.maxSegmentMs - config.prePaddingMs + 1;
    QVERIFY_EXCEPTION_THROWN(LiveStreamTranscriber(config, decoder.callbacks()), std::invalid_argument);
}

void TestLiveStreamTranscriber::realModelKeepsConfirmedTextAcrossStreams()
{
    const auto model = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (model.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL to test real stream reopening");
    const QString wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    QVERIFY(!wav.isEmpty());
    const auto pcm = CrispSession::readWavF32(wav.toStdString());
    QVERIFY(!pcm.empty());
    std::vector<int16_t> input;
    for (float sample : pcm) input.push_back(static_cast<int16_t>(sample * 32768.0f));
    CrispSession session(model.toStdString(), "qwen3", 8);
    QVERIFY(session.isLoaded());
    session.setLanguage("ko");
    size_t opens = 0;
    size_t closes = 0;
    size_t segmentSamples = 0;
    size_t maxSamples = 0;
    LiveStreamConfig config;
    config.maxSegmentMs = 4000;
    config.endSilenceMs = 300;
    LiveStreamDecoder decoder;
    decoder.open = [&] {
        if (!session.openStream(500, 500, 800)) throw std::runtime_error("open failed");
        ++opens;
        segmentSamples = 0;
    };
    decoder.feed = [&](const int16_t* audio, size_t count) -> std::optional<std::string> {
        segmentSamples += count;
        maxSamples = std::max(maxSamples, segmentSamples);
        std::vector<float> floats;
        for (size_t i = 0; i < count; ++i) floats.push_back(audio[i] / 32768.0f);
        const auto before = session.streamInferenceCount();
        const bool changed = session.feedStream(floats.data(), static_cast<int>(count));
        if (changed || session.streamInferenceCount() > before) return session.streamText();
        return std::nullopt;
    };
    decoder.flush = [&] { return session.flushStream(); };
    decoder.close = [&] { session.closeStream(); ++closes; };
    std::vector<std::string> previews;
    LiveStreamTranscriber stream(config, decoder,
        [&](const std::string& text) { previews.push_back(text); });
    stream.append(input.data(), input.size());
    const std::vector<int16_t> silence(16000, 0);
    stream.append(silence.data(), silence.size());
    QVERIFY(stream.confirmedSegments() > 0);
    QVERIFY(!previews.empty());
    const std::string confirmed = previews.back();
    QVERIFY(!confirmed.empty());
    const size_t firstCount = previews.size();
    stream.append(input.data(), input.size());
    const std::string final = stream.finish();
    QVERIFY(opens >= size_t(2));
    QCOMPARE(opens, closes);
    QVERIFY(maxSamples <= size_t(16000 * 4));
    for (size_t i = firstCount; i < previews.size(); ++i) {
        QVERIFY(previews[i].rfind(confirmed, 0) == 0);
    }
    QVERIFY(final.size() > confirmed.size());
    qInfo("native stream segments=%zu, max_audio_ms=%zu, final_bytes=%zu",
          opens, maxSamples / 16, final.size());
}

QTEST_GUILESS_MAIN(TestLiveStreamTranscriber)
#include "test_live_stream_transcriber.moc"
