// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LiveStreamEndpoint.h"
#include "SileroVadBackend.h"

#include <QTest>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace echoflow;

namespace {
class ProbabilityVad : public IVadBackend {
public:
    size_t maxSamples = 0;
    float threshold = 0.5f;
    float endThreshold = 0.35f;
    int minSpeechMs = 100;
    int silenceMs = 100;
    std::vector<SpeechInterval> detect(const int16_t* pcm, size_t count) override
    {
        maxSamples = std::max(maxSamples, count);
        std::vector<float> probabilities;
        for (size_t i = 0; i < count; i += 320) probabilities.push_back(pcm[i] / 100.0f);
        return segmentByDualThreshold(probabilities.data(), static_cast<int>(probabilities.size()),
            320, count, threshold, endThreshold, minSpeechMs, silenceMs);
    }
    std::string name() const override { return "probability-vad"; }
};
}

class TestLiveStreamEndpoint : public QObject {
    Q_OBJECT
private slots:
    void factoryUsesSelectedBackend();
    void reportsQuietTailBeforeEndpoint_data();
    void reportsQuietTailBeforeEndpoint();
    void quietTailDoesNotValidateShortBurst();
    void quietTailFollowsHysteresisAndPartialWindow();
    void detectsWithModelThresholdsAndDurations();
    void silenceHistoryIsBounded();
    void resetDoesNotReplayOldSpeech();
    void finalAnalysisProcessesIncompleteCadence();
    void invalidBackendAndDurationsFail();
};

void TestLiveStreamEndpoint::factoryUsesSelectedBackend()
{
    Config config;
    QCOMPARE(makeLiveStreamEndpoint(config)->name(), std::string("energy"));
    config.vadBackend = "silero";
    config.vadModelPath = "/definitely/missing/silero.bin";
    QVERIFY_EXCEPTION_THROWN(makeLiveStreamEndpoint(config), std::runtime_error);
    config.vadBackend = "unknown";
    QVERIFY_EXCEPTION_THROWN(makeLiveStreamEndpoint(config), std::invalid_argument);
}

void TestLiveStreamEndpoint::reportsQuietTailBeforeEndpoint_data()
{
    QTest::addColumn<bool>("neural");
    QTest::newRow("energy") << false;
    QTest::newRow("model-probabilities") << true;
}

void TestLiveStreamEndpoint::reportsQuietTailBeforeEndpoint()
{
    QFETCH(bool, neural);
    auto backend = std::make_unique<ProbabilityVad>();
    backend->silenceMs = 500;
    auto endpoint = neural ? makeVadStreamEndpoint(std::move(backend), 100, 8000, 500)
                           : makeEnergyStreamEndpoint(100, 500);
    std::vector<int16_t> voice(3200);
    for (size_t i = 0; i < voice.size(); ++i) voice[i] = i % 2 ? -80 : 80;
    const std::vector<int16_t> quiet(1600, 0);
    const auto speaking = endpoint->append(voice.data(), voice.size());
    QVERIFY(speaking.active);
    QVERIFY(speaking.enoughSpeech);
    QCOMPARE(speaking.trailingSilenceSamples.value_or(123), size_t(0));
    const auto paused = endpoint->append(quiet.data(), quiet.size());
    QVERIFY(paused.active);
    QVERIFY(!paused.ended);
    QCOMPARE(paused.trailingSilenceSamples.value_or(0), size_t(1600));
    const auto resumed = endpoint->append(voice.data(), voice.size());
    QVERIFY(resumed.active);
    QCOMPARE(resumed.trailingSilenceSamples.value_or(123), size_t(0));
}

void TestLiveStreamEndpoint::quietTailDoesNotValidateShortBurst()
{
    auto backend = std::make_unique<ProbabilityVad>();
    backend->silenceMs = 500;
    auto endpoint = makeVadStreamEndpoint(std::move(backend), 100, 8000, 500);
    std::vector<int16_t> audio(3200, 0);
    std::fill(audio.begin(), audio.begin() + 640, 80);
    const auto state = endpoint->append(audio.data(), audio.size());
    QVERIFY(state.active);
    QVERIFY(!state.enoughSpeech);
    QCOMPARE(state.trailingSilenceSamples.value_or(0), size_t(2560));
}

void TestLiveStreamEndpoint::quietTailFollowsHysteresisAndPartialWindow()
{
    const float probabilities[] = {0.8f, 0.8f, 0.2f, 0.4f};
    const auto intervals = segmentByDualThreshold(probabilities, 4, 320, 1100,
                                                   0.5f, 0.35f, 100, 500);
    QCOMPARE(intervals.size(), size_t(1));
    QCOMPARE(intervals.front().trailingSilenceSamples.value_or(0), size_t(460));
}

void TestLiveStreamEndpoint::detectsWithModelThresholdsAndDurations()
{
    auto backend = std::make_unique<ProbabilityVad>();
    auto endpoint = makeVadStreamEndpoint(std::move(backend), 100, 1000, 100);
    QCOMPARE(endpoint->name(), std::string("probability-vad"));
    const std::vector<int16_t> below(1600, 40);
    QVERIFY(!endpoint->append(below.data(), below.size()).active);
    const std::vector<int16_t> above(1600, 80);
    const auto active = endpoint->append(above.data(), above.size());
    QVERIFY(active.active);
    QVERIFY(active.enoughSpeech);
    const std::vector<int16_t> silence(1600, 0);
    const auto ended = endpoint->append(silence.data(), silence.size());
    QVERIFY(ended.ended);
    QVERIFY(!ended.active);
}

void TestLiveStreamEndpoint::silenceHistoryIsBounded()
{
    auto backend = std::make_unique<ProbabilityVad>();
    auto* observed = backend.get();
    auto endpoint = makeVadStreamEndpoint(std::move(backend), 100, 1000, 300);
    const std::vector<int16_t> silence(1600, 0);
    for (int i = 0; i < 600; ++i) {
        QVERIFY(!endpoint->append(silence.data(), silence.size()).active);
    }
    QVERIFY(observed->maxSamples <= size_t((1000 + 1000 + 300) * 16));
}

void TestLiveStreamEndpoint::resetDoesNotReplayOldSpeech()
{
    auto endpoint = makeVadStreamEndpoint(std::make_unique<ProbabilityVad>(), 100, 1000, 100);
    const std::vector<int16_t> speech(1600, 90);
    const std::vector<int16_t> silence(1600, 0);
    QVERIFY(endpoint->append(speech.data(), speech.size()).active);
    QVERIFY(endpoint->append(silence.data(), silence.size()).ended);
    endpoint->reset();
    const auto after = endpoint->append(silence.data(), silence.size());
    QVERIFY(!after.active);
    QVERIFY(!after.ended);
    QVERIFY(endpoint->append(speech.data(), speech.size()).active);
}

void TestLiveStreamEndpoint::finalAnalysisProcessesIncompleteCadence()
{
    auto backend = std::make_unique<ProbabilityVad>();
    backend->minSpeechMs = 20;
    auto endpoint = makeVadStreamEndpoint(std::move(backend), 20, 1000, 300);
    const std::vector<int16_t> speech(640, 80);
    QVERIFY(!endpoint->append(speech.data(), speech.size()).active);
    const auto state = endpoint->finish();
    QVERIFY(state.active);
    QVERIFY(state.enoughSpeech);
}

void TestLiveStreamEndpoint::invalidBackendAndDurationsFail()
{
    QVERIFY_EXCEPTION_THROWN(makeVadStreamEndpoint({}, 100, 1000, 300), std::invalid_argument);
    QVERIFY_EXCEPTION_THROWN(makeVadStreamEndpoint(std::make_unique<ProbabilityVad>(),
        0, 1000, 300), std::invalid_argument);
}

QTEST_GUILESS_MAIN(TestLiveStreamEndpoint)
#include "test_live_stream_endpoint.moc"
