// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LiveSegmenterFactory.h"

#include <QTest>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace echoflow;

class TestLiveSegmenterFactory : public QObject {
    Q_OBJECT

private slots:
    void energyIsTheDefault();
    void energyEmitsContinuousSpeechAtTwoSeconds();
    void energyHonorsConfiguredInterval();
    void unknownBackendFailsClosed();
    void missingSileroModelFailsClosed();
};

void TestLiveSegmenterFactory::energyIsTheDefault()
{
    Config config;
    auto segmenter = makeLiveSegmenter(config);
    QCOMPARE(QString::fromStdString(segmenter->name()), QStringLiteral("energy"));
}

void TestLiveSegmenterFactory::energyEmitsContinuousSpeechAtTwoSeconds()
{
    Config config;
    config.streamStepMs = 2000;
    auto segmenter = makeLiveSegmenter(config);
    std::vector<int16_t> speech(2 * 16000 + 3200);
    for (size_t i = 0; i < speech.size(); ++i) {
        speech[i] = (i % 2 == 0) ? int16_t(3000) : int16_t(-3000);
    }

    const auto emitted = segmenter->append(speech.data(), speech.size());
    QCOMPARE(emitted.size(), size_t(1));
    QCOMPARE(emitted.front().sampleCount(), size_t(32000));
}

void TestLiveSegmenterFactory::energyHonorsConfiguredInterval()
{
    for (int interval : {100, 500, 1000, 4000}) {
        Config config;
        config.streamStepMs = interval;
        auto segmenter = makeLiveSegmenter(config);
        const size_t expectedSamples = static_cast<size_t>(interval) * 16;
        std::vector<int16_t> speech(expectedSamples + 320);
        for (size_t i = 0; i < speech.size(); ++i) {
            speech[i] = (i % 2 == 0) ? int16_t(3000) : int16_t(-3000);
        }
        const auto emitted = segmenter->append(speech.data(), speech.size());
        QVERIFY(!emitted.empty());
        QCOMPARE(emitted.front().sampleCount(), expectedSamples);
    }
}

void TestLiveSegmenterFactory::unknownBackendFailsClosed()
{
    Config config;
    config.vadBackend = "unknown";
    QVERIFY_EXCEPTION_THROWN(makeLiveSegmenter(config), std::invalid_argument);
}

void TestLiveSegmenterFactory::missingSileroModelFailsClosed()
{
    Config config;
    config.vadBackend = "silero";
    config.vadModelPath = "/definitely/missing/silero.bin";
    QVERIFY_EXCEPTION_THROWN(makeLiveSegmenter(config), std::runtime_error);
}

QTEST_GUILESS_MAIN(TestLiveSegmenterFactory)
#include "test_live_segmenter_factory.moc"
