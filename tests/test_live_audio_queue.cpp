// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LiveAudioQueue.h"
#include "LiveStreamTranscriber.h"

#include <QTest>
#include <vector>
#include <algorithm>
#include <optional>

using namespace echoflow;

class TestLiveAudioQueue : public QObject {
    Q_OBJECT
private slots:
    void largeBacklogKeepsSampleOrder();
    void appendDuringSpillPreservesOrder();
    void cancelClearsBothBuffers();
    void waitingTimeSurvivesSpoolingAndPartialReads();
    void backlogStillConfirmsEverySegment();
};

void TestLiveAudioQueue::largeBacklogKeepsSampleOrder()
{
    LiveAudioQueue queue;
    std::vector<int16_t> expected;
    for (int packet = 0; packet < 200; ++packet) {
        std::vector<int16_t> samples(1600);
        for (size_t i = 0; i < samples.size(); ++i) samples[i] = static_cast<int16_t>(packet * 17 + i);
        expected.insert(expected.end(), samples.begin(), samples.end());
        queue.append(std::move(samples));
    }
    QCOMPARE(queue.size(), size_t(16000 * 20));
    QVERIFY(queue.spilling());
    std::vector<int16_t> actual;
    while (!queue.empty()) {
        const auto samples = queue.take();
        QVERIFY(!samples.empty());
        actual.insert(actual.end(), samples.begin(), samples.end());
    }
    QCOMPARE(actual, expected);
    QVERIFY(!queue.spilling());
    QVERIFY(queue.take().empty());
}

void TestLiveAudioQueue::appendDuringSpillPreservesOrder()
{
    LiveAudioQueue queue(3);
    queue.append({1, 2, 3});
    queue.append({4, 5});
    QCOMPARE(queue.take(), (std::vector<int16_t>{1, 2, 3}));
    queue.append({6, 7});
    QCOMPARE(queue.take(), (std::vector<int16_t>{4, 5, 6, 7}));
    QVERIFY(queue.empty());
    QVERIFY(!queue.spilling());
    queue.append({8, 9});
    QCOMPARE(queue.take(), (std::vector<int16_t>{8, 9}));
}

void TestLiveAudioQueue::waitingTimeSurvivesSpoolingAndPartialReads()
{
    using namespace std::chrono_literals;
    const auto start = LiveAudioQueue::Clock::time_point{};
    LiveAudioQueue queue(3);
    queue.append({1, 2, 3}, start);
    queue.append(std::vector<int16_t>(4000, 4), start + 100ms);
    queue.append(std::vector<int16_t>(3000, 5), start + 200ms);
    std::chrono::milliseconds waited;
    QCOMPARE(queue.take(&waited, start + 1s), (std::vector<int16_t>{1, 2, 3}));
    QCOMPARE(waited, 1000ms);
    QCOMPARE(queue.take(&waited, start + 1s), std::vector<int16_t>(3200, 4));
    QCOMPARE(waited, 900ms);
    const auto mixed = queue.take(&waited, start + 1100ms);
    QCOMPARE(mixed.size(), size_t(3200));
    QVERIFY(std::all_of(mixed.begin(), mixed.begin() + 800, [](int value) { return value == 4; }));
    QVERIFY(std::all_of(mixed.begin() + 800, mixed.end(), [](int value) { return value == 5; }));
    QCOMPARE(waited, 1000ms);
    QCOMPARE(queue.take(&waited, start + 1200ms), std::vector<int16_t>(600, 5));
    QCOMPARE(waited, 1000ms);
    QVERIFY(queue.empty());
    QVERIFY(!queue.spilling());
    QVERIFY(queue.take(&waited, start + 2s).empty());
    QCOMPARE(waited, 0ms);
    queue.append({6}, start + 3s);
    queue.clear();
    queue.append({7}, start + 4s);
    QCOMPARE(queue.take(&waited, start + 4500ms), (std::vector<int16_t>{7}));
    QCOMPARE(waited, 500ms);
}

void TestLiveAudioQueue::cancelClearsBothBuffers()
{
    LiveAudioQueue queue(3);
    queue.append({1, 2, 3});
    queue.append({4, 5});
    QVERIFY(queue.spilling());
    queue.clear();
    QVERIFY(queue.empty());
    QVERIFY(!queue.spilling());
    QVERIFY(queue.take().empty());
    queue.append({9});
    QCOMPARE(queue.take(), (std::vector<int16_t>{9}));
}

void TestLiveAudioQueue::backlogStillConfirmsEverySegment()
{
    LiveAudioQueue queue(3200);
    std::vector<int16_t> expected(16000 * 20);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = i % 2 ? -5000 : 5000;
    for (size_t offset = 0; offset < expected.size(); offset += 3200) {
        queue.append(std::vector<int16_t>(expected.begin() + offset, expected.begin() + offset + 3200));
    }
    QVERIFY(queue.spilling());
    std::vector<int16_t> decoded;
    int segments = 0;
    LiveStreamDecoder decoder;
    decoder.open = [] {};
    decoder.close = [] {};
    decoder.feed = [&](const int16_t* samples, size_t count) {
        decoded.insert(decoded.end(), samples, samples + count);
        return std::nullopt;
    };
    decoder.flush = [&] { return std::string("段") + std::to_string(++segments); };
    LiveStreamTranscriber transcriber({}, std::move(decoder));
    while (!queue.empty()) {
        const auto samples = queue.take();
        transcriber.append(samples.data(), samples.size());
    }
    const auto text = transcriber.finish();
    QCOMPARE(decoded, expected);
    QCOMPARE(segments, 3);
    QCOMPARE(text, std::string("段1段2段3"));
}

QTEST_GUILESS_MAIN(TestLiveAudioQueue)
#include "test_live_audio_queue.moc"
