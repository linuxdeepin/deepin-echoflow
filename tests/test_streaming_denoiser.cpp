// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "StreamingDenoiser.h"

#include <QTest>
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <vector>

using namespace echoflow;
namespace {
std::vector<float> speechLike(size_t count)
{
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        const double t = static_cast<double>(i) / 16000;
        const double envelope = 0.1 + 0.07 * std::sin(2 * 3.141592653589793 * 4 * t);
        result[i] = envelope * (std::sin(2 * 3.141592653589793 * 180 * t)
            + 0.3 * std::sin(2 * 3.141592653589793 * 720 * t));
    }
    return result;
}
std::vector<float> clean(const std::vector<float>& input, size_t packet)
{
    StreamingDenoiser denoiser;
    std::vector<float> output;
    for (size_t offset = 0; offset < input.size(); offset += packet) {
        const auto block = denoiser.append(input.data() + offset, std::min(packet, input.size() - offset));
        output.insert(output.end(), block.begin(), block.end());
    }
    const auto tail = denoiser.finish();
    output.insert(output.end(), tail.begin(), tail.end());
    return output;
}
}

class TestStreamingDenoiser : public QObject {
    Q_OBJECT
private slots:
    void preservesLength_data();
    void preservesLength();
    void packetizationKeepsPersistentState();
    void emitsBeforeRecordingEnds();
    void silenceAndInvalidSamplesRemainFinite();
    void finalTailIsNotLost();
    void s16UsesSameProcessing();
    void recordingsHaveIndependentState();
    void concurrentInitializationIsSafe();
};

void TestStreamingDenoiser::preservesLength_data()
{
    QTest::addColumn<int>("samples");
    for (int samples : {0, 1, 15, 159, 160, 161, 319, 320, 321, 800, 1600, 16003}) {
        QTest::newRow(std::to_string(samples).c_str()) << samples;
    }
}
void TestStreamingDenoiser::preservesLength()
{
    QFETCH(int, samples);
    const auto input = speechLike(static_cast<size_t>(samples));
    const auto output = clean(input, 37);
    QCOMPARE(output.size(), input.size());
    for (float value : output) QVERIFY(std::isfinite(value) && value >= -1 && value <= 1);
}

void TestStreamingDenoiser::packetizationKeepsPersistentState()
{
    const auto input = speechLike(8003);
    const auto expected = clean(input, input.size());
    for (size_t packet : {size_t(1), size_t(7), size_t(160), size_t(320), size_t(1000)}) {
        const auto actual = clean(input, packet);
        QCOMPARE(actual.size(), expected.size());
        double maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) maximum = std::max(maximum, double(std::abs(actual[i] - expected[i])));
        QVERIFY2(maximum < 1e-5, qPrintable(QString::number(maximum, 'g', 12)));
    }
}

void TestStreamingDenoiser::emitsBeforeRecordingEnds()
{
    StreamingDenoiser denoiser;
    const auto input = speechLike(1600);
    const auto output = denoiser.append(input.data(), input.size());
    // Less than 30 ms of buffered audio, no wait for a whole utterance.
    QVERIFY2(output.size() >= 1600 - 480, qPrintable(QString::number(output.size())));
    QVERIFY(output.size() < input.size());
    QCOMPARE(output.size() + denoiser.finish().size(), input.size());
    QVERIFY(denoiser.finish().empty());
    QVERIFY_EXCEPTION_THROWN(denoiser.append(input.data(), 1), std::logic_error);
}

void TestStreamingDenoiser::silenceAndInvalidSamplesRemainFinite()
{
    const auto silence = clean(std::vector<float>(3200), 173);
    for (float value : silence) QCOMPARE(value, 0.0f);
    auto bad = std::vector<float>(3200, 0);
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    bad[1] = std::numeric_limits<float>::infinity();
    // Sanitize at the front end before the resampler's persistent state.
    const auto output = clean(bad, 320);
    for (float value : output) QVERIFY(std::isfinite(value));
}

void TestStreamingDenoiser::finalTailIsNotLost()
{
    auto input = speechLike(16000);
    std::fill(input.begin(), input.end() - 800, 0);
    const auto output = clean(input, 320);
    double energy = 0;
    for (size_t i = output.size() - 800; i < output.size(); ++i) energy += output[i] * output[i];
    QVERIFY(energy > 1e-7); // STOP must drain RNNoise lookahead, not zero the final syllable.
}

void TestStreamingDenoiser::s16UsesSameProcessing()
{
    auto input = speechLike(3201);
    std::vector<int16_t> s16;
    for (auto& sample : input) {
        const auto integer = static_cast<int16_t>(std::lround(sample * 32768));
        s16.push_back(integer);
        sample = integer / 32768.0f;
    }
    const auto expected = clean(input, 137);
    StreamingDenoiser denoiser;
    std::vector<int16_t> output;
    for (size_t offset = 0; offset < s16.size(); offset += 137) {
        const auto part = denoiser.append(s16.data() + offset, std::min<size_t>(137, s16.size() - offset));
        output.insert(output.end(), part.begin(), part.end());
    }
    const auto tail = denoiser.finishS16();
    output.insert(output.end(), tail.begin(), tail.end());
    QCOMPARE(output.size(), input.size());
    for (size_t i = 0; i < output.size(); ++i) {
        QCOMPARE(output[i], static_cast<int16_t>(std::lround(std::clamp(expected[i] * 32768, -32768.0f, 32767.0f))));
    }
}

void TestStreamingDenoiser::recordingsHaveIndependentState()
{
    const auto input = speechLike(4800);
    const auto first = clean(input, 320);
    const auto silence = clean(std::vector<float>(4800, 0), 320);
    for (const auto value : silence) QCOMPARE(value, 0.0f);
    const auto next = clean(input, 320);
    QCOMPARE(next, first);
}

void TestStreamingDenoiser::concurrentInitializationIsSafe()
{
    std::vector<std::future<std::vector<float>>> tasks;
    for (int i = 0; i < 4; ++i) {
        tasks.push_back(std::async(std::launch::async, [] { return clean(speechLike(3200), 320); }));
    }
    const auto expected = tasks.front().get();
    for (size_t i = 1; i < tasks.size(); ++i) QCOMPARE(tasks[i].get(), expected);
}

QTEST_GUILESS_MAIN(TestStreamingDenoiser)
#include "test_streaming_denoiser.moc"
