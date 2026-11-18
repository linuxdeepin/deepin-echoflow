// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QtTest/QtTest>
#include <QScopeGuard>

#include "../third_party/crispasr/src/core/qwen3_stream.h"
#include "../third_party/crispasr/src/core/qwen3_stream_model.h"
#include "CrispSession.h"
#include "LivePreviewScheduler.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using core_qwen3_stream::BufferedStream;

namespace {

struct Decodes {
    std::vector<std::vector<float>> audio;
    std::vector<std::string> prefixes;
    std::vector<int> budgets;
};

core_qwen3_stream::Hooks mockHooks(Decodes& decodes)
{
    core_qwen3_stream::Hooks hooks;
    hooks.encode = [](const std::string& text) {
        return std::vector<int32_t>(text.begin(), text.end());
    };
    hooks.decode_bytes = [](const std::vector<int32_t>& ids) {
        return std::string(ids.begin(), ids.end());
    };
    hooks.generate = [&decodes](const std::vector<float>& audio,
                                const std::string& prefix, int budget) {
        decodes.audio.push_back(audio);
        decodes.prefixes.push_back(prefix);
        decodes.budgets.push_back(budget);
        return (prefix.empty() ? "language English<asr_text>" : " ")
            + std::string("word") + std::to_string(decodes.audio.size());
    };
    return hooks;
}

std::vector<float> samples(size_t count)
{
    std::vector<float> audio(count);
    for (size_t i = 0; i < count; ++i)
        audio[i] = static_cast<float>(i % 37) / 37.0f;
    return audio;
}

} // namespace

class TestQwenStream : public QObject {
    Q_OBJECT

private slots:
    void boundaries_data();
    void boundaries();
    void packetization_data();
    void packetization();
    void deferredPreviewCoalescesWithoutLosingAudio_data();
    void deferredPreviewCoalescesWithoutLosingAudio();
    void deferredFlushDecodesOnce_data();
    void deferredFlushDecodesOnce();
    void flushShortTail();
    void flushAtBoundary();
    void recipeDefaults();
    void inferenceCountIncludesUnchangedOutput();
    void adaptiveAdmissionsCoalescePendingAudio();
    void tokenLimitAppliesToUpdatesAndFlush_data();
    void tokenLimitAppliesToUpdatesAndFlush();
    void sessionTokenLimitMatchesReferenceWhenModelAvailable();
    void rejectsInvalidIntervals();
    void fullSegmentModeRevisesEarlyText_data();
    void fullSegmentModeRevisesEarlyText();
    void fullSegmentFlushAtBoundary_data();
    void fullSegmentFlushAtBoundary();
    void fullSegmentDeferredFlushDecodesOnce();
    void fullSegmentMatchesIndependentDecodeWhenModelAvailable();
    void prefixKvReuseRequiresMatchingStreamState();
    void failedDecodeClearsProfile();
    void persistentPrefixInvalidatesOnKvMutationWhenModelAvailable();
    void reopenedStreamsMatchFreshStreamsWhenModelAvailable();
    void prefixKvReuseMatchesFullPrefillWhenModelAvailable();
};

void TestQwenStream::boundaries_data()
{
    QTest::addColumn<bool>("r2t2");
    QTest::addColumn<int>("interval");
    for (bool r2t2 : {false, true}) {
        for (int interval : {100, 160, 500, 1000, 2000, 4000}) {
            const auto name = QByteArray(r2t2 ? "r2t2-" : "stock-")
                + QByteArray::number(interval);
            QTest::newRow(name.constData()) << r2t2 << interval;
        }
    }
}

void TestQwenStream::boundaries()
{
    QFETCH(bool, r2t2);
    QFETCH(int, interval);
    BufferedStream stream(r2t2, interval);
    Decodes decodes;
    stream.hooks = mockHooks(decodes);
    std::vector<std::string> texts;
    const auto publish = [&](const std::string& text) { texts.push_back(text); };
    const size_t step = static_cast<size_t>(interval) * 16;
    const size_t first = step + (r2t2 ? 2560 : 0);
    const auto audio = samples(first + step);

    stream.append(audio.data(), first - 1, false, publish);
    QVERIFY(decodes.audio.empty());
    stream.append(audio.data() + first - 1, 1, false, publish);
    QCOMPARE(decodes.audio.size(), size_t(1));
    QCOMPARE(decodes.audio.front().size(), first);
    stream.append(audio.data() + first, step - 1, false, publish);
    QCOMPARE(decodes.audio.size(), size_t(1));
    stream.append(audio.data() + audio.size() - 1, 1, false, publish);
    QCOMPARE(decodes.audio.size(), size_t(2));
    QVERIFY(decodes.audio.back() == audio);
    QCOMPARE(texts.size(), size_t(2));
    QVERIFY(!texts.back().empty());
    QVERIFY(stream.pending.empty());
    QCOMPARE(decodes.budgets.front(), r2t2 ? int(first / 1280) : 256);
}

void TestQwenStream::packetization_data()
{
    boundaries_data();
}

void TestQwenStream::packetization()
{
    QFETCH(bool, r2t2);
    QFETCH(int, interval);
    BufferedStream bulk(r2t2, interval);
    BufferedStream packets(r2t2, interval);
    Decodes bulkDecodes;
    Decodes packetDecodes;
    bulk.hooks = mockHooks(bulkDecodes);
    packets.hooks = mockHooks(packetDecodes);
    const size_t step = static_cast<size_t>(interval) * 16;
    const auto audio = samples(3 * step + (r2t2 ? 2560 : 0) + 777);
    std::vector<std::string> bulkTexts;
    std::vector<std::string> packetTexts;
    const auto bulkPublish = [&](const std::string& text) { bulkTexts.push_back(text); };
    const auto packetPublish = [&](const std::string& text) { packetTexts.push_back(text); };

    bulk.append(audio.data(), audio.size(), false, bulkPublish);
    const size_t sizes[] = {1, 319, 17, 2048, 63, 640};
    size_t offset = 0;
    size_t packet = 0;
    while (offset < audio.size()) {
        const size_t count = std::min(sizes[packet++ % 6], audio.size() - offset);
        packets.append(audio.data() + offset, count, false, packetPublish);
        offset += count;
    }
    QCOMPARE(bulkDecodes.audio.size(), size_t(3));
    bulk.append(nullptr, 0, true, bulkPublish);
    packets.append(nullptr, 0, true, packetPublish);
    QCOMPARE(bulkDecodes.audio.size(), size_t(4));
    QVERIFY(bulkDecodes.audio == packetDecodes.audio);
    QVERIFY(bulkDecodes.prefixes == packetDecodes.prefixes);
    QVERIFY(bulkDecodes.budgets == packetDecodes.budgets);
    QVERIFY(bulkTexts == packetTexts);
    QVERIFY(bulkDecodes.audio.back() == audio);
    QVERIFY(!bulkTexts.back().empty());
    QVERIFY(packets.pending.empty());
}

void TestQwenStream::deferredPreviewCoalescesWithoutLosingAudio_data()
{
    QTest::addColumn<bool>("r2t2");
    QTest::newRow("stock") << false;
    QTest::newRow("r2t2") << true;
}

void TestQwenStream::deferredPreviewCoalescesWithoutLosingAudio()
{
    QFETCH(bool, r2t2);
    BufferedStream stream(r2t2, 500);
    Decodes decodes;
    stream.hooks = mockHooks(decodes);
    const auto audio = samples(16000 * 4);
    size_t updates = 0;
    const auto publish = [&](const std::string&) { ++updates; };
    stream.set_live_decode(false);
    for (size_t offset = 0; offset < audio.size(); offset += 320) {
        stream.append(audio.data() + offset, 320, false, publish);
    }
    QVERIFY(decodes.audio.empty());
    QCOMPARE(updates, size_t(0));
    stream.set_live_decode(true);
    stream.append(nullptr, 0, false, publish);
    QCOMPARE(decodes.audio.size(), size_t(1));
    QCOMPARE(decodes.audio.front(), audio);
    QCOMPARE(updates, size_t(1));
    QVERIFY(decodes.budgets.front() >= 50);
    QVERIFY(stream.pending.empty());
    stream.append(audio.data(), 8000, false, publish);
    QCOMPARE(decodes.audio.size(), size_t(2));
    QCOMPARE(decodes.audio.back().size(), audio.size() + 8000);
}

void TestQwenStream::deferredFlushDecodesOnce_data()
{
    QTest::addColumn<bool>("r2t2");
    QTest::addColumn<int>("limit");
    for (bool r2t2 : {false, true}) {
        for (int limit : {0, 7}) {
            const auto name = QByteArray(r2t2 ? "r2t2-" : "stock-") + QByteArray::number(limit);
            QTest::newRow(name.constData()) << r2t2 << limit;
        }
    }
}

void TestQwenStream::deferredFlushDecodesOnce()
{
    QFETCH(bool, r2t2);
    QFETCH(int, limit);
    BufferedStream stream(r2t2, 500, limit);
    Decodes decodes;
    stream.hooks = mockHooks(decodes);
    const auto audio = samples(16000 * 8);
    size_t updates = 0;
    const auto publish = [&](const std::string&) { ++updates; };
    stream.set_live_decode(false);
    stream.append(audio.data(), audio.size(), false, publish);
    QVERIFY(decodes.audio.empty());
    stream.append(nullptr, 0, true, publish);
    QCOMPARE(decodes.audio.size(), size_t(1));
    QCOMPARE(decodes.audio.front(), audio);
    QCOMPARE(updates, size_t(1));
    QCOMPARE(decodes.budgets.front(), limit ? limit : (r2t2 ? 100 : 256));
    QVERIFY(stream.pending.empty());
}

void TestQwenStream::flushShortTail()
{
    for (bool r2t2 : {false, true}) {
        BufferedStream stream(r2t2, 500);
        Decodes decodes;
        stream.hooks = mockHooks(decodes);
        std::string text;
        const auto publish = [&](const std::string& value) { text = value; };
        const auto audio = samples(257);
        stream.append(audio.data(), audio.size(), false, publish);
        QVERIFY(decodes.audio.empty());
        stream.append(nullptr, 0, true, publish);
        QCOMPARE(decodes.audio.size(), size_t(1));
        QVERIFY(decodes.audio.back() == audio);
        QVERIFY(!text.empty());
        const auto finalText = text;
        stream.append(nullptr, 0, true, publish);
        QCOMPARE(decodes.audio.size(), size_t(1));
        QVERIFY(text == finalText);
    }
}

void TestQwenStream::flushAtBoundary()
{
    for (bool r2t2 : {false, true}) {
        BufferedStream stream(r2t2, 500);
        Decodes decodes;
        stream.hooks = mockHooks(decodes);
        const auto publish = [](const std::string&) {};
        stream.append(nullptr, 0, true, publish);
        QCOMPARE(stream.inference_count, int64_t(0));
        QVERIFY(decodes.audio.empty());
        const auto audio = samples(8000 + (r2t2 ? 2560 : 0));
        stream.append(audio.data(), audio.size(), false, publish);
        QCOMPARE(decodes.audio.size(), size_t(1));
        QCOMPARE(stream.inference_count, int64_t(1));
        stream.append(nullptr, 0, true, publish);
        QCOMPARE(decodes.audio.size(), size_t(1));
        QCOMPARE(stream.inference_count, int64_t(1));
    }
}

void TestQwenStream::adaptiveAdmissionsCoalescePendingAudio()
{
    using namespace std::chrono_literals;
    for (bool r2t2 : {false, true}) {
        auto now = echoflow::LivePreviewScheduler::Clock::time_point{};
        echoflow::LivePreviewScheduler scheduler(500, now);
        BufferedStream stream(r2t2, 500);
        Decodes decodes;
        stream.hooks = mockHooks(decodes);
        const auto audio = samples(48000 + 123);
        const auto publish = [](const std::string&) {};
        for (size_t offset = 0; offset < audio.size();) {
            const size_t count = std::min(size_t(320), audio.size() - offset);
            now += 20ms;
            stream.set_live_decode(scheduler.admit(count, false, false, now));
            const auto before = stream.inference_count;
            stream.append(audio.data() + offset, count, false, publish);
            if (stream.inference_count > before) {
                now += 800ms;
                scheduler.decoded(800ms, now);
            }
            offset += count;
        }
        stream.set_live_decode(false);
        stream.append(nullptr, 0, true, publish);
        QVERIFY(decodes.audio.size() <= size_t(4));
        QVERIFY(decodes.audio.size() >= size_t(2));
        // Every update contains precisely the original segment prefix;
        // finalization contains all samples, including the sub-frame tail.
        for (const auto& decoded : decodes.audio) {
            QVERIFY(decoded.size() <= audio.size());
            QVERIFY(std::equal(decoded.begin(), decoded.end(), audio.begin()));
        }
        QCOMPARE(decodes.audio.back(), audio);
        QCOMPARE(stream.inference_count, static_cast<int64_t>(decodes.audio.size()));
    }
}

void TestQwenStream::inferenceCountIncludesUnchangedOutput()
{
    for (bool r2t2 : {false, true}) {
        BufferedStream stream(r2t2, 500);
        Decodes decodes;
        stream.hooks = mockHooks(decodes);
        stream.hooks.generate = [](const std::vector<float>&, const std::string&, int) {
            return std::string();
        };
        const auto audio = samples(16000);
        const auto publish = [](const std::string&) {};
        QCOMPARE(stream.inference_count, int64_t(0));
        stream.set_live_decode(false);
        stream.append(audio.data(), audio.size(), false, publish);
        QCOMPARE(stream.inference_count, int64_t(0));
        stream.set_live_decode(true);
        stream.append(nullptr, 0, false, publish);
        QCOMPARE(stream.inference_count, int64_t(1));
        stream.append(audio.data(), audio.size(), false, publish);
        QCOMPARE(stream.inference_count, int64_t(3));
        stream.set_live_decode(false);
        stream.append(audio.data(), 123, false, publish);
        stream.append(nullptr, 0, true, publish);
        QCOMPARE(stream.inference_count, int64_t(4));
    }
}

void TestQwenStream::recipeDefaults()
{
    BufferedStream stock(false);
    BufferedStream r2t2(true);
    QCOMPARE(stock.chunk, 32000);
    QVERIFY(stock.algorithm.cfg.stock_qwen);
    QCOMPARE(stock.algorithm.cfg.unfixed_chunk_num, 2);
    QCOMPARE(stock.algorithm.cfg.unfixed_token_num, 5);
    QCOMPARE(r2t2.schedule.next_chunk_samples(), 5120);
    r2t2.schedule.update(U"test");
    QCOMPARE(r2t2.schedule.next_chunk_samples(), 2560);
    QVERIFY(!r2t2.algorithm.cfg.stock_qwen);
    QCOMPARE(r2t2.algorithm.cfg.unfixed_chunk_num, 0);
    QCOMPARE(r2t2.algorithm.cfg.unfixed_token_num, 1);
}

void TestQwenStream::tokenLimitAppliesToUpdatesAndFlush_data()
{
    QTest::addColumn<bool>("r2t2");
    QTest::addColumn<int>("limit");
    for (bool r2t2 : {false, true}) {
        for (int limit : {0, 1, 3, 512}) {
            const QByteArray name = QByteArray(r2t2 ? "r2t2-" : "stock-")
                + QByteArray::number(limit);
            QTest::newRow(name.constData()) << r2t2 << limit;
        }
    }
}

void TestQwenStream::tokenLimitAppliesToUpdatesAndFlush()
{
    QFETCH(bool, r2t2);
    QFETCH(int, limit);
    BufferedStream baseline(r2t2, 160);
    BufferedStream limited(r2t2, 160, limit);
    Decodes original;
    Decodes capped;
    baseline.hooks = mockHooks(original);
    limited.hooks = mockHooks(capped);
    const auto audio = samples(3 * 2560 + (r2t2 ? 2560 : 0) + 137);
    const auto publish = [](const std::string&) {};
    baseline.append(audio.data(), audio.size(), false, publish);
    limited.append(audio.data(), audio.size(), false, publish);
    QCOMPARE(capped.budgets.size(), size_t(3));
    baseline.append(nullptr, 0, true, publish);
    limited.append(nullptr, 0, true, publish);
    QCOMPARE(capped.budgets.size(), size_t(4));
    for (size_t i = 0; i < capped.budgets.size(); ++i) {
        const int expected = limit == 0 ? original.budgets[i]
            : (r2t2 ? std::min(original.budgets[i], limit) : limit);
        QCOMPARE(capped.budgets[i], expected);
    }
    QVERIFY_EXCEPTION_THROWN(BufferedStream(r2t2, 160, -1), std::invalid_argument);
}

void TestQwenStream::sessionTokenLimitMatchesReferenceWhenModelAvailable()
{
    const QByteArray modelPath = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (modelPath.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL to test session token limits");
    const QString wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    QVERIFY(!wav.isEmpty());
    const auto pcm = echoflow::CrispSession::readWavF32(wav.toStdString());
    QVERIFY(pcm.size() >= size_t(8000));
    std::string expectedUpdate;
    std::string expectedTail;
    {
        auto params = qwen3_asr_context_default_params();
        params.n_threads = 2;
        params.verbosity = 0;
        std::unique_ptr<qwen3_asr_context, decltype(&qwen3_asr_free)> reference(
            qwen3_asr_init_from_file(modelPath.constData(), params), &qwen3_asr_free);
        QVERIFY(reference != nullptr);
        for (size_t count : {size_t(8000), size_t(4000)}) {
            const std::vector<float> audio(pcm.begin(), pcm.begin() + count);
            bool success = false;
            const auto raw = core_qwen3_stream_model::generate(reference.get(), false, audio,
                "language Korean<asr_text>", 1, &success);
            QVERIFY(success);
            const auto parsed = core_qwen3_stream::parse_asr_output(
                core_qwen3_stream::utf8_decode_replace(raw), U"Korean");
            const auto text = core_qwen3_stream::utf8_encode(parsed.text);
            if (count == 8000) expectedUpdate = text;
            else expectedTail = text;
        }
    }
    echoflow::CrispSession session(modelPath.toStdString(), "qwen3", 2);
    QVERIFY(session.isLoaded());
    session.setLanguage("ko");
    session.setMaxNewTokens(1);
    QVERIFY(session.openStream(500, 500, 800));
    QCOMPARE(session.streamInferenceCount(), std::int64_t(0));
    session.feedStream(pcm.data(), 8000);
    QCOMPARE(session.streamInferenceCount(), std::int64_t(1));
    QCOMPARE(session.streamText(), expectedUpdate);
    session.closeStream();
    QVERIFY(session.openStream(500, 500, 800));
    QVERIFY(!session.feedStream(pcm.data(), 4000));
    QCOMPARE(session.streamInferenceCount(), std::int64_t(0));
    QCOMPARE(session.flushStream(), expectedTail);
    QCOMPARE(session.streamInferenceCount(), std::int64_t(1));
    session.closeStream();
    QVERIFY(session.openStream(100, 100, 800));
    session.setStreamLiveDecode(false);
    QVERIFY(!session.feedStream(pcm.data(), 8000));
    QVERIFY(session.streamText().empty());
    QCOMPARE(session.flushStream(), expectedUpdate);
    session.closeStream();
    QVERIFY(session.openStream(500, 500, 800));
    session.setStreamLiveDecode(false);
    QVERIFY(!session.feedStream(pcm.data(), 4000));
    session.setStreamLiveDecode(true);
    session.feedStream(pcm.data() + 4000, 4000);
    QCOMPARE(session.streamText(), expectedUpdate);
}

void TestQwenStream::rejectsInvalidIntervals()
{
    for (int interval : {-1, 1, 99, 4001, 1000000}) {
        QVERIFY_EXCEPTION_THROWN(BufferedStream(false, interval), std::invalid_argument);
        QVERIFY_EXCEPTION_THROWN(BufferedStream(true, interval), std::invalid_argument);
    }
}

void TestQwenStream::fullSegmentModeRevisesEarlyText_data()
{
    QTest::addColumn<bool>("r2t2");
    QTest::newRow("stock") << false;
    QTest::newRow("r2t2") << true;
}

void TestQwenStream::fullSegmentModeRevisesEarlyText()
{
    QFETCH(bool, r2t2);
    BufferedStream stream(r2t2, 500);
    stream.set_full_segment_decode(true);
    Decodes decodes;
    stream.hooks = mockHooks(decodes);
    stream.hooks.generate = [&](const std::vector<float>& audio, const std::string& prefix, int budget) {
        decodes.audio.push_back(audio);
        decodes.prefixes.push_back(prefix);
        decodes.budgets.push_back(budget);
        return std::string("language English<asr_text>")
            + (decodes.audio.size() == 1 ? "wrong words at the beginning of this sentence"
                                         : "corrected whole sentence");
    };
    std::string latest;
    const auto publish = [&](const std::string& text) { latest = text; };
    const auto audio = samples(8000 + (r2t2 ? 2560 : 0));
    stream.append(audio.data(), audio.size(), false, publish);
    QVERIFY(latest.find("wrong") == 0);
    for (int update = 0; update < 3; ++update) {
        stream.append(audio.data(), 8000, false, publish);
    }
    QCOMPARE(latest, std::string("corrected whole sentence"));
    for (const auto& prefix : decodes.prefixes) QVERIFY(prefix.empty());
    for (const auto budget : decodes.budgets) QCOMPARE(budget, 256);
}

void TestQwenStream::fullSegmentFlushAtBoundary_data()
{
    QTest::addColumn<bool>("r2t2");
    QTest::addColumn<int>("tokenLimit");
    for (const bool r2t2 : {false, true}) {
        for (const int limit : {0, 7}) {
            const auto name = QByteArray(r2t2 ? "r2t2-" : "stock-") + QByteArray::number(limit);
            QTest::newRow(name.constData()) << r2t2 << limit;
        }
    }
}

void TestQwenStream::fullSegmentFlushAtBoundary()
{
    QFETCH(bool, r2t2);
    QFETCH(int, tokenLimit);
    BufferedStream stream(r2t2, 500, tokenLimit);
    stream.set_full_segment_decode(true);
    Decodes decodes;
    stream.hooks = mockHooks(decodes);
    const auto publish = [](const std::string&) {};
    stream.append(nullptr, 0, true, publish);
    QCOMPARE(stream.inference_count, int64_t(0));
    const auto audio = samples(8000 + (r2t2 ? 2560 : 0));
    stream.append(audio.data(), audio.size(), false, publish);
    QVERIFY(stream.pending.empty());
    QCOMPARE(stream.inference_count, int64_t(1));
    stream.append(nullptr, 0, true, publish);
    QCOMPARE(stream.inference_count, int64_t(2));
    QCOMPARE(decodes.audio.size(), size_t(2));
    QVERIFY(decodes.audio.back() == audio);
    QVERIFY(decodes.prefixes.back().empty());
    QCOMPARE(decodes.budgets.back(), tokenLimit ? tokenLimit : 256);
}

void TestQwenStream::fullSegmentDeferredFlushDecodesOnce()
{
    for (const bool r2t2 : {false, true}) {
        BufferedStream stream(r2t2, 500);
        stream.set_full_segment_decode(true);
        stream.set_live_decode(false);
        Decodes decodes;
        stream.hooks = mockHooks(decodes);
        const auto audio = samples(8 * 16000 + 123);
        const auto publish = [](const std::string&) {};
        stream.append(audio.data(), audio.size(), false, publish);
        QVERIFY(decodes.audio.empty());
        stream.append(nullptr, 0, true, publish);
        QCOMPARE(stream.inference_count, int64_t(1));
        QVERIFY(decodes.audio.back() == audio);
        QVERIFY(decodes.prefixes.back().empty());
    }
}

void TestQwenStream::fullSegmentMatchesIndependentDecodeWhenModelAvailable()
{
    const auto modelPath = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (modelPath.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL for full-segment C ABI parity");
    const QString wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    const auto audio = echoflow::CrispSession::readWavF32(wav.toStdString());
    QVERIFY(audio.size() >= size_t(24000));
    auto params = qwen3_asr_context_default_params();
    params.n_threads = 2;
    params.use_gpu = false;
    params.verbosity = 0;
    std::unique_ptr<qwen3_asr_context, decltype(&qwen3_asr_free)> reference(
        qwen3_asr_init_from_file(modelPath.constData(), params), &qwen3_asr_free);
    QVERIFY(reference != nullptr);
    bool ok = false;
    const auto raw = core_qwen3_stream_model::generate(reference.get(), false,
        std::vector<float>(audio.begin(), audio.begin() + 24000),
        "language Korean<asr_text>", 64, &ok);
    QVERIFY(ok);
    const auto expected = core_qwen3_stream::utf8_encode(core_qwen3_stream::parse_asr_output(
        core_qwen3_stream::utf8_decode_replace(raw), U"Korean").text);
    echoflow::CrispSession session(modelPath.toStdString(), "qwen3", 2);
    QVERIFY(session.isLoaded());
    session.setLanguage("ko");
    session.setMaxNewTokens(64);
    QVERIFY(session.openStream(500, 500, 800));
    session.setStreamFullSegmentDecode(true);
    for (int offset = 0; offset < 24000; offset += 8000) {
        session.feedStream(audio.data() + offset, 8000);
        if (offset == 0) session.refreshStream(); // A pause checkpoint must allow later audio.
    }
    QCOMPARE(session.streamInferenceCount(), std::int64_t(4));
    QCOMPARE(session.flushStream(), expected);
    QCOMPARE(session.streamInferenceCount(), std::int64_t(5));
    QCOMPARE(session.flushStream(), expected);
    QCOMPARE(session.streamInferenceCount(), std::int64_t(5));
    QVERIFY_EXCEPTION_THROWN(session.refreshStream(), std::runtime_error);
    session.closeStream();
    QVERIFY(session.openStream(500, 500, 800));
    QVERIFY_EXCEPTION_THROWN(session.refreshStream(), std::runtime_error);
    QCOMPARE(session.streamInferenceCount(), std::int64_t(0));
}

void TestQwenStream::prefixKvReuseRequiresMatchingStreamState()
{
    using namespace core_qwen3_stream_model;
    qwen3_asr_context* context = reinterpret_cast<qwen3_asr_context*>(0x1);
    const std::vector<int32_t> prompt = {10, 11, 12, 99, 99, 99, 13, 14};
    const int prefixLength = stable_prefix_length(prompt, 99);
    QCOMPARE(prefixLength, 3);

    PrefixKvCache cache;
    QVERIFY(!can_reuse_prefix_kv(cache, context, 4096, prompt, prefixLength));
    cache.context = context;
    cache.max_ctx = 4096;
    cache.prefix_ids = {10, 11, 12};
    cache.ready = true;
    QVERIFY(can_reuse_prefix_kv(cache, context, 4096, prompt, prefixLength));
    QVERIFY(!can_reuse_prefix_kv(cache, context, 4096, prompt, prefixLength, 1));
    cache.revision = 7;
    QVERIFY(can_reuse_prefix_kv(cache, context, 4096, prompt, prefixLength, 7));
    QVERIFY(!can_reuse_prefix_kv(cache, context, 4096, prompt, prefixLength, 8));
    cache.revision = 0;
    QVERIFY(can_reuse_prefix_kv(cache, context, 2048, prompt, prefixLength));
    QVERIFY(!can_reuse_prefix_kv(cache, context, 8192, prompt, prefixLength));

    auto changedPrompt = prompt;
    changedPrompt[1] = 15;
    QVERIFY(!can_reuse_prefix_kv(cache, context, 4096, changedPrompt, prefixLength));
    auto otherContext = reinterpret_cast<qwen3_asr_context*>(0x2);
    QVERIFY(!can_reuse_prefix_kv(cache, otherContext, 4096, prompt, prefixLength));
    QVERIFY(!can_reuse_prefix_kv(cache, context, 4096, prompt, 0));
    QVERIFY(!can_reuse_prefix_kv(cache, context, 4096, prompt, 99));
    QCOMPARE(stable_prefix_length(std::vector<int32_t>{10, 11}, 99), 0);
}


void TestQwenStream::failedDecodeClearsProfile()
{
    using namespace core_qwen3_stream_model;
    DecodeMetrics timing;
    timing.success = true;
    timing.generated_tokens = 99;
    timing.encoder_ms = 100;
    bool ok = true;
    QVERIFY(generate(nullptr, false, samples(320), "", 8, &ok, nullptr, &timing).empty());
    QVERIFY(!ok);
    QVERIFY(!timing.success);
    QCOMPARE(timing.audio_samples, std::int64_t(320));
    QCOMPARE(timing.generated_tokens, 0);
    QCOMPARE(timing.encoder_ms, 0.0);
    QVERIFY(timing.total_ms >= 0);
    QCOMPARE(timing.total_ms, timing.prepare_ms);
}

void TestQwenStream::persistentPrefixInvalidatesOnKvMutationWhenModelAvailable()
{
    using namespace core_qwen3_stream_model;
    const auto model = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (model.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL to test persistent prefix invalidation");
    const auto wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    QVERIFY(!wav.isEmpty());
    const auto clip = echoflow::CrispSession::readWavF32(wav.toStdString());
    QVERIFY(clip.size() >= size_t(24000));
    const std::vector<float> audio(clip.begin(), clip.begin() + 24000);
    auto params = qwen3_asr_context_default_params();
    params.n_threads = 2;
    params.verbosity = 0;
    params.use_gpu = false;
    std::unique_ptr<qwen3_asr_context, decltype(&qwen3_asr_free)> ctx(
        qwen3_asr_init_from_file(model.constData(), params), &qwen3_asr_free);
    QVERIFY(ctx != nullptr);
    bool ok = false;
    const auto reference = generate(ctx.get(), false, audio, "language Korean<asr_text>", 64, &ok);
    QVERIFY(ok);
    QVERIFY(!reference.empty());
    PrefixKvCache cache;
    DecodeMetrics timing;
    auto decode = [&] {
        return generate(ctx.get(), false, audio, "language Korean<asr_text>", 64, &ok, &cache, &timing);
    };
    QCOMPARE(decode(), reference);
    QVERIFY(ok);
    QVERIFY(!timing.prefix_reused);
    QCOMPARE(decode(), reference);
    QVERIFY(timing.prefix_reused);
    QVERIFY(timing.success);
    QVERIFY(timing.audio_frames > 0);
    QVERIFY(timing.prompt_tokens > timing.audio_frames);
    QVERIFY(timing.generated_tokens > 0);
    const auto parts = timing.mel_ms + timing.encoder_ms + timing.prepare_ms
        + timing.prefill_ms + timing.generate_ms;
    QVERIFY(std::abs(parts - timing.total_ms) < 0.001);
    QVERIFY(timing.llm_compute_ms > 0);
    QVERIFY(timing.llm_compute_ms <= timing.prefill_ms + timing.generate_ms);
    const auto revision = qwen3_asr_kv_revision(ctx.get());
    qwen3_asr_kv_reset(ctx.get());
    QVERIFY(qwen3_asr_kv_revision(ctx.get()) != revision);
    QCOMPARE(decode(), reference);
    QVERIFY(!timing.prefix_reused);
    QCOMPARE(decode(), reference);
    QVERIFY(timing.prefix_reused);
    // A nonstreaming or uncached inference may overwrite every KV position.
    QCOMPARE(generate(ctx.get(), false, audio, "language Korean<asr_text>", 64, &ok), reference);
    QCOMPARE(decode(), reference);
    QVERIFY(!timing.prefix_reused);
    QVERIFY(qwen3_asr_kv_init(ctx.get(), 8192));
    QCOMPARE(decode(), reference);
    QVERIFY(!timing.prefix_reused);
    QCOMPARE(decode(), reference);
    QVERIFY(timing.prefix_reused);
}

void TestQwenStream::reopenedStreamsMatchFreshStreamsWhenModelAvailable()
{
    const auto model = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (model.isEmpty()) QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL to test stream reopen parity");
    const auto wav = QFINDTESTDATA("../third_party/crispasr/samples/ko-369.wav");
    QVERIFY(!wav.isEmpty());
    const auto clip = echoflow::CrispSession::readWavF32(wav.toStdString());
    QVERIFY(clip.size() >= size_t(32000));
    echoflow::CrispSession session(model.toStdString(), "qwen3", 2);
    QVERIFY(session.isLoaded());
    const auto oldSetting = qgetenv("CRISPASR_QWEN3_STREAM_REUSE_PREFIX");
    const auto oldSplit = qgetenv("CRISPASR_QWEN3_STREAM_SPLIT_PREFILL");
    const bool hadSetting = qEnvironmentVariableIsSet("CRISPASR_QWEN3_STREAM_REUSE_PREFIX");
    const bool hadSplit = qEnvironmentVariableIsSet("CRISPASR_QWEN3_STREAM_SPLIT_PREFILL");
    const auto restore = qScopeGuard([&] {
        if (hadSetting) qputenv("CRISPASR_QWEN3_STREAM_REUSE_PREFIX", oldSetting);
        else qunsetenv("CRISPASR_QWEN3_STREAM_REUSE_PREFIX");
        if (hadSplit) qputenv("CRISPASR_QWEN3_STREAM_SPLIT_PREFILL", oldSplit);
        else qunsetenv("CRISPASR_QWEN3_STREAM_SPLIT_PREFILL");
    });
    std::vector<std::string> expected;
    for (bool reuse : {false, true}) {
        qputenv("CRISPASR_QWEN3_STREAM_REUSE_PREFIX", reuse ? "1" : "0");
        qputenv("CRISPASR_QWEN3_STREAM_SPLIT_PREFILL", reuse ? "0" : "1");
        size_t index = 0;
        for (const auto& language : {"ko", "zh"}) {
            session.setLanguage(language);
            for (int count : {8000, 24000, 32000}) {
                QVERIFY(session.openStream(500, 500, 800));
                session.setStreamLiveDecode(false);
                QVERIFY(!session.feedStream(clip.data(), count));
                const auto text = session.flushStream();
                if (!reuse) expected.push_back(text);
                else QCOMPARE(text, expected[index]);
                session.closeStream();
                ++index;
            }
        }
    }

}

void TestQwenStream::prefixKvReuseMatchesFullPrefillWhenModelAvailable()
{
    const QByteArray modelPath = qgetenv("ECHOFLOW_QWEN3_TEST_MODEL");
    if (modelPath.isEmpty())
        QSKIP("set ECHOFLOW_QWEN3_TEST_MODEL to run the model-backed KV parity test");

    auto params = qwen3_asr_context_default_params();
    params.n_threads = 2;
    params.verbosity = 0;
    params.use_gpu = false;
    params.flash_attn = false;
    std::unique_ptr<qwen3_asr_context, decltype(&qwen3_asr_free)> context(
        qwen3_asr_init_from_file(modelPath.constData(), params), &qwen3_asr_free);
    QVERIFY2(context != nullptr, "failed to load ECHOFLOW_QWEN3_TEST_MODEL");

    std::vector<float> firstAudio(16000);
    std::vector<float> secondAudio(24000);
    for (size_t i = 0; i < secondAudio.size(); ++i) {
        const float sample = static_cast<float>(0.12 * std::sin(2.0 * 3.141592653589793
            * 220.0 * static_cast<double>(i) / 16000.0));
        secondAudio[i] = sample;
        if (i < firstAudio.size())
            firstAudio[i] = sample;
    }

    core_qwen3_stream_model::PrefixKvCache cache;
    auto transcribe = [&](const std::vector<float>& audio,
                          core_qwen3_stream_model::PrefixKvCache* prefixCache) {
        bool success = false;
        std::string text = core_qwen3_stream_model::generate(
            context.get(), false, audio, "", 32, &success, prefixCache);
        return std::make_pair(success, std::move(text));
    };

    const auto [fullFirstOk, fullFirst] = transcribe(firstAudio, nullptr);
    QVERIFY2(fullFirstOk, "uncached Qwen generation failed for first audio");
    const auto [cachedFirstOk, cachedFirst] = transcribe(firstAudio, &cache);
    QVERIFY2(cachedFirstOk, "cached Qwen generation failed for first audio");
    QCOMPARE(QString::fromStdString(cachedFirst), QString::fromStdString(fullFirst));

    const auto [cachedSecondOk, cachedSecond] = transcribe(secondAudio, &cache);
    QVERIFY2(cachedSecondOk, "cached Qwen generation failed for extended audio");
    const auto [fullSecondOk, fullSecond] = transcribe(secondAudio, nullptr);
    QVERIFY2(fullSecondOk, "uncached Qwen generation failed for extended audio");
    QCOMPARE(QString::fromStdString(cachedSecond), QString::fromStdString(fullSecond));
}

QTEST_GUILESS_MAIN(TestQwenStream)
#include "test_qwen_stream.moc"
