// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "StreamingTextPolisher.h"
#include "LocalLlmPolisher.h"

#include <QTest>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace echoflow;
using namespace std::chrono_literals;

namespace {
struct ScopeExit {
    std::function<void()> action;
    ~ScopeExit() { action(); }
};

class YieldingPolisher : public ITextPolisher {
public:
    std::mutex mutex;
    std::condition_variable ready;
    int entered = 0;
    int yielded = 0;

    std::string polish(const std::string& text, const std::function<bool()>& aborted) override
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (++entered == 1) {
            ready.notify_all();
            while (!aborted()) ready.wait_for(lock, 2ms);
            ++yielded;
            ready.notify_all();
            return "stale";
        }
        ready.notify_all();
        return "P(" + text + ")";
    }
    bool wait(bool forYield)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return ready.wait_for(lock, 1s, [&] { return forYield ? yielded > 0 : entered > 0; });
    }
};

class ControlledPolisher : public ITextPolisher {
public:
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::string> inputs;
    std::string blocked;
    bool released = false;
    bool fail = false;
    bool empty = false;
    size_t returned = 0;

    std::string polish(const std::string& text, const std::function<bool()>&) override
    {
        std::unique_lock<std::mutex> lock(mutex);
        inputs.push_back(text);
        condition.notify_all();
        // Deliberately ignores cancellation to test the caller's hard deadline.
        if (text == blocked) condition.wait(lock, [&] { return released; });
        ++returned;
        condition.notify_all();
        if (fail) throw std::runtime_error("unavailable");
        return empty ? "" : "P(" + text + ")";
    }
    bool waitForInput(const std::string& text)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, 1s, [&] {
            return std::find(inputs.begin(), inputs.end(), text) != inputs.end();
        });
    }
    bool waitForReturned(size_t count)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, 1s, [&] { return returned >= count; });
    }
    void release()
    {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_all();
    }
};
} // namespace

class TestStreamingTextPolisher : public QObject {
    Q_OBJECT
private slots:
    void confirmedTextStartsBeforeFinish();
    void recognitionBusyDefersQueuedPolish();
    void cancelledDeferredPolishDoesNotLeakIntoNextSession();
    void interruptedPolishIsRetried();
    void briefRecognitionInvalidatesUncooperativeResult();
    void updatesCarrySessionAndRawPrefix();
    void cancelledCompletionDoesNotPublish();
    void completedPiecesAreReusedAndOnlyTailIsQueued();
    void timeoutKeepsCompletedEditsAndRawTail();
    void cancelDoesNotWaitForUncooperativeModel();
    void lateResultsDoNotCrossSessions();
    void changedConfirmedPrefixFallsBackToRaw();
    void whitespaceIsPreserved();
    void failuresAndEmptyResultsFallBack();
    void queueOverflowRetainsEveryRawPiece();
    void realModelRespectsFinishWaitWhenAvailable();
    void realModelPublishesEditedSegmentWhenAvailable();
};

void TestStreamingTextPolisher::recognitionBusyDefersQueuedPolish()
{
    ControlledPolisher model;
    StreamingTextPolisher worker(model);
    worker.start();
    worker.setRecognitionBusy(true);
    worker.appendConfirmed("first");
    QTest::qWait(30);
    {
        std::lock_guard<std::mutex> lock(model.mutex);
        QVERIFY(model.inputs.empty());
    }
    worker.setRecognitionBusy(false);
    QCOMPARE(worker.finish("first second", 1s), std::string("P(first) P(second)"));
}

void TestStreamingTextPolisher::cancelledDeferredPolishDoesNotLeakIntoNextSession()
{
    ControlledPolisher model;
    StreamingTextPolisher worker(model);
    worker.start();
    worker.setRecognitionBusy(true);
    worker.appendConfirmed("old");
    worker.cancel();
    worker.start();
    worker.appendConfirmed("new");
    worker.setRecognitionBusy(false);
    QCOMPARE(worker.finish("new", 1s), std::string("P(new)"));
    std::lock_guard<std::mutex> lock(model.mutex);
    QCOMPARE(model.inputs, (std::vector<std::string>{"new"}));
}

void TestStreamingTextPolisher::interruptedPolishIsRetried()
{
    YieldingPolisher model;
    StreamingTextPolisher worker(model);
    worker.start();
    worker.appendConfirmed("first");
    QVERIFY(model.wait(false));
    worker.setRecognitionBusy(true);
    QVERIFY(model.wait(true));
    worker.appendConfirmed("first second");
    worker.setRecognitionBusy(false);
    QCOMPARE(worker.finish("first second", 1s), std::string("P(first) P(second)"));
    std::lock_guard<std::mutex> lock(model.mutex);
    QCOMPARE(model.entered, 3);
}

void TestStreamingTextPolisher::briefRecognitionInvalidatesUncooperativeResult()
{
    ControlledPolisher model;
    model.blocked = "first";
    std::atomic<int> updates{0};
    StreamingTextPolisher worker(model, [&](std::uint64_t, const std::string&, const std::string&) {
        ++updates;
    });
    ScopeExit release{[&] { model.release(); }};
    worker.start();
    worker.appendConfirmed("first");
    QVERIFY(model.waitForInput("first"));
    worker.setRecognitionBusy(true);
    worker.setRecognitionBusy(false);
    model.release();
    QCOMPARE(worker.finish("first", 1s), std::string("P(first)"));
    std::lock_guard<std::mutex> lock(model.mutex);
    QCOMPARE(model.inputs.size(), size_t(2));
    QVERIFY(updates.load() <= 1);
}

void TestStreamingTextPolisher::confirmedTextStartsBeforeFinish()
{
    ControlledPolisher model;
    StreamingTextPolisher worker(model);
    worker.start();
    worker.appendConfirmed("first");
    QVERIFY(model.waitForInput("first"));
    QCOMPARE(worker.finish("first"), std::string("P(first)"));
    QCOMPARE(model.inputs.size(), size_t(1));
}

void TestStreamingTextPolisher::updatesCarrySessionAndRawPrefix()
{
    ControlledPolisher model;
    std::mutex mutex;
    std::condition_variable changed;
    struct Update { std::uint64_t id; std::string raw; std::string polished; };
    std::vector<Update> updates;
    StreamingTextPolisher worker(model,
        [&](std::uint64_t id, const std::string& raw, const std::string& polished) {
            std::lock_guard<std::mutex> lock(mutex);
            updates.push_back({id, raw, polished});
            changed.notify_all();
        });
    const auto first = worker.start();
    worker.appendConfirmed("first");
    {
        std::unique_lock<std::mutex> lock(mutex);
        QVERIFY(changed.wait_for(lock, 1s, [&] { return updates.size() == 1; }));
        QCOMPARE(updates.back().id, first);
        QCOMPARE(updates.back().raw, std::string("first"));
        QCOMPARE(updates.back().polished, std::string("P(first)"));
    }
    QCOMPARE(worker.finish("first"), std::string("P(first)"));
    const auto second = worker.start();
    QVERIFY(second != first);
    worker.appendConfirmed("new");
    {
        std::unique_lock<std::mutex> lock(mutex);
        QVERIFY(changed.wait_for(lock, 1s, [&] { return updates.size() == 2; }));
        QCOMPARE(updates.back().id, second);
        QCOMPARE(updates.back().raw, std::string("new"));
        QCOMPARE(updates.back().polished, std::string("P(new)"));
    }
    QCOMPARE(worker.finish("new"), std::string("P(new)"));
}

void TestStreamingTextPolisher::cancelledCompletionDoesNotPublish()
{
    ControlledPolisher model;
    model.blocked = "old";
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::uint64_t> delivered;
    StreamingTextPolisher worker(model,
        [&](std::uint64_t id, const std::string&, const std::string&) {
            std::lock_guard<std::mutex> lock(mutex);
            delivered.push_back(id);
            changed.notify_all();
        });
    ScopeExit release{[&] { model.release(); }};
    const auto oldSession = worker.start();
    worker.appendConfirmed("old");
    QVERIFY(model.waitForInput("old"));
    worker.cancel();
    const auto newSession = worker.start();
    worker.appendConfirmed("new");
    model.release();
    {
        std::unique_lock<std::mutex> lock(mutex);
        QVERIFY(changed.wait_for(lock, 1s, [&] { return !delivered.empty(); }));
        QVERIFY(std::find(delivered.begin(), delivered.end(), oldSession) == delivered.end());
        QCOMPARE(delivered.back(), newSession);
    }
    QCOMPARE(worker.finish("new"), std::string("P(new)"));
}

void TestStreamingTextPolisher::completedPiecesAreReusedAndOnlyTailIsQueued()
{
    ControlledPolisher model;
    StreamingTextPolisher worker(model);
    worker.start();
    worker.appendConfirmed("first");
    QVERIFY(model.waitForReturned(1));
    worker.appendConfirmed("first second");
    QVERIFY(model.waitForReturned(2));
    QCOMPARE(worker.finish("first second tail"), std::string("P(first) P(second) P(tail)"));
    QCOMPARE(model.inputs, (std::vector<std::string>{"first", "second", "tail"}));
}

void TestStreamingTextPolisher::timeoutKeepsCompletedEditsAndRawTail()
{
    ControlledPolisher model;
    model.blocked = "tail";
    StreamingTextPolisher worker(model);
    ScopeExit release{[&] { model.release(); }};
    worker.start();
    worker.appendConfirmed("first");
    QVERIFY(model.waitForReturned(1));
    worker.appendConfirmed("first tail");
    QVERIFY(model.waitForInput("tail"));
    const auto started = std::chrono::steady_clock::now();
    QCOMPARE(worker.finish("first tail", 25ms), std::string("P(first) tail"));
    QVERIFY(std::chrono::steady_clock::now() - started < 500ms);
}

void TestStreamingTextPolisher::cancelDoesNotWaitForUncooperativeModel()
{
    ControlledPolisher model;
    model.blocked = "blocked";
    StreamingTextPolisher worker(model);
    ScopeExit release{[&] { model.release(); }};
    worker.start();
    worker.appendConfirmed("blocked");
    QVERIFY(model.waitForInput("blocked"));
    const auto started = std::chrono::steady_clock::now();
    worker.cancel();
    QCOMPARE(worker.finish("blocked"), std::string("blocked"));
    QVERIFY(std::chrono::steady_clock::now() - started < 500ms);
}

void TestStreamingTextPolisher::lateResultsDoNotCrossSessions()
{
    ControlledPolisher model;
    model.blocked = "old";
    StreamingTextPolisher worker(model);
    ScopeExit release{[&] { model.release(); }};
    worker.start();
    worker.appendConfirmed("old");
    QVERIFY(model.waitForInput("old"));
    QCOMPARE(worker.finish("old", 1ms), std::string("old"));
    worker.start();
    worker.appendConfirmed("new");
    model.release();
    QCOMPARE(worker.finish("new", 1s), std::string("P(new)"));
}

void TestStreamingTextPolisher::changedConfirmedPrefixFallsBackToRaw()
{
    ControlledPolisher model;
    StreamingTextPolisher worker(model);
    worker.start();
    worker.appendConfirmed("first");
    QVERIFY(model.waitForReturned(1));
    QCOMPARE(worker.finish("changed"), std::string("changed"));
}

void TestStreamingTextPolisher::whitespaceIsPreserved()
{
    ControlledPolisher model;
    StreamingTextPolisher worker(model);
    worker.start();
    worker.appendConfirmed("  first ");
    QCOMPARE(worker.finish("  first second\n"), std::string("  P(first) P(second)\n"));
}

void TestStreamingTextPolisher::failuresAndEmptyResultsFallBack()
{
    for (bool fail : {false, true}) {
        ControlledPolisher model;
        model.fail = fail;
        model.empty = !fail;
        StreamingTextPolisher worker(model);
        worker.start();
        QCOMPARE(worker.finish("raw text"), std::string("raw text"));
    }
}

void TestStreamingTextPolisher::queueOverflowRetainsEveryRawPiece()
{
    ControlledPolisher model;
    model.blocked = "blocked";
    StreamingTextPolisher worker(model);
    ScopeExit release{[&] { model.release(); }};
    worker.start();
    std::string raw = "blocked";
    worker.appendConfirmed(raw);
    QVERIFY(model.waitForInput("blocked"));
    for (int i = 0; i < 20; ++i) {
        raw += " word" + std::to_string(i);
        worker.appendConfirmed(raw);
    }
    QCOMPARE(worker.finish(raw, 1ms), raw);
    QCOMPARE(model.inputs.size(), size_t(1));
}

void TestStreamingTextPolisher::realModelRespectsFinishWaitWhenAvailable()
{
    const auto path = qgetenv("ECHOFLOW_POLISH_TEST_MODEL");
    if (path.isEmpty()) QSKIP("set ECHOFLOW_POLISH_TEST_MODEL to test the real polish model");
    LocalLlmPolisher model(path.toStdString(), 2);
    model.preload();
    StreamingTextPolisher worker(model);
    worker.start();
    const std::string first = "嗯那个我们明天上午九点开会，请大家提前准备好相关材料。";
    worker.appendConfirmed(first);
    const auto started = std::chrono::steady_clock::now();
    const std::string result = worker.finish(first + "另外请确认会议室已经预订。", 50ms);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    QVERIFY(elapsed < 500ms);
    QVERIFY(!result.empty());
    qInfo("real polish finish wait_ms=%lld", static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));
}

void TestStreamingTextPolisher::realModelPublishesEditedSegmentWhenAvailable()
{
    const auto path = qgetenv("ECHOFLOW_POLISH_TEST_MODEL");
    if (path.isEmpty()) QSKIP("set ECHOFLOW_POLISH_TEST_MODEL to test polished previews");
    LocalLlmPolisher model(path.toStdString(), 2);
    const std::string raw = "嗯那个我们明天上午九点开会，请大家提前准备好相关材料";
    const std::string expected = model.polish(raw, {});
    if (expected == raw) QSKIP("this model leaves the sample unchanged");
    std::mutex mutex;
    std::condition_variable ready;
    std::uint64_t deliveredSession = 0;
    std::string deliveredRaw;
    std::string deliveredText;
    StreamingTextPolisher worker(model,
        [&](std::uint64_t session, const std::string& source, const std::string& text) {
            std::lock_guard<std::mutex> lock(mutex);
            deliveredSession = session;
            deliveredRaw = source;
            deliveredText = text;
            ready.notify_all();
        });
    const auto session = worker.start();
    worker.appendConfirmed(raw);
    {
        std::unique_lock<std::mutex> lock(mutex);
        QVERIFY(ready.wait_for(lock, 8s, [&] { return deliveredSession != 0; }));
        QCOMPARE(deliveredSession, session);
        QCOMPARE(deliveredRaw, raw);
        QCOMPARE(deliveredText, expected);
    }
    QCOMPARE(worker.finish(raw), expected);
}

QTEST_GUILESS_MAIN(TestStreamingTextPolisher)
#include "test_streaming_text_polisher.moc"
