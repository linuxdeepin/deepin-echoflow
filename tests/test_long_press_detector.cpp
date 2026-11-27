// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QSignalSpy>
#include <QtTest>

#include "LongPressDetector.h"

#include <memory>

namespace {

constexpr int kHoldMs = 40;

struct Harness {
    QPoint cursor;
    bool canStart = true;
    bool inActiveWindow = true;
    std::unique_ptr<LongPressDetector> detector;
    std::unique_ptr<QSignalSpy> started;
    std::unique_ptr<QSignalSpy> finished;
    std::unique_ptr<QSignalSpy> cancelled;

    Harness() {
        LongPressDetector::Options options;
        options.holdMs = kHoldMs;
        options.pollMs = 5;
        detector = std::make_unique<LongPressDetector>(
            [this] { return cursor; }, [this] { return canStart; }, options);
        detector->setPointerInActiveWindow([this] { return inActiveWindow; });
        started = std::make_unique<QSignalSpy>(detector.get(), &LongPressDetector::holdStarted);
        finished = std::make_unique<QSignalSpy>(detector.get(), &LongPressDetector::holdFinished);
        cancelled = std::make_unique<QSignalSpy>(detector.get(), &LongPressDetector::holdCancelled);
    }

    void press(QPoint position) {
        cursor = position;
        detector->handlePress(position);
    }

    void waitPastThreshold() { QTest::qWait(kHoldMs * 3); }
};

} // namespace

class LongPressDetectorTest final : public QObject {
    Q_OBJECT

private slots:
    void focusedWindowHoldStartsAndReleaseFinishes();
    void releaseBeforeThresholdIsAClick();
    void movingBeforeThresholdIsASelection();
    void movingWhileHoldingCancels();
    void releaseAwayFromPressCancels();
    void focusAcquiredDuringPressQualifies();
    void pressOutsideActiveWindowIsIgnored();
    void missingOrLostFocusDoesNotStart();
    void busySessionIsNotStarted();
    void disablingCancelsHoldAndIgnoresPresses();
};

void LongPressDetectorTest::focusedWindowHoldStartsAndReleaseFinishes() {
    Harness h;
    h.detector->handleInputFocus(true);
    h.waitPastThreshold();

    h.press(QPoint(400, 208));
    QTRY_COMPARE(h.started->count(), 1);
    QVERIFY(h.detector->holding());

    h.detector->handleRelease(QPoint(401, 208));
    QCOMPARE(h.finished->count(), 1);
    QCOMPARE(h.cancelled->count(), 0);
    QVERIFY(!h.detector->holding());
}

void LongPressDetectorTest::releaseBeforeThresholdIsAClick() {
    Harness h;
    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    h.detector->handleRelease(QPoint(101, 208));
    h.waitPastThreshold();

    QCOMPARE(h.started->count(), 0);
    QCOMPARE(h.finished->count(), 0);
}

void LongPressDetectorTest::movingBeforeThresholdIsASelection() {
    Harness h;
    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    h.cursor = QPoint(140, 208);
    h.waitPastThreshold();
    h.detector->handleRelease(h.cursor);

    QCOMPARE(h.started->count(), 0);
    QCOMPARE(h.cancelled->count(), 0);
}

void LongPressDetectorTest::movingWhileHoldingCancels() {
    Harness h;
    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    QTRY_COMPARE(h.started->count(), 1);

    h.cursor = QPoint(101, 240);
    QTRY_COMPARE(h.cancelled->count(), 1);
    QVERIFY(!h.detector->holding());

    // The eventual release belongs to the cancelled hold.
    h.detector->handleRelease(h.cursor);
    QCOMPARE(h.finished->count(), 0);
    QCOMPARE(h.cancelled->count(), 1);
}

void LongPressDetectorTest::releaseAwayFromPressCancels() {
    Harness h;
    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    QTRY_COMPARE(h.started->count(), 1);

    h.detector->handleRelease(QPoint(160, 208));
    QCOMPARE(h.cancelled->count(), 1);
    QCOMPARE(h.finished->count(), 0);
}

void LongPressDetectorTest::focusAcquiredDuringPressQualifies() {
    Harness h;
    h.press(QPoint(300, 208));
    h.detector->handleInputFocus(true);
    QTRY_COMPARE(h.started->count(), 1);
}

void LongPressDetectorTest::pressOutsideActiveWindowIsIgnored() {
    Harness h;
    // e.g. the dock, which never takes input focus from the text field.
    h.inActiveWindow = false;
    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    h.waitPastThreshold();
    QCOMPARE(h.started->count(), 0);
}

void LongPressDetectorTest::missingOrLostFocusDoesNotStart() {
    Harness h;
    h.press(QPoint(101, 208));
    h.waitPastThreshold();
    QCOMPARE(h.started->count(), 0);

    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    h.detector->handleInputFocus(false);
    h.waitPastThreshold();
    QCOMPARE(h.started->count(), 0);
}

void LongPressDetectorTest::busySessionIsNotStarted() {
    Harness h;
    h.canStart = false;
    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    h.waitPastThreshold();
    h.detector->handleRelease(QPoint(101, 208));

    QCOMPARE(h.started->count(), 0);
    QCOMPARE(h.finished->count(), 0);
}

void LongPressDetectorTest::disablingCancelsHoldAndIgnoresPresses() {
    Harness h;
    h.detector->handleInputFocus(true);
    h.press(QPoint(101, 208));
    QTRY_COMPARE(h.started->count(), 1);

    h.detector->setEnabled(false);
    QCOMPARE(h.cancelled->count(), 1);

    h.press(QPoint(101, 208));
    h.waitPastThreshold();
    QCOMPARE(h.started->count(), 1);
}

QTEST_GUILESS_MAIN(LongPressDetectorTest)

#include "test_long_press_detector.moc"
