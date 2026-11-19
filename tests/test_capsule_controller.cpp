// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QGuiApplication>
#include <QScreen>
#include <QQmlComponent>
#include <QQmlContext>
#include <QQuickItem>
#include <QQuickView>
#include <QImage>
#include <QSignalSpy>
#include <QtTest>

#include "CapsuleController.h"

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CapsuleState {
    bool visible = false;
    QString phase;
    QString message;
    bool hasPosition = false;
    int moveX = 0;
    int moveY = 0;
};

struct HintState {
    bool visible = false;
    QString text;
    int moveX = 0;
    int moveY = 0;
};

HintState hintAt(const QSignalSpy &spy, int index) {
    const QList<QVariant> arguments = spy.at(index);
    return {
        arguments.at(0).toBool(),
        arguments.at(1).toString(),
        arguments.at(2).toInt(),
        arguments.at(3).toInt(),
    };
}

// Hint updates are coalesced, so let them land before inspecting the spy.
void settleHint() {
    QTest::qWait(CapsuleController::kHintSettleMs * 3);
}

CapsuleController::SendCommand ignoreCommands() {
    return [](std::string_view) { return true; };
}

CapsuleState stateAt(const QSignalSpy &spy, int index) {
    const QList<QVariant> arguments = spy.at(index);
    return {
        arguments.at(0).toBool(),
        arguments.at(1).toString(),
        arguments.at(2).toString(),
        arguments.at(3).toBool(),
        arguments.at(4).toInt(),
        arguments.at(5).toInt(),
    };
}

} // namespace

class CapsuleControllerTest final : public QObject {
    Q_OBJECT

private slots:
    void persistentIdleIsVisibleAndRequestsStatus();
    void transientIdleIsHidden();
    void closingPersistentCapsuleHidesAndCancels();
    void reopeningBeforeIdleShowsCancelling();
    void persistentIdleMessageRemainsVisible();
    void transientRecordingAndTranscribingAreVisibleUntilIdle();
    void cancellingIsVisibleOnlyInPersistentMode();
    void audioLevelTracksSoundWithoutChangingCapsule();
    void waveformReplacesOneSnapshotAndClearsAtSilence();
    void renderedWaveformFollowsRealLevels();
    void renderedSpectrumAnimatesAndReturnsToSilence();
    void recordingTimerSurvivesStatusAndStopsWithRecording();
    void silentLevelOverridesOldFullHeightBars();
    void invalidAudioLevelsAreIgnored();
    void audioLevelClearsAfterStopAndCancel();
    void missingAudioLevelsReturnToSilence();
    void streamTextDoesNotChangeCapsule();
    void releasingShowsTranscribingUntilIdle();
    void toggleAndCancelSendCommands();
    void persistentFlagOnlySignalsActualChanges();
    void focusShowsHintBelowCaret();
    void hintWaitsForCaretAfterFocus();
    void hintFollowsCaretUntilTimeout();
    void typingOrBlurHidesHintForGood();
    void hintTextReflectsLongPress();
    void hintSuppressedWhenDisabledPersistentOrBusy();
    void focusHintIsDisabledByDefault();
    void recordingCapsuleStaysAtLowerCenter();
    void recordingCapsuleSelectsMonitorBeforeShowing();
    void inputFocusIsReportedForLongPress();
    void refocusTracksCaretWithoutHinting();
    void clickFocusAnchorsAtPress();
    void clickNearReportedCaretTrustsTheReport();
    void pressArrivingAfterFocusReanchors();
    void pressAfterFocusNearCaretKeepsReportedCaret();
    void staleCaretEchoKeepsPressAnchor();
    void focusChurnDoesNotFlickerHint();
};

void CapsuleControllerTest::persistentIdleIsVisibleAndRequestsStatus() {
    std::vector<std::string> commands;
    CapsuleController controller(
        [&](std::string_view command) {
            commands.emplace_back(command);
            return true;
        });
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    controller.setPersistent(true);

    QCOMPARE(commands, std::vector<std::string>{"STATUS"});
    QCOMPARE(capsuleSpy.count(), 1);
    const CapsuleState state = stateAt(capsuleSpy, 0);
    QVERIFY(state.visible);
    QCOMPARE(state.phase, QStringLiteral("idle"));
    QVERIFY(state.message.isEmpty());
    if (QScreen *screen = QGuiApplication::primaryScreen()) {
        const QRect available = screen->availableGeometry();
        QVERIFY(state.hasPosition);
        QCOMPARE(state.moveX, available.left() + available.width() / 2);
        QCOMPARE(state.moveY, available.bottom()
                 - available.height() / CapsuleController::kCapsuleBottomDivisor);
    } else {
        QVERIFY(!state.hasPosition);
    }
}

void CapsuleControllerTest::transientIdleIsHidden() {
    CapsuleController controller([](std::string_view) { return true; });
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    controller.applyServiceMessage(QStringLiteral("IDLE"));

    QCOMPARE(capsuleSpy.count(), 1);
    const CapsuleState state = stateAt(capsuleSpy, 0);
    QVERIFY(!state.visible);
    QCOMPARE(state.phase, QStringLiteral("idle"));
    QVERIFY(state.message.isEmpty());
}

void CapsuleControllerTest::closingPersistentCapsuleHidesAndCancels() {
    std::vector<std::string> commands;
    CapsuleController controller(
        [&](std::string_view command) {
            commands.emplace_back(command);
            return true;
        });
    controller.setPersistent(true);
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    controller.setPersistent(false);

    QCOMPARE(commands, (std::vector<std::string>{"STATUS", "CANCEL"}));
    QCOMPARE(capsuleSpy.count(), 1);
    const CapsuleState state = stateAt(capsuleSpy, 0);
    QVERIFY(!state.visible);
    QCOMPARE(state.phase, QStringLiteral("cancelling"));
    QCOMPARE(state.message, QStringLiteral("Cancelling"));
}

void CapsuleControllerTest::reopeningBeforeIdleShowsCancelling() {
    CapsuleController controller([](std::string_view) { return true; });
    controller.setPersistent(true);
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    controller.setPersistent(false);
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    controller.setPersistent(true);

    QCOMPARE(capsuleSpy.count(), 1);
    const CapsuleState state = stateAt(capsuleSpy, 0);
    QVERIFY(state.visible);
    QCOMPARE(state.phase, QStringLiteral("cancelling"));
    QCOMPARE(state.message, QStringLiteral("Cancelling"));
}

void CapsuleControllerTest::persistentIdleMessageRemainsVisible() {
    CapsuleController controller([](std::string_view) { return true; });
    controller.setPersistent(true);
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    controller.applyServiceMessage(QStringLiteral("IDLE"));

    QCOMPARE(capsuleSpy.count(), 1);
    const CapsuleState state = stateAt(capsuleSpy, 0);
    QVERIFY(state.visible);
    QCOMPARE(state.phase, QStringLiteral("idle"));
    QVERIFY(state.message.isEmpty());
}

void CapsuleControllerTest::transientRecordingAndTranscribingAreVisibleUntilIdle() {
    CapsuleController controller([](std::string_view) { return true; });
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    controller.applyServiceMessage(QStringLiteral("TRANSCRIBING"));
    controller.applyServiceMessage(QStringLiteral("IDLE"));

    QCOMPARE(capsuleSpy.count(), 3);
    CapsuleState state = stateAt(capsuleSpy, 0);
    QVERIFY(state.visible);
    QCOMPARE(state.phase, QStringLiteral("recording"));
    QCOMPARE(state.message, QStringLiteral("Listening"));
    state = stateAt(capsuleSpy, 1);
    QVERIFY(state.visible);
    QCOMPARE(state.phase, QStringLiteral("transcribing"));
    QCOMPARE(state.message, QStringLiteral("Transcribing"));
    state = stateAt(capsuleSpy, 2);
    QVERIFY(!state.visible);
    QCOMPARE(state.phase, QStringLiteral("idle"));
}

void CapsuleControllerTest::cancellingIsVisibleOnlyInPersistentMode() {
    CapsuleController controller([](std::string_view) { return true; });
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    controller.applyServiceMessage(QStringLiteral("CANCELLING"));
    controller.setPersistent(true);
    controller.applyServiceMessage(QStringLiteral("CANCELLING"));

    QCOMPARE(capsuleSpy.count(), 3);
    QVERIFY(!stateAt(capsuleSpy, 0).visible);
    QVERIFY(stateAt(capsuleSpy, 1).visible);
    const CapsuleState persistentState = stateAt(capsuleSpy, 2);
    QVERIFY(persistentState.visible);
    QCOMPARE(persistentState.phase, QStringLiteral("cancelling"));
    QCOMPARE(persistentState.message, QStringLiteral("Cancelling"));
}

void CapsuleControllerTest::audioLevelTracksSoundWithoutChangingCapsule()
{
    CapsuleController controller(ignoreCommands());
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);
    QSignalSpy levelSpy(&controller, &CapsuleController::audioLevelChanged);
    QVERIFY(!controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 800")));
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QVERIFY(controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600")));
    QCOMPARE(controller.audioLevel(), 0.6);
    QCOMPARE(levelSpy.count(), 1);
    QVERIFY(controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600")));
    QCOMPARE(levelSpy.count(), 1);
    QVERIFY(controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 0")));
    QCOMPARE(controller.audioLevel(), 0.0);
    QCOMPARE(levelSpy.count(), 2);
    QVERIFY(controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 0")));
    QCOMPARE(levelSpy.count(), 2); // Silent input does not animate or repaint.
    QCOMPARE(capsuleSpy.count(), 1); // Audio must not resize or move the capsule.
}

void CapsuleControllerTest::waveformReplacesOneSnapshotAndClearsAtSilence()
{
    CapsuleController controller(ignoreCommands());
    QSignalSpy spy(&controller, &CapsuleController::audioLevelsChanged);
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QCOMPARE(controller.audioLevels(), (QVariantList{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    QVERIFY(controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600 100 800 300 1000 500")));
    QCOMPARE(controller.audioLevel(), 0.6);
    QCOMPARE(controller.audioLevels(), (QVariantList{0.1, 0.8, 0.3, 1.0, 0.5}));
    QCOMPARE(spy.count(), 1); // One message supplies every bar in the frame.
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600 100 800 300 1000 500"));
    QCOMPARE(spy.count(), 1); // Identical samples must not create synthetic movement.
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600 900 200 700 100 600"));
    QCOMPARE(controller.audioLevels(), (QVariantList{0.9, 0.2, 0.7, 0.1, 0.6}));
    QCOMPARE(spy.count(), 2); // Whole snapshot replaces the previous waveform.
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600 100 800 300 1000 500 200 700"));
    QCOMPARE(controller.audioLevels(), (QVariantList{0.1, 0.8, 0.3, 1.0, 0.5, 0.2, 0.7}));
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 0 0 0 0 0 0 0 0"));
    QCOMPARE(controller.audioLevels(), (QVariantList{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    QCOMPARE(spy.count(), 4);
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 0 0 0 0 0 0 0 0"));
    QCOMPARE(spy.count(), 4);
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 800 200 400 900 700 300"));
    controller.applyServiceMessage(QStringLiteral("TRANSCRIBING"));
    QCOMPARE(controller.audioLevels(), (QVariantList{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

void CapsuleControllerTest::renderedWaveformFollowsRealLevels()
{
    CapsuleController controller(ignoreCommands());
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QQuickView view;
    view.setResizeMode(QQuickView::SizeViewToRootObject);
    view.rootContext()->setContextProperty(QStringLiteral("meterController"), &controller);
    const auto source = QFINDTESTDATA("../qml/AudioWaveform.qml");
    QVERIFY(!source.isEmpty());
    QQmlComponent component(view.engine());
    component.setData(R"QML(import QtQuick 2.15
AudioWaveform {
    active: true
    animate: false
    levels: meterController.audioLevels
}
)QML", QUrl::fromLocalFile(source).resolved(QUrl(QStringLiteral("WaveformTest.qml"))));
    QVERIFY2(component.isReady(), qPrintable(component.errorString()));
    auto* waveform = qobject_cast<QQuickItem*>(component.create());
    QVERIFY(waveform);
    view.setContent(component.url(), &component, waveform);
    view.show();
    QVERIFY(QTest::qWaitForWindowExposed(&view));
    QList<QQuickItem*> bars;
    for (int i = 0; i < 7; ++i) {
        QQuickItem* bar = nullptr;
        for (auto* child : waveform->childItems()) {
            if (child->objectName() == QStringLiteral("audioBar%1").arg(i)) bar = child;
        }
        QVERIFY(bar);
        QCOMPARE(bar->height(), 3.0);
        bars.append(bar);
    }
    const auto quietImage = view.grabWindow();
    QVERIFY(!quietImage.isNull());
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600 100 800 300 1000 500 200 700"));
    QVERIFY(bars[1]->height() > bars[0]->height());
    QVERIFY(bars[1]->height() > bars[2]->height());
    QCOMPARE(bars[3]->height(), 24.0);
    QVERIFY(bars[4]->height() > 10.0);
    // Pixel mapping preserves relative amplitudes instead of compressing them
    // again with a square root. Changing the shape moves the tallest bar.
    QVERIFY(std::abs((bars[0]->height() - 3.0) / (bars[3]->height() - 3.0) - 0.1) < 0.001);
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 600 1000 200 300 100 500 800 700"));
    QCOMPARE(bars[0]->height(), 24.0);
    QVERIFY(bars[0]->height() > bars[3]->height());
    QTest::qWait(30);
    const auto soundImage = view.grabWindow();
    QVERIFY(!soundImage.isNull());
    QVERIFY(soundImage != quietImage); // Verify actual rendered pixels change.
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 0"));
    for (auto* bar : bars) QCOMPARE(bar->height(), 3.0);
    QTest::qWait(30);
    QCOMPARE(view.grabWindow(), quietImage);
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 900"));
    controller.applyServiceMessage(QStringLiteral("TRANSCRIBING"));
    for (auto* bar : bars) QCOMPARE(bar->height(), 3.0);
}

void CapsuleControllerTest::renderedSpectrumAnimatesAndReturnsToSilence()
{
    QQuickView view;
    view.setSource(QUrl::fromLocalFile(QFINDTESTDATA("../qml/AudioWaveform.qml")));
    QVERIFY(view.status() == QQuickView::Ready);
    auto* waveform = view.rootObject();
    QVERIFY(waveform);
    waveform->setWidth(39);
    waveform->setHeight(24);
    waveform->setProperty("active", true);
    view.show();
    QVERIFY(QTest::qWaitForWindowExposed(&view));
    QQuickItem* first = nullptr;
    for (auto* child : waveform->childItems()) {
        if (child->objectName() == QStringLiteral("audioBar0")) first = child;
    }
    QVERIFY(first);
    QCOMPARE(first->height(), 3.0);
    waveform->setProperty("levels", QVariantList{1.0, 0.2, 0.1, 0.4, 0.2, 0.1, 0.05});
    QTRY_COMPARE_WITH_TIMEOUT(first->height(), 24.0, 500);
    waveform->setProperty("levels", QVariantList{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    QTRY_COMPARE_WITH_TIMEOUT(first->height(), 3.0, 500);
    QTest::qWait(180);
    QCOMPARE(first->height(), 3.0); // No perpetual activity on a silent frame.
    waveform->setProperty("levels", QVariantList{0.9, 0.2, 0.1, 0.4, 0.2, 0.1, 0.05});
    QTRY_VERIFY_WITH_TIMEOUT(first->height() > 20, 500);
    waveform->setProperty("active", false);
    QTRY_COMPARE_WITH_TIMEOUT(first->height(), 3.0, 100); // STOP cancels the release on the next UI update.
}

void CapsuleControllerTest::recordingTimerSurvivesStatusAndStopsWithRecording()
{
    CapsuleController controller(ignoreCommands());
    QSignalSpy spy(&controller, &CapsuleController::recordingDurationChanged);
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QCOMPARE(controller.recordingDuration(), QStringLiteral("00:00"));
    QTRY_COMPARE(controller.recordingDuration(), QStringLiteral("00:01"));
    QTRY_VERIFY(spy.count() >= 2);
    controller.applyServiceMessage(QStringLiteral("RECORDING")); // STATUS must not reset time.
    QCOMPARE(controller.recordingDuration(), QStringLiteral("00:01"));
    controller.applyServiceMessage(QStringLiteral("TRANSCRIBING"));
    QCOMPARE(controller.recordingDuration(), QStringLiteral("00:00"));
    const auto count = spy.count();
    QTest::qWait(1050);
    QCOMPARE(spy.count(), count);
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QCOMPARE(controller.recordingDuration(), QStringLiteral("00:00"));
    controller.applyServiceMessage(QStringLiteral("CANCELLING"));
    QCOMPARE(controller.recordingDuration(), QStringLiteral("00:00"));
}

void CapsuleControllerTest::silentLevelOverridesOldFullHeightBars()
{
    CapsuleController controller(ignoreCommands());
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 900 900 900 900 900 900 900 900"));
    QVERIFY(controller.audioLevel() > 0);
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 0 1000 1000 1000 1000 1000 1000 1000"));
    QCOMPARE(controller.audioLevel(), 0.0);
    QCOMPARE(controller.audioLevels(), (QVariantList{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

void CapsuleControllerTest::invalidAudioLevelsAreIgnored()
{
    CapsuleController controller(ignoreCommands());
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 500"));
    for (const auto& message : {QStringLiteral("AUDIO_LEVEL"), QStringLiteral("AUDIO_LEVEL "),
                               QStringLiteral("AUDIO_LEVEL nan"), QStringLiteral("AUDIO_LEVEL -1"),
                               QStringLiteral("AUDIO_LEVEL 1001"), QStringLiteral("AUDIO_LEVEL 1 2"),
                               QStringLiteral("AUDIO_LEVEL 500 100 200 300 400"),
                               QStringLiteral("AUDIO_LEVEL 500 100 200 300 400 1001"),
                               QStringLiteral("AUDIO_LEVEL 500 100 200 300 400 nan"),
                               QStringLiteral("AUDIO_LEVEL 500 100 200 300 400 500 600")}) {
        QVERIFY(!controller.applyServiceMessage(message));
        QCOMPARE(controller.audioLevel(), 0.5);
    }
}

void CapsuleControllerTest::audioLevelClearsAfterStopAndCancel()
{
    CapsuleController controller(ignoreCommands());
    for (const auto& phase : {QStringLiteral("TRANSCRIBING"), QStringLiteral("CANCELLING"),
                             QStringLiteral("IDLE"), QStringLiteral("HIDE_TOOLTIP")}) {
        controller.applyServiceMessage(QStringLiteral("RECORDING"));
        controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 1000"));
        controller.applyServiceMessage(phase);
        QCOMPARE(controller.audioLevel(), 0.0);
        QVERIFY(!controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 900")));
    }
}

void CapsuleControllerTest::missingAudioLevelsReturnToSilence()
{
    CapsuleController controller(ignoreCommands());
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 700"));
    QTest::qWait(CapsuleController::kAudioLevelTimeoutMs * 3 / 5);
    controller.applyServiceMessage(QStringLiteral("AUDIO_LEVEL 700"));
    QTest::qWait(CapsuleController::kAudioLevelTimeoutMs * 3 / 5);
    QCOMPARE(controller.audioLevel(), 0.7); // Equal levels still refresh the watchdog.
    QTRY_COMPARE(controller.audioLevel(), 0.0);
}

void CapsuleControllerTest::streamTextDoesNotChangeCapsule() {
    CapsuleController controller([](std::string_view) { return true; });
    QSignalSpy spy(&controller, &CapsuleController::capsuleChanged);

    for (const auto& phase : {QStringLiteral("IDLE"), QStringLiteral("RECORDING"),
                             QStringLiteral("TRANSCRIBING"), QStringLiteral("CANCELLING")}) {
        QVERIFY(controller.applyServiceMessage(phase));
        const auto count = spy.count();
        QVERIFY(!controller.applyServiceMessage(QStringLiteral("STREAM_TEXT 实时文字")));
        QVERIFY(!controller.applyServiceMessage(QStringLiteral("STREAM_TEXT ")));
        QVERIFY(!controller.applyServiceMessage(QStringLiteral("STREAM_TEXT")));
        QCOMPARE(spy.count(), count);
    }
}

void CapsuleControllerTest::releasingShowsTranscribingUntilIdle()
{
    CapsuleController controller([](std::string_view) { return true; });
    QSignalSpy spy(&controller, &CapsuleController::capsuleChanged);
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    controller.applyServiceMessage(QStringLiteral("STREAM_TEXT 识别预览只在输入框显示"));
    QCOMPARE(stateAt(spy, 0).message, QStringLiteral("Listening"));
    controller.applyServiceMessage(QStringLiteral("TRANSCRIBING"));
    QCOMPARE(stateAt(spy, 1).phase, QStringLiteral("transcribing"));
    QCOMPARE(stateAt(spy, 1).message, QStringLiteral("Transcribing"));
    QVERIFY(stateAt(spy, 1).visible);
    controller.applyServiceMessage(QStringLiteral("STREAM_TEXT 尾音和润色更新"));
    controller.applyServiceMessage(QStringLiteral("TRANSCRIBING")); // STATUS refresh.
    QCOMPARE(stateAt(spy, 2).message, QStringLiteral("Transcribing"));
    controller.applyServiceMessage(QStringLiteral("IDLE"));
    QVERIFY(!stateAt(spy, 3).visible);
}

void CapsuleControllerTest::toggleAndCancelSendCommands() {
    std::vector<std::string> commands;
    CapsuleController controller(
        [&](std::string_view command) {
            commands.emplace_back(command);
            return true;
        });

    controller.requestToggle();
    controller.requestCancel();

    QCOMPARE(commands, (std::vector<std::string>{"CTRL_DOWN", "CANCEL"}));
    QVERIFY(!controller.persistent());
}

void CapsuleControllerTest::persistentFlagOnlySignalsActualChanges() {
    std::vector<std::string> commands;
    CapsuleController controller(
        [&](std::string_view command) {
            commands.emplace_back(command);
            return true;
        });
    QSignalSpy persistentSpy(&controller, &CapsuleController::persistentChanged);

    QVERIFY(!controller.persistent());
    controller.setPersistent(false);
    controller.setPersistent(true);
    controller.setPersistent(true);
    controller.setPersistent(false);
    controller.setPersistent(false);

    QVERIFY(!controller.persistent());
    QCOMPARE(persistentSpy.count(), 2);
    QCOMPARE(persistentSpy.at(0).at(0).toBool(), true);
    QCOMPARE(persistentSpy.at(1).at(0).toBool(), false);
    QCOMPARE(commands, (std::vector<std::string>{"STATUS", "CANCEL"}));
}

void CapsuleControllerTest::focusShowsHintBelowCaret() {
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);

    QVERIFY(controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18")));

    settleHint();

    QCOMPARE(hintSpy.count(), 1);
    const HintState hint = hintAt(hintSpy, 0);
    QVERIFY(hint.visible);
    QCOMPARE(hint.moveX, 100);
    QCOMPARE(hint.moveY, 200 + 18 + CapsuleController::kCaretGap);
}

void CapsuleControllerTest::hintWaitsForCaretAfterFocus() {
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS"));

    settleHint();
    QVERIFY(hintSpy.isEmpty() || !hintAt(hintSpy, hintSpy.count() - 1).visible);

    controller.applyServiceMessage(QStringLiteral("CARET 40 60 0 20"));

    settleHint();
    const HintState hint = hintAt(hintSpy, hintSpy.count() - 1);
    QVERIFY(hint.visible);
    QCOMPARE(hint.moveX, 40);
    QCOMPARE(hint.moveY, 60 + 20 + CapsuleController::kCaretGap);

    // A caret without a preceding focus-in never shows a hint.
    CapsuleController unfocused(ignoreCommands());
    unfocused.setFocusHintEnabled(true);
    QSignalSpy unfocusedSpy(&unfocused, &CapsuleController::hintChanged);
    unfocused.applyServiceMessage(QStringLiteral("CARET 40 60 0 20"));
    settleHint();
    QVERIFY(unfocusedSpy.isEmpty()
            || !hintAt(unfocusedSpy, unfocusedSpy.count() - 1).visible);
}

void CapsuleControllerTest::hintFollowsCaretUntilTimeout() {
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    controller.setHintTimeout(300);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));

    settleHint();
    controller.applyServiceMessage(QStringLiteral("CARET 150 200 2 18"));
    settleHint();
    QCOMPARE(hintSpy.count(), 2);
    QVERIFY(hintAt(hintSpy, 1).visible);
    QCOMPARE(hintAt(hintSpy, 1).moveX, 150);

    QTRY_VERIFY(!hintAt(hintSpy, hintSpy.count() - 1).visible);

    // After timing out, caret moves in the same field stay quiet.
    const int count = hintSpy.count();
    controller.applyServiceMessage(QStringLiteral("CARET 180 200 2 18"));
    settleHint();
    QCOMPARE(hintSpy.count(), count);

    // A new focus-in gets a new hint.
    controller.applyServiceMessage(QStringLiteral("FOCUS 10 20 2 18"));
    settleHint();
    QVERIFY(hintAt(hintSpy, hintSpy.count() - 1).visible);
}

void CapsuleControllerTest::typingOrBlurHidesHintForGood() {
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));

    settleHint();
    QVERIFY(controller.applyServiceMessage(QStringLiteral("TYPED")));
    settleHint();
    QVERIFY(!hintAt(hintSpy, hintSpy.count() - 1).visible);
    controller.applyServiceMessage(QStringLiteral("CARET 110 200 2 18"));
    settleHint();
    QVERIFY(!hintAt(hintSpy, hintSpy.count() - 1).visible);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));

    settleHint();
    QVERIFY(hintAt(hintSpy, hintSpy.count() - 1).visible);
    QVERIFY(controller.applyServiceMessage(QStringLiteral("BLUR")));
    settleHint();
    QVERIFY(!hintAt(hintSpy, hintSpy.count() - 1).visible);
}

void CapsuleControllerTest::hintTextReflectsLongPress() {
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));

    settleHint();
    QCOMPARE(hintAt(hintSpy, hintSpy.count() - 1).text, QStringLiteral("Hold right Ctrl to talk"));

    controller.setLongPressHint(true);

    settleHint();
    QCOMPARE(hintAt(hintSpy, hintSpy.count() - 1).text,
             QStringLiteral("Hold the left mouse button or right Ctrl to talk"));
}

void CapsuleControllerTest::hintSuppressedWhenDisabledPersistentOrBusy() {
    CapsuleController disabled(ignoreCommands());
    disabled.setFocusHintEnabled(false);
    QSignalSpy disabledSpy(&disabled, &CapsuleController::hintChanged);
    disabled.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    settleHint();
    QVERIFY(disabledSpy.isEmpty() || !hintAt(disabledSpy, disabledSpy.count() - 1).visible);

    CapsuleController persistent(ignoreCommands());
    persistent.setFocusHintEnabled(true);
    persistent.setPersistent(true);
    QSignalSpy persistentSpy(&persistent, &CapsuleController::hintChanged);
    persistent.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    settleHint();
    QVERIFY(persistentSpy.isEmpty()
            || !hintAt(persistentSpy, persistentSpy.count() - 1).visible);

    CapsuleController busy(ignoreCommands());
    busy.setFocusHintEnabled(true);
    QSignalSpy busySpy(&busy, &CapsuleController::hintChanged);
    busy.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    settleHint();
    QVERIFY(hintAt(busySpy, busySpy.count() - 1).visible);
    busy.applyServiceMessage(QStringLiteral("RECORDING"));
    settleHint();
    QVERIFY(!hintAt(busySpy, busySpy.count() - 1).visible);
    // The hint does not come back once the session finishes.
    busy.applyServiceMessage(QStringLiteral("IDLE"));
    settleHint();
    QVERIFY(!hintAt(busySpy, busySpy.count() - 1).visible);
}

void CapsuleControllerTest::focusHintIsDisabledByDefault() {
    CapsuleController controller(ignoreCommands());
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    settleHint();

    QVERIFY(hintSpy.isEmpty()
            || !hintAt(hintSpy, hintSpy.count() - 1).visible);
}

void CapsuleControllerTest::recordingCapsuleStaysAtLowerCenter() {
    QScreen *screen = QGuiApplication::screenAt(QPoint(100, 200));
    if (!screen) screen = QGuiApplication::primaryScreen();
    if (!screen) {
        QSKIP("no screen available");
    }
    const QRect available = screen->availableGeometry();
    const int expectedX = available.left() + available.width() / 2;
    const int expectedY = available.bottom()
        - available.height() / CapsuleController::kCapsuleBottomDivisor;

    CapsuleController controller(ignoreCommands());
    QSignalSpy capsuleSpy(&controller, &CapsuleController::capsuleChanged);

    // The caret selects the screen. Moving within the screen keeps the anchor.
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    controller.applyServiceMessage(QStringLiteral("CARET 300 400 2 18"));
    controller.applyServiceMessage(QStringLiteral("TRANSCRIBING"));
    controller.setPersistent(true);

    QVERIFY(capsuleSpy.count() >= 3);
    for (int i = 0; i < capsuleSpy.count(); ++i) {
        const CapsuleState state = stateAt(capsuleSpy, i);
        QVERIFY(state.hasPosition);
        QCOMPARE(state.moveX, expectedX);
        QCOMPARE(state.moveY, expectedY);
    }
}

void CapsuleControllerTest::recordingCapsuleSelectsMonitorBeforeShowing() {
    if (QGuiApplication::platformName() != QStringLiteral("offscreen")
        || !qEnvironmentVariable("QT_QPA_PLATFORM").contains(QStringLiteral("capsule-test-screens"))) {
        QSKIP("requires the offscreen multi-monitor fixture");
    }

    QScreen *primary = QGuiApplication::primaryScreen();
    QScreen *left = QGuiApplication::screenAt(QPoint(-1100, 10));
    QScreen *above = QGuiApplication::screenAt(QPoint(500, -500));
    QVERIFY(primary && left && above);
    QVERIFY(left != primary && above != primary && left != above);

    CapsuleController controller(ignoreCommands());
    QScreen *target = nullptr;
    QPoint anchor;
    connect(&controller, &CapsuleController::capsulePlacementChanged, this,
            [&](QScreen *screen, QPoint position) {
                target = screen;
                anchor = position;
            });
    connect(&controller, &CapsuleController::capsuleChanged, this,
            [&](bool visible, const QString &, const QString &, bool hasPosition, int x, int y) {
                if (visible) {
                    QVERIFY(target);
                    QVERIFY(hasPosition);
                    QCOMPARE(QPoint(x, y), anchor);
                }
            });

    controller.applyServiceMessage(QStringLiteral("FOCUS -1100 10 2 18"));
    controller.applyServiceMessage(QStringLiteral("RECORDING"));
    QCOMPARE(target, left);
    QCOMPARE(anchor, QPoint(-640, 695));

    // The capsule is already visible when the caret moves to another output.
    controller.applyServiceMessage(QStringLiteral("FOCUS 500 -500 2 18"));
    QTRY_COMPARE(target, above);
    QCOMPARE(anchor, QPoint(1000, -113));

    controller.applyServiceMessage(QStringLiteral("BLUR"));
    QTRY_COMPARE(target, primary);
    QCOMPARE(anchor, QPoint(400, 699));
}

void CapsuleControllerTest::inputFocusIsReportedForLongPress() {
    CapsuleController controller(ignoreCommands());
    QSignalSpy focusSpy(&controller, &CapsuleController::inputFocusChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    controller.applyServiceMessage(QStringLiteral("CARET 120 200 2 18"));
    controller.applyServiceMessage(QStringLiteral("TYPED"));
    controller.applyServiceMessage(QStringLiteral("CARET 1 2 bad 4"));
    controller.applyServiceMessage(QStringLiteral("BLUR"));

    QCOMPARE(focusSpy.count(), 4);
    QVERIFY(focusSpy.at(0).at(0).toBool());
    QCOMPARE(focusSpy.at(0).at(1).toRect(), QRect(100, 200, 2, 18));
    QCOMPARE(focusSpy.at(1).at(1).toRect(), QRect(120, 200, 2, 18));
    QVERIFY(focusSpy.at(2).at(0).toBool());
    QVERIFY(!focusSpy.at(2).at(1).toRect().isValid());
    QVERIFY(!focusSpy.at(3).at(0).toBool());
}

void CapsuleControllerTest::refocusTracksCaretWithoutHinting() {
    // REFOCUS is a repeat focus inside a program that already showed its
    // one hint: focus and caret tracking continue, but no new hint appears.
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);
    QSignalSpy focusSpy(&controller, &CapsuleController::inputFocusChanged);

    controller.applyServiceMessage(QStringLiteral("REFOCUS 100 200 2 18"));
    settleHint();
    QVERIFY(hintSpy.isEmpty() || !hintAt(hintSpy, hintSpy.count() - 1).visible);
    QVERIFY(focusSpy.at(focusSpy.count() - 1).at(0).toBool());
    QCOMPARE(focusSpy.at(focusSpy.count() - 1).at(1).toRect(), QRect(100, 200, 2, 18));

    // The program's first FOCUS still gets its one hint.
    controller.applyServiceMessage(QStringLiteral("FOCUS 120 220 2 18"));
    settleHint();
    QVERIFY(hintAt(hintSpy, hintSpy.count() - 1).visible);

    // A REFOCUS while the hint is up moves it along with the caret.
    controller.applyServiceMessage(QStringLiteral("REFOCUS 150 240 2 18"));
    settleHint();
    const HintState moved = hintAt(hintSpy, hintSpy.count() - 1);
    QVERIFY(moved.visible);
    QCOMPARE(moved.moveX, 150);
    QCOMPARE(moved.moveY, 240 + 18 + CapsuleController::kCaretGap);
}

void CapsuleControllerTest::clickFocusAnchorsAtPress() {
    // Chromium reports the previous field's caret when focus moves between
    // its inputs; the click that focused the new field is the better anchor.
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);
    QSignalSpy focusSpy(&controller, &CapsuleController::inputFocusChanged);

    controller.notePointerPress(QPoint(700, 500));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    settleHint();

    const QRect anchor(700, 491, 0, 18);
    QCOMPARE(focusSpy.at(focusSpy.count() - 1).at(1).toRect(), anchor);
    const HintState hint = hintAt(hintSpy, hintSpy.count() - 1);
    QVERIFY(hint.visible);
    QCOMPARE(hint.moveX, 700);
    QCOMPARE(hint.moveY, anchor.bottom() + 1 + CapsuleController::kCaretGap);

    // Keyboard focus long after the press keeps using the reported caret.
    QTest::qWait(CapsuleController::kClickFocusWindowMs + 50);
    controller.applyServiceMessage(QStringLiteral("BLUR"));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    QCOMPARE(focusSpy.at(focusSpy.count() - 1).at(1).toRect(), QRect(100, 200, 2, 18));
}

void CapsuleControllerTest::clickNearReportedCaretTrustsTheReport() {
    // A press on the caret's line, in its field's padding, or in the empty
    // space below the last line moved that caret itself, so the reported
    // rectangle is fresher than the raw press point and anchors the hint.
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);
    QSignalSpy focusSpy(&controller, &CapsuleController::inputFocusChanged);
    const QRect reported(100, 200, 2, 18);

    controller.notePointerPress(QPoint(104, 206));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    settleHint();
    QCOMPARE(focusSpy.at(focusSpy.count() - 1).at(1).toRect(), reported);
    const HintState onLine = hintAt(hintSpy, hintSpy.count() - 1);
    QVERIFY(onLine.visible);
    QCOMPARE(onLine.moveX, 100);
    QCOMPARE(onLine.moveY, reported.bottom() + 1 + CapsuleController::kCaretGap);

    // Chat-box empty space below the last line still counts as near.
    controller.applyServiceMessage(QStringLiteral("BLUR"));
    controller.notePointerPress(QPoint(105, 500));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    QCOMPARE(focusSpy.at(focusSpy.count() - 1).at(1).toRect(), reported);

    // A press far away (a title-bar activation, another field) still anchors
    // at the press, keeping the Chromium stale-caret workaround intact.
    controller.applyServiceMessage(QStringLiteral("BLUR"));
    controller.notePointerPress(QPoint(700, 500));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    QCOMPARE(focusSpy.at(focusSpy.count() - 1).at(1).toRect(),
             QRect(700, 491, 0, 18));
}

void CapsuleControllerTest::pressArrivingAfterFocusReanchors() {
    CapsuleController controller(ignoreCommands());
    QSignalSpy focusSpy(&controller, &CapsuleController::inputFocusChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    controller.notePointerPress(QPoint(700, 500));

    QCOMPARE(focusSpy.count(), 2);
    QCOMPARE(focusSpy.at(1).at(1).toRect(), QRect(700, 491, 0, 18));
}

void CapsuleControllerTest::pressAfterFocusNearCaretKeepsReportedCaret() {
    // The D-Bus press can trail the Fcitx focus report; when it lands near
    // the reported caret the report stays authoritative.
    CapsuleController controller(ignoreCommands());
    QSignalSpy focusSpy(&controller, &CapsuleController::inputFocusChanged);

    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    controller.notePointerPress(QPoint(104, 206));

    QCOMPARE(focusSpy.count(), 2);
    QCOMPARE(focusSpy.at(1).at(1).toRect(), QRect(100, 200, 2, 18));
}

void CapsuleControllerTest::staleCaretEchoKeepsPressAnchor() {
    CapsuleController controller(ignoreCommands());
    QSignalSpy focusSpy(&controller, &CapsuleController::inputFocusChanged);

    controller.notePointerPress(QPoint(700, 500));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    const int count = focusSpy.count();

    // The stale value echoed again is ignored...
    controller.applyServiceMessage(QStringLiteral("CARET 100 200 2 18"));
    QCOMPARE(focusSpy.count(), count);

    // ...while a genuinely new caret replaces the press anchor.
    controller.applyServiceMessage(QStringLiteral("CARET 720 495 2 18"));
    QCOMPARE(focusSpy.at(focusSpy.count() - 1).at(1).toRect(), QRect(720, 495, 2, 18));
}

void CapsuleControllerTest::focusChurnDoesNotFlickerHint() {
    CapsuleController controller(ignoreCommands());
    controller.setFocusHintEnabled(true);
    QSignalSpy hintSpy(&controller, &CapsuleController::hintChanged);

    controller.notePointerPress(QPoint(700, 500));
    controller.applyServiceMessage(QStringLiteral("BLUR"));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    controller.applyServiceMessage(QStringLiteral("BLUR"));
    controller.applyServiceMessage(QStringLiteral("FOCUS 100 200 2 18"));
    settleHint();

    QCOMPARE(hintSpy.count(), 1);
    QVERIFY(hintAt(hintSpy, 0).visible);
}

QTEST_MAIN(CapsuleControllerTest)

#include "test_capsule_controller.moc"
