// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string_view>

#include <QElapsedTimer>
#include <QObject>
#include <QPoint>
#include <QRect>
#include <QScreen>
#include <QString>
#include <QTimer>
#include <QVariantList>

class CapsuleController final : public QObject {
    Q_OBJECT
    Q_PROPERTY(double audioLevel READ audioLevel NOTIFY audioLevelChanged)
    Q_PROPERTY(QVariantList audioLevels READ audioLevels NOTIFY audioLevelsChanged)
    Q_PROPERTY(QString recordingDuration READ recordingDuration NOTIFY recordingDurationChanged)

public:
    using SendCommand = std::function<bool(std::string_view)>;

    // The capsule's bottom edge sits this fraction of the available screen
    // height above the screen's bottom edge.
    static constexpr int kCapsuleBottomDivisor = 8;
    static constexpr int kHintHeight = 28;
    static constexpr int kCaretGap = 6;
    static constexpr int kDefaultHintTimeoutMs = 3000;
    static constexpr int kAudioLevelTimeoutMs = 250;
    static constexpr int kHintSettleMs = 30;
    // A focus report this close to a left press is taken as caused by it.
    static constexpr int kClickFocusWindowMs = 400;
    // A caret reported this close to the causing press was moved by that
    // press itself, so the report is fresh and wins over the press point.
    // kCaretNearBelow matches the long-press reach below the caret line.
    static constexpr int kCaretNearX = 150;
    static constexpr int kCaretNearAbove = 150;
    static constexpr int kCaretNearBelow = 400;

    explicit CapsuleController(SendCommand sendCommand, QObject *parent = nullptr);

    bool persistent() const;
    bool idle() const;
    double audioLevel() const { return audioLevel_; }
    QVariantList audioLevels() const { return audioLevels_; }
    QString recordingDuration() const;
    void setHintTimeout(int milliseconds);

public slots:
    void setPersistent(bool persistent);
    void setFocusHintEnabled(bool enabled);
    void setLongPressHint(bool longPressAvailable);
    bool applyServiceMessage(QString message);
    // position is logical; used to anchor click-focused fields.
    void notePointerPress(QPoint position);
    void requestToggle();
    void requestCancel();

signals:
    void audioLevelChanged();
    void audioLevelsChanged();
    void recordingDurationChanged();
    // Delivered before capsuleChanged so Wayland can place the surface on map.
    void capsulePlacementChanged(QScreen *screen, QPoint anchor);
    // (moveX, moveY) is the bottom-center anchor on the focused caret's screen,
    // falling back to the primary screen.
    void capsuleChanged(bool visible, QString phase, QString message,
                        bool hasPosition, int moveX, int moveY);
    void hintChanged(bool visible, QString text, int moveX, int moveY);
    void persistentChanged(bool persistent);
    // caret is in logical coordinates and null when unknown.
    void inputFocusChanged(bool focused, QRect caret);

private:
    enum class Phase {
        Idle,
        Recording,
        Transcribing,
        Cancelling,
    };

    static QString phaseName(Phase phase);
    static QPoint placeBesideCaret(const QRect &caret, int height);
    bool applyFocusMessage(const QString &verb, const QString &geometry);
    void setAudioLevel(double level, const QVariantList& waveform = {});
    void resetAudioLevel();
    void stopRecordingClock();
    bool capsuleVisible() const;
    QScreen *capsuleScreen() const;
    void schedulePlacementUpdate();
    void publish();
    void updateHint();
    void publishHint();
    bool recentPress() const;
    bool caretNearPress(const QRect &caret) const;
    QRect caretAtPress();

    SendCommand sendCommand_;
    Phase phase_ = Phase::Idle;
    QString message_;
    double audioLevel_ = 0;
    QVariantList audioLevels_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    QTimer audioLevelTimer_;
    QElapsedTimer recordingClock_;
    QTimer recordingClockTimer_;
    bool persistent_ = false;

    bool focused_ = false;
    // caret_ is the effective anchor; reportedCaret_ is what Fcitx reported.
    QRect caret_;
    QRect reportedCaret_;
    bool anchoredAtPress_ = false;
    QElapsedTimer clock_;
    QPoint lastPress_;
    qint64 lastPressAt_ = -1;
    qint64 lastFocusAt_ = -1;

    bool focusHintEnabled_ = false;
    bool longPressHint_ = false;
    // Set on focus-in and cleared once the hint has had its turn, so caret
    // moves never bring back a hint that timed out or was typed over.
    bool hintRequested_ = false;
    bool hintVisible_ = false;
    QString hintText_;
    QPoint hintPosition_;
    QTimer placementTimer_;
    QTimer hintTimer_;
    QTimer hintPublishTimer_;
};
