// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>

#include <QObject>
#include <QPoint>
#include <QTimer>

// A stationary hold in the active window can target the current input focus.
// The recording service checks the live Fcitx context before opening the mic.
class LongPressDetector final : public QObject {
    Q_OBJECT

public:
    struct Options {
        int holdMs = 500;
        // Pointer travel that turns a hold into a click or text selection.
        int moveTolerance = 6;
        int pollMs = 30;
    };

    using CursorProvider = std::function<QPoint()>;
    using CanStart = std::function<bool()>;
    // Whether the pointer is over the active window, which keeps dock and
    // other non-focusable surfaces from triggering.
    using PointerInActiveWindow = std::function<bool()>;

    LongPressDetector(CursorProvider cursor, CanStart canStart,
                      QObject *parent = nullptr);
    LongPressDetector(CursorProvider cursor, CanStart canStart,
                      Options options, QObject *parent = nullptr);

    void setPointerInActiveWindow(PointerInActiveWindow check);
    bool enabled() const;
    bool holding() const;

public slots:
    void setEnabled(bool enabled);
    void handlePress(QPoint position);
    void handleRelease(QPoint position);
    void handleInputFocus(bool focused);

signals:
    void holdStarted();
    void holdFinished();
    void holdCancelled();

private:
    enum class State {
        Idle,
        Pending,
        Holding,
    };

    bool movedFromPress(const QPoint &position) const;
    void pollCursor();
    void holdElapsed();
    void reset();

    CursorProvider cursor_;
    CanStart canStart_;
    Options options_;
    bool enabled_ = true;
    State state_ = State::Idle;
    QPoint pressPosition_;
    PointerInActiveWindow pointerInActiveWindow_;

    bool focused_ = false;

    QTimer holdTimer_;
    QTimer pollTimer_;
};
