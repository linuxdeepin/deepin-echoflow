// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "LongPressDetector.h"

#include <utility>

LongPressDetector::LongPressDetector(CursorProvider cursor, CanStart canStart,
                                     QObject *parent)
    : LongPressDetector(std::move(cursor), std::move(canStart), Options{}, parent) {}

LongPressDetector::LongPressDetector(CursorProvider cursor, CanStart canStart,
                                     Options options, QObject *parent)
    : QObject(parent)
    , cursor_(std::move(cursor))
    , canStart_(std::move(canStart))
    , options_(options) {
    holdTimer_.setSingleShot(true);
    holdTimer_.setInterval(options_.holdMs);
    connect(&holdTimer_, &QTimer::timeout, this, &LongPressDetector::holdElapsed);
    pollTimer_.setInterval(options_.pollMs);
    connect(&pollTimer_, &QTimer::timeout, this, &LongPressDetector::pollCursor);
}

void LongPressDetector::setPointerInActiveWindow(PointerInActiveWindow check) {
    pointerInActiveWindow_ = std::move(check);
}

bool LongPressDetector::enabled() const {
    return enabled_;
}

bool LongPressDetector::holding() const {
    return state_ == State::Holding;
}

void LongPressDetector::setEnabled(bool enabled) {
    if (enabled_ == enabled) {
        return;
    }
    enabled_ = enabled;
    if (!enabled_) {
        const bool wasHolding = state_ == State::Holding;
        reset();
        if (wasHolding) {
            emit holdCancelled();
        }
    }
}

void LongPressDetector::handlePress(QPoint position) {
    if (!enabled_) {
        return;
    }
    if (state_ == State::Holding) {
        // A second button press cannot happen while one is held; treat it as
        // a lost release rather than leaving the session recording forever.
        reset();
        emit holdFinished();
        return;
    }
    state_ = State::Pending;
    pressPosition_ = position;
    holdTimer_.start();
    pollTimer_.start();
}

void LongPressDetector::handleRelease(QPoint position) {
    const State state = state_;
    if (state == State::Idle) {
        return;
    }
    const bool moved = movedFromPress(position);
    reset();
    if (state == State::Holding) {
        if (moved) {
            emit holdCancelled();
        } else {
            emit holdFinished();
        }
    }
}

void LongPressDetector::handleInputFocus(bool focused) {
    focused_ = focused;
}

bool LongPressDetector::movedFromPress(const QPoint &position) const {
    return (position - pressPosition_).manhattanLength() > options_.moveTolerance;
}

void LongPressDetector::pollCursor() {
    if (!cursor_ || !movedFromPress(cursor_())) {
        return;
    }
    const bool wasHolding = state_ == State::Holding;
    reset();
    if (wasHolding) {
        emit holdCancelled();
    }
}

void LongPressDetector::holdElapsed() {
    if (state_ != State::Pending) {
        return;
    }
    const bool moved = cursor_ && movedFromPress(cursor_());
    const bool inActiveWindow = !pointerInActiveWindow_ || pointerInActiveWindow_();
    const bool canStart = !canStart_ || canStart_();
    if (moved || !focused_ || !inActiveWindow || !canStart) {
        reset();
        return;
    }
    state_ = State::Holding;
    emit holdStarted();
}

void LongPressDetector::reset() {
    state_ = State::Idle;
    holdTimer_.stop();
    pollTimer_.stop();
}
