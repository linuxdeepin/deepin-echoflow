// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

#include <QObject>
#include <QPoint>

class PointerEventSource : public QObject {
    Q_OBJECT

public:
    // Returns nullptr when the session has no supported backend.
    static std::unique_ptr<PointerEventSource> createForSession();

    virtual bool start() = 0;
    virtual void stop() = 0;

    // Position and button events share Qt logical screen coordinates.
    virtual QPoint pointerPosition() const = 0;

    // Return false if the window under the pointer cannot be identified.
    virtual bool pointerInActiveWindow() const = 0;

signals:
    void leftPressed(QPoint position);
    void leftReleased(QPoint position);
    void activeChanged(bool active);

protected:
    using QObject::QObject;
};
