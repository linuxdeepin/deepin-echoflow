// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "PointerEventSource.h"

#include <QString>

class QDBusServiceWatcher;

// Adapts deepin's XEventMonitor session service, which reports global X11
// button events without grabbing the pointer.
class XEventMonitorSource final : public PointerEventSource {
    Q_OBJECT

public:
    struct Endpoint {
        const char *service;
        const char *path;
        const char *interface;
    };

    explicit XEventMonitorSource(QObject *parent = nullptr);
    ~XEventMonitorSource() override;

    // Only X11 sessions can observe global buttons this way.
    static bool platformSupported();

    bool start() override;
    void stop() override;
    QPoint pointerPosition() const override;
    bool pointerInActiveWindow() const override;

private slots:
    void handleButtonPress(int button, int x, int y, const QString &id);
    void handleButtonRelease(int button, int x, int y, const QString &id);

private:
    bool active() const;
    bool registerWith(const Endpoint &endpoint);
    void disconnectSignals();
    void setRegistration(const Endpoint *endpoint, const QString &id);

    bool wanted_ = false;
    const Endpoint *endpoint_ = nullptr;
    QString registrationId_;
    QDBusServiceWatcher *watcher_ = nullptr;
};
