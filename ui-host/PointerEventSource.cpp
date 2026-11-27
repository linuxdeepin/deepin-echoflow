// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "PointerEventSource.h"

#include "XEventMonitorSource.h"

#include <QGuiApplication>

std::unique_ptr<PointerEventSource> PointerEventSource::createForSession() {
    if (XEventMonitorSource::platformSupported()) {
        return std::make_unique<XEventMonitorSource>();
    }
    qInfo("deepin-echoflow long-press: unsupported platform %s",
          qPrintable(QGuiApplication::platformName()));
    return nullptr;
}
