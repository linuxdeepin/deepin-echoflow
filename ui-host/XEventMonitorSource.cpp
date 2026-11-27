// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "XEventMonitorSource.h"

#include "ScreenGeometry.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include <QCursor>
#include <QDBusConnection>
#include <QDBusConnectionInterface>
#include <QDBusMessage>
#include <QDBusReply>
#include <QDBusServiceWatcher>
#include <QGuiApplication>
#include <QtGlobal>

#include <xcb/xcb.h>

namespace {

constexpr int kLeftButton = 1;

// Current deepin first, then the name used before the dde rename.
constexpr XEventMonitorSource::Endpoint kEndpoints[] = {
    {"org.deepin.dde.XEventMonitor1", "/org/deepin/dde/XEventMonitor1",
     "org.deepin.dde.XEventMonitor1"},
    {"com.deepin.api.XEventMonitor", "/com/deepin/api/XEventMonitor",
     "com.deepin.api.XEventMonitor"},
};

struct FreeDeleter {
    void operator()(void *pointer) const { std::free(pointer); }
};

template <typename T>
using Reply = std::unique_ptr<T, FreeDeleter>;

xcb_window_t activeWindow(xcb_connection_t *connection, xcb_window_t root) {
    constexpr const char kName[] = "_NET_ACTIVE_WINDOW";
    Reply<xcb_intern_atom_reply_t> atom(xcb_intern_atom_reply(
        connection, xcb_intern_atom(connection, 1, std::strlen(kName), kName), nullptr));
    if (!atom || atom->atom == XCB_ATOM_NONE) {
        return XCB_WINDOW_NONE;
    }
    Reply<xcb_get_property_reply_t> property(xcb_get_property_reply(
        connection,
        xcb_get_property(connection, 0, root, atom->atom, XCB_ATOM_WINDOW, 0, 1),
        nullptr));
    if (!property || xcb_get_property_value_length(property.get()) < 4) {
        return XCB_WINDOW_NONE;
    }
    return *static_cast<xcb_window_t *>(xcb_get_property_value(property.get()));
}

// The active window is a client window; with a reparenting window manager
// the root's direct child is its frame.
xcb_window_t topLevelAncestor(xcb_connection_t *connection, xcb_window_t root,
                              xcb_window_t window) {
    for (int depth = 0; window != XCB_WINDOW_NONE && depth < 16; ++depth) {
        Reply<xcb_query_tree_reply_t> tree(
            xcb_query_tree_reply(connection, xcb_query_tree(connection, window), nullptr));
        if (!tree) {
            return XCB_WINDOW_NONE;
        }
        if (tree->parent == root) {
            return window;
        }
        window = tree->parent;
    }
    return XCB_WINDOW_NONE;
}

} // namespace

XEventMonitorSource::XEventMonitorSource(QObject *parent)
    : PointerEventSource(parent) {}

XEventMonitorSource::~XEventMonitorSource() {
    stop();
}

bool XEventMonitorSource::platformSupported() {
    // deepin runs DTK apps on its "dxcb" wrapper around the xcb platform.
    const QString platform = QGuiApplication::platformName();
    return (platform == QStringLiteral("xcb") || platform == QStringLiteral("dxcb"))
        && qEnvironmentVariable("XDG_SESSION_TYPE") != QStringLiteral("wayland");
}

QPoint XEventMonitorSource::pointerPosition() const {
    return QCursor::pos();
}

bool XEventMonitorSource::pointerInActiveWindow() const {
    auto *x11 = qGuiApp ? qGuiApp->nativeInterface<QNativeInterface::QX11Application>()
                        : nullptr;
    xcb_connection_t *connection = x11 ? x11->connection() : nullptr;
    if (!connection) {
        return false;
    }
    const xcb_setup_t *setup = xcb_get_setup(connection);
    const xcb_window_t root = xcb_setup_roots_iterator(setup).data->root;

    const xcb_window_t active = activeWindow(connection, root);
    if (active == XCB_WINDOW_NONE) {
        return false;
    }
    Reply<xcb_query_pointer_reply_t> pointer(
        xcb_query_pointer_reply(connection, xcb_query_pointer(connection, root), nullptr));
    if (!pointer) {
        return false;
    }
    return pointer->child != XCB_WINDOW_NONE
        && pointer->child == topLevelAncestor(connection, root, active);
}

bool XEventMonitorSource::active() const {
    return endpoint_ != nullptr;
}

bool XEventMonitorSource::start() {
    wanted_ = true;
    if (active()) {
        return true;
    }

    QDBusConnection bus = QDBusConnection::sessionBus();
    if (!watcher_) {
        watcher_ = new QDBusServiceWatcher(this);
        watcher_->setConnection(bus);
        for (const Endpoint &endpoint : kEndpoints) {
            watcher_->addWatchedService(QString::fromLatin1(endpoint.service));
        }
        // The monitor forgets registrations when it restarts.
        connect(watcher_, &QDBusServiceWatcher::serviceRegistered, this, [this] {
            if (wanted_ && !active()) {
                start();
            }
        });
        connect(watcher_, &QDBusServiceWatcher::serviceUnregistered, this,
                [this](const QString &service) {
                    if (endpoint_ && service == QString::fromLatin1(endpoint_->service)) {
                        disconnectSignals();
                        setRegistration(nullptr, {});
                    }
                });
    }

    for (const Endpoint &endpoint : kEndpoints) {
        if (registerWith(endpoint)) {
            return true;
        }
    }
    qWarning("deepin-echoflow long-press: XEventMonitor service is unavailable");
    return false;
}

bool XEventMonitorSource::registerWith(const Endpoint &endpoint) {
    QDBusConnection bus = QDBusConnection::sessionBus();
    const QString service = QString::fromLatin1(endpoint.service);
    if (!bus.interface() || !bus.interface()->isServiceRegistered(service)) {
        return false;
    }

    const QString path = QString::fromLatin1(endpoint.path);
    const QString interface = QString::fromLatin1(endpoint.interface);
    QDBusMessage call = QDBusMessage::createMethodCall(
        service, path, interface, QStringLiteral("RegisterFullScreen"));
    const QDBusReply<QString> reply = bus.call(call, QDBus::Block, 2000);
    if (!reply.isValid() || reply.value().isEmpty()) {
        qWarning("deepin-echoflow long-press: RegisterFullScreen failed on %s: %s",
                 endpoint.service, qPrintable(reply.error().message()));
        return false;
    }

    const bool pressConnected = bus.connect(
        service, path, interface, QStringLiteral("ButtonPress"), this,
        SLOT(handleButtonPress(int,int,int,QString)));
    const bool releaseConnected = bus.connect(
        service, path, interface, QStringLiteral("ButtonRelease"), this,
        SLOT(handleButtonRelease(int,int,int,QString)));
    if (!pressConnected || !releaseConnected) {
        qWarning("deepin-echoflow long-press: cannot subscribe to %s button signals",
                 endpoint.service);
        return false;
    }

    setRegistration(&endpoint, reply.value());
    qInfo("deepin-echoflow long-press: listening for buttons via %s", endpoint.service);
    return true;
}

void XEventMonitorSource::stop() {
    wanted_ = false;
    if (!endpoint_) {
        return;
    }
    QDBusConnection bus = QDBusConnection::sessionBus();
    QDBusMessage call = QDBusMessage::createMethodCall(
        QString::fromLatin1(endpoint_->service), QString::fromLatin1(endpoint_->path),
        QString::fromLatin1(endpoint_->interface), QStringLiteral("UnregisterArea"));
    call << registrationId_;
    bus.call(call, QDBus::NoBlock);
    disconnectSignals();
    setRegistration(nullptr, {});
}

void XEventMonitorSource::disconnectSignals() {
    if (!endpoint_) {
        return;
    }
    QDBusConnection bus = QDBusConnection::sessionBus();
    const QString service = QString::fromLatin1(endpoint_->service);
    const QString path = QString::fromLatin1(endpoint_->path);
    const QString interface = QString::fromLatin1(endpoint_->interface);
    bus.disconnect(service, path, interface, QStringLiteral("ButtonPress"), this,
                   SLOT(handleButtonPress(int,int,int,QString)));
    bus.disconnect(service, path, interface, QStringLiteral("ButtonRelease"), this,
                   SLOT(handleButtonRelease(int,int,int,QString)));
}

void XEventMonitorSource::setRegistration(const Endpoint *endpoint, const QString &id) {
    const bool wasActive = active();
    endpoint_ = endpoint;
    registrationId_ = id;
    if (wasActive != active()) {
        emit activeChanged(active());
    }
}

void XEventMonitorSource::handleButtonPress(int button, int x, int y, const QString &id) {
    if (button == kLeftButton && id == registrationId_) {
        emit leftPressed(echoflow::nativeToLogical(QPoint(x, y)));
    }
}

void XEventMonitorSource::handleButtonRelease(int button, int x, int y, const QString &id) {
    if (button == kLeftButton && id == registrationId_) {
        emit leftReleased(echoflow::nativeToLogical(QPoint(x, y)));
    }
}
