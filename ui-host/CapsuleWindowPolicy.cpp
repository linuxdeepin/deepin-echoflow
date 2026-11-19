// SPDX-FileCopyrightText: 2026 Hualet Wang
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CapsuleWindowPolicy.h"

#include "CapsuleController.h"

#include <QGuiApplication>
#include <QPointer>
#include <QPlatformSurfaceEvent>
#include <QScreen>
#include <QWindow>
#include <qscreen_platform.h>

#ifdef ECHOFLOW_HAVE_TREELAND_DDE_SHELL
#include <QWaylandClientExtension>
#include <qpa/qplatformwindow_p.h>

#include <memory>

#include "qwayland-treeland-dde-shell-unstable-v2.h"

#ifdef ECHOFLOW_HAVE_LAYER_SHELL
#include <QtWaylandClient/private/qwaylandshellintegration_p.h>
#include <QtWaylandClient/private/qwaylandshellsurface_p.h>
#include <QtWaylandClient/private/qwaylandwindow_p.h>

#include "qwayland-wlr-layer-shell-unstable-v1.h"
#endif

namespace {

class TreelandDdeShellManager final
    : public QWaylandClientExtensionTemplate<TreelandDdeShellManager>
    , public QtWayland::treeland_dde_shell_manager_v2 {
public:
    TreelandDdeShellManager()
        : QWaylandClientExtensionTemplate<TreelandDdeShellManager>(2) {
        initialize();
    }
};

class TreelandDdeShellSurface final : public QtWayland::treeland_dde_shell_surface_v2 {
public:
    explicit TreelandDdeShellSurface(struct ::treeland_dde_shell_surface_v2 *surface)
        : QtWayland::treeland_dde_shell_surface_v2(surface) {}

    ~TreelandDdeShellSurface() { destroy(); }
};

#ifdef ECHOFLOW_HAVE_LAYER_SHELL
class CapsuleLayerSurface final : public QtWaylandClient::QWaylandShellSurface,
                                  public QtWayland::zwlr_layer_surface_v1 {
public:
    CapsuleLayerSurface(QtWaylandClient::QWaylandWindow *window,
                        struct ::zwlr_layer_surface_v1 *surface, int bottomMargin)
        : QWaylandShellSurface(window), zwlr_layer_surface_v1(surface) {
        set_anchor(anchor_bottom);
        // The margin already includes any insets from QScreen's work area.
        set_exclusive_zone(-1);
        set_margin(0, 0, bottomMargin, 0);
        set_keyboard_interactivity(keyboard_interactivity_none);
        size_ = window->window()->size().expandedTo(QSize(1, 1));
        set_size(size_.width(), size_.height());
    }

    ~CapsuleLayerSurface() override { destroy(); }

    bool isExposed() const override { return configured_; }

    void setWindowGeometry(const QRect &geometry) override {
        const QSize size = geometry.size().expandedTo(QSize(1, 1));
        if (size == size_) {
            return;
        }
        size_ = size;
        set_size(size_.width(), size_.height());
        window()->commit();
    }

private:
    void zwlr_layer_surface_v1_configure(uint32_t serial, uint32_t, uint32_t) override {
        ack_configure(serial);
        if (!configured_) {
            configured_ = true;
            applyConfigureWhenPossible();
        }
    }

    void zwlr_layer_surface_v1_closed() override {
        // Hiding destroys this shell surface, so defer it until dispatch ends.
        QMetaObject::invokeMethod(this, [this] { window()->window()->hide(); },
                                  Qt::QueuedConnection);
    }

    QSize size_;
    bool configured_ = false;
};

class CapsuleLayerShell final
    : public QtWaylandClient::QWaylandShellIntegrationTemplate<CapsuleLayerShell>
    , public QtWayland::zwlr_layer_shell_v1 {
public:
    CapsuleLayerShell() : QWaylandShellIntegrationTemplate(3) {
        initialize(nullptr);
    }

    ~CapsuleLayerShell() override {
        if (isInitialized()) {
            if (QtWayland::zwlr_layer_shell_v1::version() >= 3) {
                destroy();
            } else {
                wl_proxy_destroy(reinterpret_cast<wl_proxy *>(object()));
            }
        }
    }

    void setPlacement(QScreen *screen, int bottomMargin) {
        screen_ = screen;
        bottomMargin_ = bottomMargin;
    }

    QtWaylandClient::QWaylandShellSurface *createShellSurface(
        QtWaylandClient::QWaylandWindow *window) override {
        const auto *screen = screen_ ? screen_.data() : QGuiApplication::primaryScreen();
        auto *waylandScreen = screen ? screen->nativeInterface<QNativeInterface::QWaylandScreen>()
                                     : nullptr;
        auto *output = waylandScreen ? waylandScreen->output() : nullptr;
        // This assigns the surface to the output before its first commit.
        auto *surface = get_layer_surface(wlSurfaceForWindow(window), output,
                                          layer_overlay, QStringLiteral("deepin-echoflow"));
        return new CapsuleLayerSurface(window, surface, bottomMargin_);
    }

private:
    QPointer<QScreen> screen_;
    int bottomMargin_ = 0;
};
#endif

class TreelandCapsulePolicy final : public QObject {
public:
    TreelandCapsulePolicy(QWindow *window, CapsuleController &controller)
        : QObject(window), window_(window) {
        window_->installEventFilter(this);
        connect(&controller, &CapsuleController::capsulePlacementChanged,
                this, &TreelandCapsulePolicy::setPlacement);
        connectPlatformSurface();
#ifdef ECHOFLOW_HAVE_LAYER_SHELL
        if (layerShell_.isActive()) {
            // Install the per-window integration before Qt assigns a shell role.
            window_->create();
            return;
        }
#endif
        connect(&manager_, &QWaylandClientExtension::activeChanged,
                this, [this] { attachSurface(); });
    }

    bool eventFilter(QObject *watched, QEvent *event) override {
        if (watched == window_ && event->type() == QEvent::PlatformSurface) {
            const auto *surfaceEvent = static_cast<QPlatformSurfaceEvent *>(event);
            if (surfaceEvent->surfaceEventType()
                == QPlatformSurfaceEvent::SurfaceCreated) {
                connectPlatformSurface();
            } else if (surfaceEvent->surfaceEventType()
                       == QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed) {
                shellSurface_.reset();
            }
        }
        return false;
    }

private:
    void setPlacement(QScreen *screen, QPoint anchor) {
        const bool moved = targetScreen_ != screen || targetAnchor_ != anchor;
        targetScreen_ = screen;
        targetAnchor_ = anchor;

#ifdef ECHOFLOW_HAVE_LAYER_SHELL
        if (layerShellInstalled_) {
            const int bottomMargin = screen ? qMax(0, screen->geometry().bottom() - anchor.y()) : 0;
            layerShell_.setPlacement(screen, bottomMargin);
            if (moved && window_->isVisible()) {
                // A layer surface's output is fixed for the lifetime of its role.
                window_->hide();
            }
            return;
        }
#endif

        if (moved && positionHintSent_ && window_->isVisible()) {
            // Treeland reads the hint at initial placement. capsuleChanged
            // will show the window again after updating the QML content.
            window_->hide();
        }
    }

    void connectPlatformSurface() {
#ifdef ECHOFLOW_HAVE_LAYER_SHELL
        if (layerShell_.isActive() && window_->handle()) {
            auto *waylandWindow = static_cast<QtWaylandClient::QWaylandWindow *>(window_->handle());
            waylandWindow->setShellIntegration(&layerShell_);
            layerShellInstalled_ = true;
            return;
        }
#endif
        auto *waylandWindow =
            window_->nativeInterface<QNativeInterface::Private::QWaylandWindow>();
        if (!waylandWindow) {
            return;
        }
        if (connectedPlatformWindow_ == waylandWindow) {
            if (waylandWindow->surface()) {
                attachSurface();
            }
            return;
        }
        connectedPlatformWindow_ = waylandWindow;

        connect(waylandWindow,
                &QNativeInterface::Private::QWaylandWindow::surfaceCreated,
                this, [this] { attachSurface(); });
        connect(waylandWindow,
                &QNativeInterface::Private::QWaylandWindow::surfaceDestroyed,
                this, [this] { shellSurface_.reset(); });
        if (waylandWindow->surface()) {
            attachSurface();
        }
    }

    void attachSurface() {
        if (!manager_.isActive() || shellSurface_) {
            return;
        }

        auto *waylandWindow =
            window_->nativeInterface<QNativeInterface::Private::QWaylandWindow>();
        if (!waylandWindow || !waylandWindow->surface()) {
            return;
        }

        auto *surface = manager_.get_shell_surface(waylandWindow->surface());
        if (!surface) {
            return;
        }

        shellSurface_ = std::make_unique<TreelandDdeShellSurface>(surface);
        positionHintSent_ = false;
        const uint32_t role = shellSurface_->version() >= 2
            ? QtWayland::treeland_dde_shell_surface_v2::role_always_on_top
            : QtWayland::treeland_dde_shell_surface_v2::role_overlay;
        shellSurface_->set_role(role);
        shellSurface_->set_accept_keyboard_focus(0);
        updatePositionHint();
    }

    void updatePositionHint() {
        if (!shellSurface_ || positionHintSent_ || !targetScreen_) {
            return;
        }
        auto *waylandScreen = targetScreen_->nativeInterface<QNativeInterface::QWaylandScreen>();
        auto *output = waylandScreen ? waylandScreen->output() : nullptr;
        // A null output uses the primary output's origin in the protocol.
        const QScreen *coordinateScreen = output ? targetScreen_.data()
                                                 : QGuiApplication::primaryScreen();
        const QPoint origin = coordinateScreen ? coordinateScreen->geometry().topLeft() : QPoint();
        const QPoint topLeft = targetAnchor_ - origin
            - QPoint((window_->width() + 1) / 2, window_->height());
        shellSurface_->set_position_hint(output, topLeft.x(), topLeft.y());
        positionHintSent_ = true;
    }

    QWindow *window_;
    QPointer<QObject> connectedPlatformWindow_;
    QPointer<QScreen> targetScreen_;
    QPoint targetAnchor_;
#ifdef ECHOFLOW_HAVE_LAYER_SHELL
    CapsuleLayerShell layerShell_;
    bool layerShellInstalled_ = false;
#endif
    TreelandDdeShellManager manager_;
    std::unique_ptr<TreelandDdeShellSurface> shellSurface_;
    bool positionHintSent_ = false;
};

} // namespace
#endif

void configureCapsuleWindow(QWindow *window, CapsuleController &controller) {
    if (!window || !QGuiApplication::platformName().startsWith(QStringLiteral("wayland"))) {
        return;
    }

    // The DDE shell fallback needs a regular toplevel, not an xdg_popup.
    window->setFlags(Qt::Window | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint
                     | Qt::WindowDoesNotAcceptFocus);

#ifdef ECHOFLOW_HAVE_TREELAND_DDE_SHELL
    new TreelandCapsulePolicy(window, controller);
#else
    Q_UNUSED(controller);
#endif
}
