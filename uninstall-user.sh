#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
STATE_DIR="${STATE_DIR:-$HOME/.local/share/deepin-echoflow}"
CONFIG_ROOT="$HOME/.config"
if [[ "${XDG_CONFIG_HOME:-}" == /* ]]; then
  CONFIG_ROOT="$XDG_CONFIG_HOME"
fi
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
AUTOSTART_DIR="${AUTOSTART_DIR:-$HOME/.config/autostart}"

systemctl --user disable --now echoflow.socket echoflow.service echoflow-ui.service deepin-echoflow.socket deepin-echoflow.service deepin-echoflow-ui.service 2>/dev/null || true
"$ROOT_DIR/scripts/deepin-echoflow-session-stop" "$PREFIX/bin"
rm -f "$SYSTEMD_USER_DIR/deepin-echoflow.service" "$SYSTEMD_USER_DIR/deepin-echoflow.socket" "$SYSTEMD_USER_DIR/deepin-echoflow-ui.service"
rm -f "$AUTOSTART_DIR/deepin-echoflow-autostart.desktop"
rm -f "$SYSTEMD_USER_DIR/echoflow.service" "$SYSTEMD_USER_DIR/echoflow.socket" \
      "$SYSTEMD_USER_DIR/echoflow-ui.service" "$AUTOSTART_DIR/echoflow-autostart.desktop"
systemctl --user daemon-reload 2>/dev/null || true

rm -f "$HOME/.local/share/fcitx5/addon/deepin-echoflow.conf"
rm -f "$PREFIX/bin/deepin-echoflow-service"
rm -f "$PREFIX/bin/deepin-echoflow-ui"
rm -f "$PREFIX/bin/deepin-echoflow-session-start"
rm -f "$PREFIX/bin/deepin-echoflow-session-stop"
rm -f "$PREFIX/lib/fcitx5/libdeepin-echoflow.so"
rm -f "$PREFIX/lib/x86_64-linux-gnu/fcitx5/libdeepin-echoflow.so"
rm -f "$PREFIX/share/applications/deepin-echoflow.desktop"
rm -f "$PREFIX/share/deepin-echoflow/deepin-echoflow-autostart.desktop"
rm -f "$PREFIX/etc/xdg/autostart/deepin-echoflow-autostart.desktop"
rm -f "$PREFIX/share/icons/hicolor/scalable/apps/deepin-echoflow.svg"

# Superseded local entry points can remain after an interrupted upgrade.
rm -f "$HOME/.local/share/fcitx5/addon/echoflow.conf" \
      "$PREFIX/share/fcitx5/addon/echoflow.conf" \
      "$PREFIX/share/fcitx5/addon/deepin-echoflow.conf" \
      "$PREFIX/bin/echoflow-service" "$PREFIX/bin/echoflow-ui" \
      "$PREFIX/bin/echoflow-session-start" "$PREFIX/bin/echoflow-session-stop" \
      "$PREFIX/lib/fcitx5/libechoflow.so" \
      "$PREFIX/lib/x86_64-linux-gnu/fcitx5/libechoflow.so" \
      "$PREFIX/share/applications/echoflow.desktop" \
      "$PREFIX/share/echoflow/echoflow-autostart.desktop" \
      "$PREFIX/etc/xdg/autostart/echoflow-autostart.desktop" \
      "$PREFIX/share/icons/hicolor/scalable/apps/echoflow.svg"

if command -v fcitx5 >/dev/null 2>&1; then
  fcitx5 -rd || echo "Warning: restart Fcitx manually: fcitx5 -rd" >&2
fi

echo "deepin-echoflow user services and binaries removed."
echo "Preserved state/config:"
echo "  $STATE_DIR"
echo "  $CONFIG_ROOT/deepin/deepin-echoflow"
echo "  $CONFIG_ROOT/deepin-echoflow (legacy)"
echo "  $CONFIG_ROOT/echoflow (legacy)"
echo "  $HOME/.local/share/echoflow (legacy)"
