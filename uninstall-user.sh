#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
STATE_DIR="${STATE_DIR:-$HOME/.local/share/echoflow}"
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
AUTOSTART_DIR="${AUTOSTART_DIR:-$HOME/.config/autostart}"

systemctl --user disable --now echoflow.socket echoflow.service echoflow-ui.service 2>/dev/null || true
"$ROOT_DIR/scripts/echoflow-session-stop" "$PREFIX/bin"
rm -f "$SYSTEMD_USER_DIR/echoflow.service" "$SYSTEMD_USER_DIR/echoflow.socket" "$SYSTEMD_USER_DIR/echoflow-ui.service"
rm -f "$AUTOSTART_DIR/echoflow-autostart.desktop"
systemctl --user daemon-reload 2>/dev/null || true

rm -f "$HOME/.local/share/fcitx5/addon/echoflow.conf"
rm -f "$PREFIX/bin/echoflow-service"
rm -f "$PREFIX/bin/echoflow-ui"
rm -f "$PREFIX/bin/echoflow-session-start"
rm -f "$PREFIX/bin/echoflow-session-stop"
rm -f "$PREFIX/lib/fcitx5/libechoflow.so"
rm -f "$PREFIX/lib/x86_64-linux-gnu/fcitx5/libechoflow.so"
rm -f "$PREFIX/share/applications/echoflow.desktop"
rm -f "$PREFIX/share/echoflow/echoflow-autostart.desktop"
rm -f "$PREFIX/etc/xdg/autostart/echoflow-autostart.desktop"
rm -f "$PREFIX/share/icons/hicolor/scalable/apps/echoflow.svg"

if command -v fcitx5 >/dev/null 2>&1; then
  fcitx5 -rd || echo "Warning: restart Fcitx manually: fcitx5 -rd" >&2
fi

echo "EchoFlow user services and binaries removed."
echo "Preserved state/config:"
echo "  $STATE_DIR"
echo "  $HOME/.config/echoflow"
