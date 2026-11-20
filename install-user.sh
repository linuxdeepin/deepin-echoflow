#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
STATE_DIR="${STATE_DIR:-$HOME/.local/share/deepin-echoflow}"
CONFIG_ROOT="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR_EXPLICIT="${CONFIG_DIR+x}"
CONFIG_DIR="${CONFIG_DIR:-$CONFIG_ROOT/deepin-echoflow}"
CONFIG_FILE="$CONFIG_DIR/deepin-echoflow.conf"
# Match ProjectPaths.h: prefer the new config, otherwise retain the old config
# and its model directory in place. Explicit CONFIG_DIR always takes priority.
if [[ -z "$CONFIG_DIR_EXPLICIT" && ! -e "$CONFIG_FILE" \
      && -f "$CONFIG_ROOT/echoflow/echoflow.conf" ]]; then
  CONFIG_DIR="$CONFIG_ROOT/echoflow"
  CONFIG_FILE="$CONFIG_DIR/echoflow.conf"
fi
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
AUTOSTART_DIR="${AUTOSTART_DIR:-$HOME/.config/autostart}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/install-user}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
JOBS="${JOBS:-$(nproc)}"
START_SERVICES=1

usage() {
  cat <<EOF
Usage: $0 [--no-start]

Options:
  --no-start  Install files without starting the service or UI.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start)
      START_SERVICES=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$STATE_DIR" "$CONFIG_DIR" "$SYSTEMD_USER_DIR" "$AUTOSTART_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_INSTALL_SYSCONFDIR="$PREFIX/etc" \
  -DECHOFLOW_BUILD_TESTS=OFF
cmake --build "$BUILD_DIR" -j "$JOBS"

# Remove the legacy boot-enabled service before replacing its unit. Otherwise
# an old default.target.wants link can start the new activation-only service
# without its socket descriptor on the next login.
# Stop the previous project name before replacing its launchers/addon.
systemctl --user disable --now echoflow.socket echoflow.service echoflow-ui.service 2>/dev/null || true
systemctl --user disable --now deepin-echoflow.service 2>/dev/null || true
systemctl --user stop deepin-echoflow.socket 2>/dev/null || true

systemctl --user disable --now deepin-echoflow-ui.service 2>/dev/null || true
"$ROOT_DIR/scripts/deepin-echoflow-session-stop" "$PREFIX/bin"

cmake --install "$BUILD_DIR"

# Remove superseded entry points so Fcitx and the desktop load one version.
rm -f "$SYSTEMD_USER_DIR/echoflow.service" "$SYSTEMD_USER_DIR/echoflow.socket" \
      "$SYSTEMD_USER_DIR/echoflow-ui.service" "$AUTOSTART_DIR/echoflow-autostart.desktop"
rm -f "$HOME/.local/share/fcitx5/addon/echoflow.conf" \
      "$PREFIX/share/fcitx5/addon/echoflow.conf"
rm -f "$PREFIX/bin/echoflow-service" "$PREFIX/bin/echoflow-ui" \
      "$PREFIX/bin/echoflow-session-start" "$PREFIX/bin/echoflow-session-stop" \
      "$PREFIX/lib/fcitx5/libechoflow.so" \
      "$PREFIX/lib/x86_64-linux-gnu/fcitx5/libechoflow.so" \
      "$PREFIX/share/applications/echoflow.desktop" \
      "$PREFIX/share/applications/deepin-echoflow.desktop" \
      "$PREFIX/share/echoflow/echoflow-autostart.desktop" \
      "$PREFIX/etc/xdg/autostart/echoflow-autostart.desktop" \
      "$PREFIX/share/icons/hicolor/scalable/apps/echoflow.svg"
rm -f "$PREFIX/share/deepin-echoflow/deepin-echoflow-autostart.desktop"


if [[ ! -e "$CONFIG_FILE" ]]; then
  cat > "$CONFIG_FILE" <<EOF
[basic.model.model_name]
value=qwen3-asr-0.6b
[basic.model.mirror]
value=hf-mirror
[basic.recognition.prompt]
value=
[basic.recognition.strip_trailing_punctuation]
value=false
[basic.recording.min_record_seconds]
value=0.25
[basic.recording.rate]
value=16000
[basic.recording.channels]
value=1
[basic.recording.format]
value=s16
[advanced.runtime.asr_timeout_seconds]
value=120
[advanced.runtime.openblas_threads]
value=4
[advanced.storage.recordings_dir]
value=\$HOME/.local/share/deepin-echoflow/recordings
EOF
fi

ADDON_LIB="$PREFIX/lib/fcitx5/libdeepin-echoflow.so"
if [[ ! -e "$ADDON_LIB" ]]; then
  ADDON_LIB="$(find "$PREFIX" -path '*/fcitx5/libdeepin-echoflow.so' -print -quit)"
fi
if [[ -z "$ADDON_LIB" || ! -e "$ADDON_LIB" ]]; then
  echo "Could not find installed libdeepin-echoflow.so under $PREFIX" >&2
  exit 1
fi

FCITX_ADDON_DIR="$HOME/.local/share/fcitx5/addon"
mkdir -p "$FCITX_ADDON_DIR"
sed "s|^Library=.*|Library=${ADDON_LIB%.so}|" \
  "$ROOT_DIR/fcitx-addon/deepin-echoflow.conf.in" > "$FCITX_ADDON_DIR/deepin-echoflow.conf"

install -m 0644 "$BUILD_DIR/systemd/user/deepin-echoflow.service" "$SYSTEMD_USER_DIR/deepin-echoflow.service"
install -m 0644 "$BUILD_DIR/systemd/user/deepin-echoflow.socket" "$SYSTEMD_USER_DIR/deepin-echoflow.socket"
install -m 0644 "$BUILD_DIR/deepin-echoflow-autostart.desktop" "$AUTOSTART_DIR/deepin-echoflow-autostart.desktop"
rm -f "$SYSTEMD_USER_DIR/deepin-echoflow-ui.service"

systemctl --user daemon-reload
if [[ "$START_SERVICES" == "1" ]]; then
  if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    "$PREFIX/bin/deepin-echoflow-session-start" >/dev/null 2>&1 &
  else
    "$PREFIX/bin/deepin-echoflow-session-start" --socket-only
  fi
  if command -v fcitx5 >/dev/null 2>&1; then
    fcitx5 -rd || echo "Warning: restart Fcitx manually: fcitx5 -rd" >&2
  fi
fi

echo "deepin-echoflow installed."
echo "Config: $CONFIG_FILE"
echo "Fcitx addon: $FCITX_ADDON_DIR/deepin-echoflow.conf"
if [[ "$START_SERVICES" == "0" ]]; then
  echo "Start UI and activation socket later: $PREFIX/bin/deepin-echoflow-session-start"
  echo "Then reload Fcitx: fcitx5 -rd"
fi
