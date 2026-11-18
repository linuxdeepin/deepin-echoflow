#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/deepin-echoflow-install-spec.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

FAKE_BIN="$TEST_ROOT/fake-bin"
TEST_LOG="$TEST_ROOT/commands.log"
mkdir -p "$FAKE_BIN"
export TEST_LOG

cat > "$FAKE_BIN/cmake" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'cmake' >> "$TEST_LOG"
printf ' <%s>' "$@" >> "$TEST_LOG"
printf '\n' >> "$TEST_LOG"

if [[ "${1:-}" == "-S" ]]; then
  build_dir=""
  for ((i = 1; i <= $#; ++i)); do
    if [[ "${!i}" == "-B" ]]; then
      next=$((i + 1))
      build_dir="${!next}"
    fi
  done
  mkdir -p "$build_dir/systemd/user"
  : > "$build_dir/systemd/user/deepin-echoflow.service"
  : > "$build_dir/systemd/user/deepin-echoflow.socket"
  : > "$build_dir/deepin-echoflow-autostart.desktop"
elif [[ "${1:-}" == "--install" ]]; then
  mkdir -p "$PREFIX/bin" "$PREFIX/lib/fcitx5"
  : > "$PREFIX/bin/deepin-echoflow-service"
  : > "$PREFIX/bin/deepin-echoflow-ui"
  cat > "$PREFIX/bin/deepin-echoflow-session-start" <<'SCRIPT'
#!/usr/bin/env bash
printf 'deepin-echoflow-session-start %s\n' "$*" >> "$TEST_LOG"
if [[ "${FAIL_RESTART:-0}" == "1" ]]; then
  exit 23
fi
SCRIPT
  chmod +x "$PREFIX/bin/deepin-echoflow-session-start"
  : > "$PREFIX/lib/fcitx5/libdeepin-echoflow.so"
fi
EOF

cat > "$FAKE_BIN/systemctl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'systemctl %s\n' "$*" >> "$TEST_LOG"
EOF

cat > "$FAKE_BIN/fcitx5" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'fcitx5 %s\n' "$*" >> "$TEST_LOG"
EOF
chmod +x "$FAKE_BIN/cmake" "$FAKE_BIN/systemctl" "$FAKE_BIN/fcitx5"

fail() {
  echo "FAIL - $*" >&2
  exit 1
}

assert_logged() {
  grep -qFx -- "$1" "$TEST_LOG" || fail "missing command: $1"
}

assert_not_logged() {
  if grep -qF -- "$1" "$TEST_LOG"; then
    fail "unexpected command: $1"
  fi
}

assert_ordered() {
  local previous=0 needle line
  for needle in "$@"; do
    line="$(grep -nF -- "$needle" "$TEST_LOG" | head -1 | cut -d: -f1)"
    [[ -n "$line" && "$line" -gt "$previous" ]] || fail "command order at: $needle"
    previous="$line"
  done
}

run_install() {
  local home="$1"
  shift
  HOME="$home" \
  PREFIX="$home/prefix with spaces" \
  STATE_DIR="$home/state" \
  CONFIG_DIR="$home/config" \
  SYSTEMD_USER_DIR="$home/systemd" \
  AUTOSTART_DIR="$home/autostart" \
  DISPLAY= WAYLAND_DISPLAY= \
  BUILD_DIR="$home/build with spaces" \
  PATH="$FAKE_BIN:/usr/bin:/bin" \
    bash "$ROOT/install-user.sh" "$@"
}

DEFAULT_HOME="$TEST_ROOT/default-home"
: > "$TEST_LOG"
default_output="$(run_install "$DEFAULT_HOME")"
assert_logged "systemctl --user daemon-reload"
assert_logged "systemctl --user disable --now deepin-echoflow-ui.service"
assert_logged "systemctl --user disable --now deepin-echoflow.service"
assert_logged "systemctl --user stop deepin-echoflow.socket"
assert_logged "deepin-echoflow-session-start --socket-only"
[[ -f "$DEFAULT_HOME/autostart/deepin-echoflow-autostart.desktop" ]] || fail "user autostart desktop entry was not installed"
assert_logged "fcitx5 -rd"
assert_ordered \
  "systemctl --user disable --now deepin-echoflow.service" \
  "systemctl --user stop deepin-echoflow.socket" \
  "cmake <--install>" \
  "systemctl --user daemon-reload" \
  "deepin-echoflow-session-start --socket-only" \
  "fcitx5 -rd"
grep -qF 'deepin-echoflow installed.' <<< "$default_output" || fail "default install did not report success"

NO_START_HOME="$TEST_ROOT/no-start-home"
: > "$TEST_LOG"
no_start_output="$(run_install "$NO_START_HOME" --no-start)"
assert_logged "systemctl --user daemon-reload"
assert_logged "systemctl --user disable --now deepin-echoflow-ui.service"
assert_logged "systemctl --user disable --now deepin-echoflow.service"
assert_logged "systemctl --user stop deepin-echoflow.socket"
assert_not_logged "deepin-echoflow-session-start"
assert_not_logged "fcitx5 -rd"
grep -qF 'Start UI and activation socket later:' <<< "$no_start_output" || fail "no-start guidance missing"

FAIL_HOME="$TEST_ROOT/failure-home"
: > "$TEST_LOG"
failure_output="$TEST_ROOT/failure.out"
if FAIL_RESTART=1 run_install "$FAIL_HOME" > "$failure_output" 2>&1; then
  fail "launcher failure returned success"
fi
assert_logged "deepin-echoflow-session-start --socket-only"
assert_not_logged "fcitx5 -rd"
if grep -qF 'deepin-echoflow installed.' "$failure_output"; then
  fail "launcher failure printed install success"
fi

# Upgrade uses the existing configuration/model location and retires old entry points.
UPGRADE_HOME="$TEST_ROOT/upgrade-home"
UPGRADE_PREFIX="$UPGRADE_HOME/prefix with spaces"
UPGRADE_CONFIG="$UPGRADE_HOME/xdg-config"
mkdir -p "$UPGRADE_CONFIG/echoflow/qwen3-asr-0.6b" "$UPGRADE_HOME/systemd" \
  "$UPGRADE_HOME/autostart" "$UPGRADE_HOME/.local/share/fcitx5/addon" \
  "$UPGRADE_PREFIX/bin" "$UPGRADE_PREFIX/share/applications"
printf 'existing user config\n' > "$UPGRADE_CONFIG/echoflow/echoflow.conf"
printf 'existing model\n' > "$UPGRADE_CONFIG/echoflow/qwen3-asr-0.6b/model.gguf"
: > "$UPGRADE_HOME/systemd/echoflow.service"
: > "$UPGRADE_HOME/autostart/echoflow-autostart.desktop"
: > "$UPGRADE_HOME/.local/share/fcitx5/addon/echoflow.conf"
: > "$UPGRADE_PREFIX/bin/echoflow-ui"
: > "$UPGRADE_PREFIX/share/applications/echoflow.desktop"
: > "$TEST_LOG"
upgrade_output="$(env -u CONFIG_DIR HOME="$UPGRADE_HOME" XDG_CONFIG_HOME="$UPGRADE_CONFIG" \
  PREFIX="$UPGRADE_PREFIX" STATE_DIR="$UPGRADE_HOME/state" \
  SYSTEMD_USER_DIR="$UPGRADE_HOME/systemd" AUTOSTART_DIR="$UPGRADE_HOME/autostart" \
  DISPLAY= WAYLAND_DISPLAY= BUILD_DIR="$UPGRADE_HOME/build" \
  PATH="$FAKE_BIN:/usr/bin:/bin" bash "$ROOT/install-user.sh" --no-start)"
grep -qFx 'existing user config' "$UPGRADE_CONFIG/echoflow/echoflow.conf" || fail "legacy config changed"
grep -qFx 'existing model' "$UPGRADE_CONFIG/echoflow/qwen3-asr-0.6b/model.gguf" || fail "legacy model changed"
grep -qF "Config: $UPGRADE_CONFIG/echoflow/echoflow.conf" <<< "$upgrade_output" || fail "legacy config not selected"
[[ ! -e "$UPGRADE_CONFIG/deepin-echoflow/deepin-echoflow.conf" ]] || fail "upgrade replaced config with defaults"
for old_entry in "$UPGRADE_HOME/systemd/echoflow.service" \
  "$UPGRADE_HOME/autostart/echoflow-autostart.desktop" \
  "$UPGRADE_HOME/.local/share/fcitx5/addon/echoflow.conf" \
  "$UPGRADE_PREFIX/bin/echoflow-ui" "$UPGRADE_PREFIX/share/applications/echoflow.desktop"; do
  [[ ! -e "$old_entry" ]] || fail "legacy entry remains: $old_entry"
done
[[ -f "$UPGRADE_HOME/.local/share/fcitx5/addon/deepin-echoflow.conf" ]] || fail "renamed addon missing"
assert_logged "systemctl --user disable --now echoflow.socket echoflow.service echoflow-ui.service"

UNINSTALL_HOME="$TEST_ROOT/uninstall-home"
UNINSTALL_PREFIX="$UNINSTALL_HOME/prefix with spaces"
mkdir -p \
  "$UNINSTALL_PREFIX/share/applications" \
  "$UNINSTALL_PREFIX/share/icons/hicolor/scalable/apps" \
  "$UNINSTALL_HOME/state" \
  "$UNINSTALL_HOME/.config/deepin-echoflow" \
  "$UNINSTALL_HOME/.config/autostart"
: > "$UNINSTALL_PREFIX/share/applications/deepin-echoflow.desktop"
: > "$UNINSTALL_PREFIX/share/icons/hicolor/scalable/apps/deepin-echoflow.svg"
: > "$UNINSTALL_HOME/state/preserved"
: > "$UNINSTALL_HOME/.config/deepin-echoflow/preserved"
: > "$UNINSTALL_HOME/.config/autostart/deepin-echoflow-autostart.desktop"
: > "$TEST_LOG"
HOME="$UNINSTALL_HOME" \
PREFIX="$UNINSTALL_PREFIX" \
STATE_DIR="$UNINSTALL_HOME/state" \
SYSTEMD_USER_DIR="$UNINSTALL_HOME/systemd" \
PATH="$FAKE_BIN:/usr/bin:/bin" \
  bash "$ROOT/uninstall-user.sh" > "$TEST_ROOT/uninstall.out"
[[ ! -e "$UNINSTALL_PREFIX/share/applications/deepin-echoflow.desktop" ]] || fail "desktop entry not removed"
[[ ! -e "$UNINSTALL_PREFIX/share/icons/hicolor/scalable/apps/deepin-echoflow.svg" ]] || fail "icon not removed"
[[ -e "$UNINSTALL_HOME/state/preserved" ]] || fail "state was removed"
[[ -e "$UNINSTALL_HOME/.config/deepin-echoflow/preserved" ]] || fail "config was removed"
[[ ! -e "$UNINSTALL_HOME/.config/autostart/deepin-echoflow-autostart.desktop" ]] || fail "autostart entry was not removed"

echo "ok   - executable install and uninstall behavior"
