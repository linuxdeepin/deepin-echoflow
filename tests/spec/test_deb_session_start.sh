#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
tmp=$(mktemp -d)
desktop_pid=
cleanup() {
    if [[ -n "$desktop_pid" ]]; then
        kill "$desktop_pid" 2>/dev/null || true
        wait "$desktop_pid" 2>/dev/null || true
    fi
    rm -rf -- "$tmp"
}
trap cleanup EXIT
export PATH="$tmp:$PATH" TEST_LOG="$tmp/log"
unset DPKG_ROOT

cat > "$tmp/systemd-detect-virt" <<'SCRIPT'
#!/bin/sh
[ "${TEST_CHROOT:-0}" = 1 ]
SCRIPT
cat > "$tmp/dpkg-query" <<'SCRIPT'
#!/bin/sh
# Never migrate the host's conffile while testing a live-root invocation.
exit 1
SCRIPT
cat > "$tmp/loginctl" <<'SCRIPT'
#!/bin/sh
if [ "$1" = list-sessions ]; then
    if [ "${TEST_HEADLESS:-0}" = 0 ]; then
        printf '%s\n' gone x11 duplicate greeter tty remote inactive root wayland
    fi
    exit 0
fi
[ "$1" = show-session ] || exit 1
uid=1001 class=user type=x11 active=yes remote=no
case "$2" in
    gone) exit 1 ;;
    x11|duplicate) ;;
    greeter) uid=1002 class=greeter ;;
    tty) uid=1003 type=tty ;;
    remote) uid=1004 remote=yes ;;
    inactive) uid=1005 active=no ;;
    root) uid=0 ;;
    wayland) uid=1006 type=wayland ;;
    live|different) uid=$TEST_UID ;;
    live-wayland) uid=$TEST_UID type=wayland ;;
    *) exit 1 ;;
esac
printf 'User=%s\nClass=%s\nType=%s\nActive=%s\nRemote=%s\n' "$uid" "$class" "$type" "$active" "$remote"
printf 'Display=:42\nLeader=%s\n' "${TEST_LEADER:-0}"
SCRIPT
cat > "$tmp/id" <<'SCRIPT'
#!/bin/sh
[ "$1" = -g ] && printf '2000\n'
SCRIPT
cat > "$tmp/setpriv" <<'SCRIPT'
#!/bin/sh
printf 'setpriv %s\n' "$*" | sed -E 's/deepin-echoflow-install-[0-9]+\.scope/deepin-echoflow-install-PID.scope/' >> "$TEST_LOG"
[ "$1" != "--reuid=${TEST_FAILED_UID:-}" ]
SCRIPT
cat > "$tmp/systemd-run" <<'SCRIPT'
#!/bin/sh
exit 99
SCRIPT
cat > "$tmp/deb-systemd-invoke" <<'SCRIPT'
#!/bin/sh
printf 'deb-systemd-invoke %s\n' "$*" >> "$TEST_LOG"
SCRIPT
cat > "$tmp/systemctl" <<'SCRIPT'
#!/bin/sh
if [ "$1" = list-units ]; then
    printf 'user@1001.service loaded active running\nuser@0.service loaded active running\n'
    exit 0
fi
printf 'systemctl %s\n' "$*" >> "$TEST_LOG"
SCRIPT
cat > "$tmp/pgrep" <<'SCRIPT'
#!/bin/sh
[ -n "${TEST_DDE_PID:-}" ] && printf '%s\n' "$TEST_DDE_PID"
SCRIPT
cat > "$tmp/busctl" <<'SCRIPT'
#!/bin/sh
printf 'Fcitx %s\n' "$8" >> "$TEST_LOG"
case "$8" in
    GetConnectionUnixProcessID) [ -n "${TEST_FCITX_PID:-}" ] && printf 'u %s\n' "$TEST_FCITX_PID" ;;
    NameHasOwner) [ "${TEST_BUS_FAILED:-0}" = 0 ] && printf 'b %s\n' "${TEST_HAS_OWNER:-true}" ;;
    CanRestart) [ "${TEST_CAN_RESTART_FAILED:-0}" = 0 ] && printf 'b %s\n' "${TEST_CAN_RESTART:-true}" ;;
    Restart) [ "${TEST_RESTART_FAILED:-0}" = 0 ] ;;
    *) exit 1 ;;
esac
SCRIPT
cat > "$tmp/deepin-echoflow-ui" <<'SCRIPT'
#!/bin/sh
printf 'UI DISPLAY=<%s> XAUTHORITY=<%s> WAYLAND_DISPLAY=<%s>\n' "${DISPLAY:-}" "${XAUTHORITY:-}" "${WAYLAND_DISPLAY:-}" >> "$TEST_LOG"
SCRIPT
cat > "$tmp/deepin-echoflow-session-stop" <<'SCRIPT'
#!/bin/sh
printf 'stop previous UI\n' >> "$TEST_LOG"
SCRIPT
cp "$ROOT/scripts/deepin-echoflow-session-start" "$tmp/"
chmod +x "$tmp"/*

expect_log() {
    diff -u <(printf '%s\n' "$@") "$TEST_LOG"
    : > "$TEST_LOG"
}

run_isolated() {
    # Resolve the UID after env -i drops fakeroot's preload.
    env -i PATH="$PATH" TEST_LOG="$TEST_LOG" bash -c '
        export TEST_UID="$UID"
        exec env "$@"
    ' bash "$@"
}

postinst="$ROOT/debian/postinst"
prerm="$ROOT/debian/prerm"
launcher="$tmp/deepin-echoflow-session-start"
scope='setsid --fork systemd-cat --identifier=deepin-echoflow-install systemd-run --user --scope --collect --quiet --unit=deepin-echoflow-install-PID.scope --property=PartOf=graphical-session.target /usr/bin/deepin-echoflow-session-start --after-install'
expected_scope() {
    printf 'setpriv --reuid=%s --regid=2000 --init-groups --reset-env env XDG_RUNTIME_DIR=/run/user/%s DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/%s/bus %s %s' "$1" "$1" "$1" "$scope" "$2"
}

sh "$postinst" configure
expect_log "$(expected_scope 1001 x11)" "$(expected_scope 1006 wayland)"

warning=$(TEST_FAILED_UID=1006 sh "$postinst" configure 2>&1)
[[ "$warning" == *'Could not start EchoFlow for desktop user 1006'* ]]
expect_log "$(expected_scope 1001 x11)" "$(expected_scope 1006 wayland)"

TEST_HEADLESS=1 sh "$postinst" configure
TEST_CHROOT=1 sh "$postinst" configure
DPKG_ROOT="$tmp/image" sh "$postinst" configure
sh "$postinst" abort-upgrade
TEST_CHROOT=1 sh "$prerm" remove
DPKG_ROOT="$tmp/image" sh "$prerm" remove
sh "$prerm" upgrade
[[ ! -s "$TEST_LOG" ]]

sh "$prerm" remove
expect_log 'deb-systemd-invoke --user stop deepin-echoflow-install-*.scope' \
           'setpriv --reuid=1001 --regid=2000 --init-groups --reset-env /usr/bin/deepin-echoflow-session-stop' \
           'deb-systemd-invoke --user stop deepin-echoflow.socket deepin-echoflow.service'

env -i DISPLAY=:42 XAUTHORITY="$tmp/path with spaces" XDG_SESSION_ID=live /usr/bin/sleep 60 &
desktop_pid=$!
for ((attempt = 0; attempt < 50; ++attempt)); do
    if [[ $(readlink "/proc/$desktop_pid/exe") == /usr/bin/sleep ]]; then break; fi
    sleep 0.01
done

reload='systemctl --user daemon-reload'
start='systemctl --user start deepin-echoflow.socket'
clear='systemctl --user unset-environment WAYLAND_DISPLAY QT_QPA_PLATFORM'
import='systemctl --user import-environment DISPLAY XAUTHORITY XDG_SESSION_ID XDG_SESSION_TYPE'
resume='systemctl --user try-restart deepin-echoflow.service'
prefix=('Fcitx GetConnectionUnixProcessID' "$reload" "$clear" "$import" "$start" 'stop previous UI' "$resume" 'Fcitx NameHasOwner')
ui="UI DISPLAY=<:42> XAUTHORITY=<$tmp/path with spaces> WAYLAND_DISPLAY=<>"
run_isolated TEST_FCITX_PID="$desktop_pid" "$launcher" --after-install live
expect_log "${prefix[@]}" 'Fcitx CanRestart' 'Fcitx Restart' "$ui"

warning=$(run_isolated TEST_DDE_PID="$desktop_pid" TEST_CAN_RESTART=false "$launcher" --after-install live 2>&1)
[[ "$warning" == *'does not support self-restart'* && "$warning" != *'-rd'* ]]
expect_log "${prefix[@]}" 'Fcitx CanRestart' "$ui"

warning=$(run_isolated TEST_LEADER="$desktop_pid" TEST_RESTART_FAILED=1 "$launcher" --after-install live 2>&1)
[[ "$warning" == *'Could not request a Fcitx5 restart'* ]]
expect_log "${prefix[@]}" 'Fcitx CanRestart' 'Fcitx Restart' "$ui"

warning=$(run_isolated TEST_LEADER="$desktop_pid" TEST_HAS_OWNER=false DISPLAY=:999 WAYLAND_DISPLAY=stale "$launcher" --after-install live 2>&1)
[[ "$warning" == *'Fcitx5 is not running'* ]]
expect_log "${prefix[@]}" "$ui"

warning=$(run_isolated TEST_LEADER="$desktop_pid" TEST_CAN_RESTART_FAILED=1 "$launcher" --after-install live 2>&1)
[[ "$warning" == *'Could not request a Fcitx5 restart'* && "$warning" != *'does not support self-restart'* ]]
expect_log "${prefix[@]}" 'Fcitx CanRestart' "$ui"

warning=$(run_isolated TEST_LEADER="$desktop_pid" TEST_BUS_FAILED=1 "$launcher" --after-install live 2>&1)
[[ "$warning" == *'Could not check whether Fcitx5 is running'* ]]
expect_log "${prefix[@]}" "$ui"

warning=$(run_isolated TEST_FCITX_PID="$desktop_pid" DISPLAY=:999 "$launcher" --after-install different 2>&1)
[[ "$warning" == *'No graphical environment'* ]]
expect_log 'Fcitx GetConnectionUnixProcessID'

kill "$desktop_pid"
wait "$desktop_pid" || true
wayland_socket="$tmp/wayland-0"
env -i WAYLAND_DISPLAY="$wayland_socket" XDG_SESSION_ID=live-wayland /usr/bin/python3 -c '
import os
import socket
import time

with socket.socket(socket.AF_UNIX) as compositor:
    compositor.bind(os.environ["WAYLAND_DISPLAY"])
    time.sleep(60)
' &
desktop_pid=$!
for ((attempt = 0; attempt < 50; ++attempt)); do
    if [[ -S "$wayland_socket" ]]; then break; fi
    sleep 0.01
done
run_isolated TEST_FCITX_PID="$desktop_pid" "$launcher" --after-install live-wayland
expect_log 'Fcitx GetConnectionUnixProcessID' "$reload" \
           'systemctl --user unset-environment DISPLAY XAUTHORITY QT_QPA_PLATFORM' \
           'systemctl --user import-environment WAYLAND_DISPLAY XDG_SESSION_ID XDG_SESSION_TYPE' \
           "$start" 'stop previous UI' "$resume" 'Fcitx NameHasOwner' 'Fcitx CanRestart' 'Fcitx Restart' \
           "UI DISPLAY=<> XAUTHORITY=<> WAYLAND_DISPLAY=<$wayland_socket>"

run_isolated DISPLAY=:0 "$launcher"
expect_log 'systemctl --user import-environment DISPLAY' "$start" 'UI DISPLAY=<:0> XAUTHORITY=<> WAYLAND_DISPLAY=<>'

echo 'ok   - Debian install recovers the selected desktop environment and removes session processes on uninstall'
