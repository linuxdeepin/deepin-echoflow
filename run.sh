#!/bin/sh
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

set -eu

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

if ! cmake --build build -j "$JOBS" 2>/dev/null; then
    cmake -S . -B build
    cmake --build build -j "$JOBS"
fi

if [ "$#" -gt 0 ]; then
    exec ./build/service/echoflow-service "$@"
fi

runtime_dir="/run/user/$(id -u)"
if [ ! -d "$runtime_dir" ] || [ ! -w "$runtime_dir" ]; then
    runtime_dir="${XDG_RUNTIME_DIR:-/tmp}"
fi
socket_path="$runtime_dir/echoflow-wakeup.sock"

if command -v systemctl >/dev/null 2>&1 \
    && systemctl --user is-active --quiet echoflow.socket; then
    echo "echoflow.socket is already active; use the installed service instead of run.sh." >&2
    exit 1
fi
if ! command -v systemd-socket-activate >/dev/null 2>&1; then
    echo "systemd-socket-activate is required to run the service from the build tree." >&2
    exit 1
fi

while :; do
    if systemd-socket-activate --datagram --listen="$socket_path" \
        ./build/service/echoflow-service; then
        sleep 1
    fi
    sleep 1
done
