#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/deepin-echoflow-ui-install-spec.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

CUSTOM_BUILD="$TEST_ROOT/custom-build"
DESTDIR_ROOT="$TEST_ROOT/destdir"
CUSTOM_PREFIX="/opt/Echo Flow"

cmake -S "$ROOT/ui-host" -B "$CUSTOM_BUILD" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$CUSTOM_PREFIX" >/dev/null
cmake --build "$CUSTOM_BUILD" -j2 >/dev/null
DESTDIR="$DESTDIR_ROOT" cmake --install "$CUSTOM_BUILD" >/dev/null

INSTALLED_DESKTOP="$DESTDIR_ROOT$CUSTOM_PREFIX/share/applications/deepin-echoflow.desktop"
INSTALLED_ICON="$DESTDIR_ROOT$CUSTOM_PREFIX/share/icons/hicolor/scalable/apps/deepin-echoflow.svg"
INSTALLED_BINARY="$DESTDIR_ROOT$CUSTOM_PREFIX/bin/deepin-echoflow-ui"

[[ -x "$INSTALLED_BINARY" ]] || { echo "FAIL - standalone UI binary not installed" >&2; exit 1; }
[[ ! -e "$INSTALLED_DESKTOP" ]] || { echo "FAIL - obsolete application launcher was installed" >&2; exit 1; }
[[ -f "$INSTALLED_ICON" ]] || { echo "FAIL - icon not installed" >&2; exit 1; }
echo "ok   - standalone UI configure, build, and DESTDIR install"
