#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/echoflow-systemd-spec.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
HELPER="$(command -v deb-systemd-helper || true)"
if [[ -z "$HELPER" ]]; then
  echo 'SKIP - deb-systemd-helper is not installed'
  exit 0
fi

fail() { echo "FAIL - $*" >&2; exit 1; }

# Exercise the real Debian helper without touching the host service state.
mkdir -p "$TEST_ROOT/bin"
export HELPER
export DPKG_MAINTSCRIPT_PACKAGE=echoflow
cat > "$TEST_ROOT/bin/deb-systemd-helper" <<'EOF'
#!/bin/sh
DPKG_ROOT="$TEST_PACKAGE_ROOT" exec "$HELPER" "$@"
EOF
chmod +x "$TEST_ROOT/bin/deb-systemd-helper"

for scenario in enabled disabled fresh preset-disabled; do
  export TEST_PACKAGE_ROOT="$TEST_ROOT/$scenario"
  units="$TEST_PACKAGE_ROOT/usr/lib/systemd/user"
  mkdir -p "$units" "$TEST_PACKAGE_ROOT/usr/bin" "$TEST_PACKAGE_ROOT/etc/systemd/user-preset"
  printf '[Unit]\nDescription=Test user manager\n' > "$units/default.target"
  printf '[Unit]\nDescription=Test graphical session\n' > "$units/graphical-session.target"
  ln -s "$(command -v systemctl)" "$TEST_PACKAGE_ROOT/usr/bin/systemctl"
  if [[ "$scenario" == preset-disabled ]]; then
    printf 'disable echoflow*\n' > "$TEST_PACKAGE_ROOT/etc/systemd/user-preset/00-test.preset"
  fi
  for unit in echoflow.service echoflow-ui.service; do
    sed -e 's|@CMAKE_INSTALL_FULL_BINDIR@|/usr/bin|g' \
        -e 's/WantedBy=graphical-session.target/WantedBy=default.target/' \
        "$ROOT/systemd/user/$unit.in" > "$units/$unit"
    if [[ "$scenario" == enabled || "$scenario" == disabled ]]; then
      DPKG_ROOT="$TEST_PACKAGE_ROOT" "$HELPER" --user enable "$unit"
      if [[ "$scenario" == disabled ]]; then
        DPKG_ROOT="$TEST_PACKAGE_ROOT" "$HELPER" --user disable "$unit"
      fi
    fi
    sed 's|@CMAKE_INSTALL_FULL_BINDIR@|/usr/bin|g' \
      "$ROOT/systemd/user/$unit.in" > "$units/$unit"
  done

  # The maintscript must run before debhelper updates its recorded link state.
  env -u DPKG_ROOT PATH="$TEST_ROOT/bin:$PATH" sh "$ROOT/debian/postinst" configure
  for unit in echoflow.service echoflow-ui.service; do
    if [[ "$scenario" == fresh || "$scenario" == preset-disabled ]]; then
      if DPKG_ROOT="$TEST_PACKAGE_ROOT" "$HELPER" --quiet --user debian-installed "$unit"; then
        fail "migration bypassed first-install preset handling: $unit"
      fi
    fi
    # Complete the normal debhelper enable/update-state sequence.
    if DPKG_ROOT="$TEST_PACKAGE_ROOT" "$HELPER" --quiet --user was-enabled "$unit"; then
      DPKG_ROOT="$TEST_PACKAGE_ROOT" "$HELPER" --user enable "$unit"
    else
      DPKG_ROOT="$TEST_PACKAGE_ROOT" "$HELPER" --user update-state "$unit"
    fi
    old="$TEST_PACKAGE_ROOT/etc/systemd/user/default.target.wants/$unit"
    new="$TEST_PACKAGE_ROOT/etc/systemd/user/graphical-session.target.wants/$unit"
    [[ ! -L "$old" ]] || fail "$scenario retained boot-only link: $unit"
    if [[ "$scenario" == disabled || "$scenario" == preset-disabled ]]; then
      [[ ! -L "$new" ]] || fail "upgrade enabled a disabled service: $unit"
    else
      [[ -L "$new" ]] || fail "$scenario missing session autostart: $unit"
    fi
  done
done
echo 'ok   - Debian session autostart migration preserves disabled services and presets'
