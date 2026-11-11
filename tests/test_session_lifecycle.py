#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise launcher shutdown using isolated installations and real processes."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class SessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="echoflow-session-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindir = self.root / "prefix with spaces" / "bin"
        self.bindir.mkdir(parents=True)
        self.marker = self.root / "ui-pids"
        self.children = []
        self.ui_pids = []
        self.addCleanup(self.stop_children)
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        self.write_executable(fake_bin / "systemctl", "#!/bin/sh\nexit 0\n")
        self.env = dict(os.environ, PATH=f"{fake_bin}:/usr/bin:/bin", UI_PIDS=str(self.marker))
        self.write_executable(self.bindir / "echoflow-ui", f"""#!{sys.executable}
import os
import time
with open(os.environ['UI_PIDS'], 'a') as marker:
    marker.write(str(os.getpid()) + '\\n')
time.sleep(60)
""")

    @staticmethod
    def write_executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def spawn(self, command):
        process = subprocess.Popen(command, env=self.env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        self.children.append(process)
        return process

    def wait_ui(self, count=1):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pids = self.marker.read_text().splitlines() if self.marker.exists() else []
            if len(pids) >= count:
                self.ui_pids = [int(pid) for pid in pids]
                return self.ui_pids[-1]
            time.sleep(0.02)
        self.fail("launcher did not start the UI")

    def stop_children(self):
        # Also clean up if an assertion fails before the normal stop path.
        pids = self.marker.read_text().splitlines() if self.marker.exists() else []
        for pid in [p.pid for p in self.children] + [int(pid) for pid in pids]:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in self.children:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def stop(self):
        result = subprocess.run([str(ROOT / "scripts/echoflow-session-stop"), str(self.bindir)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stops_launcher_and_ui_preserving_other_installation(self):
        shutil.copy(ROOT / "scripts/echoflow-session-start", self.bindir)
        other = self.root / "other-installation"
        other.mkdir()
        shutil.copy(self.bindir / "echoflow-ui", other)
        unrelated = self.spawn([str(other / "echoflow-ui")])
        self.wait_ui()
        launcher = self.spawn([str(self.bindir / "echoflow-session-start")])
        ui_pid = self.wait_ui(2)
        self.stop()
        self.assertEqual(launcher.wait(timeout=3), 0)
        self.assertFalse(running(ui_pid))
        self.assertIsNone(unrelated.poll())

    def test_stops_legacy_supervisor_without_respawning_ui(self):
        launcher_path = self.bindir / "echoflow-session-start"
        self.write_executable(launcher_path, '#!/bin/sh\nwhile :; do\n'
                              '  "$(dirname "$0")/echoflow-ui"\n  sleep 0.05\ndone\n')
        launcher = self.spawn([str(launcher_path)])
        ui_pid = self.wait_ui()
        self.stop()
        launcher.wait(timeout=3)
        self.assertFalse(running(ui_pid))
        time.sleep(0.15)
        self.assertEqual(len(self.marker.read_text().splitlines()), 1)

    def test_stops_launcher_through_symlinked_prefix(self):
        shutil.copy(ROOT / "scripts/echoflow-session-start", self.bindir)
        alias = self.root / "linked-prefix"
        alias.symlink_to(self.bindir.parent, target_is_directory=True)
        launcher = self.spawn([str(alias / "bin/echoflow-session-start")])
        ui_pid = self.wait_ui()
        self.stop()
        self.assertEqual(launcher.wait(timeout=3), 0)
        self.assertFalse(running(ui_pid))

    def test_stops_replaced_executable(self):
        binary = self.bindir / "echoflow-ui"
        binary.unlink()
        shutil.copyfile(shutil.which("sleep"), binary)
        binary.chmod(0o755)
        process = self.spawn([str(binary), "60"])
        replacement = self.bindir / "replacement"
        shutil.copyfile(binary, replacement)
        replacement.replace(binary)
        self.stop()
        process.wait(timeout=3)

    def test_clean_ui_exit_stops_launcher(self):
        shutil.copy(ROOT / "scripts/echoflow-session-start", self.bindir)
        self.write_executable(self.bindir / "echoflow-ui", "#!/bin/sh\nexit 0\n")
        launcher = self.spawn([str(self.bindir / "echoflow-session-start")])
        self.assertEqual(launcher.wait(timeout=5), 0)

    def test_crashed_ui_restarts_and_can_be_stopped(self):
        shutil.copy(ROOT / "scripts/echoflow-session-start", self.bindir)
        ui = self.bindir / "echoflow-ui"
        ui.write_text(ui.read_text().replace("time.sleep(60)",
            "if not os.path.exists(os.environ['UI_PIDS'] + '.crashed'):\n"
            "    open(os.environ['UI_PIDS'] + '.crashed', 'w').close()\n"
            "    raise SystemExit(1)\ntime.sleep(60)"))
        launcher = self.spawn([str(self.bindir / "echoflow-session-start")])
        ui_pid = self.wait_ui(2)
        self.stop()
        self.assertEqual(launcher.wait(timeout=3), 0)
        self.assertFalse(running(ui_pid))


if __name__ == "__main__":
    unittest.main()
