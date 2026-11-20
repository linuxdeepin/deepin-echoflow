#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the autostart migration in isolated dpkg roots."""

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
AUTOSTART_DIR = Path("etc/xdg/autostart")
OLD_ENTRY = AUTOSTART_DIR / "echoflow-autostart.desktop"
NEW_ENTRY = AUTOSTART_DIR / "deepin-echoflow-autostart.desktop"
DESKTOP_ENTRY = "[Desktop Entry]\nType=Application\nName=EchoFlow\nExec=/bin/true\n"


def build_package(work, name, entry, extra_control="", postinst=None):
    staging = work / name
    metadata = staging / "DEBIAN"
    metadata.mkdir(parents=True)
    (metadata / "control").write_text(
        f"Package: {name}\nVersion: 1.0\nArchitecture: all\n"
        "Maintainer: EchoFlow test <test@example.org>\n"
        f"Description: EchoFlow autostart test fixture\n{extra_control}",
        encoding="utf-8",
    )
    (metadata / "conffiles").write_text(f"/{entry}\n", encoding="utf-8")
    (staging / entry).parent.mkdir(parents=True)
    desktop = DESKTOP_ENTRY + ("NoDisplay=true\n" if entry == NEW_ENTRY else "")
    (staging / entry).write_text(desktop, encoding="utf-8")
    if postinst is not None:
        script = metadata / "postinst"
        script.write_text(postinst, encoding="utf-8")
        script.chmod(0o755)

    deb = work / f"{name}.deb"
    subprocess.run(
        ["dpkg-deb", "--root-owner-group", "--build", staging, deb], check=True,
        stdout=subprocess.DEVNULL,
    )
    return deb


def create_dpkg_root(root):
    admin_dir = root / "var/lib/dpkg"
    admin_dir.mkdir(parents=True)
    (admin_dir / "status").touch()


def install(root, deb, action="--install"):
    env = os.environ.copy()
    env.pop("DPKG_ADMINDIR", None)
    env.pop("DPKG_ROOT", None)
    result = subprocess.run(
        ["dpkg", "--force-not-root", "--force-script-chrootless", f"--root={root}",
         f"--log={root / 'dpkg.log'}", action, deb],
        env=env, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"dpkg {action} failed for {deb}:\n{result.stdout}{result.stderr}")


def main():
    with tempfile.TemporaryDirectory(prefix="deepin-echoflow-deb-upgrade-") as tmp:
        work = Path(tmp)
        old_deb = build_package(work, "echoflow", OLD_ENTRY)
        control = (ROOT / "debian/control").read_text(encoding="utf-8")
        relationships = "".join(
            f"{line}\n" for line in control.splitlines()
            if line.startswith(("Conflicts:", "Replaces:", "Provides:"))
        )
        new_deb = build_package(
            work, "deepin-echoflow", NEW_ENTRY, relationships,
            (ROOT / "debian/postinst").read_text(encoding="utf-8"),
        )

        upgrade = work / "upgrade"
        create_dpkg_root(upgrade)
        install(upgrade, old_deb)
        old_entry = upgrade / OLD_ENTRY
        custom_content = DESKTOP_ENTRY + "# Local edit\n"
        old_entry.write_text(custom_content, encoding="utf-8")
        install(upgrade, new_deb)
        assert not old_entry.exists(), "old package conffile can still appear in DDE"
        backups = list(old_entry.parent.glob(f"{old_entry.name}.bak.*"))
        assert len(backups) == 1, "old conffile was not backed up once"
        assert backups[0].read_text(encoding="utf-8") == custom_content
        assert "NoDisplay=true\n" in (upgrade / NEW_ENTRY).read_text(encoding="utf-8")
        install(upgrade, new_deb)
        assert list(old_entry.parent.glob(f"{old_entry.name}.bak.*")) == backups

        fresh = work / "fresh"
        create_dpkg_root(fresh)
        unowned_entry = fresh / OLD_ENTRY
        unowned_entry.parent.mkdir(parents=True)
        unowned_entry.write_text(custom_content, encoding="utf-8")
        install(fresh, new_deb)
        assert unowned_entry.read_text(encoding="utf-8") == custom_content
        assert not list(unowned_entry.parent.glob(f"{unowned_entry.name}.bak.*"))

    print("ok   - Debian upgrade archives the old package entry and preserves unowned files")


if __name__ == "__main__":
    main()
