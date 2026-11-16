"""Propagate VERSION into every file that carries a version string.

Two modes:

    python scripts/sync-version.py            rewrite the files (used by bump-version.sh)
    python scripts/sync-version.py --check    verify only; exit 1 on drift (used by CI)

`--check` exists because this tooling only ever ran at release time, so its bugs
stayed invisible until a release was already half-published. Two real ones:

  * `python/crispasr/__init__.py` was added here but not to bump-version.sh's
    staged file list, so v0.8.24 was first tagged with `crispasr.__version__`
    reading 0.8.23 while the wheel metadata said 0.8.24.
  * `flutter/crispasr/CHANGELOG.md` is not touched by either script, and pub.dev
    DRY-RUN-FAILS a publish whose changelog lacks the version being published —
    which is why 0.8.23 never reached pub.dev at all.

Running this on every push turns both into a red check on the commit that caused
them, instead of a surprise during a release.
"""
import sys
import re
import os


def update_file(file_path, patterns, version, check=False):
    """Sync (or verify) one file. Returns a list of human-readable problems."""
    if not os.path.exists(file_path):
        if not check:
            print(f"Skipping {file_path} (not found)")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    problems = []
    new_content = content
    for pattern, replacement in patterns:
        # A pattern that matches NOTHING is the dangerous case: re.sub happily
        # returns the text unchanged, so the file silently never gets synced and
        # nothing anywhere complains. That is how a renamed field would slip
        # through, so treat it as an error rather than a no-op.
        if not re.search(pattern, content, flags=re.MULTILINE):
            problems.append(
                f"{file_path}: pattern {pattern!r} matched nothing — the field was "
                f"renamed or removed, so this file is silently never synced")
            continue
        new_content = re.sub(pattern, replacement.replace("{version}", version),
                             new_content, flags=re.MULTILINE)

    if check:
        if new_content != content:
            problems.append(f"{file_path}: version does not match VERSION ({version})")
        return problems

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes for {file_path}")
    return problems


def check_flutter_changelog(version):
    """pub.dev refuses to publish a version absent from the changelog.

    bump-version.sh does not touch this file, so it silently falls behind and the
    failure only surfaces at `dart pub publish` — after the tag exists.
    """
    path = 'flutter/crispasr/CHANGELOG.md'
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        text = f.read()
    # Accept `## 0.8.24` and `## [0.8.24]`.
    if re.search(rf'^##\s*\[?{re.escape(version)}\]?', text, flags=re.MULTILINE):
        return []
    return [f"{path}: no '## {version}' entry — pub.dev will refuse to publish "
            f"this version (bump-version.sh does not write this file; add it by hand)"]


def run(version, check):
    problems = []

    # Rust
    problems += update_file('crispasr/Cargo.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"'),
        (r'crispasr-sys = \{ path = "\.\./crispasr-sys", version = "[^"]+" \}',
         'crispasr-sys = { path = "../crispasr-sys", version = "{version}" }')
    ], version, check)
    problems += update_file('crispasr-sys/Cargo.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"')
    ], version, check)

    # Python
    problems += update_file('python/pyproject.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"')
    ], version, check)
    # The package __version__ is hand-maintained (not read from pyproject at
    # runtime), so keep it in lockstep too — otherwise `crispasr.__version__`
    # drifts from the published wheel version.
    problems += update_file('python/crispasr/__init__.py', [
        (r'^__version__ = "[^"]+"', '__version__ = "{version}"')
    ], version, check)

    # Dart/Flutter
    problems += update_file('flutter/crispasr/pubspec.yaml', [
        (r'^version: [^\n]+', 'version: {version}')
    ], version, check)

    # JavaScript / Bindings
    problems += update_file('bindings/javascript/package.json', [
        (r'^  "version": "[^"]+"', '  "version": "{version}"')
    ], version, check)

    problems += check_flutter_changelog(version)
    return problems


if __name__ == "__main__":
    check = "--check" in sys.argv[1:]

    version_file = 'VERSION'
    if not os.path.exists(version_file):
        print(f"Error: {version_file} file not found")
        sys.exit(1)

    with open(version_file, 'r', encoding='utf-8') as f:
        version = f.read().strip()

    if not check:
        print(f"Synchronizing version to {version}...")

    problems = run(version, check)

    if check:
        if problems:
            print(f"VERSION is {version}, but {len(problems)} thing(s) disagree:\n",
                  file=sys.stderr)
            for p in problems:
                print(f"  ::error::{p}", file=sys.stderr)
            print("\nRun `python scripts/sync-version.py` to fix the derived files "
                  "(the changelog entry must be written by hand).", file=sys.stderr)
            sys.exit(1)
        print(f"OK: every versioned file agrees with VERSION ({version}).")
        sys.exit(0)

    # In write mode a non-matching pattern is still worth shouting about.
    for p in problems:
        print(f"WARNING: {p}", file=sys.stderr)
    print("Version synchronization complete.")
