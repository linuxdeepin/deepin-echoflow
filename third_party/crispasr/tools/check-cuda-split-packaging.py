#!/usr/bin/env python3
"""Every Windows package that bundles the CUDA runtime must also ship a split one (#342).

#342 asks for the three Windows CUDA runtime DLLs (cudart64, cublas64,
cublasLt64) as separate downloads, with each release shipping an archive that
leaves them out: they change far less often than we cut releases, and for users
on a slow link to github.com re-downloading them every time is pure waste.

That was implemented for `build-windows-cuda` in 0ae65086 — and missed on
`build-libs-windows-x86_64-cuda`, which at 874 MB is the *largest* asset in the
release and therefore the one where a slow connection hurts most. Two surfaces,
one fixed. This script is the reason the next one cannot be missed: it finds
every job that copies those DLLs into a package and requires the full treatment
of each.

Enforced, per bundling job:

  1. A split archive is produced whose name ends in `-non-cuda`.
  2. The split archive is attached to the release, not merely uploaded as a
     workflow artifact — an artifact is invisible to the people in #342.
  3. The bundling is verified rather than best-effort. Both jobs copy with
     `-ErrorAction SilentlyContinue`, so a wrong CUDA_PATH yields a silently
     CPU-only package named "cuda"; a job must therefore assert it ended up
     with exactly three DLLs.
  4. All bundling jobs on the SAME CUDA major pin the SAME CUDA version. The
     three DLLs are published once per major and shared across that major's
     packages, which is only sound while every producer in the major builds
     against the same toolkit — bumping one job alone would leave the other's
     users installing mismatched DLLs. Different majors are fine (#400 added
     CUDA 13 packages next to the CUDA 12 ones): the DLL file names carry the
     major (cudart64_12.dll vs cudart64_13.dll), so the assets cannot collide
     or be cross-installed silently.

Usage:
    python tools/check-cuda-split-packaging.py [WORKFLOW ...]
"""

from __future__ import annotations

import os
import re
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not a finding
    print("error: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    raise SystemExit(2)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = [os.path.join(REPO, ".github", "workflows", "release.yml")]

# The three DLLs #342 is about. Matched against the copy commands, so this also
# catches a job that bundles only some of them.
DLL_PATTERN = re.compile(r"\b(cudart64|cublas64|cublasLt64)_")
SPLIT_NAME = re.compile(r"[\w.-]*-non-cuda[\w.-]*")
CUDA_VERSION_KEYS = ("cuda", "cuda-version")


def steps_of(job: dict) -> list:
    return job.get("steps") or []


def release_files(job: dict) -> str:
    """Concatenated `files:` of every step that attaches assets to a release."""
    out = []
    for st in steps_of(job):
        if "action-gh-release" in (st.get("uses") or ""):
            files = (st.get("with") or {}).get("files")
            if files:
                out.append(files if isinstance(files, str) else "\n".join(files))
    return "\n".join(out)


def cuda_version(job: dict) -> str | None:
    for st in steps_of(job):
        if "cuda-toolkit" in (st.get("uses") or "").lower():
            with_ = st.get("with") or {}
            for k in CUDA_VERSION_KEYS:
                if k in with_:
                    return str(with_[k])
    return None


def main() -> int:
    paths = sys.argv[1:] or DEFAULT
    problems: list[str] = []
    bundling: dict[str, str | None] = {}
    examined = 0

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        for job_name, job in (doc.get("jobs") or {}).items():
            runs = "\n".join((st.get("run") or "") for st in steps_of(job))
            if not DLL_PATTERN.search(runs):
                continue
            examined += 1
            where = f"{os.path.basename(path)}:{job_name}"
            bundling[where] = cuda_version(job)

            split = SPLIT_NAME.findall(runs)
            if not split:
                problems.append(
                    f"{where}: bundles the CUDA runtime DLLs but produces no '-non-cuda' "
                    f"split archive (#342)"
                )
                continue

            attached = release_files(job)
            if not any(s in attached for s in split):
                problems.append(
                    f"{where}: builds a split archive ({split[0]}) but never attaches it to "
                    f"the release — a workflow artifact is not a download"
                )

            if "throw" not in runs:
                problems.append(
                    f"{where}: copies the DLLs with -ErrorAction SilentlyContinue and never "
                    f"asserts all three arrived; a wrong CUDA_PATH would ship a CPU-only "
                    f"package named 'cuda'"
                )

    if examined == 0:
        # A checker that examined nothing must not report success.
        print(
            "error: no CUDA-bundling job found — has the packaging moved, or the DLL names changed?",
            file=sys.stderr,
        )
        return 2

    # Lockstep is per CUDA major: the runtime DLL names carry the major
    # (cudart64_12.dll vs cudart64_13.dll), so majors publish disjoint assets,
    # but within a major every producer must pin the identical toolkit.
    by_major: dict[str, dict[str, str]] = {}
    for where, ver in bundling.items():
        if ver:
            by_major.setdefault(ver.split(".")[0], {})[where] = ver
    for major, group in sorted(by_major.items()):
        if len(set(group.values())) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(group.items()))
            problems.append(
                f"CUDA {major}.x versions differ across bundling jobs ({detail}). The three "
                f"DLLs are published once per major and shared between that major's packages, "
                f"which only holds while every producer builds against the same toolkit."
            )
    if None in bundling.values():
        missing = ", ".join(sorted(k for k, v in bundling.items() if v is None))
        problems.append(f"could not determine the pinned CUDA version for: {missing}")

    if problems:
        for p in problems:
            print(f"FAIL  {p}", file=sys.stderr)
        print(f"\n{len(problems)} problem(s) across {examined} CUDA-bundling job(s).", file=sys.stderr)
        return 1

    vers = ", ".join(sorted({v for g in by_major.values() for v in g.values()})) or "?"
    print(f"RESULT: PASS — {examined} CUDA-bundling job(s), all split and attached, on CUDA {vers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
