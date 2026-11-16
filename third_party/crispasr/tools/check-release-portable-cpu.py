#!/usr/bin/env python
"""Every redistributable x86_64 build must pin a portable CPU baseline.

    python tools/check-release-portable-cpu.py

Why this exists
---------------
Issue #374: the official Windows CUDA build of v0.8.29 died with SIGILL on any
CPU without AVX-512. `build-windows-cuda` set `-DGGML_CUDA=ON` and nothing else,
so `GGML_NATIVE` stayed ON and ggml compiled with `-march=native` — against a
GitHub runner that HAS AVX-512. The binary then executed AVX-512 on user
machines that do not.

Nine of eleven GPU jobs were in that state, including every Linux CUDA/HIP/Vulkan
one. The near-miss is the instructive part: `build-libs-windows-x86_64-cuda` DID
pin the baseline while `build-windows-cuda` — the crispasr.exe users actually
download — did not. The class was already known (#261 added a startup warning for
exactly this SIGILL) and the flag had been applied to some jobs and not others,
which is precisely the failure a per-job convention cannot prevent and a check
can.

release.yml only executes on a tag push, so its first run IS the release. A
missing flag here is not caught by CI, by review, or by anything short of a user
on the wrong CPU — which is how it reached v0.8.29.

What it enforces
----------------
Any job that builds for x86_64 must pin the CPU baseline, by either mechanism:

  * `-DGGML_NATIVE=OFF` (optionally with explicit `-DGGML_AVX2=ON` etc.), or
  * `-DCRISPASR_PORTABLE_CPU=ON`, which force-disables NATIVE and every optional
    ISA in CMakeLists.txt.

Declaring an ISA is not enough on its own — the avx512 artifacts set
`-DGGML_AVX512=ON` and still inherited `-march=native`, so they could pick up
AMX or AVX512-BF16 the artifact never promised. NATIVE=OFF is what makes the
shipped ISA equal to the declared one.

ARM jobs are exempt (they pin `GGML_CPU_ARM_ARCH` instead), as are jobs that run
no cmake at all.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"

# Jobs that legitimately build no redistributable x86_64 binary.
EXEMPT_SUBSTRINGS = ("arm64", "aarch64", "android", "wasm", "macos", "xcframework")


def jobs(lines: list[str]) -> list[tuple[str, int, int]]:
    start = next(i for i, line in enumerate(lines) if re.match(r"^jobs:\s*$", line))
    marks: list[tuple[int, str]] = []
    for i in range(start + 1, len(lines)):
        m = re.match(r"^  ([A-Za-z0-9_-]+):", lines[i])
        if m and not lines[i].startswith("    "):
            marks.append((i, m.group(1)))
    marks.append((len(lines), ""))
    return [(n, a, b) for (a, n), (b, _) in zip(marks, marks[1:]) if n]


def main() -> int:
    lines = WORKFLOW.read_text().splitlines()
    failures: list[str] = []
    checked = 0
    for name, a, b in jobs(lines):
        body = "\n".join(lines[a:b])
        if not re.search(r"\bcmake\b", body):
            continue
        if any(s in name.lower() for s in EXEMPT_SUBSTRINGS):
            continue
        if "runs-on: macos" in body:
            continue
        checked += 1
        if "GGML_NATIVE=OFF" not in body and "CRISPASR_PORTABLE_CPU=ON" not in body:
            gpu = [g for g, k in (("CUDA", "GGML_CUDA=ON"), ("HIP", "GGML_HIP=ON"),
                                  ("Vulkan", "GGML_VULKAN=ON")) if k in body]
            failures.append(
                f"  {name}: no GGML_NATIVE=OFF"
                + (f" (builds {'+'.join(gpu)})" if gpu else "")
            )

    if failures:
        print("RESULT: FAIL — redistributable x86_64 jobs compiled with -march=native\n")
        print("\n".join(failures))
        print(
            "\nThese ship binaries that execute whatever ISA the RUNNER happens to have.\n"
            "GitHub's x86_64 runners have AVX-512; most user CPUs do not, and the result\n"
            "is a SIGILL at startup on their machine (#374, and #261 before it).\n\n"
            "Add to the cmake invocation:\n"
            "    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON\n"
            "or -DCRISPASR_PORTABLE_CPU=ON for the generic x86-64 baseline.\n"
        )
        return 1

    print(f"RESULT: PASS — {checked} redistributable x86_64 job(s) pin a portable CPU baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
