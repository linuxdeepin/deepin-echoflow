#!/usr/bin/env python3
"""`core_cpu_backend::` is used in ~100 files; its header must be reachable in all of them.

src/core/ggml_cpu_backend.h was introduced for #355 and its call sites were
converted mechanically. The rewrite inserted the new `#include` after the first
`#include` already in each file — which in seven of them was inside a
conditional block:

    #ifndef _WIN32
    #include <dlfcn.h>
    #include "core/ggml_cpu_backend.h"   <- unreachable on Windows
    #endif

    #ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
    #include "core/ggml_cpu_backend.h"   <- unreachable everywhere else
    #endif

Every use of the namespace, though, is at top level. So the code compiled on
macOS — where `__APPLE__`, `HAVE_ACCELERATE` and `!_WIN32` all hold — and broke
on Windows with "'core_cpu_backend': is not a class or namespace name". Two of
the seven sat behind build options (`INDEXTTS_HAS_SUBPROCESS`,
`CRISPASR_ESPEAK_DLOPEN`) and so were configuration-dependent even on Linux.

A local build cannot catch this: the guards that hide it are exactly the ones
satisfied on the machine doing the building. Hence a preprocessor-independent
check — it reads the conditional nesting rather than evaluating it, so one run
covers every platform at once.

Rule: in any file that uses `core_cpu_backend::` outside all `#if` blocks, at
least one `#include` of the header must also be outside all `#if` blocks.

Usage:
    python tools/check-cpu-backend-include.py [DIR ...]
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIRS = ["src", "examples", "tests"]
SKIP_DIRS = {"ggml", "third_party", "build", ".git"}

HEADER = "core/ggml_cpu_backend.h"
NAMESPACE = "core_cpu_backend::"
IF = re.compile(r"^#\s*(if|ifdef|ifndef)\b")
ENDIF = re.compile(r"^#\s*endif\b")
INCLUDE = re.compile(r'^#\s*include\s*[<"]')
SOURCE_EXT = (".cpp", ".cc", ".cxx", ".h", ".hpp")


def scan(path: str):
    """-> (unconditional_include, unconditional_use_line or None, saw_use)"""
    depth = 0
    inc_uncond = False
    use_line = None
    saw_use = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            s = line.strip()
            if IF.match(s):
                depth += 1
                continue
            if ENDIF.match(s):
                depth = max(0, depth - 1)
                continue
            if INCLUDE.match(s) and HEADER in s and depth == 0:
                inc_uncond = True
            if NAMESPACE in s:
                # Comments mentioning the namespace are not uses.
                if s.startswith(("//", "*", "/*")):
                    continue
                saw_use = True
                if depth == 0 and use_line is None:
                    use_line = n
    return inc_uncond, use_line, saw_use


def main() -> int:
    roots = sys.argv[1:] or [os.path.join(REPO, d) for d in DEFAULT_DIRS]
    problems = []
    users = 0

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(SOURCE_EXT):
                    continue
                path = os.path.join(dirpath, fn)
                inc_uncond, use_line, saw_use = scan(path)
                if not saw_use:
                    continue
                users += 1
                if use_line is not None and not inc_uncond:
                    problems.append(
                        f"{os.path.relpath(path, REPO)}:{use_line}: uses {NAMESPACE} outside any "
                        f"#if, but every #include of {HEADER} is inside one — this builds only on "
                        f"platforms where that condition happens to hold"
                    )

    if users == 0:
        # A checker that examined nothing must not report success.
        print(
            f"error: no file uses {NAMESPACE} — has the shim been renamed or removed?",
            file=sys.stderr,
        )
        return 2
    if problems:
        for p in problems:
            print(f"FAIL  {p}", file=sys.stderr)
        print(f"\n{len(problems)} problem(s) across {users} files using {NAMESPACE}", file=sys.stderr)
        return 1
    print(f"RESULT: PASS — {users} files use {NAMESPACE}, all reach the header unconditionally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
