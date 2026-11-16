#!/usr/bin/env python3
"""Every chat C-ABI entry point must be bound by every chat binding.

`include/crispasr_chat.h` is hand-mirrored into five languages. Nothing in CI
compiles four of them: `bindings-go.yml` is the only binding workflow that
touches the chat surface, and even it did not trigger on `crispasr_chat.h` or
`src/chat.cpp` until this check landed beside it. `bindings/java` is built only
by `build.yml`, whose PR filter is `ggml`/`.gitmodules`; there is no PR-time
`cargo`, `dart test` or Java job at all. So a new entry point — or a renamed
one — could land bound in one language and missing from the rest, and every
required check would stay green.

Compiling five toolchains in CI to catch that is the expensive answer. The
cheap one is this: the risk is *drift*, and drift is visible in the text. This
parses the header for its exported functions and asserts each one is named in
each binding, the same way `tests/test-copies-in-sync.cpp` pins the duplicated
module sources and `tools/check-backend-wiring.py` pins the backend surface.

What it does NOT do: check signatures. A binding that declares
`crispasr_chat_count_tokens` with the wrong argument types passes here. This
catches the omission, not the mistranslation — per-language tests are what
catch the second, and those are the suites that need a model.

Usage:
    python tools/check-chat-binding-coverage.py          # exit 1 on a gap
    python tools/check-chat-binding-coverage.py --list   # print the symbols
"""

from __future__ import annotations

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(REPO, "include", "crispasr_chat.h")

# Each binding, and the file(s) that must name every entry point. A binding
# split across files (Java: a JNA interface plus the session wrapper) may
# satisfy a symbol from any of them.
BINDINGS: dict[str, list[str]] = {
    "python": ["python/crispasr/_binding.py"],
    "go": ["bindings/go/chat.go"],
    "rust": ["crispasr-sys/src/lib.rs"],
    "dart": ["flutter/crispasr/lib/src/chat.dart"],
    "java": [
        "bindings/java/src/main/java/io/github/ggerganov/whispercpp/chat/ChatNative.java",
        "bindings/java/src/main/java/io/github/ggerganov/whispercpp/chat/ChatLib.java",
    ],
}

# `CRISPASR_CHAT_API <return type> <name>(` — the exported functions. The
# trailing `(` is what separates them from the typedefs and the handle type,
# which are not entry points and are not expected in every binding.
DECL = re.compile(r"CRISPASR_CHAT_API\s+[A-Za-z_][A-Za-z0-9_ *]*?\b(crispasr_chat_[a-z0-9_]+)\s*\(")


def header_symbols(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # Drop block comments so a symbol named only in prose does not count.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return sorted(set(DECL.findall(text)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the exported symbols and exit")
    args = ap.parse_args()

    if not os.path.exists(HEADER):
        print(f"error: {HEADER} not found", file=sys.stderr)
        return 2
    symbols = header_symbols(HEADER)
    if args.list:
        for s in symbols:
            print(s)
        return 0
    if not symbols:
        # A header that parses to nothing would make every binding vacuously
        # complete, which is the one way this check could pass while blind.
        print("error: no CRISPASR_CHAT_API declarations found — has the header or the regex changed?",
              file=sys.stderr)
        return 2

    gaps = 0
    for lang, rel_paths in sorted(BINDINGS.items()):
        blobs = []
        for rel in rel_paths:
            path = os.path.join(REPO, rel)
            if not os.path.exists(path):
                print(f"MISSING FILE  {lang}: {rel}", file=sys.stderr)
                gaps += 1
                continue
            with open(path, encoding="utf-8") as fh:
                blobs.append(fh.read())
        joined = "\n".join(blobs)
        missing = [s for s in symbols if s not in joined]
        if missing:
            gaps += len(missing)
            for s in missing:
                print(f"UNBOUND  {lang}: {s}  (add it to {' or '.join(rel_paths)})", file=sys.stderr)
        else:
            print(f"ok  {lang:6s} binds all {len(symbols)} entry points")

    if gaps:
        print(f"\n{gaps} gap(s): a chat ABI entry point is not bound everywhere.", file=sys.stderr)
        return 1
    print(f"\nRESULT: PASS — {len(symbols)} entry points bound by {len(BINDINGS)} bindings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
