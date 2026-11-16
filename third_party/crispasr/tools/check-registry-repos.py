#!/usr/bin/env python3
"""check-registry-repos.py — audit that every registry download URL is publicly reachable.

Why this exists: issue #331. `cstr/piper-en_US-lessac-medium-GGUF` — the default
Piper voice — was flipped **private** by an out-of-band step during the 2026-08-03
speaker_identity stamping pass. Nothing in the repo changed, no test failed, and
the file was still there; it simply stopped being downloadable by anyone but the
owner. The first report came from a downstream catalog consumer a day later.

The check that matters is therefore **unauthenticated**: it reproduces exactly
what `-m auto --auto-download` and any outside consumer see. Running this with a
token in the environment would have found nothing wrong — the owner's token
returns 200 for a private repo. So this deliberately strips HF credentials.

Note the failure mode is indistinguishable from deletion from outside: HF answers
`401 Invalid username or password` for private *and* nonexistent repos. That is
why the reporter of #331 concluded the repo had been deleted. When this script
flags a repo, check `private` with an owner token before assuming data loss.

Run after ANY bulk operation over the published repos (card rewrites, metadata
stamping, re-uploads, visibility sweeps) — that is when repos silently change
state.

Usage:
    python tools/check-registry-repos.py [--verbose] [--jobs N]

Exit code: 0 if every referenced repo is publicly reachable, 1 otherwise.
"""

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "src" / "crispasr_model_registry.cpp"

# Matches the resolve URLs in the registry's string literals.
REPO_RE = re.compile(r"https://huggingface\.co/([^/\"]+/[^/\"]+)/resolve")

# HF credentials would mask exactly the bug this looks for (an owner token sees
# a private repo as 200), so they are cleared for the duration of the run.
for _var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
    os.environ.pop(_var, None)


def referenced_repos(path):
    """Every distinct `owner/name` the registry links to, in first-seen order."""
    text = path.read_text(encoding="utf-8")
    seen = {}
    for m in REPO_RE.finditer(text):
        seen.setdefault(m.group(1), True)
    return list(seen)


def check(repo):
    """(repo, ok, detail) — is this repo readable with no credentials at all?"""
    url = f"https://huggingface.co/api/models/{repo}"
    req = urllib.request.Request(url, headers={"User-Agent": "crispasr-registry-audit"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read(4096).decode("utf-8", "replace")
        # A gated repo is reachable but not freely downloadable — still a break
        # for --auto-download, so it is reported rather than passed.
        if '"gated":true' in body.replace(" ", ""):
            return repo, False, "gated (needs manual acceptance)"
        return repo, True, "public"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return repo, False, "401 — private or deleted (check with an owner token)"
        return repo, False, f"HTTP {e.code}"
    except Exception as e:  # network/DNS/timeout — inconclusive, not a repo fault
        return repo, False, f"unreachable: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="list every repo, not just failures")
    ap.add_argument("--jobs", type=int, default=8, help="parallel requests (default 8)")
    args = ap.parse_args()

    if not REGISTRY.exists():
        print(f"registry not found at {REGISTRY}", file=sys.stderr)
        return 1

    repos = referenced_repos(REGISTRY)
    if not repos:
        print(f"no huggingface.co resolve URLs found in {REGISTRY.name} — regex stale?",
              file=sys.stderr)
        return 1

    print(f"checking {len(repos)} repos referenced by {REGISTRY.name} (unauthenticated)\n")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(check, repos))

    bad = [r for r in results if not r[1]]
    if args.verbose:
        for repo, ok, detail in results:
            print(f"  {'ok  ' if ok else 'FAIL'}  {repo:<55} {detail}")
        print()

    if bad:
        print(f"{len(bad)} of {len(repos)} NOT publicly reachable:\n")
        for repo, _, detail in bad:
            print(f"  {repo:<55} {detail}")
        print("\nAn outside user cannot --auto-download these.")
        return 1

    print(f"all {len(repos)} repos publicly reachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
