#!/usr/bin/env python3
"""Does this branch ITSELF change the files a workflow watches?

A heavy verify workflow triggered by `push: paths:` fires on every branch
whose push touches those paths -- and a push that REBASES a branch onto main
touches every path main changed, even when the branch changed none of them.
One rebase of many branches onto a main that edited CMakeLists.txt then queues
one Windows/CUDA verify per branch, and they starve every other workflow on the
account (measured 2026-09-28: 409 push runs across 77 branches in 35 minutes).

This compares the branch against its merge-base with main, not against the
previous tip, and matches the workflow's OWN `on.push.paths` list (the single
source of truth, so the gate cannot drift from the trigger). Prints
`relevant=true|false` for $GITHUB_OUTPUT.

    branch_touches_paths.py <workflow.yml> [<base-ref>=origin/main]

Anything it cannot decide (no paths list, no merge-base) answers true: this
gate may only SKIP work it can prove is unrelated, never hide work.
"""
import re
import subprocess
import sys


def glob_to_regex(pattern: str) -> re.Pattern:
    """GitHub path-filter glob -> regex: `**` any depth, `*` within a segment, `?` one char."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith('**/', i):
            out.append('(?:.*/)?'); i += 3
        elif pattern.startswith('**', i):
            out.append('.*'); i += 2
        elif c == '*':
            out.append('[^/]*'); i += 1
        elif c == '?':
            out.append('[^/]'); i += 1
        else:
            out.append(re.escape(c)); i += 1
    return re.compile('^' + ''.join(out) + '$')


def watched(workflow: str):
    import yaml  # inside the fail-open guard: a runner without PyYAML must RUN the job
    doc = yaml.safe_load(open(workflow))
    on = doc.get('on', doc.get(True)) or {}      # YAML 1.1 reads a bare `on:` key as True
    push = on.get('push') if isinstance(on, dict) else None
    return (push or {}).get('paths') if isinstance(push, dict) else None


def main() -> int:
    workflow = sys.argv[1]
    base_ref = sys.argv[2] if len(sys.argv) > 2 else 'origin/main'
    paths = watched(workflow)
    if not paths:
        print('relevant=true'); return 0
    try:
        base = subprocess.check_output(['git', 'merge-base', base_ref, 'HEAD'], text=True).strip()
        changed = subprocess.check_output(['git', 'diff', '--name-only', base, 'HEAD'], text=True).split()
    except subprocess.CalledProcessError:
        print('relevant=true'); return 0
    include = [glob_to_regex(p) for p in paths if not p.startswith('!')]
    exclude = [glob_to_regex(p[1:]) for p in paths if p.startswith('!')]
    hit = [f for f in changed
           if any(r.match(f) for r in include) and not any(r.match(f) for r in exclude)]
    for f in hit[:20]:
        print(f'  watched and changed on this branch: {f}', file=sys.stderr)
    print(f'relevant={"true" if hit else "false"}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:  # never let a gate crash skip the heavy job
        print(f'  scope gate error ({e!r}); running the job', file=sys.stderr)
        print('relevant=true')
