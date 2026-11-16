#!/usr/bin/env python3
"""Consolidate scattered GGUFs into one dir, then purge non-Q4_K files that
have a verified copy on HuggingFace.

Two phases:
  1. MOVE every *.gguf under --src-dirs to --dest (skipping *-ref.gguf and
     files already in --dest). Symlinks are resolved; the real file is moved.
  2. PURGE: for each *.gguf in --dest, keep if filename contains '-q4_k';
     otherwise check cstr/<basename>-GGUF on HuggingFace. Delete only if
     filename matches AND size matches (within 1 KB).

Defaults to dry-run. Pass --apply to execute.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

DEFAULT_SRC_DIRS = [
    Path.home() / "code" / "parakeet-rnnt-temp",
    Path.home() / "code" / "parakeet-rnnt-hf-staging",
    Path.home() / "code" / "parakeet-rnnt-1.1b-hf-staging",
    Path.home() / "code" / "sensevoice-quant-stash",
    Path("/Volumes/backups/ai/crispasr-staging"),
    Path("/Volumes/backups/ai"),  # root, non-recursive — see SCAN_NONRECURSIVE
]

# Dirs scanned non-recursively (root of /Volumes/backups/ai has many GGUFs
# but its subdirs are typically tool-specific caches we don't want to touch).
SCAN_NONRECURSIVE = {Path("/Volumes/backups/ai")}

DEFAULT_DEST = Path("/Volumes/backups/ai/crispasr")
DEFAULT_HF_ORG = "cstr"
SIZE_TOLERANCE = 1024  # bytes

# Matches *-ref.gguf, *-ref-<tag>.gguf, *-ref_<tag>.gguf — local validation
# fixtures that must never be deleted.
REF_PATTERN = re.compile(r"-ref([._-]|\.gguf$)")

# Strip these suffixes from filename stem to get the model basename used in
# the HuggingFace repo name (cstr/<basename>-GGUF).
QUANT_SUFFIXES = [
    "-q4_k", "-q4_0", "-q4_1",
    "-q5_k", "-q5_0", "-q5_1",
    "-q6_k", "-q8_0",
    "-f16", "-f32", "-bf16",
]


def find_ggufs(src_dirs: list[Path]) -> list[Path]:
    """Walk src dirs and return all *.gguf files (excluding *-ref.gguf)."""
    found = []
    seen = set()
    for d in src_dirs:
        if not d.exists():
            continue
        recursive = d not in SCAN_NONRECURSIVE
        if recursive:
            it = d.rglob("*.gguf")
        else:
            it = d.glob("*.gguf")
        for p in it:
            if REF_PATTERN.search(p.name):
                continue
            if p.name.endswith(".download.gguf"):
                continue
            # resolve symlinks to dedupe; skip dangling symlinks
            try:
                real = p.resolve(strict=True)
            except (FileNotFoundError, OSError):
                print(f"  WARN dangling symlink, skipping: {p}", file=sys.stderr)
                continue
            if real in seen:
                continue
            seen.add(real)
            found.append(p)
    return found


def derive_basenames(filename: str) -> list[str]:
    """Strip the quant suffix and return candidate basenames to try as the
    HF repo name. Returns progressively shorter candidates so e.g.
    'chatterbox-t3-q8_0' yields ['chatterbox-t3', 'chatterbox']."""
    stem = filename[:-5] if filename.endswith(".gguf") else filename
    base = None
    for suf in sorted(QUANT_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            break
        # also handle '-regen' / '-fix' tail like chatterbox-t3-q4_k-regen
        m = re.match(rf"^(.+){re.escape(suf)}(-[a-z0-9]+)?$", stem)
        if m:
            base = m.group(1)
            break
    if base is None:
        return []
    candidates = [base]
    # try progressively shorter prefixes (drop trailing -components)
    parts = base.split("-")
    while len(parts) > 1:
        parts.pop()
        candidates.append("-".join(parts))
    return candidates


def is_q4_k(filename: str) -> bool:
    return "-q4_k" in filename.lower()


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def hf_file_size(api, repo: str, filename: str) -> int | None:
    """Return size of `filename` in `repo` if present, else None."""
    try:
        tree = api.list_repo_tree(repo, repo_type="model")
        for f in tree:
            if f.rfilename == filename and f.size is not None:
                return f.size
    except Exception:
        return None
    return None


def move_phase(ggufs: list[Path], dest: Path, apply: bool) -> list[Path]:
    """Move all source files into dest. Returns list of files now in dest
    (whether moved this run or already there)."""
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    dest_real = dest.resolve()
    for src in ggufs:
        try:
            src_real = src.resolve(strict=True)
        except (FileNotFoundError, OSError):
            print(f"  WARN dangling, skipping: {src}", file=sys.stderr)
            continue
        if src_real.parent == dest_real:
            moved.append(src_real)
            continue
        target = dest / src_real.name
        if target.is_symlink() and not target.exists():
            print(f"  DANGLING target symlink, removing: {target} -> {os.readlink(target)}")
            if apply:
                target.unlink()
        if target.exists():
            tgt_size = target.stat().st_size
            src_size = src_real.stat().st_size
            if tgt_size == src_size:
                print(f"  SKIP (already in dest, same size): {src} -> {target}")
                # delete the duplicate at the source (if it's the real file)
                if apply:
                    if src.is_symlink():
                        src.unlink()
                    elif src_real == src:
                        src.unlink()
                moved.append(target)
                continue
            else:
                print(f"  CONFLICT (size mismatch): {src} ({fmt_size(src_size)}) vs {target} ({fmt_size(tgt_size)}) — leaving both", file=sys.stderr)
                continue
        size = src_real.stat().st_size
        print(f"  MOVE: {src_real} -> {target}  ({fmt_size(size)})")
        if apply:
            # cross-device move: shutil.move handles that
            shutil.move(str(src_real), str(target))
            # if the original was a symlink, remove the dangling link
            if src.is_symlink():
                try:
                    src.unlink()
                except FileNotFoundError:
                    pass
        moved.append(target)
    return moved


def purge_phase(dest: Path, apply: bool, hf_org: str) -> None:
    """Delete non-Q4_K files in dest that are verified to exist on HF."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed; cannot verify HF presence")
    api = HfApi()

    files = sorted(dest.glob("*.gguf"))
    kept_q4k = 0
    deleted = 0
    kept_no_hf = 0
    dangling = 0
    bytes_freed = 0

    for f in files:
        # dangling symlink — clean up
        if f.is_symlink() and not f.exists():
            print(f"  DANGLING symlink, removing: {f.name} -> {os.readlink(f)}")
            if apply:
                f.unlink()
            dangling += 1
            continue
        if REF_PATTERN.search(f.name):
            print(f"  KEEP (ref fixture): {f.name}")
            continue
        if is_q4_k(f.name):
            kept_q4k += 1
            continue
        candidates = derive_basenames(f.name)
        if not candidates:
            print(f"  KEEP (unrecognized quant suffix): {f.name}")
            kept_no_hf += 1
            continue
        local_size = f.stat().st_size
        matched_repo = None
        hf_size = None
        for basename in candidates:
            repo = f"{hf_org}/{basename}-GGUF"
            sz = hf_file_size(api, repo, f.name)
            if sz is not None:
                matched_repo = repo
                hf_size = sz
                break
        if matched_repo is None:
            tried = ", ".join(f"{hf_org}/{b}-GGUF" for b in candidates)
            print(f"  KEEP (not on HF — tried {tried}): {f.name}  ({fmt_size(local_size)})")
            kept_no_hf += 1
            continue
        if abs(hf_size - local_size) > SIZE_TOLERANCE:
            print(f"  KEEP (size mismatch on {matched_repo}: HF={hf_size}B vs local={local_size}B, diff={hf_size-local_size}B): {f.name}")
            kept_no_hf += 1
            continue
        print(f"  DELETE (verified on {matched_repo}, {fmt_size(local_size)}): {f.name}")
        if apply:
            f.unlink()
        deleted += 1
        bytes_freed += local_size

    print()
    print(f"Summary: kept Q4_K={kept_q4k}  kept (no HF match)={kept_no_hf}  "
          f"dangling={dangling}  deleted={deleted}  freed={fmt_size(bytes_freed)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dirs", nargs="*", type=Path, default=DEFAULT_SRC_DIRS,
                    help="directories to scan for GGUFs (default: parakeet stash dirs + /Volumes/backups/ai root)")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"destination dir (default: {DEFAULT_DEST})")
    ap.add_argument("--hf-org", default=DEFAULT_HF_ORG,
                    help=f"HuggingFace org to check (default: {DEFAULT_HF_ORG})")
    ap.add_argument("--apply", action="store_true",
                    help="actually move/delete (default: dry-run)")
    ap.add_argument("--skip-move", action="store_true",
                    help="skip the move phase, only run purge on --dest")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN — no files will be touched. Re-run with --apply to execute.\n")

    if not args.skip_move:
        print(f"=== MOVE phase ===  dest={args.dest}")
        ggufs = find_ggufs(args.src_dirs)
        print(f"  found {len(ggufs)} candidate .gguf files in src dirs")
        moved = move_phase(ggufs, args.dest, args.apply)
        print(f"  moved/already-in-dest: {len(moved)}")
        print()

    print(f"=== PURGE phase ===  dest={args.dest}  hf-org={args.hf_org}")
    purge_phase(args.dest, args.apply, args.hf_org)


if __name__ == "__main__":
    main()
