#!/usr/bin/env python3
"""Upload tada-ref-<lang>.gguf files to both TADA HF repos.

Usage:
    python tools/upload_tada_lang_refs.py \
        --dir /Volumes/backups/code/tada-lang-refs-stash/kaggle-out/lang-refs/
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi

REPOS = [
    "cstr/tada-tts-1b-GGUF",
    "cstr/tada-tts-3b-ml-GGUF",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--langs", nargs="+", default=None,
                    help="Only upload these language codes (default: all found)")
    args = ap.parse_args()

    api = HfApi()
    d = Path(args.dir)
    files = sorted(d.glob("tada-ref-*.gguf"))
    if args.langs:
        files = [f for f in files
                 if any(f.stem == f"tada-ref-{l}" for l in args.langs)]

    if not files:
        print(f"No tada-ref-*.gguf found in {d}")
        return

    for path in files:
        for repo in REPOS:
            print(f"  {path.name} → {repo} … ", end="", flush=True)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo,
                repo_type="model",
                commit_message=f"Add {path.name} (FLEURS CC-BY-4.0)",
            )
            print("done")

    print(f"\nUploaded {len(files)} file(s) to {len(REPOS)} repo(s).")

if __name__ == "__main__":
    main()
