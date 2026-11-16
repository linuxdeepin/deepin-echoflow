#!/usr/bin/env python3
"""Run the staged CUDA archive checks using the existing Heavy CPU workflow."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli-run", type=int, required=True)
    parser.add_argument("--library-run", type=int, required=True)
    args = parser.parse_args()
    assert args.cli_run > 0 and args.library_run > 0
    subprocess.run(["uptime"], check=True)
    subprocess.run(["free", "-h"], check=True)
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "cuda-package-pair"
    output = Path(os.environ["HEAVY_OUT"])
    output.mkdir(parents=True, exist_ok=True)
    checker = Path(__file__).with_name("check_cuda_package_pair.py")
    repository = os.environ.get("GITHUB_REPOSITORY", "CrispStrobe/CrispASR")
    for flavor in ("cuda", "cuda126"):
        cli = scratch / flavor / "cli"
        libraries = scratch / flavor / "libraries"
        for run, artifact, destination in (
            (args.cli_run, f"crispasr-windows-x86_64-{flavor}-split", cli),
            (args.library_run, f"libcrispasr-windows-x86_64-{flavor}", libraries),
        ):
            subprocess.run(["gh", "run", "download", str(run), "--repo", repository,
                            "--name", artifact, "--dir", str(destination)], check=True)
        subprocess.run([sys.executable, str(checker), "--flavor", flavor,
                        "--cli", str(cli), "--libraries", str(libraries),
                        "--output", str(output / f"package-pair-{flavor}.json")], check=True)

    # Reproduce #483's mixed-version installation using the actual DLL bytes:
    # match the CUDA 12.8 CLI manifest against the CUDA 12.6 library runtime.
    wrong = scratch / "wrong-runtime"
    wrong.mkdir(parents=True, exist_ok=True)
    legacy = list((scratch / "cuda126" / "libraries").rglob("libcrispasr-windows-x86_64-cuda126.tar.gz"))
    assert len(legacy) == 1, legacy
    (wrong / "libcrispasr-windows-x86_64-cuda.tar.gz").symlink_to(legacy[0])
    negative = subprocess.run(
        [sys.executable, str(checker), "--flavor", "cuda", "--cli", str(scratch / "cuda" / "cli"),
         "--libraries", str(wrong), "--output", str(output / "must-not-pass.json")],
        capture_output=True, text=True)
    (output / "mixed-runtime-negative.log").write_text(negative.stdout + negative.stderr)
    assert negative.returncode != 0 and "library runtime differs from CLI runtime" in negative.stderr
    (output / "mixed-runtime-negative.json").write_text(json.dumps({
        "status": "correctly rejected", "cli_toolkit": "12.8", "library_runtime": "12.6",
        "exit_status": negative.returncode,
    }, indent=2) + "\n")
    (output / "summary.md").write_text(
        "CUDA 12.8 and CUDA 12.6 CLI/runtime/library SHA-256 pairing PASS.\n"
        "Mixed 12.8 CLI / 12.6 runtime negative control correctly rejected.\n"
        f"CLI run: {args.cli_run}; library run: {args.library_run}.\n")


if __name__ == "__main__":
    main()
