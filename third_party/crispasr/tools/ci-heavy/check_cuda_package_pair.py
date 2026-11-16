#!/usr/bin/env python3
"""Verify released CLI/library archives contain the same three CUDA DLLs."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def exactly_one(root, filename):
    paths = list(root.rglob(filename))
    assert len(paths) == 1, (filename, paths)
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flavor", choices=("cuda", "cuda126"), required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--libraries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = exactly_one(args.cli, f"crispasr-windows-x86_64-{args.flavor}-runtime-sha256.txt")
    expected = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})\s+((?:cudart|cublas|cublasLt)64_\d+\.dll)", line.strip())
        assert match, line
        assert match[2] not in expected, match[2]
        expected[match[2]] = match[1]
    assert len(expected) == 3, expected
    archive = exactly_one(args.cli, f"{args.flavor}-runtime.zip")
    runtime = {}
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            name = Path(entry.filename).name
            assert name in expected and name not in runtime, entry.filename
            with package.open(entry) as stream:
                runtime[name] = digest(stream)
    assert runtime == expected, ("runtime ZIP differs from CLI manifest", runtime, expected)
    library_archive = exactly_one(args.libraries, f"libcrispasr-windows-x86_64-{args.flavor}.tar.gz")
    libraries = {}
    with tarfile.open(library_archive, "r|gz") as package:
        for entry in package:
            name = Path(entry.name).name
            if name in expected:
                assert entry.isfile() and name not in libraries, entry.name
                with package.extractfile(entry) as stream:
                    libraries[name] = digest(stream)
    assert libraries == expected, ("library runtime differs from CLI runtime", libraries, expected)
    receipt = {"status": "PASS", "flavor": args.flavor, "runtime_sha256": expected,
               "checked": ["CLI runtime manifest", "runtime ZIP", "shared-library archive"]}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
