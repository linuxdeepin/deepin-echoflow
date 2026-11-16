#!/bin/bash
# Stage the canonical sweep script + harness alongside the kernel metadata and
# push to Kaggle (${KAGGLE_ACCOUNT} -- NOT ${KAGGLE_ACCOUNT}; kernel-metadata.json's id and its
# dataset_sources are both ${KAGGLE_ACCOUNT}, and private datasets are per-account, so a
# ${KAGGLE_ACCOUNT} push fails on the dataset references). Avoids committing a duplicate of
# the 800-line script.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
STAGE="$(mktemp -d)"
cp "$ROOT/tools/kaggle-benchmark-all-backends.py" "$STAGE/"
cp "$ROOT/tools/kaggle/kaggle_harness.py" "$STAGE/"
cp "$HERE/kernel-metadata.json" "$STAGE/"
echo "Staged in $STAGE; pushing to Kaggle..."
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$STAGE"
rm -rf "$STAGE"
