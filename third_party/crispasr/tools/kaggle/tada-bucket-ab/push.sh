#!/usr/bin/env bash
# Push + trigger the tada §176b bucket-floor timing A/B kernel on Kaggle (CUDA).
# The feat branch must be pushed to GitHub first (the kernel clones it by ref).
# Usage: bash tools/kaggle/tada-bucket-ab/push.sh
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$dir"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI not found. Install with: pip install kaggle" >&2
    exit 1
fi

echo "Pushing $(jq -r .id kernel-metadata.json) ..."
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$dir"
echo
echo "Triggered. Monitor with:"
echo "  kaggle kernels status ${KAGGLE_ACCOUNT}/crispasr-tada-bucket-ab"
echo "  kaggle kernels output ${KAGGLE_ACCOUNT}/crispasr-tada-bucket-ab -p ./out"
