#!/usr/bin/env bash
# Push + trigger the parakeet ggml-decode A/B kernel on Kaggle (P100).
# The feat branch must be pushed to GitHub first (the kernel clones it by ref).
# Usage: bash tools/kaggle/parakeet-ggml-decode-ab/push.sh
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
echo "  kaggle kernels status ${KAGGLE_ACCOUNT}/crispasr-parakeet-ggml-decode-ab"
echo "  kaggle kernels output ${KAGGLE_ACCOUNT}/crispasr-parakeet-ggml-decode-ab -p ./out"
