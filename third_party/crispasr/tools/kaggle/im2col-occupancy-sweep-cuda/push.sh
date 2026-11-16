#!/usr/bin/env bash
# Push the CUDA im2col occupancy wider-sweep kernel (P100). Needs ${KAGGLE_ACCOUNT} Kaggle auth.
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found" >&2; exit 1; }
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$dir"
echo "Monitor: kaggle kernels status ${KAGGLE_ACCOUNT}/crispasr-im2col-occupancy-sweep-cuda"
