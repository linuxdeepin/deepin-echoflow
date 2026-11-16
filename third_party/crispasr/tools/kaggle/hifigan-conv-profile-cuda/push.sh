#!/usr/bin/env bash
# Push + trigger the melotts HiFi-GAN conv per-op CUDA profile on Kaggle (P100).
# Clones stock `main` (nsys gives the CUDA per-kernel breakdown — no branch push
# needed). Needs Kaggle auth (${KAGGLE_ACCOUNT} ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KEY).
# Usage: bash tools/kaggle/hifigan-conv-profile-cuda/push.sh
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found (pip install kaggle)" >&2; exit 1; }
echo "Pushing $(python3 -c 'import json;print(json.load(open("kernel-metadata.json"))["id"])') ..."
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$dir"
echo; echo "Monitor:"
echo "  kaggle kernels status ${KAGGLE_ACCOUNT}/crispasr-hifigan-conv-profile-cuda"
echo "  kaggle kernels output ${KAGGLE_ACCOUNT}/crispasr-hifigan-conv-profile-cuda -p ./out"
