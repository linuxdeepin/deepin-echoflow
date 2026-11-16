#!/usr/bin/env bash
# Push + trigger the parakeet memory-policy CUDA proof kernel (improvements Phase 2).
# Push the branch (or merge to main) to GitHub FIRST — the kernel clones by ref
# (CRISPASR_REF, default main). Usage: bash tools/kaggle/parakeet-mem-policy-cuda/push.sh
set -euo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
command -v kaggle >/dev/null || { echo "kaggle CLI not found (pip install kaggle)" >&2; exit 1; }
echo "Pushing $(jq -r .id kernel-metadata.json) ..."
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$dir"
echo "Monitor:  kaggle kernels status ${KAGGLE_ACCOUNT}/crispasr-parakeet-mem-policy-cuda"
echo "Output:   kaggle kernels output ${KAGGLE_ACCOUNT}/crispasr-parakeet-mem-policy-cuda -p ./out"
