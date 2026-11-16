#!/usr/bin/env bash
# Push the fusion-A/B kernel: PRE = d758fe69~1, POST = d758fe69.
# GPU on (T4 ×1 or P100), internet on (HF model download).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "kaggle kernels push -p $DIR"
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$DIR"

ID="$(python -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")"
echo
echo "Watch: https://www.kaggle.com/code/${ID}"
echo "Poll : kaggle kernels status $ID"
