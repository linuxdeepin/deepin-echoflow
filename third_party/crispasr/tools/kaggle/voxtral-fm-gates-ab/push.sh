#!/usr/bin/env bash
# Push + run the Voxtral-TTS FM perf-gate A/B kernel on Kaggle GPU.
# `kaggle kernels push` uploads this dir and triggers an immediate run.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$DIR/../kaggle_harness.py" "$DIR/kaggle_harness.py" 2>/dev/null || true
echo "kaggle kernels push -p $DIR"
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$DIR"
ID="$(python -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")"
echo
echo "Watch:   https://www.kaggle.com/code/${ID}"
echo "Status:  kaggle kernels status ${ID}"
echo "Output:  kaggle kernels output ${ID} -p ${DIR}/out"
