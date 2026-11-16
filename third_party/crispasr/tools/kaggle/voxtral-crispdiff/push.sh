#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$DIR"
echo "Watch: https://www.kaggle.com/code/${KAGGLE_ACCOUNT}/crispasr-voxtral-tts-crispdiff"
