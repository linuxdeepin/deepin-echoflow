#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jupytext --to ipynb "$DIR/crispasr-vibevoice-asr-bench.py" \
    --output "$DIR/crispasr-vibevoice-asr-bench.ipynb"
python - <<'PY' "$DIR/crispasr-vibevoice-asr-bench.ipynb"
import json, sys
nb = json.load(open(sys.argv[1]))
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
PY
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$DIR"
ID="$(python -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")"
echo "Watch: https://www.kaggle.com/code/${ID}"
