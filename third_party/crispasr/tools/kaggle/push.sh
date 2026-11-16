#!/usr/bin/env bash
# Push the CrispASR regression notebook to Kaggle.
#
# Kaggle's `kernels push` API uploads the contents of the given
# directory AND triggers an immediate run with the freshly-uploaded
# code. There is no "upload without running" flag — running is the
# implicit "this is now the current version" signal.
#
# After the first successful push, set up Kaggle's native weekly
# schedule via the kernel's "Settings → Schedule a notebook run"
# UI; the CLI doesn't expose scheduling. The script itself
# `git clone`s CrispASR at the start of every run, so as long as
# the local `tools/kaggle/crispasr-regression.py` we pushed is the
# bootstrap, Kaggle picks up `main` automatically each cycle.
#
# Re-push when the bootstrap script itself changes (env knob
# defaults, new pip deps). Pure C++ changes to CrispASR don't
# need a re-push — they get pulled at runtime.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "kaggle kernels push -p $DIR"
python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" "$DIR"

echo
echo "Push triggered. Watch live at:"
ID="$(python -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")"
echo "  https://www.kaggle.com/code/${ID}"
echo
echo "Poll status via CLI:"
echo "  kaggle kernels status $ID"
echo
echo "Once the first run lands cleanly, enable the weekly schedule:"
echo "  → open the URL above"
echo "  → Settings → 'Schedule a notebook run' → Weekly · Sun 04:00 UTC"
