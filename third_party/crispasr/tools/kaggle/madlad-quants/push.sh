#!/bin/bash
# Push under ${KAGGLE_ACCOUNT} (CrispASR kernel convention).
# ⚠ If a run of this kernel is already live: `yes | kaggle kernels delete
# ${KAGGLE_ACCOUNT}/crispasr-madlad-quants` FIRST — a re-push STACKS a second GPU session
# and both keep burning quota (kaggle_usage gotcha #25).
# Kaggle CLI auth: ~/.kaggle/access_token (never commit a token; see tests/test_no_secrets.py)
cd "$(dirname "$0")" && cp ../kaggle_harness.py . && python3 "$(git rev-parse --show-toplevel)/tools/kaggle/kpush.py" .
