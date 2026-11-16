#!/usr/bin/env bash
# Local-side companion to the Kaggle rebake kernel.
#
# The Kaggle kernel (${KAGGLE_ACCOUNT}/crispasr-auto-rebake-refs) generates
# fresh reference dumps and writes them to
# `/kaggle/working/rebake-stage/`. Because Kaggle Secrets has been
# flaky (HTTPError 400 from GetUserSecretByLabel even after multiple
# Attach toggles), we do the publish step from local where HF_TOKEN
# from `.env` is already write-tested against
# `cstr/crispasr-regression-fixtures`.
#
# Usage:
#
#   ./tools/kaggle/rebake/fetch-and-upload.sh
#
# Runs after the Kaggle kernel completes successfully. Steps:
#
#   1. `kaggle kernels output` → pull staged refs to
#      /Volumes/backups/ai/crispasr-regression/rebake-out/.
#   2. Stage `rebake-stage/` for upload.
#   3. `hf upload` to cstr/crispasr-regression-fixtures using
#      HF_TOKEN from .env.
#   4. Print the new fixtures commit SHA to paste into
#      `tests/regression/manifest.json`'s `fixtures.revision`.

set -euo pipefail

KAGGLE_KERNEL="${KAGGLE_ACCOUNT}/crispasr-auto-rebake-refs"
LOCAL_OUT="/Volumes/backups/ai/crispasr-regression/rebake-out"
FIXTURES_REPO="cstr/crispasr-regression-fixtures"
ENV_FILE="/Users/christianstrobele/code/.env"

# ── Load HF_TOKEN from .env without printing it ───────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found" >&2
    exit 1
fi
# `grep -E` + parameter expansion, stays out of stdout
HF_TOKEN_LINE="$(grep -E '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN[[:space:]]*=' "$ENV_FILE" || true)"
if [ -z "$HF_TOKEN_LINE" ]; then
    echo "ERROR: no HF_TOKEN line in $ENV_FILE" >&2
    exit 1
fi
HF_TOKEN_VALUE="${HF_TOKEN_LINE#*=}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\"}"; HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\"}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE%\'}"; HF_TOKEN_VALUE="${HF_TOKEN_VALUE#\'}"
export HF_TOKEN="$HF_TOKEN_VALUE"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VALUE"

# ── Pull Kaggle output ────────────────────────────────────────────────
rm -rf "$LOCAL_OUT"
mkdir -p "$LOCAL_OUT"
echo "fetching $KAGGLE_KERNEL output → $LOCAL_OUT"
kaggle kernels output "$KAGGLE_KERNEL" -p "$LOCAL_OUT"

# Kaggle preserves the whole /kaggle/working/ tree; we want rebake-stage/.
STAGE="$LOCAL_OUT/rebake-stage"
if [ ! -d "$STAGE" ]; then
    # Some Kaggle output paths nest differently; find it.
    STAGE="$(find "$LOCAL_OUT" -type d -name rebake-stage | head -1)"
    if [ -z "$STAGE" ] || [ ! -d "$STAGE" ]; then
        echo "ERROR: rebake-stage/ not found under $LOCAL_OUT" >&2
        echo "Kaggle output contents:" >&2
        find "$LOCAL_OUT" -maxdepth 3 -type d >&2
        exit 1
    fi
fi
echo "found rebake-stage: $STAGE"
echo "contents:"
find "$STAGE" -type f | sort

# ── Upload to fixtures HF repo ────────────────────────────────────────
echo
echo "uploading to $FIXTURES_REPO …"
cd "$STAGE"
hf upload "$FIXTURES_REPO" . . \
    --commit-message "rebake from ${KAGGLE_ACCOUNT}/crispasr-auto-rebake-refs $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    2>&1 | tail -10

# ── Report new fixtures commit SHA ────────────────────────────────────
echo
echo "new fixtures commit SHA:"
NEW_SHA="$(curl -s "https://huggingface.co/api/models/$FIXTURES_REPO" \
    | python -c "import json,sys; print(json.load(sys.stdin)['sha'])")"
echo "  $NEW_SHA"
echo
echo "to adopt this rebake in CI, bump tests/regression/manifest.json:"
echo "  fixtures.revision = \"$NEW_SHA\""
echo "then commit + push. The next nightly GH workflow run will pick it up."
