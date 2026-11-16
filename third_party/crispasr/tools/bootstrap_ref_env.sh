#!/usr/bin/env bash
# bootstrap_ref_env.sh — create a python venv that can run a specific
# `tools/dump_reference.py --backend <name>` invocation.
#
# Usage:
#   tools/bootstrap_ref_env.sh <backend>           # create + install
#   tools/bootstrap_ref_env.sh <backend> --upgrade # update deps in place
#
# Reads tools/reference_envs/<backend>/requirements.txt for the dep
# scaffold. Creates the venv at:
#
#   ${CRISPASR_REF_ENVS_ROOT:-/Volumes/backups/ai/refenvs}/<backend>/
#
# After bootstrap, activate + dump:
#
#   source <env>/bin/activate
#   python tools/dump_reference.py --backend <backend> \
#       --model-dir <model-snapshot> --audio samples/jfk.wav \
#       --output /Volumes/backups/ai/<backend>-ref.gguf
#
# Use tools/audit_diff_coverage.py to see what's covered.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <backend> [--upgrade]" >&2
    exit 2
fi

BACKEND="$1"
shift
UPGRADE=0
for arg in "$@"; do
    case "$arg" in
        --upgrade) UPGRADE=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$REPO_ROOT/tools/reference_envs/$BACKEND/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
    echo "error: no requirements scaffold at $REQ_FILE" >&2
    echo "  run: python tools/audit_diff_coverage.py --emit-env-scaffolds" >&2
    exit 1
fi

ENVS_ROOT="${CRISPASR_REF_ENVS_ROOT:-/Volumes/backups/ai/refenvs}"
ENV_DIR="$ENVS_ROOT/$BACKEND"

if [[ -d "$ENV_DIR" && "$UPGRADE" -eq 0 ]]; then
    echo "venv already exists at $ENV_DIR (pass --upgrade to refresh deps)"
    echo "activate with: source $ENV_DIR/bin/activate"
    exit 0
fi

if [[ ! -d "$ENV_DIR" ]]; then
    PYTHON="${PYTHON:-python3}"
    echo "creating venv at $ENV_DIR using $PYTHON"
    mkdir -p "$ENVS_ROOT"
    "$PYTHON" -m venv "$ENV_DIR"
fi

# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null
echo "installing deps from $REQ_FILE"
python -m pip install -r "$REQ_FILE"
deactivate

# Check for TODO markers and remind the user
if grep -q '^# TODO' "$REQ_FILE"; then
    echo
    echo "NOTE: $REQ_FILE has unresolved TODO lines for git-only deps."
    echo "Add the correct git+https URL (with pinned sha) and re-run with --upgrade."
fi

echo
echo "done. activate with:"
echo "  source $ENV_DIR/bin/activate"
echo "dump a reference archive with:"
echo "  python tools/dump_reference.py --backend $BACKEND --model-dir <dir> \\"
echo "      --audio samples/jfk.wav --output /Volumes/backups/ai/$BACKEND-ref.gguf"
