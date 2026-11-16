#!/usr/bin/env bash
# Download microsoft/VibeVoice-ASR from HuggingFace, convert to GGUF,
# and optionally quantize to Q4_K_M.
#
# Usage:
#   bash models/download-and-convert-vibevoice.sh [OPTIONS]
#
# Options:
#   --hf-cache DIR      HuggingFace cache dir   (default: ~/.cache/huggingface)
#   --out-dir  DIR      Output directory         (default: .)
#   --model    HF_ID    Model to download        (default: microsoft/VibeVoice-ASR)
#   --quantize          Also produce a Q4_K_M GGUF (requires built quantize binary)
#   --skip-download     Use already-cached model (skip HF download)
#   --python   PATH     Python interpreter        (default: python3)
#   -h / --help
#
# Examples:
#   # Basic: download + convert to F16 GGUF in current directory
#   bash models/download-and-convert-vibevoice.sh
#
#   # Store large files on an external SSD, also quantize
#   bash models/download-and-convert-vibevoice.sh \
#       --hf-cache /Volumes/backups/ai/hub \
#       --out-dir  /Volumes/backups/ai \
#       --quantize
#
#   # Already downloaded, just convert
#   bash models/download-and-convert-vibevoice.sh \
#       --skip-download \
#       --hf-cache /Volumes/backups/ai/hub \
#       --out-dir  /Volumes/backups/ai

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
OUT_DIR="."
MODEL_ID="microsoft/VibeVoice-ASR"
DO_QUANTIZE=0
SKIP_DOWNLOAD=0
PYTHON="${PYTHON:-python3}"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-cache)  HF_CACHE="$2";    shift 2 ;;
        --out-dir)   OUT_DIR="$2";     shift 2 ;;
        --model)     MODEL_ID="$2";    shift 2 ;;
        --python)    PYTHON="$2";      shift 2 ;;
        --quantize)  DO_QUANTIZE=1;    shift   ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

echo "=== VibeVoice-ASR: download + convert ==="
echo "  Repo:      $REPO_DIR"
echo "  HF cache:  $HF_CACHE"
echo "  Output:    $OUT_DIR"
echo "  Model:     $MODEL_ID"
echo "  Python:    $("$PYTHON" --version 2>&1)"

# ── 1. Download from HuggingFace ─────────────────────────────────────────────
MODEL_SLUG="${MODEL_ID//\//_}"
SNAP_DIR="$HF_CACHE/models--${MODEL_ID//\//-}/snapshots"

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    echo ""
    echo "── Step 1: download $MODEL_ID ──"
    HF_HOME="$HF_CACHE" "$PYTHON" - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download("$MODEL_ID", cache_dir="$HF_CACHE")
print("  snapshot:", path)
EOF
else
    echo ""
    echo "── Step 1: skipping download (--skip-download set) ──"
fi

# ── 2. Resolve snapshot directory ────────────────────────────────────────────
SNAP=$(ls -td "$SNAP_DIR"/*/ 2>/dev/null | head -1)
SNAP="${SNAP%/}"
if [[ -z "$SNAP" ]]; then
    echo "ERROR: no snapshot found under $SNAP_DIR" >&2
    exit 1
fi
echo ""
echo "── Step 2: snapshot = $SNAP"

# ── 3. Convert to F16 GGUF ───────────────────────────────────────────────────
# Derive a filename from the model ID slug and local snapshot hash
SNAP_HASH="$(basename "$SNAP")"
F16_NAME="${MODEL_ID##*/}-f16.gguf"
F16_NAME="${F16_NAME,,}"  # lowercase
F16_OUT="$OUT_DIR/$F16_NAME"

echo ""
echo "── Step 3: convert → $F16_OUT"
HF_HOME="$HF_CACHE" "$PYTHON" "$REPO_DIR/models/convert-vibevoice-to-gguf.py" \
    --input "$SNAP" \
    --output "$F16_OUT"
echo "  F16 GGUF: $(du -sh "$F16_OUT" | cut -f1)"

# ── 4. Quantize to Q4_K_M (optional) ─────────────────────────────────────────
if [[ "$DO_QUANTIZE" -eq 1 ]]; then
    Q4_NAME="${F16_NAME/f16/q4_k}"
    Q4_OUT="$OUT_DIR/$Q4_NAME"
    QUANTIZE="$REPO_DIR/build/bin/crispasr-quantize"

    echo ""
    echo "── Step 4: quantize Q4_K_M → $Q4_OUT"
    if [[ ! -x "$QUANTIZE" ]]; then
        echo "  quantize binary not found — building …"
        cmake -B "$REPO_DIR/build" -S "$REPO_DIR" \
            -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON 2>&1 | tail -3
        cmake --build "$REPO_DIR/build" --target quantize \
            -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)" 2>&1 | tail -5
    fi
    "$QUANTIZE" "$F16_OUT" "$Q4_OUT" Q4_K_M
    echo "  Q4_K: $(du -sh "$Q4_OUT" | cut -f1)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo "  F16:  $F16_OUT"
if [[ "$DO_QUANTIZE" -eq 1 ]]; then
    echo "  Q4_K: $Q4_OUT"
    MODEL_FILE="$Q4_OUT"
else
    MODEL_FILE="$F16_OUT"
fi
echo ""
echo "Test with:"
echo "  $REPO_DIR/build/bin/crispasr --model $MODEL_FILE \\"
echo "      --file /path/to/audio.wav --backend vibevoice"
