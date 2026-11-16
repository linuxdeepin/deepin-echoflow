#!/usr/bin/env bash
# Stamp a directory of published GGUFs with their speaker_identity verdict.
#
# Why this exists: the runtime prefers a stamp inside the file over its built-in
# table of file-name patterns, because a stamped file answers for itself and
# survives being renamed, re-quantised or moved. Everything CrispASR has already
# published predates the stamp, so the table is still load-bearing for those —
# this closes that gap for a whole directory in one pass.
#
# The verdicts are NOT restated here. Each file is handed to
# `crispasr --print-speaker-identity`, which runs the same resolution the
# disclosure gate runs. A shell copy of the table would be a third place for it
# to drift, and a drift fails open: the stamp is simply never found.
#
#   ./models/stamp-published-voices.sh <dir> [outdir]
#
# Files that resolve to `unknown` are SKIPPED, not stamped: absence of the key
# is how "not established" is encoded, and writing a guess would be the one
# error that silently removes a disclosure.
#
# Writes to <outdir> (default: <dir>/stamped) and never modifies the inputs.
# Re-uploading the results is a separate, deliberate step — this script does not
# touch the network.
set -euo pipefail

SRC="${1:?usage: stamp-published-voices.sh <dir> [outdir]}"
OUT="${2:-$SRC/stamped}"
CRISPASR="${CRISPASR_BIN:-build/bin/crispasr}"
STAMPER="$(dirname "$0")/stamp-speaker-identity.py"

[ -x "$CRISPASR" ] || { echo "crispasr binary not found at $CRISPASR (set CRISPASR_BIN)" >&2; exit 1; }
[ -f "$STAMPER" ]  || { echo "stamper not found at $STAMPER" >&2; exit 1; }
mkdir -p "$OUT"

stamped=0 skipped=0 failed=0
for f in "$SRC"/*.gguf; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    # Exit 3 == unknown. Ask the binary; never decide here.
    if identity="$("$CRISPASR" --print-speaker-identity "$f" --no-prints 2>/dev/null)"; then
        :
    else
        rc=$?
        if [ "$rc" -eq 3 ]; then
            printf '%-46s %s\n' "$base" "SKIP (unknown — read the model card, then use stamp-speaker-identity.py)"
            skipped=$((skipped + 1))
            continue
        fi
        printf '%-46s %s\n' "$base" "FAILED (rc=$rc)"
        failed=$((failed + 1))
        continue
    fi
    identity="$(printf '%s' "$identity" | tr -d '[:space:]')"
    if [ -z "$identity" ] || [ "$identity" = "unknown" ]; then
        printf '%-46s %s\n' "$base" "SKIP (unknown)"
        skipped=$((skipped + 1))
        continue
    fi
    if python "$STAMPER" --input "$f" --output "$OUT/$base" \
        --speaker-identity "$identity" \
        --evidence "resolved by crispasr --print-speaker-identity; see examples/cli/crispasr_speaker_identity_models.h" \
        --force >/dev/null 2>&1; then
        printf '%-46s %s\n' "$base" "stamped $identity"
        stamped=$((stamped + 1))
    else
        printf '%-46s %s\n' "$base" "FAILED to stamp"
        failed=$((failed + 1))
    fi
done

echo
echo "stamped=$stamped skipped=$skipped failed=$failed -> $OUT"
[ "$failed" -eq 0 ] || exit 1
