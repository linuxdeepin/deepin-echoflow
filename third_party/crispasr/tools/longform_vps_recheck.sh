#!/bin/bash
# Re-test voxtral + cohere on all 4 lengths after the opt-out fixes
# (commits dc2295b2 cohere, 6fef8790 voxtral).
set -uo pipefail

CRISPASR=/mnt/akademie_storage/whisper.cpp/build/bin/crispasr
OUT_DIR=/mnt/akademie_storage/longform_results_recheck
CLIPS_DIR=/mnt/akademie_storage
mkdir -p "$OUT_DIR"

LENGTHS=(60 120 300 600)
CONFIGS=(
  "voxtral-mini-3b      /mnt/akademie_storage/test_cohere/voxtral-mini-3b-2507-q4_k.gguf  -l ja"
  "cohere-transcribe    /mnt/akademie_storage/test_cohere/cohere-transcribe-q4_k.gguf     -l ja"
)

MEM_FREE_MIN_MB=600
wait_for_memory() {
  while true; do
    local free_mb=$(free -m | awk '/^Mem:/ {print $4+$6}')
    [ "$free_mb" -ge "$MEM_FREE_MIN_MB" ] && return
    sleep 10
  done
}

echo "label,rc,wall_s" > "$OUT_DIR/_summary.csv"
for length in "${LENGTHS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    label=$(echo "$cfg" | awk '{print $1}')
    model=$(echo "$cfg" | awk '{print $2}')
    extra=$(echo "$cfg" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}')
    stem="${label// /_}_${length}s"
    clip="$CLIPS_DIR/yt_${length}s.wav"
    of_prefix="$OUT_DIR/${stem}"
    wait_for_memory
    echo "==> $stem"
    t0=$(date +%s)
    eval timeout 900 "$CRISPASR" -m "$model" $extra \
      --split-on-punct -ml 1 -of "$of_prefix" --output-json-full \
      -f "$clip" 2>"${of_prefix}.stderr" > "${of_prefix}.stdout"
    rc=$?
    t1=$(date +%s)
    wall=$((t1 - t0))
    echo "  rc=$rc wall=${wall}s"
    echo "${stem},${rc},${wall}" >> "$OUT_DIR/_summary.csv"
    sleep 2
  done
done
cat "$OUT_DIR/_summary.csv"
