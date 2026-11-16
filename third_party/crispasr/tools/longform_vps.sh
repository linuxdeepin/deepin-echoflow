#!/bin/bash
# Multi-backend × multi-length live test of long-audio handling on
# lenhone's fresh yt-dlp Japanese audio. Runs on VPS to dodge local
# memory pressure. Sequential, with memory-free check between iterations.
set -uo pipefail

CRISPASR=/mnt/akademie_storage/whisper.cpp/build/bin/crispasr
GGUF_DIR=/mnt/akademie_storage
OUT_DIR=/mnt/akademie_storage/longform_results
CLIPS_DIR=/mnt/akademie_storage
mkdir -p "$OUT_DIR"

# Order: light → heavy by RAM footprint
# Skip mimo-asr (4.5 GB Q4_K — too big for 7.6 GB total RAM)
LENGTHS=(60 120 300 600)
CONFIGS=(
  "parakeet-streamed                 parakeet-tdt-0.6b-ja.gguf                                          -l ja"
  "parakeet-vad-silero               parakeet-tdt-0.6b-ja.gguf                                          -l ja --vad"
  "parakeet-singlepass               parakeet-tdt-0.6b-ja.gguf                                          -l ja"
  "parakeet-ctc-head                 parakeet-tdt-0.6b-ja.gguf                                          -l ja --parakeet-decoder ctc"
)

# Heavier — run only on shorter clips first
HEAVY=(
  "voxtral-mini-3b                   /mnt/akademie_storage/test_cohere/voxtral-mini-3b-2507-q4_k.gguf   -l ja"
  "cohere-transcribe                 /mnt/akademie_storage/test_cohere/cohere-transcribe-q4_k.gguf      -l ja"
  "cohere-vad                        /mnt/akademie_storage/test_cohere/cohere-transcribe-q4_k.gguf      -l ja --vad"
  "canary-1b-v2                      /mnt/akademie_storage/test_cohere/canary-1b-v2-q4_k.gguf           -l ja"
)

MEM_FREE_MIN_MB=600    # require at least 600 MB free before launching a model
wait_for_memory() {
  while true; do
    local free_mb=$(free -m | awk '/^Mem:/ {print $4+$6}')   # free + buff/cache
    if [ "$free_mb" -ge "$MEM_FREE_MIN_MB" ]; then return; fi
    echo "  wait: free+cache=${free_mb}MB < ${MEM_FREE_MIN_MB}MB, sleeping 10s"
    sleep 10
  done
}

run_one() {
  local label=$1 model=$2 length=$3 extra=$4
  local stem="${label// /_}_${length}s"
  local clip="$CLIPS_DIR/yt_${length}s.wav"
  local model_path
  if [[ "$model" == /* ]]; then
    model_path="$model"
  else
    model_path="$GGUF_DIR/$model"
  fi
  local of_prefix="$OUT_DIR/${stem}"

  [ -f "$clip" ] || { echo "MISSING_CLIP $clip"; echo "${stem},missing_clip,0" >> "$OUT_DIR/_summary.csv"; return; }
  [ -f "$model_path" ] || { echo "MISSING_MODEL $model_path"; echo "${stem},missing_model,0" >> "$OUT_DIR/_summary.csv"; return; }

  local env_prefix=""
  if [[ "$label" == *singlepass* ]]; then
    env_prefix="CRISPASR_PARAKEET_STREAM_THRESHOLD=999"
  fi

  wait_for_memory

  echo "==> $stem"
  local t0=$(date +%s)
  eval $env_prefix timeout 900 "$CRISPASR" -m "$model_path" $extra \
    --split-on-punct -ml 1 -of "$of_prefix" --output-json-full \
    -f "$clip" 2>"${of_prefix}.stderr" > "${of_prefix}.stdout"
  local rc=$?
  local t1=$(date +%s)
  local wall=$((t1 - t0))
  echo "  rc=$rc wall=${wall}s"
  echo "${stem},${rc},${wall}" >> "$OUT_DIR/_summary.csv"

  # let the kernel reclaim before next iter
  sleep 2
}

echo "label,rc,wall_s" > "$OUT_DIR/_summary.csv"

echo "=== light backends (parakeet family) ==="
for length in "${LENGTHS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    label=$(echo "$cfg" | awk '{print $1}')
    model=$(echo "$cfg" | awk '{print $2}')
    extra=$(echo "$cfg" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}')
    run_one "$label" "$model" "$length" "$extra"
  done
done

echo
echo "=== heavy backends — short clips first then long ==="
for length in 60 120 300 600; do
  for cfg in "${HEAVY[@]}"; do
    label=$(echo "$cfg" | awk '{print $1}')
    model=$(echo "$cfg" | awk '{print $2}')
    extra=$(echo "$cfg" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}')
    run_one "$label" "$model" "$length" "$extra"
  done
done

echo
echo "=== summary ==="
cat "$OUT_DIR/_summary.csv"
