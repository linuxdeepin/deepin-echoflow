#!/bin/bash
# CrispASR RunPod — MOSS-Audio reference activation dump
#
# Spins up RTX 3090, downloads model, runs Python reference forward pass,
# SCPs the ref GGUF back to this machine.
#
# Usage:
#   ./tools/runpod-moss-refdump.sh                  # run dump
#   ./tools/runpod-moss-refdump.sh teardown          # terminate pod
#
# Cost: ~$0.22/hr RTX 3090. Typical run: ~10 min = ~$0.04.

set -euo pipefail

RPOD_API="${RUNPOD_API_KEY:-$(grep RPOD_API ~/.env 2>/dev/null | cut -d= -f2)}"
GPU_TYPE="NVIDIA GeForce RTX 3090"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PUBKEY=$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "")
POD_FILE="/tmp/crispasr-runpod-refdump-pod-id.txt"
BRANCH="${1:-feature/moss-audio}"
LOCAL_OUT="/mnt/volume1/moss-audio-4b-instruct-ref.gguf"

if [ "$BRANCH" = "teardown" ]; then
    if [ -f "$POD_FILE" ]; then
        POD_ID=$(cat "$POD_FILE")
        echo "Terminating pod $POD_ID..."
        python -c "
import runpod
runpod.api_key = '$RPOD_API'
runpod.terminate_pod('$POD_ID')
print('Terminated')
"
        rm -f "$POD_FILE"
    else
        echo "No pod file at $POD_FILE"
    fi
    exit 0
fi

[ -z "$RPOD_API" ] && { echo "ERROR: RPOD_API not set"; exit 1; }
[ -z "$PUBKEY" ] && { echo "ERROR: no SSH key"; exit 1; }

# ── Create pod ──────────────────────────────────────────────────────
echo "Creating RunPod RTX 3090..."
POD_INFO=$(python << PYEOF
import runpod, json
runpod.api_key = '$RPOD_API'
pod = runpod.create_pod(
    name="crispasr-moss-refdump",
    image_name="$IMAGE",
    gpu_type_id="$GPU_TYPE",
    gpu_count=1,
    volume_in_gb=40,
    container_disk_in_gb=20,
    min_vcpu_count=4,
    min_memory_in_gb=16,
    ports="22/tcp",
    env={"PUBLIC_KEY": "$PUBKEY"},
)
print(json.dumps({"id": pod["id"]}))
PYEOF
)
POD_ID=$(echo "$POD_INFO" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "$POD_ID" > "$POD_FILE"
echo "Pod ID: $POD_ID"

# ── Wait for SSH ────────────────────────────────────────────────────
echo "Waiting for SSH..."
SSH_IP="" SSH_PORT=""
for i in $(seq 1 60); do
    SSH_INFO=$(python << PYEOF
import runpod, json
runpod.api_key = '$RPOD_API'
pod = runpod.get_pod('$POD_ID')
runtime = pod.get('runtime', {}) or {}
for p in (runtime.get('ports', []) or []):
    if p.get('privatePort') == 22:
        print(json.dumps({"ip": p["ip"], "port": p["publicPort"]}))
        break
PYEOF
    )
    if [ -n "$SSH_INFO" ]; then
        SSH_IP=$(echo "$SSH_INFO" | python -c "import sys,json; print(json.load(sys.stdin)['ip'])")
        SSH_PORT=$(echo "$SSH_INFO" | python -c "import sys,json; print(json.load(sys.stdin)['port'])")
        echo "SSH ready: root@$SSH_IP -p $SSH_PORT"
        break
    fi
    echo "  [$i] waiting..."
    sleep 10
done
[ -z "$SSH_IP" ] && { echo "ERROR: SSH never came up"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@$SSH_IP -p $SSH_PORT"
SCP="scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -P $SSH_PORT"

# ── Install deps + clone ───────────────────────────────────────────
echo "=== Setting up ==="
$SSH << REMOTE
set -e
pip3 install -q torch transformers==4.57.1 safetensors gguf huggingface_hub hf_transfer 2>&1 | tail -3
cd /runpod-volume
git clone --depth 1 --branch $BRANCH https://github.com/CrispStrobe/CrispASR.git 2>&1 | tail -2
git clone --depth 1 https://github.com/OpenMOSS/MOSS-Audio.git 2>&1 | tail -2
echo "=== Clones done ==="
REMOTE

# ── Download model ─────────────────────────────────────────────────
echo "=== Downloading MOSS-Audio model ==="
$SSH << 'REMOTE'
set -e
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('OpenMOSS-Team/MOSS-Audio-4B-Instruct', cache_dir='/runpod-volume/cache')
print(f'Downloaded: {p}')
open('/tmp/moss-model-dir.txt','w').write(p)
"
echo "=== Model downloaded ==="
REMOTE

# ── Run reference dump ─────────────────────────────────────────────
echo "=== Running reference dump ==="
$SSH << 'REMOTE'
set -e
export MOSS_AUDIO_DIR=$(cat /tmp/moss-model-dir.txt)
export MOSS_AUDIO_GITHUB=/runpod-volume/MOSS-Audio
export MOSS_AUDIO_PROMPT="Transcribe this audio."
export MOSS_AUDIO_MAX_NEW=128

cd /runpod-volume/CrispASR
python3 tools/dump_reference.py \
  --backend moss-audio \
  --model-dir "$MOSS_AUDIO_DIR" \
  --audio samples/jfk.wav \
  --output /runpod-volume/moss-audio-ref.gguf 2>&1

echo "=== Ref dump done ==="
ls -lh /runpod-volume/moss-audio-ref.gguf
REMOTE

# ── SCP back ───────────────────────────────────────────────────────
echo "=== Downloading ref GGUF ==="
$SCP "root@$SSH_IP:/runpod-volume/moss-audio-ref.gguf" "$LOCAL_OUT"
echo "Saved: $LOCAL_OUT ($(du -h "$LOCAL_OUT" | cut -f1))"

# ── Upload to HF ──────────────────────────────────────────────────
echo "=== Uploading to HF ==="
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='$LOCAL_OUT',
    path_in_repo='moss-audio-4b-instruct-ref.gguf',
    repo_id='cstr/MOSS-Audio-4B-Instruct-GGUF',
    repo_type='model',
    commit_message='Add reference activation dump (jfk.wav, bf16 forward)',
)
print('Uploaded to HF')
"

echo ""
echo "=== DONE. Teardown with: ./tools/runpod-moss-refdump.sh teardown ==="
echo "Pod ID: $POD_ID  Cost: ~\$0.22/hr"
