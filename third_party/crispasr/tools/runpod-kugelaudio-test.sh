#!/bin/bash
# CrispASR RunPod GPU test — KugelAudio TTS (Qwen2.5-7B + diffusion)
#
# Spins up an RTX 3090 (24 GB VRAM), builds CUDA, downloads the Q4_K
# GGUF (~5.3 GB), runs TTS synthesis, and validates via whisper ASR
# roundtrip.
#
# Prerequisites:
#   pip install runpod
#   RUNPOD_API_KEY set (from ~/.env RPOD_API)
#   SSH key at ~/.ssh/id_ed25519
#
# Usage:
#   ./tools/runpod-kugelaudio-test.sh                    # create pod, build, test
#   ./tools/runpod-kugelaudio-test.sh feature/kugelaudio-tts  # test a branch
#   ./tools/runpod-kugelaudio-test.sh teardown           # terminate pod
#
# Cost: ~$0.22/hr (RTX 3090 community). Typical run: 20-30 min = ~$0.10.
# Always terminate when done!

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
RPOD_API="${RUNPOD_API_KEY:-$(grep RPOD_API ~/.env 2>/dev/null | cut -d= -f2)}"
GPU_TYPE="NVIDIA GeForce RTX 3090"
GPU_ARCH=86  # sm_86 for RTX 3090
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
PUBKEY=$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "")
POD_FILE="/tmp/crispasr-kugelaudio-pod-id.txt"
BRANCH="${1:-feature/kugelaudio-tts}"

if [ "$BRANCH" = "teardown" ]; then
    if [ -f "$POD_FILE" ]; then
        POD_ID=$(cat "$POD_FILE")
        echo "Terminating pod $POD_ID..."
        python3 -c "
import runpod
runpod.api_key = '$RPOD_API'
runpod.terminate_pod('$POD_ID')
print('Terminated')
"
        rm -f "$POD_FILE"
    else
        echo "No pod ID file found at $POD_FILE"
    fi
    exit 0
fi

[ -z "$RPOD_API" ] && { echo "ERROR: RUNPOD_API_KEY not set"; exit 1; }
[ -z "$PUBKEY" ] && { echo "ERROR: no SSH key at ~/.ssh/id_ed25519.pub"; exit 1; }

# ── Create pod ──────────────────────────────────────────────────────
echo "Creating RunPod $GPU_TYPE pod for KugelAudio..."
POD_INFO=$(python3 << PYEOF
import runpod, json
runpod.api_key = '$RPOD_API'
pod = runpod.create_pod(
    name="crispasr-kugelaudio-test",
    image_name="$IMAGE",
    gpu_type_id="$GPU_TYPE",
    gpu_count=1,
    volume_in_gb=40,
    container_disk_in_gb=20,
    min_vcpu_count=4,
    min_memory_in_gb=24,
    ports="22/tcp",
    env={"PUBLIC_KEY": "$PUBKEY"},
)
print(json.dumps({"id": pod["id"]}))
PYEOF
)
POD_ID=$(echo "$POD_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "$POD_ID" > "$POD_FILE"
echo "Pod ID: $POD_ID (saved to $POD_FILE)"

# ── Wait for SSH ────────────────────────────────────────────────────
echo "Waiting for SSH..."
for i in $(seq 1 60); do
    SSH_INFO=$(python3 << PYEOF
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
        SSH_IP=$(echo "$SSH_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['ip'])")
        SSH_PORT=$(echo "$SSH_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['port'])")
        echo "SSH ready: root@$SSH_IP -p $SSH_PORT"
        break
    fi
    echo "  [$i] waiting..."
    sleep 10
done

[ -z "${SSH_IP:-}" ] && { echo "ERROR: SSH never came up"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 root@$SSH_IP -p $SSH_PORT"

# ── Clone + Build ──────────────────────────────────────────────────
echo "=== Clone + CUDA Build ==="
$SSH << REMOTE
set -e
export PATH=/usr/local/cuda/bin:\$PATH
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq ninja-build >/dev/null 2>&1
pip3 install cmake huggingface_hub hf_transfer -q 2>/dev/null

export CMAKE=\$(find /usr -path '*/cmake/data/bin/cmake' 2>/dev/null | head -1)
[ -z "\$CMAKE" ] && CMAKE=cmake

cd /runpod-volume
rm -rf CrispASR build
git clone --depth 1 --branch $BRANCH https://github.com/CrispStrobe/CrispASR.git
cd CrispASR && git submodule update --init --depth 1 ggml
echo "SHA: \$(git rev-parse --short HEAD)"

mkdir -p /runpod-volume/build
\$CMAKE -G Ninja -S /runpod-volume/CrispASR -B /runpod-volume/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCRISPASR_BUILD_TESTS=OFF \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH
ninja -C /runpod-volume/build -j\$(nproc) crispasr-cli
echo "=== Build OK ==="
ls -la /runpod-volume/build/bin/crispasr
REMOTE

# ── Download GGUF ─────────────────────────────────────────────────
echo "=== Downloading KugelAudio Q4_K GGUF ==="
$SSH << 'REMOTE'
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('cstr/kugelaudio-0-open-GGUF', 'kugelaudio-0-open-q4_k.gguf',
                       cache_dir='/runpod-volume/cache')
print(f'GGUF: {path}')
# Symlink for easy access
import os
os.symlink(path, '/runpod-volume/kugelaudio-q4k.gguf') if not os.path.exists('/runpod-volume/kugelaudio-q4k.gguf') else None
"
ls -lh /runpod-volume/kugelaudio-q4k.gguf
REMOTE


# ── Download whisper model for ASR roundtrip ─────────────────
echo "=== Downloading whisper base.en model ==="
$SSH << 'REMOTE'
set -e
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('ggerganov/whisper.cpp', 'ggml-base.en.bin',
                       cache_dir='/runpod-volume/cache')
shutil.copy2(path, '/runpod-volume/cache/ggml-base.en.bin')
print(f'whisper model: /runpod-volume/cache/ggml-base.en.bin')
"
ls -lh /runpod-volume/cache/ggml-base.en.bin
REMOTE

# ── Test: TTS synthesis ──────────────────────────────────────────
echo "=== Test: TTS synthesis ==="
$SSH << 'REMOTE'
set -e
export LD_LIBRARY_PATH=/runpod-volume/build/src:/runpod-volume/build/ggml/src:/runpod-volume/build/ggml/src/ggml-cuda
CLI=/runpod-volume/build/bin/crispasr

echo "--- Backend list ---"
$CLI --list-backends 2>&1 | grep kugelaudio

echo "--- TTS: English ---"
$CLI --backend kugelaudio -m /runpod-volume/kugelaudio-q4k.gguf \
  --tts "Hello, this is a test of the speech synthesis system." \
  --tts-output /tmp/kugelaudio-en.wav 2>&1 | tail -10
ls -lh /tmp/kugelaudio-en.wav 2>/dev/null || echo "FAILED: no English audio"

echo "--- TTS: German ---"
$CLI --backend kugelaudio -m /runpod-volume/kugelaudio-q4k.gguf \
  --tts "Hallo, dies ist ein Test." -l de \
  --tts-output /tmp/kugelaudio-de.wav 2>&1 | tail -10
ls -lh /tmp/kugelaudio-de.wav 2>/dev/null || echo "FAILED: no German audio"
REMOTE

# ── Test: ASR roundtrip ──────────────────────────────────────────
echo "=== Test: ASR roundtrip ==="
$SSH << 'REMOTE'
set -e
export LD_LIBRARY_PATH=/runpod-volume/build/src:/runpod-volume/build/ggml/src:/runpod-volume/build/ggml/src/ggml-cuda
CLI=/runpod-volume/build/bin/crispasr
WHISPER=/runpod-volume/cache/ggml-base.en.bin

echo "=== English roundtrip ==="
if [ -f /tmp/kugelaudio-en.wav ]; then
    echo "Input:  Hello, this is a test of the speech synthesis system."
    echo -n "Output: "
    $CLI -m $WHISPER -f /tmp/kugelaudio-en.wav --no-timestamps -otxt -of /tmp/rt-en 2>/dev/null
    cat /tmp/rt-en.txt 2>/dev/null || echo "(no transcript)"
else
    echo "SKIPPED: no English audio"
fi

echo ""
echo "=== German roundtrip ==="
if [ -f /tmp/kugelaudio-de.wav ]; then
    echo "Input:  Hallo, dies ist ein Test."
    echo -n "Output: "
    $CLI -m $WHISPER -f /tmp/kugelaudio-de.wav --no-timestamps -l de -otxt -of /tmp/rt-de 2>/dev/null
    cat /tmp/rt-de.txt 2>/dev/null || echo "(no transcript)"
else
    echo "SKIPPED: no German audio"
fi
REMOTE

echo ""
echo "=== Done. Run './tools/runpod-kugelaudio-test.sh teardown' to terminate ==="
echo "Pod ID: $POD_ID  Cost: ~\$0.22/hr"
