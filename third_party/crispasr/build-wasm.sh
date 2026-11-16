#!/bin/bash
# CrispASR WASM Build Script — all backends for browser use.
#
# Usage:
#   ./build-wasm.sh                    # default build
#   ./build-wasm.sh --clean            # remove build-wasm/ first
#   ./build-wasm.sh --simd             # enable WASM SIMD128 (default: on)
#   ./build-wasm.sh --no-simd          # disable WASM SIMD128
#   ./build-wasm.sh --single-file      # embed WASM inside .js (larger, simpler deploy)
#   ./build-wasm.sh -- -DFOO=BAR       # extra cmake flags
#
# Prerequisites:
#   - Emscripten SDK activated (source emsdk_env.sh)
#
# Output:
#   build-wasm/bin/libwhisper.js        Emscripten JS loader
#   build-wasm/bin/libwhisper.wasm      WebAssembly binary
#   build-wasm/bin/libwhisper.worker.js Worker for pthreads
#
# Note: this build uses -pthread (multithreaded). The hosting page MUST set:
#   Cross-Origin-Opener-Policy: same-origin
#   Cross-Origin-Embedder-Policy: require-corp
# to enable SharedArrayBuffer.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="build-wasm"
CLEAN=false
SIMD=ON
SINGLE_FILE=OFF
C2PA=OFF
CMAKE_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)       CLEAN=true; shift ;;
        --simd)        SIMD=ON; shift ;;
        --no-simd)     SIMD=OFF; shift ;;
        --single-file) SINGLE_FILE=ON; shift ;;
        --single-thread) CMAKE_EXTRA+=("-DCRISPASR_WASM_SINGLE_THREAD=ON"); shift ;;
        --proxy-to-pthread) CMAKE_EXTRA+=("-DCRISPASR_WASM_PROXY_TO_PTHREAD=ON"); shift ;;
        --c2pa)        C2PA=ON; shift ;;   # opt-in: bundle C2PA signing (+~10 MB, needs wasm-EH)
        --)            shift; CMAKE_EXTRA+=("$@"); break ;;
        *)             CMAKE_EXTRA+=("$1"); shift ;;
    esac
done

# Check emcc is available
if ! command -v emcc &>/dev/null; then
    echo "[ERROR] emcc not found. Activate Emscripten SDK first:"
    echo "  source <path-to-emsdk>/emsdk_env.sh"
    exit 1
fi

echo "============================================"
echo "  CrispASR - WASM Build (all backends)"
echo "============================================"

# Check ggml submodule
if [ ! -f "$SCRIPT_DIR/ggml/CMakeLists.txt" ]; then
    echo "[INFO] Initializing ggml submodule..."
    cd "$SCRIPT_DIR" && git submodule update --init --recursive
fi

# Clean if requested
if [ "$CLEAN" = true ] && [ -d "$SCRIPT_DIR/$BUILD_DIR" ]; then
    echo "[INFO] Cleaning $BUILD_DIR..."
    rm -rf "$SCRIPT_DIR/$BUILD_DIR"
fi

# SIMD flags
SIMD_FLAGS=""
if [ "$SIMD" = "ON" ]; then
    SIMD_FLAGS="-msimd128"
    echo "[INFO] WASM SIMD128 enabled"
fi

# Configure
# C2PA in wasm is OPT-IN (--c2pa) because it adds ~10 MB (the full c2pa-rs stack)
# to the module — enough to trip the web build's size budget by default. When
# enabled: the prebuilt c2pa-rs wasm32-emscripten lib is compiled with NATIVE
# wasm exceptions (its Rust deps lopdf/rayon import the `__cpp_exception` tag),
# which is provided ONLY by `-fwasm-exceptions` (native wasm EH) — NOT
# `-fexceptions` (JS-based EH) — so the whole module must use -fwasm-exceptions
# or the link fails `undefined symbol: __cpp_exception`. We also switch longjmp
# to the wasm-EH-compatible impl (-sSUPPORT_LONGJMP=wasm) since CrispASR uses
# setjmp/longjmp. Needs a browser with the wasm-EH proposal (all modern, 2023+).
# Without --c2pa the module is unchanged and Module.c2paSign() returns empty.
C2PA_FETCH_FLAG=""
C2PA_WASM_FLAGS=""
C2PA_WASM_LINK_FLAGS=""
if [ "$C2PA" = "ON" ]; then
    C2PA_FETCH_FLAG="-DCRISPASR_C2PA_FETCH=ON"
    C2PA_WASM_FLAGS="-fwasm-exceptions"
    C2PA_WASM_LINK_FLAGS="-fwasm-exceptions -sSUPPORT_LONGJMP=wasm"
    echo "[INFO] C2PA signing enabled in wasm (+~10 MB, needs wasm-EH browser)"
fi
echo "[INFO] Configuring with emcmake..."
cd "$SCRIPT_DIR"
# Use ninja if available (faster parallel builds) + ccache
GENERATOR=""
if command -v ninja &>/dev/null; then
    GENERATOR="-G Ninja"
    echo "[INFO] Using Ninja generator"
fi
export CCACHE_DIR="${CCACHE_DIR:-${HOME}/.ccache}"

emcmake cmake -S . -B "$BUILD_DIR" $GENERATOR \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_OPENMP=OFF \
    -DCRISPASR_BUILD_TESTS=OFF \
    -DCRISPASR_BUILD_EXAMPLES=OFF \
    -DCRISPASR_BUILD_SERVER=OFF \
    -DCRISPASR_SDL2=OFF \
    -DCRISPASR_CURL=OFF \
    -DCRISPASR_OPUS_FETCH=ON \
    -DOPUS_DISABLE_INTRINSICS=ON \
    $C2PA_FETCH_FLAG \
    -DCRISPASR_WASM_SINGLE_FILE="$SINGLE_FILE" \
    -DCRISPASR_WASM=ON \
    -DCMAKE_C_FLAGS="$SIMD_FLAGS $C2PA_WASM_FLAGS" \
    -DCMAKE_CXX_FLAGS="$SIMD_FLAGS $C2PA_WASM_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$C2PA_WASM_LINK_FLAGS" \
    "${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"}"

# Build
echo "[INFO] Building..."
cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || echo 4) --target libwhisper

echo ""
echo "[SUCCESS] WASM build complete!"
ls -lh "$BUILD_DIR/bin/libwhisper.js" "$BUILD_DIR/bin/libwhisper.wasm" "$BUILD_DIR/bin/libwhisper.worker.js" 2>/dev/null || \
ls -lh "$BUILD_DIR/bin/libwhisper.js" 2>/dev/null
echo ""
echo "Files to deploy:"
echo "  $BUILD_DIR/bin/libwhisper.js"
echo "  $BUILD_DIR/bin/libwhisper.wasm"
echo "  $BUILD_DIR/bin/libwhisper.worker.js"
echo ""
echo "Note: hosting page MUST set COOP/COEP headers for SharedArrayBuffer."
