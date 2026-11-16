#!/usr/bin/env bash
# Shared flags for release packaging and the #484 Intel A/B proof.
set -euo pipefail
BUILD="${1:?usage: configure-macos-intel.sh <build-dir> <avx2|legacy> [cmake options]}"
ISA="${2:?ISA required}"
shift 2
case "$ISA" in
    avx2) PORTABLE=OFF; SIMD=ON ;;
    legacy) PORTABLE=ON; SIMD=OFF ;;
    *) echo "Unknown Intel ISA: $ISA" >&2; exit 1 ;;
esac
cmake -G Ninja -S . -B "$BUILD" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple \
    -DCRISPASR_PORTABLE_CPU="$PORTABLE" -DGGML_NATIVE=OFF \
    -DGGML_SSE42="$SIMD" -DGGML_AVX="$SIMD" -DGGML_AVX2="$SIMD" \
    -DGGML_FMA="$SIMD" -DGGML_F16C="$SIMD" -DGGML_BMI2=OFF \
    -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF \
    -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF \
    -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF \
    -DCRISPASR_BUILD_TESTS=OFF -DCRISPASR_BUILD_SERVER=OFF \
    -DCRISPASR_OPUS_FETCH=ON "$@"
