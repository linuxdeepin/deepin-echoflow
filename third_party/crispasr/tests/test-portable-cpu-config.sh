#!/usr/bin/env bash
# Configure-only regression guard for the opt-in generic x86-64 CPU baseline
# used by legacy/portable artifacts (#261). No compiler or model execution is
# needed: the CMake cache is the contract that controls ggml-cpu's emitted ISA.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${TMPDIR:-/tmp}/crispasr-portable-cpu-config-$$"
trap 'rm -rf "${build_dir}"' EXIT

cmake -S "${repo_root}" -B "${build_dir}" \
    -DCRISPASR_PORTABLE_CPU=ON \
    -DCRISPASR_BUILD_TESTS=OFF \
    -DCRISPASR_BUILD_EXAMPLES=OFF \
    -DCRISPASR_BUILD_SERVER=OFF \
    -DCRISPASR_NO_C2PA_NATIVE=ON >/dev/null

cache="${build_dir}/CMakeCache.txt"
for option in \
    GGML_NATIVE GGML_SSE42 GGML_AVX GGML_AVX_VNNI GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 \
    GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 \
    GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16; do
    if ! grep -q "^${option}:BOOL=OFF$" "${cache}"; then
        echo "FAIL: CRISPASR_PORTABLE_CPU left ${option} enabled" >&2
        grep "^${option}:" "${cache}" >&2 || true
        exit 1
    fi
done

echo "PASS: CRISPASR_PORTABLE_CPU forces the generic x86-64 baseline (15/15 ISA flags OFF)"
