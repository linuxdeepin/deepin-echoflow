# CUDA and upstream refresh implementation plan

**Goal:** Enable CUDA inference and update the vendored runtime without losing
EchoFlow's existing integration fixes.

**Architecture:** Retain the upstream `GGML_CUDA` build switch. Let the chat
runtime select GPU offload using its default parameters. Preserve CPU defaults
and merge the official runtime into the existing fork integration.

**Tech Stack:** CMake, C++17, CrispASR/ggml, NVIDIA CUDA 12.8, shell specs, QTest.

- [x] Extend `tests/spec/test_onboarding_install.sh` with a CPU default check
  and an install using `GGML_CUDA=ON CMAKE_CUDA_ARCHITECTURES=120`; require both
  CMake arguments in the mocked command log. Run the spec and confirm failure
  before implementing forwarding.
- [x] Declare `GGML_CUDA` OFF in `CMakeLists.txt`. Forward that environment
  option and an optional architecture argument in `install-user.sh`. Remove
  `params.n_gpu_layers = 0` in `service/LocalLlmPolisher.cpp` so the runtime's
  default all-layer offload applies when a GPU backend is available.
- [x] Merge upstream `830ca6024` (including planned `04200fae7`) into the existing CrispASR fork revision,
  resolve any slim-build incompatibility, and update nested ggml. Inspect
  changes affecting Qwen3, chat, mel preprocessing, GPU selection and build.
- [x] Run `bash tests/spec/test_onboarding_install.sh`, shell syntax checks,
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo`,
  `cmake --build build -j4`, and
  `ctest --test-dir build --output-on-failure`.
- [x] Configure `build/cuda` with `-DGGML_CUDA=ON`, an isolated toolkit, and
  `-DCMAKE_CUDA_ARCHITECTURES=120`; build and run available checks. A toolkit
  or driver failure must be reported separately from passing CPU tests.
- [x] Update README build/install commands and add an upstream review note
  with exact revisions, retained patches, relevant changes and validation.
  Review `git diff --check` and `git status --short` before delivery.

GPU execution and a small CPU/CUDA latency comparison were verified on
2026-11-02 after driver recovery. Record the input identity, raw samples and
bounded interpretation in `docs/performance/cuda-upstream-benchmark.json` and
`docs/performance/cuda-upstream-refresh.md`; this is not a release gate.
