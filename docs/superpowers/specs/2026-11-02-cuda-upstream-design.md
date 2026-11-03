# CUDA inference and CrispASR refresh

Enable optional NVIDIA acceleration for ASR and transcript polish, while keeping
the default build CPU-only. Use upstream's `GGML_CUDA` CMake option rather than
introducing a second backend switch. Forward it through the user installer, with
an optional `CMAKE_CUDA_ARCHITECTURES` override for builds without a working GPU.
Remove EchoFlow's forced zero-layer polish setting so the chat runtime can use
its automatic GPU offload policy. Keep VAD on CPU.

Refresh CrispASR against official main `830ca6024ece6477da959236ee4f825a4d085300`
(including the planned `04200fae791c8d5113fa47d5896ccd13f18f6bad` and PR #565)
and preserve the four existing fork patches (slim embedding, VAD compute error
handling, native GQA, and slim C API guards). Review shared preprocessing and
ggml changes; do not enable additional model families or automatic downloads.
The resulting submodule integration commit stays local until explicitly pushed.

Verify installer argument propagation with its executable spec; configure, build
and run all tests in CPU mode. Build CUDA with an isolated NVIDIA toolkit and
an explicit architecture, then run model smoke tests when the driver works.
Record unavailable GPU validation explicitly. Do not change system drivers,
install the application, or restart the user's services as part of this work.
