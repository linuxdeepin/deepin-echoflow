**Title:** `vulkan: vkResetCommandPool faults on native RADV/NVIDIA under a many-small-graph workload`

---

**Bug report, no patch — the queue-drain fix was DISPROVEN on real hardware.**
File against **ggml-org/llama.cpp** (vulkan) only if it reproduces in isolation;
right now the app-level fix is device-scoped CPU fallback (see below), not a ggml
change. We have no native RADV/NVIDIA hardware — analysis is from a reporter's
debug backtrace.

## Symptom

A small ASR model (Qwen3-Omni audio encoder + Qwen3-1.7B, our moss-transcribe
backend) segfaults on the Vulkan backend on **both** an NVIDIA GPU and an AMD
Radeon (RADV), q8_0 and q4_k, inside the vendor driver one frame under
`vkResetCommandPool`:

```
#2 vk::Device::resetCommandPool  vulkan_funcs.hpp:2479
#3 ggml_vk_command_pool_cleanup  ggml-vulkan.cpp
#4 ggml_vk_graph_cleanup
#5 ggml_backend_synchronize      ggml-backend.cpp:420
#6 ggml_backend_sched_compute_splits
#9 moss_transcribe_run_encoder   (2nd audio slice, conv chunk 0)
```

It needs "multiple slices": a short clip runs clean; a ~33 s clip (processed as 2
CLI slices sharing one backend context) crashes on the **2nd** slice.

## What we tried and ruled out

1. **Encoder `flash_attn_ext` → manual softmax** — no effect. Not the FA path.
2. **Disable async submission (`GGML_VK_DISABLE_ASYNC`)** — no effect (and it is
   timing-fragile: `support_async` is frozen at device creation, so setting the
   env from the backend after any earlier device enumeration is a no-op).
3. **Drain the pool's queue before the reset** (`p.q->queue.waitIdle()` in
   `ggml_vk_command_pool_cleanup`) — **no effect.** The reporter's debug `bt full`
   confirms the guard was compiled in (`resetCommandPool` shifted to the expected
   line) and `p.q` was a valid queue, so `waitIdle` ran — and `resetCommandPool`
   *still* faulted. If the buffers were merely still executing, a queue drain would
   have finished them. So the fault is **not** "reset a pool with pending buffers."

## Current read

A `resetCommandPool` that faults *after* the queue is provably idle points to
corrupted pool / driver state, i.e. an earlier out-of-bounds write or a graph-scale
allocation/aliasing problem in the large encoder graph that only surfaces at the
next Vulkan API call. This is the same signature as our TADA #192 (unreproducible
in isolation, native-only, never on MoltenVK). MoltenVK auto-manages
command-buffer lifetime and never reproduces it, so we cannot bisect locally.

## App-level fix (shipped)

Because the failure is native-Vulkan-specific and MoltenVK/CUDA/Metal are verified
safe, the model backend inspects the device description of the backend it actually
received (`ggml_backend_dev_description`, no env / ordering race) and runs on CPU
for native Vulkan; MoltenVK/CUDA/Metal keep the GPU. Only AMD-on-Linux (no CUDA)
loses acceleration, deterministically, instead of crashing. `..._VULKAN_NATIVE=1`
forces the GPU path for hardware A/B.

## To make this a ggml/llama.cpp report

Needs an isolated repro: a `test-backend-ops`-style loop that submits many small
graphs interleaved with `tensor_set/get` on one context, on RADV, and watches for
the `resetCommandPool` fault — plus a `GGML_VULKAN_CHECK_RESULTS=1` run to find the
first diverging op. Pending native hardware access; do not file until we can point
at either an isolated repro or the specific overflow.

## Status

Disproven queue-drain patch removed. Bug report only — awaiting native-hardware
isolation before filing at ggml-org/llama.cpp.
