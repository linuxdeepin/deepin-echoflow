# Concurrency, parallelism & scaling

How CrispASR uses multiple cores, how it handles concurrent requests, and how
to scale to large batch workloads (thousands of hours of audio). If you skimmed
the docs looking for "concurrency", "parallel", "batch", or "throughput" and
found nothing — this is the page.

**TL;DR**

- **One transcription already uses multiple CPU threads** (ggml intra-op
  parallelism, `--threads`). You don't configure anything for this.
- **The HTTP server accepts connections concurrently, but by default runs
  inference one-at-a-time** through a single loaded model (mutex-serialized).
- **`--server-workers N` / `CRISPASR_SERVER_WORKERS=N`** loads N independent
  model instances so N pure-ASR requests run *concurrently* — a real throughput
  win on a GPU (or an under-utilised box), a no-op-or-loss on a CPU model that
  already saturates memory bandwidth.
- **For bulk offline work (your "few thousand hours"), the simplest and fastest
  answer is process-level parallelism over the CLI** (`xargs -P` / GNU
  `parallel`) — see [Offline bulk transcription](#offline-bulk-transcription-the-thousands-of-hours-case).
- **For a serving deployment, run N replicas behind a load balancer**
  (docker-compose `--scale`, Caddy/nginx). This is the same thing you may have
  already discovered by hand, and it works — see [Horizontal scaling](#horizontal-scaling-replicas--load-balancer).
- **There is no batched multi-stream inference** (feeding N different clips
  through one instance as a single forward pass), and **PagedAttention is not the
  right fit** for this engine — see [What is *not* supported](#what-is-not-supported-and-why).

---

## The three layers of parallelism

CrispASR parallelises at three levels. They compose: a bulk run typically uses
all three (many processes × server workers × intra-op threads only if you size
them to your hardware — over-subscribing all three at once contends).

### 1. Intra-op threads (inside one transcription)

Every inference already runs multi-threaded inside ggml. One `crispasr`
invocation transcribing one file uses several CPU cores for the matmuls/convs.

- CLI: `--threads N` (default `min(4, hardware_concurrency())`).
- C-ABI / bindings: `crispasr_params_set_n_threads()` (default 4).
- Server: process-wide, from the same `--threads N` startup flag. There is
  **no** per-request thread-count field — every request inherits the server's
  setting.

This is *within* a single stream — it does not let two files transcribe at once.
On a GPU backend the heavy math runs on the device and thread count matters less.

### 2. The HTTP server: concurrent transport, serialized model

`crispasr --server` loads the model **once** and reuses it across requests
(no per-request reload). The HTTP layer (cpp-httplib) has its own thread pool
(≥8 threads), so it **accepts and parses many requests concurrently**.

**But by default there is a single model instance, and inference is
mutex-serialized.** N simultaneous uploads are received in parallel, then run
strictly one-at-a-time through the one model. This is deliberate: a single
loaded model/context is **not** safe to use from multiple threads at once
(see `include/crispasr.h`), so the server holds a `model_mutex` around the
inference call.

So out of the box the server gives you a *persistent, no-reload* model with
*serialized* execution — great latency per request, throughput of one stream.

### 3. `--server-workers N`: concurrent inference via independent instances

To run several transcriptions **at the same time** inside one server process,
load more than one model instance:

```bash
# Flag form (shown in --help):
crispasr --server -m model.gguf --port 8080 --server-workers 4

# Env form (overrides the flag; the ultimate gate):
CRISPASR_SERVER_WORKERS=4 crispasr --server -m model.gguf --port 8080
```

Each worker owns its own backend instance (and, on CUDA, its own device
context / command queue). A request is routed to a free worker via a blocking
RAII lease from a bounded pool (`src/core/worker_pool.h`), so up to N run
concurrently and different workers never contend.

**Important caveats — read before setting this above 1:**

- **Costs N× model memory.** Four workers = four copies of the weights resident.
- **Only "pure-ASR" requests run concurrently.** A request qualifies when it has
  an **explicit `language`** (not `auto` — auto-LID uses a shared model), **no
  aligner**, and **no punctuation / truecaser post-processing** (those contexts
  are shared and non-re-entrant). Any request that needs auto-language-ID, CTC
  alignment, or punctuation/truecasing falls back to the single shared model and
  serializes as before. If you want the concurrency, pass `language` explicitly
  and keep post-processing off (or do punctuation as a separate pass).
- **`/load` (runtime model hot-swap) is disabled** while a pool is active —
  restart the server to change models.
- **It is not always a win.** On a **GPU** where one request under-utilises the
  card (small model, spare SM/VRAM), N workers give real concurrency and higher
  throughput — this is validated on CUDA (`tools/kaggle/server-workers-cuda/`:
  WORKERS=2 makes a concurrent pair meaningfully faster than a serial pair, with
  byte-identical transcripts). On a **CPU model that already saturates memory
  bandwidth**, the workers contend for the same bottleneck and you get **no
  speedup or a net loss** — plus N× the RAM. Measure your model on your box
  before committing memory to it. When one request already saturates the
  hardware, the honest expectation is speedup ≈ 1.0 (that is not a bug).

Transcripts are identical whether a request ran on a pooled worker or the shared
model — the pool changes *scheduling*, never the math.

---

## Offline bulk transcription (the "thousands of hours" case)

If your goal is to get through a large corpus of files (not to serve live
requests), **you do not need the server at all.** Bulk ASR is
embarrassingly parallel across *files*, and the simplest robust approach is to
run several `crispasr` processes at once, each transcribing whole files.

The CLI itself processes multiple input files **sequentially** (no `-j` flag), so
drive the parallelism from the shell:

```bash
# N parallel processes, one file each. Tune -P to your hardware
# (see "sizing" below). Auto-download resolves the model once into the cache;
# point every process at the same cached model to avoid N downloads.
mkdir -p out
ls corpus/*.wav | xargs -P 4 -I{} \
  sh -c 'crispasr -m model.gguf --threads 4 -otxt -of "out/$(basename "$1" .wav)" -f "$1"' _ {}

# GNU parallel is nicer for progress, retries, and remote nodes
# (here {/.} = input basename without extension):
parallel -j 4 --bar \
  'crispasr -m model.gguf --threads 4 -otxt -of out/{/.} -f {}' ::: corpus/*.wav
```

**Sizing `-P` × `--threads`.** The two multiply. On a CPU box, total worker
threads ≈ `P × threads` should roughly match physical cores; over-subscribing
makes every process slower. A common sweet spot is a handful of processes each
with a few threads (e.g. an 8-core box: `-P 4 --threads 2`, or `-P 2
--threads 4`). On a **GPU** box, run enough processes to keep the card busy but
watch VRAM — each process loads its own copy of the model. Benchmark two or
three `(P, threads)` points on a representative subset before committing to the
full corpus; the [benchmarking guide](benchmarking.md) has the proof-of-work
rules (a crash or wrong-model load can fake a "fast" run).

This scales linearly with cores/machines, needs no server, and each file's
success/failure is independent (easy to retry just the failures).

If you'd rather push files at a **persistent** server (model stays warm, no
per-file process spawn), run it with `--server-workers N` and fan out client
requests — but for a one-shot corpus the CLI fan-out above is usually simpler and
at least as fast.

---

## Horizontal scaling (replicas + load balancer)

For a serving deployment that must handle sustained concurrent traffic — or to
scale past one box — run **N independent server processes/containers behind a
load balancer.** This is the standard, robust pattern, and it is exactly the
"8 replicas behind a proxy → several-× throughput" setup people arrive at by
hand. It composes with (or replaces) the in-process `--server-workers` pool:
each replica is a full server, so it also sidesteps the pure-ASR routing caveat
and the shared-model `/load` restriction.

### Docker Compose

The repo ships a [`docker-compose.yml`](../docker-compose.yml) with a service
named `crispasr`. Scale it and put a proxy in front:

```bash
# N independent server containers, each with the model loaded once.
docker compose up --build --scale crispasr=4
```

> **Port gotcha.** The shipped compose **publishes one host port for the whole
> service** (`"${CRISPASR_HOST_PORT:-${CRISPASR_PORT:-8080}}:${CRISPASR_PORT:-8080}"`
> — `8080:8080` unless you override it, and the same value for every replica).
> Scaling as-is makes the replicas collide on that host port
> ("port is already allocated"). The fix is to **not** publish the backend port
> on the host and instead let the load balancer reach the replicas over the
> compose network (as below) — remove the `ports:` mapping from the `crispasr`
> service and publish only the proxy. (Docker's DNS round-robins the service
> name to all replica IPs; a real LB with health checks is better.)

Then front them with any load balancer. A minimal Caddy config (round-robin /
least-conn over the replicas):

```caddyfile
:8080 {
    reverse_proxy crispasr:8080 {
        lb_policy least_conn
        health_uri /health
    }
}
```

`GET /health` is public (no API key) precisely so a load balancer / orchestrator
can health-check each replica. nginx `upstream { least_conn; server … }` works
the same way.

### In-process pool vs. replicas — which to use

| | `--server-workers N` (in-process pool) | N replicas + load balancer |
|---|---|---|
| Setup | one flag | compose `--scale` + a proxy |
| Memory | N× weights in one process | N× weights across N processes |
| Concurrency scope | **pure-ASR requests only** (see caveats) | every request, no routing caveat |
| Model hot-swap (`/load`) | disabled while pooled | per-replica (roll one at a time) |
| Fault isolation | one process — a crash takes all workers | a crashed replica is drained by the LB |
| Cross-machine | no (single process) | **yes** |
| Best for | one GPU under-utilised by a single stream | production serving, multi-box, mixed requests |

Rule of thumb: **one GPU, small model, want concurrency cheaply →
`--server-workers`. Production serving / multiple machines / mixed request types
→ replicas behind a load balancer.** They stack: e.g. 2 replicas × 2 workers.

---

## What is *not* supported, and why

### No batched multi-stream inference

There is no API that takes N *different* audio clips and runs them through **one**
model instance as a single batched forward pass. "Batch" inside the codebase
means splitting **one** long stream into sequential chunks with context carry —
not cross-clip batching. Concurrency is achieved by running independent
instances (workers / processes / replicas), each handling one stream, never by
batching several streams on one instance.

Why: the backend zoo is heterogeneous (encoder-heavy CTC/transducer models,
small AR decoders, flow-matching TTS, …) with per-model preprocessing, chunking,
and decoding. A generic cross-stream batch path would have to be built and
verified per backend, and for the dominant use case (offline bulk) file-level
process parallelism already saturates the hardware with none of that complexity.

### A single loaded context is not thread-safe

One `crispasr_context` / session must not be driven from multiple threads at
once (documented in `include/crispasr.h`). Concurrency comes from **separate
instances**, not shared-context threading. If you use the C-ABI / a binding
directly, serialize calls per context or create one context per thread.

### PagedAttention is not the right fit

PagedAttention (vLLM) solves KV-cache **fragmentation and memory pressure** when
serving *many concurrent, long, autoregressive LLM sequences* that share a giant
paged KV cache — it lets one model instance pack many sequences into VRAM
efficiently.

That is not where CrispASR's ASR throughput goes:

- Most ASR backends are **encoder-dominated** (Whisper, Parakeet/Conformer,
  Canary, FunASR). The compute is a bounded-length encoder forward pass, not a
  long autoregressive decode with a sprawling KV cache — there is little to page.
- The autoregressive ASR/TTS backends decode **short** sequences relative to an
  LLM chat context; KV-cache paging is not the bottleneck.
- The bottleneck for bulk ASR is **per-stream throughput and keeping the
  hardware busy**, which process/worker/replica parallelism already delivers —
  and which you can measure yourself (the manual replica experiment that
  motivated this doc is exactly this effect).

So the honest recommendation is the replica/worker model documented above, not a
PagedAttention port. If a specific large-AR-decoder backend ever becomes the
throughput bottleneck under concurrent serving, KV-cache paging could be
revisited *for that backend* — but it would not be a general CrispASR feature.

---

## See also

- [Server mode (HTTP API)](server.md) — endpoints, form fields, Docker Compose,
  `CRISPASR_SERVER_WORKERS`, prebuilt CUDA images.
- [Benchmarking CrispASR](benchmarking.md) — how to measure throughput honestly
  (proof-of-work rules; a crash or wrong-model load can fake a "fast" run).
- `tools/kaggle/server-workers-cuda/` — the CUDA A/B that proves the worker pool
  gives real concurrency (and documents the CPU null).
