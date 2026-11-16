# # CrispASR vs transcribe.cpp — head-to-head GPU benchmark
#
# Systematic evaluation of CrispASR against transcribe.cpp
# (https://github.com/handy-computer/transcribe.cpp) on shared ASR models.
#
# For each overlapping model family this kernel:
#   1. Downloads the CrispASR GGUF from cstr/ HF repos
#   2. Downloads the transcribe.cpp GGUF from handy-computer/ HF repos
#   3. Runs both on jfk.wav and measures:
#        - Transcript text (WER vs reference)
#        - Wall-clock inference time → RTF
#        - Model load time
#        - Peak RSS (via /proc/self/status)
#   4. Cleans up both GGUFs before the next model to stay within ~20 GB scratch
#   5. Streams per-model results to cstr/crispasr-kaggle-progress on HF
#
# "transcribe.cpp only" models (GigaAM family) are tested with transcribe.cpp
# only — these represent a CrispASR coverage gap.
#
# References: tools/kaggle-benchmark-all-backends.py (CrispASR full sweep),
# transcribe.cpp docs/models/*, scripts/wer/run.py
#
# Account:  $KAGGLE_ACCOUNT (GPU quota, separate from $KAGGLE_ACCOUNT's 30 h/week)
# Datasets: ${KAGGLE_ACCOUNT}/crispasr-hf-token  (HF auth)
#           ${KAGGLE_ACCOUNT}/crispasr-ccache     (warm CrispASR build, ~3 min vs ~20 min)

import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
_PROGRESS = WORK / "progress.jsonl"
_T0 = time.time()
_HF_LAST_PUSH = 0.0
_HF_REPO = "cstr/crispasr-kaggle-progress"
_HF_PATH = (
    f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    f"-transcribe-cpp-bench.jsonl"
)
MAX_WALL_S = 8 * 3600  # leave 1 h margin vs Kaggle's 9 h session limit

# ── Reference JFK transcript ──────────────────────────────────────────────────
JFK_REF = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
JFK_DURATION_S = 11.0  # approximate duration of jfk.wav

# ── Model pairs to benchmark ─────────────────────────────────────────────────
# Format:
#   family:              human-readable name
#   ca_backend:          CrispASR -b flag (or None = whisper file-based)
#   ca_url:              CrispASR GGUF download URL
#   ca_file:             local filename for CrispASR GGUF
#   tc_url:              transcribe.cpp GGUF download URL
#   tc_file:             local filename for transcribe.cpp GGUF
#   ca_wer_libri:        CrispASR LibriSpeech test-clean WER (from docs, None if unknown)
#   tc_wer_libri:        transcribe.cpp LibriSpeech test-clean WER (from their docs)
#   timeout_s:           per-model timeout for both inference calls
#   notes:               benchmark notes

SHARED_MODELS = [
    {
        "family":        "Whisper base",
        "ca_backend":    "whisper",
        "ca_url":        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "ca_file":       "ggml-base.bin",
        "tc_url":        "https://huggingface.co/handy-computer/whisper-base-gguf/resolve/main/whisper-base-Q8_0.gguf",
        "tc_file":       "whisper-base-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "5.12%",
        "timeout_s":     60,
        "notes":         "CrispASR ggml-base.bin vs t.cpp Q8_0 (same weights, different GGUF schemas)",
    },
    {
        "family":        "Moonshine Tiny",
        "ca_backend":    "moonshine",
        "ca_url":        "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/moonshine-tiny-q4_k.gguf",
        "ca_file":       "moonshine-tiny-q4_k.gguf",
        "ca_companion":  "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/tokenizer.bin",
        "tc_url":        "https://huggingface.co/handy-computer/moonshine-tiny-gguf/resolve/main/moonshine-tiny-Q8_0.gguf",
        "tc_file":       "moonshine-tiny-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "4.60%",
        "timeout_s":     60,
        "notes":         "CrispASR Q4_K vs t.cpp Q8_0",
    },
    {
        "family":        "SenseVoice Small",
        "ca_backend":    "sensevoice",
        "ca_url":        "https://huggingface.co/cstr/sensevoice-small-GGUF/resolve/main/sensevoice-small-q4_k.gguf",
        "ca_file":       "sensevoice-small-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/SenseVoiceSmall-gguf/resolve/main/SenseVoiceSmall-Q8_0.gguf",
        "tc_file":       "SenseVoiceSmall-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "3.13%",
        "timeout_s":     60,
        "notes":         "CrispASR Q4_K vs t.cpp Q8_0; SenseVoice emits emotion/LID tags in CrispASR",
    },
    {
        "family":        "Moonshine Streaming Tiny",
        "ca_backend":    "moonshine-streaming",
        "ca_url":        "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/moonshine-streaming-tiny-q4_k.gguf",
        "ca_file":       "moonshine-streaming-tiny-q4_k.gguf",
        "ca_companion":  "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/tokenizer.bin",
        "tc_url":        "https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-Q8_0.gguf",
        "tc_file":       "moonshine-streaming-tiny-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "4.52%",
        "timeout_s":     60,
        "notes":         "Both run in offline mode on jfk.wav",
    },
    {
        "family":        "Parakeet TDT 0.6B",
        "ca_backend":    "parakeet",
        "ca_url":        "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf",
        "ca_file":       "parakeet-tdt-0.6b-v3-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-Q4_K_M.gguf",
        "tc_file":       "parakeet-tdt-0.6b-v2-Q4_K_M.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "1.72%",
        "timeout_s":     90,
        "notes":         "CrispASR v3 (25 EU langs) vs t.cpp v2 (EN only) — different checkpoints",
    },
    {
        "family":        "Qwen3-ASR 0.6B",
        "ca_backend":    "qwen3",
        "ca_url":        "https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-q4_k.gguf",
        "ca_file":       "qwen3-asr-0.6b-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "tc_file":       "Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "2.26%",
        "timeout_s":     90,
        "notes":         "CrispASR Q4_K vs t.cpp Q4_K_M; same upstream checkpoint; Q8_0 t.cpp=2.11%",
    },
    {
        "family":        "Canary 1B v2",
        "ca_backend":    "canary",
        "ca_url":        "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf",
        "ca_file":       "canary-1b-v2-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q4_K_M.gguf",
        "tc_file":       "canary-1b-v2-Q4_K_M.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  None,  # canary-1b-v2 specific docs may not have WER yet
        "timeout_s":     120,
        "notes":         "Both Q4_K on same NVIDIA canary-1b-v2 checkpoint",
    },
    {
        "family":        "FunASR Nano 2512",
        "ca_backend":    "funasr",
        "ca_url":        "https://huggingface.co/cstr/funasr-nano-GGUF/resolve/main/funasr-nano-2512-q8_0.gguf",
        "ca_file":       "funasr-nano-2512-q8_0.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/Fun-ASR-Nano-2512-gguf/resolve/main/Fun-ASR-Nano-2512-Q8_0.gguf",
        "tc_file":       "Fun-ASR-Nano-2512-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "1.79%",
        "timeout_s":     120,
        "notes":         "Both Q8_0 (CrispASR uses Q8_0 to avoid CUDA F16×F32 saturation bug)",
    },
    {
        "family":        "Nemotron 3.5 ASR Streaming 0.6B",
        "ca_backend":    "nemotron",
        "ca_url":        "https://huggingface.co/cstr/nemotron-3.5-asr-streaming-0.6b-GGUF/resolve/main/nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        "ca_file":       "nemotron-3.5-asr-streaming-0.6b-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
        "tc_file":       "nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  None,
        "timeout_s":     120,
        "notes":         "CrispASR Q4_K vs t.cpp Q8_0",
    },
    {
        "family":        "Cohere Transcribe",
        "ca_backend":    "cohere",
        "ca_url":        "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf",
        "ca_file":       "cohere-transcribe-q4_k.gguf",
        "tc_url":        "https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf/resolve/main/cohere-transcribe-03-2026-Q4_K_M.gguf",
        "tc_file":       "cohere-transcribe-03-2026-Q4_K_M.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "1.25%",
        "timeout_s":     180,
        "notes":         "Both Q4_K; ~1.5 GB each; encoder-decoder with cross-attention",
    },
    {
        "family":        "Whisper Large v3 Turbo",
        "ca_backend":    "whisper",
        "ca_url":        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        "ca_file":       "ggml-large-v3-turbo.bin",
        "tc_url":        "https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf/resolve/main/whisper-large-v3-turbo-Q8_0.gguf",
        "tc_file":       "whisper-large-v3-turbo-Q8_0.gguf",
        "ca_wer_libri":  None,
        "tc_wer_libri":  "2.01%",
        "timeout_s":     180,
        "notes":         "CrispASR ggml .bin vs t.cpp Q8_0; larger model tests GPU scaling",
    },
]

# Models that exist ONLY in transcribe.cpp (CrispASR coverage gaps)
TC_ONLY_MODELS = [
    {
        "family":        "GigaAM v3 E2E-CTC",
        "tc_url":        "https://huggingface.co/handy-computer/gigaam-v3-e2e-ctc-gguf/resolve/main/gigaam-v3-e2e-ctc-Q8_0.gguf",
        "tc_file":       "gigaam-v3-e2e-ctc-Q8_0.gguf",
        "tc_wer_libri":  "5.50%",
        "timeout_s":     90,
        "notes":         "Russian-focused + EN ASR; no CrispASR equivalent yet",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _push_hf_progress():
    global _HF_LAST_PUSH
    if time.time() - _HF_LAST_PUSH < 30:
        return
    if not os.environ.get("HF_TOKEN") or not _PROGRESS.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(_PROGRESS),
            path_in_repo=_HF_PATH,
            repo_id=_HF_REPO,
            repo_type="dataset",
            commit_message=f"bench progress @ {int(time.time() - _T0)}s",
        )
        _HF_LAST_PUSH = time.time()
    except Exception:
        pass


def step(name, **extra):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - _T0, 2),
        "step": name,
        **extra,
    }
    _PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with _PROGRESS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(
        f"[step {rec['elapsed_s']:>7.1f}s] {name}"
        + (f"  {extra}" if extra else ""),
        flush=True,
    )
    _push_hf_progress()


def run_cmd(cmd: str, timeout: int = 600, stream: bool = False):
    """Run shell command; return (ok, stdout, stderr, elapsed_s)."""
    t0 = time.time()
    if stream:
        try:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stderr_buf = []
            import select
            while True:
                if time.time() - t0 > timeout:
                    proc.kill()
                    return False, "", "TIMEOUT", time.time() - t0
                ready, _, _ = select.select([proc.stderr], [], [], 1.0)
                if ready:
                    line = proc.stderr.readline()
                    if line:
                        stderr_buf.append(line)
                        print(f"  [build] {line.rstrip()}", flush=True)
                if proc.poll() is not None:
                    for line in proc.stderr:
                        stderr_buf.append(line)
                    break
            stdout = proc.stdout.read()
            return proc.returncode == 0, stdout, "".join(stderr_buf), time.time() - t0
        except Exception as e:
            return False, "", str(e), time.time() - t0
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0, r.stdout, r.stderr, time.time() - t0
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT", time.time() - t0


def peak_rss_mb() -> float:
    """Read current process peak RSS in MB from /proc/self/status."""
    try:
        for line in open("/proc/self/status").readlines():
            if line.startswith("VmPeak:"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def wer_simple(ref: str, hyp: str) -> float:
    """Simple word error rate (substitutions + deletions + insertions) / len(ref)."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    if not ref_words:
        return 0.0
    # DP levenshtein on words
    m, n = len(ref_words), len(hyp_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[n] / m


def strip_tags(text: str) -> str:
    """Remove SenseVoice-style <|TAG|> tokens and normalise whitespace."""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    return " ".join(text.split())


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, normalise whitespace for WER."""
    text = strip_tags(text)
    text = text.lower()
    # Strip inline language tags emitted by some backends (e.g. nemotron "en-us")
    text = re.sub(r"\b[a-z]{2}-[a-z]{2}\b", "", text)
    text = re.sub(r"[^a-z0-9 ''-]", " ", text)
    return " ".join(text.split())


def download_gguf(url: str, dest: Path, timeout: int = 600) -> bool:
    """Download a GGUF file with wget (HF_HUB_ENABLE_HF_TRANSFER=1 for hf_transfer)."""
    if dest.exists():
        step(f"download.skip", file=dest.name, reason="already exists")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    auth = f'--header "Authorization: Bearer {token}"' if token else ""
    ok, _, err, elapsed = run_cmd(
        f'wget -q --show-progress {auth} -O "{dest}" "{url}"',
        timeout=timeout,
    )
    if not ok or not dest.exists() or dest.stat().st_size < 1024:
        step("download.failed", url=url, error=err[-200:])
        dest.unlink(missing_ok=True)
        return False
    step("download.done", file=dest.name, size_mb=round(dest.stat().st_size / 1e6, 1),
         elapsed_s=round(elapsed, 1))
    return True


def run_crispasr(binary: Path, model: Path, audio: Path, backend: str | None,
                 timeout: int = 120, no_gpu: bool = False) -> dict:
    """Run crispasr; return structured result with per-stage timing breakdown.

    CrispASR outputs transcript to stdout, timing to stderr:
      stderr: "crispasr: transcribed X.Xs audio in Y.YYs (Z.Zx realtime)"
    Per-stage bench lines (when *_BENCH=1):
      stderr: "  <backend>_bench: <stage>  <ms> ms"
    """
    b_flag = f"--backend {backend}" if backend else ""
    ng_flag = "-ng" if no_gpu else ""
    # Enable per-stage bench output for the backend + common subsystems
    bench_env = {**os.environ}
    if backend:
        # Map backend name to bench env var prefix
        bench_name = backend.upper().replace("-", "_")
        bench_env[f"{bench_name}_BENCH"] = "1"
    # Also enable common subsystems + batched decode for GPU runs
    for k in ["MOONSHINE_BENCH", "MOONSHINE_STREAMING_BENCH", "NEMOTRON_BENCH",
              "COHERE_BENCH", "PARAKEET_BENCH", "SENSEVOICE_BENCH", "FUNASR_BENCH",
              "QWEN3_ASR_BENCH", "VOXTRAL_BENCH", "WHISPER_BENCH"]:
        bench_env[k] = "1"
    # §232: batched TDT/RNNT decode DISABLED — v14 showed 5-9x SLOWER on GPU.
    # The CPU sgemm for 32 frames computes unused logits; sequential sgemv is
    # faster because each call is tiny and terminates at first blank.
    # The real fix is porting LSTM+joint to a ggml graph (GPU-native decode).
    t0 = time.time()
    # -l en: skip LID probe (saves ~1-2s); --auto-download for companion files
    try:
        proc = subprocess.run(
            f'"{binary}" -m "{model}" {b_flag} {ng_flag} -l en --auto-download "{audio}"',
            shell=True, capture_output=True, text=True, timeout=timeout,
            env=bench_env,
        )
        ok = proc.returncode == 0
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired:
        return {"transcript": "", "infer_s": timeout, "rtf": None, "ok": False,
                "stderr_tail": "TIMEOUT", "bench": {}}
    infer_s = round(time.time() - t0, 3)

    # Extract transcript from stdout — CrispASR outputs plain text, one segment per line
    transcript = out.strip()

    # Extract RTF from stderr: "crispasr: transcribed X.Xs audio in Y.YYs (Z.Zx realtime)"
    rtf = None
    for line in err.splitlines():
        m = re.search(r"transcribed.*?(\d+\.?\d*)x realtime", line)
        if m:
            try:
                rtf_factor = float(m.group(1))
                rtf = round(1.0 / rtf_factor, 4)  # convert Nx to RTF (< 1 means faster than real-time)
            except Exception:
                pass

    # Parse per-stage bench timing from stderr lines like:
    #   "  moonshine_bench: encoder                1623.89 ms"
    #   "  nemotron_bench: mel                      366.78 ms"
    bench = {}
    for line in err.splitlines():
        m2 = re.search(r"_bench:\s+(\S+)\s+([\d.]+)\s*ms", line)
        if m2:
            bench[m2.group(1)] = round(float(m2.group(2)), 1)

    return {
        "transcript": transcript,
        "infer_s": infer_s,
        "rtf": rtf,
        "ok": ok,
        "stderr_tail": err[-500:],  # always include for diagnostics
        "bench": bench,  # per-stage timing in ms
    }


def run_transcribe_cpp(binary: Path, model: Path, audio: Path,
                       timeout: int = 120) -> dict:
    """Run transcribe-cli in --batch-jsonl mode; return structured result.

    transcribe-cli single-file mode outputs human-readable text; batch-jsonl
    mode outputs clean JSON with per-utterance timings:
      line 0:  {"type":"batch_header","load_ms":...}
      line 1+: {"file":"...","text":"...","mel_ms":...,"encode_ms":...,"decode_ms":...}
    """
    import tempfile
    # Write a one-line batch file pointing at the audio file
    batch_file = Path(tempfile.mktemp(suffix=".txt", dir=str(audio.parent)))
    batch_file.write_text(str(audio) + "\n")
    t0 = time.time()
    try:
        ok, out, err, elapsed = run_cmd(
            f'"{binary}" -m "{model}" --batch "{batch_file}" --batch-jsonl 2>/dev/null',
            timeout=timeout,
        )
    finally:
        batch_file.unlink(missing_ok=True)
    infer_s = time.time() - t0

    transcript = ""
    mel_ms = encode_ms = decode_ms = load_ms = 0.0

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") == "batch_header":
            load_ms = d.get("load_ms", 0)
        elif "text" in d:
            # Per-utterance result line
            transcript = d.get("text", "")
            mel_ms = d.get("mel_ms", 0)
            encode_ms = d.get("encode_ms", 0)
            decode_ms = d.get("decode_ms", 0)
            if d.get("error"):
                ok = False

    return {
        "transcript": transcript,
        "infer_s": round(infer_s, 3),
        "mel_ms": mel_ms,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "load_ms": load_ms,
        "ok": ok,
        "stderr_tail": err[-300:] if not ok else "",
    }


# ── Stream results to HF dataset ─────────────────────────────────────────────
SWEEP_REPO = "cstr/crispasr-kaggle-progress"
RUN_TAG = f"transcribe-cpp-bench-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
SWEEP_PREFIX = f"transcribe-cpp-bench/{RUN_TAG}"
_hf_api = None
_sweep_ok = False


def _init_hf_streaming():
    global _hf_api, _sweep_ok
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[sweep] No HF_TOKEN — results will be LOCAL ONLY", flush=True)
        return
    try:
        from huggingface_hub import HfApi
        _hf_api = HfApi(token=token)
        _hf_api.upload_file(
            path_or_fileobj=io.BytesIO(b"alive\n"),
            path_in_repo=f"{SWEEP_PREFIX}/_heartbeat.txt",
            repo_type="dataset",
            repo_id=SWEEP_REPO,
            commit_message="transcribe-cpp bench heartbeat",
        )
        _sweep_ok = True
        print(f"[sweep] HF streaming OK → {SWEEP_REPO}/{SWEEP_PREFIX}", flush=True)
    except Exception as e:
        print(f"[sweep] HF streaming disabled: {e!r}", flush=True)


def sweep_publish(key: str, payload: dict):
    if not _sweep_ok or _hf_api is None:
        return
    data = json.dumps({**payload, "_run": RUN_TAG}, indent=2, default=str).encode()
    try:
        _hf_api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=f"{SWEEP_PREFIX}/results/{key}.json",
            repo_type="dataset",
            repo_id=SWEEP_REPO,
            commit_message=f"bench {RUN_TAG}: {key}",
        )
        print(f"  [sweep] ↑ {key}.json", flush=True)
    except Exception as e:
        print(f"  [sweep] ! upload failed: {e!r}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: Install Python deps
# ─────────────────────────────────────────────────────────────────────────────
step("install_deps.begin")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub", "hf_transfer",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
step("install_deps.done")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Clone CrispASR (feature branch) + kaggle_harness
# ─────────────────────────────────────────────────────────────────────────────
CRISPASR_DIR = WORK / "CrispASR"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_BRANCH = "feat/transcribe-cpp-eval"

step("clone_crispasr.begin", branch=CRISPASR_BRANCH)
if CRISPASR_DIR.exists():
    run_cmd(
        f"cd {CRISPASR_DIR} && git fetch --depth 1 origin {CRISPASR_BRANCH} "
        f"&& git reset --hard FETCH_HEAD",
        timeout=120,
    )
    step("clone_crispasr.updated")
else:
    ok, _, err, _ = run_cmd(
        f"git clone --depth 1 --branch {CRISPASR_BRANCH} {CRISPASR_URL} {CRISPASR_DIR}",
        timeout=300,
    )
    if not ok:
        # Branch may not be pushed yet — fall back to main
        step("clone_crispasr.branch_missing", fallback="main")
        run_cmd(
            f"git clone --depth 1 {CRISPASR_URL} {CRISPASR_DIR}",
            timeout=300,
        )
    step("clone_crispasr.done")

# git clone --depth 1 does not pull submodules; ggml is one since 2026-07-07
# and cmake fails at add_subdirectory(ggml) without it (#238 lesson).
if not (CRISPASR_DIR / "ggml" / "CMakeLists.txt").exists():
    step("clone_crispasr.submodule_init.begin")
    subprocess.run(
        ["git", "-C", str(CRISPASR_DIR), "submodule", "update",
         "--init", "--recursive", "--depth", "1"],
        check=True,
    )
    step("clone_crispasr.submodule_init.done")

sys.path.insert(0, str(CRISPASR_DIR / "tools" / "kaggle"))
# Also accept bundled harness in same dir as this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(progress_path=str(WORK / "progress.jsonl"))
kh.resolve_hf_token()

_init_hf_streaming()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build CrispASR with CUDA (using kaggle_harness)
# ─────────────────────────────────────────────────────────────────────────────
kh.step("build_crispasr.begin")
_build_log = (WORK / "build_crispasr.log").open("a")

_tc = kh.install_build_toolchain()
has_gpu = os.path.exists("/usr/local/cuda/bin/nvcc")
CRISPASR_BUILD = CRISPASR_DIR / "build"
CRISPASR_BUILD.mkdir(exist_ok=True)

_cuda_arch = kh.detect_cuda_arch() if has_gpu else None
_cuda_flags = kh.cuda_build_flags(_cuda_arch) if has_gpu else []
_cache_flags = kh.cache_and_link_flags()
_jobs = kh.safe_build_jobs(gpu=has_gpu)
_gen = ["-G", "Ninja"] if _tc.get("ninja") else []

_cmake_ret = subprocess.run(
    ["cmake"] + _gen + [
        "-B", str(CRISPASR_BUILD),
        "-DCMAKE_BUILD_TYPE=Release",
    ] + _cuda_flags + _cache_flags + [str(CRISPASR_DIR)],
    stdout=_build_log, stderr=_build_log,
    env={**os.environ, "CCACHE_DIR": "/kaggle/working/.ccache"},
)
if _cmake_ret.returncode != 0:
    _build_log.flush()
    print("=== build_crispasr.log (tail) ===", flush=True)
    print(open(str(WORK / "build_crispasr.log")).read()[-4000:], flush=True)
    raise subprocess.CalledProcessError(_cmake_ret.returncode, "cmake configure (CrispASR)")

with kh.build_heartbeat("cmake.build.crispasr"):
    subprocess.run(
        ["cmake", "--build", str(CRISPASR_BUILD), f"-j{_jobs}", "--target", "crispasr"],
        check=True, stdout=_build_log, stderr=_build_log, timeout=30 * 60,
    )

CRISPASR_BIN = CRISPASR_BUILD / "bin" / "crispasr"
assert CRISPASR_BIN.is_file(), f"crispasr binary missing: {CRISPASR_BIN}"
kh.step("build_crispasr.done", binary=str(CRISPASR_BIN))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Clone and build transcribe.cpp
# ─────────────────────────────────────────────────────────────────────────────
TC_DIR = WORK / "transcribe.cpp"
TC_URL = "https://github.com/handy-computer/transcribe.cpp.git"

step("clone_transcribecpp.begin")
if TC_DIR.exists():
    run_cmd(
        f"cd {TC_DIR} && git fetch --depth 1 origin main && git reset --hard FETCH_HEAD",
        timeout=120,
    )
    step("clone_transcribecpp.updated")
else:
    ok, _, err, _ = run_cmd(
        f"git clone --depth 1 {TC_URL} {TC_DIR}", timeout=300
    )
    assert ok, f"transcribe.cpp clone failed: {err[-500:]}"
    step("clone_transcribecpp.done")

step("build_transcribecpp.begin")
_tc_build_log = (WORK / "build_transcribecpp.log").open("a")
TC_BUILD = TC_DIR / "build"

# Fix CUDA::cuda_driver on Kaggle: the target_link_libraries(ggml-cuda ...
# CUDA::cuda_driver) in transcribe.cpp's ggml fork fails because libcuda.so
# only lives in /usr/local/cuda/lib64/stubs/. The real fix is GGML_CUDA_NO_VMM=ON
# which gates the CUDA::cuda_driver link entirely (same flag CrispASR uses).
# Also export LIBRARY_PATH so the linker finds the stub at link time.
_tc_cuda_flags = ["-DTRANSCRIBE_CUDA=ON"] if has_gpu else []
if has_gpu and _cuda_arch:
    _tc_cuda_flags += [
        f"-DCMAKE_CUDA_ARCHITECTURES={_cuda_arch}",
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
        "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        "-DGGML_CUDA_NO_VMM=ON",  # skip CUDA::cuda_driver link
    ]
    # Belt-and-suspenders: linker needs the stub too
    os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs:" + os.environ.get("LIBRARY_PATH", "")

def _cmake_configure_tc(cuda_flags):
    return subprocess.run(
        ["cmake"] + _gen + [
            "-B", str(TC_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
        ] + cuda_flags + _cache_flags + [str(TC_DIR)],
        stdout=_tc_build_log, stderr=_tc_build_log,
        env={**os.environ, "CCACHE_DIR": "/kaggle/working/.ccache"},
    )

_tc_cmake_ret = _cmake_configure_tc(_tc_cuda_flags)
if _tc_cmake_ret.returncode != 0 and _tc_cuda_flags:
    # CUDA configure failed — fall back to CPU-only build
    _tc_build_log.flush()
    print("=== build_transcribecpp.log (tail, CUDA attempt) ===", flush=True)
    print(open(str(WORK / "build_transcribecpp.log")).read()[-2000:], flush=True)
    step("build_transcribecpp.cuda_fallback_cpu")
    import shutil as _shutil
    _shutil.rmtree(TC_BUILD, ignore_errors=True)
    TC_BUILD.mkdir(exist_ok=True)
    _tc_cmake_ret = _cmake_configure_tc([])  # CPU-only retry

if _tc_cmake_ret.returncode != 0:
    _tc_build_log.flush()
    print("=== build_transcribecpp.log (tail) ===", flush=True)
    print(open(str(WORK / "build_transcribecpp.log")).read()[-4000:], flush=True)
    raise subprocess.CalledProcessError(_tc_cmake_ret.returncode, "cmake configure (transcribe.cpp)")

with kh.build_heartbeat("cmake.build.transcribecpp"):
    subprocess.run(
        ["cmake", "--build", str(TC_BUILD), f"-j{_jobs}", "--target", "transcribe-cli"],
        check=True, stdout=_tc_build_log, stderr=_tc_build_log, timeout=30 * 60,
    )

TC_BIN = TC_BUILD / "bin" / "transcribe-cli"
assert TC_BIN.is_file(), f"transcribe-cli binary missing: {TC_BIN}"
step("build_transcribecpp.done", binary=str(TC_BIN))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Prepare test audio
# ─────────────────────────────────────────────────────────────────────────────
JFK_WAV = WORK / "jfk.wav"
# Use jfk.wav from CrispASR samples (16 kHz mono WAV)
if not JFK_WAV.exists():
    shutil.copy(CRISPASR_DIR / "samples" / "jfk.wav", JFK_WAV)
step("audio.ready", file=str(JFK_WAV))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Run head-to-head benchmark on shared models (GPU + CPU)
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = WORK / "models"
MODELS_DIR.mkdir(exist_ok=True)
results = []


def _bench_crispasr(ca_gguf, backend, timeout, no_gpu, label):
    """Run CrispASR and return a dict of prefixed result keys."""
    res = run_crispasr(CRISPASR_BIN, ca_gguf, JFK_WAV,
                       backend=backend, timeout=timeout, no_gpu=no_gpu)
    norm = normalise(res["transcript"])
    wer = wer_simple(JFK_REF, norm)
    rtf = res.get("rtf") or round(res["infer_s"] / JFK_DURATION_S, 3)
    bench = res.get("bench", {})
    step(f"bench.ca_{label}_done", ok=res["ok"], rtf=rtf, wer=round(wer, 4),
         transcript=norm[:80], bench=bench)
    d = {
        f"ca_{label}_ok": res["ok"],
        f"ca_{label}_transcript_norm": norm,
        f"ca_{label}_jfk_wer": round(wer, 4),
        f"ca_{label}_infer_s": res["infer_s"],
        f"ca_{label}_rtf": rtf,
        f"ca_{label}_stderr_tail": res.get("stderr_tail", ""),
    }
    # Include per-stage bench breakdown (e.g. ca_gpu_bench_encoder_ms)
    for stage, ms in bench.items():
        d[f"ca_{label}_bench_{stage}_ms"] = ms
    return d


def _bench_transcribe_cpp(tc_gguf, timeout, label):
    """Run transcribe-cli and return a dict of prefixed result keys."""
    res = run_transcribe_cpp(TC_BIN, tc_gguf, JFK_WAV, timeout=timeout)
    norm = normalise(res["transcript"])
    wer = wer_simple(JFK_REF, norm)
    tc_infer_ms = res["mel_ms"] + res["encode_ms"] + res["decode_ms"]
    rtf = (round(tc_infer_ms / (JFK_DURATION_S * 1000), 3)
           if tc_infer_ms > 0 else round(res["infer_s"] / JFK_DURATION_S, 3))
    step(f"bench.tc_{label}_done", ok=res["ok"], rtf=rtf, wer=round(wer, 4),
         transcript=norm[:80])
    return {
        f"tc_{label}_ok": res["ok"],
        f"tc_{label}_transcript_norm": norm,
        f"tc_{label}_jfk_wer": round(wer, 4),
        f"tc_{label}_infer_s": res["infer_s"],
        f"tc_{label}_rtf": rtf,
        f"tc_{label}_mel_ms": res["mel_ms"],
        f"tc_{label}_encode_ms": res["encode_ms"],
        f"tc_{label}_decode_ms": res["decode_ms"],
        f"tc_{label}_load_ms": res["load_ms"],
        f"tc_{label}_stderr_tail": res.get("stderr_tail", ""),
    }


for m in SHARED_MODELS:
    fam = m["family"]
    if time.time() - _T0 > MAX_WALL_S:
        step("bench.wall_limit_reached", remaining_models=fam)
        break

    step(f"bench.start", family=fam)
    ca_gguf = MODELS_DIR / m["ca_file"]
    tc_gguf = MODELS_DIR / m["tc_file"]

    ca_dl = download_gguf(m["ca_url"], ca_gguf, timeout=600)
    # Download companion files (e.g. moonshine tokenizer.bin) into the same dir
    if m.get("ca_companion") and ca_dl:
        comp_dest = MODELS_DIR / Path(m["ca_companion"]).name
        download_gguf(m["ca_companion"], comp_dest, timeout=120)
    tc_dl = download_gguf(m["tc_url"], tc_gguf, timeout=600)

    row = {
        "family": fam,
        "notes": m["notes"],
        "ca_wer_libri_docs": m["ca_wer_libri"],
        "tc_wer_libri_docs": m["tc_wer_libri"],
        "ca_model": m["ca_file"],
        "tc_model": m["tc_file"],
    }

    # ── GPU runs ────────────────────────────────────────────────────────
    if ca_dl:
        row.update(_bench_crispasr(ca_gguf, m["ca_backend"], m["timeout_s"],
                                   no_gpu=False, label="gpu"))
    if tc_dl:
        row.update(_bench_transcribe_cpp(tc_gguf, m["timeout_s"], label="gpu"))

    # ── CPU runs (-ng / no CUDA) ────────────────────────────────────────
    if ca_dl:
        row.update(_bench_crispasr(ca_gguf, m["ca_backend"], m["timeout_s"],
                                   no_gpu=True, label="cpu"))
    if tc_dl:
        # transcribe.cpp: set CUDA_VISIBLE_DEVICES="" to force CPU
        _old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        row.update(_bench_transcribe_cpp(tc_gguf, m["timeout_s"], label="cpu"))
        if _old_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _old_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # ── Print side-by-side comparison ──────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {fam}")
    print(f"  Reference: {JFK_REF}")
    for mode in ["gpu", "cpu"]:
        ca_r = row.get(f"ca_{mode}_rtf")
        tc_r = row.get(f"tc_{mode}_rtf")
        ca_w = row.get(f"ca_{mode}_jfk_wer")
        tc_w = row.get(f"tc_{mode}_jfk_wer")
        ca_s = f"RTF={ca_r:.3f} WER={ca_w*100:.0f}%" if ca_r is not None else "FAIL"
        tc_s = f"RTF={tc_r:.3f} WER={tc_w*100:.0f}%" if tc_r is not None else "FAIL"
        print(f"  [{mode.upper()}] CA: {ca_s}  |  TC: {tc_s}")
    print(f"{'='*72}\n", flush=True)

    results.append(row)
    sweep_publish(f"shared__{fam.replace(' ', '_')}", row)

    ca_gguf.unlink(missing_ok=True)
    tc_gguf.unlink(missing_ok=True)
    if m.get("ca_companion"):
        (MODELS_DIR / Path(m["ca_companion"]).name).unlink(missing_ok=True)
    step(f"bench.cleanup", family=fam)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: transcribe.cpp-only models (CrispASR coverage gaps)
# ─────────────────────────────────────────────────────────────────────────────
tc_only_results = []
for m in TC_ONLY_MODELS:
    fam = m["family"]
    if time.time() - _T0 > MAX_WALL_S:
        step("bench.wall_limit_reached", remaining_models=fam)
        break
    step(f"bench.tc_only.start", family=fam)
    tc_gguf = MODELS_DIR / m["tc_file"]
    dl = download_gguf(m["tc_url"], tc_gguf, timeout=600)
    row = {
        "family": fam,
        "notes": m["notes"],
        "tc_wer_libri_docs": m["tc_wer_libri"],
        "tc_model": m["tc_file"],
        "ca_equivalent": None,
    }
    if dl:
        tc_res = run_transcribe_cpp(TC_BIN, tc_gguf, JFK_WAV, timeout=m["timeout_s"])
        tc_norm = normalise(tc_res["transcript"])
        tc_wer = wer_simple(JFK_REF, tc_norm)
        row.update({
            "tc_ok": tc_res["ok"],
            "tc_transcript": tc_res["transcript"],
            "tc_transcript_norm": tc_norm,
            "tc_jfk_wer": round(tc_wer, 4),
            "tc_infer_s": tc_res["infer_s"],
            "tc_rtf": round(tc_res["infer_s"] / JFK_DURATION_S, 3),
        })
        print(f"\n  {fam} (t.cpp only)")
        print(f"  t.cpp: {tc_norm}  [RTF={row['tc_rtf']}, WER={row['tc_jfk_wer']}]")
        step(f"bench.tc_only.done", family=fam, rtf=row["tc_rtf"], wer=row["tc_jfk_wer"])
    else:
        row.update({"tc_ok": False, "tc_jfk_wer": None, "tc_rtf": None})
        step(f"bench.tc_only.download_failed", family=fam)

    tc_only_results.append(row)
    sweep_publish(f"tc_only__{fam.replace(' ', '_')}", row)
    tc_gguf.unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Print final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 95)
print("BENCHMARK SUMMARY — CrispASR vs transcribe.cpp")
print(f"CUDA arch: {_cuda_arch or 'N/A'}  | Branch: {CRISPASR_BRANCH}")
print("=" * 95)
for mode in ["gpu", "cpu"]:
    print(f"\n  [{mode.upper()} mode]")
    print(f"  {'Family':<32} {'CA RTF':>8} {'TC RTF':>8} {'CA WER':>8} {'TC WER':>8}")
    print(f"  {'-'*72}")
    for row in results:
        ca_rtf = f"{row.get(f'ca_{mode}_rtf', ''):.3f}" if row.get(f"ca_{mode}_rtf") is not None else "FAIL"
        tc_rtf = f"{row.get(f'tc_{mode}_rtf', ''):.3f}" if row.get(f"tc_{mode}_rtf") is not None else "FAIL"
        ca_wer = f"{row.get(f'ca_{mode}_jfk_wer', 0)*100:.1f}%" if row.get(f"ca_{mode}_jfk_wer") is not None else "FAIL"
        tc_wer = f"{row.get(f'tc_{mode}_jfk_wer', 0)*100:.1f}%" if row.get(f"tc_{mode}_jfk_wer") is not None else "FAIL"
        print(f"  {row['family']:<32} {ca_rtf:>8} {tc_rtf:>8} {ca_wer:>8} {tc_wer:>8}")
print("=" * 95)

if tc_only_results:
    print("\ntranscribe.cpp-only models (CrispASR coverage gaps):")
    for row in tc_only_results:
        tc_rtf = f"{row.get('tc_rtf', ''):.3f}" if row.get("tc_rtf") is not None else "FAIL"
        tc_wer = f"{row.get('tc_jfk_wer', 0)*100:.1f}%" if row.get("tc_jfk_wer") is not None else "FAIL"
        print(f"  {row['family']:<40} RTF={tc_rtf}  WER={tc_wer}")

print(f"\nTotal elapsed: {round(time.time() - _T0, 0):.0f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Save and upload full summary JSON
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "run_tag": RUN_TAG,
    "branch": CRISPASR_BRANCH,
    "gpu": _cuda_arch,
    "elapsed_s": round(time.time() - _T0, 1),
    "shared_models": results,
    "tc_only_models": tc_only_results,
}
summary_path = WORK / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, default=str))
step("summary.written", path=str(summary_path))
sweep_publish("summary", summary)
step("bench.complete", total_models=len(results) + len(tc_only_results))
