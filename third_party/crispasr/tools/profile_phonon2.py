"""Load-excluded, warmed Phonon-2 measurements with transcript proof of work.

Run runtime and reference in separate processes, sequentially on the same host.
Scheduler traces belong in a separate run: callbacks perturb dispatch timings.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tools")]
EXPECTED = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."


def words(text):
    return re.findall(r"\w+(?:'\w+)?", text.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["runtime", "reference"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lib")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true", help="one short inference; timings are diagnostic only")
    args = parser.parse_args()
    if args.repeat < 3 and not args.trace:
        parser.error("use at least three measurements")
    if not args.trace and any(os.environ.get(k, "0") != "0" for k in
                              ("CRISPASR_SCHED_PROFILE", "CRISPASR_FC_PROFILE", "CRISPASR_PARAKEET_BENCH",
                               "CRISPASR_PARAKEET_ENC_PROBE", "CRISPASR_PARAKEET_DECODE_TIMING",
                               "CRISPASR_PARAKEET_FFN_TRACE")):
        parser.error("disable instrumentation for benchmark measurements")
    with wave.open(str(ROOT / "samples/jfk.wav")) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    loaded = time.perf_counter()
    session = None
    if args.engine == "runtime":
        from crispasr import Session
        session = Session(args.model, lib_path=args.lib, n_threads=args.threads, backend="phonon2")
        assert session.backend == "parakeet"
        def transcribe(pcm):
            return " ".join(segment.text for segment in session.transcribe(pcm)).strip()
    else:
        import torch
        from transformers import AutoTokenizer, ParakeetFeatureExtractor
        from reference_backends import parakeet_hf
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        model, base = parakeet_hf._load(Path(args.model))
        feature = ParakeetFeatureExtractor(feature_size=model.config.encoder_config.num_mel_bins)
        tokenizer = AutoTokenizer.from_pretrained(base)
        def transcribe(pcm):
            inputs = feature(pcm, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                generated = model.generate(**inputs)
            return tokenizer.batch_decode(getattr(generated, "sequences", generated), skip_special_tokens=True)[0].strip()
    report = {"engine": args.engine, "model": args.model, "host": platform.platform(),
              "threads": args.threads, "load_s": time.perf_counter() - loaded,
              "trace_only": args.trace, "clips": [],
              "controls": {key: os.environ.get(key) for key in (
                  "CRISPASR_PARAKEET_FFN", "CRISPASR_PARAKEET_FFN_BLAS_THREADS",
                  "CRISPASR_PARAKEET_CPU_BLAS", "CRISPASR_PARAKEET_GGML_DECODE",
                  "CRISPASR_PARAKEET_ENCODER_BLAS", "CRISPASR_PARAKEET_ENCODER_BLAS_THREADS",
                  "CRISPASR_RNNT_GPU_ENC_PROJ", "CRISPASR_PARAKEET_FORCE_SCALAR",
                  "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}
    try:
        for count in ([1] if args.trace else [1, 5]):
            pcm = np.tile(audio, count)
            reference = transcribe(pcm)  # shape-specific warmup, excluded
            assert reference and abs(len(words(reference)) / len(words(EXPECTED)) - count) <= 0.5, reference
            if count == 1:
                assert words(reference) == words(EXPECTED), reference
            timings = []
            for _ in range(1 if args.trace else args.repeat):
                start = time.perf_counter()
                text = transcribe(pcm)
                elapsed = time.perf_counter() - start
                assert text == reference, (reference, text)
                timings.append(elapsed)
            median = statistics.median(timings)
            row = {"audio_s": len(pcm) / 16000, "seconds": timings, "median_s": median,
                   "realtime_factor": median / (len(pcm) / 16000),
                   "realtime_speed": (len(pcm) / 16000) / median,
                   "transcript": reference, "word_count": len(words(reference))}
            report["clips"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        if session:
            session.close()
    if args.engine == "runtime" and args.lib:
        # Query the linked BLAS, rather than assuming its thread count still
        # equals the startup environment (the ggml BLAS backend can change it).
        import ctypes
        try:
            getter = ctypes.CDLL(args.lib).openblas_get_num_threads
            getter.argtypes = []
            getter.restype = ctypes.c_int
            report["openblas_threads_after_inference"] = getter()
        except (AttributeError, OSError):
            pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mb"] = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
