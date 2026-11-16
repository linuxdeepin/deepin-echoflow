#!/usr/bin/env python3
"""Speaker-count estimator A/B on VoxConverse dev — BIC+silhouette vs NME-SC.

Why on Kaggle: the sweep is 200+ diarization runs over 20 h of audio. The dev
box it was written on sat between load 13 and 197 all session and killed the
run twice, so the numbers that matter never got produced.

What it measures: speaker-COUNT accuracy first, DER second. The estimator was
found wrong on 4 of 8 files while DER still averaged 7.32%, because merging two
speakers only costs the frames of the one absorbed — so DER alone cannot see
this failure. See tools/diarize_eval.py.

Runs a 40-file SUBSET of the TUNE split. Holdout is deliberately not computed:
not calculating it is a stronger guarantee than calculating it and promising not
to look. If the subset shows an effect, the full tune split is the next run.

Kernel notes (tools/../kaggle_usage.md):
  * GPU is enabled ONLY because Kaggle CPU workers get no internet (#3) and a
    script kernel cannot import bundled siblings (#26) — the repo has to be
    cloned. The work itself is CPU-bound.
  * Repo goes to /kaggle/temp, not /kaggle/working, so `kernels output` is not
    page-capped past the artifacts we need (#22).
  * Long phases are wrapped in kh.build_heartbeat so Kaggle does not idle-kill
    a silent build or sweep (#26).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)

CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = TEMP / "CrispASR"

# ── clone + harness (must come from the clone, not a bundled sibling) ────────
if not REPO.exists():
    # --recursive: the build needs the bundled ggml submodule, and a plain
    # --depth 1 clone leaves cmake dying on a missing ggml/CMakeLists.txt.
    subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", CRISPASR_URL, str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("clone", status="ok", repo=str(REPO))

token = kh.resolve_hf_token()
kh.step("hf_token", status="ok" if token else "MISSING")

# ── build (CPU only; the diarizer never uses the GPU) ────────────────────────
with kh.build_heartbeat("build", 30):
    kh.install_build_toolchain()
    build = REPO / "build"
    # cache_and_link_flags() folds in ccache/mold AND -DCRISPASR_NO_C2PA_NATIVE
    # (the c2pa-audio submodule is irrelevant here and breaks generate).
    subprocess.check_call(
        ["cmake", "-S", str(REPO), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
         "-DCRISPASR_BUILD_TESTS=OFF", "-DGGML_CUDA=OFF"] + kh.cache_and_link_flags(),
    )
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {build} --target crispasr-cli -j{kh.safe_build_jobs(gpu=False)}"
    )
cli = build / "bin" / "crispasr"
if not cli.exists():
    raise SystemExit(f"build produced no {cli}")
kh.step("build", status="ok")

# ── corpus: HF parquet -> wav/ + ref.json ────────────────────────────────────
from huggingface_hub import hf_hub_download  # noqa: E402

CORPUS = TEMP / "corpus"
with kh.build_heartbeat("corpus", 30):
    for i in range(5):
        p = hf_hub_download(
            repo_id="diarizers-community/voxconverse",
            filename=f"data/dev-0000{i}-of-00005.parquet",
            repo_type="dataset",
            token=token,
            cache_dir=str(TEMP / "hf"),
        )
        subprocess.check_call([sys.executable, str(REPO / "tools" / "voxconverse_extract.py"),
                               "--parquet", p, "--out", str(CORPUS)])
ref = json.load(open(CORPUS / "ref.json"))
kh.step("corpus", status="ok", files=len(ref))

# ── models ──────────────────────────────────────────────────────────────────
asr = hf_hub_download(repo_id="ggerganov/whisper.cpp", filename="ggml-tiny.bin",
                      token=token, cache_dir=str(TEMP / "hf"))
emb = hf_hub_download(repo_id="cstr/wespeaker-resnet34-lm-GGUF",
                      filename="wespeaker-resnet34-lm.gguf",
                      token=token, cache_dir=str(TEMP / "hf"))
kh.step("models", status="ok")

# ── the A/B ─────────────────────────────────────────────────────────────────
# --diarize-max-speakers stays at the SHIPPING DEFAULT of 8, not 20.
#
# A first run used 20 so that files with up to 20 speakers were countable. That
# made the silhouette scan evaluate 19 candidate k instead of 7, each an O(n^2)
# eigendecomposition, and the longest tune file has ~1800 embedding windows —
# the BIC arm alone took 6.4 h. Two reasons 8 is the better measurement anyway:
# it is what users actually get, and the cap applies identically to BOTH arms so
# the comparison stays fair. diarize_eval.py flags the files the cap makes
# unwinnable rather than letting them read as model error.
# Threads per file x concurrent files must not exceed the box. The first run
# used -t cpu_count WITH --jobs 2 on a 4-CPU worker: 8 threads on 4 cores,
# every file fighting the other for the same cores.
NCPU = os.cpu_count() or 4
JOBS = 2
THREADS = max(1, NCPU // JOBS)

CMD = (
    f"{cli} -m {asr} -f {{wav}} -t {THREADS} --diarize "
    f"--diarize-method foxnose --diarize-embedder {emb} "
    f"-oj -of {{out}}"
)
# SUBSET of the tune split, not all 101. A full arm cost 6.2 h and returned
# nothing; if NME-SC has a real effect on speaker counting it will show on 40
# files, and learning that in an hour beats learning nothing in seven. The
# subset is the FIRST 40 tune files by name, so it is deterministic and the
# same 40 in both arms.
SUBSET = int(os.environ.get("DIARIZE_EVAL_SUBSET", "40"))

BASE = [
    sys.executable, str(REPO / "tools" / "diarize_eval.py"),
    "--wav-dir", str(CORPUS / "wav"), "--ref", str(CORPUS / "ref.json"),
    "--workdir", str(TEMP / "evalwork"), "--jobs", str(JOBS),
    "--max-speakers", "8", "--split", "tune", "--subset", str(SUBSET),
    "--cmd", CMD,
]

summary = {}
for arm, env_extra in (("bic", {}), ("nme-sc", {"CRISPASR_DIARIZE_COUNT": "nme-sc"})):
    env = dict(os.environ)
    env.update(env_extra)
    out_json = WORK / f"eval_{arm}.json"
    # NOT capture_output. The first run buffered 101 files of output into a
    # variable that was only written after the arm returned — the arm failed,
    # the kernel was killed, and six hours produced heartbeats and no reason.
    # Tee to the kernel log (live) AND to a file (retrievable).
    log_path = WORK / f"eval_{arm}.txt"
    with kh.build_heartbeat(f"eval:{arm}", 30), open(log_path, "w", buffering=1) as lf:
        proc = subprocess.Popen(BASE + ["--json-out", str(out_json)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
        captured = []
        for line in proc.stdout:
            print(f"[{arm}] {line.rstrip()}", flush=True)
            lf.write(line)
            captured.append(line)
        rc = proc.wait()
    tune = [l for l in captured if l.startswith("tune ")]
    summary[arm] = tune[0].rstrip() if tune else f"NO RESULT (rc={rc})"
    kh.step(f"eval:{arm}", status="ok" if tune else "FAILED", line=summary[arm])
    print(f"[{arm}] {summary[arm]}", flush=True)

(WORK / "summary.json").write_text(json.dumps(summary, indent=1))
print("\n=== TUNE split, speaker-count accuracy is the metric ===")
for arm, line in summary.items():
    print(f"{arm:8} {line}")
kh.step("done", status="ok")
