"""#300: validate streaming speaker-diarization label passthrough on Kaggle.

Follows the CrispASR harness regime: build the ACTUAL commits from source
(no stale release binary), run the DECODED-OUTPUT roundtrip (HARD RULE #3),
and prove the work rather than trusting exit codes (gotcha #24).

The change under test (commit a8e9ef8d) surfaces a backend's structured
`seg.speaker` field in streaming output — inline in plain `--stream` and as a
`"speaker"` field on single-speaker `--stream-json` `final` events. Before it,
the streaming emit path built text from `seg.text` only and dropped
`seg.speaker`.

Ground-truth A/B on the SAME backend (moss-diarize, which natively populates
`seg.speaker`):
    BASE = d4f2824b  (parent of #300)  → streamed output has NO structured labels
    FIX  = a8e9ef8d  (#300)            → streamed output HAS structured labels

Plus:
  * FIX file-mode moss-diarize      → labels present (model actually diarizes)
  * FIX --stream-json moss-diarize  → a `final` event carries a "speaker" field
  * FIX --stream whisper (control)  → NO structured labels (change is gated)
  * FIX moss-diarize on a NON-diarizing observation for vibevoice-asr: its
    adapter never sets seg.speaker (speaker info is inline transcript text), so
    the structured-label change is a no-op for it — recorded, not asserted.

Structured moss label form is "(Speaker N) " (capital S, parenthesised); the
regex \(Speaker \d+\) matches ONLY that, not a model's inline "Speaker 0" text
or the word "speaker" — so a positive count means the structured field flowed
through, which is exactly the code under test.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

TEMP = Path("/kaggle/temp"); OUT = Path("/kaggle/working")
REPO = TEMP / "CrispASR"; MODELS = TEMP / "models"
for d in (TEMP, OUT, MODELS):
    d.mkdir(parents=True, exist_ok=True)

BASE = "d4f2824b"  # parent of #300 — old streaming emit (drops seg.speaker)
FIX = "a8e9ef8d"   # #300 — surfaces seg.speaker in --stream / --stream-json

SPK_RE = re.compile(r"\(Speaker \d+\)")  # moss structured label ONLY


def _eh(et, ev, tb):
    try:
        (OUT / "error.txt").write_text("".join(traceback.format_exception(et, ev, tb)))
    except Exception:
        pass
    sys.__excepthook__(et, ev, tb)


sys.excepthook = _eh


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


print(json.dumps({"step": "start"}), flush=True)

# ── Clone (full enough to reach BASE) + submodules ───────────────────────────
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "15", "https://github.com/CrispStrobe/CrispASR.git",
     str(REPO)], capture_output=False)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive", "--depth", "1"],
    capture_output=False, timeout=1800)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("cloned")

# Confirm both commits are in the shallow history (fail loud if not).
for c in (BASE, FIX):
    r = run(["git", "-C", str(REPO), "cat-file", "-t", c])
    if r.returncode or "commit" not in (r.stdout or ""):
        kh.step("commit.MISSING", commit=c)
        (OUT / "results.json").write_text(json.dumps({"error": f"commit {c} not in shallow clone"}))
        raise SystemExit(1)
kh.step("commits.present", base=BASE, fix=FIX)

kh.install_build_toolchain()
run(["apt-get", "install", "-y", "-q", "libopenblas-dev", "ffmpeg"], capture_output=False)
JOBS = str(min(4, os.cpu_count() or 2))


def build_at(commit, bdir):
    run(["git", "-C", str(REPO), "checkout", "-q", commit], capture_output=False)
    r = run(["cmake", "-G", "Ninja", "-B", str(bdir), "-S", str(REPO),
             "-DCMAKE_BUILD_TYPE=Release"] + kh.cache_and_link_flags(), capture_output=False)
    if r.returncode:
        kh.step(f"cfg.FAIL.{commit}")
        raise SystemExit(1)
    with kh.build_heartbeat(f"build.{commit}"):
        r = run(["cmake", "--build", str(bdir), "--target", "crispasr-cli", "-j", JOBS],
                capture_output=False)
    if r.returncode:
        kh.step(f"build.FAIL.{commit}")
        raise SystemExit(1)
    cli = bdir / "bin" / "crispasr"
    if not cli.exists():
        c = [p for p in bdir.rglob("crispasr") if p.is_file() and os.access(p, os.X_OK)]
        cli = c[0] if c else None
    if cli is None:
        kh.step(f"MISSING.{commit}")
        raise SystemExit(1)
    return cli


# BASE first, then FIX (delta rebuild — only crispasr_run.cpp differs → fast).
BASE_DIR = TEMP / "b-base"; FIX_DIR = TEMP / "b-fix"
base_cli = build_at(BASE, BASE_DIR); kh.step("build.base.done", cli=str(base_cli))
fix_cli = build_at(FIX, FIX_DIR); kh.step("build.fix.done", cli=str(fix_cli))

# ── Sample: multispeaker.wav (ships in repo) → raw s16le 16k mono for --stream ─
WAV = REPO / "samples" / "multispeaker.wav"
assert WAV.is_file(), f"missing sample: {WAV}"
PCM = TEMP / "multispeaker.s16le"
run(["ffmpeg", "-y", "-i", str(WAV), "-f", "s16le", "-ar", "16000", "-ac", "1", str(PCM)],
    capture_output=False)
assert PCM.is_file() and PCM.stat().st_size > 0, "ffmpeg produced no PCM"
kh.step("pcm.ready", bytes=PCM.stat().st_size)

# ── Models (public HF; no token needed). Stage under /kaggle/temp (gotcha #18). ─
from huggingface_hub import hf_hub_download  # noqa: E402

kh.step("dl.moss.begin")
MOSS = Path(hf_hub_download(repo_id="cstr/MOSS-Transcribe-Diarize-GGUF",
                            filename="moss-transcribe-diarize-0.9b-q4_k.gguf",
                            local_dir=str(MODELS)))
kh.step("dl.moss.done", gb=round(MOSS.stat().st_size / 1e9, 2))


def _env(bdir):
    return {**os.environ, "LD_LIBRARY_PATH": f"{bdir}/src:" + os.environ.get("LD_LIBRARY_PATH", "")}


def _summ(out):
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return {
        "labels": len(SPK_RE.findall(out)),       # structured (Speaker N) count
        "chars": len(out),
        "nonempty_lines": len(lines),
        "tail": "\n".join(lines[-4:])[-600:],
    }


def run_file(cli, bdir, backend, model, timeout=1800):
    t0 = time.time()
    r = run([str(cli), "-m", str(model), "-f", str(WAV), "--backend", backend],
            env=_env(bdir), timeout=timeout)
    s = _summ(r.stdout or ""); s.update(exit=r.returncode, wall=round(time.time() - t0, 1),
                                        stderr_tail=(r.stderr or "")[-400:] if r.returncode else "")
    return s


def run_stream(cli, bdir, backend, model, extra=None, timeout=1800):
    cmd = [str(cli), "--stream", "--backend", backend, "-m", str(model)] + (extra or [])
    t0 = time.time()
    with open(PCM, "rb") as fin:
        r = run(cmd, env=_env(bdir), stdin=fin, timeout=timeout)
    s = _summ(r.stdout or ""); s.update(exit=r.returncode, wall=round(time.time() - t0, 1),
                                        stderr_tail=(r.stderr or "")[-400:] if r.returncode else "")
    # For --stream-json: count `final` events that carry a "speaker" field.
    finals = 0; finals_with_spk = 0
    for ln in (r.stdout or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        if ev.get("type") == "final":
            finals += 1
            if ev.get("speaker"):
                finals_with_spk += 1
    s.update(json_finals=finals, json_finals_with_speaker=finals_with_spk)
    return s


R = {}
# T1: FIX file-mode moss — model actually diarizes (baseline).
R["moss_fix_file"] = run_file(fix_cli, FIX_DIR, "moss-diarize", MOSS); kh.step("t1", **R["moss_fix_file"])
# T2: BASE --stream moss — OLD emit drops seg.speaker → expect 0 structured labels.
R["moss_base_stream"] = run_stream(base_cli, BASE_DIR, "moss-diarize", MOSS); kh.step("t2", **R["moss_base_stream"])
# T3: FIX --stream moss — NEW emit surfaces seg.speaker → expect >0 (THE change).
R["moss_fix_stream"] = run_stream(fix_cli, FIX_DIR, "moss-diarize", MOSS); kh.step("t3", **R["moss_fix_stream"])
# T4: FIX --stream-json + VAD moss — expect a `final` event with a "speaker" field.
R["moss_fix_streamjson"] = run_stream(
    fix_cli, FIX_DIR, "moss-diarize", MOSS,
    extra=["--stream-json", "--vad", "--stream-final-on-silence-ms", "800"]); kh.step("t4", **R["moss_fix_streamjson"])

# Persist partial results NOW so a later crash/timeout can't lose the core A/B.
(OUT / "results.json").write_text(json.dumps({"base": BASE, "fix": FIX, "results": R}, indent=2))

# T5: control — whisper --stream (FIX) must NOT inject structured labels.
try:
    R["whisper_fix_stream"] = run_stream(fix_cli, FIX_DIR, "whisper", "auto", timeout=1200)
    kh.step("t5", **R["whisper_fix_stream"])
except Exception as e:
    R["whisper_fix_stream"] = {"error": f"{type(e).__name__}: {e}"}
    kh.step("t5.err", error=str(e))
(OUT / "results.json").write_text(json.dumps({"base": BASE, "fix": FIX, "results": R}, indent=2))

# T6: vibevoice observation (best-effort, time/disk permitting). Its adapter
# never sets seg.speaker → structured label count expected 0 even on FIX; any
# "speaker" info it shows is inline transcript text (recorded, not asserted).
try:
    free_gb = shutil.disk_usage(str(TEMP)).free / 1e9
    remaining = 8 * 3600 - (time.time() - kh._T0 if hasattr(kh, "_T0") else 0)
    if free_gb >= 6.0:
        kh.step("dl.vibe.begin", free_gb=round(free_gb, 1))
        VIBE = Path(hf_hub_download(repo_id="cstr/vibevoice-asr-GGUF",
                                    filename="vibevoice-asr-q4_k.gguf", local_dir=str(MODELS)))
        kh.step("dl.vibe.done", gb=round(VIBE.stat().st_size / 1e9, 2))
        R["vibe_fix_file"] = run_file(fix_cli, FIX_DIR, "vibevoice", VIBE, timeout=2400)
        kh.step("t6.file", **R["vibe_fix_file"])
        R["vibe_fix_stream"] = run_stream(fix_cli, FIX_DIR, "vibevoice", VIBE, timeout=2400)
        kh.step("t6.stream", **R["vibe_fix_stream"])
    else:
        R["vibe_skipped"] = {"reason": f"disk {free_gb:.1f} GB"}
        kh.step("t6.skip", free_gb=round(free_gb, 1))
except Exception as e:
    R["vibe_error"] = {"error": f"{type(e).__name__}: {e}"}
    kh.step("t6.err", error=str(e))

# ── Verdict ──────────────────────────────────────────────────────────────────
def _ok(k):
    return isinstance(R.get(k), dict) and R[k].get("exit") == 0


verdict = {}
verdict["moss_diarizes_file"] = _ok("moss_fix_file") and R["moss_fix_file"]["labels"] > 0
verdict["base_stream_no_labels"] = _ok("moss_base_stream") and R["moss_base_stream"]["labels"] == 0 \
    and R["moss_base_stream"]["nonempty_lines"] > 0  # non-empty ⇒ it actually ran (gotcha #24)
verdict["fix_stream_has_labels"] = _ok("moss_fix_stream") and R["moss_fix_stream"]["labels"] > 0
verdict["fix_streamjson_speaker_field"] = _ok("moss_fix_streamjson") \
    and R["moss_fix_streamjson"].get("json_finals_with_speaker", 0) > 0
verdict["control_whisper_no_labels"] = (not _ok("whisper_fix_stream")) or \
    R["whisper_fix_stream"].get("labels", 0) == 0

# The A/B core: label count strictly increased BASE→FIX on the SAME backend.
core = (verdict["base_stream_no_labels"] and verdict["fix_stream_has_labels"]
        and verdict["fix_streamjson_speaker_field"])
verdict["PASS"] = bool(core)

payload = {"base": BASE, "fix": FIX, "verdict": verdict, "results": R}
(OUT / "results.json").write_text(json.dumps(payload, indent=2))

print("\n=== #300 STREAMING DIARIZATION VALIDATION ===", flush=True)
for k in ("moss_fix_file", "moss_base_stream", "moss_fix_stream", "moss_fix_streamjson",
          "whisper_fix_stream", "vibe_fix_stream"):
    v = R.get(k)
    if isinstance(v, dict):
        print(f"  {k:22s} exit={v.get('exit')} labels={v.get('labels')} "
              f"json_final_spk={v.get('json_finals_with_speaker')} lines={v.get('nonempty_lines')} "
              f"wall={v.get('wall')}s", flush=True)
print("  --- verdict ---", flush=True)
for k, val in verdict.items():
    print(f"  {k:32s} {val}", flush=True)
print(f"\n  OVERALL: {'PASS ✅' if verdict['PASS'] else 'FAIL ❌'}", flush=True)

kh.step("done", **verdict)
print(json.dumps({"step": "done", "PASS": verdict["PASS"]}), flush=True)
if not verdict["PASS"]:
    raise SystemExit(2)
