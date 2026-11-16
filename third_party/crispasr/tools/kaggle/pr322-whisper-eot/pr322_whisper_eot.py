#!/usr/bin/env python3
"""PR #322 verification — whisper token_eot corrupted when specials are half-serialized.

THE BUG (verified by reading main): `set_token_id()` ASSIGNS its destination before
returning true, and it sits on the left of a short-circuit `&&`:

    const bool has_serialized_specials =
        set_token_id(vocab.token_eot, "<|endoftext|>") && set_token_id(vocab.token_sot, "<|startoftranscript|>");

So when `<|endoftext|>` is serialized but `<|startoftranscript|>` is not, the flag is
correctly false — but `token_eot` has already been overwritten. The legacy multilingual
fixup (`token_eot++; token_sot++`) then increments an ALREADY-RESOLVED id:

    defaults        eot=50256  sot=50257
    probe assigns   eot=50257            (from the serialized <|endoftext|>)
    fixup ++        eot=50258  sot=50258  <-- COLLISION, and sot is force-suppressed

`whisper_process_logits()` does `logits[token_sot] = -INFINITY` unconditionally, and the
timestamp rule masks `for (i = 0; i < token_eot; ++i)`, which now covers the REAL eot at
50257 — so the decoder can never end a segment at a timestamp.

WHICH MODELS: measured locally on eight whisper .bin files. The PREBUILT models from
ggerganov (what download-ggml-model.sh fetches) serialize 50257 entries and do NOT contain
`<|endoftext|>` at all -> unaffected. `.en` models have it at 50256 but are not multilingual
so the fixup never runs -> unaffected. The trigger is a MULTILINGUAL model whose serialized
vocab carries `<|endoftext|>`, which is what upstream's convert-h5-to-ggml.py produces:
openai/whisper-tiny's vocab.json has 50258 entries with `<|endoftext|>` at 50257 and no
`<|startoftranscript|>` (confirmed against the HF file).

WHAT THIS KERNEL DOES — the local box could not build (load average 31), so the runtime
half of the review runs here:

  1. build crispasr at main            (BEFORE)
  2. build crispasr at refs/pull/322/head (AFTER)   — same clone, warm ccache
  3. control  = prebuilt ggml-tiny.bin        (unaffected: no serialized eot)
  4. affected = the same file with `<|endoftext|>` appended as id 50257 and the vocab
     count bumped, weights untouched — reproducing the converter's output shape. The
     transform was verified locally: it yields exactly eot_ser=50257 / sot absent /
     multilingual, and the old-code arithmetic collides at 50258.
  5. decode samples/jfk.wav across the 2x2 matrix and compare.

VERDICT LOGIC (all three must hold):
  * control BEFORE == control AFTER      -> the fix regresses nothing
  * affected BEFORE != affected AFTER    -> the fix actually does something
  * affected AFTER == control AFTER      -> same weights now decode the same, i.e. FIXED

Follows the harness regime: clone in-kernel under /kaggle/temp (a script kernel does not
ship its siblings), read fixtures from the clone, heartbeat every long op, keep
/kaggle/working down to the artifacts we must retrieve.
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp/pr322")
TMP.mkdir(parents=True, exist_ok=True)
MODELS = Path("/kaggle/temp/models")
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "pr322_results.json"

HERE = Path(__file__).resolve().parent
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
CLONE = Path("/kaggle/temp/CrispASR")  # NOT /kaggle/working — see gotcha #22

if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--recurse-submodules", CRISPASR_URL, str(CLONE)],
                       check=True, timeout=1800)
    except Exception as e:  # noqa: BLE001
        print(f"clone failed: {e}", flush=True)

sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists() else str(HERE))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()


def sh(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def die(stage, **extra):
    kh.step(f"{stage}_FAILED", **extra)
    RESULTS.write_text(json.dumps({"verdict": "ERROR", "stage": stage, **extra}, indent=2))
    raise SystemExit(f"FATAL at {stage}: {extra}")


if not CLONE.exists():
    die("clone", err="CrispASR clone missing; a script kernel cannot rely on bundled files")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer", "huggingface_hub"],
               check=False)
HF_TOKEN = kh.resolve_hf_token()
kh.step("hf.token", present=bool(HF_TOKEN))
from huggingface_hub import hf_hub_download  # noqa: E402

# ── the two revisions ────────────────────────────────────────────────────────────
r = sh("git rev-parse HEAD", cwd=str(CLONE))
MAIN_SHA = r.stdout.strip()[:12]
r = sh("git fetch origin refs/pull/322/head:pr322", cwd=str(CLONE), timeout=900)
if r.returncode != 0:
    die("fetch.pr", err=r.stderr[-2000:])
PR_SHA = sh("git rev-parse pr322", cwd=str(CLONE)).stdout.strip()[:12]
kh.step("revs", main=MAIN_SHA, pr=PR_SHA)

# Show the diff being tested, so the log is self-contained.
diff = sh("git diff HEAD pr322 -- src/crispasr.cpp", cwd=str(CLONE)).stdout
print("=== PR #322 diff under test ===\n" + diff[:4000], flush=True)


def build(rev_label, git_ref):
    """Build the crispasr CLI at `git_ref`; return the binary path (copied aside)."""
    r = sh(f"git checkout --force {git_ref}", cwd=str(CLONE), timeout=600)
    if r.returncode != 0:
        die(f"checkout.{rev_label}", err=r.stderr[-2000:])
    if not (CLONE / "ggml" / "CMakeLists.txt").exists():
        sh("git submodule update --init ggml", cwd=str(CLONE), timeout=1200)
    # CPU-only: this is a vocab/decode-logic test, no GPU maths involved. The GPU is
    # enabled purely because Kaggle CPU workers get no internet (gotcha #3).
    flags = (["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON",
              "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DCRISPASR_BUILD_TESTS=OFF"]
             + kh.cache_and_link_flags())
    kh.step(f"build.configure.{rev_label}")
    with kh.build_heartbeat(f"build.cmake.{rev_label}", 30):
        r = sh("cmake -S . -B build -G Ninja " + " ".join(flags), cwd=str(CLONE), timeout=1800)
    if r.returncode != 0:
        die(f"build.configure.{rev_label}", err=r.stderr[-3000:])
    jobs = kh.safe_build_jobs(gpu=True)
    kh.step(f"build.compile.{rev_label}", jobs=jobs)
    with kh.build_heartbeat(f"build.ninja.{rev_label}", 30):
        try:
            kh.sh_with_progress(f"cmake --build build -j{jobs} --target crispasr-cli", cwd=str(CLONE))
        except Exception as e:  # noqa: BLE001
            die(f"build.compile.{rev_label}", err=str(e)[-3000:])
    src_bin = CLONE / "build" / "bin" / "crispasr"
    if not src_bin.exists():
        die(f"build.artifact.{rev_label}", err="crispasr binary not produced")
    # Copy the whole bin/ per revision, not just the executable: the SECOND build
    # overwrites build/bin, and if anything links a shared ggml the first binary
    # would silently start running the second revision's libraries.
    dst_dir = TMP / f"bin-{rev_label}"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(CLONE / "build" / "bin", dst_dir)
    dst = dst_dir / "crispasr"
    os.chmod(dst, 0o755)
    kh.step(f"build.ready.{rev_label}", size_mb=round(dst.stat().st_size / 1e6, 1))
    return dst


# ── models ───────────────────────────────────────────────────────────────────────
def vocab_facts(path):
    """Parse the whisper ggml header + vocab and report what drives the bug."""
    f = open(path, "rb")
    if f.read(4) != b"lmgg":
        return {"error": "not a whisper ggml file"}
    hp = struct.unpack("<11i", f.read(44))
    n_mel, n_fft = struct.unpack("<2i", f.read(8))
    f.seek(n_mel * n_fft * 4, 1)
    (ns,) = struct.unpack("<i", f.read(4))
    ids = {}
    for i in range(ns):
        (ln,) = struct.unpack("<I", f.read(4))
        w = f.read(ln)
        if w in (b"<|endoftext|>", b"<|startoftranscript|>", b"<|0.00|>"):
            ids[w.decode()] = i
    eot_ser = ids.get("<|endoftext|>")
    sot_ser = ids.get("<|startoftranscript|>")
    multiling = hp[0] >= 51865
    eot, sot = 50256, 50257
    if eot_ser is not None:
        eot = eot_ser                       # the old code's stray assignment
    if not (eot_ser is not None and sot_ser is not None) and multiling:
        eot += 1
        sot += 1
    return {"hparams_n_vocab": hp[0], "serialized": ns, "eot_serialized": eot_ser,
            "sot_serialized": sot_ser, "multilingual": multiling,
            "old_code_eot": eot, "old_code_sot": sot, "old_code_collides": eot == sot}


def make_affected(src, dst):
    """Append `<|endoftext|>` as the next vocab id, bump the count, keep weights verbatim.

    Reproduces what upstream's convert-h5-to-ggml.py emits for a multilingual model
    (openai/whisper-tiny's vocab.json: 50258 entries, <|endoftext|> at 50257, no
    <|startoftranscript|>). Verified locally before this kernel was written.
    """
    f = open(src, "rb")
    assert f.read(4) == b"lmgg"
    f.seek(44, 1)
    n_mel, n_fft = struct.unpack("<2i", f.read(8))
    f.seek(n_mel * n_fft * 4, 1)
    off = f.tell()
    (n,) = struct.unpack("<i", f.read(4))
    for _ in range(n):
        (ln,) = struct.unpack("<I", f.read(4))
        f.seek(ln, 1)
    end = f.tell()
    raw = open(src, "rb").read()
    tok = b"<|endoftext|>"
    out = (raw[:off] + struct.pack("<i", n + 1) + raw[off + 4:end]
           + struct.pack("<I", len(tok)) + tok + raw[end:])
    open(dst, "wb").write(out)
    return dst


SEG_RE = re.compile(r"\[([\d:.]+)\s*-->\s*([\d:.]+)\]\s*(.*)")


def decode(binary, model, wav, print_special=False):
    env_prefix = f"LD_LIBRARY_PATH={Path(binary).parent}:$LD_LIBRARY_PATH "
    ps = " -ps" if print_special else ""
    r = sh(env_prefix + f"{binary} -m {model} -f {wav} -l en -t 4{ps}", timeout=2400)
    segs = []
    for line in r.stdout.splitlines():
        m = SEG_RE.search(line)
        if m:
            segs.append({"t0": m.group(1), "t1": m.group(2), "text": m.group(3).strip()})
    return {"rc": r.returncode, "n_segments": len(segs), "segments": segs,
            "text": " ".join(s["text"] for s in segs).strip(),
            "stderr_tail": r.stderr[-600:] if r.returncode != 0 else ""}


def main():
    results = {"main_sha": MAIN_SHA, "pr_sha": PR_SHA, "diff": diff[:4000]}

    with kh.build_heartbeat("models.download", 30):
        control = hf_hub_download(repo_id="ggerganov/whisper.cpp", filename="ggml-tiny.bin",
                                  local_dir=str(MODELS), token=HF_TOKEN or None)
    affected = make_affected(control, str(MODELS / "ggml-tiny-affected.bin"))
    results["control_vocab"] = vocab_facts(control)
    results["affected_vocab"] = vocab_facts(affected)
    kh.step("models.ready", control=results["control_vocab"], affected=results["affected_vocab"])

    # The whole premise: the control must NOT collide and the affected MUST.
    if results["affected_vocab"].get("old_code_collides") is not True:
        die("premise", msg="constructed model does not reproduce the bug condition",
            facts=results["affected_vocab"])
    if results["control_vocab"].get("old_code_collides") is not False:
        die("premise", msg="control model unexpectedly collides", facts=results["control_vocab"])

    # Fixture choice is the whole experiment. A single-utterance 11 s clip fits one
    # 30 s window and terminates on end-of-audio, so a broken EOT changes NOTHING —
    # run 1 of this kernel produced byte-identical output on all four cells and the
    # "fix has an effect" check failed for that reason, not because the bug is absent.
    # multispeaker.wav is 31.5 s across two windows with several utterances, so the
    # decoder must actually end segments at timestamps — the path the bug breaks.
    FIXTURES = [("jfk", CLONE / "samples" / "jfk.wav"),
                ("multispeaker", CLONE / "samples" / "multispeaker.wav")]
    for nm, pth in FIXTURES:
        if not pth.exists():
            die("fixture", err=f"{pth} missing from the clone")

    # Once, not per build: a second call re-extracts the seed tar over the cache the
    # first build just populated.
    kh.install_build_toolchain()

    bins = {}
    for label, ref in (("before", MAIN_SHA), ("after", "pr322")):
        bins[label] = build(label, ref)

    for label in ("before", "after"):
        for mname, mpath in (("control", control), ("affected", affected)):
            for fname, fpath in FIXTURES:
                key = f"{mname}_{label}_{fname}"
                with kh.build_heartbeat(f"decode.{key}", 30):
                    results[key] = decode(bins[label], mpath, str(fpath))
                kh.step(f"decode.{key}", rc=results[key]["rc"],
                        n_segments=results[key]["n_segments"],
                        text=results[key]["text"][:100])

    # Special-token stream on the affected model: shows the EOT/SOT handling directly
    # rather than inferring it from the transcript.
    for label in ("before", "after"):
        key = f"affected_{label}_multispeaker_ps"
        with kh.build_heartbeat(f"decode.{key}", 30):
            results[key] = decode(bins[label], affected,
                                  str(CLONE / "samples" / "multispeaker.wav"), print_special=True)

    per_fixture = {}
    for fname, _ in FIXTURES:
        cb = results[f"control_before_{fname}"]
        ca = results[f"control_after_{fname}"]
        ab = results[f"affected_before_{fname}"]
        aa = results[f"affected_after_{fname}"]
        per_fixture[fname] = {
            "control_unchanged": cb["segments"] == ca["segments"],
            "affected_changed": ab["segments"] != aa["segments"],
            "affected_matches_control": aa["segments"] == ca["segments"],
            "all_ok": all(x["rc"] == 0 for x in (cb, ca, ab, aa)),
            "n_segments": {"control": ca["n_segments"], "affected_before": ab["n_segments"],
                           "affected_after": aa["n_segments"]},
        }
    results["per_fixture"] = per_fixture

    no_regression = all(f["control_unchanged"] and f["affected_matches_control"]
                        for f in per_fixture.values())
    all_ok = all(f["all_ok"] for f in per_fixture.values())
    demonstrated = any(f["affected_changed"] for f in per_fixture.values())

    if not all_ok:
        results["verdict"] = "ERROR"
    elif not no_regression:
        results["verdict"] = "REGRESSION"
    elif demonstrated:
        results["verdict"] = "FIXED"          # observable decode difference, corrected
    else:
        results["verdict"] = "LATENT"         # safe, but no fixture exposed a difference
    results["verdict_note"] = {
        "FIXED": "affected model decoded differently before/after and now matches the control",
        "LATENT": "no regression anywhere, but no fixture produced an observable difference",
        "REGRESSION": "the fix changed something it should not have",
        "ERROR": "a decode run failed",
    }[results["verdict"]]

    # Persist BEFORE printing: v2 computed everything correctly and then died in a
    # leftover print loop, so the run showed ERROR despite valid results on disk.
    RESULTS.write_text(json.dumps(results, indent=2))
    print("\n=== PR #322 VERDICT:", results["verdict"], "===", flush=True)
    print("   ", results["verdict_note"], flush=True)
    for fname, f in per_fixture.items():
        print(f"\n  [{fname}] segments control={f['n_segments']['control']} "
              f"affected_before={f['n_segments']['affected_before']} "
              f"affected_after={f['n_segments']['affected_after']}", flush=True)
        for k, v in f.items():
            if isinstance(v, bool):
                print(f"      {'PASS' if v else 'FAIL'}  {k}", flush=True)
    for key in sorted(k for k in results if k.startswith(("control_", "affected_"))):
        rr = results[key]
        if not isinstance(rr, dict) or "segments" not in rr:
            continue
        print(f"\n--- {key}: rc={rr['rc']} segments={rr['n_segments']}", flush=True)
        for sg in rr["segments"][:8]:
            print(f"    [{sg['t0']} --> {sg['t1']}] {sg['text'][:90]}", flush=True)
    kh.step("verdict", result=results["verdict"], per_fixture=per_fixture)

    RESULTS.write_text(json.dumps(results, indent=2))

    try:
        kh.export_ccache_tar()
    except Exception:  # noqa: BLE001
        pass


main()
