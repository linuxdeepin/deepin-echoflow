# %% [markdown]
# # CrispASR — madlad400-3b-mt: the missing F16/Q8_0 + a per-stage reference (#333)
#
# `cstr/madlad400-3b-mt-GGUF` ships ONLY `madlad400-3b-mt-q4_k.gguf`, while its
# README lists F16 and Q8_0 and its own copy-paste quickstart tells you to
# download `…-q8_0.gguf` — which 404s. That is what #333 reports.
#
# This does more than fill the gap. madlad had no diff-harness coverage at all
# until now, so every published quant was "it loaded and the text looked fine".
# This run produces the reference archive and reports **per-stage cosine parity
# for all three quants against the PyTorch blueprint**, so the answer on the
# issue can be a number instead of an assurance.
#
#     download source (11.76 GB fp32)
#       → convert F16 → validate → upload
#       → dump madlad-ref.gguf (needs the source, so BEFORE deleting it) → upload
#       → rm source
#       → quantize Q8_0 → validate → upload
#       → crispasr-diff F16 / Q8_0 / Q4_K vs the reference → parity table
#
# Ordering is the port-pipeline rule "never crash before a produced artifact is
# checkpointed to HF": each artifact is uploaded the moment it exists and is
# validated, so a failure in a later step cannot lose an earlier one.
#
# It BUILDS FROM SOURCE rather than using a release tarball: the madlad arm in
# `crispasr-diff` and `tools/reference_backends/madlad.py` are newer than
# v0.8.25. That is what `${KAGGLE_ACCOUNT}/crispasr-ccache` is attached for — a warm
# ccache turns a ~20 min build into ~3 min.

# %% [code]
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)

# ── Kaggle regime: clone + harness from the CLONE, not from bundled siblings ──
# gotcha #26: a script kernel runs only its code_file. gotcha #22: keep
# /kaggle/working tiny so `kernels output` isn't page-capped past our artifacts.
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
REPO = TEMP / "CrispASR"
if not REPO.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", "--recursive",
                               CRISPASR_URL, str(REPO)])
        sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    except Exception:
        pass
if str(REPO / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
step = kh.step
step("script.start", issue=333)

CONVERTER = REPO / "models" / "convert-madlad-to-gguf.py"
DUMPER = REPO / "tools" / "dump_reference.py"
for f in (CONVERTER, DUMPER):
    if not f.is_file():
        step("fatal.clone-incomplete", missing=str(f))
        raise SystemExit(f"{f} missing — the clone failed; refusing to continue (HARD RULE #8)")

TOKEN = kh.resolve_hf_token("HF_TOKEN")
if not TOKEN:
    step("fatal.no-token")
    raise SystemExit("no HF token — every upload would fail after ~40 min of work")
step("hf_token.resolved")

step("install-deps.begin")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub", "hf_transfer", "sentencepiece", "safetensors", "gguf"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

step("install-deps.done")

# ── build (warm ccache from the attached dataset) ─────────────────────────────
step("toolchain.begin")
kh.install_build_toolchain()
# gotcha #22: move ccache out of /kaggle/working so it can't flood the outputs.
CC = TEMP / ".ccache"
if (WORK / ".ccache").exists() and not CC.exists():
    shutil.move(str(WORK / ".ccache"), str(CC))
os.environ["CCACHE_DIR"] = str(CC)
step("toolchain.done")

BUILD = REPO / "build"
step("build.begin")
with kh.build_heartbeat("cmake-configure", 30):
    kh.sh(f"cmake -S {REPO} -B {BUILD} -DCMAKE_BUILD_TYPE=Release "
          f"-DCRISPASR_BUILD_TESTS=OFF -DCRISPASR_BUILD_SERVER=OFF "
          f"-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
with kh.build_heartbeat("cmake-build", 30):
    kh.sh(f"cmake --build {BUILD} -j {kh.safe_build_jobs(gpu=True)} "
          f"--target crispasr-cli crispasr-quantize crispasr-diff")
CRISPASR = BUILD / "bin" / "crispasr"
QUANT = BUILD / "bin" / "crispasr-quantize"
DIFF = BUILD / "bin" / "crispasr-diff"
for b in (CRISPASR, QUANT, DIFF):
    if not b.is_file():
        step("fatal.binary-missing", which=b.name)
        raise SystemExit(f"{b.name} did not build — judge by the artifact, not the exit code")
step("build.done", ccache_hits=kh.sh("ccache -s | head -5", check=False))
kh.export_ccache_tar(str(WORK / "ccache.tar"))

# ── staging on the ~70 GB ephemeral layer, not the ~20 GB working mount (#18) ─
MODELS = Path("/tmp/madlad")
MODELS.mkdir(parents=True, exist_ok=True)
SRC = Path("/tmp/madlad-src")

SRC_REPO = "google/madlad400-3b-mt"
DST_REPO = "cstr/madlad400-3b-mt-GGUF"
F16 = MODELS / "madlad400-3b-mt-f16.gguf"
Q8 = MODELS / "madlad400-3b-mt-q8_0.gguf"
Q4 = MODELS / "madlad400-3b-mt-q4_k.gguf"
REF = MODELS / "madlad400-3b-mt-ref.gguf"

MADLAD_TEXT = "Hello world, how are you today?"
os.environ["MADLAD_TEXT"] = MADLAD_TEXT
os.environ["MADLAD_TL"] = "de"


def gb(p):
    return round(Path(p).stat().st_size / 1e9, 2)


def free_gb(p="/tmp"):
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except Exception:
        return -1.0


summary = {"issue": 333, "source": SRC_REPO, "target": DST_REPO, "artifacts": {}, "parity": {}}

# ── resumable: what is already on the repo? ───────────────────────────────────
# The first run of this kernel produced and uploaded F16 and Q8_0 and then died
# in the reference dump on a wrong tensor key. Re-running everything to get the
# parity table would redo ~15 minutes of conversion and quantization for files
# that are already published, so each produce-step is skipped when its artifact
# exists and the file is fetched for the diff phase instead.
try:
    HAVE = {f.rfilename for f in HfApi().model_info(DST_REPO, token=TOKEN).siblings}
except Exception as e:
    HAVE = set()
    step("repo-listing.failed", err=str(e)[:200])
step("repo-listing", present=sorted(HAVE))


def save():
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


# ── validation: three real translations, not a load ───────────────────────────
CASES = [
    (MADLAD_TEXT, "en", "de", ["hallo", "welt", "wie geht", "heute"]),
    ("Machine learning is changing the world.", "en", "fr", ["apprentissage", "monde", "machine"]),
    ("Bonjour le monde!", "fr", "en", ["hello", "world", "good"]),
]


def _norm(s):
    return re.sub(r"[^a-z0-9äöüàéèç ]", " ", (s or "").lower())


def validate(model_path, label):
    step(f"{label}.validate.begin", gb=gb(model_path))
    rows = []
    for text, sl, tl, wants in CASES:
        with kh.build_heartbeat(f"{label}.translate.{sl}-{tl}", 30):
            p = subprocess.run([str(CRISPASR), "--backend", "madlad", "-m", str(model_path),
                                "--text", text, "-sl", sl, "-tl", tl],
                               capture_output=True, text=True, timeout=1800)
        out = _norm(p.stdout)
        # A non-zero exit or an empty transcript is a FAIL, never a quiet pass —
        # a fast crash must not read as success (kaggle_usage #24).
        ok = p.returncode == 0 and bool(out.strip())
        rows.append({"pair": f"{sl}->{tl}", "exit": p.returncode, "ok": ok,
                     "matched": any(w in out for w in wants),
                     "echoed": _norm(text).strip() in out,
                     "out": p.stdout.strip()[:200]})
        step(f"{label}.translate.{sl}-{tl}", **{k: rows[-1][k] for k in ("exit", "matched", "echoed")},
             out=p.stdout.strip()[:120])
    passed = all(r["ok"] and r["matched"] and not r["echoed"] for r in rows)
    step(f"{label}.validate.done", passed=passed)
    return passed, rows


def upload(path, msg):
    step("upload.begin", file=Path(path).name, gb=gb(path))
    with kh.build_heartbeat(f"upload.{Path(path).name}", 30):
        HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=Path(path).name,
                            repo_id=DST_REPO, repo_type="model", token=TOKEN, commit_message=msg)
    step("upload.done", file=Path(path).name)


# ── 1. source ─────────────────────────────────────────────────────────────────
step("source-download.begin", free_gb=free_gb())
with kh.build_heartbeat("source-download", 30):
    snapshot_download(repo_id=SRC_REPO, local_dir=str(SRC), token=TOKEN,
                      allow_patterns=["model.safetensors", "config.json", "spiece.model",
                                      "tokenizer*.json", "special_tokens_map.json",
                                      "added_tokens.json", "generation_config.json"])
step("source-download.done", free_gb=free_gb())

# ── 2. F16 ────────────────────────────────────────────────────────────────────
if F16.name in HAVE:
    step("convert.skipped", why="already on the repo")
    with kh.build_heartbeat("f16-download", 30):
        F16 = Path(hf_hub_download(repo_id=DST_REPO, filename=F16.name,
                                   local_dir=str(MODELS), token=TOKEN))
    summary["artifacts"]["f16"] = {"size_gb": gb(F16), "preexisting": True}
    save()
else:
  step("convert.begin", free_gb=free_gb())
  with kh.build_heartbeat("convert-f16", 30):
    p = subprocess.run([sys.executable, str(CONVERTER), "--input", str(SRC), "--output", str(F16)],
                       capture_output=True, text=True, timeout=7200)
  if p.returncode != 0 or not F16.is_file():
    step("fatal.convert-failed", exit=p.returncode, tail=p.stdout[-600:], err=p.stderr[-600:])
    raise SystemExit("F16 conversion failed")
  step("convert.done", f16_gb=gb(F16), free_gb=free_gb())

  ok, rows = validate(F16, "f16")
  summary["artifacts"]["f16"] = {"size_gb": gb(F16), "validated": ok, "cases": rows}
  save()
  if not ok:
    step("fatal.f16-invalid")
    raise SystemExit("F16 failed validation — refusing to upload a broken artifact")
  upload(F16, "add F16 (#333: the README listed it but the repo never had it)")
  summary["artifacts"]["f16"]["uploaded"] = True
  save()

# ── 3. reference archive — BEFORE the source is deleted ───────────────────────
step("refdump.begin", free_gb=free_gb())
with kh.build_heartbeat("refdump", 30):
    p = subprocess.run([sys.executable, str(DUMPER), "--backend", "madlad",
                        "--model-dir", str(SRC), "--output", str(REF),
                        "--audio", str(REPO / "samples" / "jfk.wav")],
                       capture_output=True, text=True, timeout=7200, cwd=str(REPO / "tools"))
ref_ok = p.returncode == 0 and REF.is_file()
step("refdump.done", exit=p.returncode, ok=ref_ok,
     ref_mb=round(REF.stat().st_size / 1e6, 1) if ref_ok else 0,
     tail=p.stdout[-800:], err=p.stderr[-800:] if p.returncode else "")
if ref_ok:
    upload(REF, "add per-stage reference archive for crispasr-diff (#333)")
    summary["artifacts"]["ref"] = {"size_mb": round(REF.stat().st_size / 1e6, 1), "uploaded": True}
else:
    # Not fatal: the quants are the issue, the reference is the bonus.
    summary["artifacts"]["ref"] = {"failed": True, "tail": p.stdout[-400:], "err": p.stderr[-400:]}
save()

# ── does the BLUEPRINT itself run away on a short input? (#333) ───────────────
# Our runtime emits "Hello world! – 1000000…" for fr→en "Bonjour le monde!" —
# correct translation, then a digit loop to the token cap. Two arms of our own
# already agree (incremental KV vs full re-forward each step), and F16 per-stage
# parity is 1.000000 with a matching step-0 argmax, so the runtime reproduces
# the model faithfully at every point we can measure. The one thing that decides
# whether this is a PORT bug or the MODEL's behaviour is what the PyTorch
# blueprint does on the same input, and only this kernel has the checkpoint.
step("decode-probe.begin")
probe = {}
for text, tl in (("Bonjour le monde!", "en"), ("Hello world, how are you today?", "de"),
                 ("Machine learning is changing the world.", "fr")):
    env = dict(os.environ, MADLAD_TEXT=text, MADLAD_TL=tl)
    out = MODELS / f"probe-{tl}.gguf"
    with kh.build_heartbeat(f"decode-probe.{tl}", 30):
        pr = subprocess.run([sys.executable, str(DUMPER), "--backend", "madlad",
                             "--model-dir", str(SRC), "--output", str(out),
                             "--max-new-tokens", "60",
                             "--audio", str(REPO / "samples" / "jfk.wav")],
                            capture_output=True, text=True, timeout=3600, env=env,
                            cwd=str(REPO / "tools"))
    line = [l for l in pr.stdout.splitlines() if "madlad reference:" in l]
    probe[f"{tl}:{text[:32]}"] = {"exit": pr.returncode, "report": line,
                                  "err": pr.stderr[-300:] if pr.returncode else ""}
    step(f"decode-probe.{tl}", exit=pr.returncode, report=line)
    out.unlink(missing_ok=True)
summary["blueprint_decode_probe"] = probe
save()
step("decode-probe.done")

shutil.rmtree(SRC, ignore_errors=True)
step("source.deleted", free_gb=free_gb())

# ── 4. Q8_0 ───────────────────────────────────────────────────────────────────
# No t5/madlad rule in examples/crispasr-quantize/main.cpp, so this takes the
# generic path — the same one that produced the published Q4_K. The validation
# and the parity table below are what prove that was the right call.
if Q8.name in HAVE:
    step("quantize.skipped", why="already on the repo")
    with kh.build_heartbeat("q8-download", 30):
        Q8 = Path(hf_hub_download(repo_id=DST_REPO, filename=Q8.name,
                                  local_dir=str(MODELS), token=TOKEN))
    summary["artifacts"]["q8_0"] = {"size_gb": gb(Q8), "preexisting": True}
    save()
else:
  step("quantize.begin", free_gb=free_gb())
  with kh.build_heartbeat("quantize-q8_0", 30):
    p = subprocess.run([str(QUANT), str(F16), str(Q8), "q8_0"],
                       capture_output=True, text=True, timeout=7200)
  if p.returncode != 0 or not Q8.is_file():
    step("fatal.quantize-failed", exit=p.returncode, tail=p.stdout[-600:])
    raise SystemExit("Q8_0 quantization failed (F16 is already on HF, so nothing is lost)")
  step("quantize.done", q8_gb=gb(Q8), free_gb=free_gb())

  ok, rows = validate(Q8, "q8_0")
  summary["artifacts"]["q8_0"] = {"size_gb": gb(Q8), "validated": ok, "cases": rows}
  save()
  if not ok:
    step("fatal.q8-invalid")
    raise SystemExit("Q8_0 failed validation — refusing to upload a broken artifact")
  upload(Q8, "add Q8_0 (#333)")
  summary["artifacts"]["q8_0"]["uploaded"] = True
  save()

# ── 5. per-stage parity for all three quants ──────────────────────────────────
# This is the part that turns "it loaded" into a number. Q4_K is pulled back
# from HF so the file people already have is measured too.
if ref_ok:
    step("q4k-download.begin")
    try:
        got = hf_hub_download(repo_id=DST_REPO, filename=Q4.name, local_dir=str(MODELS), token=TOKEN)
        Q4 = Path(got)
        step("q4k-download.done", gb=gb(Q4))
    except Exception as e:
        step("q4k-download.failed", err=str(e)[:200])

    for label, path in (("f16", F16), ("q8_0", Q8), ("q4_k", Q4)):
        if not Path(path).is_file():
            continue
        step(f"diff.{label}.begin")
        with kh.build_heartbeat(f"diff.{label}", 30):
            p = subprocess.run([str(DIFF), "madlad", str(path), str(REF)],
                               capture_output=True, text=True, timeout=3600)
        # Parse the per-stage lines the diff prints:
        #   t5 enc_out          n=...  cos=0.999998  max_abs=...  |mine|=... |ref|=...  PASS
        stages = {}
        for line in p.stdout.splitlines():
            m = re.match(r"t5 (\S+)\s+n=\S+\s+cos=([-\d.]+)\s+max_abs=([\d.]+)\s+"
                         r"\|mine\|=([\d.]+) \|ref\|=([\d.]+)\s+(PASS|FAIL)", line)
            if m:
                stages[m.group(1)] = {"cos": float(m.group(2)), "max_abs": float(m.group(3)),
                                      "mine": float(m.group(4)), "ref": float(m.group(5)),
                                      "verdict": m.group(6)}
        am = re.search(r"argmax_step0\s+mine=(\d+) ref=(\d+)\s+(MATCH|DIFFER)", p.stdout)
        summary["parity"][label] = {
            "exit": p.returncode,
            "stages": stages,
            "argmax_step0": {"mine": int(am.group(1)), "ref": int(am.group(2)), "verdict": am.group(3)}
            if am else None,
            "worst_cos": min((v["cos"] for v in stages.values()), default=None),
            "stdout_tail": p.stdout[-1500:],
        }
        step(f"diff.{label}.done", exit=p.returncode, n_stages=len(stages),
             worst_cos=summary["parity"][label]["worst_cos"],
             argmax=summary["parity"][label]["argmax_step0"])
        save()
else:
    step("diff.skipped", why="no reference archive")

# ── 6. requant with the t5 rule, and measure whether it is worth the size ────
# The published Q4_K/Q8_0 predate examples/crispasr-quantize/main.cpp's t5 rule
# (shared.embed.* and lm_head.* stay at source precision). Those two tensors
# carry the 256K vocabulary and are exactly where the per-stage table says Q4_K
# loses — enc_embed 0.9974, enc_out 0.9937 against 1.000000 at F16. Keeping them
# wide should lift those stages; it also makes the file bigger, and that trade
# needs numbers rather than a preference.
#
# Only uploads when parity actually improves. A bigger file that measures the
# same is a worse artifact, not a better one.
if ref_ok and F16.is_file():
    step("requant.begin")
    for qtype in ("q8_0", "q4_k"):
        newf = MODELS / f"madlad400-3b-mt-{qtype}.new.gguf"
        with kh.build_heartbeat(f"requant.{qtype}", 30):
            pq = subprocess.run([str(QUANT), str(F16), str(newf), qtype],
                                capture_output=True, text=True, timeout=7200)
        if pq.returncode != 0 or not newf.is_file():
            step(f"requant.{qtype}.failed", exit=pq.returncode, tail=pq.stdout[-400:])
            continue
        with kh.build_heartbeat(f"requant.{qtype}.diff", 30):
            pd = subprocess.run([str(DIFF), "madlad", str(newf), str(REF)],
                                capture_output=True, text=True, timeout=3600)
        stages = {}
        for line in pd.stdout.splitlines():
            m = re.match(r"t5 (\S+)\s+n=\S+\s+cos=([-\d.]+)", line)
            if m:
                stages[m.group(1)] = float(m.group(2))
        worst_new = min(stages.values()) if stages else None
        before = summary["parity"].get(qtype, {})
        worst_old = before.get("worst_cos")
        ok_val, val_rows = validate(newf, f"{qtype}-requant")
        better = (worst_new is not None and worst_old is not None and worst_new > worst_old)
        summary.setdefault("requant", {})[qtype] = {
            "size_gb_old": before.get("size_gb"), "size_gb_new": gb(newf),
            "worst_cos_old": worst_old, "worst_cos_new": worst_new,
            "stages_new": stages, "validated": ok_val, "better": better,
        }
        step(f"requant.{qtype}.measured", worst_old=worst_old, worst_new=worst_new,
             gb_new=gb(newf), better=better, validated=ok_val)
        save()
        if better and ok_val:
            upload(newf, f"re-quantize {qtype} with the t5 rule "
                         f"(embeddings/lm_head at source precision) — #333")
            summary["requant"][qtype]["uploaded"] = True
        else:
            step(f"requant.{qtype}.not-uploaded",
                 why="parity did not improve" if not better else "failed validation")
        newf.unlink(missing_ok=True)
        save()
    step("requant.done")

save()
step("script.done", artifacts=list(summary["artifacts"]), parity=list(summary["parity"]))
print(json.dumps(summary, indent=2, ensure_ascii=False)[:6000])
