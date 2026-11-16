#!/usr/bin/env python3
"""Backfill `cohere_transcribe.supported_languages` into every published
Cohere GGUF, then prove the whole path works in the product.

Why: Cohere Transcribe accepts a FIXED language set and answers a wrong
language *fluently* instead of failing. The set cannot be recovered from the
GGUF — the tokenizer carries all 183 ISO-639-1 `<|xx|>` tokens whatever the
model supports — so the runtime fix (0b858088) is INERT until the whitelist
actually ships in the weights. That is what this kernel does.

Phases, in this order on purpose:
  1. Build crispasr.
  2. Rewrite ONE GGUF, verify byte-identical tensors, and TRANSCRIBE with it
     before uploading anything. Never overwrite 23 GB of published artifacts
     on the strength of a checksum alone.
  3. Rewrite + verify + upload the remaining files, one at a time, deleting
     as we go, and re-check the uploaded copy over HTTP Range.
  4. Run the converter end-to-end on the real safetensors — it has never been
     executed since it learned to write this key — and diff its tensors
     against the published f16.
  5. Measure in the product, from GGUF metadata alone (no env override):
     substitution on an unsupported -l, the probe on the 2-language model,
     the cost gate on the 14-language model, and the forced 14-way probe
     whose accuracy is the open question.
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
# Keep every HF byte off /kaggle/working: it is the 20 GB output mount and is
# page-capped at 500 files on retrieval (gotchas #18/#22). ~23 GB moves through
# here, so the cache goes on the big ephemeral layer too.
os.environ.setdefault("HF_HOME", str(TMP / "hf"))
REPO = TMP / "CrispASR"                       # off /kaggle/working (gotcha #22)
BUILD = TMP / "build"
STAGE = TMP / "stage"; STAGE.mkdir(parents=True, exist_ok=True)
RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")

# (gguf repo, source model repo whose config.json holds supported_languages, files)
TARGETS = [
    ("cstr/cohere-transcribe-arabic-07-2026-GGUF", "CohereLabs/cohere-transcribe-arabic-07-2026", [
        "cohere-transcribe-arabic-q4_k-imatrix.gguf",   # smoke-tested first
        "cohere-transcribe-arabic-q4_k.gguf",
        "cohere-transcribe-arabic-q8_0.gguf",
        "cohere-transcribe-arabic-f16.gguf",
    ]),
    ("cstr/cohere-transcribe-03-2026-GGUF", "CohereLabs/cohere-transcribe-03-2026", [
        "cohere-transcribe-q4_k.gguf",
        "cohere-transcribe-q5_0.gguf",
        "cohere-transcribe-q5_1.gguf",
        "cohere-transcribe-q6_k.gguf",
        "cohere-transcribe-q8_0.gguf",
        "cohere-transcribe.gguf",
    ]),
]

REPORT = {"phases": {}, "files": [], "product": {}, "converter": {}}


def jstep(name, **kv):
    print(f"[STEP] {name} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def save_report():
    (RESULTS / "report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False))


# ───────────────────────── clone + harness + build ────────────────────────
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                       "--recursive", CRISPASR_REPO, str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()
SHA = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
REPORT["sha"] = SHA
jstep("cloned", sha=SHA)

TOKEN = kh.resolve_hf_token()
assert TOKEN, "no HF token — the whole kernel is pointless without upload rights"

for pkg in ("gguf", "huggingface_hub"):
    try:
        __import__(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
from huggingface_hub import hf_hub_download, upload_file  # noqa: E402

kh.install_build_toolchain()
subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(BUILD),
                       "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON"])
with kh.build_heartbeat("build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} "
                        f"--target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}")
CLI = BUILD / "bin" / "crispasr"
assert CLI.exists(), "crispasr binary missing"
REPORT["phases"]["build"] = "ok"
jstep("built")
save_report()

ADD_LANGS = REPO / "tools" / "gguf-add-cohere-langs.py"
assert ADD_LANGS.exists(), f"{ADD_LANGS} missing — is the tool on {CRISPASR_REF}?"

AR_WAV = REPO / "tools" / "kaggle" / "cohere-arabic-verify" / "ar_clean_8s.wav"
JFK_WAV = REPO / "samples" / "jfk.wav"
assert AR_WAV.exists() and JFK_WAV.exists()


# ───────────────────────── helpers ────────────────────────────────────────
def run_cli(model, wav, *extra, timeout=900):
    """Return (rc, stdout, stderr). Never raises — a failure is a data point."""
    cmd = [str(CLI), "--backend", "cohere", "-m", str(model), str(wav), *extra]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, "", f"TIMEOUT after {timeout}s"


def transcript_of(stdout):
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    return lines[-1] if lines else ""


_GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}
_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_kv_from_bytes(buf, want_key):
    """Minimal GGUF KV scanner over a PREFIX of a file.

    Used to confirm the key landed in the copy that is now live on HF, without
    re-downloading multi-GB files. Returns the value, None if absent, or
    raises EOFError when the prefix was too short to decide.
    """
    off = 0

    def take(n):
        nonlocal off
        if off + n > len(buf):
            raise EOFError
        b = buf[off:off + n]
        off += n
        return b

    if take(4) != b"GGUF":
        raise ValueError("not a GGUF")
    version = struct.unpack("<I", take(4))[0]
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    struct.unpack("<Q", take(8))[0]              # n_tensors
    n_kv = struct.unpack("<Q", take(8))[0]

    def read_str():
        n = struct.unpack("<Q", take(8))[0]
        return take(n).decode("utf-8", "replace")

    def read_value(t):
        if t == 8:
            return read_str()
        if t == 9:
            sub = struct.unpack("<I", take(4))[0]
            count = struct.unpack("<Q", take(8))[0]
            return [read_value(sub) for _ in range(count)]
        if t in _GGUF_SIZES:
            return struct.unpack("<" + _GGUF_TYPES[t], take(_GGUF_SIZES[t]))[0]
        raise ValueError(f"unknown GGUF value type {t}")

    for _ in range(n_kv):
        key = read_str()
        vtype = struct.unpack("<I", take(4))[0]
        val = read_value(vtype)
        if key == want_key:
            return val
    return None


def remote_key(repo_id, filename, key, prefix_mb=16):
    """Range-GET the head of the uploaded file and look for `key`."""
    import urllib.request
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    for mb in (prefix_mb, prefix_mb * 4):
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {TOKEN}",
            "Range": f"bytes=0-{mb * 1024 * 1024 - 1}",
        })
        with urllib.request.urlopen(req, timeout=300) as r:
            buf = r.read()
        try:
            return gguf_kv_from_bytes(buf, key)
        except EOFError:
            continue
    raise EOFError(f"{filename}: KV section longer than {prefix_mb * 4} MB")


def rewrite_one(gguf_repo, src_repo, filename):
    """download -> add key -> verify tensors -> return local path of the new file"""
    src = Path(hf_hub_download(gguf_repo, filename, token=TOKEN, local_dir=str(STAGE)))
    dst = STAGE / (filename.replace(".gguf", "") + ".langs.gguf")
    subprocess.check_call([sys.executable, str(ADD_LANGS),
                           "--hf-config", src_repo, "--verify",
                           str(src), str(dst)])
    return src, dst


def upload_one(gguf_repo, filename, path, expect_langs):
    for attempt in range(1, 6):
        try:
            upload_file(path_or_fileobj=str(path), path_in_repo=filename,
                        repo_id=gguf_repo, token=TOKEN,
                        commit_message="add cohere_transcribe.supported_languages "
                                       "(the runtime cannot validate -l without it)")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  upload attempt {attempt} failed: {e}", flush=True)
            if attempt == 5:
                raise
            time.sleep(20 * attempt)
    live = remote_key(gguf_repo, filename, "cohere_transcribe.supported_languages")
    if live != expect_langs:
        raise RuntimeError(f"{filename}: live copy has {live}, expected {expect_langs}")
    return live


def langs_of(src_repo):
    cfg = json.load(open(hf_hub_download(src_repo, "config.json", token=TOKEN)))
    return [str(c).strip().lower() for c in cfg["supported_languages"]]


# ───────── phase 2: one file, smoke-tested in the RUNTIME before upload ────
gguf_repo, src_repo, files = TARGETS[0]
first = files[0]
expect = langs_of(src_repo)
jstep("pilot.rewrite", file=first, langs=expect)
with kh.build_heartbeat("pilot-rewrite", 30):
    src_path, new_path = rewrite_one(gguf_repo, src_repo, first)

pilot = {"file": first, "langs": expect}
rc, out, err = run_cli(new_path, AR_WAV, "-l", "ar", "--lid-backend", "off")
pilot["transcribe_ar"] = {"rc": rc, "text": transcript_of(out)}
rc2, out2, err2 = run_cli(new_path, AR_WAV, "-l", "de", "--lid-backend", "off")
pilot["reject_de"] = {"rc": rc2, "warned": "is not supported by this model" in err2,
                      "text": transcript_of(out2)}
REPORT["phases"]["pilot"] = pilot
save_report()
jstep("pilot.runtime", rc=rc, warned=pilot["reject_de"]["warned"])

assert rc == 0 and pilot["transcribe_ar"]["text"], "rewritten GGUF failed to transcribe — NOT uploading"
assert pilot["reject_de"]["warned"], "rewritten GGUF did not trigger the whitelist — NOT uploading"

upload_one(gguf_repo, first, new_path, expect)
REPORT["files"].append({"repo": gguf_repo, "file": first, "langs": expect, "uploaded": True})
save_report()
jstep("pilot.uploaded", file=first)
# keep the pilot file: phase 5 uses it
PILOT_GGUF = STAGE / "pilot-arabic.gguf"
shutil.move(str(new_path), str(PILOT_GGUF))
src_path.unlink(missing_ok=True)

# ───────────────────────── phase 3: the rest ──────────────────────────────
for gguf_repo, src_repo, files in TARGETS:
    expect = langs_of(src_repo)
    for filename in files:
        if any(f["file"] == filename for f in REPORT["files"]):
            continue
        jstep("rewrite", file=filename, free_gb=kh.free_gb("/kaggle/temp"))
        entry = {"repo": gguf_repo, "file": filename, "langs": expect, "uploaded": False}
        try:
            with kh.build_heartbeat(f"rewrite {filename}", 30):
                src_path, new_path = rewrite_one(gguf_repo, src_repo, filename)
            upload_one(gguf_repo, filename, new_path, expect)
            entry["uploaded"] = True
            # keep one base-model quant for phase 5
            if filename == "cohere-transcribe-q4_k.gguf":
                shutil.move(str(new_path), str(STAGE / "base-q4k.gguf"))
            else:
                new_path.unlink(missing_ok=True)
            src_path.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            entry["error"] = repr(e)
            print(f"  FAILED {filename}: {e}", flush=True)
        REPORT["files"].append(entry)
        save_report()

jstep("republish.done", ok=sum(1 for f in REPORT['files'] if f['uploaded']),
      total=sum(len(t[2]) for t in TARGETS))

# ───────── phase 4: the converter, end-to-end, for the first time ─────────
# It has written this key since 0b858088 but has never actually been RUN.
conv = {}
try:
    CONV_SRC = TMP / "arabic-src"; CONV_SRC.mkdir(exist_ok=True)
    for f in ("config.json", "model.safetensors", "tokenizer_config.json"):
        try:
            hf_hub_download("CohereLabs/cohere-transcribe-arabic-07-2026", f,
                            token=TOKEN, local_dir=str(CONV_SRC))
        except Exception as e:  # noqa: BLE001
            conv[f"missing_{f}"] = repr(e)
    out_gguf = TMP / "converted-f16.gguf"
    with kh.build_heartbeat("convert", 30):
        p = subprocess.run([sys.executable, str(REPO / "models" / "convert-cohere-asr-to-gguf.py"),
                            "--model-dir", str(CONV_SRC), "--output", str(out_gguf)],
                           capture_output=True, text=True, timeout=3600)
    conv["rc"] = p.returncode
    conv["tail"] = (p.stdout + p.stderr)[-2000:]
    if p.returncode == 0 and out_gguf.exists():
        import gguf as gguflib
        r = gguflib.GGUFReader(str(out_gguf))
        f = r.fields.get("cohere_transcribe.supported_languages")
        conv["langs"] = None if f is None else [
            str(bytes(f.parts[i]), encoding="utf-8") for i in f.data]
        mc = r.fields.get("cohere_transcribe.audio.max_clip_s")
        conv["max_clip_s"] = None if mc is None else int(mc.parts[-1][0])
        conv["n_tensors"] = len(r.tensors)
        # does a fresh conversion match the published f16 we just republished?
        pub = Path(hf_hub_download("cstr/cohere-transcribe-arabic-07-2026-GGUF",
                                   "cohere-transcribe-arabic-f16.gguf",
                                   token=TOKEN, local_dir=str(STAGE)))
        rp = gguflib.GGUFReader(str(pub))
        def digest(rd):
            return {str(t.name): hashlib.sha256(t.data.tobytes()).hexdigest() for t in rd.tensors}
        a, b = digest(r), digest(rp)
        conv["tensors_match_published_f16"] = (a == b)
        conv["n_differing"] = len([k for k in a if b.get(k) != a[k]])
        pub.unlink(missing_ok=True)
        out_gguf.unlink(missing_ok=True)
    shutil.rmtree(CONV_SRC, ignore_errors=True)
except Exception as e:  # noqa: BLE001
    conv["error"] = repr(e)
REPORT["converter"] = conv
save_report()
jstep("converter", rc=conv.get("rc"), langs=conv.get("langs"),
      match=conv.get("tensors_match_published_f16"))

# ───────── phase 5: the product, driven by GGUF metadata alone ────────────
prod = {}
BASE_GGUF = STAGE / "base-q4k.gguf"


def record(name, model, wav, *extra, **checks):
    rc, out, err = run_cli(model, wav, *extra)
    rec = {"rc": rc, "text": transcript_of(out)}
    lid = [l for l in err.splitlines() if "cohere[lid]" in l or "LID ->" in l]
    if lid:
        rec["lid"] = lid
    rec["substituted"] = "is not supported by this model" in err
    rec["probe_declined"] = "skipping the self-probe" in err
    for k, needle in checks.items():
        rec[k] = needle in err
    prod[name] = rec
    save_report()
    jstep(f"product.{name}", rc=rc, sub=rec["substituted"])
    return rec


if PILOT_GGUF.exists():
    # 2-language model: whitelist + probe, both from GGUF metadata only
    record("ar_reject_ru", PILOT_GGUF, AR_WAV, "-l", "ru", "--lid-backend", "off")
    record("ar_probe_arabic", PILOT_GGUF, AR_WAV, "-l", "auto")
    record("ar_probe_english", PILOT_GGUF, JFK_WAV, "-l", "auto")

if BASE_GGUF.exists():
    # 14-language model: unsupported code, cost gate, and the forced probe
    record("base_reject_ru", BASE_GGUF, JFK_WAV, "-l", "ru", "--lid-backend", "off")
    record("base_gate", BASE_GGUF, JFK_WAV, "-l", "auto")
    record("base_probe_forced_en", BASE_GGUF, JFK_WAV, "-l", "auto", "--lid-backend", "probe")
    record("base_probe_forced_ar", BASE_GGUF, AR_WAV, "-l", "auto", "--lid-backend", "probe")

REPORT["product"] = prod
save_report()

print("\n================ SUMMARY ================")
print(json.dumps({
    "sha": REPORT["sha"],
    "uploaded": [f["file"] for f in REPORT["files"] if f["uploaded"]],
    "failed": [f["file"] for f in REPORT["files"] if not f["uploaded"]],
    "converter_rc": REPORT["converter"].get("rc"),
    "converter_langs": REPORT["converter"].get("langs"),
    "converter_tensors_match": REPORT["converter"].get("tensors_match_published_f16"),
    "product": {k: {"rc": v["rc"], "substituted": v["substituted"],
                    "declined": v["probe_declined"], "lid": v.get("lid", [])[-1:],
                    "text": v["text"][:80]}
                for k, v in prod.items()},
}, indent=2, ensure_ascii=False))
