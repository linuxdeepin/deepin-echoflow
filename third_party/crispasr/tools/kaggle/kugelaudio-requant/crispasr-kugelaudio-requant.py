#!/usr/bin/env python3
"""
Find a KugelAudio quant that fits a 16 GB GPU AND still speaks.

The F16 is 17.3 GB, so `-m auto` downloads 17 GB and then fails to allocate on
every 16 GB card. The published Q4_K fits but is broken, not merely lossy —
measured on Kaggle it stutters and loops:

  Q4_K  WER 0.7222, 13.73 s  "The quick brown fast The quick brown the quick
                              brown fox jobs over the lazy job with a morning
                              light spills across the MBC."
  F16   WER 0.0556,  7.47 s  (reference-quality)

That Q4_K was produced with the generic 2-D rule, which quantizes the DiT
diffusion head and the VAE decoder along with the 7B Qwen2.5 backbone. Both
hypotheses are testable, so test both rather than guessing:

  q8_0        uniform, near-lossless. If plain precision was the problem, this
              fixes it. ~9.2 GB, fits comfortably.
  q6_k        uniform, smaller. The cheap win if it holds.
  q4_k_mixed  backbone Q4_K, DiT head + VAE decoder + semantic tokenizer kept
              F16. If THIS is the one that works while uniform q4_k does not,
              the fault was the rule, not the bit-width — and the payoff is the
              smallest file of the three.

Every candidate is ASR-roundtripped (feedback_tts_validation); nothing is judged
by file size or by loading without error. The winner gets uploaded.

Follows the harness regime (clone in-kernel, import from the clone, heartbeat).
"""
import os, subprocess, sys, json, re, wave, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
REPO = TMP / "CrispASR"
BUILD = TMP / "build"
GGUF_REPO = "cstr/kugelaudio-0-open-GGUF"
F16_NAME = "kugelaudio-0-open-f16.gguf"
TEXT = ("The quick brown fox jumps over the lazy dog while the morning light "
        "spills across the empty street.")
UPLOAD = os.environ.get("KUGEL_UPLOAD", "1") == "1"

# The DiT diffusion head, the acoustic VAE decoder and the semantic tokenizer are
# the parts a coarse quant is most likely to wreck: they carry the acoustics,
# not the token stream. Names follow the converter, which preserves the PyTorch
# hierarchy with the "model." prefix stripped.
KEEP_F16 = r"(dit|diffusion|vae|decoder|vocoder|semantic_tokenizer|semantic_connector)"

CANDIDATES = [
    ("q8_0",       "q8_0", []),
    ("q6_k",       "q6_k", []),
    ("q4_k_mixed", "q4_k", ["--tensor-type", f"{KEEP_F16}=f16"]),
]

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--recurse-submodules",
                    "--shallow-submodules", "https://github.com/CrispStrobe/CrispASR.git", str(REPO)],
                   check=True, timeout=2400)

_h = REPO / "tools" / "kaggle"
sys.path.insert(0, str(_h if (_h / "kaggle_harness.py").exists() else Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh._HF_PROGRESS_PATH = "runs/kugelaudio-requant-live.jsonl"
kh.init_progress()
res = {"text": TEXT, "candidates": [c[0] for c in CANDIDATES]}


def sh(cmd, timeout=3600):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


kh.step("toolchain")
kh.install_build_toolchain()
token = kh.resolve_hf_token()
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

cuda = kh.detect_cuda_arch()
kh.step("cmake", cuda_arch=cuda)
flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCRISPASR_BUILD_TESTS=OFF",
         "-DCRISPASR_BUILD_EXAMPLES=ON", "-DCRISPASR_BUILD_SERVER=OFF",
         "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda}"] + kh.cache_and_link_flags()
with kh.build_heartbeat("cmake.configure"):
    rc, out = sh(f"cmake -S {REPO} -B {BUILD} -G Ninja " + " ".join(flags))
if rc != 0:
    print(out[-6000:], flush=True); raise SystemExit("configure failed")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli crispasr-quantize "
                        f"-j{kh.safe_build_jobs(gpu=True)}")
cli = BUILD / "bin" / "crispasr"
quantize = BUILD / "bin" / "crispasr-quantize"
for exe in (cli, quantize):
    if not exe.exists():
        raise SystemExit(f"{exe.name} missing after build")

kh.step("download f16")
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402
f16 = Path(hf_hub_download(repo_id=GGUF_REPO, filename=F16_NAME, token=token, local_dir=str(TMP / "gguf")))
res["f16_gb"] = round(f16.stat().st_size / 1e9, 2)
print(f"f16: {res['f16_gb']} GB", flush=True)

# List the real tensor namespaces before trusting KEEP_F16. The regex was
# written from the converter's "preserve the PyTorch hierarchy" comment, i.e.
# from inference — and a rule that matches nothing produces a "mixed" quant
# identical to the uniform one, which would look like a result. Print what is
# actually in the file, and how many names the rule catches.
kh.step("inspect tensors")
try:
    sys.path.insert(0, str(REPO / "ggml" / "gguf-py"))
    import gguf  # noqa: E402
    rdr = gguf.GGUFReader(str(f16))
    names = [t.name for t in rdr.tensors]
    prefixes = sorted({n.split(".")[0] for n in names})
    matched = [n for n in names if re.search(KEEP_F16, n)]
    res["tensor_count"] = len(names)
    res["tensor_prefixes"] = prefixes
    res["keep_f16_matches"] = len(matched)
    res["keep_f16_sample"] = matched[:12]
    print(f"{len(names)} tensors; top-level prefixes: {prefixes}", flush=True)
    print(f"KEEP_F16 matches {len(matched)} tensors, e.g. {matched[:6]}", flush=True)
    if not matched:
        print("WARNING: KEEP_F16 matches nothing — q4_k_mixed would equal uniform q4_k", flush=True)
except Exception as e:  # noqa: BLE001
    res["tensor_inspect_error"] = repr(e)
    print(f"tensor inspection failed: {e!r}", flush=True)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


def dur_s(p):
    try:
        with wave.open(str(p)) as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:  # noqa: BLE001
        return 0.0


results = {}
for label, ftype, extra in CANDIDATES:
    kh.step(f"quantize {label}")
    out_gguf = TMP / "gguf" / f"kugelaudio-0-open-{label}.gguf"
    r = {"ftype": ftype, "extra": " ".join(extra)}
    with kh.build_heartbeat(f"quantize.{label}", 30.0):
        rc, out = sh(f"{quantize} {f16} {out_gguf} {ftype} " + " ".join(extra), timeout=5400)
    if rc != 0 or not out_gguf.exists():
        print(out[-4000:], flush=True)
        r["quantize_rc"] = rc
        r["quantize_log"] = out[-2500:]
        results[label] = r
        continue
    r["gb"] = round(out_gguf.stat().st_size / 1e9, 2)
    r["fits_16gb"] = r["gb"] < 15.0  # leave room for KV + activations
    print(f"{label}: {r['gb']} GB", flush=True)

    kh.step(f"synth {label}")
    wav = TMP / f"kugel_{label}.wav"
    with kh.build_heartbeat(f"synth.{label}", 30.0):
        rc, out = sh(f"{cli} --backend kugelaudio -m {out_gguf} "
                     f"--tts \"{TEXT}\" --tts-output {wav} --no-prints", timeout=2400)
    r["synth_rc"] = rc
    r["synth_log"] = out[-2000:]
    if rc != 0 or not wav.exists() or wav.stat().st_size < 1000:
        print(out[-4000:], flush=True)
        results[label] = r
        continue
    shutil.copy(str(wav), str(WORK / wav.name))  # for a human listen
    r["dur_s"] = dur_s(wav)

    rc, out = sh(f"{cli} --backend parakeet -m auto --auto-download -f {wav} --no-prints", timeout=1800)
    text = out.strip().splitlines()[-1] if out.strip() else ""
    r["asr"] = text
    r["wer"] = round(wer(TEXT, text), 4)
    # The Q4_K failure showed up in DURATION as much as in words: 13.73 s of
    # audio for a 7.47 s sentence, because it looped. Judge both.
    r["usable"] = r["wer"] <= 0.20 and r["dur_s"] <= 12.0
    print(f"[{label}] {r['gb']} GB  wer={r['wer']}  dur={r['dur_s']}s  usable={r['usable']}\n  {text}", flush=True)
    results[label] = r

res["results"] = results

# Winner: smallest usable candidate that fits. Size only breaks ties among
# things that already passed the roundtrip — never the other way round.
usable = [(v["gb"], k) for k, v in results.items() if v.get("usable") and v.get("fits_16gb")]
winner = min(usable)[1] if usable else None
res["winner"] = winner
print(f"\nwinner: {winner}", flush=True)

if winner and UPLOAD:
    kh.step("upload winner")
    src = TMP / "gguf" / f"kugelaudio-0-open-{winner}.gguf"
    # Published under its ftype name, so --model-quant <ftype> resolves it via
    # the registry's filename substitution. q4_k_mixed ships as the q4_k name:
    # it REPLACES the broken uniform q4_k rather than sitting beside it, since
    # leaving that one reachable is leaving a trap.
    dest = f"kugelaudio-0-open-{'q4_k' if winner == 'q4_k_mixed' else winner}.gguf"
    api = HfApi()
    try:
        with kh.build_heartbeat("upload", 30.0):
            api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                            repo_id=GGUF_REPO, repo_type="model", token=token)
        remote = {f.rfilename: f for f in api.model_info(GGUF_REPO, files_metadata=True).siblings}
        res["uploaded"] = dest
        res["upload_verified"] = dest in remote
        print(f"uploaded {dest} (verified={res['upload_verified']})", flush=True)
    except Exception as e:  # noqa: BLE001
        res["upload_error"] = repr(e)
        print(f"upload failed: {e!r}", flush=True)

(WORK / "kugelaudio_requant.json").write_text(json.dumps(res, indent=2))
kh.step("done", winner=winner, uploaded=res.get("uploaded"))
print(json.dumps(res, indent=2), flush=True)
