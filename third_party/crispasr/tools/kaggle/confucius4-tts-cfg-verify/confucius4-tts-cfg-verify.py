#!/usr/bin/env python3
"""Kaggle kernel: verify the Confucius4-TTS S2A fixes (#377, feat/confucius4-cfg).

Runs the full pipeline and then does what the VPS cannot: drives the REAL
PyTorch S2A (confuciustts/flow) on exactly the inputs and initial noise the C++
runtime produced, and compares per stage.  Finishes with a TTS -> ASR roundtrip,
which is the only acceptance gate (HARD RULE #3).

What is under test (all on feat/confucius4-cfg):
  1. s2a_sinusoidal_embed  -- scale=1000, cos-then-sin
  2. InterpolateRegulator  -- the learned conv stack, was skipped entirely
  3. classifier-free guidance in the ODE
  4. T2S sampling temperature (the CLI was forcing greedy)
  5. linear ODE time schedule (the port used cosine)
  6. the prompt string -- LANGUAGE_TOKEN_MAP["en"] is CHINESE, and the old test
     kernel invented an English one, so every previous run fed the T2S a prompt
     it was never trained on

Push ($KAGGLE_ACCOUNT is the default account):
  python -m kaggle kernels push -p tools/kaggle/confucius4-tts-cfg-verify
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

BRANCH = "main"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
REF = TEMP / "confucius4-ref"
DUMP = TEMP / "s2a_dump"

# Keep the run tractable on a Kaggle CPU box: short text -> few semantic codes.
TEST_TEXT = "The quick brown fox jumps over the lazy dog."
LANG = "en"
ODE_STEPS = int(os.environ.get("ODE_STEPS", "25"))

# ── Phase 0: clone repo (the branch under test) ─────────────────────────────
print(f"=== Phase 0: clone {BRANCH} ===", flush=True)
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "-b", BRANCH,
        "https://github.com/CrispStrobe/CrispASR", str(REPO),
    ])
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO))
head = subprocess.run(["git", "log", "--oneline", "-3"], cwd=str(REPO),
                      capture_output=True, text=True).stdout
print("  HEAD:\n   " + "\n   ".join(head.strip().split("\n")))

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh
kh.init_progress()

# ── Phase 1: deps + HF token ────────────────────────────────────────────────
kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer tokenizers pyyaml librosa scipy "
                    "torchaudio safetensors transformers")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
from huggingface_hub import hf_hub_download

# ── Phase 2: the Python blueprint ───────────────────────────────────────────
kh.step("clone Python blueprint")
if not REF.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/netease-youdao/Confucius4-TTS", str(REF),
    ])
sys.path.insert(0, str(REF))
from confuciustts.utils.text_utils import LANGUAGE_TOKEN_MAP

# ── Phase 3: models ─────────────────────────────────────────────────────────
kh.step("download GGUFs")
mdir = str(TEMP / "models")


def grab(repo, fname, **kw):
    p = hf_hub_download(repo, fname, local_dir=kw.pop("d", mdir), token=hf_token, **kw)
    print(f"  {fname}: {os.path.getsize(p) / 1024**2:.0f} MB")
    return p


t2s_path = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-t2s-q4_k.gguf")
s2a_path = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-s2a-q4_k.gguf")
# F16 too: running the S2A at F16 separates quantization error compounding
# through the 25-step ODE from a genuine remaining port bug.
s2a_f16 = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-s2a-f16.gguf")
voc_path = grab("cstr/confucius4-tts-GGUF", "confucius4-tts-bigvgan-22k-f16.gguf")
s2a_ckpt = grab("netease-youdao/Confucius4-TTS", "s2a_model.pt", d=str(TEMP / "torch"))
try:  # encoder-only w2v-BERT: enables the FULLY native --voice arm (sibling of the T2S)
    grab("cstr/confucius4-tts-GGUF", "confucius4-tts-w2v-f16.gguf")
    HAVE_W2V = True
except Exception as _e:  # noqa: BLE001
    print(f"  (no w2v GGUF yet: {_e}) -- NATIVE-full arm will be skipped")
    HAVE_W2V = False

# ── Phase 4: tokenize with the CORRECT prompt ───────────────────────────────
kh.step("tokenize (real LANGUAGE_TOKEN_MAP)")
tok_path = hf_hub_download("netease-youdao/Confucius4-TTS", "tokenizer.json",
                           local_dir=str(TEMP / "tokenizer"), token=hf_token)
from tokenizers import Tokenizer

tok = Tokenizer.from_file(tok_path)
lang_token = LANGUAGE_TOKEN_MAP.get(LANG, f"请用{LANG}朗读接下来的文字")
formatted = f"You are a helpful assistant. {lang_token}:{TEST_TEXT}"
ids = tok.encode(formatted).ids
# The raw tokenizers post_processor only prepends <s>, but the reference loads
# AutoTokenizer -> LlamaTokenizerFast, whose add_eos_token=True (from
# tokenizer_config.json) rebuilds the template to ALSO append </s> (id 2).
# Verified: AutoTokenizer ids end [..., 28723, 2]; raw ids end [..., 28723].
if ids and ids[-1] != 2:
    ids = ids + [2]
token_ids_str = ",".join(str(x) for x in ids)
print(f"  lang_token : {lang_token!r}   <-- Chinese, per LANGUAGE_TOKEN_MAP")
print(f"  formatted  : {formatted}")
print(f"  n_ids={len(ids)}  ids={token_ids_str}")

# ── Phase 5: build ──────────────────────────────────────────────────────────
kh.step("build CrispASR")
BUILD = TEMP / "build"
BUILD.mkdir(parents=True, exist_ok=True)
flags = kh.cache_and_link_flags()
kh.sh_with_progress(
    f"cmake -G Ninja -B {BUILD} -S {REPO} -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
    + " ".join(flags)
)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} -j{kh.safe_build_jobs(gpu=False)} --target crispasr-cli"
    )
crispasr_bin = str(BUILD / "bin" / "crispasr")
print(f"  binary: {crispasr_bin} ({os.path.getsize(crispasr_bin) / 1024**2:.0f} MB)")

# ── Phase 6: synthesize, dumping the S2A stages ─────────────────────────────
kh.step(f"TTS ({ODE_STEPS} ODE steps, CFG on) -- q4_k and f16")

env_base = os.environ.copy()
env_base["CRISPASR_CONFUCIUS4_TEXT_IDS"] = token_ids_str

runs = {}
for tag, s2a in (("q4_k", s2a_path), ("f16", s2a_f16)):
    dump = TEMP / f"dump_{tag}"
    dump.mkdir(parents=True, exist_ok=True)
    wav = TEMP / f"confucius4_{tag}.wav"
    env = dict(env_base)
    env["CRISPASR_CONFUCIUS4_DUMP_S2A"] = str(dump)
    r = subprocess.run(
        [crispasr_bin, "--backend", "confucius4-tts", "-m", t2s_path,
         "--codec-model", s2a, "--tts", TEST_TEXT,
         "--tts-output", str(wav), "--tts-steps", str(ODE_STEPS), "-v"],
        capture_output=True, text=True, timeout=7200, env=env,
    )
    ok = wav.exists() and os.path.getsize(str(wav)) > 100
    print(f"  [{tag}] rc={r.returncode}  wav={'%d B' % os.path.getsize(str(wav)) if ok else 'NONE'}")
    for line in r.stderr.split("\n"):
        if any(k in line for k in ("flow-matching:", "regulator OK", "time schedule", "BigVGAN:")):
            print(f"  [{tag}] " + line.strip())
    if r.returncode != 0:
        for line in [l for l in r.stderr.split("\n") if l.strip()][-20:]:
            print(f"  [{tag}] ! " + line.strip())
    runs[tag] = {"rc": r.returncode, "stderr": r.stderr, "wav": wav, "ok": ok, "dump": dump}

res = runs["q4_k"]["stderr"]
wav_ok = runs["q4_k"]["ok"]
tts_wav = runs["q4_k"]["wav"]


# ── Phase 6b: speaker conditioning ──────────────────────────────────────────
# The model is zero-shot -- run 3 showed the PyTorch reference itself babbles
# with zero conditioning.  Feed real conditioning derived from a reference wav
# and see whether the roundtrip passes.  Two arms, cheap one first:
#   s2a-only : CAMPPlus style + reference mel  (needs only campplus, ~28 MB)
#   full     : + the T2S condition_emb         (needs w2v-BERT 2.4 GB + T2S 2.6 GB)
# If s2a-only already yields speech, the semantic codes were fine all along and
# the T2S condition_emb is a voice-identity refinement rather than a blocker.
kh.step("speaker conditioning")
dumper = REPO / "tools" / "confucius4_dump_conditioning.py"
prompt_wav = REPO / "samples" / "jfk.wav"
w2v_stats = None

cond_runs = {}
t2s_ckpt = None
for arm in ("s2a-only", "full"):
    cdir = TEMP / f"cond_{arm}"
    cmd = [sys.executable, str(dumper), "--ref-repo", str(REF),
           "--prompt-wav", str(prompt_wav), "--out-dir", str(cdir)]
    if arm == "s2a-only":
        cmd.append("--no-w2v")
    else:
        if w2v_stats is None:
            w2v_stats = grab("netease-youdao/Confucius4-TTS", "wav2vec2bert_stats.pt",
                             d=str(TEMP / "torch"))
        t2s_ckpt = grab("netease-youdao/Confucius4-TTS", "t2s_model.safetensors",
                        d=str(TEMP / "torch"))
        cmd += ["--t2s-ckpt", t2s_ckpt, "--w2v-stats", w2v_stats]

    dres = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print(f"  [{arm}] dumper rc={dres.returncode}")
    for line in dres.stdout.split("\n"):
        if line.strip():
            print(f"  [{arm}] " + line.strip())
    if dres.returncode != 0:
        for line in [l for l in dres.stderr.split("\n") if l.strip()][-20:]:
            print(f"  [{arm}] ! " + line.strip())
        continue

    wav = TEMP / f"confucius4_cond_{arm}.wav"
    dump = TEMP / f"dump_cond_{arm}"
    dump.mkdir(parents=True, exist_ok=True)
    env = dict(env_base)
    env["CRISPASR_CONFUCIUS4_COND_DIR"] = str(cdir)
    env["CRISPASR_CONFUCIUS4_DUMP_S2A"] = str(dump)
    r = subprocess.run(
        [crispasr_bin, "--backend", "confucius4-tts", "-m", t2s_path,
         "--codec-model", s2a_f16, "--tts", TEST_TEXT,
         "--tts-output", str(wav), "--tts-steps", str(ODE_STEPS), "-v"],
        capture_output=True, text=True, timeout=7200, env=env,
    )
    ok = wav.exists() and os.path.getsize(str(wav)) > 100
    print(f"  [{arm}] TTS rc={r.returncode} wav={'%d B' % os.path.getsize(str(wav)) if ok else 'NONE'}")
    for line in r.stderr.split("\n"):
        if any(k in line for k in ("conditioning set", "flow-matching:", "EOS at", "semantic codes",
                                   "BigVGAN:")):
            print(f"  [{arm}] " + line.strip())
    if r.returncode != 0:
        for line in [l for l in r.stderr.split("\n") if l.strip()][-20:]:
            print(f"  [{arm}] ! " + line.strip())
    cond_runs[arm] = {"wav": wav, "ok": ok}

# ── Phase 6b2: native-path arms (need the re-converted GGUFs) ───────────────
# NATIVE-tok  : no CRISPASR_CONFUCIUS4_TEXT_IDS — the runtime's SP-BPE
#               tokenizer (baked vocab) must reproduce the AutoTokenizer ids.
# NATIVE-voice: --voice jfk.wav — CAMPPlus style + prompt mel computed
#               in-process from the campplus.* bake; the cond dir supplies
#               ONLY w2v_features.bin (ECAPA condition_emb runs natively).
import shutil

_native_arms = {}
_nat_cond = TEMP / "cond_native"
_nat_cond.mkdir(parents=True, exist_ok=True)
if (TEMP / "cond_full" / "w2v_features.bin").exists():
    shutil.copy(TEMP / "cond_full" / "w2v_features.bin", _nat_cond / "w2v_features.bin")
_arms = [
    ("NATIVE-tok", {"CRISPASR_CONFUCIUS4_COND_DIR": str(TEMP / "cond_full")}, []),
    ("NATIVE-voice", {"CRISPASR_CONFUCIUS4_COND_DIR": str(_nat_cond)},
     ["--voice", str(prompt_wav), "--i-have-rights"]),
]
if HAVE_W2V:
    # THE fully native zero-shot path: no COND_DIR, no TEXT_IDS — tokenizer,
    # w2v-BERT layer-17, ECAPA, CAMPPlus, prompt mel all in-process.
    _arms.append(("NATIVE-full", {}, ["--voice", str(prompt_wav), "--i-have-rights"]))
    # Persistent-decode-graph timing A/B on a clean box (the VPS is too noisy).
    # PCM must stay bit-identical to NATIVE-full (only the timestamped C2PA
    # manifest may differ); the wall-time delta decides the default.
    _arms.append(("PERSIST-ab", {"CRISPASR_CONFUCIUS4_PERSIST": "1"},
                  ["--voice", str(prompt_wav), "--i-have-rights"]))
import time as _time
for arm, extra_env, extra_args in _arms:
    wav = TEMP / f"confucius4_{arm}.wav"
    env = dict(os.environ)  # NATIVE-tok / NATIVE-full run the native tokenizer
    if arm == "NATIVE-voice":
        env["CRISPASR_CONFUCIUS4_TEXT_IDS"] = token_ids_str
    env.update(extra_env)
    _t0 = _time.monotonic()
    r = subprocess.run(
        [crispasr_bin, "--backend", "confucius4-tts", "-m", t2s_path,
         "--codec-model", s2a_f16, "--tts", TEST_TEXT, "-l", LANG,
         "--tts-output", str(wav), "--tts-steps", str(ODE_STEPS), "-v"] + extra_args,
        capture_output=True, text=True, timeout=7200, env=env,
    )
    _wall = _time.monotonic() - _t0
    ok = wav.exists() and os.path.getsize(str(wav)) > 100
    print(f"  [{arm}] TTS rc={r.returncode} wall={_wall:.1f}s "
          f"wav={'%d B' % os.path.getsize(str(wav)) if ok else 'NONE'}")
    for line in r.stderr.split("\n"):
        if any(k in line for k in ("conditioning set", "tokenized", "voice set", "CAMPPlus",
                                   "beam decode", "BigVGAN:", "WARNING", "condition_emb computed")):
            print(f"  [{arm}] " + line.strip())
    if r.returncode != 0:
        for line in [l for l in r.stderr.split("\n") if l.strip()][-15:]:
            print(f"  [{arm}] ! " + line.strip())
    _native_arms[arm] = {"wav": wav, "ok": ok}
print(f"  expected AutoTokenizer ids: n={len(ids)} (runtime 'tokenized' line above must match)")
if "PERSIST-ab" in _native_arms and _native_arms.get("NATIVE-full", {}).get("ok"):
    import wave as _wave
    def _pcm(p):
        w = _wave.open(str(p)); d = w.readframes(w.getnframes()); w.close(); return d
    _same = _pcm(_native_arms["NATIVE-full"]["wav"]) == _pcm(_native_arms["PERSIST-ab"]["wav"])
    print(f"  PERSIST-ab PCM vs NATIVE-full: {'BIT-IDENTICAL' if _same else 'DIFFERS (investigate!)'}")


# ── Phase 6c: REFERENCE end-to-end control ──────────────────────────────────
# T2S parity now passes (prefill argmax matches, lm_latent cos 0.999) yet the
# roundtrip still fails — including the reference-S2A-on-runtime-codes arm.
# The missing control: does the FULL reference pipeline (beam-sample T2S +
# reference S2A + torch BigVGAN) pass the roundtrip on THIS text and THIS
# prompt wav? If yes, the remaining delta is the decode strategy
# (num_beams=3 beam-sample vs pure sampling). If no, the harness/conditioning
# is the problem and the C++ is off the hook.
kh.step("REFERENCE end-to-end control")
ref_e2e_wav = TEMP / "confucius4_ref_e2e.wav"
# transformers pinned to the reference release: beam-sample internals moved
# across HF versions, and this arm defines the ground-truth decode behavior.
kh.sh_with_progress("pip install -q regex inflect pykakasi 'transformers==4.52.4'")
ref_dcond_wav = TEMP / "confucius4_ref_dumpercond.wav"
runtime_codes_wav = TEMP / "confucius4_runtimecodes_refs2a.wav"
_driver = TEMP / "ref_e2e_driver.py"
_driver.write_text(f"""
import sys, os
import numpy as np
import torch, torchaudio
torch.set_grad_enabled(False)
sys.path.insert(0, {str(REF)!r})
os.chdir({str(REF)!r})   # config uses relative ./checkpoints paths
from confuciustts.cli.inference import ConfuciusTTS
m = ConfuciusTTS(config_path="config/inference_config.yaml", device="cpu")
audio = m.generate({TEST_TEXT!r}, {LANG!r}, {str(prompt_wav)!r}, raw=True, verbose=True)
if audio.dim() == 1:
    audio = audio.unsqueeze(0)
torchaudio.save({str(ref_e2e_wav)!r}, audio.cpu(), m.sample_rate)
print("saved", {str(ref_e2e_wav)!r})

# ---- conditioning cross-check: dumper .bins vs the reference's own values ----
def cmp(tag, mine, ref):
    mine = np.asarray(mine, np.float64).ravel(); ref = np.asarray(ref, np.float64).ravel()
    if mine.shape != ref.shape:
        print(f"COND-XCHK {{tag:16s}} SHAPE mine={{mine.shape}} ref={{ref.shape}}"); return
    cos = float(mine @ ref / (np.linalg.norm(mine) * np.linalg.norm(ref) + 1e-12))
    print(f"COND-XCHK {{tag:16s}} cos={{cos:.6f}} |mine|={{np.linalg.norm(mine):.3f}} |ref|={{np.linalg.norm(ref):.3f}}")

cd = {str(TEMP / 'cond_full')!r}
wav16, wavt = m._load_prompt({str(prompt_wav)!r})
sem = m._extract_semantic(wav16)[0].numpy()
sty = m._extract_style(wav16)[0].numpy()
rmel = m._ref_mel(wavt)[0].numpy()
cmp("w2v_features", np.fromfile(cd + "/w2v_features.bin", np.float32).reshape(-1, 1024), sem)
cmp("style_embedding", np.fromfile(cd + "/style_embedding.bin", np.float32), sty)
cmp("prompt_mel", np.fromfile(cd + "/prompt_mel.bin", np.float32).reshape(-1, 80), rmel)

# ---- reference generate under the DUMPER's conditioning -> ref S2A -> voc ----
def synth_from_codes(codes, latent, tag, out_path):
    T = codes.shape[1]
    mel = m.s2a_model.inference(
        semantic_token=codes, lm_latent=latent,
        prompt_feat=torch.from_numpy(rmel).unsqueeze(0), embedding=torch.from_numpy(sty).unsqueeze(0),
        target_feat_len=torch.tensor([int(T * 1.72)]), n_timesteps=25, inference_cfg_rate=0.7)
    wav = m.bigvgan(mel.float()).squeeze(1)
    torchaudio.save(out_path, wav.cpu(), m.sample_rate)
    print(f"{{tag}}: {{T}} codes -> {{wav.shape[-1] / m.sample_rate:.2f}}s saved")

feats = torch.from_numpy(np.fromfile(cd + "/w2v_features.bin", np.float32).reshape(1, -1, 1024))
tid = torch.from_numpy(np.fromfile(cd.replace("cond_full", "dump_cond_full") + "/text_ids_i32.bin",
                                   np.int32).astype(np.int64)).unsqueeze(0)
out = m.t2s_model.generate(text_inputs=tid, condition_vector=feats, max_length=1520, num_beams=3,
                           do_sample=True, top_p=0.8, top_k=30, temperature=0.8,
                           repetition_penalty=10.0, early_stopping=True, return_latent=True)
print("ref codes under dumper conditioning:", out["semantic_codes"].shape[1])
synth_from_codes(out["semantic_codes"], out["latent"], "REF-dumpercond", {str(ref_dcond_wav)!r})

# ---- runtime's own conditioned codes + latents through ref S2A + voc ----
dd = cd.replace("cond_full", "dump_cond_full")
rc = np.fromfile(dd + "/semantic_codes_i32.bin", np.int32).astype(np.int64)
rl = np.fromfile(dd + "/lm_latent.bin", np.float32).reshape(-1, 1280)[: len(rc)]
synth_from_codes(torch.from_numpy(rc).unsqueeze(0), torch.from_numpy(rl).unsqueeze(0),
                 "RUNTIME-codes/ref-S2A", {str(runtime_codes_wav)!r})
""")
try:
    eres = subprocess.run([sys.executable, str(_driver)], capture_output=True,
                          text=True, timeout=7200)
    print(f"  ref e2e rc={eres.returncode}")
    for line in eres.stdout.split("\n")[-15:]:
        if line.strip():
            print("  " + line.rstrip())
    if eres.returncode != 0:
        for line in [l for l in eres.stderr.split("\n") if l.strip()][-25:]:
            print("  ! " + line.strip())
except Exception as e:  # noqa: BLE001 — control arm must never sink the run
    print(f"  ref e2e failed: {e}")


# ── Phase 7: per-stage parity against the real PyTorch S2A ──────────────────
kh.step("S2A parity vs PyTorch reference")
parity = REPO / "tools" / "s2a_parity.py"
# third arm: the CONDITIONED dump — exercises the prompt path (prompt_cond
# prepend, prompt_x, spks, per-step re-zero, strip), which had zero parity
# coverage when bug 13 (CFG-uncond over T_mel) shipped.
_parity_arms = [("q4_k", runs["q4_k"]["dump"], None),
                ("f16", runs["f16"]["dump"], None),
                ("cond-full", TEMP / "dump_cond_full", TEMP / "cond_full")]
for tag, dump, cdir in _parity_arms:
    print(f"  --- {tag} ---")
    if not (dump / "shapes.txt").exists():
        print("  SKIP: no dump produced")
        continue
    cmd = [sys.executable, str(parity), "--dump-dir", str(dump),
           "--ref-repo", str(REF), "--s2a-ckpt", s2a_ckpt,
           "--steps", str(ODE_STEPS), "--cfg", "0.7"]
    if cdir is not None:
        cmd += ["--style", str(cdir / "style_embedding.bin"),
                "--prompt-mel", str(cdir / "prompt_mel.bin")]
    if tag == "q4_k":                       # vocode once, from the shipped quant
        cmd += ["--vocode-out", str(TEMP / "vocoded")]
    pres = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print(f"  parity rc={pres.returncode}")
    for line in pres.stdout.split("\n"):
        if line.strip():
            print("  " + line.rstrip())
    if pres.returncode != 0:
        for line in [l for l in pres.stderr.split("\n") if l.strip()][-25:]:
            print("  ! " + line.strip())


# ── Phase 7b: T2S parity ────────────────────────────────────────────────────
# S2A is exact yet the reference S2A babbles on these codes, so the codes are
# wrong and the bug is upstream.  This is the stage the harness never covered.
kh.step("T2S parity vs PyTorch reference")
t2s_parity = REPO / "tools" / "t2s_parity.py"
_full = TEMP / "dump_cond_full"
_w2v = TEMP / "cond_full" / "w2v_features.bin"
if not (t2s_parity.exists() and (_full / "shapes.txt").exists() and _w2v.exists()):
    print("  SKIP: needs the full-conditioning run (dump + w2v features)")
else:
    tres = subprocess.run(
        [sys.executable, str(t2s_parity), "--dump-dir", str(_full), "--ref-repo", str(REF),
         "--t2s-ckpt", t2s_ckpt, "--w2v-features", str(_w2v)],
        capture_output=True, text=True, timeout=7200,
    )
    print(f"  t2s parity rc={tres.returncode}")
    for line in tres.stdout.split("\n"):
        if line.strip():
            print("  " + line.rstrip())
    if tres.returncode != 0:
        for line in [l for l in tres.stderr.split("\n") if l.strip()][-25:]:
            print("  ! " + line.strip())


# ── Phase 8: ASR roundtrip (the acceptance gate) ────────────────────────────
kh.step("ASR roundtrip")
import urllib.request

whisper_model = str(TEMP / "models" / "ggml-base.en.bin")
os.makedirs(os.path.dirname(whisper_model), exist_ok=True)
if not os.path.exists(whisper_model):
    urllib.request.urlretrieve(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        whisper_model,
    )

ORIG = set(w.strip(".,!?").lower() for w in TEST_TEXT.split())


def asr_score(tag, path):
    """Transcribe and word-overlap score one wav.  The decisive comparison is
    cpp vs ref: if the REFERENCE audio is also unintelligible then the S2A port
    is not the blocker, the missing speaker conditioning is."""
    if not (path and os.path.exists(path) and os.path.getsize(path) > 100):
        print(f"  [{tag}] SKIP: no wav")
        return
    a = subprocess.run([crispasr_bin, "-m", whisper_model, "-f", str(path), "--no-prints"],
                       capture_output=True, text=True, timeout=600)
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", a.stdout)
    hit = {w for w in ORIG if w in clean.lower()}
    try:  # duration + rms so a silent / NaN wav is visible next to its score
        import soundfile as _sf
        _d, _sr = _sf.read(str(path))
        _rms = float(np.sqrt(np.mean(np.square(_d)))) if len(_d) else 0.0
        _stat = f"{len(_d) / _sr:.2f}s rms={_rms:.4f} nan={int(np.isnan(_d).any())}"
    except Exception as _e:  # noqa: BLE001
        _stat = f"stat-err {_e}"
    print(f"  [{tag}] rc={a.returncode}  overlap {len(hit)}/{len(ORIG)} = "
          f"{100 * len(hit) / max(len(ORIG), 1):.0f}%  ({_stat})")
    print(f"  [{tag}] transcript: {clean.strip()[:400]}")
    return len(hit)


asr_score("nocond-q4_k", str(runs["q4_k"]["wav"]) if runs["q4_k"]["ok"] else None)
asr_score("nocond-f16", str(runs["f16"]["wav"]) if runs["f16"]["ok"] else None)
for _arm, _r in cond_runs.items():
    asr_score(f"COND-{_arm}", str(_r["wav"]) if _r["ok"] else None)
asr_score("cpp-mel/torch-voc", str(TEMP / "vocoded" / "cpp_mel.wav"))
asr_score("REF-mel/torch-voc", str(TEMP / "vocoded" / "ref_mel.wav"))
asr_score("REF-E2E-control", str(ref_e2e_wav) if ref_e2e_wav.exists() else None)
asr_score("REF-dumpercond", str(ref_dcond_wav) if ref_dcond_wav.exists() else None)
asr_score("RUNTIME-codes/ref-S2A", str(runtime_codes_wav) if runtime_codes_wav.exists() else None)
for _arm, _r in _native_arms.items():
    asr_score(_arm, str(_r["wav"]) if _r["ok"] else None)
print("  NOTE: if REF-mel is also ~0%, the port is not the blocker -- the model is")
print("        zero-shot and always has a speaker prompt, which is still all zeros.")

# ── Phase 9: summary ────────────────────────────────────────────────────────
kh.step("summary")
m = re.search(r"generated (\d+) semantic codes", res)
print(f"  semantic codes : {m.group(1) if m else '?'}")
m = re.search(r"time schedule: (\w+)", res)
print(f"  time schedule  : {m.group(1) if m else '?'}   (expect linear)")
m = re.search(r"cfg=([0-9.]+)", res)
print(f"  cfg rate       : {m.group(1) if m else '?'}   (expect 0.70)")
m = re.search(r"regulator OK \(([0-9]+)→([0-9]+)→([0-9]+) dims\)", res)
print(f"  regulator dims : {'->'.join(m.groups()) if m else '?'}   (expect 2304->1024->512)")
print(f"  TTS rc         : q4_k={runs['q4_k']['rc']} f16={runs['f16']['rc']}")
print("\n=== Done ===", flush=True)
