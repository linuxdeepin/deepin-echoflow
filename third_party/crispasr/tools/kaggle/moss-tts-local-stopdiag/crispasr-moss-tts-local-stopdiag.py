# CrispASR — MOSS-TTS-Local 4B STOP-HEAD diagnostic (#249, P5 debug) — CPU-only
#
# run1/run2 proved the codec decode is correct (ASR overlap 1.0) but the C++
# generation RUNS AWAY (binary stop head never fires) for short text + Q4_K; the
# stop logit sits at continue~8 / stop~-3 every frame. Static inspection shows
# the C++ port matches the HF reference structurally. THE decisive question:
#   Does the HF reference MossTTSLocal.generate() STOP for "Hello world"?
#
# This runs the HF reference only (no C++, no codec, no ASR) so it needs NO GPU
# (GPU weekly quota is exhausted) — a CPU kernel. It hooks local_text_lm_head and
# runs generate(do_sample=True, text_temperature=1.0) on SHORT + LONG text,
# reporting: frames, stopped?, where the stop logit first exceeds continue, and
# the [continue, stop] trajectory.
#
# Verdict:
#   ref stops short  -> C++ has a subtle bug (compare trajectories next).
#   ref runs away too -> model/param behavior; fix = generation handling
#                        (e.g. text-conditioned max frames, or the "- Tokens:"
#                        prompt field, or sampling config).

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Kaggle sets =1 but the pkg isn't installed
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
TMP = Path("/tmp")
REPO = TMP / "CrispASR"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
HF_MODEL = os.environ.get("MOSS_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5")
MAXF = int(os.environ.get("MOSS_MAXF", "256"))
# Multiple seeds (incl. 0 = our C++ default p.seed=0) to measure STOP RELIABILITY,
# not a single lucky/unlucky draw — the whole point is whether the reference's
# sampled stop is dependable or as coin-flippy as our C++ port.
SEEDS = [int(s) for s in os.environ.get("MOSS_SEEDS", "0,1234,7").split(",")]
# Audio sampling MUST match our C++ backend defaults (model card: 1.7/0.8/25),
# else we'd be comparing different trajectories. The stop head stays sampled at
# text_temperature 1.0 (reference default).
AUDIO_TEMP, AUDIO_TOP_P, AUDIO_TOP_K = 1.7, 0.8, 25

SHORT_TEXT = "Hello world."
LONG_TEXT = ("The quick brown fox jumps over the lazy dog. "
             "Speech synthesis should stay intelligible over a longer passage.")
_T0 = time.time()


def log(m):
    print(f"[{round(time.time() - _T0, 1)}s] {m}", flush=True)


def run_reference(hf_token):
    # huggingface_hub freezes HF_HUB_ENABLE_HF_TRANSFER at import (Kaggle sets =1 but
    # the pkg is absent) -> force it off on the already-imported module.
    import huggingface_hub.constants as _hfc
    _hfc.HF_HUB_ENABLE_HF_TRANSFER = False
    try:
        import huggingface_hub.file_download as _fd
        _fd.HF_HUB_ENABLE_HF_TRANSFER = False
    except Exception:  # noqa: BLE001
        pass
    import torch
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(os.cpu_count() or 4)
    log("load reference model (float32, cpu)")
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                      token=hf_token).eval()
    proc = AutoProcessor.from_pretrained(HF_MODEL, trust_remote_code=True)  # public; token kwarg rejected by custom proc
    log("model + processor loaded")

    import kaggle_harness as kh
    out = {}
    for tag, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
        msg = proc.build_user_message(text=text)
        feat = proc([msg], mode="generation")
        input_ids = feat["input_ids"] if isinstance(feat, dict) else feat.input_ids
        try:
            prompt_txt = proc.tokenizer.decode(input_ids.reshape(-1, input_ids.shape[-1])[:, 0].tolist())
            (RESULTS / f"ref_prompt_{tag}.txt").write_text(prompt_txt)
        except Exception:  # noqa: BLE001
            pass
        runs = []
        for seed in SEEDS:
            stop_logits = []

            def hook(_m, _i, o):
                t = o.detach().float().reshape(-1, o.shape[-1])
                stop_logits.append(t[-1].tolist())  # [continue, stop]

            h = model.local_text_lm_head.register_forward_hook(hook)
            try:
                torch.manual_seed(seed)
                t0 = time.time()
                with kh.build_heartbeat(f"ref.{tag}.seed{seed}"):
                    model.generate(input_ids=input_ids, max_new_frames=MAXF, do_sample=True,
                                   text_temperature=1.0, temperature=AUDIO_TEMP,
                                   top_p=AUDIO_TOP_P, top_k=AUDIO_TOP_K)
                n = len(stop_logits)
                stopped = n < MAXF
                cross = next((i for i, (cl, sl) in enumerate(stop_logits) if sl >= cl), None)
                traj = [(i, round(cl, 3), round(sl, 3)) for i, (cl, sl) in enumerate(stop_logits)]
                runs.append({"seed": seed, "frames": n, "stopped": stopped, "stop_crosses_at": cross,
                             "min_gap": round(min((cl - sl for cl, sl in stop_logits), default=0), 3),
                             "elapsed_s": round(time.time() - t0, 1),
                             "logit_first": traj[:8], "logit_last": traj[-6:]})
                log(f"REF {tag} seed={seed}: frames={n} stopped={stopped} cross_at={cross}")
            except Exception as e:  # noqa: BLE001
                runs.append({"seed": seed, "error": f"{type(e).__name__}: {e}"})
                log(f"REF {tag} seed={seed} ERROR: {e}")
            finally:
                h.remove()
        n_stopped = sum(1 for r in runs if r.get("stopped"))
        out[tag] = {"runs": runs, "n_seeds": len(SEEDS), "n_stopped": n_stopped,
                    "reliable": n_stopped == len(SEEDS)}
        log(f"REF {tag}: {n_stopped}/{len(SEEDS)} seeds stopped (reliable={out[tag]['reliable']})")
    return out


def main():
    summary = {"ref_branch": CRISPASR_REF, "seed": SEED, "maxf": MAXF, "hf_model": HF_MODEL}
    log(f"clone {CRISPASR_REF} (for kaggle_harness)")
    if not REPO.exists():
        import subprocess
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
                               CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    hf_token = kh.resolve_hf_token()

    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers==4.57.6", "accelerate", "huggingface_hub", "hf_transfer"])
    try:
        summary["reference"] = run_reference(hf_token)
    except Exception as e:  # noqa: BLE001
        summary["reference"] = {"fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]}
        log(f"reference fatal: {e}")

    ref = summary.get("reference", {})
    rs = ref.get("short") if isinstance(ref, dict) else None
    if isinstance(rs, dict) and "n_stopped" in rs:
        ns, nt = rs["n_stopped"], rs["n_seeds"]
        if ns == nt:
            summary["verdict"] = (f"PORT/QUANT DIVERGENCE: reference STOPS 'Hello world' on all {nt} seeds "
                                  f"(card params) — our C++ q4_k runs away on the same text, so the stop "
                                  f"path or quant sensitivity is OURS to fix; compare trajectories.")
        elif ns == 0:
            summary["verdict"] = (f"MODEL BEHAVIOR: reference runs away on ALL {nt} seeds too — inherent; "
                                  f"fix = a robust stop policy (repetition / frame-budget), not the port.")
        else:
            summary["verdict"] = (f"FRAGILE BY DESIGN: reference stops on only {ns}/{nt} seeds — the sampled "
                                  f"stop is coin-flippy even in the reference; fix = a deterministic stop net.")
    else:
        summary["verdict"] = "INCONCLUSIVE (reference error — see tb)"
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60 + "\n" + json.dumps(summary, indent=2) + "\n" + "=" * 60)
    log(f"VERDICT: {summary['verdict']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
