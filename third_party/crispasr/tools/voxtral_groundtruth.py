#!/usr/bin/env python
"""Run upstream Mistral Voxtral via transformers on each of the
60/120/300/600 s lenhone clips. Output is logged with timing and
character count so we can diff against the C++ streamed output."""
from __future__ import annotations
import sys, time, json, os, gc

# overrides shim (transformers loads enforce-overrides transitively)
import overrides as _ro
import overrides.enforce as _re
from abc import ABCMeta
_lo = lambda f=None, **kw: (f if callable(f) else lambda fn: fn)
class _LEM(ABCMeta): pass
class _LEO(metaclass=_LEM): pass
_lm = type(sys)("overrides"); _lm.__dict__.update(_ro.__dict__)
_lm.override = _lo; _lm.overrides = _lo; _lm.EnforceOverrides = _LEO
_rm = _re.EnforceOverridesMeta; _re.EnforceOverridesMeta = _LEM
sys.modules["overrides"] = _lm
sys.modules["overrides.enforce"] = _re

import torch
import scipy.io.wavfile as wav
import numpy as np
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

sys.modules["overrides"] = _ro
sys.modules["overrides.enforce"] = _re
_re.EnforceOverridesMeta = _rm

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
LANG = "ja"
CLIPS = [
    ("60s",  "/Users/christianstrobele/code/issue89-stash/o_9dWkRPYC0_60s.wav"),
    ("120s", "/Users/christianstrobele/code/issue89-stash/o_9dWkRPYC0_120s.wav"),
    ("300s", "/Users/christianstrobele/code/issue89-stash/o_9dWkRPYC0_300s.wav"),
    ("600s", "/Users/christianstrobele/code/issue89-stash/o_9dWkRPYC0_600s.wav"),
]

OUT = "/Users/christianstrobele/code/issue89-stash/voxtral_groundtruth.json"

def load_wav_to_array(path):
    sr, x = wav.read(path)
    assert sr == 16000, f"need 16k mono, got {sr}"
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    return x

def main():
    print(f">> loading {MODEL_ID} (this takes a minute)", flush=True)
    t0 = time.time()
    processor = VoxtralProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                            device_map={"": "cpu"})
    model.eval()
    print(f"   loaded in {time.time()-t0:.0f}s", flush=True)

    results = {}
    for label, path in CLIPS:
        if not os.path.exists(path):
            print(f"SKIP {label}: missing {path}", flush=True)
            continue
        audio = load_wav_to_array(path)
        print(f"\n>> {label} ({len(audio)/16000:.1f}s of audio)", flush=True)

        t0 = time.time()
        inputs = processor.apply_transcription_request(language=LANG, audio=audio, model_id=MODEL_ID,
                                                       sampling_rate=16000, return_tensors="pt",
                                                       tokenize=True, return_dict=True)
        # Move to model device
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        # Strip the prompt prefix
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, prompt_len:].tolist()
        text = processor.tokenizer.tokenizer.decode(gen_ids)
        n_chars = sum(1 for c in text if not c.isspace())
        wall = time.time() - t0
        print(f"   wall={wall:.0f}s chars={n_chars}", flush=True)
        print(f"   head={text[:80]!r}", flush=True)
        print(f"   tail={text[-80:]!r}", flush=True)
        results[label] = {"chars": n_chars, "wall_s": round(wall, 1), "text": text}

        # Release GPU memory after each clip
        del inputs, out_ids
        gc.collect()

    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {OUT}", flush=True)

if __name__ == "__main__":
    main()
