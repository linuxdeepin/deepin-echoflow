"""Shared guard against HF `_init_weights` clobbering pretrained weights.

Some custom remote model classes (loaded via `from_pretrained(...,
trust_remote_code=True)`) run a `_init_weights` that re-initializes all
Linear/Conv weights to `normal(0, 0.02)` on load, silently overwriting the
pretrained values -> a reference dump of pure noise. This was first hit with
CohereAsrPreTrainedModel (fixed inline in cohere.py, commit c9a4de65).

`reload_if_random_init` detects the symptom (a conv-like weight whose RMS is
far below any trained value) and reloads every parameter from the model's
safetensors shard(s). It is a no-op when the weights are already pretrained,
so it is safe to call unconditionally after loading any such model.
"""
from __future__ import annotations

import glob
from pathlib import Path


def reload_if_random_init(model, model_dir, rms_threshold: float = 0.1) -> bool:
    """If `model` looks random-initialized, reload all params from safetensors.

    Returns True if a reload was performed, False otherwise. Never raises.
    """
    try:
        conv0 = next(
            (m.weight for m in model.modules()
             if getattr(getattr(m, "weight", None), "dim", lambda: 0)() >= 3),
            None,
        )
        if conv0 is None:
            return False
        rms = float(conv0.norm() / conv0.numel() ** 0.5)
        if rms >= rms_threshold:
            return False
        print(f"  WARNING: weights look random-init (conv rms={rms:.4f} < "
              f"{rms_threshold}); reloading all params from safetensors")
        from safetensors import safe_open
        sd = dict(model.named_parameters())
        shards = sorted(glob.glob(str(Path(model_dir) / "*.safetensors")))
        n = 0
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if key in sd:
                        try:
                            sd[key].data.copy_(f.get_tensor(key).float())
                            n += 1
                        except Exception:
                            pass
        print(f"  reloaded {n} params from {len(shards)} safetensors shard(s)")
        return n > 0
    except Exception as e:  # never let the guard break a dump
        print(f"  (init-weights reload guard skipped: {e})")
        return False
