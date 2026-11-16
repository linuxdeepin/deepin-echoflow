"""Independent Phonon-2 reference: upstream reader + stock ParakeetForTDT.

fermion_container.py is vendored unchanged from FermionResearch/Phonon-2
(Copyright 2026 Fermion Research, Apache-2.0; see LICENSE). Its decoding is
independent of models/phonon2_container.py. No downloaded Python is executed.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np


def load(snapshot: Path, converter):
    import torch
    from transformers import GenerationConfig, ParakeetForTDT, ParakeetTDTConfig
    from huggingface_hub import snapshot_download

    from .fermion_container import read_container

    md = converter.phonon2_reader().materialize(snapshot)
    base = os.environ.get("PHONON2_BASE_DIR")
    if base is None:
        base = snapshot_download("nvidia/parakeet-tdt-0.6b-v3", allow_patterns=["*.json"])
    cfg = ParakeetTDTConfig.from_pretrained(base)
    # Avoid allocating a second 2.5 GB weight copy on CPU validation hosts.
    with torch.device("meta"):
        model = ParakeetForTDT(cfg)
    # This analytic basis is nonpersistent: assign=True cannot restore it
    # from the checkpoint. Construct it through the actual upstream module.
    model.encoder.encode_positions = type(model.encoder.encode_positions)(cfg.encoder_config)
    tensors, _ = read_container(str(md / "model.fermion"))
    sd = {}
    for name in list(tensors):
        arr = tensors.pop(name)  # release each fp16 source as its fp32 copy is made
        if name.endswith("num_batches_tracked"):
            value = float(arr)
            sd[name] = torch.tensor(int(value) if np.isfinite(value) else 0, dtype=torch.int64)
        else:
            value = np.ascontiguousarray(arr.astype(np.float32))
            if value.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", name):
                value = value[:, :, None]
            sd[name] = torch.from_numpy(value)
    del tensors
    model.load_state_dict(sd, strict=True, assign=True)
    model.generation_config = GenerationConfig.from_pretrained(base)
    return model.eval(), Path(base)
