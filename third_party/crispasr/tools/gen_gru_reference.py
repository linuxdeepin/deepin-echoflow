#!/usr/bin/env python3
"""Generate the PyTorch nn.GRU fixture that tests/test-core-gru.cpp checks against.

    python tools/gen_gru_reference.py [out.bin] [--in 5 --hidden 4 --steps 7]

Writes: int32 I,H,T; float32 x(T,I); then weight_ih_l0, weight_hh_l0,
bias_ih_l0, bias_hh_l0 and their *_reverse counterparts (all row-major, as
torch stores them); then the expected output (T, 2H).

Run the test with CRISPASR_GRU_REF pointing at the result. Without it the
parity case skips, so CI needs neither torch nor network.
"""
import sys, struct
import numpy as np
import torch

def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "gru_ref.bin"
    I, H, T = 5, 4, 7
    torch.manual_seed(0)
    g = torch.nn.GRU(I, H, num_layers=1, bidirectional=True, batch_first=False)
    x = torch.randn(T, 1, I)
    with torch.no_grad():
        y, _ = g(x)
    sd = {k: v.detach().numpy() for k, v in g.state_dict().items()}
    with open(out, "wb") as f:
        f.write(struct.pack("<iii", I, H, T))
        f.write(x[:, 0, :].numpy().astype(np.float32).tobytes())
        for k in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
                  "weight_ih_l0_reverse", "weight_hh_l0_reverse",
                  "bias_ih_l0_reverse", "bias_hh_l0_reverse"]:
            f.write(np.ascontiguousarray(sd[k]).astype(np.float32).tobytes())
        f.write(y[:, 0, :].numpy().astype(np.float32).tobytes())
    print(f"wrote {out}  I={I} H={H} T={T}")

if __name__ == "__main__":
    main()
