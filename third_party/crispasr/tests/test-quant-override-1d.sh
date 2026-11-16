#!/usr/bin/env bash
# Regression guard for #461: `--tensor-type <regex>=f16` must not lower 1-D
# tensors (biases, norm weights) below F32. ggml's CPU add aborts on F32 + F16,
# so '^locdit\.=f16' used to build a voxcpm2 file that crashed CPU synthesis
# (Vulkan tolerated it). 2-D weights must still take the override, and an
# explicit =f32 override on a 1-D tensor must still apply.
set -euo pipefail

QUANT="${1:-}"
[ -x "$QUANT" ] || { echo "SKIP: crispasr-quantize binary not found"; exit 0; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP/in.gguf" <<'PY'
import struct
import sys

# name -> shape (ggml ne order). Hand-written GGUF: no NumPy in the unit image.
tensors = [
    ("locdit.blk.0.attn_q.weight", (256, 32)),  # 2-D: takes the f16 override
    ("locdit.blk.0.attn_q.bias", (32,)),        # 1-D: must stay F32
    ("locdit.output_norm.weight", (256,)),      # 1-D: must stay F32
    ("tslm.blk.0.attn_q.weight", (256, 32)),    # not matched: base q8_0
]


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


with open(sys.argv[1], "wb") as f:
    f.write(b"GGUF")
    f.write(struct.pack("<IQQ", 3, len(tensors), 1))
    f.write(string("general.architecture"))
    f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
    f.write(string("voxcpm2"))
    offset = 0
    sizes = []
    for name, shape in tensors:
        f.write(string(name))
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<IQ", 0, offset))  # GGML_TYPE_F32
        n = 1
        for d in shape:
            n *= d
        sizes.append(n)
        offset += (n * 4 + 31) // 32 * 32
    f.write(b"\0" * ((-f.tell()) % 32))
    for n in sizes:
        data = b"".join(struct.pack("<f", ((i % 17) - 8) / 8.0) for i in range(n))
        f.write(data + b"\0" * ((-len(data)) % 32))
PY

line_for() { grep -F "$1" "$2" | head -1 || true; }
require() {  # require <log> <tensor> <regex the decision line must match>
    local line
    line="$(line_for "$2" "$1")"
    [ -n "$line" ] || { echo "FAIL: no quantizer decision for $2"; cat "$1"; exit 1; }
    echo "$line" | grep -Eqi "$3" || { echo "FAIL: $2 should match /$3/, got: $line"; exit 1; }
}
forbid() {  # forbid <log> <tensor> <regex the decision line must NOT match>
    local line
    line="$(line_for "$2" "$1")"
    [ -n "$line" ] || { echo "FAIL: no quantizer decision for $2"; cat "$1"; exit 1; }
    if echo "$line" | grep -Eqi "$3"; then echo "FAIL: $2 must not match /$3/, got: $line"; exit 1; fi
}

# 1) the #461 override: 2-D LocDiT weight -> F16, 1-D bias/norm untouched
LOG1="$TMP/q1.log"
"$QUANT" "$TMP/in.gguf" "$TMP/out1.gguf" q8_0 --tensor-type '^locdit\.=f16' >"$LOG1" 2>&1
require "$LOG1" "locdit.blk.0.attn_q.weight" "f16"
forbid "$LOG1" "locdit.blk.0.attn_q.bias" "f16"
forbid "$LOG1" "locdit.output_norm.weight" "f16"
require "$LOG1" "tslm.blk.0.attn_q.weight" "q8_0"

# 2) an explicit =f32 override on a 1-D tensor is still honoured (keeps norms precise)
LOG2="$TMP/q2.log"
"$QUANT" "$TMP/in.gguf" "$TMP/out2.gguf" q8_0 --tensor-type 'output_norm=f32' >"$LOG2" 2>&1
forbid "$LOG2" "locdit.output_norm.weight" "f16|q8_0"

echo "PASS: --tensor-type lowers only 2-D tensors; 1-D biases/norms stay F32"
