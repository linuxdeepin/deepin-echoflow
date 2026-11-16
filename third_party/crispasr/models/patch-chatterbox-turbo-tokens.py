#!/usr/bin/env python
"""
Patch a chatterbox-turbo T3 GGUF so its embedded tokenizer covers the full
text_vocab_size (#181). The published turbo GGUFs embed only the 50257-token
base GPT-2 vocab (vocab.json), but the T3 text embedding is 50276 — the extra
19 ids (50257..50275) are the turbo emotion/style control tokens ([laugh],
[whispering], [angry], …) that live in added_tokens.json, which the original
converter dropped. This rewrites `tokenizer.ggml.tokens` (and the legacy
`chatterbox.t3.text_tokens` if present) to the full 50276-token array; all other
KV (merges, hparams) and all tensors are copied byte-for-byte — NO re-quantize.

Usage:
  python patch-chatterbox-turbo-tokens.py MODEL_DIR INPUT.gguf OUTPUT.gguf
where MODEL_DIR holds vocab.json + added_tokens.json (the ResembleAI/chatterbox-
turbo HF snapshot).
"""
import sys
import json
from pathlib import Path

try:
    import gguf
except ImportError:
    sys.exit("install: pip install gguf")

TOKEN_KEYS = ("tokenizer.ggml.tokens", "chatterbox.t3.text_tokens")


def build_tokens(model_dir: Path):
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    added_path = model_dir / "added_tokens.json"
    added = json.loads(added_path.read_text(encoding="utf-8")) if added_path.exists() else {}
    max_id = max([max(vocab.values())] + list(added.values()))
    tokens = [""] * (max_id + 1)
    for t, i in vocab.items():
        tokens[i] = t
    for t, i in added.items():
        if 0 <= i < len(tokens):
            tokens[i] = t
    print(f"built tokens: {len(tokens)} ({len(vocab)} base + {len(added)} added_tokens)")
    return tokens


def main(argv):
    if len(argv) != 4:
        sys.exit(f"usage: {argv[0]} MODEL_DIR INPUT.gguf OUTPUT.gguf")
    model_dir, inp, out = Path(argv[1]), Path(argv[2]), Path(argv[3])
    tokens = build_tokens(model_dir)

    reader = gguf.GGUFReader(inp)
    arch = "chatterbox"
    for kv in reader.fields.values():
        if kv.name == "general.architecture":
            arch = bytes(kv.parts[kv.data[0]]).decode("utf-8", "replace")
            break
    writer = gguf.GGUFWriter(out, arch, use_temp_file=False)

    AUTO_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for name, field in reader.fields.items():
        if name in AUTO_KEYS or name in TOKEN_KEYS:
            continue
        try:
            val = field.contents()
        except Exception:
            val = None
        if val is None:
            continue
        t = field.types[0] if field.types else None
        T = gguf.GGUFValueType
        if t == T.UINT32:
            writer.add_uint32(name, int(val))
        elif t == T.INT32:
            writer.add_int32(name, int(val))
        elif t == T.FLOAT32:
            writer.add_float32(name, float(val))
        elif t == T.BOOL:
            writer.add_bool(name, bool(val))
        elif t == T.STRING:
            writer.add_string(name, str(val))
        elif t == T.UINT64:
            writer.add_uint64(name, int(val))
        elif t == T.INT64:
            writer.add_int64(name, int(val))
        elif t == T.FLOAT64:
            writer.add_float64(name, float(val))
        elif t == T.ARRAY:
            elem_t = field.types[1] if len(field.types) > 1 else None
            if elem_t == T.STRING:
                writer.add_array(name, [str(v) for v in val])
            else:
                writer.add_array(name, list(val))
        else:
            print(f"skip unknown type for {name}: {t}")

    # Rewrite the token arrays present in the source (don't invent text_tokens
    # if the original didn't carry it — keep the file's shape).
    present = {k for k in reader.fields if k in TOKEN_KEYS}
    for k in present:
        writer.add_array(k, tokens)
        print(f"rewrote {k}: {len(tokens)} tokens")
    if "tokenizer.ggml.tokens" not in present:
        writer.add_array("tokenizer.ggml.tokens", tokens)
        print("added tokenizer.ggml.tokens (was absent)")

    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote: {out}")


if __name__ == "__main__":
    main(sys.argv)
