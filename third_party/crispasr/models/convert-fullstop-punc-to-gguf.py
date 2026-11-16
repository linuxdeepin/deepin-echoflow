#!/usr/bin/env python3
"""Convert oliverguhr/fullstop-punctuation-multilang-large to GGUF.

Architecture: XLM-RoBERTa-large (24L, d=1024, 16 heads, d_ffn=4096)
              + Linear(1024, 6) classifier head.
              6 classes: 0(none), .(period), ,(comma), ?(question), -(dash), :(colon)

Usage:
    python models/convert-fullstop-punc-to-gguf.py \
        --input oliverguhr/fullstop-punctuation-multilang-large \
        --output fullstop-punc.gguf
"""

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--restore-zeroed-embeddings-from", metavar="HF_MODEL", default=None,
        help="Fill all-zero token-embedding rows from this model (e.g. xlm-roberta-base). "
             "kredor/punctuate-all zeroes 9531 rows — four contiguous ranges that look like "
             "pruned token ranges for languages it does not serve. A zero embedding gives the "
             "encoder no information for those tokens; the base model's pretrained vector is "
             "what the fine-tune started from, and kredor's surviving rows sit only ~0.05 away "
             "from base, so restoring base is close to undoing the prune. OFF by default: it "
             "makes the artifact diverge from what transformers loads, so it is a deliberate "
             "choice, not a silent improvement.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

    print(f"Loading model from {args.input}...")
    config = AutoConfig.from_pretrained(args.input)
    model = AutoModelForTokenClassification.from_pretrained(args.input, dtype=torch.float32)
    sd = model.state_dict()
    print(f"Loaded {len(sd)} tensors")

    # Architecture params
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    d_ffn = config.intermediate_size
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size
    max_pos = config.max_position_embeddings
    n_classes = config.num_labels

    print(f"  Layers: {n_layers}, d_model: {d_model}, d_ffn: {d_ffn}, heads: {n_heads}")
    print(f"  Vocab: {vocab_size}, max_pos: {max_pos}, classes: {n_classes}")
    print(f"  Labels: {config.id2label}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.input)

    # Get vocab as list of strings
    vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]

    # Unigram piece scores (log-probs) — required for correct Viterbi tokenization.
    # XLM-RoBERTa's SP model is a Unigram model; greedy longest-match mis-segments
    # multi-subword words, so the engine runs Viterbi over these scores.
    # Two sources, because not every checkpoint ships the same files. This used
    # to demand `sentencepiece.bpe.model` unconditionally and die with a 404 on
    # any repo that ships only the fast tokenizer — kredor/punctuate-all, for
    # one, which is a model this script is otherwise perfectly able to convert.
    #
    # Getting the scores is not optional. Without them the runtime falls back to
    # greedy longest-match, and XLM-R's SP model is Unigram, where greedy is
    # simply the wrong algorithm: `fox` has no `▁fox` piece, so Viterbi gives
    # `▁`+`fox` while greedy takes the longest prefix `▁fo` and is left with `x`.
    # Different ids, different embeddings, silently wrong output. The shipped
    # punctuate-all GGUF carries only `tokenizer.ggml.tokens` and mis-segments
    # exactly that way — it predates this block.
    from huggingface_hub import hf_hub_download

    scores = None
    try:
        import sentencepiece as spm

        sp_path = hf_hub_download(args.input, "sentencepiece.bpe.model")
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_path)
        scores = []
        for tok in vocab:
            sid = sp.PieceToId(tok)  # 0 (<unk>) for HF specials not in the SP model
            scores.append(float(sp.GetScore(sid)) if sid > 0 else 0.0)
        print(f"  SP unigram scores from sentencepiece.bpe.model: {len(scores)}")
    except Exception as e:
        print(f"  no sentencepiece.bpe.model ({type(e).__name__}); trying tokenizer.json")

    if scores is None:
        # tokenizer.json's Unigram model carries [piece, score] pairs directly —
        # the same numbers, just serialised by `tokenizers` instead of by
        # sentencepiece.
        import json

        tj = json.load(open(hf_hub_download(args.input, "tokenizer.json")))
        model = tj.get("model", {})
        if model.get("type") != "Unigram":
            raise RuntimeError(f"tokenizer.json model type is {model.get('type')!r}, expected Unigram")
        piece_score = {p: float(sc) for p, sc in model["vocab"]}
        scores = [piece_score.get(tok, 0.0) for tok in vocab]
        print(f"  SP unigram scores from tokenizer.json: {len(scores)}")

    nonzero = sum(1 for x in scores if x != 0.0)
    print(f"  nonzero scores: {nonzero}/{len(scores)}")
    # A vocab of 250k with almost nothing scored means the lookup silently missed
    # (a piece-naming mismatch, say) and every score defaulted to 0.0 — which
    # makes Viterbi degenerate and is worse than not shipping scores at all,
    # because the runtime would then trust them.
    if nonzero < len(scores) // 2:
        raise RuntimeError(f"only {nonzero}/{len(scores)} pieces scored — the score lookup did not match the vocab")

    # Get labels
    labels = [config.id2label[i] for i in range(n_classes)]
    # Map '0' to space
    labels = [' ' if l == '0' else l for l in labels]
    print(f"  Labels (mapped): {labels}")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "fireredpunc")
    # Name the model this actually IS, not the one this script was first written
    # for. The literal "fullstop-punctuation-multilang-large" used to be
    # hard-coded here regardless of --input, so a GGUF converted from any other
    # checkpoint self-identified as the large model. That is not cosmetic: it is
    # the provenance a reader checks when a result looks wrong, and it is the
    # attribution that has to be right for the licence. The shipped
    # `punctuate-all-*.gguf` carries the wrong name for exactly this reason — it
    # is kredor/punctuate-all, XLM-R BASE (12L, d=768), while the name claims the
    # 24L/1024 large model. Re-convert to correct an existing artifact.
    writer.add_name(os.path.basename(str(args.input).rstrip("/")))

    writer.add_uint32("fireredpunc.d_model", d_model)
    writer.add_uint32("fireredpunc.d_ffn", d_ffn)
    writer.add_uint32("fireredpunc.n_heads", n_heads)
    writer.add_uint32("fireredpunc.n_layers", n_layers)
    writer.add_uint32("fireredpunc.vocab_size", vocab_size)
    writer.add_uint32("fireredpunc.max_pos", max_pos)
    writer.add_uint32("fireredpunc.n_classes", n_classes)
    writer.add_uint32("fireredpunc.cls_id", 0)   # <s> for RoBERTa
    writer.add_uint32("fireredpunc.pad_id", 1)   # <pad>
    writer.add_string("fireredpunc.tokenizer_type", "sentencepiece")

    writer.add_array("tokenizer.ggml.tokens", vocab)
    writer.add_array("tokenizer.ggml.scores", scores)
    writer.add_array("fireredpunc.labels", labels)

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Shorten tensor names — same as FireRedPunc but with roberta. prefix
    def shorten(name):
        name = name.replace("roberta.embeddings.", "emb.")
        name = name.replace("roberta.encoder.layer.", "enc.")
        name = name.replace("attention.self.query.", "attn.q.")
        name = name.replace("attention.self.key.", "attn.k.")
        name = name.replace("attention.self.value.", "attn.v.")
        name = name.replace("attention.output.dense.", "attn.out.")
        name = name.replace("attention.output.LayerNorm.", "attn.ln.")
        name = name.replace("intermediate.dense.", "ffn.up.")
        name = name.replace("output.dense.", "ffn.down.")
        name = name.replace("output.LayerNorm.", "ffn.ln.")
        name = name.replace("word_embeddings.", "tok_emb.")
        name = name.replace("position_embeddings.", "pos_emb.")
        name = name.replace("token_type_embeddings.", "type_emb.")
        name = name.replace("LayerNorm.", "ln.")
        name = name.replace("classifier.", "cls.")
        return name

    # Optional: undo the embedding prune. Done on the state dict BEFORE the write
    # loop so the restored rows go through the same f16 path as everything else.
    if args.restore_zeroed_embeddings_from:
        from transformers import AutoModel

        emb_key = next(k for k in sd if k.endswith("word_embeddings.weight"))
        W = sd[emb_key]
        norms = torch.linalg.norm(W.float(), dim=1)
        zero_rows = (norms < 1e-3).nonzero().flatten()
        print(f"  zeroed embedding rows: {len(zero_rows)} / {W.shape[0]}")
        if len(zero_rows):
            base = AutoModel.from_pretrained(args.restore_zeroed_embeddings_from, dtype=torch.float32)
            Wb = base.get_input_embeddings().weight.detach()
            if Wb.shape != W.shape:
                raise RuntimeError(f"embedding shape mismatch: base {tuple(Wb.shape)} vs model {tuple(W.shape)}")
            W = W.clone()
            W[zero_rows] = Wb[zero_rows].to(W.dtype)
            sd[emb_key] = W
            print(f"  restored {len(zero_rows)} rows from {args.restore_zeroed_embeddings_from}")

    tensor_count = 0
    for name in sorted(sd.keys()):
        t = sd[name].float().numpy()
        gguf_name = shorten(name)

        # Norms/biases as F32, weights as F16.
        # Embedding table stays F32 — F16 causes precision loss through 24 layers.
        if "ln." in gguf_name or "LayerNorm" in name or name.endswith(".bias") or len(t.shape) <= 1:
            data = f32(t)
        elif "tok_emb" in gguf_name or "pos_emb" in gguf_name or "type_emb" in gguf_name:
            data = f32(t)
        else:
            data = f16(t)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 50 == 0:
            print(f"  [{tensor_count}] {gguf_name:50s} {str(data.shape):20s}")

    print(f"  Total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e6:.1f} MB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
