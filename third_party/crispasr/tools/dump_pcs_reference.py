#!/usr/bin/env python3
"""PCS diff-harness reference: run the source ONNX model and dump per-token head
predictions, to compare against the CrispEmbed/CrispASR pcs engine.

The engine mirrors these heads (post-punct, pre-punct, sbd/segmentation,
truecase); dump the engine side with `PCS_DEBUG=1 test-punct-diff <gguf> "<text>"`
(and `PCS_FORCE_CPU=1` to match onnxruntime's CPU float order on borderline
logits). Token ids must match between the two for an apples-to-apples compare.

Usage:
    HF_HOME=/path/to/cache python tools/dump_pcs_reference.py "hello world how are you today"

Model: 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase (ONNX).
Outputs argmax predictions (not logits): post_preds, pre_preds, seg_preds,
cap_preds[seq,16]. cap_preds[t][0] is the ▁ (word-start) slot; the first visible
character uses cap_preds[t][1] (the engine's "predictions start after ▁").
"""
import sys

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download

REPO = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
POST = ["<NULL>", "<ACRONYM>", ".", ",", "?", "？", "，", "。", "、", "・", "।", "؟", "،", ";", "።", "፣", "፧"]
PRE = ["<NULL>", "¿"]


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "hello world how are you today i am fine thanks"
    onnx_path = hf_hub_download(REPO, "model.onnx")
    sp_path = hf_hub_download(REPO, "sp.model")

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_path)
    pieces = sp.EncodeAsPieces(text)
    ids = [sp.PieceToId(p) for p in pieces]
    # RoBERTa wrap: <s>=0 ... </s>=2
    inp = np.array([[0] + ids + [2]], dtype=np.int64)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input_ids": inp})))
    strip = lambda x: x[0][1:-1]  # drop CLS/SEP
    post, pre, seg, cap = (strip(out[k]) for k in ("post_preds", "pre_preds", "seg_preds", "cap_preds"))

    print(f"token_ids: {' '.join(map(str, ids))}")
    print(f"{'idx':>3} {'piece':<12} {'post':<8} {'pre':<6} {'seg':<3} cap[0:6]")
    txt = ""
    for i, p in enumerate(pieces):
        print(f"{i:>3} {p.replace(chr(0x2581), '_'):<12} {POST[post[i]]:<8} {PRE[pre[i]]:<6} "
              f"{int(seg[i]):<3} {list(map(int, cap[i][:6]))}")
        raw = p.replace("▁", " ")
        lead, chars = raw.startswith(" "), raw.strip()
        capped = "".join(c.upper() if (j + 1 < 16 and cap[i][j + 1]) else c for j, c in enumerate(chars))
        if lead and txt:
            txt += " "
        if PRE[pre[i]] != "<NULL>":
            txt += PRE[pre[i]]
        txt += capped
        if POST[post[i]] not in ("<NULL>", "<ACRONYM>"):
            txt += POST[post[i]]
    print(f"\nreference decoded: {txt}")


if __name__ == "__main__":
    main()
