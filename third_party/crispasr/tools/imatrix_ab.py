#!/usr/bin/env python3
"""imatrix_ab.py — A/B harness for importance-matrix (imatrix) quantization.

Measures whether feeding an importance matrix to the quantizer improves the
quantized ASR model's fidelity to the f16 gold, using the model's **prefill
logits** (the first-token distribution over the audio) as a fixed-length,
generation-independent parity signal — the ASR analog of CrispEmbed's
embedding-cosine A/B (tools/imatrix_ab.py there).

Everything runs serially (16 GB Mac constraint: never load two heavy models at
once). Pipeline:
  1. calibration : run crispasr with CRISPASR_IMATRIX_OUT over the calib clips
                   (merges across clips) -> <model>.imatrix.gguf
  2. quant A     : crispasr-quantize <src> <out_a> <qtype>              (baseline)
  3. quant B     : crispasr-quantize <src> <out_b> <qtype> --imatrix    (candidate)
  4. eval        : for each held-out clip, dump prefill logits (via
                   CRISPASR_ACTDUMP_OUT) from src / A / B and report
                   mean cosine(A, src) vs mean cosine(B, src).

Two signals vs the f16 gold: transcript **CER** (primary — the real quality
metric) and prefill-logit **cosine** (a proxy). Accept criterion: imatrix must
not worsen CER; cosine only breaks the tie on easy quants where transcripts
already match. NOTE: the two can diverge at aggressive bit-widths — measured on
q3_k, imatrix improved CER (0.37→0.13) while cosine dipped — so CER is the gate.

Usage:
  python tools/imatrix_ab.py --cli build/bin/crispasr \\
      --quant build/bin/crispasr-quantize --src model-f16.gguf --qtype q4_k \\
      --calib a.wav b.wav --eval c.wav d.wav [--workdir DIR] [--keep]

Note: the src model's backend must have the imatrix/actdump collector installed
on its decode scheduler (crispasr_imatrix_install) — see docs/quantize.md.
"""
import argparse, math, os, struct, subprocess, sys, time


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def read_dump(path):
    """Read an actdump file: int64 n (LE) then n float32. Returns list[float]."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<q", f.read(8))
        return list(struct.unpack(f"<{n}f", f.read(n * 4)))


def logits(cli, model, wav, out):
    """Run the model once: dump prefill logits to `out`, return (vec, transcript).
    The transcript (last non-empty stdout line) gives a second, coarser parity
    signal (CER vs the f16 gold) alongside the logit cosine."""
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ, CRISPASR_ACTDUMP_OUT=out)
    r = run([cli, "-m", model, "-f", wav], env=env)
    if not os.path.exists(out):
        sys.exit(f"no logits dumped for {os.path.basename(model)} on {os.path.basename(wav)}:\n{r.stderr[-1500:]}")
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    transcript = lines[-1] if lines else ""
    return read_dump(out), transcript


def cosine(a, b):
    n = min(len(a), len(b))
    if n == 0 or len(a) != len(b):
        return float("nan")
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(y * y for y in b[:n]))
    return dot / (na * nb) if na and nb else 0.0


def cer(hyp, ref):
    """Character error rate of hyp vs ref (edit distance / len(ref))."""
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--src", required=True, help="f16/f32 (or q8_0) source GGUF = gold")
    ap.add_argument("--qtype", default="q4_k")
    ap.add_argument("--calib", nargs="+", required=True, help="calibration wav clips")
    ap.add_argument("--eval", nargs="+", required=True, help="held-out eval wav clips")
    ap.add_argument("--workdir", default="/tmp/imatrix_ab")
    ap.add_argument("--keep", action="store_true", help="keep intermediate GGUFs")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.src))[0]
    imat = os.path.join(args.workdir, base + ".imatrix.gguf")
    out_a = os.path.join(args.workdir, f"{base}-{args.qtype}.gguf")
    out_b = os.path.join(args.workdir, f"{base}-{args.qtype}-imatrix.gguf")
    dump = os.path.join(args.workdir, "acts.bin")

    # 1. calibration (one process per clip; the collector merges into `imat`)
    if os.path.exists(imat):
        os.remove(imat)
    print(f"[1/4] calibration over {len(args.calib)} clip(s) -> {imat}")
    t0 = time.time()
    for w in args.calib:
        env = dict(os.environ, CRISPASR_IMATRIX_OUT=imat)
        r = run([args.cli, "-m", args.src, "-f", w], env=env)
        if not os.path.exists(imat):
            sys.exit(f"calibration failed on {w}:\n{r.stderr[-1500:]}")
    print(f"      done in {time.time()-t0:.1f}s  ({os.path.getsize(imat)//1024} KB)")

    # 2 + 3. quantize baseline and imatrix
    print(f"[2/4] quantize baseline  -> {out_a}")
    r = run([args.quant, args.src, out_a, args.qtype])
    if r.returncode != 0:
        sys.exit(f"baseline quant failed:\n{r.stderr[-1500:]}\n{r.stdout[-1500:]}")
    print(f"[3/4] quantize +imatrix  -> {out_b}")
    r = run([args.quant, args.src, out_b, args.qtype, "--imatrix", imat])
    if r.returncode != 0:
        sys.exit(f"imatrix quant failed:\n{r.stderr[-1500:]}\n{r.stdout[-1500:]}")
    n_im = sum(1 for l in r.stdout.splitlines() if "(imatrix)" in l)
    print(f"      {n_im} tensor(s) quantized with imatrix weighting")

    # 4. eval (serial: one model in memory at a time). Two parity signals vs the
    # f16 gold: prefill-logit cosine (higher=closer) and transcript CER (lower=closer).
    print(f"[4/4] eval over {len(args.eval)} held-out clip(s): prefill-logit cosine + transcript CER")
    rows, cos_a_all, cos_b_all, cer_a_all, cer_b_all = [], [], [], [], []
    for w in args.eval:
        name = os.path.basename(w)
        g, tg = logits(args.cli, args.src, w, dump)
        a, ta = logits(args.cli, out_a, w, dump)
        b, tb = logits(args.cli, out_b, w, dump)
        ca, cb = cosine(a, g), cosine(b, g)
        ra, rb = cer(ta, tg), cer(tb, tg)  # CER vs the f16 gold transcript
        cos_a_all.append(ca); cos_b_all.append(cb)
        cer_a_all.append(ra); cer_b_all.append(rb)
        rows.append((name, ca, cb, ra, rb))

    sz = lambda p: os.path.getsize(p) / 1e6
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("\n===== A/B RESULT (" + args.qtype + ", vs f16 gold) =====")
    print(f"  source     {os.path.basename(args.src):40s} {sz(args.src):7.1f} MB")
    print(f"  A baseline {os.path.basename(out_a):40s} {sz(out_a):7.1f} MB")
    print(f"  B +imatrix {os.path.basename(out_b):40s} {sz(out_b):7.1f} MB")
    print(f"  {'clip':22s}  cosA     cosB     dcos      CER_A   CER_B   dCER")
    for name, ca, cb, ra, rb in rows:
        print(f"  {name:22s}  {ca:.4f}   {cb:.4f}  {cb-ca:+.4f}    {ra:.4f}  {rb:.4f}  {rb-ra:+.4f}")
    ma, mb = mean(cos_a_all), mean(cos_b_all)
    ra_, rb_ = mean(cer_a_all), mean(cer_b_all)
    print(f"  {'MEAN':22s}  {ma:.4f}   {mb:.4f}  {mb-ma:+.4f}    {ra_:.4f}  {rb_:.4f}  {rb_-ra_:+.4f}")
    # Verdict gates on CER — transcript fidelity to the f16 gold is the real
    # quality signal. The prefill-logit cosine is only a proxy and can move the
    # OTHER way at aggressive bit-widths (measured: q3_k imatrix improves CER
    # while cosine dips), so it is reported but not the gate; it only breaks the
    # tie on easy quants where transcripts already match f16 (CER≈0 either way).
    eps = 1e-6
    if rb_ < ra_ - eps:
        verdict = "PASS — imatrix improves transcript CER"
    elif rb_ > ra_ + eps:
        verdict = "REGRESSION — imatrix worsens transcript CER"
    else:
        verdict = "PASS — CER tied, cosine improves" if mb >= ma - eps else "WEAK — CER tied, cosine dips"
    print(f"  VERDICT: {verdict}")
    print("  (CER vs f16 = transcript fidelity = primary signal; prefill-logit")
    print("   cosine is a proxy that can diverge at aggressive bit-widths.)")

    if not args.keep:
        for p in (out_a, out_b, dump):
            if os.path.exists(p):
                os.remove(p)
        print("  (removed intermediate GGUFs; --keep to retain)")


if __name__ == "__main__":
    main()
