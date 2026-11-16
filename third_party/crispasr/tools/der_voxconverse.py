#!/usr/bin/env python3
"""Score a CrispASR diarization arm against VoxConverse dev reference labels.

The driver behind the DER table in PLAN.md "#326". `der_score.py` is the metric;
this is the harness that produces the hypothesis: run the CLI once per file,
read `--output-json`'s per-segment `offsets` + `speaker`, and score against the
human labels.

Reference labels come from the HuggingFace VoxConverse dataset (`diarizers-community/
voxconverse`), whose parquet rows carry `timestamps_start` / `timestamps_end` /
`speakers` alongside the audio. `--prepare` extracts both from the parquet shards
so the corpus is reproducible from one command; everything after reads the
extracted `<stem>.wav` + `<stem>.ref.json` pairs.

    # one-time corpus extraction (~2 GB of parquet in, ~40 MB out)
    python tools/der_voxconverse.py --prepare /path/to/voxconverse/data --audio-dir /tmp/vox

    # score an arm
    python tools/der_voxconverse.py --audio-dir /tmp/vox \\
        --bin build/bin/crispasr --model ggml-tiny.bin --cache-dir ~/.cache/crispasr \\
        --args "--diarize --diarize-method pyannote --diarize-embedder auto" \\
        --label baseline

Speaker COUNT is reported next to DER on purpose: an over-clustering path can
post a mediocre DER for the specific reason that it pinned to
`--diarize-max-speakers`, and the count is what says so.
"""

import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from der_score import score

# The 8-file dev shard the #326 / #324 numbers are quoted on, with the reference
# speaker count of each. Pinned so an arm is never accidentally scored on a
# different subset than the number it is being compared to.
SHARD = {
    "esrit": 5, "fsaal": 7, "jyirt": 4, "mesob": 4,
    "nnqfq": 5, "rcxzg": 4, "tiams": 5, "willh": 2,
}


def prepare(parquet_dir, out_dir):
    import glob
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    found = {}
    for f in sorted(glob.glob(os.path.join(parquet_dir, "dev-*.parquet"))):
        t = pq.read_table(f).combine_chunks()
        au = t.column("audio").chunk(0)
        paths, byts = au.field("path").to_pylist(), au.field("bytes")
        ts = t.column("timestamps_start").chunk(0)
        te = t.column("timestamps_end").chunk(0)
        sp = t.column("speakers").chunk(0)
        for i, p in enumerate(paths):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem not in SHARD or stem in found:
                continue
            open(os.path.join(out_dir, stem + ".wav"), "wb").write(byts[i].as_py())
            ref = list(zip(ts[i].as_py(), te[i].as_py(), sp[i].as_py()))
            json.dump(ref, open(os.path.join(out_dir, stem + ".ref.json"), "w"))
            found[stem] = len({s for _, _, s in ref})
        if len(found) == len(SHARD):
            break
    missing = set(SHARD) - set(found)
    if missing:
        sys.exit(f"missing from the parquet shards: {sorted(missing)}")
    for stem, n in sorted(found.items()):
        if n != SHARD[stem]:
            sys.exit(f"{stem}: extracted {n} reference speakers, expected {SHARD[stem]}")
    print(f"prepared {len(found)} files in {out_dir}; reference speaker counts match")


def hypothesis(js):
    """CLI --output-json → [(start_s, end_s, speaker), ...]."""
    out = []
    for s in js.get("transcription", []):
        spk = (s.get("speaker") or "").strip()
        if not spk:
            continue
        o = s["offsets"]
        a, b = o["from"] / 1000.0, o["to"] / 1000.0
        if b > a:
            out.append((a, b, spk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", metavar="PARQUET_DIR")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--bin", default="build/bin/crispasr")
    ap.add_argument("--model")
    ap.add_argument("--cache-dir")
    ap.add_argument("--args", default="", help="extra CLI args, one string")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--out-json", help="write the per-file table here")
    a = ap.parse_args()

    if a.prepare:
        prepare(a.prepare, a.audio_dir)
        return

    rows, ders = [], []
    for stem in sorted(SHARD):
        wav = os.path.join(a.audio_dir, stem + ".wav")
        ref = json.load(open(os.path.join(a.audio_dir, stem + ".ref.json")))
        ref = [(s, e, spk) for s, e, spk in ref if e > s]
        outbase = os.path.join(a.audio_dir, f".{a.label}.{stem}")
        cmd = [a.bin, "-m", a.model, "-f", wav, "-l", "en", "-oj", "-of", outbase]
        if a.cache_dir:
            cmd += ["--cache-dir", a.cache_dir]
        cmd += a.args.split()
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        if r.returncode != 0 or not os.path.exists(outbase + ".json"):
            # A failed run is a FAIL, never a silently-skipped file — a missing
            # arm would otherwise flatter the mean it is dropped from.
            sys.exit(f"{stem}: rc={r.returncode}\n{r.stderr[-1500:]}")
        hyp = hypothesis(json.load(open(outbase + ".json")))
        s = score(ref, hyp, collar=a.collar)
        n_hyp = len({x[2] for x in hyp})
        rows.append((stem, SHARD[stem], n_hyp, 100 * s["der"], wall))
        ders.append(100 * s["der"])

    print(f"\n=== {a.label} (collar {a.collar}s) ===")
    print(f"{'file':8} {'GT':>3} {'hyp':>4} {'DER%':>8} {'wall_s':>8}")
    for stem, gt, nh, der, wall in rows:
        flag = "  <-- capped?" if nh >= 8 else ("  <-- undercount" if nh < gt else "")
        print(f"{stem:8} {gt:3d} {nh:4d} {der:8.2f} {wall:8.1f}{flag}")
    mean = sum(ders) / len(ders)
    print(f"{'MEAN':8} {'':3} {'':4} {mean:8.2f}")
    if a.out_json:
        json.dump({"label": a.label, "mean_der": mean,
                   "files": [{"file": s, "gt": g, "hyp": h, "der": d} for s, g, h, d, _ in rows]},
                  open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
