#!/usr/bin/env python3
"""Diarization evaluation: speaker-count accuracy AND DER, on a tune/holdout split.

Why this exists
---------------
The diarization work in #324/#326 was steered by DER on an 8-file subset, and
that turned out to be too weak a signal in two separate ways:

  * DER HIDES COUNT ERRORS. A survey of the estimator found 4 of 8 files
    predicting the wrong number of speakers while the shard still averaged
    7.32% DER, because merging two speakers only costs the frames of the one
    that got absorbed. If the speaker count is part of what you ship, measure
    it directly.
  * 8 FILES IS NOT ENOUGH TO TUNE ON. With a handful of files it is trivially
    easy to "fix" a constant against the one file that happens to be
    interesting. This script therefore splits the corpus and reports TUNE and
    HOLDOUT separately: change things against TUNE, report the number from
    HOLDOUT, and never the other way round.

The split is by hash of the file name, so it is stable when files are added and
does not depend on directory order.

Usage
-----
    python tools/diarize_eval.py --wav-dir DIR --ref DIR/ref.json \\
        --cmd './build/bin/crispasr -m MODEL -f {wav} -t 8 --diarize \\
               --diarize-method foxnose --diarize-embedder EMB -oj -of {out}'

`{wav}` and `{out}` are substituted per file; the command must write {out}.json
in CrispASR's JSON layout. Add --jobs N to run files concurrently.

Corpus
------
tools/voxconverse_extract.py builds one from the HF dataset
`diarizers-community/voxconverse` (CC, no login): dev is 5 parquet shards
(~216 files) and test is 11 (~232). Develop against dev's TUNE/HOLDOUT and keep
**test** untouched, so there is still an honest number left to quote at the end.
One dev shard alone is 44 files / 4.2 h with 1-17 speakers.

    --subset N     evaluate only the first N files of each split (smoke runs)
    --max-speakers N   report how many files are unwinnable under that cap
"""

import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from der_score import score


def split_of(name, holdout_frac=0.5):
    """Stable tune/holdout assignment from the file name alone."""
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    return "holdout" if (h % 100) / 100.0 < holdout_frac else "tune"


def run_one(name, wav, cmd_tpl, workdir):
    # NOTHING in here may raise. A 101-file sweep died after 6 hours because an
    # exception in one file propagated out of ThreadPoolExecutor.map and took
    # the whole arm with it — losing the other 100 results AND the reason. A
    # per-file failure must be recorded and stepped over, never fatal.
    try:
        out = os.path.join(workdir, name)
        cmd = cmd_tpl.replace("{wav}", wav).replace("{out}", out)
        # errors="replace": decoding is done on the CHILD's output, and a
        # transcript carrying odd bytes would otherwise raise UnicodeDecodeError
        # inside communicate() — a failure of the harness, not of the run.
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace")
        path = out + ".json"
        if r.returncode != 0 or not os.path.exists(path):
            return name, None, f"rc={r.returncode} " + (r.stderr or "")[-400:]
        segs = json.load(open(path))["transcription"]
    except Exception as e:  # noqa: BLE001 — deliberately total
        return name, None, f"{type(e).__name__}: {e}"
    try:
        hyp = [
            (s["offsets"]["from"] / 1000.0, s["offsets"]["to"] / 1000.0, s["speaker"])
            for s in segs
            if s.get("speaker") not in (None, "", "?")
        ]
    except Exception as e:  # noqa: BLE001
        return name, None, f"bad segments: {type(e).__name__}: {e}"
    return name, hyp, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument(
        "--split",
        choices=["tune", "holdout", "both"],
        default="both",
        help="which split to RUN. Use 'tune' while developing: not computing holdout at all is a stronger "
        "guarantee than computing it and promising not to look.",
    )
    ap.add_argument("--max-speakers", type=int, default=8)
    ap.add_argument("--json-out", default="")
    ap.add_argument(
        "--workdir",
        default="",
        help="where per-file transcripts are written (default: system temp). Point this at a disk with "
        "room: a full system volume gets runs killed mid-sweep, which looks like a crash and is not one.",
    )
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    names = sorted(ref)
    by_split = {"tune": [], "holdout": []}
    for n in names:
        by_split[split_of(n)].append(n)
    if args.split != "both":
        for k in list(by_split):
            if k != args.split:
                by_split[k] = []
    if args.subset:
        for k in by_split:
            by_split[k] = by_split[k][: args.subset]

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="diarize-eval-", dir=args.workdir)
    else:
        workdir = tempfile.mkdtemp(prefix="diarize-eval-")
    free_gb = shutil.disk_usage(workdir).free / 2**30
    if free_gb < 2.0:
        print(f"warning: only {free_gb:.1f} GB free on the work volume ({workdir}); "
              f"use --workdir to point somewhere with room", file=sys.stderr)
    results = {}
    todo = [(n, os.path.join(args.wav_dir, n + ".wav")) for k in by_split for n in by_split[k]]

    def work(item):
        return run_one(item[0], item[1], args.cmd, workdir)

    done = 0
    total = len(todo)

    # Append each result as it lands. A sweep that is killed at hour six must
    # still leave behind what it finished — the Kaggle run that motivated this
    # lost 101 files of work AND the reason, because everything was held in
    # memory until a final write that never happened.
    inc = open(args.json_out + ".partial", "w", buffering=1) if args.json_out else None

    def note(name, err):
        nonlocal done
        done += 1
        print(f"  [{done}/{total}] {name}{'  FAILED: ' + err[:120] if err else ''}", file=sys.stderr, flush=True)
        if inc:
            inc.write(json.dumps({"file": name, "error": err}) + "\n")

    if args.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for name, hyp, err in ex.map(work, todo):
                results[name] = (hyp, err)
                note(name, err)
    else:
        for item in todo:
            name, hyp, err = work(item)
            results[name] = (hyp, err)
            note(name, err)

    rows = []
    for n in names:
        if n not in results:
            continue
        hyp, err = results[n]
        gt = len({s["speaker"] for s in ref[n]})
        if hyp is None:
            rows.append({"file": n, "split": split_of(n), "gt": gt, "hyp_k": None, "der": None, "error": err})
            continue
        turns = [(s["start"], s["end"], s["speaker"]) for s in ref[n]]
        try:
            d = score(turns, hyp, collar=args.collar)
        except Exception as e:  # noqa: BLE001
            rows.append({"file": n, "split": split_of(n), "gt": gt, "hyp_k": None, "der": None,
                         "error": f"scoring: {type(e).__name__}: {e}"})
            continue
        rows.append(
            {
                "file": n,
                "split": split_of(n),
                "gt": gt,
                "hyp_k": len({h[2] for h in hyp}),
                "der": d["der"] if d else None,
                "unwinnable": gt > args.max_speakers,
            }
        )

    def report(split):
        rs = [r for r in rows if r["split"] == split and r.get("der") is not None]
        if not rs:
            print(f"{split:8}  (no results)")
            return
        n = len(rs)
        exact = sum(1 for r in rs if r["hyp_k"] == r["gt"])
        within1 = sum(1 for r in rs if abs(r["hyp_k"] - r["gt"]) <= 1)
        under = sum(1 for r in rs if r["hyp_k"] < r["gt"])
        over = sum(1 for r in rs if r["hyp_k"] > r["gt"])
        unwin = sum(1 for r in rs if r.get("unwinnable"))
        der = sum(r["der"] for r in rs) / n
        nfail = len([r for r in rows if r["split"] == split and r.get("der") is None])
        print(
            f"{split:8} n={n:3d}{f' (+{nfail} FAILED)' if nfail else ''}  DER {der*100:6.2f}%"
            f"  |  count exact {exact}/{n} ({exact/n*100:.0f}%)"
            f"  within1 {within1}/{n}  under {under}  over {over}"
            + (f"  [{unwin} unwinnable at max-speakers={args.max_speakers}]" if unwin else "")
        )

    print()
    for r in sorted(rows, key=lambda r: (r["split"], r["file"])):
        if r.get("der") is None:
            print(f"  {r['split']:8} {r['file']:10} FAILED: {r.get('error','')[:80]}")
        else:
            flag = ""
            if r["hyp_k"] != r["gt"]:
                flag = "  <-- count off" + (" (capped)" if r.get("unwinnable") else "")
            print(f"  {r['split']:8} {r['file']:10} gt={r['gt']:2d} hyp={r['hyp_k']:2d}  DER {r['der']*100:6.2f}%{flag}")
    print()
    report("tune")
    report("holdout")
    print("\nTune the change against TUNE; quote the number from HOLDOUT.")

    if args.json_out:
        json.dump(rows, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
