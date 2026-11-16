"""Summarize same-runner CPU measurements and transcript agreement."""
import argparse
import json
from pathlib import Path
import re


def normalized(text):
    return re.findall(r"\w+(?:'\w+)?", text.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for path in sorted(args.results.glob("parity-*.json")):
        config = path.stem.removeprefix("parity-")
        parity = json.loads(path.read_text())
        reports[config] = {"parity": parity, "quants": {}}
        for quant in ("f16", "q8_0", "q4_k"):
            runtime_path = args.results / f"runtime-{config}-{quant}.json"
            if not runtime_path.exists():
                continue
            measured = json.loads(runtime_path.read_text())
            baseline = json.loads((args.results / f"runtime-scalar-{quant}.json").read_text())
            assert measured["host"] == baseline["host"] and measured["threads"] == baseline["threads"]
            assert len(measured["clips"]) == len(baseline["clips"]) == 2
            for row, base in zip(measured["clips"], baseline["clips"]):
                assert row["audio_s"] == base["audio_s"] and row["median_s"] > 0
                row["speedup_vs_scalar"] = base["median_s"] / row["median_s"]
                row["exact_scalar_transcript"] = row["transcript"] == base["transcript"]
                row["same_scalar_words"] = normalized(row["transcript"]) == normalized(base["transcript"])
            reports[config]["quants"][quant] = measured
            timings = " / ".join(f"{row['median_s']:.4f}s ({row['speedup_vs_scalar']:.3f}x)" for row in measured["clips"])
            exact = all(row["exact_scalar_transcript"] for row in measured["clips"])
            print(f"{config:12} {quant:5} {timings} exact={exact}")
    report = {"configurations": reports}
    for name in ("commit.txt", "host.txt"):
        report[name] = (args.results / name).read_text().strip()
    report["reference"] = json.loads((args.results / "reference-cpu.json").read_text())
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
