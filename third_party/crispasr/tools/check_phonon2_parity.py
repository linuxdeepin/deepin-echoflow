"""Require stage cosine AND an RMS-based bound on magnitude error."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs = {t.name: t.data for t in GGUFReader(str(args.reference)).tensors}
    rows = []
    for line in args.log.read_text().splitlines():
        match = re.search(r"^\[(PASS|FAIL|SKIP)\]\s+(\w+)\s+shape=.*cos_min=([\d.]+).*\srms=([\d.eE+-]+)", line)
        if not match:
            continue
        status, name, cosine, rms = match.groups()
        arr = np.asarray(refs["encoder_output" if name == "encoder_output_ref_mel" else name], dtype=np.float64)
        ref_rms = float(np.sqrt(np.mean(arr * arr)))
        bound = float(rms) / ref_rms
        row = dict(stage=name, cos_min=float(cosine), relative_rms_error=bound,
                   norm_ratio_error_bound=bound * 1.01)
        assert status == "PASS" and float(cosine) >= 0.999 and bound < 0.01, row
        rows.append(row)
    expected = {"mel_spectrogram", "pre_encode_output", "encoder_output", "encoder_output_ref_mel",
                "encoder_output_projected", "decoder_sos", "joint_sos_t0"}
    expected.update(f"encoder_layer_{i}" for i in range(24))
    assert {row["stage"] for row in rows} == expected, rows
    report = dict(stages=rows, min_cos=min(row["cos_min"] for row in rows),
                  max_norm_ratio_error_bound=max(row["norm_ratio_error_bound"] for row in rows))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(rows)} stages pass: min cosine {report['min_cos']}, magnitude bound {report['max_norm_ratio_error_bound']:.6g}")


if __name__ == "__main__":
    main()
