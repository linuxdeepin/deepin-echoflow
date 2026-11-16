"""Same-runner FFN/thread A/B, preserving load/RSS and every transcript."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from gguf import GGUFReader
from check_phonon2_ffn_stages import compare_stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    refs = {t.name: t.data.reshape(-1, 1024) for t in GGUFReader(str(args.results / 'full-ref.gguf')).tensors
            if t.name == 'pre_encode_output' or t.name.startswith('encoder_layer_')}
    assert len(refs) == 25
    configs = {}
    for path in sorted(args.results.glob('parity-*.json')):
        config = path.stem.removeprefix('parity-')
        parity = json.loads(path.read_text())
        configs[config] = {'parity': parity, 'quants': {}}
        for quant in ('f16', 'q8_0', 'q4_k'):
            runtime_path = args.results / f'runtime-{config}-{quant}.json'
            if not runtime_path.exists():
                continue
            row = json.loads(runtime_path.read_text())
            base = json.loads((args.results / f'runtime-ggml-t4-{quant}.json').read_text())
            assert row['host'] == base['host'] and len(row['clips']) == len(base['clips']) == 2
            for clip, baseline in zip(row['clips'], base['clips']):
                assert clip['audio_s'] == baseline['audio_s'] and clip['median_s'] > 0
                clip['speedup_vs_default_t4'] = baseline['median_s'] / clip['median_s']
                clip['exact_default_transcript'] = clip['transcript'] == baseline['transcript']
                clip['same_default_words'] = re.findall(r"\w+(?:'\w+)?", clip['transcript'].lower()) == re.findall(r"\w+(?:'\w+)?", baseline['transcript'].lower())
            capture_path = args.results / f'stages-{config}-{quant}.npz'
            if capture_path.exists():
                with np.load(capture_path) as captures:
                    row['encoder_stages_vs_python_f32'] = compare_stages(captures, refs)
            stage_path = args.results / f'stages-{config}-{quant}.json'
            if stage_path.exists():
                row['same_quant_stage_parity'] = json.loads(stage_path.read_text())
            configs[config]['quants'][quant] = row
            times = ' / '.join(f"{c['median_s']:.4f}s ({c['speedup_vs_default_t4']:.3f}x)" for c in row['clips'])
            print(config, quant, times, 'exact=', all(c['exact_default_transcript'] for c in row['clips']),
                  f"load={row['load_s']:.3f}s RSS={row['peak_rss_mb']:.1f}MiB")
    report = {'configurations': configs, 'reference': json.loads((args.results / 'reference-cpu.json').read_text())}
    for name in ('commit.txt', 'host.txt', 'SHA256SUMS.txt'):
        report[name] = (args.results / name).read_text()
    report['kernel_logs'] = {p.name: p.read_text() for p in sorted(args.results.glob('kernel-*.txt'))}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
