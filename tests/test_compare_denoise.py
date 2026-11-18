#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Hualet Wang
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_denoise", ROOT / "scripts/compare-denoise.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareDenoiseTest(unittest.TestCase):
    def test_character_errors_do_not_count_spaces_or_punctuation(self):
        self.assertEqual(MODULE.normalized("你好，世界！ Ａ"), "你好世界a")
        self.assertEqual(MODULE.edits("你好世界", "你好世间"), 1)
        self.assertEqual(MODULE.edits("你好", ""), 2)
        self.assertEqual(MODULE.edits("你好", "你们好"), 1)

    def test_same_audio_is_used_and_polishing_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            audio = folder / "sample.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                writer.writeframes(b"\x00\x00" * 1600)
            model = folder / "model.gguf"
            model.write_bytes(b"fixture")
            binary = folder / "fake-service"
            binary.write_text(f"#!{sys.executable}\n" + '''import configparser, json, sys
from pathlib import Path
config = configparser.ConfigParser()
config.read(sys.argv[sys.argv.index('--config') + 1])
assert config['basic.polish.enabled']['value'] == 'false'
assert config['basic.recognition.language']['value'] == 'Chinese'
assert Path(config['advanced.crisp.model']['value']).is_file()
assert Path(sys.argv[sys.argv.index('--transcribe-file') + 1]).read_bytes().startswith(b'RIFF')
assert '--transcribe-json' in sys.argv
noise = config['basic.recognition.noise_suppression']['value'] == 'true'
print('diagnostic log before the JSON result')
print(json.dumps({'transcript': '你好' if noise else '你号', 'noise_suppression': noise, 'recognition_ms': 12}))
''')
            binary.chmod(0o755)
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps([{"id": "sample", "audio": "sample.wav",
                "reference": "你好！", "condition": "noisy"}], ensure_ascii=False))
            output = folder / "results.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/compare-denoise.py"),
                "--binary", str(binary), "--model", str(model), "--manifest", str(manifest),
                "--output", str(output), "--repeat", "1"], check=True, capture_output=True)
            report = json.loads(output.read_text())
            self.assertFalse(report['polish_enabled'])
            self.assertEqual(len(report['samples']), 2)
            self.assertEqual(report['samples'][0]['sha256'], report['samples'][1]['sha256'])
            self.assertEqual(report['conditions']['noisy']['raw']['cer'], 0.5)
            self.assertEqual(report['conditions']['noisy']['rnnoise']['cer'], 0)
            self.assertTrue(report['latency_includes_model_loading'])


if __name__ == "__main__":
    unittest.main()
