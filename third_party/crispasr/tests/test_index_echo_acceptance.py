"""Guards against accepting lost bilingual cues, arbitrary edits or large drift."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('index_echo_acceptance',
    Path(__file__).resolve().parents[1] / 'tools/index_echo_acceptance.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AcceptanceBounds(unittest.TestCase):
    def setUp(self):
        self.source = [dict(start=.43, end=4.57, text='原文。\nTranslation.'),
                       dict(start=5.23, end=7.47, text='次。\nNext.')]

    def test_f16_must_match_a_complete_independent_variant(self):
        alternate = [dict(c) for c in self.source]
        alternate[0]['text'] = '原文。\nAlternate.'
        variants = dict(released=self.source, f32=alternate)
        self.assertEqual(audit.compare_case(alternate, variants, 'f16')['source_variant'], 'f32')
        altered = [dict(c) for c in alternate]
        altered[1]['text'] = '次。\nInvented.'
        with self.assertRaises(ValueError):
            audit.compare_case(altered, variants, 'f16')

    def test_quant_timestamp_allowance_is_bounded(self):
        candidate = [dict(c) for c in self.source]
        candidate[0]['start'] += .02
        audit.compare_case(candidate, dict(source=self.source), 'q8_0')
        with self.assertRaises(ValueError):
            audit.compare_case(candidate, dict(source=self.source), 'f16')
        candidate[0]['start'] += .01
        with self.assertRaises(ValueError):
            audit.compare_case(candidate, dict(source=self.source), 'q8_0')

    def test_no_missing_cues_or_transcript_only_match(self):
        with self.assertRaises(ValueError):
            audit.compare_case(self.source[-1:], dict(source=self.source), 'q8_0')
        candidate = [dict(c, text=c['text'].split('\n')[0]) for c in self.source]
        with self.assertRaises(ValueError):
            audit.compare_case(candidate, dict(source=self.source), 'q8_0')


if __name__ == '__main__':
    unittest.main()
