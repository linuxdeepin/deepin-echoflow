"""Hand-packed format fixtures, including ragged rows and corrupt records."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "phonon2_container", Path(__file__).resolve().parents[1] / "models/phonon2_container.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class PhononContainerTests(unittest.TestCase):
    def test_five_levels_ragged_rows_and_nonzero_bit_order(self):
        # Row 0: [-hi, -lo, 0, +lo, +hi, 0]; row 1: [+hi, 0, -lo, 0, +lo, -hi].
        # Magnitude bits run across nonzeros only, in row-major little-endian order.
        signs = bytes([0 + 0*3 + 1*9 + 2*27 + 2*81, 1,
                       2 + 1*3 + 0*9 + 1*27 + 2*81, 0])
        levels = np.array([0.5, 1.0, 2.0, 4.0], dtype="<f2").tobytes()
        blob = signs + bytes([0b10011001]) + levels
        actual = reader.expand_five_value(blob, (2, 6))
        np.testing.assert_array_equal(actual, [[-2, -.5, 0, .5, 2, 0], [4, 0, -1, 0, 1, -4]])

    def test_int6_signed_extremes_padding_and_row_scales(self):
        # 6 actual values, padded to 8; -32, -1, 0, 1, 30, 31.
        packed = sum(v << (6*i) for i, v in enumerate([0, 31, 32, 33]))
        tail = 62 | (63 << 6)
        blob = packed.to_bytes(3, "little") + tail.to_bytes(3, "little")
        blob += np.array([.5, 2], dtype="<f2").tobytes()
        np.testing.assert_array_equal(reader.expand_int(blob, (2, 3), 6), [[-16, -.5, 0], [2, 60, 62]])

    def test_int8_and_scalar_fp16(self):
        blob = np.array([-128, -1, 0, 127], dtype=np.int8).tobytes() + np.array([.25], dtype="<f2").tobytes()
        np.testing.assert_array_equal(reader.expand_int(blob, (1, 4), 8), [[-32, -.25, 0, 31.75]])
        got = self.read([{"n": "counter", "k": "fp16", "shape": [], "b": 2}], np.array(3, dtype="<f2").tobytes())
        self.assertEqual(got[0][1].shape, ())
        self.assertEqual(float(got[0][1]), 3)

    def read(self, index, body, extra=b"", format=reader.FORMAT):
        header = json.dumps({"format": format, "index": index}).encode()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.fermion"
            p.write_bytes(len(header).to_bytes(8, "little") + header + body + extra)
            return list(reader.iter_tensors(p))

    def test_rejects_corrupt_record_lengths_duplicates_and_trailing_bytes(self):
        entry = {"n": "x", "k": "fp16", "shape": [1], "b": 2}
        for index, body, extra in [([entry], b"x", b""), ([entry], b"xx", b"x"),
                                   ([entry, entry], b"xxxx", b""),
                                   ([{**entry, "b": -1}], b"", b"")]:
            with self.subTest(index=index, body=body, extra=extra), self.assertRaises(ValueError):
                self.read(index, body, extra)
        with self.assertRaises(ValueError):
            self.read([entry], b"xx", format="unknown")

    def test_rejects_invalid_base3_and_missing_magnitude_bit(self):
        for blob in [bytes([243]) + b"\0"*5, bytes([0]) + b"\0"*4]:
            with self.assertRaises(ValueError):
                reader.expand_five_value(blob, (1, 1))

    def test_checks_container_checksum_and_rejects_artifact_paths(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "model.fermion").write_bytes(b"fixture")
            digest = hashlib.sha256(b"fixture").hexdigest()
            (p / "config.json").write_text(json.dumps({"fermion": {"container_sha256": digest}}))
            self.assertEqual(reader.materialize(p), p)
            (p / "model.fermion").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                reader.materialize(p)
            (p / "config.json").write_text(json.dumps({"model_type": "parakeet_tdt_five_value",
                                                       "artifact": {"filename": "../escape"}}))
            with self.assertRaisesRegex(ValueError, "basename"):
                reader.materialize(p)


if __name__ == "__main__":
    unittest.main()
