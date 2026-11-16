// Live end-to-end test for the §251b beat-tracking binding.
//
// Opens a CrispasrSession against a beat-this GGUF, runs [beats] on a
// synthetic 120 BPM click track, and asserts the contract the Dart docs
// promise — not just "it returned something":
//
//   (a) Beats come back in strictly increasing time order.
//   (b) Downbeats are a STRICT SUBSET of beats. This is the property the
//       postprocessor's snap-to-nearest-beat step exists to guarantee, and
//       the one a caller would otherwise have to reconstruct by merging two
//       lists. If it ever breaks, every bar-grid UI downstream breaks.
//   (c) The median-interval tempo lands on 120 BPM, which is a real check
//       rather than a smoke test: the fixture's tempo is known exactly, so a
//       half/double-time error or a seam bug moves it off 120.
//   (d) beatsSampleRate reports 22050 and is usable as a capability probe.
//
// Skips silently when CRISPASR_LIB / CRISPASR_BEAT_MODEL are absent — same
// pattern as transcription_test.dart + alt_tokens_live_test.dart so `dart
// test` stays green on model-less runners.
//
// Run locally with:
//   CRISPASR_LIB=../../build/src/libcrispasr.dylib \
//     CRISPASR_BEAT_MODEL=/path/to/beat-this-f16.gguf \
//     dart test test/beats_live_test.dart

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:crispasr/crispasr.dart';
import 'package:test/test.dart';

const int _sr = 22050;
const double _bpm = 120.0;

/// A 20 s click track at 120 BPM, accented every 4th beat.
///
/// Synthesised rather than shipped as a fixture so the expected tempo and bar
/// length are exact by construction, with no WAV in the repo to drift.
Float32List _clickTrack({double seconds = 20.0}) {
  final n = (seconds * _sr).round();
  final x = Float32List(n);
  final period = 60.0 / _bpm;
  for (var i = 0;; i++) {
    final t0 = (i * period * _sr).round();
    if (t0 >= n) break;
    final downbeat = i % 4 == 0;
    final freq = downbeat ? 180.0 : 330.0;
    final amp = downbeat ? 0.9 : 0.5;
    final dur = downbeat ? 0.12 : 0.06;
    final m = math.min((dur * _sr).round(), n - t0);
    for (var j = 0; j < m; j++) {
      final t = j / _sr;
      x[t0 + j] +=
          amp * math.sin(2 * math.pi * freq * t) * math.exp(-t / (dur / 4));
    }
  }
  return x;
}

void main() {
  final libPath = Platform.environment['CRISPASR_LIB'];
  final modelPath = Platform.environment['CRISPASR_BEAT_MODEL'];

  test('beats: order, downbeat subset, and 120 BPM on a 120 BPM fixture', () {
    if (libPath == null || modelPath == null) {
      // ignore: avoid_print
      print('skip: set CRISPASR_LIB and CRISPASR_BEAT_MODEL to run');
      return;
    }
    final s =
        CrispasrSession.open(modelPath, libPath: libPath, backend: 'beat-this');
    try {
      expect(s.beatsSampleRate, _sr,
          reason: 'the binding must probe the rate, not hard-code it');

      final grid = s.beats(_clickTrack());
      expect(grid, isNotEmpty);

      // (a) strictly increasing
      for (var i = 1; i < grid.length; i++) {
        expect(grid[i].timeS, greaterThan(grid[i - 1].timeS),
            reason: 'beats must be in strictly increasing time order');
      }

      // (b) downbeats are a strict subset of beats
      final downbeats = grid.where((b) => b.isDownbeat).toList();
      expect(downbeats, isNotEmpty, reason: 'a 4/4 click track has downbeats');
      expect(downbeats.length, lessThan(grid.length),
          reason: 'not every beat is a downbeat');

      // (c) tempo — the fixture is exactly 120 BPM, so this is a real assert.
      expect(s.beatsTempoBpm, closeTo(_bpm, 1.0));

      // ...and the inter-beat interval should be the 0.5 s that implies.
      final iois = <double>[
        for (var i = 1; i < grid.length; i++) grid[i].timeS - grid[i - 1].timeS
      ]..sort();
      expect(iois[iois.length ~/ 2], closeTo(60.0 / _bpm, 0.05));

      // ignore: avoid_print
      print('beats=${grid.length} downbeats=${downbeats.length} '
          'tempo=${s.beatsTempoBpm.toStringAsFixed(2)} BPM');
    } finally {
      s.close();
    }
  });
}
