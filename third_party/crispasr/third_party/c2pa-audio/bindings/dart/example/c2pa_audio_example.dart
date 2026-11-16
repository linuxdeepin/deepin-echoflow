// Minimal example: sign a WAV with the bundled default cert, then verify it.
// Set C2PA_AUDIO_LIB to the built native library before running.
import 'dart:typed_data';

import 'package:c2pa_audio/c2pa_audio.dart';

Uint8List makeWav({int n = 4800, int sr = 24000}) {
  final b = BytesBuilder();
  void u32(int v) =>
      b.add([v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff]);
  void u16(int v) => b.add([v & 0xff, (v >> 8) & 0xff]);
  b.add('RIFF'.codeUnits);
  u32(36 + n * 2);
  b.add('WAVE'.codeUnits);
  b.add('fmt '.codeUnits);
  u32(16);
  u16(1);
  u16(1);
  u32(sr);
  u32(sr * 2);
  u16(2);
  u16(16);
  b.add('data'.codeUnits);
  u32(n * 2);
  for (var i = 0; i < n; i++) {
    u16(0);
  }
  return b.toBytes();
}

void main() {
  final c2pa = C2paAudio.open();
  print('c2pa-audio ${c2pa.version}');
  final signed = c2pa.signWav(makeWav());
  final result = c2pa.verify(signed);
  print('signed ${signed.length} bytes; valid: ${result.valid}');
}
