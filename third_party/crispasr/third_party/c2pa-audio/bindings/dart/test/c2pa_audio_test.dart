// Dart FFI binding tests. Set C2PA_AUDIO_LIB to the built shared library:
//   dart test  (with C2PA_AUDIO_LIB=/path/to/libc2pa_audio.dylib)
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:c2pa_audio/c2pa_audio.dart';
import 'package:test/test.dart';

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
    final s = (3000 * sin(2 * pi * 220 * i / sr)).round();
    u16(s & 0xffff);
  }
  return b.toBytes();
}

void main() {
  final libEnv = Platform.environment['C2PA_AUDIO_LIB'];
  final c2pa = C2paAudio.open();

  test('version', () {
    expect(c2pa.version, isNotEmpty);
  });

  test('sign -> verify round-trip is valid', () {
    final signed = c2pa.signWav(makeWav());
    expect(signed.length, greaterThan(makeWav().length));
    final r = c2pa.verify(signed);
    expect(r.valid, isTrue, reason: r.toString());
    expect(r.signatureValid, isTrue);
    expect(r.dataHashValid, isTrue);
    expect(r.assertionsValid, isTrue);
  });

  test('tampering the audio fails verification', () {
    final signed = c2pa.signWav(makeWav());
    signed[46] ^= 0xff;
    final r = c2pa.verify(signed);
    expect(r.valid, isFalse);
    expect(r.dataHashValid, isFalse);
  });

  test('non-C2PA WAV is not valid', () {
    final r = c2pa.verify(makeWav());
    expect(r.valid, isFalse);
  });

  test('c2pa-rs reference vector validates (their signer -> our verifier)', () {
    // fixture lives at repo test/assets; skip if not found from CWD
    final candidates = [
      'test/assets/reference-c2pa-rs.wav',
      '../../test/assets/reference-c2pa-rs.wav',
      if (libEnv != null)
        '${File(libEnv).parent.parent.path}/test/assets/reference-c2pa-rs.wav',
    ];
    final path =
        candidates.firstWhere((p) => File(p).existsSync(), orElse: () => '');
    if (path.isEmpty) return; // fixture optional
    final r = c2pa.verify(File(path).readAsBytesSync());
    expect(r.valid, isTrue, reason: r.toString());
  });

  test('MP3 round-trip (sign audio/mpeg -> verify)', () {
    // locate an unsigned sample MP3 next to the fixtures
    final candidates = [
      'test/assets/sample.mp3',
      '../../test/assets/sample.mp3',
      if (libEnv != null)
        '${File(libEnv).parent.parent.path}/test/assets/sample.mp3',
    ];
    final path =
        candidates.firstWhere((p) => File(p).existsSync(), orElse: () => '');
    if (path.isEmpty) return; // fixture optional
    final signed = c2pa.signMp3(File(path).readAsBytesSync());
    final r = c2pa.verify(signed);
    expect(r.valid, isTrue, reason: r.toString());
  });
}
