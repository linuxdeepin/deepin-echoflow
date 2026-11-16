// c2pa_audio.dart — Dart FFI binding for the standalone C2PA signer/verifier.
//
// Loads libc2pa_audio (shared library) and exposes signWav / verifyWav.
// Pure dart:ffi + package:ffi. Works on Dart VM and Flutter (desktop/mobile).
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

// ---- C signatures ----
typedef _SignNative = Int32 Function(
    Pointer<Uint8> in_,
    IntPtr inLen,
    Pointer<Utf8> mime,
    Pointer<Utf8> cert,
    Pointer<Utf8> key,
    Pointer<Pointer<Uint8>> out,
    Pointer<IntPtr> outLen);
typedef _SignDart = int Function(
    Pointer<Uint8> in_,
    int inLen,
    Pointer<Utf8> mime,
    Pointer<Utf8> cert,
    Pointer<Utf8> key,
    Pointer<Pointer<Uint8>> out,
    Pointer<IntPtr> outLen);
typedef _VerifyNative = Int32 Function(Pointer<Uint8> in_, IntPtr inLen);
typedef _VerifyDart = int Function(Pointer<Uint8> in_, int inLen);
typedef _FreeNative = Void Function(Pointer<Uint8> p);
typedef _FreeDart = void Function(Pointer<Uint8> p);
typedef _VersionNative = Pointer<Utf8> Function();
typedef _VersionDart = Pointer<Utf8> Function();

/// Result of [C2paAudio.verifyWav].
class VerifyResult {
  final bool signatureValid;
  final bool dataHashValid;
  final bool assertionsValid;
  final bool valid;
  const VerifyResult(this.signatureValid, this.dataHashValid,
      this.assertionsValid, this.valid);
  @override
  String toString() =>
      'VerifyResult(valid: $valid, sig: $signatureValid, data: $dataHashValid, assertions: $assertionsValid)';
}

/// Native C2PA signer/verifier for WAV — no c2pa-rs.
class C2paAudio {
  final DynamicLibrary _lib;
  late final _SignDart _sign =
      _lib.lookupFunction<_SignNative, _SignDart>('c2pa_audio_sign');
  late final _VerifyDart _verify =
      _lib.lookupFunction<_VerifyNative, _VerifyDart>('c2pa_audio_verify');
  late final _FreeDart _free =
      _lib.lookupFunction<_FreeNative, _FreeDart>('c2pa_audio_free');
  late final _VersionDart _version =
      _lib.lookupFunction<_VersionNative, _VersionDart>('c2pa_audio_version');

  C2paAudio(this._lib);

  /// Open the shared library. Pass an explicit [path], else use C2PA_AUDIO_LIB
  /// env var, else the platform default (libc2pa_audio.{dylib,so,dll}).
  factory C2paAudio.open([String? path]) {
    path ??= Platform.environment['C2PA_AUDIO_LIB'];
    if (path == null) {
      if (Platform.isMacOS) {
        path = 'libc2pa_audio.dylib';
      } else if (Platform.isWindows) {
        path = 'c2pa_audio.dll';
      } else {
        path = 'libc2pa_audio.so';
      }
    }
    return C2paAudio(DynamicLibrary.open(path));
  }

  String get version => _version().toDartString();

  /// Sign [data] with a C2PA manifest. [mime] is "audio/wav" (default) or
  /// "audio/mpeg". Pass PEM strings for a custom identity, or leave null to use
  /// the bundled self-signed default cert. Returns the signed bytes; throws
  /// [StateError] on failure.
  Uint8List sign(Uint8List data,
      {String mime = 'audio/wav', String? certPem, String? keyPem}) {
    final dataPtr = malloc<Uint8>(data.length);
    dataPtr.asTypedList(data.length).setAll(0, data);
    final mimePtr = mime.toNativeUtf8();
    final certPtr = certPem == null ? nullptr : certPem.toNativeUtf8();
    final keyPtr = keyPem == null ? nullptr : keyPem.toNativeUtf8();
    final outPtr = malloc<Pointer<Uint8>>();
    final outLenPtr = malloc<IntPtr>();
    try {
      final rc = _sign(dataPtr, data.length, mimePtr, certPtr.cast(),
          keyPtr.cast(), outPtr, outLenPtr);
      if (rc != 0) throw StateError('c2pa_audio_sign failed (rc=$rc)');
      final out = Uint8List.fromList(outPtr.value.asTypedList(outLenPtr.value));
      _free(outPtr.value);
      return out;
    } finally {
      malloc.free(dataPtr);
      malloc.free(mimePtr);
      if (certPtr != nullptr) malloc.free(certPtr);
      if (keyPtr != nullptr) malloc.free(keyPtr);
      malloc.free(outPtr);
      malloc.free(outLenPtr);
    }
  }

  /// Convenience: sign a WAV.
  Uint8List signWav(Uint8List wav, {String? certPem, String? keyPem}) =>
      sign(wav, mime: 'audio/wav', certPem: certPem, keyPem: keyPem);

  /// Convenience: sign an MP3.
  Uint8List signMp3(Uint8List mp3, {String? certPem, String? keyPem}) =>
      sign(mp3, mime: 'audio/mpeg', certPem: certPem, keyPem: keyPem);

  /// Convenience: sign an M4A/MP4.
  Uint8List signM4a(Uint8List m4a, {String? certPem, String? keyPem}) =>
      sign(m4a, mime: 'audio/mp4', certPem: certPem, keyPem: keyPem);

  /// Convenience: sign a FLAC.
  Uint8List signFlac(Uint8List flac, {String? certPem, String? keyPem}) =>
      sign(flac, mime: 'audio/flac', certPem: certPem, keyPem: keyPem);

  /// Verify a signed audio file (WAV / MP3 / M4A, auto-detected). Never throws.
  VerifyResult verify(Uint8List data) {
    final dataPtr = malloc<Uint8>(data.length);
    dataPtr.asTypedList(data.length).setAll(0, data);
    try {
      final f = _verify(dataPtr, data.length);
      return VerifyResult(
          (f & 0x1) != 0, (f & 0x2) != 0, (f & 0x4) != 0, (f & 0x8) != 0);
    } finally {
      malloc.free(dataPtr);
    }
  }
}
