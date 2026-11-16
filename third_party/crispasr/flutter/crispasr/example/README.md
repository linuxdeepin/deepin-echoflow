# crispasr examples

`crispasr` runs on-device speech recognition (Whisper-family and other ASR
models) via ggml. It needs the native `libcrispasr` at runtime (see the package
README for how to supply it).

## Transcribe an audio file

```dart
import 'package:crispasr/crispasr.dart';

void main() {
  final model = CrispASR.open('ggml-base.en.bin');
  try {
    final segments = model.transcribe('audio.wav');
    for (final seg in segments) {
      print('[${seg.start}s - ${seg.end}s] ${seg.text}');
    }
  } finally {
    model.close();
  }
}
```

The library also exposes a chat/LLM surface (`package:crispasr/crispasr.dart`
re-exports `src/chat.dart`) for the bundled generation backends.
