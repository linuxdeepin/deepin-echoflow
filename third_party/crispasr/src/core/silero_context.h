#pragma once
#include "crispasr.h"

#ifdef __cplusplus
extern "C" {
#endif
// Internal recipe switch: Silero's 16 kHz TorchScript wrapper prepends the
// previous 64 samples (zeros at file start), then reflects only the right edge.
// Enable on a fresh VAD context; other backends keep their existing recipe.
CRISPASR_API bool crispasr_silero_enable_context(struct whisper_vad_context* ctx);
#ifdef __cplusplus
}
#endif
