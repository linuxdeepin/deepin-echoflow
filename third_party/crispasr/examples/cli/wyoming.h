// wyoming.h — Wyoming protocol TCP server (STT + TTS for Home Assistant).
//
// Wyoming is a peer-to-peer JSONL-over-TCP protocol used by Home Assistant's
// Assist voice pipeline. One CrispASR server instance can replace both a
// wyoming-faster-whisper STT service and a wyoming-piper TTS service.
//
// Wire format:
//   {"type":"...","data":{...},"payload_length":N}\n
//   [N bytes of binary payload, present only when payload_length > 0]
//
// Handled event types:
//   describe       → info           (capability advertisement)
//   transcribe     +
//   audio-start    +                (batch: buffer until audio-stop)
//   audio-chunk    +
//   audio-stop     → transcript
//   synthesize     → audio-start +
//                    audio-chunk* +
//                    audio-stop
//
// Usage from crispasr_server.cpp:
//   wyoming_start(backend, model_mutex, params, port);
//   wyoming_stop();

#pragma once

#include "crispasr_backend.h"
#include "whisper_params.h"
#include <mutex>

// Start the Wyoming listener thread on `port`. Returns 0 on success.
// `backend` and `model_mutex` must remain valid for the server lifetime.
int wyoming_start(CrispasrBackend* backend, std::mutex& model_mutex, const whisper_params& params, int port);

// Stop the listener and join the thread. Safe to call multiple times.
void wyoming_stop();
