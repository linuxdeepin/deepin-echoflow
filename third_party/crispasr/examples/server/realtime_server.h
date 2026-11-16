#pragma once

#include <mutex>
#include "../cli/whisper_params.h"
#include "../cli/crispasr_backend.h"

// Starts a raw TCP WebSocket server for the vLLM Realtime API (/v1/realtime)
int realtime_server_start(CrispasrBackend* backend, std::mutex& model_mutex, const whisper_params& base_params,
                          int port);

// Stops the server and joins the listener thread
void realtime_server_stop();
