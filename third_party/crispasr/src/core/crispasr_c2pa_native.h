#pragma once
// Vendored from the standalone c2pa-audio repo (git submodule at
// third_party/c2pa-audio). This shim keeps existing
// `#include "crispasr_c2pa_native.h"` call sites working after the extraction.
// The declarations (namespace crispasr::c2pa_native: sign_wav / sign_mp3 /
// verify_wav / VerifyResult) live in the submodule's header.
#include "c2pa_native.h"
