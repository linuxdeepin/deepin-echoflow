// env_gate.h — one value-parsed boolean environment gate for the whole
// codebase.
//
// The defect this exists to prevent: `getenv("X") != nullptr` (or bare
// `if (getenv("X"))`) treats `X=0` as ENABLED. Every operator who writes
// `CRISPEMBED_FOO_BENCH=0` to turn an instrument OFF gets it turned ON, and
// for a diagnostic gate the only symptom is unexplained output — which is
// exactly how the DS_* inversion survived until the G2b/G6 audits.
//
// Semantics (identical to deepseek_ocr2.cpp's `ds_env_on`, which established
// them, and to core_initbench::enabled): the gate is ON when the variable is
// set to a non-empty value that is not exactly "0". Unset, empty, and "0" are
// all OFF.
//
//     const bool bench = core_env::on("CRISPEMBED_FOO_BENCH");
//
// Guarded hermetically by CrispEmbed's tests/test_env_gate.cpp.
//
// ⚠ CROSS-REPO MIRROR (CrispEmbed issue #50). This header is the CrispASR half
// of a mirrored pair; the other half is CrispEmbed's src/core/env_gate.h.
// crisp_punc/src/fireredpunc.cpp reads its tokenizer gate through core_env::,
// and that file compiles against the CONSUMER's src/core — CrispASR's when
// built here, CrispEmbed's when CrispEmbed add_subdirectory()s crisp_punc. So
// the name has to resolve on both sides. CrispASR's own code keeps using
// crispasr_env:: (which carries the legacy-name aliasing this does not); this
// exists for the shared libraries. Keep the two copies in sync.
#pragma once

#include <cstdlib>
#include <cstring>

namespace core_env {

inline bool on(const char* name) {
    if (!name)
        return false;
    const char* e = std::getenv(name);
    return e && *e && std::strcmp(e, "0") != 0;
}

// The other half of a tri-state gate: EXPLICITLY set to "0". `on()` cannot
// express it, because `on()` folds unset and "0" together — fine for an
// instrument that defaults off, wrong for a knob that defaults ON and needs an
// opt-OUT. Use the pair when the default is on:
//
//     if (core_env::explicitly_off("CRISPEMBED_FOO_CLEANUP")) { /* legacy */ }
//
// Unset and empty are NOT off (they mean "default"); only a literal "0" is.
inline bool explicitly_off(const char* name) {
    if (!name)
        return false;
    const char* e = std::getenv(name);
    return e && *e && std::strcmp(e, "0") == 0;
}

} // namespace core_env
