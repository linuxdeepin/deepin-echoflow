// htdemucs_ggml_util.h — weight-free helpers for the htdemucs ggml graph port.
//
// Issue #398: the CUDA binbcast kernels (ggml-cuda/binbcast.cu) pick the
// src1 element type from the src0/dst branch — for src0=F32/dst=F32 they
// cast src1 to `const float*` unconditionally. A broadcast ggml_add/ggml_mul
// whose src1 is an F16 weight therefore trips
// `GGML_ASSERT(nb10 % sizeof(src1_t) == 0)` (nb10 = 2, sizeof(float) = 4)
// and hard-aborts the server on the FIRST /v1/audio/separation request.
// The CPU path converts per element and never notices, which is how this
// shipped: the F16 GGUF stores the DConv GroupNorm affine weights
// (`*.dconv.layers.N.4.weight`, 384/768 elements) as F16 — the converter's
// keep-F32 rule matches ".norm"/".bias"/"scale" by name, but these sit at a
// bare nn.Sequential index and are ≥256 elements, so both keep rules missed
// them. Quantized files inherit the same F16 tensors verbatim, which is why
// q8_0 crashed identically.
//
// Fix: every weight tensor used as src1 of a broadcast binary op goes
// through htd_bcast_f32(), which inserts an in-graph cast to F32 when the
// stored type differs (a few hundred elements per graph — negligible). The
// pre-fix behaviour stays reachable via CRISPASR_HTDEMUCS_NO_BCAST_CAST=1
// for regression bisection.

#pragma once

#include "ggml.h"

#include <cstdlib>

// Returns true unless CRISPASR_HTDEMUCS_NO_BCAST_CAST=1 disables the fix
// (bisection gate — never remove).
static inline bool htd_bcast_cast_enabled() {
    const char* e = std::getenv("CRISPASR_HTDEMUCS_NO_BCAST_CAST");
    return !(e && atoi(e) != 0);
}

// Wrap a weight tensor that is about to be used as src1 of a broadcast
// binary op (ADD/SUB/MUL/DIV). Non-F32 storage gets an in-graph cast to F32
// so the CUDA binbcast kernels see a stride that matches their element type.
static inline ggml_tensor* htd_bcast_f32(ggml_context* g, ggml_tensor* t, bool enabled) {
    if (!enabled || !t || t->type == GGML_TYPE_F32)
        return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

static inline ggml_tensor* htd_bcast_f32(ggml_context* g, ggml_tensor* t) {
    return htd_bcast_f32(g, t, htd_bcast_cast_enabled());
}

// True when a binary-broadcast node's src1 type is one the CUDA binbcast
// dispatch handles: F32 always works; F16 src1 is only supported when src0
// is F16 as well (the half/half and half/float branches).
static inline bool htd_binbcast_src1_ok(const ggml_tensor* node) {
    const ggml_tensor* s0 = node->src[0];
    const ggml_tensor* s1 = node->src[1];
    if (!s0 || !s1)
        return true;
    if (s1->type == GGML_TYPE_F32)
        return true;
    return s0->type == GGML_TYPE_F16 && s1->type == GGML_TYPE_F16;
}

// Walks a built graph and returns the first ADD/SUB/MUL/DIV node whose src1
// would trip the CUDA binbcast stride assert, or nullptr if the graph is
// clean. Used by the unit test as the algebraic guard for issue #398.
static inline const ggml_tensor* htd_first_bad_binbcast(ggml_cgraph* gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor* n = ggml_graph_node(gf, i);
        switch (n->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (!htd_binbcast_src1_ok(n))
                return n;
            break;
        default:
            break;
        }
    }
    return nullptr;
}
