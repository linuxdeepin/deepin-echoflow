// Quantized-weight × batched-activation matmuls: detector + avoidance.
//
// Issue #416 background. `ggml_mul_mat(W, X)` where W is a QUANTIZED weight
// (so ne2 == 1) and X carries a batch/head dimension (ne2 > 1) makes ggml
// broadcast src0 across that dimension. On Vulkan a quantized src0 is routed
// through the MMQ/Q8_1 pipeline, but only on devices advertising integer dot
// product:
//
//     quantize_y = ctx->device->integer_dot_product && ...   // ggml-vulkan.cpp
//
// Sidon's relative-position matmul in exactly that shape produced an all-zero
// predictor output — a full-length file of silence — on a GTX 1660 SUPER
// (`int dot: 1`), while CPU and `int dot: 0` devices were fine. Sidon itself is
// fixed by keeping its 9 KB lookup table unquantized, but that is only viable
// because the table is tiny; for a real weight matrix, dequantizing would
// defeat quantization entirely. Hence the fold below.
//
// ⚠ The underlying ggml defect is NOT yet confirmed to be broadcast-specific
// rather than specific to sidon's dimensions. A caller may only default the
// fold ON once it has been shown, by running the detector below, that (a) the
// backend genuinely has exposed sites and (b) folding them leaves the decoded
// output unchanged. Anything unverified stays OFF.
#pragma once

#include "ggml.h"

#include <cstdio>
#include <cstdlib>

namespace core_quant_bcast {

// A linear projection is independent per token, so folding the batch dimension
// into the token dimension is an exact restatement: the same rows are dotted
// with the same columns. It removes the src0 broadcast (ne2 becomes 1) at the
// cost of one reshape, and the result is reshaped back to the caller's layout.
//
// Falls through to a plain mul_mat when there is no batch to fold, so it is a
// drop-in replacement.
inline ggml_tensor* mul_mat_fold_batch(ggml_context* c, ggml_tensor* w, ggml_tensor* x) {
    if (x->ne[2] <= 1 && x->ne[3] <= 1)
        return ggml_mul_mat(c, w, x);
    const int64_t d = x->ne[0], n = x->ne[1], b2 = x->ne[2], b3 = x->ne[3];
    // reshape needs contiguity; most call sites already are, so this is
    // usually a no-op node.
    ggml_tensor* xf = ggml_is_contiguous(x) ? x : ggml_cont(c, x);
    xf = ggml_reshape_2d(c, xf, d, n * b2 * b3);
    ggml_tensor* y = ggml_mul_mat(c, w, xf); // (d_out, n*b2*b3)
    return ggml_reshape_4d(c, y, y->ne[0], n, b2, b3);
}

// Gate helper. `CRISPASR_<BACKEND>_FOLD_BCAST` overrides the backend's default;
// "0" forces the legacy broadcasting matmul back, anything else forces the fold.
// Per the project convention, once a path is verified the NEW path becomes the
// default and the OLD one keeps a gate — never the reverse, and never removed,
// because that gate is the bisection mechanism.
inline bool fold_enabled(const char* env_name, bool default_on = false) {
    const char* e = std::getenv(env_name);
    if (!e || !e[0])
        return default_on;
    return e[0] != '0';
}

// Detector. Static greps for this pattern undercount badly: two independent
// heuristics over src/ each found sites the other missed (one keyed on how the
// src1 variable was built, one on the trailing shape comment). Walking the
// actual graph is the only complete answer, and it also reports the true
// runtime ne values rather than a guess.
//
// Gated by CRISPASR_AUDIT_QUANT_BCAST=1; costs one pass over the node list at
// graph-build time, nothing per element. Run any backend with a QUANTIZED model
// and the env var set to enumerate its exposure.
inline int audit(const ggml_cgraph* gf, const char* tag) {
    if (!gf || !std::getenv("CRISPASR_AUDIT_QUANT_BCAST"))
        return 0;
    int found = 0;
    for (int i = 0, n = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(gf)); i < n; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(gf), i);
        if (!node || node->op != GGML_OP_MUL_MAT)
            continue;
        const ggml_tensor* a = node->src[0];
        const ggml_tensor* b = node->src[1];
        if (!a || !b || !ggml_is_quantized(a->type))
            continue;
        // src0 broadcast over the batch dims is the hazard; equal dims are not.
        if ((b->ne[2] > a->ne[2]) || (b->ne[3] > a->ne[3])) {
            ++found;
            std::fprintf(stderr,
                         "quant-bcast[%s]: %s  src0=%s%s [%lld,%lld,%lld,%lld]  "
                         "src1 [%lld,%lld,%lld,%lld]  (r2=%lld r3=%lld)\n",
                         tag, node->name[0] ? node->name : "(unnamed)", ggml_type_name(a->type),
                         a->name[0] ? a->name : "", (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2],
                         (long long)a->ne[3], (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2],
                         (long long)b->ne[3], (long long)(a->ne[2] ? b->ne[2] / a->ne[2] : 0),
                         (long long)(a->ne[3] ? b->ne[3] / a->ne[3] : 0));
        }
    }
    if (found)
        std::fprintf(stderr, "quant-bcast[%s]: %d broadcasting quantized matmul(s)\n", tag, found);
    return found;
}

} // namespace core_quant_bcast
