// Parity harness: full rel-pos attention (monolithic O(T²) BD) vs query-TILED
// full attention (per query-block BD slab, O(T·B) peak). Proves the per-block
// rel-shift offset so exact full attention can run with bounded memory.
#include "ggml-cpu.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#ifndef HD
#define HD 4
#endif
#ifndef NH
#define NH 2
#endif
#ifndef T
#define T 12
#endif
#ifndef BS
#define BS 3
#endif
static const int D = HD * NH;
static const int NB = (T + BS - 1) / BS;
static const int Tp = NB * BS;

static float frand(int i) { return sinf((float)i * 0.37f) * 0.9f; }

// standard rel_shift: a (2T-1,T,H) -> (T,T,H), out[k,q]=a[k-q+(T-1),q]
static ggml_tensor* rel_shift(ggml_context* c, ggml_tensor* a) {
    const int t = (int)a->ne[1], h = (int)a->ne[2];
    return ggml_view_3d(c, a, t, t, h, a->nb[1] - a->nb[0], a->nb[2], (size_t)(t - 1) * a->nb[0]);
}

int main() {
    ggml_init_params ip = {(size_t)512 * 1024 * 1024, nullptr, false};
    ggml_context* c = ggml_init(ip);

    ggml_tensor* Q = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    ggml_tensor* K3 = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, NH, T);
    ggml_tensor* V3 = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, NH, T);
    ggml_tensor* R0 = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 2 * T - 1);
    ggml_tensor* pbu = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    ggml_tensor* pbv = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    for (auto* t : {Q, K3, V3, R0, pbu, pbv})
        ggml_set_input(t);

    const float scale = 1.0f / sqrtf((float)HD);
    auto sh = [&](ggml_tensor* x) { return ggml_permute(c, ggml_reshape_3d(c, x, HD, NH, T), 0, 2, 1, 3); };

    ggml_tensor* Qu = ggml_cont(c, sh(ggml_add(c, Q, pbu))); // (HD,T,NH)
    ggml_tensor* Qv = ggml_cont(c, sh(ggml_add(c, Q, pbv)));
    ggml_tensor* Kh = ggml_cont(c, ggml_permute(c, K3, 0, 2, 1, 3)); // (HD,T,NH)
    ggml_tensor* Vh = ggml_cont(c, ggml_permute(c, V3, 0, 2, 1, 3));
    ggml_tensor* Rh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, R0, HD, NH, 2 * T - 1), 0, 2, 1, 3));

    ggml_tensor *ref, *tiled;
    // ---- REFERENCE: monolithic full attention ----
    {
        ggml_tensor* BDraw = ggml_mul_mat(c, Rh, Qv);          // (2T-1,T,NH)
        ggml_tensor* BD = ggml_cont(c, rel_shift(c, BDraw));   // (T,T,NH)
        ggml_tensor* sc = ggml_mul_mat(c, Kh, Qu);             // (T,T,NH)
        sc = ggml_add(c, sc, BD);
        sc = ggml_soft_max_ext(c, sc, nullptr, scale, 0.0f);
        ggml_tensor* Vt = ggml_cont(c, ggml_permute(c, Vh, 1, 0, 2, 3)); // (T,HD,NH)
        ref = ggml_cont(c, ggml_mul_mat(c, Vt, sc));                     // (HD,T,NH)
    }
    // ---- TILED: per query-block, full keys, BD slab (T,BS) ----
    {
        // pad queries to Tp
        ggml_tensor* Qup = (Tp == T) ? Qu : ggml_pad_ext(c, Qu, 0, 0, 0, Tp - T, 0, 0, 0, 0);
        ggml_tensor* Qvp = (Tp == T) ? Qv : ggml_pad_ext(c, Qv, 0, 0, 0, Tp - T, 0, 0, 0, 0);
        Qup = ggml_cont(c, Qup);
        Qvp = ggml_cont(c, Qvp);
        ggml_tensor* Vt = ggml_cont(c, ggml_permute(c, Vh, 1, 0, 2, 3)); // (T,HD,NH)
        ggml_tensor* out_blocks = nullptr;                              // will concat (HD,BS,NH) per block along dim1
        for (int b = 0; b < NB; b++) {
            // query block b: (HD,BS,NH)
            ggml_tensor* Qu_b = ggml_cont(
                c, ggml_view_3d(c, Qup, HD, BS, NH, Qup->nb[1], Qup->nb[2], (size_t)b * BS * Qup->nb[1]));
            ggml_tensor* Qv_b = ggml_cont(
                c, ggml_view_3d(c, Qvp, HD, BS, NH, Qvp->nb[1], Qvp->nb[2], (size_t)b * BS * Qvp->nb[1]));
            // BDraw_b = mul_mat(Rh, Qv_b) (2T-1,BS,NH)  -- Rh (HD,2T-1,NH), Qv_b (HD,BS,NH): heads align in ne2
            ggml_tensor* BDraw_b = ggml_mul_mat(c, Rh, Qv_b);
            // BD_b[k,i] = BDraw_b[k - i + (T-1) - b*BS, i] : rel_shift with offset (T-1-b*BS)
            // valid because b*BS <= Tp-BS < T-1 for BS>=2 (offset>=0), rows in [0,2T-2].
            int off = (T - 1) - b * BS;
            ggml_tensor* BD_b = ggml_cont(c, ggml_view_3d(c, BDraw_b, T, BS, NH, BDraw_b->nb[1] - BDraw_b->nb[0],
                                                          BDraw_b->nb[2], (size_t)off * BDraw_b->nb[0]));
            ggml_tensor* sc = ggml_mul_mat(c, Kh, Qu_b); // (T,BS,NH)
            sc = ggml_add(c, sc, BD_b);
            sc = ggml_soft_max_ext(c, sc, nullptr, scale, 0.0f);
            ggml_tensor* ob = ggml_mul_mat(c, Vt, sc); // (HD,BS,NH)
            out_blocks = out_blocks ? ggml_concat(c, out_blocks, ob, 1) : ob;
        }
        // out_blocks (HD,Tp,NH) -> slice to (HD,T,NH)
        if (Tp != T)
            out_blocks = ggml_view_3d(c, out_blocks, HD, T, NH, out_blocks->nb[1], out_blocks->nb[2], 0);
        tiled = ggml_cont(c, out_blocks);
    }

    ggml_cgraph* gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, tiled);
    auto fill = [&](ggml_tensor* t, int s) {
        float* p = (float*)t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++)
            p[i] = frand((int)i + s);
    };
    fill(Q, 1);
    fill(K3, 100);
    fill(V3, 200);
    fill(R0, 300);
    fill(pbu, 400);
    fill(pbv, 500);
    ggml_graph_compute_with_ctx(c, gf, 1);

    float *rp = (float*)ref->data, *tp = (float*)tiled->data;
    double maxabs = 0;
    for (int64_t i = 0; i < ggml_nelements(ref); i++)
        maxabs = fmax(maxabs, fabs((double)rp[i] - (double)tp[i]));
    printf("T=%d BS=%d NB=%d Tp=%d  max abs diff = %.3e  %s\n", T, BS, NB, Tp, maxabs,
           maxabs < 1e-4 ? "PARITY OK" : "PARITY FAIL");
    ggml_free(c);
    return maxabs < 1e-4 ? 0 : 1;
}
