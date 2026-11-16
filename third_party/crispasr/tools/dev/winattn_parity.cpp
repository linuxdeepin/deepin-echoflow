// Standalone parity harness: full rel-pos attention (reference, mirrors
// core_conformer::build_block manual path) vs windowed block sliding-chunks
// attention. Validates the BD rel-shift-in-block math before wiring into the
// 24-layer FastConformer encoder.
//
// Build:
//   c++ -std=c++17 -O2 -I ggml/include winattn_parity.cpp \
//       build/ggml/src/libggml.dylib build/ggml/src/libggml-base.dylib \
//       build/ggml/src/libggml-cpu.dylib -o winattn_parity
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---- dims (small, CPU) ----
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
#ifndef WL
#define WL 2
#endif
#ifndef WR
#define WR 2
#endif
static const int D = HD * NH;
static const int NB = (T + BS - 1) / BS; // n_blocks (ceil — pads T up to Tp)
static const int Tp = NB * BS;           // padded seq len (multiple of BS)

static float frand(int i) { return sinf((float)i * 0.37f) * 0.9f; } // deterministic pseudo-random

static ggml_tensor* newf(ggml_context* c, int ne0, int ne1, int ne2, const char* name) {
    ggml_tensor* t = ggml_new_tensor_3d(c, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(t, name);
    return t;
}

// rel_shift: a (2T-1, T, H) -> (T, T, H), out[k,q]=a[k-q+(T-1), q]  (matches fastconformer.h)
static ggml_tensor* rel_shift(ggml_context* c, ggml_tensor* a) {
    const int t = (int)a->ne[1], h = (int)a->ne[2];
    return ggml_view_3d(c, a, t, t, h, a->nb[1] - a->nb[0], a->nb[2], (size_t)(t - 1) * a->nb[0]);
}

int main() {
    size_t memsz = (size_t)512 * 1024 * 1024;
    ggml_init_params ip = {memsz, nullptr, false};
    ggml_context* c = ggml_init(ip);

    // Inputs (post-projection): Q,K3,V3 as in build_block, plus R (rel-pos proj) and biases.
    ggml_tensor* Q = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);       // (d,T)
    ggml_tensor* K3 = newf(c, HD, NH, T, "K3");                        // (head_dim,n_heads,T)
    ggml_tensor* V3 = newf(c, HD, NH, T, "V3");                        // (head_dim,n_heads,T)
    ggml_tensor* R0 = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 2 * T - 1); // (d, 2T-1)
    ggml_tensor* pbu = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    ggml_tensor* pbv = ggml_new_tensor_1d(c, GGML_TYPE_F32, D);
    for (auto* t : {Q, K3, V3, R0, pbu, pbv}) {
        ggml_set_input(t);
    }

    const float scale = 1.0f / sqrtf((float)HD);
    const bool NOBD = getenv("NOBD") != nullptr;

    auto shape_heads = [&](ggml_tensor* x2d /*(d,T)*/) {
        // (d,T)->reshape(head_dim,n_heads,T)->permute(0,2,1,3)->(head_dim,T,n_heads)
        return ggml_permute(c, ggml_reshape_3d(c, x2d, HD, NH, T), 0, 2, 1, 3);
    };

    ggml_tensor* Q_u = ggml_add(c, Q, pbu);
    ggml_tensor* Q_v = ggml_add(c, Q, pbv);
    ggml_tensor* Q_u_h = ggml_cont(c, shape_heads(Q_u));                // (HD,T,NH)
    ggml_tensor* Q_v_h = ggml_cont(c, shape_heads(Q_v));               // (HD,T,NH)
    ggml_tensor* K_h = ggml_cont(c, ggml_permute(c, K3, 0, 2, 1, 3));   // (HD,T,NH)
    ggml_tensor* V_h = ggml_cont(c, ggml_permute(c, V3, 0, 2, 1, 3));   // (HD,T,NH)
    // R -> (HD, 2T-1, NH)
    ggml_tensor* R_h = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, R0, HD, NH, 2 * T - 1), 0, 2, 1, 3));

    // ---------- REFERENCE: full attention with window mask ----------
    ggml_tensor *ref_out, *win_out;
    {
        ggml_tensor* BD_raw = ggml_mul_mat(c, R_h, Q_v_h);      // (2T-1, T, NH)
        ggml_tensor* BD = ggml_cont(c, rel_shift(c, BD_raw));   // (T,T,NH)  BD[k,q,h]
        ggml_set_name(BD, "BD_ref");
        ggml_tensor* scores = ggml_mul_mat(c, K_h, Q_u_h);      // (T,T,NH)  scores[k,q,h]
        if(!NOBD) scores = ggml_add(c, scores, BD);
        // window mask (T,T): mask[k,q]=0 if q-WL<=k<=q+WR else -inf
        ggml_tensor* mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, T);
        ggml_set_name(mask, "refmask");
        ggml_set_input(mask);
        scores = ggml_add(c, scores, mask);
        scores = ggml_soft_max_ext(c, scores, nullptr, scale, 0.0f);
        ggml_tensor* V_t = ggml_cont(c, ggml_permute(c, V_h, 1, 0, 2, 3)); // (T,HD,NH)
        ggml_tensor* a = ggml_mul_mat(c, V_t, scores);                     // (HD,T,NH)
        ref_out = ggml_cont(c, a);
        ggml_set_name(ref_out, "ref_out");
    }

    // ---------- WINDOWED: block sliding-chunks ----------
    {
        // Pad query/key/value along T up to Tp = NB*BS (multiple of BS) with a zero tail.
        // Padded keys are masked invalid; padded queries are sliced off at the end.
        ggml_tensor* qtail = nullptr;
        auto padT = [&](ggml_tensor* x) {
            if (Tp == T)
                return x;
            if (!qtail) {
                qtail = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, Tp - T, NH);
                ggml_set_name(qtail, "qtail");
                ggml_set_input(qtail);
            }
            return ggml_concat(c, x, qtail, 1); // (HD,Tp,NH)
        };
        ggml_tensor* Qu_p = padT(Q_u_h);
        ggml_tensor* Qv_p = padT(Q_v_h);
        ggml_tensor* Kt = padT(K_h);
        ggml_tensor* Vt = padT(V_h);

        // Pad Kt, Vt with BS zeros on each side along Tp: (HD, Tp+2*BS, NH)
        ggml_tensor* zpad = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, BS, NH); // zeroed on host below
        ggml_set_name(zpad, "zpad");
        ggml_set_input(zpad);
        ggml_tensor* Kp = ggml_concat(c, ggml_concat(c, zpad, Kt, 1), zpad, 1); // (HD,Tp+2*BS,NH)
        ggml_tensor* Vp = ggml_concat(c, ggml_concat(c, zpad, Vt, 1), zpad, 1);

        // ggml forbids overlapping views (contiguous product > source). Build the
        // 3-block band via non-overlapping block reshape + 3 stride-1 block slices + concat.
        // Kp -> blocks (HD, BS, NB+2, NH); band = [blocks bo, bo+1, bo+2] = [orig bo-1, bo, bo+1].
        auto band3 = [&](ggml_tensor* Xp) {
            ggml_tensor* Xb = ggml_reshape_4d(c, Xp, HD, BS, NB + 2, NH);
            auto slc = [&](int i) {
                return ggml_cont(c, ggml_view_4d(c, Xb, HD, BS, NB, NH, Xb->nb[1], Xb->nb[2], Xb->nb[3],
                                                 (size_t)i * Xb->nb[2]));
            };
            return ggml_concat(c, ggml_concat(c, slc(0), slc(1), 1), slc(2), 1); // (HD,3B,NB,NH)
        };
        ggml_tensor* K_band = band3(Kp); // (HD,3B,NB,NH), key j natural: k=(bo-1)BS+j
        ggml_tensor* V_band = band3(Vp);

        // Q blocks: (HD, BS, NB, NH) from padded Qu_p/Qv_p (HD,Tp,NH)
        Qu_p = ggml_cont(c, Qu_p);
        Qv_p = ggml_cont(c, Qv_p);
        ggml_tensor* Qu_blk =
            ggml_view_4d(c, Qu_p, HD, BS, NB, NH, Qu_p->nb[1], (size_t)BS * Qu_p->nb[1], Qu_p->nb[2], 0);
        ggml_tensor* Qv_blk =
            ggml_view_4d(c, Qv_p, HD, BS, NB, NH, Qv_p->nb[1], (size_t)BS * Qv_p->nb[1], Qv_p->nb[2], 0);
        Qu_blk = ggml_cont(c, Qu_blk);
        Qv_blk = ggml_cont(c, Qv_blk);

        // scores_blk = mul_mat(K_band, Qu_blk) -> (3B, BS, NB, NH)  [key j, query i]
        ggml_tensor* sc = ggml_mul_mat(c, K_band, Qu_blk);

        // BD block: R_sl = R rows [T-2B .. T-2B+RB-1] = 4B-1 rows (natural order).
        // R_h is (HD, 2T-1, NH). slice along ne1.
        // R_h row r <-> rel index r used at key-query offset k-q+(T-1)=r.
        // For block query q=bB+i, key k=(b-1)BS+j: needed row = (T-1)-BS+j-i.
        // With natural slice start (T-2B): BDraw_blk row m=(BS-1)+j-i, m in [0,4B-2].
        const int RB = 4 * BS - 1;
        ggml_tensor* R_sl = ggml_view_3d(c, R_h, HD, RB, NH, R_h->nb[1], R_h->nb[2], (size_t)(T - 2 * BS) * R_h->nb[1]);
        R_sl = ggml_cont(c, R_sl);
        // (HD,RB,NH) -> (HD,RB,1,NH): block axis size 1 so mul_mat broadcasts R over blocks
        // and matches heads in ne3 (else ggml mixes the head and block batch dims).
        R_sl = ggml_reshape_4d(c, R_sl, HD, RB, 1, NH);
        ggml_tensor* BDraw_blk = ggml_mul_mat(c, R_sl, Qv_blk); // (4B-1, BS, NB, NH)
        // in-block rel-shift (standard form): BD_blk[j,i] = BDraw_blk[(BS-1)+j-i, i]
        // ne0=3B (j, stride nb0), ne1=BS (i, stride nb1-nb0), offset (BS-1)*nb0. All strides >=0, natural key order.
        ggml_tensor* BD_blk =
            ggml_view_4d(c, BDraw_blk, 3 * BS, BS, NB, NH, BDraw_blk->nb[1] - BDraw_blk->nb[0], BDraw_blk->nb[2],
                         BDraw_blk->nb[3], (size_t)(BS - 1) * BDraw_blk->nb[0]);
        BD_blk = ggml_cont(c, BD_blk);
        ggml_set_name(BD_blk, "BD_blk");

        if(!NOBD) sc = ggml_add(c, sc, BD_blk);

        // band mask (3B, BS, NB): additive, encodes window + boundary. NATURAL key order.
        ggml_tensor* bmask = ggml_new_tensor_3d(c, GGML_TYPE_F32, 3 * BS, BS, NB);
        ggml_set_name(bmask, "bmask");
        ggml_set_input(bmask);
        // broadcast over heads: reshape sc (3B,BS,NB,NH); add bmask (3B,BS,NB,1)
        ggml_tensor* bmask4 = ggml_reshape_4d(c, bmask, 3 * BS, BS, NB, 1);
        sc = ggml_add(c, sc, bmask4);
        sc = ggml_soft_max_ext(c, sc, nullptr, scale, 0.0f);

        // attn = sum_j softmax[j,i] * V_band[:,j]  -> mul_mat(V_band_t, sc)
        // V_band (HD,3B,NB,NH) -> V_band_t (3B,HD,NB,NH); mul_mat(V_band_t, sc(3B,BS,NB,NH)) -> (HD,BS,NB,NH)
        ggml_tensor* V_band_t = ggml_cont(c, ggml_permute(c, V_band, 1, 0, 2, 3)); // (3B,HD,NB,NH)
        ggml_tensor* a_blk = ggml_mul_mat(c, V_band_t, sc);                        // (HD,BS,NB,NH)
        // reshape to (HD, Tp, NH) then slice off padded queries -> (HD, T, NH)
        ggml_tensor* a_full = ggml_cont(c, ggml_reshape_3d(c, ggml_cont(c, a_blk), HD, Tp, NH));
        if (Tp != T)
            a_full = ggml_view_3d(c, a_full, HD, T, NH, a_full->nb[1], a_full->nb[2], 0);
        win_out = ggml_cont(c, a_full);
        ggml_set_name(win_out, "win_out");
    }

    ggml_cgraph* gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, ref_out);
    ggml_build_forward_expand(gf, win_out);

    // ---- fill inputs ----
    auto fill = [&](ggml_tensor* t, int seed) {
        float* p = (float*)t->data;
        int64_t n = ggml_nelements(t);
        for (int64_t i = 0; i < n; i++)
            p[i] = frand((int)i + seed);
    };
    fill(Q, 1);
    fill(K3, 100);
    fill(V3, 200);
    fill(R0, 300);
    fill(pbu, 400);
    fill(pbv, 500);
    // zero the zpad (+ qtail if T padded)
    {
        ggml_tensor* z = ggml_get_tensor(c, "zpad");
        memset(z->data, 0, ggml_nbytes(z));
        ggml_tensor* qt = ggml_get_tensor(c, "qtail");
        if (qt)
            memset(qt->data, 0, ggml_nbytes(qt));
    }
    // ref window mask
    {
        ggml_tensor* m = ggml_get_tensor(c, "refmask");
        float* p = (float*)m->data;
        for (int q = 0; q < T; q++)
            for (int k = 0; k < T; k++)
                p[(size_t)q * T + k] = (k >= q - WL && k <= q + WR) ? 0.0f : -1e9f;
    }
    // band mask natural: bmask[j, i, b], natural key j (0..3B-1) -> global k=(b-1)BS+j
    {
        ggml_tensor* m = ggml_get_tensor(c, "bmask");
        float* p = (float*)m->data;
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < BS; i++)
                for (int j = 0; j < 3 * BS; j++) {
                    int q = b * BS + i;
                    int k = (b - 1) * BS + j;
                    bool vis = (k >= 0 && k < T && k >= q - WL && k <= q + WR);
                    p[((size_t)b * BS + i) * (3 * BS) + j] = vis ? 0.0f : -1e9f;
                }
    }

    ggml_graph_compute_with_ctx(c, gf, 1);

    // ---- compare ----
    float* rp = (float*)ref_out->data;
    float* wp = (float*)win_out->data;
    int64_t n = ggml_nelements(ref_out);
    double maxabs = 0, sumsq = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = fabs((double)rp[i] - (double)wp[i]);
        if (d > maxabs)
            maxabs = d;
        sumsq += d * d;
    }
    printf("ref_out ne=(%lld,%lld,%lld)  win_out ne=(%lld,%lld,%lld)\n", (long long)ref_out->ne[0],
           (long long)ref_out->ne[1], (long long)ref_out->ne[2], (long long)win_out->ne[0], (long long)win_out->ne[1],
           (long long)win_out->ne[2]);
    printf("max abs diff = %.3e   rms = %.3e\n", maxabs, sqrt(sumsq / n));
    printf(maxabs < 1e-4 ? "PARITY OK\n" : "PARITY FAIL\n");

    if (getenv("DBG")) {
        ggml_tensor* BDr = ggml_get_tensor(c, "BD_ref");  // (T,T,NH) [k,q,h]
        ggml_tensor* BDb = ggml_get_tensor(c, "BD_blk");  // (3B,BS,NB,NH) [j,i,b,h]
        float* rr = (float*)BDr->data;
        float* bb = (float*)BDb->data;
        int h = 0;
        printf("\n[DBG] BD ref[k,q] vs blk[j,i,b] for h=0:\n");
        for (int b = 0; b < NB; b++)
            for (int i = 0; i < BS; i++) {
                int q = b * BS + i;
                for (int j = 0; j < 3 * BS; j++) {
                    int k = (b - 1) * BS + j;
                    if (k < 0 || k >= T)
                        continue;
                    if (!(k >= q - WL && k <= q + WR))
                        continue;
                    float vref = rr[(size_t)h * T * T + (size_t)q * T + k]; // [k,q,h]: ne0=k,ne1=q
                    float vblk = bb[((size_t)h * NB * BS + (size_t)b * BS + i) * (3 * BS) + j];
                    printf("  q=%2d k=%2d (b=%d,i=%d,j=%d): ref=% .4f blk=% .4f %s\n", q, k, b, i, j, vref, vblk,
                           fabs(vref - vblk) > 1e-4 ? "<<<DIFF" : "");
                }
            }
    }
    ggml_free(c);
    return maxabs < 1e-4 ? 0 : 1;
}
