#!/usr/bin/env python3
"""Diagnose the two failures that turning the unit tier on exposed.

Both were invisible because CI ran 1 of 162 unit tests until 2026-07-29.

  1. core_adaln: the modulate6 + apply_norm_modulation graph disagrees with the
     plain-float reference in tests/test-core-adaln.cpp by max|Δ| 0.658 and 1.962
     against a 1e-4 tolerance — under CLANG only; gcc passes. Four orders of
     magnitude is not FP contraction, and a compiler-dependent numeric difference
     that large usually means undefined behaviour. core_adaln is production DiT
     modulation (f5-tts, cosyvoice3), so this may be a live correctness bug.

  2. test-vad: "Subprocess aborted" in DEBUG on gcc and gcc-arm64; Release passes.
     An assert/abort, not a comparison. The model IS tracked, so it is not a
     missing fixture. build.yml runs this test in isolation via `ctest -R
     ^test-vad$` — a Release leg, which is how it stayed green.

Guessing from source got nowhere, so this does what LEARNINGS.md prescribes: a
stage-by-stage diff. It builds the SAME tree with gcc and clang, runs both tests
under each, and dumps every intermediate of the adaln graph next to the reference
so the FIRST divergence is identified rather than inferred.

The dump isolates the two halves:
  * the six modulation views ARE the projection output (emb), so comparing them
    tells us whether modulate6 is wrong;
  * `out` given correct views tells us whether apply_norm_modulation is wrong.

adaln_diag.cpp is appended to the clone's tests/CMakeLists.txt and built as a
normal target, so it inherits the project's own include/link setup instead of a
hand-rolled compile line.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
RESULTS = WORK / "adaln_vad_diag.json"
CLONE = Path("/kaggle/temp/CrispASR")  # not /kaggle/working (gotcha #22)

if not CLONE.exists():
    try:
        subprocess.run(["git", "clone", "--recurse-submodules",
                        "https://github.com/CrispStrobe/CrispASR.git", str(CLONE)],
                       check=True, timeout=1800)
    except Exception as e:  # noqa: BLE001
        print(f"clone failed: {e}", flush=True)

sys.path.insert(0, str(CLONE / "tools" / "kaggle") if CLONE.exists()
                else str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()


def sh(cmd, cwd=None, timeout=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def die(stage, **extra):
    kh.step(f"{stage}_FAILED", **extra)
    RESULTS.write_text(json.dumps({"verdict": "ERROR", "stage": stage, **extra}, indent=2))
    raise SystemExit(f"FATAL at {stage}: {extra}")


if not CLONE.exists():
    die("clone", err="clone missing; a script kernel cannot rely on bundled files")

kh.resolve_hf_token()

# ── the instrumented dump ────────────────────────────────────────────────────────
# Mirrors tests/test-core-adaln.cpp exactly (same seed, same dims) but marks the six
# modulation views AND the output as graph outputs and prints every one, so the
# comparison is per-stage instead of a single end-to-end delta.
DIAG_CPP = r'''
#include "core/adaln.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t nx() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 11); }
    float sym() { return (float)((nx() % 20001) / 10000.0 - 1.0); }
};
float silu(float v) { return v / (1.0f + std::exp(-v)); }
void dump(const char* tag, const std::vector<float>& v) {
    std::printf("%-22s", tag);
    for (float e : v) std::printf(" % .6f", e);
    std::printf("\n");
}
} // namespace

int main() {
    // Identical to the unit test: dim = cond_dim = 6, T_x = 3, seed 0xADA1F.
    const int dim = 6, cond_dim = dim, T_x = 3;
    Rng rng(0xADA1FULL);
    std::vector<float> temb(cond_dim), w((size_t)cond_dim * 6 * dim), b(6 * dim), x((size_t)dim * T_x);
    for (auto* v : {&temb, &w, &b, &x})
        for (auto& e : *v) e = rng.sym();

    for (int silu_mode = 1; silu_mode >= 0; silu_mode--) {
        const bool apply_silu = silu_mode == 1;
        std::printf("\n===== apply_silu=%s =====\n", apply_silu ? "true" : "false");

        std::vector<uint8_t> meta(4 * 1024 * 1024);
        ggml_init_params ip{meta.size(), meta.data(), true};
        ggml_context* ctx = ggml_init(ip);

        ggml_tensor* t_temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cond_dim, 1);
        ggml_tensor* t_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cond_dim, 6 * dim);
        ggml_tensor* t_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6 * dim);
        ggml_tensor* t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, T_x);
        for (auto* t : {t_temb, t_w, t_b, t_x}) ggml_set_input(t);
        ggml_set_name(t_temb, "temb"); ggml_set_name(t_w, "w");
        ggml_set_name(t_b, "b"); ggml_set_name(t_x, "x");

        auto m = core_adaln::modulate6(ctx, t_temb, t_w, t_b, apply_silu);
        ggml_tensor* views[6] = {m.shift_msa, m.scale_msa, m.gate_msa, m.shift_mlp, m.scale_mlp, m.gate_mlp};
        const char* vnames[6] = {"shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"};
        // A view is contiguous-copied so it can be read back as its own output.
        ggml_tensor* vcopy[6];
        for (int i = 0; i < 6; i++) {
            vcopy[i] = ggml_cont(ctx, views[i]);
            ggml_set_output(vcopy[i]);
            ggml_set_name(vcopy[i], vnames[i]);
        }
        ggml_tensor* out = core_adaln::apply_norm_modulation(ctx, t_x, m.scale_msa, m.shift_msa, 1e-6f);
        ggml_set_output(out);
        ggml_set_name(out, "out");

        ggml_cgraph* gf = ggml_new_graph(ctx);
        for (int i = 0; i < 6; i++) ggml_build_forward_expand(gf, vcopy[i]);
        ggml_build_forward_expand(gf, out);

        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 2048, false, false);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "temb"), temb.data(), 0, temb.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"), w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "b"), b.data(), 0, b.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x.data(), 0, x.size() * sizeof(float));
        ggml_backend_sched_graph_compute(sched, gf);

        // ── reference projection ────────────────────────────────────────────────
        std::vector<float> emb(cond_dim);
        for (int i = 0; i < cond_dim; i++) emb[i] = apply_silu ? silu(temb[i]) : temb[i];
        std::vector<float> proj(6 * dim);
        for (int o = 0; o < 6 * dim; o++) {
            float acc = 0;
            for (int i = 0; i < cond_dim; i++) acc += w[(size_t)o * cond_dim + i] * emb[i];
            proj[o] = acc + b[o];
        }

        for (int i = 0; i < 6; i++) {
            std::vector<float> got(dim);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, vnames[i]), got.data(), 0, got.size() * sizeof(float));
            std::vector<float> ref(proj.begin() + (size_t)i * dim, proj.begin() + (size_t)(i + 1) * dim);
            float md = 0;
            for (int d = 0; d < dim; d++) md = std::fmax(md, std::fabs(got[d] - ref[d]));
            std::printf("-- %s  max|d| = %.6f %s\n", vnames[i], md, md < 1e-4f ? "OK" : "<<<< DIVERGES");
            dump("   graph", got);
            dump("   reference", ref);
        }

        // ── reference output ────────────────────────────────────────────────────
        const float* shift = proj.data() + 0 * dim;
        const float* scale = proj.data() + 1 * dim;
        std::vector<float> refout((size_t)dim * T_x);
        for (int t = 0; t < T_x; t++) {
            float mean = 0;
            for (int d = 0; d < dim; d++) mean += x[(size_t)t * dim + d];
            mean /= dim;
            float var = 0;
            for (int d = 0; d < dim; d++) { float c = x[(size_t)t * dim + d] - mean; var += c * c; }
            var /= dim;
            float inv = 1.0f / std::sqrt(var + 1e-6f);
            for (int d = 0; d < dim; d++) {
                float nx = (x[(size_t)t * dim + d] - mean) * inv;
                refout[(size_t)t * dim + d] = nx * (1.0f + scale[d]) + shift[d];
            }
        }
        std::vector<float> gotout((size_t)dim * T_x);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), gotout.data(), 0, gotout.size() * sizeof(float));
        float md = 0;
        for (size_t i = 0; i < gotout.size(); i++) md = std::fmax(md, std::fabs(gotout[i] - refout[i]));
        std::printf("-- out        max|d| = %.6f %s\n", md, md < 1e-4f ? "OK" : "<<<< DIVERGES");
        dump("   graph", gotout);
        dump("   reference", refout);

        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    return 0;
}
'''

(CLONE / "tests" / "adaln_diag.cpp").write_text(DIAG_CPP)
with open(CLONE / "tests" / "CMakeLists.txt", "a") as f:
    f.write('''
# adaln-diag (Kaggle diagnosis of the clang-only core_adaln divergence) — appended
# by tools/kaggle/adaln-vad-diag, not part of the tree.
add_executable(adaln-diag adaln_diag.cpp)
target_include_directories(adaln-diag PRIVATE ../include ../ggml/include ../src ../examples)
target_link_libraries(adaln-diag PRIVATE common)
''')

kh.install_build_toolchain()
# The Kaggle image ships gcc but NOT clang, and the whole point of this kernel is
# the gcc-vs-clang A/B — run 1 died at configure.clang-release on "CMAKE_C_COMPILER:
# clang is not a full path and was not found in the PATH".
sh("apt-get install -y --no-install-recommends clang || true", timeout=900)
for _c in ("gcc", "clang"):
    _r = sh(f"which {_c} && {_c} --version | head -1")
    print(f"  compiler {_c}: {_r.stdout.strip() or 'ABSENT'}", flush=True)
results = {}


def build_and_run(label, cc, cxx, build_type):
    bdir = f"build-{label}"
    flags = (f"-DCMAKE_BUILD_TYPE={build_type} -DGGML_NATIVE=OFF -DCRISPASR_BUILD_TESTS=ON "
             f"-DCMAKE_C_COMPILER={cc} -DCMAKE_CXX_COMPILER={cxx} "
             + " ".join(kh.cache_and_link_flags()))
    if not sh(f"which {cc}").stdout.strip():
        # Record and move on. Aborting the whole run because ONE combo cannot be
        # built is how run 1 threw away three working combos.
        results[label] = {"skipped": f"{cc} not installed"}
        kh.step(f"skip.{label}", reason=f"{cc} absent")
        return
    kh.step(f"configure.{label}")
    with kh.build_heartbeat(f"cmake.{label}", 30):
        r = sh(f"cmake -S . -B {bdir} -G Ninja {flags}", cwd=str(CLONE), timeout=1800)
    if r.returncode != 0:
        results[label] = {"configure_failed": r.stderr[-2000:]}
        kh.step(f"configure.{label}_FAILED", err=r.stderr[-800:])
        return
    jobs = kh.safe_build_jobs(gpu=True)
    with kh.build_heartbeat(f"ninja.{label}", 30):
        try:
            kh.sh_with_progress(f"cmake --build {bdir} -j{jobs} --target test-core-adaln test-vad adaln-diag",
                                cwd=str(CLONE))
        except Exception as e:  # noqa: BLE001
            results[label] = {"build_failed": str(e)[-2000:]}
            kh.step(f"build.{label}_FAILED", err=str(e)[-800:])
            return

    out = {}
    for exe in ("test-core-adaln", "test-vad", "adaln-diag"):
        r = sh(f"./{bdir}/bin/{exe}", cwd=str(CLONE), timeout=900)
        out[exe] = {"rc": r.returncode,
                    "stdout": r.stdout[-6000:],
                    "stderr": r.stderr[-4000:]}
        kh.step(f"run.{label}.{exe}", rc=r.returncode)
        print(f"\n########## [{label}] {exe}  rc={r.returncode} ##########", flush=True)
        print(r.stdout[-6000:], flush=True)
        if r.stderr.strip():
            print(f"--- stderr ---\n{r.stderr[-4000:]}", flush=True)
    results[label] = out


# gcc-debug second: it reproduces the test-vad abort and reuses the warm gcc
# ccache, so the cheapest repro lands before the two cold clang builds.
for label, cc, cxx, bt in (("gcc-release", "gcc", "g++", "Release"),
                           ("gcc-debug", "gcc", "g++", "Debug"),
                           ("clang-release", "clang", "clang++", "Release"),
                           ("clang-debug", "clang", "clang++", "Debug")):
    build_and_run(label, cc, cxx, bt)

RESULTS.write_text(json.dumps(results, indent=2))
print("\n===== SUMMARY =====", flush=True)
for label, out in results.items():
    if "test-core-adaln" not in out:
        print(f"  {label:14s} {out}", flush=True)
        continue
    print(f"  {label:14s} adaln={out['test-core-adaln']['rc']:3d}  vad={out['test-vad']['rc']:3d}", flush=True)
kh.step("done", summary={k: ({e: v[e].get("rc") for e in v if isinstance(v[e], dict)} or v)
                         for k, v in results.items()})
try:
    kh.export_ccache_tar()
except Exception:  # noqa: BLE001
    pass
