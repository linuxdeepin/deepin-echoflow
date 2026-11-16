// fuzz_verify.cpp — mutation fuzzer for the C2PA verifier (parses untrusted
// input). Seeds from the c2pa-rs reference vectors + degenerate inputs, applies
// random byte/length/truncation mutations, and calls verify_wav on each — which
// must never crash, read out of bounds, or hang. Build with AddressSanitizer +
// UndefinedBehaviorSanitizer (see CMake option C2PA_AUDIO_FUZZ). Deterministic
// (no <random>) so a failure reproduces; pass an iteration count as argv[1].
#include "c2pa_native.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

using namespace crispasr::c2pa_native;

static Bytes read_file(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static uint64_t st = 0x1234567;
static uint32_t rnd() {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return uint32_t(st >> 33);
}

// The mutation stream is deterministic, so the iteration index is all you need
// to replay a fault: report it before dying rather than leaving a bare abort.
static long g_iter = -1;
static void report_iter() {
    const char* what = "(not a C++ exception)";
    if (auto e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception& ex) {
            what = ex.what();
        } catch (...) {
            what = "(non-std exception)";
        }
    }
    std::fprintf(stderr, "\n*** fault at iteration %ld: %s (replay: rerun with the same argv) ***\n", g_iter, what);
    std::fflush(stderr);
    std::abort();
}

int main(int argc, char** argv) {
    std::set_terminate(report_iter);
    long iters = argc > 1 ? std::atol(argv[1]) : 100000;
    std::vector<Bytes> seeds;
    for (int i = 2; i < argc; i++) {
        Bytes b = read_file(argv[i]);
        if (!b.empty())
            seeds.push_back(b);
    }
    // degenerate seeds: empty, magics, all-0xFF, a "deeply nested" JUMBF chain.
    seeds.push_back(Bytes{});
    seeds.push_back(Bytes{'R', 'I', 'F', 'F'});
    seeds.push_back(Bytes{'I', 'D', '3', 4});
    seeds.push_back(Bytes(32, 0xff));
    {
        const uint32_t N = 400;
        Bytes nested = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'C', '2', 'P', 'A'};
        uint32_t cs = 8 * N;
        nested.insert(nested.end(), {uint8_t(cs), uint8_t(cs >> 8), uint8_t(cs >> 16), uint8_t(cs >> 24)});
        for (uint32_t i = 0; i < N; i++) {
            uint32_t sz = 8 * (N - i);
            nested.insert(nested.end(), {uint8_t(sz >> 24), uint8_t(sz >> 16), uint8_t(sz >> 8), uint8_t(sz), 'j', 'u', 'm', 'b'});
        }
        seeds.push_back(std::move(nested));
    }

    for (long it = 0; it < iters; it++) {
        g_iter = it;
        if (it && it % 1000000 == 0) { // a heartbeat, so a timed-out run shows its reach
            std::fprintf(stderr, "iter %ld\n", it);
            std::fflush(stderr);
        }
        Bytes b = seeds[rnd() % seeds.size()];
        int muts = 1 + rnd() % 8;
        for (int m = 0; m < muts && !b.empty(); m++) {
            int op = rnd() % 4;
            if (op == 0)
                b[rnd() % b.size()] ^= uint8_t(1 << (rnd() % 8));
            else if (op == 1 && b.size() > 1)
                b.resize(1 + rnd() % (b.size() - 1)); // keep >=1 byte
            else if (op == 2)
                b[rnd() % b.size()] = uint8_t(rnd());
            else if (op == 3 && b.size() >= 4) {
                size_t o = rnd() % (b.size() - 3);
                for (int k = 0; k < 4; k++)
                    b[o + k] = uint8_t(rnd());
            }
        }
        VerifyResult r = verify_wav(b); // must never crash / OOB / hang
        (void)r;
    }
    printf("fuzz survived %ld iterations\n", iters);
    return 0;
}
