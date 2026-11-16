// core/parallel_for.h — one place for "run this loop across the cores".
//
// Five copies of this had accumulated, each written for its own file:
//   firered_vad.cpp        firered_parallel_for   (static template)
//   marblenet_vad.cpp      marblenet_parallel_for (static template, identical)
//   omnivoice.cpp          ov_parallel_for        (std::function variant)
//   core/mel.cpp           inlined, no helper
//   core/foxnose_pipeline.cpp  inlined, no helper
//
// They come in TWO genuinely different shapes, and both are kept here rather
// than forcing one onto the other:
//
//   for_each_chunk — splits [0, n) into one contiguous block per thread.
//     Right when every item costs about the same (audio frames, feature rows),
//     which is what four of the five above do. Cheapest possible: no atomics,
//     one call per thread.
//
//   for_each_task — hands out item indices from an atomic counter, so threads
//     take more work as they finish. Right when per-item cost varies a lot,
//     where a static split would leave a thread idle holding the cheap items.
//     Also carries a SLOT index for callers that own per-thread resources
//     (foxnose_pipeline gives each worker its own model context this way).
//
// Both reuse the CALLING thread as one of the workers, so `n_threads == 1`
// costs nothing beyond the loop itself.
//
// DETERMINISM CONTRACT (why every caller here is safe to parallelise): `fn`
// must write only to storage owned by its own index/range. Any order-sensitive
// reduction — argmin, argmax, accumulation — belongs in a SERIAL pass
// afterwards, so results do not depend on thread timing.
//
// Not std::execution::par: libc++ gates <execution> behind
// _LIBCPP_HAS_EXPERIMENTAL_PSTL and ships no backend to link against (verified
// on Apple clang 21: the parallel overloads fail to resolve, and force-enabling
// them fails to link __pstl::__libdispatch::__dispatch_apply), while libstdc++
// implements it over Intel TBB, which this tree does not depend on.
//
// Not OpenMP: the tree does use it (~28 pragma sites), but every one is
// #ifdef _OPENMP-guarded and OpenMP is absent from the stock macOS toolchain,
// so those loops run serial there. core/mel.cpp keeps its OpenMP path for the
// platforms that have it and uses this helper as the portable path; the rest
// had no OpenMP path to begin with.

#pragma once

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace core_parallel {

// Number of threads to use for `n_items`, given a caller budget.
// `n_threads <= 0` means "ask the machine".
inline int resolve_threads(int n_items, int n_threads) {
    if (n_items <= 1)
        return 1;
    if (n_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        n_threads = (int)(hw == 0 ? 1u : hw);
    }
    return std::max(1, std::min(n_threads, n_items));
}

// fn(begin, end) over contiguous blocks partitioning [0, n_items).
template <typename F> void for_each_chunk(int n_items, int n_threads, F&& fn) {
    if (n_items <= 0)
        return;
    const int nt = resolve_threads(n_items, n_threads);
    if (nt == 1) {
        fn(0, n_items);
        return;
    }
    const int chunk = (n_items + nt - 1) / nt;
    std::vector<std::thread> pool;
    pool.reserve((size_t)nt - 1);
    for (int t = 1; t < nt; t++) {
        const int a = t * chunk, b = std::min(n_items, a + chunk);
        if (a >= b)
            break;
        pool.emplace_back([&fn, a, b]() { fn(a, b); });
    }
    fn(0, std::min(n_items, chunk)); // calling thread takes the first block
    for (auto& t : pool)
        t.join();
}

// fn(task_index, slot) for every task in [0, n_tasks), pulled from an atomic
// counter. `slot` is in [0, n_threads) and identifies WHICH per-thread
// resource to use; no two concurrent calls share a slot.
template <typename F> void for_each_task(int n_tasks, int n_threads, F&& fn) {
    if (n_tasks <= 0)
        return;
    const int nt = resolve_threads(n_tasks, n_threads);
    if (nt == 1) {
        for (int i = 0; i < n_tasks; i++)
            fn(i, 0);
        return;
    }
    std::atomic<int> next{0};
    auto run = [&](int slot) {
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= n_tasks)
                return;
            fn(i, slot);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve((size_t)nt - 1);
    for (int t = 1; t < nt; t++)
        pool.emplace_back(run, t);
    run(0); // calling thread is slot 0
    for (auto& t : pool)
        t.join();
}

} // namespace core_parallel
