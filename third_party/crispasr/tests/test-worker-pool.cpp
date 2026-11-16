// test-worker-pool.cpp — unit tests for core_pool::WorkerPool, the bounded
// blocking resource pool behind the server worker-pool (improvements Phase 4).

#include <catch2/catch_test_macros.hpp>

#include "core/worker_pool.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using core_pool::WorkerPool;

namespace {
struct Res {
    int id;
};
std::vector<std::unique_ptr<Res>> make(int n) {
    std::vector<std::unique_ptr<Res>> v;
    for (int i = 0; i < n; ++i)
        v.push_back(std::make_unique<Res>(Res{i}));
    return v;
}
} // namespace

TEST_CASE("pool size and initial availability", "[unit][worker-pool][improvements]") {
    WorkerPool<Res> pool(make(3));
    REQUIRE(pool.size() == 3);
    REQUIRE(pool.available() == 3);
}

TEST_CASE("lease decrements availability; releasing on scope-exit restores it", "[unit][worker-pool][improvements]") {
    WorkerPool<Res> pool(make(2));
    {
        auto a = pool.acquire();
        REQUIRE(a);
        REQUIRE(pool.available() == 1);
        {
            auto b = pool.acquire();
            REQUIRE(pool.available() == 0);
        }
        REQUIRE(pool.available() == 1); // b released
    }
    REQUIRE(pool.available() == 2); // a released
}

TEST_CASE("try_acquire returns nullopt when exhausted, resumes after release", "[unit][worker-pool][improvements]") {
    WorkerPool<Res> pool(make(1));
    auto a = pool.try_acquire();
    REQUIRE(a.has_value());
    REQUIRE_FALSE(pool.try_acquire().has_value()); // exhausted
    a.reset();                                     // release
    REQUIRE(pool.try_acquire().has_value());
}

TEST_CASE("distinct resources are handed out while leased", "[unit][worker-pool][improvements]") {
    WorkerPool<Res> pool(make(2));
    auto a = pool.acquire();
    auto b = pool.acquire();
    REQUIRE(a.get() != b.get()); // no double-hand-out
}

TEST_CASE("acquire blocks until a resource is released by another thread", "[unit][worker-pool][improvements]") {
    WorkerPool<Res> pool(make(1));
    auto held = pool.acquire();
    std::atomic<bool> got{false};
    std::thread t([&] {
        auto lease = pool.acquire(); // blocks until `held` is released
        got.store(true);
    });
    // Give the thread a moment; it must still be blocked (pool exhausted).
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE_FALSE(got.load());
    held = WorkerPool<Res>::Lease(); // release via move-assign of an empty lease
    t.join();
    REQUIRE(got.load());
}

TEST_CASE("move semantics: moved-from lease does not double-release", "[unit][worker-pool][improvements]") {
    WorkerPool<Res> pool(make(1));
    {
        auto a = pool.acquire();
        auto b = std::move(a); // a is now empty
        REQUIRE(pool.available() == 0);
        REQUIRE_FALSE(a);
        REQUIRE(b);
    } // only b releases
    REQUIRE(pool.available() == 1);
}
