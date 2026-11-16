// src/core/worker_pool.h — bounded blocking resource pool (improvements Phase 4).
//
// The HTTP server serializes every request on one model_mutex + one backend
// instance, so N concurrent uploads run strictly one-at-a-time. A WorkerPool
// holds N independent resources (backend instances) and hands each request a
// free one via an RAII lease, so up to N requests run concurrently. N=1
// reproduces the historical single-instance behaviour.
//
// The pool logic (blocking acquire, RAII release, exhaustion) is unit-tested
// independently of the server (test-worker-pool); the server just parameterizes
// it with `CRISPASR_SERVER_WORKERS`.
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace core_pool {

template <typename T> class WorkerPool {
public:
    // Move-only RAII lease. Returns the resource to the pool on destruction.
    class Lease {
    public:
        Lease() = default;
        Lease(WorkerPool* pool, T* res) : pool_(pool), res_(res) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& o) noexcept : pool_(o.pool_), res_(o.res_) {
            o.pool_ = nullptr;
            o.res_ = nullptr;
        }
        Lease& operator=(Lease&& o) noexcept {
            if (this != &o) {
                release();
                pool_ = o.pool_;
                res_ = o.res_;
                o.pool_ = nullptr;
                o.res_ = nullptr;
            }
            return *this;
        }
        ~Lease() { release(); }

        T* get() const { return res_; }
        T* operator->() const { return res_; }
        T& operator*() const { return *res_; }
        explicit operator bool() const { return res_ != nullptr; }

    private:
        void release() {
            if (pool_ && res_) {
                pool_->give_back(res_);
                pool_ = nullptr;
                res_ = nullptr;
            }
        }
        WorkerPool* pool_ = nullptr;
        T* res_ = nullptr;
    };

    // Takes ownership of the resources (at least one). Their raw pointers are
    // the free set.
    explicit WorkerPool(std::vector<std::unique_ptr<T>> resources) : owned_(std::move(resources)) {
        free_.reserve(owned_.size());
        for (auto& r : owned_)
            free_.push_back(r.get());
    }

    size_t size() const { return owned_.size(); }

    // Free resources available right now (mainly for tests / metrics).
    size_t available() const {
        std::lock_guard<std::mutex> lk(m_);
        return free_.size();
    }

    // Block until a resource is free, then lease it.
    Lease acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !free_.empty(); });
        T* r = free_.back();
        free_.pop_back();
        return Lease(this, r);
    }

    // Non-blocking: lease a resource if one is free, else nullopt.
    std::optional<Lease> try_acquire() {
        std::lock_guard<std::mutex> lk(m_);
        if (free_.empty())
            return std::nullopt;
        T* r = free_.back();
        free_.pop_back();
        return Lease(this, r);
    }

private:
    friend class Lease;
    void give_back(T* r) {
        {
            std::lock_guard<std::mutex> lk(m_);
            free_.push_back(r);
        }
        cv_.notify_one();
    }

    mutable std::mutex m_;
    std::condition_variable cv_;
    std::vector<T*> free_;
    std::vector<std::unique_ptr<T>> owned_;
};

} // namespace core_pool
