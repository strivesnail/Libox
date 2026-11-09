#ifndef INDEX_QUIESCENCE_GATE_H
#define INDEX_QUIESCENCE_GATE_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <omp.h>

namespace index_concurrency {

// Keep ordinary operations concurrent while publishing/reclaiming topology only
// after all active operations have left. Reader counters are separate cache lines
// so unrelated workers do not contend on one shared reference count.
class QuiescenceGate {
    struct alignas(64) ReaderSlot { std::atomic<size_t> count{0}; };
    const size_t slot_count_;
    std::unique_ptr<ReaderSlot[]> readers_;
    std::atomic<bool> pending_{false};
    std::mutex writers_;

public:
    explicit QuiescenceGate(int threads)
        : slot_count_(std::max(threads, 1)), readers_(new ReaderSlot[slot_count_]) {}
    QuiescenceGate(const QuiescenceGate&) = delete;
    QuiescenceGate& operator=(const QuiescenceGate&) = delete;

    class ReadGuard {
        QuiescenceGate* gate_;
        const size_t slot_;
    public:
        explicit ReadGuard(QuiescenceGate& gate)
            : gate_(&gate), slot_(static_cast<size_t>(omp_get_thread_num()) % gate.slot_count_) {
            for (;;) {
                while (gate.pending_.load()) std::this_thread::yield();
                gate.readers_[slot_].count.fetch_add(1);
                // Sequential consistency pairs this check with a writer closing
                // admission before draining counters. A late entrant backs out
                // without touching index storage.
                if (!gate.pending_.load()) break;
                gate.readers_[slot_].count.fetch_sub(1);
            }
        }
        ReadGuard(const ReadGuard&) = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;
        void unlock() {
            if (gate_) {
                gate_->readers_[slot_].count.fetch_sub(1);
                gate_ = nullptr;
            }
        }
        ~ReadGuard() { unlock(); }
    };

    class WriteGuard {
        QuiescenceGate& gate_;
        std::unique_lock<std::mutex> lock_;
    public:
        WriteGuard(QuiescenceGate& gate, std::defer_lock_t)
            : gate_(gate), lock_(gate.writers_, std::defer_lock) {}
        explicit WriteGuard(QuiescenceGate& gate) : WriteGuard(gate, std::defer_lock) { lock(); }
        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;
        bool owns_lock() const { return lock_.owns_lock(); }
        // The caller must release its own ReadGuard before requesting a writer.
        void lock() {
            lock_.lock();
            gate_.pending_.store(true);
            for (size_t i = 0; i < gate_.slot_count_; ++i)
                while (gate_.readers_[i].count.load() != 0) std::this_thread::yield();
        }
        ~WriteGuard() {
            if (lock_.owns_lock()) gate_.pending_.store(false);
            // unique_lock releases writers_ after admission reopens.
        }
    };
};

} // namespace index_concurrency
#endif
