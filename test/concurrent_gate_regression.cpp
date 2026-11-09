#include <array>
#include <atomic>
#include <barrier>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "concurrent_gate.h"

int main() {
    // std::thread workers all share OpenMP slot zero: counters must remain safe
    // even when different workers alias one slot.
    index_concurrency::QuiescenceGate gate(4);
    std::array<unsigned, 16> values{};
    std::atomic<unsigned> errors{0};
    std::barrier start(8);
    std::vector<std::thread> workers;
    for (unsigned worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&, worker] {
            start.arrive_and_wait();
            for (unsigned iteration = 0; iteration < 2000; ++iteration) {
                if (worker < 2) {
                    index_concurrency::QuiescenceGate::WriteGuard write(gate);
                    unsigned next = values[0] + 1;
                    for (auto& value : values) value = next;
                } else {
                    index_concurrency::QuiescenceGate::ReadGuard read(gate);
                    for (auto value : values) if (value != values[0]) ++errors;
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    if (errors || values[0] != 4000) throw std::runtime_error("quiescence invariant failed");
    try { index_concurrency::QuiescenceGate::ReadGuard read(gate); throw 1; } catch (int) {}
    try { index_concurrency::QuiescenceGate::WriteGuard write(gate); throw 1; } catch (int) {}
    { index_concurrency::QuiescenceGate::ReadGuard read(gate); }
    std::cout << "Concurrent admission, reclamation exclusion, and RAII checks passed\n";
}
