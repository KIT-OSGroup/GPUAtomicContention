#include "benchmarking.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>
#include <vector>

namespace bm
{
    void run_randomized(std::vector<benchmark_t> &benchmarks) {
        std::random_device rand{};

        std::cout << "Initializing benchmarks..." << std::endl;

        for (auto &benchmark : benchmarks) {
            benchmark.initialize(&benchmark);
        }

        std::vector<int> pool(benchmarks.size());
        std::iota(pool.begin(), pool.end(), 0);

        assert(rand.max() - rand.min() >= pool.size());

        size_t total = 0;
        size_t current = 0;
        for (const auto &bm : benchmarks) {
            total += bm.requested_runs;
        }

        std::cout << "Running benchmarks in randomized order..." << std::endl;

        while (!pool.empty()) {
            uint32_t pool_idx = (rand() - rand.min()) % pool.size();
            int index = pool[pool_idx];

            std::cout << "." << std::flush;

            benchmarks[index].run(&benchmarks[index]);
            benchmarks[index].current_runs++;
            current++;

            if (benchmarks[index].current_runs >= benchmarks[index].requested_runs) {
                std::cout << "\n" << "[" << current << "/" << total << "] Benchmark '" << benchmarks[index].name << "' done" << std::endl;
                pool.erase(pool.begin() + pool_idx);
            }
        }

        std::cout << "Finalizing benchmarks..." << std::endl;

        for (auto &benchmark : benchmarks) {
            benchmark.finalize(&benchmark);
        }
    }

    void run_linear(std::vector<benchmark_t> &benchmarks) {
        std::cout << "Initializing benchmarks..." << std::endl;

        for (auto &benchmark : benchmarks) {
            benchmark.initialize(&benchmark);
        }

        for (auto &benchmark : benchmarks) {
            std::cout << "Running '" << benchmark.name << "'" << std::flush;
            while (benchmark.current_runs++ < benchmark.requested_runs) {
                std::cout << "." << std::flush;
                benchmark.run(&benchmark);
            }
            std::cout << std::endl;
        }

        for (auto &benchmark : benchmarks) {
            benchmark.finalize(&benchmark);
        }

        std::cout << "Finalizing benchmarks..." << std::endl;
    }

    void try_delete_result_file(const benchmark_t &benchmark) {
        char path[512];
        snprintf(path, 512, "/out/%s.bin", benchmark.name.data());
        if (!remove(path)) {
            std::cout << "File '" << path << "' already exists! Deleted file." << std::endl;
        }
    }

    void append_binary_result_file(const benchmark_t &benchmark, const char *data, size_t count) {
        char path[512];
        snprintf(path, 512, "/out/%s.bin", benchmark.name.data());

        std::ofstream run_file(path, std::ios::out | std::ios::binary | std::ios::app);
        run_file.write(data, count);
        run_file.close();
    }
}
