#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace bm
{
    struct benchmark_t
    {
        std::string name;
        int grid_size;
        int block_size;
        int current_runs;
        int requested_runs;
        std::function<void(benchmark_t *)> initialize;
        std::function<void(benchmark_t *)> run;
        std::function<void(benchmark_t *)> finalize;
        void *user;
    };

    void run_randomized(std::vector<benchmark_t> &benchmarks);
    void run_linear(std::vector<benchmark_t> &benchmarks);
    void try_delete_result_file(const benchmark_t &benchmark);
    void append_binary_result_file(const benchmark_t &benchmark, const char *data, size_t count);
}
