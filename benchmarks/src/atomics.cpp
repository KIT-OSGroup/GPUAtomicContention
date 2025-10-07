#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <linux/limits.h>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
#include <sys/types.h>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "buffer.hpp"
#include "utility.hpp"
#include "atomic.hpp"

struct benchmark_t {
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

static std::random_device s_rand{};
static std::default_random_engine s_prand{s_rand()};
static std::vector<benchmark_t> s_benchmarks{};

template <typename RunFunctor, typename PreFunctor, typename PostFunctor>
void register_benchmark(
    std::string name,
    int grid_size,
    int block_size,
    int runs,
    PreFunctor initialize,
    RunFunctor run,
    PostFunctor finalize,
    void *user = nullptr
) {
    s_benchmarks.push_back({
        .name = name,
        .grid_size = grid_size,
        .block_size = block_size,
        .current_runs = 0,
        .requested_runs = runs,
        .initialize = initialize,
        .run = run,
        .finalize = finalize,
        .user = user,
    });
}

void initialize_benchmarks() {
    for (auto &benchmark : s_benchmarks) {
        benchmark.initialize(&benchmark);
    }
}

void run_benchmarks_randomized() {
    std::vector<int> pool(s_benchmarks.size());
    std::iota(pool.begin(), pool.end(), 0);

    assert(s_rand.max() - s_rand.min() >= pool.size());

    std::cout << "Running benchmarks in randomized order..." << std::endl;

    while (!pool.empty()) {
        uint32_t pool_idx = (s_rand() - s_rand.min()) % pool.size();
        int index = pool[pool_idx];

        std::cout << "." << std::flush;

        s_benchmarks[index].run(&s_benchmarks[index]);
        s_benchmarks[index].current_runs++;

        if (s_benchmarks[index].current_runs >= s_benchmarks[index].requested_runs) {
            std::cout << "\n" << "Benchmark '" << s_benchmarks[index].name << "' done" << std::endl;
            pool.erase(pool.begin() + pool_idx);
        }
    }
}

void run_benchmarks_linear() {
    for (auto &benchmark : s_benchmarks) {
        std::cout << "Running '" << benchmark.name << "'" << std::flush;
        while (benchmark.current_runs++ < benchmark.requested_runs) {
            std::cout << "." << std::flush;
            benchmark.run(&benchmark);
        }
        std::cout << std::endl;
    }
}

void finalize_benchmarks() {
    for (auto &benchmark : s_benchmarks) {
        benchmark.finalize(&benchmark);
    }
}

__global__ void kernel_device_lock_or_enclosing(simt::atomic<uint32_t> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();

    uint32_t sleep = 8;
    bool done = false;
    while (done == false) {
        if (var->fetch_or(true, simt::memory_order_acquire) == false) {
            var->store(false, simt::memory_order_release);
            done = true;
        }

        __sleep(sleep);
        if (sleep < 256) {
            sleep *= 2;
        }
    }

    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_device_lock_cas_enclosing(simt::atomic<uint32_t> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();

    uint32_t sleep = 8;
    bool done = false;
    uint32_t expected = false;
    while (done == false) {
        if (var->compare_exchange_strong(expected, true, simt::memory_order_acquire)) {
            var->store(false, simt::memory_order_release);
            done = true;
        }

        __sleep(sleep);
        if (sleep < 256) {
            sleep *= 2;
        }
    }

    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_system_lock_or_enclosing(simt::atomic<uint32_t, simt::thread_scope_system> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();

    uint32_t sleep = 8;
    bool done = false;
    while (done == false) {
        if (var->fetch_or(true, simt::memory_order_acquire) == false) {
            var->store(false, simt::memory_order_release);
            done = true;
        }

        __sleep(sleep);
        if (sleep < 256) {
            sleep *= 2;
        }
    }

    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_system_lock_cas_enclosing(simt::atomic<uint32_t, simt::thread_scope_system> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();

    uint32_t sleep = 8;
    bool done = false;
    uint32_t expected = false;
    while (done == false) {
        if (var->compare_exchange_strong(expected, true, simt::memory_order_acquire)) {
            var->store(false, simt::memory_order_release);
            done = true;
        }

        __sleep(sleep);
        if (sleep < 256) {
            sleep *= 2;
        }
    }

    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_store_volatile(volatile uint32_t *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();
    #pragma unroll
    for (uint32_t i = 0; i < 128; i++) {
        *var = i;
    }
    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_store_mmio(uint32_t *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();

    #pragma unroll
    for (uint32_t i = 0; i < 128; i++) {
#if defined (__HIP_PLATFORM_NVIDIA__)
        asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(var), "r"(i) : "memory");
#elif defined(__HIP_PLATFORM_AMD__)
        asm volatile(
            "s_waitcnt lgkmcnt(0) vmcnt(0) \n"
            "s_waitcnt_vscnt null, 0x0 \n"
            "global_store_dword %0, %1, off glc slc dlc \n"
            :: "v"(var), "v"(i) : "memory");
#endif
    }

    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_device_store_atomic_relaxed(simt::atomic<uint32_t> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();
    #pragma unroll
    for (uint32_t i = 0; i < 128; i++) {
        var->store(i, simt::memory_order_relaxed);
    }
    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_device_store_atomic_seqcst(simt::atomic<uint32_t> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();
    #pragma unroll
    for (uint32_t i = 0; i < 128; i++) {
        var->store(i, simt::memory_order_seq_cst);
    }
    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_system_store_atomic_relaxed(simt::atomic<uint32_t, simt::thread_scope_system> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();
    #pragma unroll
    for (uint32_t i = 0; i < 128; i++) {
        var->store(i, simt::memory_order_relaxed);
    }
    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_system_store_atomic_seqcst(simt::atomic<uint32_t, simt::thread_scope_system> *var, uint64_t *times) {
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();
    #pragma unroll
    for (uint32_t i = 0; i < 128; i++) {
        var->store(i, simt::memory_order_seq_cst);
    }
    uint64_t end = __global_clock();
    times[id] = end - start;
}

__global__ void kernel_device_lock_or_enclosing_variable_lanes(simt::atomic<uint32_t> *var, uint64_t *times, uint64_t *masks) {
    int lid = lane_id();
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t active_mask = masks[id / warpSize];

    if (active_mask & (1ull << lid)) {
        int active_lane_count = popcount(active_mask);
        int offset = mask_relative_lane_id(lid, active_mask);
        uint64_t start = __global_clock();

         uint32_t sleep = 8;
         bool done = false;
         while (done == false) {
             if (var->fetch_or(true, simt::memory_order_acquire) == false) {
                 var->store(false, simt::memory_order_release);
                 done = true;
             }

             __sleep(sleep);
             if (sleep < 256) {
                 sleep *= 2;
             }
         }

        uint64_t end = __global_clock();
        times[id / warpSize * active_lane_count + offset] = end - start;
    }
}

__global__ void kernel_device_lock_cas_enclosing_variable_lanes(simt::atomic<uint32_t> *var, uint64_t *times, uint64_t *masks) {
    int lid = lane_id();
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t active_mask = masks[id / warpSize];

    if (active_mask & (1ull << lid)) {
        int active_lane_count = popcount(active_mask);
        int offset = mask_relative_lane_id(lid, active_mask);
        uint64_t start = __global_clock();

        uint32_t sleep = 8;
        bool done = false;
        uint32_t expected = false;
        while (done == false) {
            if (var->compare_exchange_strong(expected, true, simt::memory_order_acquire)) {
                var->store(false, simt::memory_order_release);
                done = true;
            }

            __sleep(sleep);
            if (sleep < 256) {
                sleep *= 2;
            }
        }

        uint64_t end = __global_clock();
        times[id / warpSize * active_lane_count + offset] = end - start;
    }
}

__global__ void kernel_atomic_add_variable_lanes(simt::atomic<uint32_t> *var, uint64_t *times, uint64_t *masks) {
    int lid = lane_id();
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t active_mask = masks[id / warpSize];

    if (active_mask & (1ull << lid)) {
        int active_lane_count = popcount(active_mask);
        int offset = mask_relative_lane_id(lid, active_mask);
        uint64_t start = __global_clock();

        var->fetch_add(1, simt::memory_order_acquire);
        var->fetch_add(1, simt::memory_order_release);

        uint64_t end = __global_clock();
        times[id / warpSize * active_lane_count + offset] = end - start;
    }
}

__global__ void kernel_atomic_load_store_variable_lanes(simt::atomic<uint32_t> *var, uint64_t *times, uint64_t *masks) {
    int lid = lane_id();
    uint32_t id = static_cast<uint32_t>(global_id());
    uint64_t active_mask = masks[id / warpSize];

    if (active_mask & (1ull << lid)) {
        int active_lane_count = popcount(active_mask);
        int offset = mask_relative_lane_id(lid, active_mask);
        uint64_t start = __global_clock();

        uint32_t ret = var->load(simt::memory_order_acquire);
        var->store(ret, simt::memory_order_release);

        uint64_t end = __global_clock();
        times[id / warpSize * active_lane_count + offset] = end - start;
    }
}

constexpr int MAX_GRID_SIZE = 1024;
constexpr int MAX_BLOCK_SIZE = 1024;

static int s_warp_size;
static Buffer *s_times = nullptr;
static Buffer *s_masks = nullptr;

template <typename AtomicType>
void *declare_atomic() {
    AtomicType *var = nullptr;
    HIP_CHECK(hipMalloc(&var, sizeof(AtomicType)));
    HIP_CHECK(hipMemset(const_cast<std::remove_cv_t<AtomicType> *>(var), 0, sizeof(AtomicType)));
    return const_cast<std::remove_cv_t<AtomicType> *>(var);
}

template <typename AtomicType, typename Kernel>
void register_basic_benchmark(
    std::string name,
    Kernel kernel,
    int grid_size,
    int block_size,
    int runs
) {
    // Truncate file beforehand
    char path[PATH_MAX + 1];
    sprintf(path, "/out/%s.bin", name.data());
    std::ofstream kernel_run_file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    kernel_run_file.close();

    register_benchmark(
        name,
        grid_size,
        block_size,
        runs,
        [](benchmark_t *bm){
            bm->user = declare_atomic<AtomicType>();
        },
        [kernel](benchmark_t *bm) {
            char path[PATH_MAX + 1];
            sprintf(path, "/out/%s.bin", bm->name.data());

            // Launch kernel
            hipLaunchKernelGGL(kernel, dim3(bm->grid_size), dim3(bm->block_size), 0, nullptr, static_cast<AtomicType *>(bm->user), static_cast<uint64_t *>(s_times->get_device_buffer()));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Append times to file
            s_times->copy(Target::Host);

            std::ofstream kernel_run_file(path, std::ios::out | std::ios::binary | std::ios::app);
            kernel_run_file.write(static_cast<char *>(s_times->get_host_buffer()), bm->grid_size * bm->block_size * sizeof(uint64_t));
            kernel_run_file.close();
        },
        [](benchmark_t *bm) {
            HIP_CHECK(hipFree(bm->user));
        }
    );
}

template <typename AtomicType, typename Kernel>
void register_lock_benchmark(
    std::string name,
    Kernel kernel,
    int grid_size,
    int block_size,
    int active_lanes,
    int runs
) {
    // Truncate file beforehand
    char path[PATH_MAX + 1];
    sprintf(path, "/out/%s.bin", name.data());
    std::ofstream kernel_run_file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    kernel_run_file.close();

    register_benchmark(
        name,
        grid_size,
        block_size,
        runs,
        [](benchmark_t *bm){
            bm->user = declare_atomic<AtomicType>();
        },
        [active_lanes, kernel](benchmark_t *bm) {
            char path[PATH_MAX + 1];
            sprintf(path, "/out/%s.bin", bm->name.data());

            // Generate random mask per warp
            uint64_t *masks_host = static_cast<uint64_t *>(s_masks->get_host_buffer());
            for (int i = 0; i < bm->grid_size * bm->block_size / s_warp_size; i++) {
                uint64_t mask = 0;

                while (popcount(mask) < active_lanes) {
                    mask |= (1ull << ((s_prand() - s_prand.min()) % s_warp_size));
                }

                masks_host[i] = mask;
            }
            s_masks->copy(Target::Device);

            // Launch kernel
            hipLaunchKernelGGL(kernel, dim3(bm->grid_size), dim3(bm->block_size), 0, nullptr, static_cast<AtomicType *>(bm->user), static_cast<uint64_t *>(s_times->get_device_buffer()), static_cast<uint64_t *>(s_masks->get_device_buffer()));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Append times to file
            s_times->copy(Target::Host);

            std::ofstream kernel_run_file(path, std::ios::out | std::ios::binary | std::ios::app);
            kernel_run_file.write(static_cast<char *>(s_times->get_host_buffer()), bm->grid_size * bm->block_size / s_warp_size * active_lanes * sizeof(uint64_t));
            kernel_run_file.close();
        },
        [](benchmark_t *bm) {
            HIP_CHECK(hipFree(bm->user));
        }
    );
}

int main(int argc, char *argv[]) {
#if defined (__HIP_PLATFORM_AMD__)
    __init_global_clock();
#endif

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    s_warp_size = props.warpSize;

    s_times = new Buffer(MAX_GRID_SIZE * MAX_BLOCK_SIZE * sizeof(uint64_t), Target::Both);
    HIP_CHECK(hipMemset(s_times->get_device_buffer(), 0, s_times->size()));

    s_masks = new Buffer(MAX_GRID_SIZE * MAX_BLOCK_SIZE / s_warp_size * sizeof(uint64_t), Target::Both);
    HIP_CHECK(hipMemset(s_masks->get_device_buffer(), 0, s_masks->size()));

    if (s_warp_size % 32 == 0) {
        char bm_name[PATH_MAX + 1];
        for (int lanes = 1; lanes <= s_warp_size; lanes++) {
            for (int grid_bits = 5; grid_bits <= 10; grid_bits++) {
                for (int block_bits = 5; block_bits <= 10; block_bits++) {
                    int grid = (1 << grid_bits);
                    int block = (1 << block_bits);

                    sprintf(bm_name, "bm_atomics_device_lock_or_enclosing_%d_lanes_%dx%d", lanes, grid, block);
                    register_lock_benchmark<simt::atomic<uint32_t>>(bm_name, kernel_device_lock_or_enclosing_variable_lanes, grid, block, lanes, 1);

                    sprintf(bm_name, "bm_atomics_device_lock_cas_enclosing_%d_lanes_%dx%d", lanes, grid, block);
                    register_lock_benchmark<simt::atomic<uint32_t>>(bm_name, kernel_device_lock_cas_enclosing_variable_lanes, grid, block, lanes, 1);

                    sprintf(bm_name, "bm_atomics_device_atomic_add_%d_lanes_%dx%d", lanes, grid, block);
                    register_lock_benchmark<simt::atomic<uint32_t>>(std::string(bm_name), kernel_atomic_add_variable_lanes, grid, block, lanes, 1);

                    sprintf(bm_name, "bm_atomics_device_atomic_load_store_%d_lanes_%dx%d", lanes, grid, block);
                    register_lock_benchmark<simt::atomic<uint32_t>>(std::string(bm_name), kernel_atomic_load_store_variable_lanes, grid, block, lanes, 1);
                }
            }
        }
    }

    initialize_benchmarks();
    run_benchmarks_randomized();
    finalize_benchmarks();

    s_benchmarks.clear();

    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_store_device_relaxed", kernel_device_store_atomic_relaxed, 1024, 1024, 1);
    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_store_device_seqcst", kernel_device_store_atomic_seqcst, 1024, 1024, 1);
    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_store_system_relaxed", kernel_system_store_atomic_relaxed, 1024, 1024, 1);
    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_store_system_seqcst", kernel_system_store_atomic_seqcst, 1024, 1024, 1);
    register_basic_benchmark<uint32_t>("bm_atomics_store_mmio", kernel_store_mmio, 1024, 1024, 1);
    register_basic_benchmark<volatile uint32_t>("bm_atomics_store_volatile", kernel_store_volatile, 1024, 1024, 1);

    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_device_lock_or_enclosing", kernel_device_lock_or_enclosing, 256, 256, 1);
    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_device_lock_cas_enclosing", kernel_device_lock_cas_enclosing, 256, 256, 1);
    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_system_lock_or_enclosing", kernel_system_lock_or_enclosing, 256, 256, 1);
    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_system_lock_cas_enclosing", kernel_system_lock_cas_enclosing, 256, 256, 1);

    initialize_benchmarks();
    run_benchmarks_linear();
    finalize_benchmarks();

    s_benchmarks.clear();

    delete s_times;
    s_times = nullptr;
}
