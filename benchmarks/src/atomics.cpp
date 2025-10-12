#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
#include <vector>
#include <getopt.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "buffer.hpp"
#include "utility.hpp"
#include "atomic.hpp"
#include "benchmarking.hpp"

enum class VariableSpinlockModes {
    ATOMIC_OR,
    ATOMIC_CAS,
    ATOMIC_ADD,
    ATOMIC_LOAD_STORE
};

enum class StoreModes {
    DEVICE_RELAXED,
    DEVICE_SEQ_CST,
    SYSTEM_RELAXED,
    SYSTEM_SEQ_CST,
    MMIO,
    VOLATILE
};

enum class FixedSpinlockModes {
    DEVICE_OR,
    DEVICE_CAS,
    SYSTEM_OR,
    SYSTEM_CAS
};

static std::random_device s_rand{};
static std::default_random_engine s_prand{s_rand()};
static std::vector<bm::benchmark_t> s_benchmarks{};

// Benchmarking config
static int s_variable_spinlock_runs{8};
static std::vector<VariableSpinlockModes> s_variable_spinlock_modes{
    VariableSpinlockModes::ATOMIC_OR,
    VariableSpinlockModes::ATOMIC_CAS,
    VariableSpinlockModes::ATOMIC_ADD,
    VariableSpinlockModes::ATOMIC_LOAD_STORE
};
static int s_store_runs{32};
static std::vector<StoreModes> s_store_modes{
    StoreModes::DEVICE_RELAXED,
    StoreModes::DEVICE_SEQ_CST,
    StoreModes::SYSTEM_RELAXED,
    StoreModes::SYSTEM_SEQ_CST,
    StoreModes::MMIO,
    StoreModes::VOLATILE
};
static int s_fixed_spinlock_runs{32};
static std::vector<FixedSpinlockModes> s_fixed_spinlock_modes{
    FixedSpinlockModes::DEVICE_OR,
    FixedSpinlockModes::DEVICE_CAS,
    FixedSpinlockModes::SYSTEM_OR,
    FixedSpinlockModes::SYSTEM_CAS
};
static bool s_do_variable_spinlock_benchmarks{true};
static bool s_do_store_benchmarks{true};
static bool s_do_fixed_spinlock_benchmarks{true};

__global__ void kernel_device_lock_or(simt::atomic<uint32_t> *var, uint64_t *times) {
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

__global__ void kernel_device_lock_cas(simt::atomic<uint32_t> *var, uint64_t *times) {
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

__global__ void kernel_system_lock_or(simt::atomic<uint32_t, simt::thread_scope_system> *var, uint64_t *times) {
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

__global__ void kernel_system_lock_cas(simt::atomic<uint32_t, simt::thread_scope_system> *var, uint64_t *times) {
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

__global__ void kernel_device_lock_or_variable_lanes(simt::atomic<uint32_t> *var, uint64_t *times, uint64_t *masks) {
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

__global__ void kernel_device_lock_cas_variable_lanes(simt::atomic<uint32_t> *var, uint64_t *times, uint64_t *masks) {
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
    s_benchmarks.push_back({
        .name = name,
        .grid_size = grid_size,
        .block_size = block_size,
        .current_runs = 0,
        .requested_runs = runs,
        .initialize = [](bm::benchmark_t *bm){
            try_delete_result_file(*bm);
            bm->user = declare_atomic<AtomicType>();
        },
        .run = [kernel](bm::benchmark_t *bm) {
            hipLaunchKernelGGL(kernel, dim3(bm->grid_size), dim3(bm->block_size), 0, nullptr, static_cast<AtomicType *>(bm->user), static_cast<uint64_t *>(s_times->get_device_buffer()));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            s_times->copy(Target::Host);

            uint32_t total_threads = bm->grid_size * bm->block_size;
            bm::append_binary_result_file(*bm, static_cast<char *>(s_times->get_host_buffer()), total_threads * sizeof(uint64_t));
        },
        .finalize = [](bm::benchmark_t *bm) {
            HIP_CHECK(hipFree(bm->user));
        },
        .user = nullptr,
    });
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
    s_benchmarks.push_back({
        .name = name,
        .grid_size = grid_size,
        .block_size = block_size,
        .current_runs = 0,
        .requested_runs = runs,
        .initialize = [](bm::benchmark_t *bm){
            try_delete_result_file(*bm);
            bm->user = declare_atomic<AtomicType>();
        },
        .run = [active_lanes, kernel](bm::benchmark_t *bm) {
            uint64_t *masks_host = static_cast<uint64_t *>(s_masks->get_host_buffer());
            for (int i = 0; i < bm->grid_size * bm->block_size / s_warp_size; i++) {
                uint64_t mask = 0;

                while (popcount(mask) < active_lanes) {
                    mask |= (1ull << ((s_prand() - s_prand.min()) % s_warp_size));
                }

                masks_host[i] = mask;
            }
            s_masks->copy(Target::Device);

            hipLaunchKernelGGL(kernel, dim3(bm->grid_size), dim3(bm->block_size), 0, nullptr, static_cast<AtomicType *>(bm->user), static_cast<uint64_t *>(s_times->get_device_buffer()), static_cast<uint64_t *>(s_masks->get_device_buffer()));
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            s_times->copy(Target::Host);

            uint32_t total_threads = bm->grid_size * bm->block_size / s_warp_size * active_lanes;
            bm::append_binary_result_file(*bm, static_cast<char *>(s_times->get_host_buffer()), total_threads * sizeof(uint64_t));
        },
        .finalize = [](bm::benchmark_t *bm) {
            HIP_CHECK(hipFree(bm->user));
        },
        .user = nullptr,
    });
}

static void parse_options(int argc, char *argv[]) {
    static const struct option long_options[] = {
        {"variable-spinlock-runs", required_argument, nullptr, 0},
        {"variable-spinlock-modes", required_argument, nullptr, 0},
        {"store-runs", required_argument, nullptr, 0},
        {"store-modes", required_argument, nullptr, 0},
        {"fixed-spinlock-runs", required_argument, nullptr, 0},
        {"fixed-spinlock-modes", required_argument, nullptr, 0},
        {"no-variable-spinlock", no_argument, nullptr, 0},
        {"no-store", no_argument, nullptr, 0},
        {"no-fixed-spinlock", no_argument, nullptr, 0},
    };

    static const char *variable_spinlock_modes[] = {
        "or", "cas", "add", "load_store"
    };

    static const char *store_modes[] = {
        "device_relaxed", "device_seq_cst",
        "system_relaxed", "system_seq_cst",
        "mmio", "volatile"
    };

    static const char *fixed_spinlock_modes[] = {
        "device_or", "device_cas",
        "system_or", "system_cas"
    };

    int option_index = 0;
    int ret = 0;

    while ((ret = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        if (ret == '?') {
            continue;
        }

        switch (option_index) {
            case 0:
                s_variable_spinlock_runs = atoi(optarg);
                if (s_variable_spinlock_runs <= 0) {
                    std::cerr << "Invalid number of runs for variable spinlock benchmarks: " << optarg << std::endl;
                    abort();
                }
                break;
            case 1:
                parse_list<VariableSpinlockModes>(variable_spinlock_modes, sizeof(variable_spinlock_modes) / sizeof(variable_spinlock_modes[0]), s_variable_spinlock_modes);
                break;
            case 2:
                s_store_runs = atoi(optarg);
                if (s_store_runs <= 0) {
                    std::cerr << "Invalid number of runs for atomic store benchmarks: " << optarg << std::endl;
                    abort();
                }
                break;
            case 3:
                parse_list<StoreModes>(store_modes, sizeof(store_modes) / sizeof(store_modes[0]), s_store_modes);
                break;
            case 4:
                s_fixed_spinlock_runs = atoi(optarg);
                if (s_fixed_spinlock_runs <= 0) {
                    std::cerr << "Invalid number of runs for fixed spinlock benchmarks: " << optarg << std::endl;
                    abort();
                }
                break;
            case 5:
                parse_list<FixedSpinlockModes>(fixed_spinlock_modes, sizeof(fixed_spinlock_modes) / sizeof(fixed_spinlock_modes[0]), s_fixed_spinlock_modes);
                break;
            case 6:
                s_do_variable_spinlock_benchmarks = false;
                break;
            case 7:
                s_do_store_benchmarks = false;
                break;
            case 8:
                s_do_fixed_spinlock_benchmarks = false;
                break;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "Variable spinlock benchmark runs: " << s_variable_spinlock_runs << std::endl;
    std::cout << "Variable spinlock benchmark modes: ";
    for (auto mode : s_variable_spinlock_modes) {
        std::cout << variable_spinlock_modes[static_cast<int>(mode)] << " ";
    }
    std::cout << std::endl;

    std::cout << "Atomic store benchmark Runs: " << s_store_runs << std::endl;
    std::cout << "Atomic store benchmark Modes: ";
    for (auto mode : s_store_modes) {
        std::cout << store_modes[static_cast<int>(mode)] << " ";
    }
    std::cout << std::endl;

    std::cout << "Fixed spinlock benchmark runs: " << s_fixed_spinlock_runs << std::endl;
    std::cout << "Fixed spinlock benchmark modes: ";
    for (auto mode : s_fixed_spinlock_modes) {
        std::cout << fixed_spinlock_modes[static_cast<int>(mode)] << " ";
    }
    std::cout << std::endl;

    std::cout << "Variable spinlock benchmarks: " << (s_do_variable_spinlock_benchmarks ? "Yes" : "No") << std::endl;
    std::cout << "Atomic store benchmarks: " << (s_do_store_benchmarks ? "Yes" : "No") << std::endl;
    std::cout << "Fixed spinlock benchmarks: " << (s_do_fixed_spinlock_benchmarks ? "Yes" : "No") << std::endl;
};

int main(int argc, char *argv[]) {
    parse_options(argc, argv);

    __init_global_clock();

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    s_warp_size = props.warpSize;
    assert(s_warp_size == 32);

    s_times = new Buffer(MAX_GRID_SIZE * MAX_BLOCK_SIZE * sizeof(uint64_t), Target::Both);
    HIP_CHECK(hipMemset(s_times->get_device_buffer(), 0, s_times->size()));

    s_masks = new Buffer(MAX_GRID_SIZE * MAX_BLOCK_SIZE / s_warp_size * sizeof(uint64_t), Target::Both);
    HIP_CHECK(hipMemset(s_masks->get_device_buffer(), 0, s_masks->size()));

    if (s_do_variable_spinlock_benchmarks) {
        char bm_name[512];
        for (int lanes = 1; lanes <= s_warp_size; lanes++) {
            for (int grid_bits = 5; grid_bits <= 10; grid_bits++) {
                for (int block_bits = 5; block_bits <= 10; block_bits++) {
                    int grid = (1 << grid_bits);
                    int block = (1 << block_bits);

                    for (auto mode : s_variable_spinlock_modes) {
                        switch (mode) {
                            case VariableSpinlockModes::ATOMIC_OR:
                                snprintf(bm_name, 512, "bm_atomics_device_lock_or_%d_lanes_%dx%d", lanes, grid, block);
                                register_lock_benchmark<simt::atomic<uint32_t>>(bm_name, kernel_device_lock_or_variable_lanes, grid, block, lanes, s_variable_spinlock_runs);
                                break;
                            case VariableSpinlockModes::ATOMIC_CAS:
                                snprintf(bm_name, 512, "bm_atomics_device_lock_cas_%d_lanes_%dx%d", lanes, grid, block);
                                register_lock_benchmark<simt::atomic<uint32_t>>(bm_name, kernel_device_lock_cas_variable_lanes, grid, block, lanes, s_variable_spinlock_runs);
                                break;
                            case VariableSpinlockModes::ATOMIC_ADD:
                                snprintf(bm_name, 512, "bm_atomics_device_atomic_add_%d_lanes_%dx%d", lanes, grid, block);
                                register_lock_benchmark<simt::atomic<uint32_t>>(std::string(bm_name), kernel_atomic_add_variable_lanes, grid, block, lanes, s_variable_spinlock_runs);
                                break;
                            case VariableSpinlockModes::ATOMIC_LOAD_STORE:
                                snprintf(bm_name, 512, "bm_atomics_device_atomic_load_store_%d_lanes_%dx%d", lanes, grid, block);
                                register_lock_benchmark<simt::atomic<uint32_t>>(std::string(bm_name), kernel_atomic_load_store_variable_lanes, grid, block, lanes, s_variable_spinlock_runs);
                                break;
                        }
                    }
                }
            }
        }
    }

    bm::run_randomized(s_benchmarks);
    s_benchmarks.clear();

    if (s_do_store_benchmarks) {
        for (auto mode : s_store_modes) {
            switch (mode) {
                case StoreModes::DEVICE_RELAXED:
                    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_store_device_relaxed", kernel_device_store_atomic_relaxed, 1024, 1024, s_store_runs);
                    break;
                case StoreModes::DEVICE_SEQ_CST:
                    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_store_device_seqcst", kernel_device_store_atomic_seqcst, 1024, 1024, s_store_runs);
                    break;
                case StoreModes::SYSTEM_RELAXED:
                    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_store_system_relaxed", kernel_system_store_atomic_relaxed, 1024, 1024, s_store_runs);
                    break;
                case StoreModes::SYSTEM_SEQ_CST:
                    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_store_system_seqcst", kernel_system_store_atomic_seqcst, 1024, 1024, s_store_runs);
                    break;
                case StoreModes::MMIO:
                    register_basic_benchmark<uint32_t>("bm_atomics_store_mmio", kernel_store_mmio, 1024, 1024, s_store_runs);
                    break;
                case StoreModes::VOLATILE:
                    register_basic_benchmark<volatile uint32_t>("bm_atomics_store_volatile", kernel_store_volatile, 1024, 1024, s_store_runs);
                    break;
            }
        }
    }

    if (s_do_fixed_spinlock_benchmarks) {
        for (auto mode : s_fixed_spinlock_modes) {
            switch (mode) {
                case FixedSpinlockModes::DEVICE_OR:
                    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_device_lock_or", kernel_device_lock_or, 256, 256, s_fixed_spinlock_runs);
                    break;
                case FixedSpinlockModes::DEVICE_CAS:
                    register_basic_benchmark<simt::atomic<uint32_t>>("bm_atomics_device_lock_cas", kernel_device_lock_cas, 256, 256, s_fixed_spinlock_runs);
                    break;
                case FixedSpinlockModes::SYSTEM_OR:
                    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_system_lock_or", kernel_system_lock_or, 256, 256, s_fixed_spinlock_runs);
                    break;
                case FixedSpinlockModes::SYSTEM_CAS:
                    register_basic_benchmark<simt::atomic<uint32_t, simt::thread_scope_system>>("bm_atomics_system_lock_cas", kernel_system_lock_cas, 256, 256, s_fixed_spinlock_runs);
                    break;
            }
        }
    }

    bm::run_linear(s_benchmarks);
    s_benchmarks.clear();

    delete s_times;
    delete s_masks;
}
