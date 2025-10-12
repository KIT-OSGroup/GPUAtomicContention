#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <getopt.h>
#include <set>

#include "utility.hpp"
#include "atomic.hpp"
#include "buffer.hpp"
#include "benchmarking.hpp"

constexpr int MaxBlockSizeSmall = 512;
constexpr int MaxGridSizeSmall = 128;

constexpr int MaxBlockSizeLarge = 1024;
constexpr int MaxGridSizeLarge = 1024;

constexpr int MinLaneBits = 0;
constexpr int MaxLaneBits = 5;
constexpr int MaxLanes = 1 << MaxLaneBits;

constexpr int MinWarpBits = 0;
constexpr int MaxWarpBitsLarge = 15;
constexpr int MaxWarpBitsSmall = 11;
constexpr int MaxWarpsLarge = 1 << MaxWarpBitsLarge;
constexpr int MaxWarpsSmall = 1 << MaxWarpBitsSmall;

constexpr int MinGroupBits = 0;
constexpr int MaxGroupBitsLarge = 14;
constexpr int MaxGroupBitsSmall = 11;
constexpr int MaxGroupsLarge = 1 << MaxGroupBitsLarge;
constexpr int MaxGroupsSmall = 1 << MaxGroupBitsSmall;

constexpr int MinStrideBits = 2;
constexpr int MaxStrideBits = 14;
constexpr int MaxStride = 1 << MaxStrideBits;

constexpr int MinStrideOffset = -32;
constexpr int MaxStrideOffset = 32;
constexpr int StrideOffsetStepSize = 4;

constexpr uint32_t AtomicAddMinOperations = 32;
constexpr uint32_t SpinlockMinOperations = 1;
constexpr uint32_t BaselineNumOperationsShort = 128;
constexpr uint32_t BaselineNumOperationsLong = 1024;

static const char *ModeStrings[] = {
    "add_acquire", "add_seq_cst", "add_relaxed",
    "add_acquire_sync", "add_seq_cst_sync", "add_relaxed_sync",
    "spinlock"
};

static const char *ScopeStrings[] = {
    "block", "device", "system"
};

enum class Mode {
    ATOMIC_ADD_SEQ_CST,
    ATOMIC_ADD_ACQUIRE,
    ATOMIC_ADD_RELAXED,
    ATOMIC_ADD_SEQ_CST_SYNC,
    ATOMIC_ADD_ACQUIRE_SYNC,
    ATOMIC_ADD_RELAXED_SYNC,
    SPINLOCK_MUTEX
};

enum class Scope {
    BLOCK = simt::thread_scope_block,
    DEVICE = simt::thread_scope_device,
    SYSTEM = simt::thread_scope_system
};

enum class Grid {
    LARGE,
    SMALL
};

enum class Transpose {
    DIRECT,
    SCATTERED
};

static uint32_t s_warp_size{};
static std::vector<bm::benchmark_t> s_benchmarks{};
static Buffer *s_atomic_buffer{};
static Buffer *s_mask_buffer{};
static Buffer *s_times_buffer{};
static hipEvent_t s_completion_event{};
static std::random_device s_rand{};
static std::default_random_engine s_prand{s_rand()};

// Benchmarking config
static std::vector<Mode> s_modes{
    Mode::ATOMIC_ADD_SEQ_CST,
    Mode::ATOMIC_ADD_ACQUIRE,
    Mode::ATOMIC_ADD_RELAXED,
    Mode::ATOMIC_ADD_SEQ_CST_SYNC,
    Mode::ATOMIC_ADD_ACQUIRE_SYNC,
    Mode::ATOMIC_ADD_RELAXED_SYNC,
    Mode::SPINLOCK_MUTEX
};
static std::vector<Scope> s_scopes{Scope::BLOCK, Scope::DEVICE, Scope::SYSTEM};
static std::vector<Grid> s_grids{Grid::LARGE, Grid::SMALL};
static std::vector<Transpose> s_transpose_modes{Transpose::DIRECT, Transpose::SCATTERED};
static int s_benchmark_runs{10};
static int s_baseline_runs{1024};
static bool s_do_baseline_benchmarks{true};
static bool s_do_varying_threads_benchmarks{true};
static bool s_do_varying_strides_benchmarks{true};
static bool s_do_offsetted_strides_benchmarks{true};

// MurmurHash3 - finalization mix
__device__ uint32_t hash(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

__always_inline constexpr bool is_power_of_two(std::unsigned_integral auto val) {
    return val == 0 || (val & (val - 1)) == 0;
}

template <simt::thread_scope Scope, simt::memory_order Order, bool Synchronize>
__attribute__((always_inline)) __device__ inline uint64_t atomic_add(simt::atomic<uint32_t, Scope> *atomic, bool active, int count) {
    uint64_t start = __global_clock();

    while (count--) {
        if (active) {
            atomic->fetch_add(1, Order);
        }

        if constexpr (Synchronize) {
            __syncthreads();
        }
    }

    uint64_t end = __global_clock();
    return end - start;
}

template <simt::thread_scope Scope>
__attribute__((always_inline)) __device__ inline uint64_t spinlock_mutex(simt::atomic<uint32_t, Scope> *atomic, bool active, int count) {
    uint32_t gid = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();

    while (count--) {
        if (active) {
            uint32_t sleep = 128;
            bool done = false;
            while (done == false) {
                if (atomic->fetch_or(true, simt::memory_order_acquire) == false) {
                    atomic->store(false, simt::memory_order_release);
                    done = true;
                }

                __sleep(sleep + hash(gid + count) & 0x1ff);
                if (sleep < 1024) {
                    sleep *= 2;
                }
            }
        }
    }

    uint64_t end = __global_clock();
    return end - start;
}

__device__ uint64_t dispatch_atomic(char *atomic, bool active, int count, Mode mode, simt::thread_scope scope) {
#define DISPATCH_WITH_MEMORDER(memorder, sync)                                                                                                                              \
    do {                                                                                                                                                                    \
        switch (scope) {                                                                                                                                                    \
            case simt::thread_scope_block:                                                                                                                                  \
                return atomic_add<simt::thread_scope_block, memorder, sync>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_block> *>(atomic), active, count);   \
            case simt::thread_scope_device:                                                                                                                                 \
                return atomic_add<simt::thread_scope_device, memorder, sync>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_device> *>(atomic), active, count); \
            case simt::thread_scope_system:                                                                                                                                 \
                return atomic_add<simt::thread_scope_system, memorder, sync>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_system> *>(atomic), active, count); \
        }                                                                                                                                                                   \
    } while (0)

    switch (mode) {
        case Mode::ATOMIC_ADD_SEQ_CST:
            DISPATCH_WITH_MEMORDER(simt::memory_order_seq_cst, false);
        case Mode::ATOMIC_ADD_ACQUIRE:
            DISPATCH_WITH_MEMORDER(simt::memory_order_acquire, false);
        case Mode::ATOMIC_ADD_RELAXED:
            DISPATCH_WITH_MEMORDER(simt::memory_order_relaxed, false);
        case Mode::ATOMIC_ADD_SEQ_CST_SYNC:
            DISPATCH_WITH_MEMORDER(simt::memory_order_seq_cst, true);
        case Mode::ATOMIC_ADD_ACQUIRE_SYNC:
            DISPATCH_WITH_MEMORDER(simt::memory_order_acquire, true);
        case Mode::ATOMIC_ADD_RELAXED_SYNC:
            DISPATCH_WITH_MEMORDER(simt::memory_order_relaxed, true);
        case Mode::SPINLOCK_MUTEX:
            switch (scope) {
                case simt::thread_scope_block:
                    return spinlock_mutex<simt::thread_scope_block>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_block> *>(atomic), active, count);
                case simt::thread_scope_device:
                    return spinlock_mutex<simt::thread_scope_device>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_device> *>(atomic), active, count);
                case simt::thread_scope_system:
                    return spinlock_mutex<simt::thread_scope_system>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_system> *>(atomic), active, count);
            }
    }
#undef DISPATCH_WITH_MEMORDER
}

template <simt::thread_scope Scope, uint32_t NumOperations, simt::memory_order Order, bool Synchronize>
__device__ uint64_t baseline_atomic_add(simt::atomic<uint32_t, Scope> *atomic) {
    uint64_t start = __global_clock();

    #pragma unroll
    for (int i = 0; i < NumOperations; i++) {
        atomic->fetch_add(1, Order);

        if constexpr (Synchronize) {
            __syncthreads();
        }
    }

    uint64_t end = __global_clock();
    return end - start;
}

template <simt::thread_scope Scope, uint32_t NumOperations>
__device__ uint64_t baseline_spinlock_mutex(simt::atomic<uint32_t, Scope> *atomic) {
    uint32_t gid = static_cast<uint32_t>(global_id());
    uint64_t start = __global_clock();

    #pragma unroll
    for (int i = 0; i < NumOperations; i++) {
        uint32_t sleep = 128;
        bool done = false;
        while (done == false) {
            if (atomic->fetch_or(true, simt::memory_order_acquire) == false) {
                atomic->store(false, simt::memory_order_release);
                done = true;
            }

            __sleep(sleep + hash(gid + i) & 0x1ff);
            if (sleep < 1024) {
                sleep *= 2;
            }
        }
    }

    uint64_t end = __global_clock();
    return end - start;
}

template <uint32_t NumOperations>
__device__ uint64_t dispatch_baseline_atomic(char *atomic, Mode mode, simt::thread_scope scope) {
#define DISPATCH_WITH_MEMORDER(memorder, sync)                                                                                                                                       \
    do {                                                                                                                                                                             \
        switch (scope) {                                                                                                                                                             \
            case simt::thread_scope_block:                                                                                                                                           \
                return baseline_atomic_add<simt::thread_scope_block, NumOperations, memorder, sync>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_block> *>(atomic));   \
            case simt::thread_scope_device:                                                                                                                                          \
                return baseline_atomic_add<simt::thread_scope_device, NumOperations, memorder, sync>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_device> *>(atomic)); \
            case simt::thread_scope_system:                                                                                                                                          \
                return baseline_atomic_add<simt::thread_scope_system, NumOperations, memorder, sync>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_system> *>(atomic)); \
        }                                                                                                                                                                            \
    } while (0)

    switch (mode) {
        case Mode::ATOMIC_ADD_SEQ_CST:
            DISPATCH_WITH_MEMORDER(simt::memory_order_seq_cst, false);
        case Mode::ATOMIC_ADD_ACQUIRE:
            DISPATCH_WITH_MEMORDER(simt::memory_order_acquire, false);
        case Mode::ATOMIC_ADD_RELAXED:
            DISPATCH_WITH_MEMORDER(simt::memory_order_relaxed, false);
        case Mode::ATOMIC_ADD_SEQ_CST_SYNC:
            DISPATCH_WITH_MEMORDER(simt::memory_order_seq_cst, true);
        case Mode::ATOMIC_ADD_ACQUIRE_SYNC:
            DISPATCH_WITH_MEMORDER(simt::memory_order_acquire, true);
        case Mode::ATOMIC_ADD_RELAXED_SYNC:
            DISPATCH_WITH_MEMORDER(simt::memory_order_relaxed, true);
        case Mode::SPINLOCK_MUTEX:
            switch (scope) {
                case simt::thread_scope_block:
                    return baseline_spinlock_mutex<simt::thread_scope_block, NumOperations>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_block> *>(atomic));
                case simt::thread_scope_device:
                    return baseline_spinlock_mutex<simt::thread_scope_device, NumOperations>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_device> *>(atomic));
                case simt::thread_scope_system:
                    return baseline_spinlock_mutex<simt::thread_scope_system, NumOperations>(reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_system> *>(atomic));
            }  
    }
#undef DISPATCH_WITH_MEMORDER
}

__global__ void baseline_kernel(
    uint64_t *times,
    char *atomic,
    Mode mode,
    simt::thread_scope scope,
    bool do_long_benchmark
) {
    if (global_id() == 0) {
        if (do_long_benchmark) {
            times[0] = dispatch_baseline_atomic<BaselineNumOperationsLong>(atomic, mode, scope);
        } else {
            times[0] = dispatch_baseline_atomic<BaselineNumOperationsShort>(atomic, mode, scope);
        }
    }
}

__global__ void kernel(
    uint64_t *times,
    char *atomics,
    int mem_stride,
    int groups,
    int active_warps,
    bool transpose,
    uint64_t *active_masks,
    uint32_t target_value,
    Mode mode,
    simt::thread_scope scope
) {
    int lid = lane_id();
    uint32_t gid = static_cast<uint32_t>(global_id());
    uint64_t active_mask = active_masks[gid / warpSize];
    int active_lanes = popcount(active_mask);
    int offset = mask_relative_lane_id(lid, active_mask);
    uint32_t total_threads = active_warps * active_lanes;
    int linearized_gid = gid / warpSize * active_lanes + offset;

    uint32_t group_size = total_threads / groups;
    uint32_t group = 0;

    if (transpose) {
        group = linearized_gid % groups;
    } else {
        group = linearized_gid / group_size;
    }

    auto *atomic = atomics + mem_stride * group;

    uint32_t local_target = target_value / total_threads;

    bool active = (active_mask & (1ull << lid)) && gid < active_warps * warpSize;

    // NOTE: to ensure equal work, i.e., equal number of atomicAdd(1) calls, the following should hold
    assert(active_lanes == 1 || active_lanes % 2 == 0);
    assert(total_threads % groups == 0);
    assert(target_value % total_threads == 0);

    times[linearized_gid] = dispatch_atomic(atomic, active, local_target, mode, scope);
}

void add_baseline_benchmarks(bool do_long_benchmark) {
    char name[512];

    for (auto mode : s_modes) {
        const char *mode_name = ModeStrings[static_cast<int>(mode)];

        for (auto scope : s_scopes) {
            simt::thread_scope thread_scope = static_cast<simt::thread_scope>(scope);
            const char *scope_name = ScopeStrings[static_cast<int>(scope)];

            snprintf(name, 512, "bm_contention_baseline_%s_%s_%s", mode_name, scope_name, (do_long_benchmark ? "long" : "short"));

            s_benchmarks.push_back({
                .name = name,
                .grid_size = 0,
                .block_size = 0,
                .current_runs = 0,
                .requested_runs = s_baseline_runs,
                .initialize = [](bm::benchmark_t *bm) { try_delete_result_file(*bm); },
                .run = [mode, thread_scope, do_long_benchmark](bm::benchmark_t *bm) {
                    HIP_CHECK(hipMemset(s_times_buffer->get_device_buffer(), 0, s_times_buffer->size()));
                    HIP_CHECK(hipMemset(s_atomic_buffer->get_device_buffer(), 0, s_atomic_buffer->size()));

                    hipLaunchKernelGGL(baseline_kernel, dim3(1), dim3(1), 0, nullptr,
                        static_cast<uint64_t *>(s_times_buffer->get_device_buffer()),
                        static_cast<char *>(s_atomic_buffer->get_device_buffer()),
                        mode,
                        thread_scope,
                        do_long_benchmark
                    );
                    HIP_CHECK(hipGetLastError());
                    HIP_CHECK(hipDeviceSynchronize());

                    uint32_t num_operations = do_long_benchmark ? BaselineNumOperationsLong : BaselineNumOperationsShort;
                    if (mode != Mode::SPINLOCK_MUTEX) {
                        s_atomic_buffer->copy(Target::Host);
                        uint32_t value = *reinterpret_cast<uint32_t *>(s_atomic_buffer->get_host_buffer());
                        if (value != num_operations) {
                            std::cout << "Invalid value. Got: " << value << "   Expected:" << num_operations << std::endl;
                            abort();
                        }
                    }

                    s_times_buffer->copy(Target::Host);
                    bm::append_binary_result_file(*bm, static_cast<char *>(s_times_buffer->get_host_buffer()), sizeof(uint64_t));
                },
                .finalize = [](bm::benchmark_t *bm){},
                .user = nullptr,
            });
        }
    }
}

static void add_benchmark(
    int runs,
    int active_lanes,
    int active_warps,
    int groups,
    int mem_stride,
    Mode mode,
    Scope scope,
    bool transpose,
    bool small_grid
) {
    // NOTE: We still add all benchmark configs to the list. Configs for the
    //       larger grid will need to be skipped for the smaller grid.
    if (small_grid && (active_warps > MaxWarpsSmall || groups > MaxGroupsSmall)) {
        return;
    }

    char name[512];

    int grid_size = small_grid ? MaxGridSizeSmall : MaxGridSizeLarge;
    int block_size = small_grid ? MaxBlockSizeSmall : MaxBlockSizeLarge;
    uint32_t target_value = grid_size * block_size;

    if (mode == Mode::SPINLOCK_MUTEX) {
        target_value *= SpinlockMinOperations;
    } else {
        target_value *= AtomicAddMinOperations;
    }

    snprintf(name, 512, "bm_contention_%s_%s_t_%d_%d_m_%d_%d%s%s",
        ModeStrings[static_cast<int>(mode)],
        ScopeStrings[static_cast<int>(scope)],
        active_lanes,
        active_warps,
        groups,
        mem_stride,
        (transpose ? "_transposed" : ""),
        (small_grid ? "_small_grid" : "")
    );

    simt::thread_scope thread_scope = static_cast<simt::thread_scope>(scope);

    s_benchmarks.push_back({
        .name = name,
        .grid_size = grid_size,
        .block_size = block_size,
        .current_runs = 0,
        .requested_runs = runs,
        .initialize = [](bm::benchmark_t *bm){ try_delete_result_file(*bm); },
        .run = [active_lanes, active_warps, groups, mem_stride, transpose, target_value, mode, thread_scope](bm::benchmark_t *bm) {
            HIP_CHECK(hipMemset(s_times_buffer->get_device_buffer(), 0, s_times_buffer->size()));
            HIP_CHECK(hipMemset(s_atomic_buffer->get_device_buffer(), 0, s_atomic_buffer->size()));

            uint64_t *masks_host = static_cast<uint64_t *>(s_mask_buffer->get_host_buffer());
            for (int i = 0; i < bm->grid_size * bm->block_size / s_warp_size; i++) {
                uint64_t mask = 0;

                while (popcount(mask) < active_lanes) {
                    mask |= (1ull << ((s_prand() - s_prand.min()) % s_warp_size));
                }

                masks_host[i] = mask;
            }
            s_mask_buffer->copy(Target::Device);

            hipLaunchKernelGGL(kernel, dim3(bm->grid_size), dim3(bm->block_size), 0, nullptr,
                static_cast<uint64_t *>(s_times_buffer->get_device_buffer()),
                static_cast<char *>(s_atomic_buffer->get_device_buffer()),
                mem_stride,
                groups,
                active_warps,
                transpose,
                static_cast<uint64_t *>(s_mask_buffer->get_device_buffer()),
                target_value,
                mode,
                thread_scope
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            if (mode != Mode::SPINLOCK_MUTEX) {
                s_atomic_buffer->copy(Target::Host);
                char *atomics = static_cast<char *>(s_atomic_buffer->get_host_buffer());

                uint32_t total = 0;
                for (int i = 0; i < groups; i++) {
                    total += *reinterpret_cast<uint32_t *>(atomics + i * mem_stride);

                    if (mem_stride == 0) {
                        // All threads act on the same atomic so we only need to add once
                        break;
                    }
                }

                if (total != target_value) {
                    std::cout << "Invalid value. Got: " << total << "   Expected:" << target_value << std::endl;
                    abort();
                }
            }

            uint32_t total_threads = active_warps * active_lanes;
            s_times_buffer->copy(Target::Host);
            bm::append_binary_result_file(*bm, static_cast<char *>(s_times_buffer->get_host_buffer()), total_threads * sizeof(uint64_t));
        },
        .finalize = [](bm::benchmark_t *bm){},
        .user = nullptr,
    });
}

void add_benchmarks(int active_lanes, int active_warps, int groups, int mem_stride) {
    for (auto mode : s_modes) {
        for (auto scope : s_scopes) {
            for (auto grid : s_grids) {
                for (auto transpose_mode : s_transpose_modes) {
                    add_benchmark(
                        s_benchmark_runs,
                        active_lanes,
                        active_warps,
                        groups,
                        mem_stride,
                        mode,
                        scope,
                        transpose_mode == Transpose::SCATTERED,
                        grid == Grid::SMALL
                    );
                }
            }
        }
    }
}

static void parse_options(int argc, char *argv[]) {
    static const struct option long_options[] = {
        {"modes", required_argument, nullptr, 0},
        {"scopes", required_argument, nullptr, 0},
        {"grids", required_argument, nullptr, 0},
        {"transpose", required_argument, nullptr, 0},
        {"runs", required_argument, nullptr, 0},
        {"baseline-runs", required_argument, nullptr, 0},
        {"no-baseline", no_argument, nullptr, 0},
        {"no-varying-threads", no_argument, nullptr, 0},
        {"no-varying-strides", no_argument, nullptr, 0},
        {"no-offsetted-strides", no_argument, nullptr, 0},
    };

    static const char *grids[] = {
        "large", "small"
    };

    static const char *transpose_modes[] = {
        "direct", "scattered"
    };

    int option_index = 0;
    int ret = 0;

    while ((ret = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        if (ret == '?') {
            continue;
        }

        switch (option_index) {
            case 0:
                parse_list<Mode>(ModeStrings, sizeof(ModeStrings) / sizeof(ModeStrings[0]), s_modes);
                break;
            case 1:
                parse_list<Scope>(ScopeStrings, sizeof(ScopeStrings) / sizeof(ScopeStrings[0]), s_scopes);
                break;
            case 2:
                parse_list<Grid>(grids, sizeof(grids) / sizeof(grids[0]), s_grids);
                break;
            case 3:
                parse_list<Transpose>(transpose_modes, sizeof(transpose_modes) / sizeof(transpose_modes[0]), s_transpose_modes);
                break;
            case 4:
                s_benchmark_runs = atoi(optarg);
                if (s_benchmark_runs <= 0) {
                    std::cerr << "Invalid number of runs: " << optarg << std::endl;
                    abort();
                }
                break;
            case 5:
                s_baseline_runs = atoi(optarg);
                if (s_baseline_runs <= 0) {
                    std::cerr << "Invalid number of baseline runs: " << optarg << std::endl;
                    abort();
                }
                break;
            case 6:
                s_do_baseline_benchmarks = false;
                break;
            case 7:
                s_do_varying_threads_benchmarks = false;
                break;
            case 8:
                s_do_varying_strides_benchmarks = false;
                break;
            case 9:
                s_do_offsetted_strides_benchmarks = false;
                break;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "Modes: ";
    for (auto mode : s_modes) {
        std::cout << ModeStrings[static_cast<int>(mode)] << " ";
    }
    std::cout << std::endl;

    std::cout << "Scopes: ";
    for (auto scope : s_scopes) {
        std::cout << ScopeStrings[static_cast<int>(scope)] << " ";
    }
    std::cout << std::endl;

    std::cout << "Grids: ";
    for (auto grid : s_grids) {
        std::cout << grids[static_cast<int>(grid)] << " ";
    }
    std::cout << std::endl;

    std::cout << "Transpose: ";
    for (auto transpose_mode : s_transpose_modes) {
        std::cout << transpose_modes[static_cast<int>(transpose_mode)] << " ";
    }
    std::cout << std::endl;

    std::cout << "Runs: " << s_benchmark_runs << std::endl;
    std::cout << "Baseline runs: " << s_baseline_runs << std::endl;
    std::cout << "Baseline benchmarks: " << (s_do_baseline_benchmarks ? "Yes" : "No") << std::endl;
    std::cout << "Varying threads benchmarks: " << (s_do_varying_threads_benchmarks ? "Yes" : "No") << std::endl;
    std::cout << "Varying strides benchmarks: " << (s_do_varying_strides_benchmarks ? "Yes" : "No") << std::endl;
    std::cout << "Offsetted strides benchmarks: " << (s_do_offsetted_strides_benchmarks ? "Yes" : "No") << std::endl;
};

int main(int argc, char *argv[]) {
    parse_options(argc, argv);

    __init_global_clock();

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    s_warp_size = props.warpSize;
    assert(s_warp_size == 32);

    s_times_buffer = new Buffer(MaxGridSizeLarge * MaxBlockSizeLarge * sizeof(uint64_t), Target::Both);
    s_atomic_buffer = new Buffer(MaxGroupsLarge * (MaxStride + MaxStrideOffset), Target::Both);
    s_mask_buffer = new Buffer(MaxGridSizeLarge * MaxBlockSizeLarge * sizeof(uint64_t), Target::Both);

    HIP_CHECK(hipEventCreateWithFlags(&s_completion_event, hipEventDisableTiming));

    if (s_do_baseline_benchmarks) {
        add_baseline_benchmarks(false);
        add_baseline_benchmarks(true);
    }

    struct config_t {
        int active_lanes;
        int active_warps;
        int groups;
        int mem_stride;
    };

    auto config_compare = [](const config_t &a, const config_t &b){
        return (a.active_lanes < b.active_lanes)
            || (a.active_warps < b.active_warps)
            || (a.groups < b.groups)
            || (a.mem_stride < b.mem_stride);
    };
    auto configs = std::set<config_t, decltype(config_compare)>();

    // NOTE: We add all possible benchmark configs to the list. Configs for the
    //       larger grid will be skipped for the smaller grid.

    // varying number of threads on same variable
    if (s_do_varying_threads_benchmarks) {
        for (int active_lane_bits = MinLaneBits; active_lane_bits <= MaxLaneBits; active_lane_bits++) {
            for (int active_warp_bits = MinWarpBits; active_warp_bits <= MaxWarpBitsLarge; active_warp_bits++) {
                configs.insert({
                    .active_lanes = 1 << active_lane_bits,
                    .active_warps = 1 << active_warp_bits,
                    .groups = 1,
                    .mem_stride = 4,
                });
            }
        }
    }

    // varying group sizes and strides
    if (s_do_varying_strides_benchmarks) {
        for (int group_bits = MinGroupBits; group_bits <= MaxGroupBitsLarge; group_bits++) {
            for (int stride_bits = MinStrideBits; stride_bits <= MaxStrideBits; stride_bits++) {
                configs.insert({
                    .active_lanes = (int)s_warp_size,
                    .active_warps = MaxWarpsLarge,
                    .groups = 1 << group_bits,
                    .mem_stride = 1 << stride_bits,
                });

                configs.insert({
                    .active_lanes = (int)s_warp_size,
                    .active_warps = MaxWarpsSmall,
                    .groups = 1 << group_bits,
                    .mem_stride = 1 << stride_bits,
                });
            }

            configs.insert({
                .active_lanes = (int)s_warp_size,
                .active_warps = MaxWarpsLarge,
                .groups = 1 << group_bits,
                .mem_stride = 0,
            });

            configs.insert({
                .active_lanes = (int)s_warp_size,
                .active_warps = MaxWarpsSmall,
                .groups = 1 << group_bits,
                .mem_stride = 0,
            });
        }
    }

    // varying memory stride offsets
    if (s_do_offsetted_strides_benchmarks) {
        for (int stride_bits = MinStrideBits; stride_bits <= MaxStrideBits; stride_bits++) {
            for (int stride_offset = MinStrideOffset; stride_offset <= MaxStrideOffset; stride_offset += StrideOffsetStepSize) {
                int mem_stride = (1 << stride_bits) + stride_offset;

                if (mem_stride < 0) {
                    continue;
                }

                configs.insert({
                    .active_lanes = (int)s_warp_size,
                    .active_warps = MaxWarpsLarge,
                    .groups = MaxGroupsLarge,
                    .mem_stride = mem_stride,
                });

                configs.insert({
                    .active_lanes = (int)s_warp_size,
                    .active_warps = MaxWarpsSmall,
                    .groups = MaxGroupsSmall,
                    .mem_stride = mem_stride,
                });
            }
        }
    }

    for (const auto &config : configs) {
        add_benchmarks(config.active_lanes, config.active_warps, config.groups, config.mem_stride);
    }

    bm::run_randomized(s_benchmarks);

    delete s_times_buffer;
    delete s_atomic_buffer;
    delete s_mask_buffer;

    return 0;
}
