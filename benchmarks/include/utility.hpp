#ifndef INCLUDED___base___utility_hpp
#define INCLUDED___base___utility_hpp

#include <climits>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <utility>
#include <chrono>
#include <iostream>
#include <ostream>
#include <thread>
#include <unistd.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_AMD__
__device__ extern uint64_t d_wall_clock_rate;
#endif

#define HIP_CHECK(command) do {                                                         \
    hipError_t __status = command;                                                        \
    if (__status != hipSuccess) {                                                         \
        std::cerr << __FILE__ << ":" << __LINE__ << ": " << #command << std::endl;      \
        std::cerr << "Error: HIP reports " << hipGetErrorString(__status) << std::endl;   \
        std::abort();                                                                   \
    }                                                                                   \
} while(0)

#ifdef ENABLE_LOGGING
#define DEVICE_LOG(format, ...) do {                                 \
    printf("[%d @ %s] " format, global_id(), __func__, __VA_ARGS__); \
} while(0)
#else
#define DEVICE_LOG(...)
#endif

__device__ __host__ inline int block_size() {
#ifdef __HIP_DEVICE_COMPILE__
    return blockDim.x * blockDim.y * blockDim.z;
#else
    return 1;
#endif
}


__device__ __host__ inline int block_id() {
#ifdef __HIP_DEVICE_COMPILE__
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    return blockId;
#else
    return 0;
#endif
}

__device__ __host__ inline int local_id() {
#ifdef __HIP_DEVICE_COMPILE__
    int threadId = (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
#else
    return 0;
#endif
}

__device__ __host__ inline int global_id() {
#ifdef __HIP_DEVICE_COMPILE__
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
#else
    return 0;
#endif
}

__device__ __host__ inline int grid_size() {
#ifdef __HIP_DEVICE_COMPILE__
    return gridDim.x * gridDim.y * gridDim.z;
#else
    return 1;
#endif
}

__device__ __host__ inline int total_thread_count() {
#ifdef __HIP_DEVICE_COMPILE__
    return grid_size() * block_size();
#else
    return 1;
#endif
}

__device__ __host__ inline unsigned lane_id() {
#ifdef __HIP_DEVICE_COMPILE__
#ifdef __HIP_PLATFORM_NVIDIA__
    unsigned lane_id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
#else
    return __lane_id();
#endif
#else
    return 0;
#endif
}

__device__ __host__ inline uint32_t sm_id() {
#ifdef __HIP_DEVICE_COMPILE__
#ifdef __HIP_PLATFORM_NVIDIA__
    int32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return sm_id;
#else
    return __smid();
#endif
#else
    return 0;
#endif
}

__host__ inline void __init_global_clock() {
#ifdef __HIP_PLATFORM_AMD__
    int rate = 0;
    HIP_CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));

    uint64_t host_rate = 1000000ull / (uint64_t)rate; // converts kHz to ns/tick
    HIP_CHECK(hipMemcpyToSymbol(d_wall_clock_rate, &host_rate, sizeof(uint64_t)));
#endif
}

__device__ __host__ inline uint64_t __global_clock() {
#ifdef __HIP_DEVICE_COMPILE__
#ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    return ns;
#else
    return wall_clock64() * d_wall_clock_rate;
#endif
#else
    return 0;
#endif
}


__device__ __host__ inline void __sleep(uint16_t ns) {
#ifdef __HIP_DEVICE_COMPILE__
#ifdef __HIP_PLATFORM_NVIDIA__
    __nanosleep(ns);
#else
    // 's_sleep' sleeps between 0 and 960 clock cycles in 64 cycle increments
    // using a 6-bit immediate, so we'll use a switch-statement and let the
    // compiler possibly optimize futher
    uint16_t cycles = std::clamp(ns / 64, 1, 32);
    switch (cycles) {
        case 1: __builtin_amdgcn_s_sleep(1); break;
        case 2: __builtin_amdgcn_s_sleep(2); break;
        case 3: __builtin_amdgcn_s_sleep(3); break;
        case 4: __builtin_amdgcn_s_sleep(4); break;
        case 5: __builtin_amdgcn_s_sleep(5); break;
        case 6: __builtin_amdgcn_s_sleep(6); break;
        case 7: __builtin_amdgcn_s_sleep(7); break;
        case 8: __builtin_amdgcn_s_sleep(8); break;
        case 9: __builtin_amdgcn_s_sleep(9); break;
        case 10: __builtin_amdgcn_s_sleep(10); break;
        case 11: __builtin_amdgcn_s_sleep(11); break;
        case 12: __builtin_amdgcn_s_sleep(12); break;
        case 13: __builtin_amdgcn_s_sleep(13); break;
        case 14: __builtin_amdgcn_s_sleep(14); break;
        case 15: __builtin_amdgcn_s_sleep(15); break;
        case 16: __builtin_amdgcn_s_sleep(16); break;
        case 17: __builtin_amdgcn_s_sleep(17); break;
        case 18: __builtin_amdgcn_s_sleep(18); break;
        case 19: __builtin_amdgcn_s_sleep(19); break;
        case 20: __builtin_amdgcn_s_sleep(20); break;
        case 21: __builtin_amdgcn_s_sleep(21); break;
        case 22: __builtin_amdgcn_s_sleep(22); break;
        case 23: __builtin_amdgcn_s_sleep(23); break;
        case 24: __builtin_amdgcn_s_sleep(24); break;
        case 25: __builtin_amdgcn_s_sleep(25); break;
        case 26: __builtin_amdgcn_s_sleep(26); break;
        case 27: __builtin_amdgcn_s_sleep(27); break;
        case 28: __builtin_amdgcn_s_sleep(28); break;
        case 29: __builtin_amdgcn_s_sleep(29); break;
        case 30: __builtin_amdgcn_s_sleep(30); break;
        case 31: __builtin_amdgcn_s_sleep(31); break;
        default: abort(); break;
    }
#endif
#else
    //
#endif
}

__host__ __device__ inline int popcount(uint64_t val) {
    int count = 0;

    for (int i = 0; i < 64; i++) {
        if (val & (1ull << i)) {
            count++;
        }
    }

    return count;
}

__device__ inline int mask_relative_lane_id(int lane_id, uint64_t active_lane_mask)
{
    int id = 0;

    for (int bit = 0; bit < warpSize; bit++) {
        if (lane_id == bit) {
            return id;
        }

        if (active_lane_mask & (1ull << bit)) {
            id++;
        }
    }

    return -1;
}

// Waits for completion of the kernel with an approximate timeout in seconds.
static bool wait_for_completion(hipEvent_t &event, int seconds) {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;
    std::chrono::duration<double> timeout = std::chrono::seconds(seconds);

    do {
        if (hipEventQuery(event) == hipSuccess) {
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
    } while(elapsed < timeout);

    return false;
}

#endif // INCLUDED___base___utility_hpp
