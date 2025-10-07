#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <linux/limits.h>

#include "tests.hpp"
#include "atomic.hpp"
#include "utility.hpp"

static hipEvent_t s_completion_event{};
static simt::atomic<uint32_t> *s_barrier1{nullptr};
static simt::atomic<uint32_t> *s_barrier2{nullptr};

__host__ static void barrier_init(simt::atomic<uint32_t> *barrier) {
    HIP_CHECK(hipMemset(barrier, 0, sizeof(*barrier)));
}

__device__ void backoff_barrier_wait(simt::atomic<uint32_t> *barrier, uint32_t want) {
    uint32_t count = barrier->fetch_add(1) + 1;
    int ns = 8;

    while (count < want) {
        __sleep(ns);
        ns = std::min(ns * 2, 256);

        count = *barrier;
    }

    // NOTE: omitting decreasing the barrier as it is only used once
}

__device__ void simple_barrier_wait(simt::atomic<uint32_t> *barrier, uint32_t want) {
    uint32_t count = barrier->fetch_add(1) + 1;

    while (count < want) {
        count = *barrier;
    }

    // NOTE: omitting decreasing the barrier as it is only used once
}

__global__ void kernel_backoff_barrier(simt::atomic<uint32_t> *barrier1, simt::atomic<uint32_t> *barrier2, uint32_t want) {
    backoff_barrier_wait(barrier1, want);
    backoff_barrier_wait(barrier2, want);
}

__global__ void kernel_simple_barrier(simt::atomic<uint32_t> *barrier1, simt::atomic<uint32_t> *barrier2, uint32_t want) {
    simple_barrier_wait(barrier1, want);
    simple_barrier_wait(barrier2, want);
}

static auto string_to_kernel(const char *kernel) -> decltype(&kernel_simple_barrier) {
    if (std::strcmp(kernel, "kernel_simple_barrier")) {
        return kernel_simple_barrier;
    }

    if (std::strcmp(kernel, "kernel_backoff_barrier")) {
        return kernel_backoff_barrier;
    }

    return static_cast<decltype(&kernel_simple_barrier)>(nullptr);
}

void test_barrier(const char *kernel_str, int grid_size, int block_size) {
    auto kernel = string_to_kernel(kernel_str);
    assert(kernel);

    barrier_init(s_barrier1);
    barrier_init(s_barrier2);
    hipLaunchKernelGGL(kernel, dim3(grid_size), dim3(block_size), 0, nullptr, s_barrier1, s_barrier2, grid_size * block_size);
    HIP_CHECK(hipEventRecord(s_completion_event));

    if (wait_for_completion(s_completion_event, 10)) {
        return;
    }

    FAIL("timeout");
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <kernel> <grid_size> <block_size>" << std::endl;
        std::cerr << "Available kernels: " << " simple_barrier backoff_barrier" << std::endl;
        return 1;
    }

    const char *kernel_str = argv[1];
    int grid_size = atoi(argv[2]);
    int block_size = atoi(argv[3]);

    HIP_CHECK(hipEventCreateWithFlags(&s_completion_event, hipEventDisableTiming));

    HIP_CHECK(hipMalloc(&s_barrier1, sizeof(*s_barrier1)));
    HIP_CHECK(hipMalloc(&s_barrier2, sizeof(*s_barrier2)));

    char test_name[PATH_MAX];
    sprintf(test_name, "%s_%dx%d", kernel_str, grid_size, block_size);

    auto ret = test::test(test_name, [kernel_str, grid_size, block_size](){ test_barrier(kernel_str, grid_size, block_size); });

    if (ret == test::Completion::FAILURE) {
        abort();
    }

    return 0;
}
