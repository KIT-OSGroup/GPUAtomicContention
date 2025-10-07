#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <thread>
#include <unistd.h>
#include <sys/mman.h>
#include <vector>
#include <fcntl.h>
#include <thread>

#include "buffer.hpp"
#include "utility.hpp"
#include "atomic.hpp"

#define GRIDSIZE 128
#define BLOCKSIZE 128
#define ITERATIONS_GPU (1 << 12)
#define ITERATIONS_HOST (1 << 22)
#define ITERATIONS (ITERATIONS_HOST + ITERATIONS_GPU * GRIDSIZE * BLOCKSIZE)

__global__ void kernel(simt::atomic<uint32_t, simt::thread_scope_system> *var) {
    for (int i = 0; i < ITERATIONS_GPU; i++) {
        var->fetch_add(1);
    }
}

int main(int argc, char *argv[]) {
	std::atomic<int> flag;

    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " <outpath?>" << std::endl;
        return 1;
    }

    long page_size = sysconf(_SC_PAGE_SIZE);

    if (page_size == -1) {
        perror("Cannot query page size");
        return 1;
    }

    void *ptr = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_POPULATE | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (ptr == MAP_FAILED) {
        perror("Cannot allocate memory");
        return 1;
    }

    uint32_t *counter = static_cast<uint32_t *>(ptr);
    HIP_CHECK(hipHostRegister(counter, sizeof(uint32_t), hipHostRegisterDefault));

    std::atomic_ref<uint32_t> gpu_counter(*counter);
    auto *counter_ptr_device = reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_system> *>(counter);

    std::vector<uint32_t> values;
    values.reserve(ITERATIONS);
    std::thread newthread([&] {
        uint32_t current = gpu_counter.load();

        while(!flag.load()) {
            for (int i = 0; i < (1 << 30); ++i) {
                uint32_t new_value = gpu_counter.load();

                if (new_value != current) {
                    current = new_value;
                    values.push_back(current);
                }
            }
        }
    });

    hipLaunchKernelGGL(kernel, dim3(GRIDSIZE), dim3(BLOCKSIZE), 0, nullptr, counter_ptr_device);

    for (int i = 0; i < ITERATIONS_HOST; i++) {
        gpu_counter.fetch_add(1);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    uint32_t result = gpu_counter.load();
    flag += 1;
    newthread.join();

    std::cout << "Expected: " << ITERATIONS << "   Got: " << result << std::endl;
    std::cout << "Length: " << values.size() << std::endl;

    if (munmap(ptr, page_size) == -1) {
        perror("Cannot unmap memory region");
        return 1;
    }

    if (argc == 2) {
        std::ofstream outfile{argv[1]};
        outfile << "recorded_values" << '\n';
        for (auto copy : values) outfile << copy << '\n';
    }

    return 0;
}
