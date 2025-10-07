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
#define ITERATIONS (1 << 12)

__global__ void kernel(simt::atomic<uint32_t, simt::thread_scope_system> *var) {
    for (int i = 0; i < ITERATIONS; i++) {
        var->fetch_add(1);
    }
}

int main(int argc, char *argv[]) {
    std::atomic<int> flag;

    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <path> <count> <outpath?>" << std::endl;
        return 1;
    }

    const char *path = argv[1];
    int count = std::atoi(argv[2]);

    if (count < 1) {
        std::cerr << "Count should be at least 1" << std::endl;
        return 1;
    }

    std::cout << "Waiting for file creation..." << std::endl;
    while (!std::filesystem::exists(path)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    int fd = open(path, O_RDWR | O_DIRECT | O_SYNC);

    if (fd == -1) {
        perror("Cannot open file");
        return 1;
    }

    long page_size = sysconf(_SC_PAGE_SIZE);

    if (page_size == -1) {
        perror("Cannot query page size");
        return 1;
    }

    if (ftruncate(fd, page_size * 2) == -1) {
        perror("Cannot resize file");
        return 1;
    }

    void *ptr = mmap(NULL, page_size * 2, PROT_READ | PROT_WRITE, MAP_SHARED_VALIDATE | MAP_POPULATE, fd, 0);

    if (ptr == MAP_FAILED) {
        perror("Cannot map file");
        return 1;
    }

    uint32_t *barrier_ptr_1 = static_cast<uint32_t *>(ptr);
    uint32_t *barrier_ptr_2 = barrier_ptr_1 + 1;
    uint32_t *counter_ptr_host = barrier_ptr_1 + page_size / sizeof(uint32_t);
    HIP_CHECK(hipHostRegister(counter_ptr_host, sizeof(uint32_t), hipHostRegisterDefault));

    std::atomic_ref<uint32_t> barrier1(*barrier_ptr_1);
    std::atomic_ref<uint32_t> barrier2(*barrier_ptr_2);
    std::atomic_ref<uint32_t> gpu_counter(*counter_ptr_host);
    auto *counter_ptr_device = reinterpret_cast<simt::atomic<uint32_t, simt::thread_scope_system> *>(counter_ptr_host);

    uint32_t expected = ITERATIONS * GRIDSIZE * BLOCKSIZE * count;
    std::vector<uint32_t> values;
    values.reserve(expected);
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

    std::cout << "Waiting for CPU barrier..." << std::endl;
    uint32_t current = barrier1.fetch_add(1);
    while (current < count) {
        current = barrier1.load();
    }

    hipLaunchKernelGGL(kernel, dim3(GRIDSIZE), dim3(BLOCKSIZE), 0, nullptr, counter_ptr_device);

    current = gpu_counter.load();

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Kernel done. Waiting for second CPU barrier..." << std::endl;
    current = barrier2.fetch_add(1);
    while (current < count) {
        current = barrier2.load();
    }

    uint32_t result = gpu_counter.load();
    flag += 1;
    newthread.join();

    std::cout << "Expected: " << expected << "   Got: " << result << std::endl;
    std::cout << "Length: " << values.size() << std::endl;

    if (munmap(ptr, page_size * 2) == -1) {
        perror("Cannot unmap file");
        return 1;
    }

    if (close(fd) == -1) {
        perror("Cannot close file");
        return 1;
    }

    if (argc == 4) {
        std::ofstream outfile{argv[3]};
        outfile << "recorded_values" << '\n';
        for (auto copy : values) outfile << copy << '\n';
    }

    return 0;
}
