#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <ostream>
#include <vector>

#include "buffer.hpp"
#include "utility.hpp"

static constexpr int Iterations = 1 << 16;

__global__ void kernel(uint64_t *buffer) {
    if (global_id() == 0) {
        for (int i = 0; i < Iterations; i++) {
            buffer[i] = __global_clock();
        }
    }
}

int main(int, char *[]) {
#if defined (__HIP_PLATFORM_AMD__)
    __init_global_clock();
#endif

    Buffer buffer(Iterations * sizeof(uint64_t), Target::Both);

    hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, nullptr, static_cast<uint64_t *>(buffer.get_device_buffer()));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    buffer.copy(Target::Host);
    uint64_t *times = static_cast<uint64_t *>(buffer.get_host_buffer());

    uint64_t start = times[0];
    uint64_t current = times[0];
    uint64_t count = 0;

    float resoultion_sample_sum = 0.0f;
    float resoultion_sample_count = 0.0f;
    
    for (int i = 0; i < Iterations; i++) {
        if (times[i] != current) {
            resoultion_sample_sum += (float)(times[i] - current);
            resoultion_sample_count += 1.0f;
            std::cout << std::setw(16) << (current - start) << "ns x " << std::setw(2) << count << "  ---  Approx. resoultion: " << (resoultion_sample_sum / resoultion_sample_count) << "ns" << std::endl;
            count = 0;
        }

        current = times[i];
        count++;
    }
}
