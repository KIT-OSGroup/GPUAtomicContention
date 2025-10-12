#include "utility.hpp"

#ifdef __HIP_PLATFORM_AMD__
__device__ uint64_t d_wall_clock_rate;
#endif

bool wait_for_completion(hipEvent_t &event, int seconds) {
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
