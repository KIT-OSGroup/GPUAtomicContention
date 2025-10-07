#include "utility.hpp"

#ifdef __HIP_PLATFORM_AMD__
__device__ uint64_t d_wall_clock_rate;
#endif
