cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21.3...3.27)
project(gpu_nvme-project LANGUAGES CXX)

include(cmake/helpers.cmake)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/rocm;/opt/cuda")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/opt/rocm/lib/cmake/hip")

set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")

if (DEFINED ENV{AMDGPU_ARCH})
    message("AMDGPU architecture is: $ENV{AMDGPU_ARCH}")
    set(AMDGPU_TARGETS "$ENV{AMDGPU_ARCH}")
    set(ENV{AMDGPU_TARGETS} "$ENV{AMDGPU_ARCH}")
    set(CMAKE_HIP_ARCHITECTURES "$ENV{AMDGPU_ARCH}")
endif()

find_package(hip REQUIRED)

if(${HIP_PLATFORM} MATCHES "nvidia")
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};/opt/rocm/hip/cmake")
    set(ENV{HIP_PLATFORM} nvidia)
    find_package(HIP MODULE REQUIRED)
else()
    find_package(hip REQUIRED)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(${HIP_PLATFORM} MATCHES "nvidia")
    message(STATUS "HIP_PLATFORM: nvidia")
    find_package(CUDAToolkit REQUIRED)
endif()

if(${HIP_PLATFORM} MATCHES "amd")
    link_directories("/usr/local/lib")
endif()

if("$ENV{AMDGPU_ARCH}" STREQUAL "")
    set(AMDGPU_ARCH "--offload-arch=gfx1030")
else()
    set(AMDGPU_ARCH $ENV{AMDGPU_ARCH})
endif()

if("$ENV{NVIDIA_ARCH}" STREQUAL "")
    set(NVIDIA_ARCH "-gencode=arch=compute_86,code=sm_86")
else()
    set(NVIDIA_ARCH $ENV{NVIDIA_ARCH})
endif()

separate_arguments(AMD_OFFLOAD_ARCH NATIVE_COMMAND $ENV{AMDGPU_ARCH})
separate_arguments(NVIDIA_GENCODE NATIVE_COMMAND $ENV{NVIDIA_ARCH})

if(${HIP_PLATFORM} MATCHES "nvidia")
    message(STATUS "linking against CUDA::cudart")
    set(ADDITIONAL_DEFINITIONS HIP_TEMPLATE_KERNEL_LAUNCH HIP_ENABLE_WARP_SYNC_BUILTINS)
    set(ADDITIONAL_COMPILE_OPTIONS --extended-lambda --expt-relaxed-constexpr -dc -O3 ${NVIDIA_GENCODE})
    set(ADDITIONAL_LINK_OPTIONS -O3 ${NVIDIA_GENCODE})
    set(ADDITIONAL_LIBRARIES)
else()
    message(STATUS "linking against hip::device")
    set(ADDITIONAL_DEFINITIONS HIP_TEMPLATE_KERNEL_LAUNCH HIP_ENABLE_WARP_SYNC_BUILTINS)
    set(ADDITIONAL_COMPILE_OPTIONS -fgpu-rdc -Wno-deprecated-register -O3 ${AMD_OFFLOAD_ARCH})
    set(ADDITIONAL_LINK_OPTIONS -fgpu-rdc -Wno-deprecated-register -O3 ${AMD_OFFLOAD_ARCH})
endif()

set(ADDITIONAL_DEFINITIONS ${ADDITIONAL_DEFINITIONS} ENABLE_LOGGING)

set(SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarking.cpp"
)

add_library(base STATIC ${SOURCES})
add_flags(base)
target_include_directories(base PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(base ${ADDITIONAL_LIBRARIES})

add_binary(bm_contention ${CMAKE_CURRENT_SOURCE_DIR}/src/contention.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_binary(bm_atomics ${CMAKE_CURRENT_SOURCE_DIR}/src/atomics.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_binary(test_barriers ${CMAKE_CURRENT_SOURCE_DIR}/src/barriers.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_binary(test_cross_gpu_atomics ${CMAKE_CURRENT_SOURCE_DIR}/src/cross_gpu_atomics.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_binary(test_gpu_cpu_atomics ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_cpu_atomics.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_binary(test_globaltimer ${CMAKE_CURRENT_SOURCE_DIR}/src/globaltimer.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include)
