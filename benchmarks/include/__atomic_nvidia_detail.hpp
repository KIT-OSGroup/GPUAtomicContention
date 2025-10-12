/*

NOTE: The implementation of atomic loads and stores below uses PTX inline
      assembly as specified in the PTX ISA.
      See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-acquire-patterns

*/

#pragma once

#include <hip/hip_runtime.h>
#include <cassert>

#include "__atomic_detail.hpp"

#define LOAD(addr, dst, order, scope, width, reg) asm volatile("ld.global." #order "." #scope "." #width " %0, [%1];" : #reg(dst) : "l"(addr) : "memory")
#define STORE(addr, src, order, scope, width, reg) asm volatile("st.global." #order "." #scope "." #width " [%0], %1;":: "l"(addr), #reg(src) : "memory")

#define DISPATCH_LOAD(addr, dst, order, scope, width, reg)          \
do {                                                                \
    switch (order) {                                                \
        case __ATOMIC_SEQ_CST:                                      \
            LOAD(addr, dst, acquire, scope, width, reg);            \
            asm volatile("fence.sc." #scope ";":: : "memory");      \
            break;                                                  \
        case __ATOMIC_CONSUME:                                      \
        case __ATOMIC_ACQUIRE:                                      \
            LOAD(addr, dst, acquire, scope, width, reg);            \
            break;                                                  \
        case __ATOMIC_RELAXED:                                      \
            LOAD(addr, dst, relaxed, scope, width, reg);            \
            break;                                                  \
        default:                                                    \
            assert(false);                                          \
            break;                                                  \
    }                                                               \
} while (0)

#define DISPATCH_STORE(addr, src, order, scope, width, reg)         \
do {                                                                \
    switch (order) {                                                \
        case __ATOMIC_SEQ_CST:                                      \
            asm volatile("fence.sc." #scope ";":: : "memory");      \
            [[fallthrough]];                                        \
        case __ATOMIC_RELEASE:                                      \
            STORE(addr, src, release, scope, width, reg);           \
            break;                                                  \
        case __ATOMIC_RELAXED:                                      \
            STORE(addr, src, relaxed, scope, width, reg);           \
            break;                                                  \
        default:                                                    \
            assert(false);                                          \
            break;                                                  \
    }                                                               \
} while (0)


namespace simt::detail {
    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 4) {
        DISPATCH_STORE(ptr, val, memorder, cta, b32, r);
    }

    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 8) {
        DISPATCH_STORE(ptr, val, memorder, cta, b64, l);
    }

    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_device_tag) requires(sizeof(T) == 4) {
        DISPATCH_STORE(ptr, val, memorder, gpu, b32, r);
    }

    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_device_tag) requires(sizeof(T) == 8) {
        DISPATCH_STORE(ptr, val, memorder, gpu, b64, l);
    }

    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_system_tag) requires(sizeof(T) == 4) {
        DISPATCH_STORE(ptr, val, memorder, sys, b32, r);
    }

    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_system_tag) requires(sizeof(T) == 8) {
        DISPATCH_STORE(ptr, val, memorder, sys, b64, l);
    }


    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 4) {
        typename std::remove_cv<T>::type val;
        DISPATCH_LOAD(ptr, val, memorder, cta, b32, =r);
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 8) {
        typename std::remove_cv<T>::type val;
        DISPATCH_LOAD(ptr, val, memorder, cta, b64, =l);
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_device_tag) requires(sizeof(T) == 4) {
        typename std::remove_cv<T>::type val;
        DISPATCH_LOAD(ptr, val, memorder, gpu, b32, =r);
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_device_tag) requires(sizeof(T) == 8) {
        typename std::remove_cv<T>::type val;
        DISPATCH_LOAD(ptr, val, memorder, gpu, b64, =l);
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_system_tag) requires(sizeof(T) == 4) {
        typename std::remove_cv<T>::type val;
        DISPATCH_LOAD(ptr, val, memorder, sys, b32, =r);
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_system_tag) requires(sizeof(T) == 8) {
        typename std::remove_cv<T>::type val;
        DISPATCH_LOAD(ptr, val, memorder, sys, b64, =l);
        return val;
    }
}

#undef LOAD
#undef STORE
#undef DISPATCH_LOAD
#undef DISPATCH_STORE
