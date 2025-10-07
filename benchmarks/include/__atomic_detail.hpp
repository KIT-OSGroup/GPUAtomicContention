#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cassert>

#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif //__ATOMIC_RELAXED

#ifndef __ATOMIC_BLOCK
#define __ATOMIC_SYSTEM 0
#define __ATOMIC_DEVICE 1
#define __ATOMIC_BLOCK 2
#endif //__ATOMIC_BLOCK

#define ATOMIC_WITH_MEMORDER(order, atomic, fence) \
do {                                               \
    switch (order) {                               \
        case __ATOMIC_RELEASE:                     \
            (fence);                               \
            [[fallthrough]];                       \
        case __ATOMIC_RELAXED:                     \
            result = (atomic);                     \
            break;                                 \
        case __ATOMIC_SEQ_CST:                     \
        case __ATOMIC_ACQ_REL:                     \
            (fence);                               \
            [[fallthrough]];                       \
        case __ATOMIC_CONSUME:                     \
        case __ATOMIC_ACQUIRE:                     \
            result = (atomic);                     \
            (fence);                               \
            break;                                 \
        default:                                   \
            assert(false);                         \
            break;                                 \
    }                                              \
} while (0)

namespace simt {
    enum thread_scope {
        thread_scope_block = __ATOMIC_BLOCK,
        thread_scope_device = __ATOMIC_DEVICE,
        thread_scope_system = __ATOMIC_SYSTEM
    };

    namespace detail {
        struct __thread_scope_block_tag {};
        struct __thread_scope_device_tag : __thread_scope_block_tag {};
        struct __thread_scope_system_tag : __thread_scope_device_tag {};

        template<int Scope>
        struct __scope_enum_to_tag {};

        template<>
        struct __scope_enum_to_tag<(int) thread_scope_block> {
            using type = __thread_scope_block_tag;
        };

        template<>
        struct __scope_enum_to_tag<(int) thread_scope_device> {
            using type = __thread_scope_device_tag;
        };

        template<>
        struct __scope_enum_to_tag<(int) thread_scope_system> {
            using type = __thread_scope_system_tag;
        };

        template <int Scope>
        using __scope_to_tag = typename __scope_enum_to_tag<Scope>::type;

        template <typename T>
        __device__ T __atomic_exchange(T *pointer, T desired, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicExch(pointer, desired), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_exchange(T *pointer, T desired, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicExch_system(pointer, desired), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ bool __atomic_compare_exchange_strong(T *address, T expected, T desired, int memorder, __thread_scope_block_tag) noexcept {
            T result{};
            ATOMIC_WITH_MEMORDER(memorder, atomicCAS(address, expected, desired), __threadfence());
            return result == expected;
        }

        template <typename T>
        __device__ bool __atomic_compare_exchange_strong(T *address, T expected, T desired, int memorder, __thread_scope_system_tag) noexcept {
            T result{};
            ATOMIC_WITH_MEMORDER(memorder, atomicCAS_system(address, expected, desired), __threadfence_system());
            return result == expected;
        }


        template <typename T>
        __device__ T __atomic_compare_exchange_strong_raw(T *address, T expected, T desired, int memorder, __thread_scope_block_tag) noexcept {
            T result{};
            ATOMIC_WITH_MEMORDER(memorder, atomicCAS(address, expected, desired), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_compare_exchange_strong_raw(T *address, T expected, T desired, int memorder, __thread_scope_system_tag) noexcept {
            T result{};
            ATOMIC_WITH_MEMORDER(memorder, atomicCAS_system(address, expected, desired), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ T __atomic_add(T *pointer, T arg, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicAdd(pointer, arg), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_add(T *pointer, T arg, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicAdd_system(pointer, arg), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ T __atomic_sub(T *pointer, T arg, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicSub(pointer, arg), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_sub(T *pointer, T arg, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicSub_system(pointer, arg), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ T __atomic_and(T *pointer, T arg, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicAnd(pointer, arg), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_and(T *pointer, T arg, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicAnd_system(pointer, arg), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ T __atomic_or(T *pointer, T arg, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicOr(pointer, arg), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_or(T *pointer, T arg, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicOr_system(pointer, arg), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ T __atomic_xor(T *pointer, T arg, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicXor(pointer, arg), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_xor(T *pointer, T arg, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicXor_system(pointer, arg), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ T __atomic_max(T *pointer, T arg, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicMax(pointer, arg), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_max(T *pointer, T arg, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicMax_system(pointer, arg), __threadfence_system());
            return result;
        }


        template <typename T>
        __device__ T __atomic_min(T *pointer, T arg, int memorder, __thread_scope_block_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicMin(pointer, arg), __threadfence());
            return result;
        }

        template <typename T>
        __device__ T __atomic_min(T *pointer, T arg, int memorder, __thread_scope_system_tag) noexcept {
            T result;
            ATOMIC_WITH_MEMORDER(memorder, atomicMin_system(pointer, arg), __threadfence_system());
            return result;
        }
    }
}

#undef ATOMIC_WITH_MEMORDER
