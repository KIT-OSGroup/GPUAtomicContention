/*

NOTE: Our simt::atomic implementation is in part inspired by NVIDIA's
      cuda::atomic inside libcu++. Specifically, the function dispatch using
      tags for each thread scope. As such, we include the corresponding
      copyright notice.

--------------------------------------------------------------------------------

Copyright (c) 2018, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#pragma once

#include <concepts>
#include <hip/hip_runtime.h>
#include <atomic>

#if defined(__HIP_PLATFORM_AMD__)
#   include "__atomic_amd_detail.hpp"
#elif defined(__HIP_PLATFORM_NVIDIA__)
#   include "__atomic_nvidia_detail.hpp"
#endif

#include "utility.hpp"

namespace simt {
    using memory_order = std::memory_order;

    constexpr memory_order memory_order_relaxed = std::memory_order_relaxed;
    constexpr memory_order memory_order_consume = std::memory_order_consume;
    constexpr memory_order memory_order_acquire = std::memory_order_acquire;
    constexpr memory_order memory_order_release = std::memory_order_release;
    constexpr memory_order memory_order_acq_rel = std::memory_order_acq_rel;
    constexpr memory_order memory_order_seq_cst = std::memory_order_seq_cst;

    template<typename T, thread_scope Scope = thread_scope_device>
    struct atomic {
        using value_type = T;

        __host__ __device__ constexpr atomic() noexcept(std::is_nothrow_default_constructible_v<T>) = default;
        __host__ __device__ constexpr atomic(T desired) noexcept
            : m_value(desired)
        {}
        atomic(const atomic&) = delete;
        atomic& operator=(const atomic&) = delete;
        atomic& operator=(const atomic&) volatile = delete;


        __device__ T load(memory_order order = memory_order_seq_cst) const noexcept {
            return detail::__atomic_load(std::addressof(m_value), static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ operator T() const noexcept {
            return load();
        }

        __device__ void store(T desired, memory_order order = memory_order_seq_cst) noexcept {
            detail::__atomic_store(std::addressof(m_value), desired, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T operator=(T desired) noexcept {
            store(desired);
            return desired;
        }


        __device__ T exchange(T desired, memory_order order = memory_order_seq_cst) noexcept {
            return detail::__atomic_exchange(std::addressof(m_value), desired, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ bool compare_exchange_strong(T &expected, T desired, memory_order order = memory_order_seq_cst) noexcept {
            return detail::__atomic_compare_exchange_strong(std::addressof(m_value), expected, desired, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T compare_exchange_strong_raw(T &expected, T desired, memory_order order = memory_order_seq_cst) noexcept {
            return detail::__atomic_compare_exchange_strong_raw(std::addressof(m_value), expected, desired, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T fetch_add(T arg, memory_order order = memory_order_seq_cst) noexcept requires(std::integral<T>) {
            return detail::__atomic_add(std::addressof(m_value), arg, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T fetch_sub(T arg, memory_order order = memory_order_seq_cst) noexcept requires(std::integral<T>) {
            return detail::__atomic_sub(std::addressof(m_value), arg, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T fetch_and(T arg, memory_order order = memory_order_seq_cst) noexcept requires(std::integral<T>) {
            return detail::__atomic_and(std::addressof(m_value), arg, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T fetch_or(T arg, memory_order order = memory_order_seq_cst) noexcept requires(std::integral<T>) {
            return detail::__atomic_or(std::addressof(m_value), arg, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T fetch_xor(T arg, memory_order order = memory_order_seq_cst) noexcept requires(std::integral<T>) {
            return detail::__atomic_xor(std::addressof(m_value), arg, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T fetch_max(T arg, memory_order order = memory_order_seq_cst) noexcept requires(std::integral<T>) {
            return detail::__atomic_max(std::addressof(m_value), arg, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }

        __device__ T fetch_min(T arg, memory_order order = memory_order_seq_cst) noexcept requires(std::integral<T>) {
            return detail::__atomic_min(std::addressof(m_value), arg, static_cast<int>(order), detail::__scope_to_tag<Scope>());
        }


        __device__ T operator++() noexcept requires(std::integral<T>) {
            return fetch_add(1) + 1;
        }

        __device__ T operator++(int) noexcept requires(std::integral<T>) {
            return fetch_add(1);
        }

        __device__ T operator--() noexcept requires(std::integral<T>) {
            return fetch_sub(1) - 1;
        }

        __device__ T operator--(int) noexcept requires(std::integral<T>) {
            return fetch_sub(1);
        }

        __device__ T operator+=(T arg) noexcept requires(std::integral<T>) {
            return fetch_add(arg) + arg;
        }

        __device__ T operator-=(T arg) noexcept requires(std::integral<T>) {
            return fetch_sub(arg) - arg;
        }

        __device__ T operator&=(T arg) noexcept requires(std::integral<T>) {
            return fetch_and(arg) & arg;
        }

        __device__ T operator|=(T arg) noexcept requires(std::integral<T>) {
            return fetch_or(arg) | arg;
        }

        __device__ T operator^=(T arg) noexcept requires(std::integral<T>) {
            return fetch_xor(arg) ^ arg;
        }

        __host__ __device__ T *data() noexcept {
            return std::addressof(m_value);
        }

    private:
        T m_value{};
    };

    template<thread_scope Scope = thread_scope_device>
    struct atomic_flag {
        __host__ __device__ constexpr atomic_flag() noexcept = default;
        atomic_flag(const atomic_flag&) = delete;
        atomic_flag& operator=(const atomic_flag&) = delete;
        atomic_flag& operator=(const atomic_flag&) volatile = delete;

        __device__ bool test(memory_order order = memory_order_seq_cst) const noexcept {
            return m_value.load(order);
        }

        __device__ bool test_and_set(memory_order order = memory_order_seq_cst) noexcept {
            return m_value.fetch_or(true, order);
        }

        __device__ void clear(memory_order order = memory_order_seq_cst) noexcept {
            m_value.store(false, order);
        }

        __host__ __device__ uint32_t *data() noexcept {
            return std::addressof(m_value);
        }

    private:
        atomic<uint32_t, Scope> m_value{false};
    };

    template<thread_scope Scope, std::regular_invocable Functor>
    __device__ inline void critical_section(atomic_flag<Scope> &lock, const Functor &f) noexcept {
        uint32_t sleep = 8;
        bool done = false;
        while (!done) {
            if (lock.test_and_set(memory_order_acquire) == false) {
                f();
                lock.clear(memory_order_release);
                done = true;
            }

            __sleep(sleep);
            if (sleep < 256) {
                sleep *= 2;
            }
        }
    }
}



template <typename T>
__device__ inline void acquire(simt::atomic<T> &lock) {
    unsigned int ns = 8;
    T expected = 0;
    while (lock.compare_exchange_strong(expected, 1, simt::memory_order_acquire) == false) {
        __sleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}

template <typename T>
__device__ inline bool try_acquire(simt::atomic<T> &lock) {
    T expected = 0;
    return lock.compare_exchange_strong(expected, 1, simt::memory_order_acquire);
}

template <typename T>
__device__ inline void release(simt::atomic<T> &lock) {
    lock.store(0, simt::memory_order_release);
}

template <typename T>
__device__ inline bool is_locked(simt::atomic<T> &lock) {
    return lock.load(simt::memory_order_relaxed) == 1;
}
