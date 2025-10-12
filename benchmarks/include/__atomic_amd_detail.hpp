/*

NOTE: The implementation of atomic loads and stores below is written in
      accordance with the user guide for the AMDPGU backend inside LLVM,
      targeting memory model GFX10 to GFX11, i.e., from RNDA1 until RDNA3.
      See: https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx10-gfx11

*/

#pragma once

#include <hip/hip_runtime.h>
#include <cassert>

#include "__atomic_detail.hpp"

namespace simt::detail {
    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 4) {
        switch (memorder) {
            case __ATOMIC_RELEASE:
            case __ATOMIC_SEQ_CST:
                asm volatile(
                        "s_waitcnt lgkmcnt(0) vmcnt(0) \n"
                        "s_waitcnt_vscnt null, 0x0 \n"
                        :: : "memory");
                [[fallthrough]];
            case __ATOMIC_RELAXED:
                asm volatile("global_store_dword %0, %1, off \n"::"v"(ptr), "v"(val) : "memory");
                break;
            default:
                assert(false);
                break;
        }
    }

    template<typename T>
    __device__ void __atomic_store(T *ptr, T val, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 8) {
        switch (memorder) {
            case __ATOMIC_RELEASE:
            case __ATOMIC_SEQ_CST:
                asm volatile(
                        "s_waitcnt lgkmcnt(0) vmcnt(0) \n"
                        "s_waitcnt_vscnt null, 0x0 \n"
                        :: : "memory");
                [[fallthrough]];
            case __ATOMIC_RELAXED:
                asm volatile("global_store_dwordx2 %0, %1, off \n"::"v"(ptr), "v"(val) : "memory");
                break;
            default:
                assert(false);
                break;
        }
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 4) {
        typename std::remove_cv<T>::type val;
        switch (memorder) {
            case __ATOMIC_RELAXED:
                asm volatile("global_load_dword %0, %1, off glc \n": "=v"(val) : "v"(ptr) : "memory");
                break;
            case __ATOMIC_SEQ_CST:
                asm volatile(
                        "s_waitcnt lgkmcnt(0) vmcnt(0) \n"
                        "s_waitcnt_vscnt null, 0x0 \n"
                        :: : "memory");
                [[fallthrough]];
            case __ATOMIC_ACQUIRE:
            case __ATOMIC_CONSUME:
                asm volatile(
                        "global_load_dword %0, %1, off glc \n"
                        "s_waitcnt vmcnt(0) \n"
                        "buffer_gl0_inv \n"
                        : "=v"(val) : "v"(ptr) : "memory");
                break;
            default:
                assert(false);
                break;
        }
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_block_tag) requires(sizeof(T) == 8) {
        typename std::remove_cv<T>::type val;
        switch (memorder) {
            case __ATOMIC_RELAXED:
                asm volatile("global_load_dwordx2 %0, %1, off glc \n": "=v"(val) : "v"(ptr) : "memory");
                break;
            case __ATOMIC_SEQ_CST:
                asm volatile(
                        "s_waitcnt lgkmcnt(0) vmcnt(0) \n"
                        "s_waitcnt_vscnt null, 0x0 \n"
                        :: : "memory");
                [[fallthrough]];
            case __ATOMIC_ACQUIRE:
            case __ATOMIC_CONSUME:
                asm volatile(
                        "global_load_dwordx2 %0, %1, off glc \n"
                        "s_waitcnt vmcnt(0) \n"
                        "buffer_gl0_inv \n"
                        : "=v"(val) : "v"(ptr) : "memory");
                break;
            default:
                assert(false);
                break;
        }
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_device_tag) requires(sizeof(T) == 4) {
        typename std::remove_cv<T>::type val;
        switch (memorder) {
            case __ATOMIC_RELAXED:
                asm volatile("global_load_dword %0, %1, off glc dlc \n": "=v"(val) : "v"(ptr) : "memory");
                break;
            case __ATOMIC_SEQ_CST:
                asm volatile(
                        "s_waitcnt lgkmcnt(0) vmcnt(0) \n"
                        "s_waitcnt_vscnt null, 0x0 \n"
                        :: : "memory");
                [[fallthrough]];
            case __ATOMIC_ACQUIRE:
            case __ATOMIC_CONSUME:
                asm volatile(
                        "global_load_dword %0, %1, off glc dlc \n"
                        "s_waitcnt vmcnt(0) \n"
                        "buffer_gl1_inv \n"
                        "buffer_gl0_inv \n"
                        : "=v"(val) : "v"(ptr) : "memory");
                break;
            default:
                assert(false);
                break;
        }
        return val;
    }

    template<typename T>
    __device__ T __atomic_load(T *ptr, int memorder, __thread_scope_device_tag) requires(sizeof(T) == 8) {
        typename std::remove_cv<T>::type val;
        switch (memorder) {
            case __ATOMIC_RELAXED:
                asm volatile("global_load_dwordx2 %0, %1, off glc dlc \n": "=v"(val) : "v"(ptr) : "memory");
                break;
            case __ATOMIC_SEQ_CST:
                asm volatile(
                        "s_waitcnt lgkmcnt(0) vmcnt(0) \n"
                        "s_waitcnt_vscnt null, 0x0 \n"
                        :: : "memory");
                [[fallthrough]];
            case __ATOMIC_ACQUIRE:
            case __ATOMIC_CONSUME:
                asm volatile(
                        "global_load_dwordx2 %0, %1, off glc dlc \n"
                        "s_waitcnt vmcnt(0) \n"
                        "buffer_gl1_inv \n"
                        "buffer_gl0_inv \n"
                        : "=v"(val) : "v"(ptr) : "memory");
                break;
            default:
                assert(false);
                break;
        }
        return val;
    }
}
