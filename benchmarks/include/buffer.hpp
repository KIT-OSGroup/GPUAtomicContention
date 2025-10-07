#ifndef INCLUDED___gpu_nvme___buffer_hpp
#define INCLUDED___gpu_nvme___buffer_hpp

#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "utility.hpp"

enum class Target {
    Host,
    Device,
    Both
};

/// \brief generic buffer between host and device
class Buffer {
public:
    __host__ constexpr Buffer() = default;

    __host__ explicit Buffer(std::size_t size, Target allocate)
        : m_size(size)
    {
        if (allocate == Target::Host || allocate == Target::Both) {
            void *buffer = std::malloc(size);
            std::memset(buffer, 0, size);
            m_host_buffer = std::shared_ptr<void>(buffer, [](auto ptr) {
                // std::cout << "~Buffer()" << std::endl;
                std::free(ptr);
            });
        }

        if (allocate == Target::Device || allocate == Target::Both) {
            void *device_ptr = nullptr;
            HIP_CHECK(hipMalloc(&device_ptr, size));
            HIP_CHECK(hipMemset(device_ptr, 0, size));
            m_device_buffer = std::shared_ptr<void>(device_ptr, [](auto ptr) {
                // std::cout << "~Buffer()" << std::endl;
                HIP_CHECK(hipFree(ptr));
            });
        }
    }

    __host__ ~Buffer() = default;
    __host__ Buffer(const Buffer &other) = default;
    __host__ Buffer(Buffer &&other) noexcept = default;
    __host__ Buffer &operator=(const Buffer &other) = default;
    __host__ Buffer &operator=(Buffer &&other) noexcept = default;

    /// \brief Copies data from source to destination
    //! \attention Does not call the destructor of the overwritten object
    __host__ void copy(Target dst)
    {
        assert((dst != Target::Both));

        void *src_ptr = dst == Target::Host ? m_device_buffer.get() : m_host_buffer.get();

        copy(dst, src_ptr);
    }

    __host__ void copy(Target dst, void *src)
    {
        assert((dst != Target::Both));

        void *dst_ptr = dst == Target::Host ? m_host_buffer.get() : m_device_buffer.get();
        void *src_ptr = src;
        hipMemcpyKind kind = dst == Target::Host ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice;

        HIP_CHECK(hipMemcpy(dst_ptr, src_ptr, m_size, kind));
    }

    __host__ void *get_host_buffer() { return m_host_buffer.get(); }
    __host__ const void *get_host_buffer() const { return m_host_buffer.get(); }

    __host__ void *get_device_buffer() { return m_device_buffer.get(); }
    __host__ const void *get_device_buffer() const { return m_device_buffer.get(); }

    __host__ std::size_t size() const { return m_size; }

private:
    std::size_t m_size{0};
    std::shared_ptr<void> m_host_buffer{nullptr};
    std::shared_ptr<void> m_device_buffer{nullptr};
};

/// \brief buffer with type
//! \attention objects will be left uninitialized
template <typename T>
class TypedBuffer {
public:
    __host__ constexpr TypedBuffer() = default;
    __host__ explicit TypedBuffer(Target allocate)
        : m_buffer(sizeof(T), allocate)
    {}

    __host__ ~TypedBuffer() = default;
    __host__ TypedBuffer(const TypedBuffer &other) = delete;
    __host__ TypedBuffer(TypedBuffer &&other) noexcept = default;
    __host__ TypedBuffer &operator=(const TypedBuffer &other) = delete;
    __host__ TypedBuffer &operator=(TypedBuffer &&other) noexcept = default;

    __host__ void copy(Target dst) { m_buffer.copy(dst); }
    __host__ void copy(Target dst, T *src) { m_buffer.copy(dst, src); }

    __host__ T *get_host_buffer() { return reinterpret_cast<T *>(m_buffer.get_host_buffer()); }
    __host__ const T *get_host_buffer() const { return reinterpret_cast<T *>(m_buffer.get_host_buffer()); }

    __host__ T *get_device_buffer() { return reinterpret_cast<T *>(m_buffer.get_device_buffer()); }
    __host__ const T *get_device_buffer() const { return reinterpret_cast<T *>(m_buffer.get_device_buffer()); }

private:
    Buffer m_buffer{};
};

#endif // INCLUDED___gpu_nvme___buffer_hpp
