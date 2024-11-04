#pragma once

#include <new>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <string>

namespace ponos {

    template <typename T>
    class AlignedArray final {
      public:
        AlignedArray() noexcept = default;

        explicit AlignedArray(std::size_t alignment, std::size_t size)
          : m_ptr(reinterpret_cast<T *>(std::aligned_alloc(alignment, size*sizeof(T)))),
            m_size(size)
        {
            if (not m_ptr) throw std::bad_alloc();
        }

        ~AlignedArray() noexcept {
            reset();
        }

        AlignedArray(AlignedArray && rhs) noexcept
          : m_ptr(std::exchange(rhs.m_ptr, nullptr)),
            m_size(std::exchange(rhs.m_size, 0))
        {}

        auto operator=(AlignedArray && rhs) noexcept -> AlignedArray & {
            m_ptr = std::exchange(rhs.m_ptr, nullptr);
            m_size = std::exchange(rhs.m_size, 0);
            return *this;
        }

        AlignedArray(const AlignedArray &) = delete;
        auto operator=(const AlignedArray &) -> AlignedArray & = delete;

        void reset() noexcept {
            if (m_ptr) {
                std::free(m_ptr);  //NOLINT
                m_ptr = nullptr;
            }
            m_size = 0;
        }

        auto size() const noexcept -> std::size_t {
            return m_size;
        }

        auto size_bytes() const noexcept -> std::size_t {
            return m_size*sizeof(T);
        }

        auto ptr() noexcept -> T * {
            return m_ptr;
        }

        auto ptr() const noexcept -> const T * {
            return m_ptr;
        }

        auto operator[](std::size_t index) -> T & {
            return m_ptr[index];
        }

        auto operator[](std::size_t index) const -> const T & {
            return m_ptr[index];
        }

      private:
        T * m_ptr = nullptr;
        std::size_t m_size = 0;
    };

    inline auto get_xclbin_path(const std::string & dir, const std::string & name) -> std::string {
        if (dir.empty()) return name;
        if (dir == "/") return dir+name;
        if (dir[dir.length()-1] == '/') return dir+name;
        return dir+"/"+name;
    }

}
