#pragma once

#include <gmpxx.h>
#include <array>
#include <cstdint>

#include <cassert>

namespace ponos {

    template <std::size_t N>
    inline auto mpz_from(const std::array<uint64_t, N> & data)
        -> mpz_class
    {
        mpz_class x;
        mpz_import(x.get_mpz_t(), data.size(), -1, sizeof(uint64_t), 0, 0, data.data());
        return x;
    }

    template <std::size_t N>
    inline void mpz_to(const mpz_class & x, std::array<uint64_t, N> & data) {
        size_t n = 0;
        mpz_export(data.data(), &n, -1, sizeof(uint64_t), 0, 0, x.get_mpz_t());
        assert(n <= N);

        if (n < 6) {
            memset(data.data()+n, 0, (N-n)*sizeof(uint64_t));
        }
    }

    template <typename F>
    class FieldNumber {
      public:
        using field = F;

        FieldNumber()
          : m_value()
        {}

        FieldNumber(mpz_class value)
          : m_value(std::move(value))
        {
            assert(m_value >= 0 and m_value < field::modulo);
        }

        template <std::size_t N>
        explicit FieldNumber(const std::array<uint64_t, N> & raw)
          : FieldNumber(mpz_from(raw))
        {
            static_assert(N >= field::word_count);
        }

        explicit FieldNumber(const std::string & s)
          : FieldNumber(mpz_class(s))
        {}

        FieldNumber(const FieldNumber & rhs) = default;
        auto operator=(const FieldNumber & rhs) -> FieldNumber & = default;

        FieldNumber(FieldNumber && rhs) noexcept = default;
        auto operator=(FieldNumber && rhs) noexcept -> FieldNumber & = default;

        ~FieldNumber() noexcept = default;

        auto value() const -> const mpz_class & {
            return m_value;
        }

        template <std::size_t N>
        void copy_to(std::array<uint64_t, N> & raw) const {
            static_assert(N >= field::word_count);
            mpz_to(m_value, raw);
        }

        friend auto operator+(const FieldNumber & lhs, const FieldNumber & rhs) -> FieldNumber {
            return FieldNumber((lhs.m_value + rhs.m_value) % field::modulo);
        }

        friend auto operator+(const FieldNumber & lhs, const mpz_class & rhs) -> FieldNumber {
            return FieldNumber((lhs.m_value + rhs) % field::modulo);
        }

        friend auto operator-(const FieldNumber & lhs, const FieldNumber & rhs) -> FieldNumber {
            return FieldNumber((field::modulo + lhs.m_value - rhs.m_value) % field::modulo);
        }

        friend auto operator-(const FieldNumber & lhs, const mpz_class & rhs) -> FieldNumber {
            return FieldNumber((field::modulo + lhs.m_value - rhs) % field::modulo);
        }

        friend auto operator*(const FieldNumber & lhs, const FieldNumber & rhs) -> FieldNumber {
            return FieldNumber((lhs.m_value * rhs.m_value) % field::modulo);
        }

        friend auto operator*(const FieldNumber & lhs, const mpz_class & rhs) -> FieldNumber {
            return FieldNumber((lhs.m_value * rhs) % field::modulo);
        }

        friend auto operator==(const FieldNumber & lhs, const FieldNumber & rhs) noexcept -> bool {
            return lhs.m_value == rhs.m_value;
        }

        friend auto operator!=(const FieldNumber & lhs, const FieldNumber & rhs) noexcept -> bool {
            return lhs.m_value != rhs.m_value;
        }

      private:
        mpz_class m_value;
    };

    template <typename Number>
    inline auto modulo_inverse(Number a, const Number & m) -> Number {
        Number b = m;
        Number x = 0, y = 1, lastx = 1, lasty = 0;
        Number temp, quotient;

        while (b != 0) {
            quotient = a / b;
            temp = b;
            b = a % b;
            a = temp;

            temp = x;
            x = lastx - quotient * x;
            lastx = temp;

            temp = y;
            y = lasty - quotient * y;
            lasty = temp;
        }

        assert(a == 1);

        // Ensure the result is positive
        return (lastx % m + m) % m;
    }

}
