#pragma once

#include <cstdint>
#include <array>
#include <memory>

#include "rust/cxx.h"

namespace ponos {

    struct RawArkScalar;
    struct RawArkAffine;

    class Engine12377Impl;

    class Engine12377 {
      public:
        explicit Engine12377(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir);
        ~Engine12377() noexcept = default;

        Engine12377(const Engine12377 &) = delete;
        auto operator=(const Engine12377 &) -> Engine12377 & = delete;

        void upload_bases(rust::Slice<const RawArkAffine> bases) const;

        void run(std::array<rust::Slice<const RawArkScalar>, 4> scalars, std::array<RawArkAffine, 4> & results) const;

      private:
        std::shared_ptr<Engine12377Impl> impl;
    };

    auto create_engine_bls12377(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
        -> std::unique_ptr<Engine12377>;

    class Engine12381Impl;

    class Engine12381 {
      public:
        explicit Engine12381(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir);
        ~Engine12381() noexcept = default;

        Engine12381(const Engine12381 &) = delete;
        auto operator=(const Engine12381 &) -> Engine12381 & = delete;

        void upload_bases(rust::Slice<const RawArkAffine> bases) const;

        void run(std::array<rust::Slice<const RawArkScalar>, 4> scalars, std::array<RawArkAffine, 4> & results) const;

      private:
        std::shared_ptr<Engine12381Impl> impl;
    };

    auto create_engine_bls12381(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
        -> std::unique_ptr<Engine12381>;

}
