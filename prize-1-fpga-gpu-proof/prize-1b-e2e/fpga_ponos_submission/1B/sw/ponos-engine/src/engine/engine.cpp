//////////////////////////////////////////////////////////////////////////////////
// (C) Copyright Ponos Technology 2024
// All Rights Reserved
// *** Ponos Technology Confidential ***
// Description:
// Owner: Dragan Milenkovic <dragan@ponos.technology>
//////////////////////////////////////////////////////////////////////////////////

#include "engine.hpp"
#include "math.hpp"

#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#include <cassert>
#include <iostream>

#define ENABLE_TRACE 0

namespace ponos {

    inline constexpr int WINDOW_BITS = 13;
    inline constexpr int WINDOW_COUNT = 20;
    inline constexpr int FPGA_WINDOW_COUNT = 2*10;

    inline constexpr int BUCKET_COUNT = 1 << (WINDOW_BITS-1);

    struct Bls12381Field {
        static const mpz_class modulo;
        static constexpr std::size_t word_count = 6;
    };

    const mpz_class Bls12381Field::modulo {"4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787"};

    using Fp12381 = FieldNumber<Bls12381Field>;

    static const auto ZERO = Fp12381("0");
    static const auto ONE  = Fp12381("1");
    static const auto TWO  = Fp12381("2");

    static const auto R    = mpz_class("39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306816");
    static const auto R_INV = mpz_class("3231460744492646417066832100176244795738767926513225105051837195607029917124509527734802654356338138714468589979680");

    static auto into_montgomery(const mpz_class & x) -> Fp12381 {
        return Fp12381((x * R) % Fp12381::field::modulo);
    }

    static auto from_montgomery(const mpz_class & x) -> Fp12381 {
        return Fp12381((x * R_INV) % Fp12381::field::modulo);
    }

    struct Projective12381 {
        Fp12381 X, Y, ZZ, ZZZ;
    };

    struct Affine12381 {
        Fp12381 x, y;
    };

    static auto ZERO_PROJECTIVE12381 = Projective12381 {ONE, ONE, ZERO, ZERO};

    struct FpgaProjective12381 {
        RawScalar384Padded X, Y, ZZ, ZZZ;
    };

    static auto is_zero(const Projective12381 & p) -> bool {
        return p.ZZ == ZERO;
    }

    static auto is_fpga_zero(const FpgaProjective12381 & p) -> bool {
        static constexpr uint64_t fpga_zero = 0x1fffffffffffffffULL;

        return p.X.value.repr[5] >= fpga_zero;
    }

    static auto is_fpga_zero(const NBase384 & p) -> bool {
        static constexpr uint64_t fpga_zero = 0x1fffffffffffffffULL;

        return p.x.repr[5] >= fpga_zero;
    }

    static auto convert_to_fpga_input(const RawArkAffine & src) -> NBase384 {
        NBase384 dst {};

        if (not src.infinity) {
            auto x = from_montgomery(mpz_from(src.x));
            auto y = from_montgomery(mpz_from(src.y));
            x.copy_to(dst.x.repr);
            y.copy_to(dst.y.repr);
        }
        return dst;
    }

    static auto convert_to_final_output_mont(const Projective12381 & src) -> RawArkAffine {
        RawArkAffine dst {};

        if (is_zero(src)) {
            into_montgomery(ONE.value()).copy_to(dst.x);
            into_montgomery(ONE.value()).copy_to(dst.y);
            dst.infinity = true;
        }
        else {
            mpz_class ZZ_inv = modulo_inverse<mpz_class>(src.ZZ.value(), Fp12381::field::modulo);
            mpz_class ZZZ_inv = modulo_inverse<mpz_class>(src.ZZZ.value(), Fp12381::field::modulo);

            Fp12381 x = src.X * ZZ_inv;
            Fp12381 y = src.Y * ZZZ_inv;

            into_montgomery(x.value()).copy_to(dst.x);
            into_montgomery(y.value()).copy_to(dst.y);
            dst.infinity = false;
        }
        return dst;
    }

    static void double_assign_proj(Projective12381 & r) {
        if (is_zero(r)) {
            return;
        }

        Fp12381 U = r.Y*2;
        Fp12381 V = U*U;
        Fp12381 W = U*V;
        Fp12381 S = r.X*V;
        Fp12381 M = r.X*r.X*3;
        r.X = M*M-S*2;
        r.Y = M*(S-r.X)-W*r.Y;
        r.ZZ = V*r.ZZ;
        r.ZZZ = W*r.ZZZ;
    }

    static void add_assign_proj(Projective12381 & r, const Projective12381 & q) {
        if (is_zero(q)) {
            return;
        }
        if (is_zero(r)) {
            r = q;
            return;
        }

        Fp12381 U1 = r.X*q.ZZ;
        Fp12381 U2 = q.X*r.ZZ;
        Fp12381 S1 = r.Y*q.ZZZ;
        Fp12381 S2 = q.Y*r.ZZZ;
        Fp12381 P = U2-U1;
        Fp12381 R = S2-S1;
        if (P == ZERO) {
            if (R == ZERO) {
                double_assign_proj(r);
            }
            else {
                r = ZERO_PROJECTIVE12381;
            }
            return;
        }
        Fp12381 PP = P*P;
        Fp12381 PPP = P*PP;
        Fp12381 Q = U1*PP;
        r.X = R*R-PPP-TWO*Q;
        r.Y = R*(Q-r.X)-S1*PPP;
        r.ZZ = r.ZZ*q.ZZ*PP;
        r.ZZZ = r.ZZZ*q.ZZZ*PPP;
    }

    static void add_assign_fpga(Projective12381 & r, const FpgaProjective12381 & q) {
        if (is_fpga_zero(q)) return;

        add_assign_proj(r, {Fp12381(q.X.value.repr), Fp12381(q.Y.value.repr), Fp12381(q.ZZ.value.repr), Fp12381(q.ZZZ.value.repr)});
    }

    static auto aggregate_window(int n, const FpgaProjective12381 * a)
        -> Projective12381
    {
        Projective12381 res = ZERO_PROJECTIVE12381;
        Projective12381 tmp = res;

        for (int i=n-1; i>=0; --i) {
            add_assign_fpga(tmp, a[i]);
            add_assign_proj(res, tmp);
        }

        return res;
    }

    template <typename T>
    struct XrtBoAndPtr {
        xrt::bo buffer_object;
        T * ptr = nullptr;
    };

    class VitisProjRunner final {
      public:
        explicit VitisProjRunner(std::size_t max_elements, rust::Str xclbin_path)
          : m_max_elements(max_elements)
        {
#if ENABLE_TRACE
            std::cout << "[trace] initializing MSM BLS12-381 Vitis engine\n";
#endif

#if ENABLE_TRACE
            std::cout << "[trace] initializing fpga device\n";
#endif
            m_device = xrt::device(0);

#if ENABLE_TRACE
            std::cout << "[trace] loading xclbin\n";
#endif
            auto xclbin_uuid = m_device.load_xclbin(static_cast<std::string>(xclbin_path));
            m_kernel = xrt::kernel(m_device, xclbin_uuid, "krnl_msm_381");

            {
                std::size_t size_bytes = m_max_elements * sizeof(RawArkCombo12381);
#if ENABLE_TRACE
                std::cout << "[trace] initializing data buffer (size=" << size_bytes << ")\n";
#endif
                m_data_buffer.buffer_object  = xrt::bo(m_device, size_bytes, xrt::bo::flags::host_only, m_kernel.group_id(0));
                m_data_buffer.ptr = m_data_buffer.buffer_object.map<RawArkCombo12381 *>();
                memset(m_data_buffer.ptr, 0, size_bytes);
            }
            {
                std::size_t size_bytes = FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381);
#if ENABLE_TRACE
                std::cout << "[trace] initializing bucket buffer (size=" << size_bytes << ")\n";
#endif
                m_bucket_buffer.buffer_object  = xrt::bo(m_device, size_bytes, xrt::bo::flags::host_only, m_kernel.group_id(1));
                m_bucket_buffer.ptr = m_bucket_buffer.buffer_object.map<FpgaProjective12381 *>();
                memset(m_bucket_buffer.ptr, 0, size_bytes);
            }

#if ENABLE_TRACE
            std::cout << "[trace] initialize bucket memory (workaround)\n";
#endif
            {
                auto r =
                    m_kernel(
                        m_data_buffer.buffer_object,
                        m_bucket_buffer.buffer_object,
                        8192 * sizeof(RawArkCombo12381) / 64,
                        FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381) / 64,
                        4
                    );
                r.wait();
            }
        }

        ~VitisProjRunner() noexcept = default;

        VitisProjRunner(const VitisProjRunner &) = delete;
        auto operator=(const VitisProjRunner &) -> VitisProjRunner & = delete;

        void run_with_conversion(rust::Slice<const RawScalar256> scalars, rust::Slice<const RawArkAffine> bases, RawArkAffine & result) {
            auto n = std::min(scalars.size(), bases.size());
            assert(n > 0 and n < m_max_elements);

            {
#if ENABLE_TRACE
                std::cout << "[trace] copying and converting input\n";
                auto t1 = std::chrono::high_resolution_clock::now();
#endif

                #pragma omp parallel for num_threads(8)
                for (size_t i=0; i<n; ++i) {
                    const auto & base = bases[i];

                    if (base.infinity) {
                        m_data_buffer.ptr[i].scalar = {};
                    }
                    else {
                        m_data_buffer.ptr[i].scalar = scalars[i];
                        m_data_buffer.ptr[i].base = convert_to_fpga_input(base);
                    }
                }

#if ENABLE_TRACE
                auto t2 = std::chrono::high_resolution_clock::now();
                std::cout << "[trace] copy and convert TIME: " << (t2-t1).count() << " ns\n";
#endif
            }

            {
#if ENABLE_TRACE
                std::cout << "[trace] starting kernel (isize=" << (n * sizeof(RawArkCombo12381) / 64) << ", osize=" << (FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381) / 64) << ")\n";
                auto t1 = std::chrono::high_resolution_clock::now();
#endif

                auto r =
                    m_kernel(
                        m_data_buffer.buffer_object,
                        m_bucket_buffer.buffer_object,
                        n * sizeof(RawArkCombo12381) / 64,
                        FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381) / 64,
                        4
                    );

#if ENABLE_TRACE
                std::cout << "[trace] waiting for kernel\n";
#endif
                r.wait();

#if ENABLE_TRACE
                auto t2 = std::chrono::high_resolution_clock::now();
                std::cout << "[trace] kernel finished TIME: " << (t2-t1).count() << " ns\n";
#endif
            }

            aggregate(result);
        }

        void run(rust::Slice<const RawScalar256> scalars, rust::Slice<const NBase384> bases, RawArkAffine & result) {
            auto n = std::min(scalars.size(), bases.size());
            assert(n > 0 and n < m_max_elements);

            {
#if ENABLE_TRACE
                std::cout << "[trace] copying input\n";
                auto t1 = std::chrono::high_resolution_clock::now();
#endif

                #pragma omp parallel for num_threads(8)
                for (size_t i=0; i<n; ++i) {
                    const auto & base = bases[i];

                    if (is_fpga_zero(base)) {
                        m_data_buffer.ptr[i].scalar = {};
                    }
                    else {
                        m_data_buffer.ptr[i].scalar = scalars[i];
                        m_data_buffer.ptr[i].base = base;
                    }
                }

#if ENABLE_TRACE
                auto t2 = std::chrono::high_resolution_clock::now();
                std::cout << "[trace] copy TIME: " << (t2-t1).count() << " ns\n";
#endif
            }

            {
#if ENABLE_TRACE
                std::cout << "[trace] starting kernel (isize=" << (n * sizeof(RawArkCombo12381) / 64) << ", osize=" << (FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381) / 64) << ")\n";
                auto t1 = std::chrono::high_resolution_clock::now();
#endif

                auto r =
                    m_kernel(
                        m_data_buffer.buffer_object,
                        m_bucket_buffer.buffer_object,
                        n * sizeof(RawArkCombo12381) / 64,
                        FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381) / 64,
                        4
                    );

#if ENABLE_TRACE
                std::cout << "[trace] waiting for kernel\n";
#endif
                r.wait();

#if ENABLE_TRACE
                auto t2 = std::chrono::high_resolution_clock::now();
                std::cout << "[trace] kernel finished TIME: " << (t2-t1).count() << " ns\n";
#endif
            }

            aggregate(result);
        }

      private:
        void aggregate(RawArkAffine & result) {
#if ENABLE_TRACE
            std::cout << "[trace] aggregation started\n";
            auto t1 = std::chrono::high_resolution_clock::now();
#endif

            std::array<Projective12381, WINDOW_COUNT> partial_sums;

            {
                #pragma omp parallel for num_threads(8)
                for (int iwindow=0; iwindow<WINDOW_COUNT; ++iwindow) {
                    std::ptrdiff_t offset = BUCKET_COUNT * iwindow;
                    partial_sums[iwindow] = aggregate_window(BUCKET_COUNT, m_bucket_buffer.ptr+offset);
                }
            }

            Projective12381 proj_result = partial_sums[WINDOW_COUNT-1];
            for (int iwindow=WINDOW_COUNT-2; iwindow>=0; --iwindow) {
                for (int i=0; i<WINDOW_BITS; ++i) {
                    double_assign_proj(proj_result);
                }

                add_assign_proj(proj_result, partial_sums[iwindow]);
            }

#if ENABLE_TRACE
            auto t2 = std::chrono::high_resolution_clock::now();
#endif
            result = convert_to_final_output_mont(proj_result);

#if ENABLE_TRACE
            auto t3 = std::chrono::high_resolution_clock::now();
            std::cout << "[trace] aggregation finished TIME: " << (t2-t1).count() << " ns, " << (t3-t2).count() << " ns\n";
#endif
        }

        const std::size_t m_max_elements;

        xrt::device m_device;
        xrt::kernel m_kernel;

        XrtBoAndPtr<RawArkCombo12381> m_data_buffer;
        XrtBoAndPtr<FpgaProjective12381> m_bucket_buffer;
    };

    static std::unique_ptr<VitisProjRunner> vitis_runner;

    static void init_vitis() {
        if (not vitis_runner) {
            vitis_runner = std::make_unique<VitisProjRunner>(1 << 22, "msm-4coord.xclbin");
        }
    }

    void run_fpga_msm_vitis_mont(rust::Slice<const RawScalar256> scalars, rust::Slice<const RawArkAffine> bases, RawArkAffine & result) {
        init_vitis();

        vitis_runner->run_with_conversion(scalars, bases, result);
    }

    void run_fpga_msm_vitis_normal(rust::Slice<const RawScalar256> scalars, rust::Slice<const NBase384> bases, RawArkAffine & result) {
        init_vitis();

        vitis_runner->run(scalars, bases, result);
    }

}
