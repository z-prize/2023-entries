#include "engine.hpp"
#include "ponos-msm-fpga/src/engine.rs.h"

#include "fpga.hpp"
#include "math.hpp"
#include "utils.hpp"
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>

#define ENABLE_TRACE 0

namespace ponos {

    struct Bls12381Field {
        static const mpz_class modulo;
        static constexpr std::size_t word_count = 6;
    };

    const mpz_class Bls12381Field::modulo {"4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787"};

    using Fp12381 = FieldNumber<Bls12381Field>;

    struct FpgaFp12381 {
        std::array<uint64_t, 6> value;
        std::array<uint64_t, 2> padding;
    };

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

    struct FpgaAffine12381 {
        FpgaFp12381 x;
        FpgaFp12381 y;
    };

    struct Projective12381 {
        Fp12381 X, Y, ZZ, ZZZ;
    };

    static auto ZERO_PROJECTIVE12381 = Projective12381 {ONE, ONE, ZERO, ZERO};

    struct FpgaProjective12381 {
        FpgaFp12381 X, Y, ZZ, ZZZ;
    };

    static auto is_zero(const Projective12381 & p) -> bool {
        return p.ZZ == ZERO;
    }

    static auto is_fpga_zero(const FpgaProjective12381 & p) -> bool {
        static constexpr uint64_t fpga_zero = 0x1fffffffffffffffULL;

        return p.X.value[5] >= fpga_zero;
    }

    static auto convert_to_fpga_input(const RawArkAffine & src) -> FpgaAffine12381 {
        FpgaAffine12381 dst {};

        if (not src.infinity) {
            auto x = from_montgomery(mpz_from(src.x));
            auto y = from_montgomery(mpz_from(src.y));
            x.copy_to(dst.x.value);
            y.copy_to(dst.y.value);
        }
        return dst;
    }

    static auto convert_to_final_output_mont(const Projective12381 & src) -> RawArkAffine {
        RawArkAffine dst {};

        if (is_zero(src)) [[unlikely]] {
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

    static void double_assign(Projective12381 & r) {
        if (is_zero(r)) [[unlikely]] {
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

    static void add_assign(Projective12381 & r, const Projective12381 & q) {
        if (is_zero(q)) [[unlikely]] {
            return;
        }
        if (is_zero(r)) [[unlikely]] {
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
                double_assign(r);
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

    static void add_assign(Projective12381 & r, const FpgaProjective12381 & q) {
        if (is_fpga_zero(q)) return;

        add_assign(r, {Fp12381(q.X.value), Fp12381(q.Y.value), Fp12381(q.ZZ.value), Fp12381(q.ZZZ.value)});
    }

    class Engine12381Impl {
      public:
        static constexpr int SCALAR_BUFFER_COUNT = 2;
        static constexpr int BUCKET_BUFFER_COUNT = 2;

        static constexpr int WINDOW_BITS = 13;
        static constexpr int WINDOW_COUNT = 20;
        static constexpr int FPGA_WINDOW_COUNT = 2*10;

        static constexpr int BUCKET_COUNT = 1 << (WINDOW_BITS-1);

        explicit Engine12381Impl(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
          : m_elements_per_round(1 << round_pow),
            m_element_count(element_count),
            m_zeros(element_count, false)
        {
#if ENABLE_TRACE
            std::cout << "initializing MSM BLS12-381 engine\n";
#endif

            if ((element_count % m_elements_per_round) != 0) {
                throw std::invalid_argument("element count must be a multiple of elements per round");
            }

            m_round_count = int(m_element_count / m_elements_per_round);
            if (m_round_count < 2) {
                //  Because kernel doesn't support it.
                throw std::invalid_argument("element count must fill at least two rounds");
            }

#if ENABLE_TRACE
            std::cout << "initializing fpga device\n";
#endif
            m_device = xrt::device(0);

#if ENABLE_TRACE
            std::cout << "loading xclbin\n";
#endif
            auto xclbin_uuid = m_device.load_xclbin(get_xclbin_path(static_cast<std::string>(xclbin_dir), "12381.xclbin"));
            m_kernel = xrt::kernel(m_device, xclbin_uuid, "krnl_msm_381");

            {
                m_base_buffers_a.resize(m_round_count);
                m_base_buffers_b.resize(m_round_count);

                std::size_t size_bytes = m_elements_per_round * sizeof(FpgaAffine12381);
#if ENABLE_TRACE
                std::cout << "initializing base buffers (size=" << size_bytes << ")\n";
#endif
                for (int round=0; round<m_round_count; ++round) {
                    m_base_buffers_a[round] = xrt::bo(m_device, size_bytes, xrt::bo::flags::device_only, m_kernel.group_id(4));
                    m_base_buffers_b[round] = xrt::bo(m_device, size_bytes, xrt::bo::flags::device_only, m_kernel.group_id(8));
                }
            }
            {
                std::size_t size_bytes = m_elements_per_round * sizeof(RawArkScalar);
#if ENABLE_TRACE
                std::cout << "initializing scalar buffers (size=" << size_bytes << ")\n";
#endif
                for (int i=0; i<SCALAR_BUFFER_COUNT; ++i) {
                    m_scalar_buffers[i].buffer_object  = xrt::bo(m_device, size_bytes, xrt::bo::flags::host_only, m_kernel.group_id(0));
                    m_scalar_buffers[i].ptr = m_scalar_buffers[i].buffer_object.map<RawArkScalar *>();
                    memset(m_scalar_buffers[i].ptr, 0, size_bytes);
                }
            }
            {
                std::size_t size_bytes = FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381);
#if ENABLE_TRACE
                std::cout << "initializing bucket buffers (size=" << size_bytes << ")\n";
#endif
                for (int i=0; i<BUCKET_BUFFER_COUNT; ++i) {
                    m_bucket_buffers[i].buffer_object  = xrt::bo(m_device, size_bytes, xrt::bo::flags::host_only, m_kernel.group_id(1));
                    m_bucket_buffers[i].ptr = m_bucket_buffers[i].buffer_object.map<FpgaProjective12381 *>();
                    memset(m_bucket_buffers[i].ptr, 0, size_bytes);
                }
            }
        }

        ~Engine12381Impl() noexcept = default;

        Engine12381Impl(const Engine12381Impl &) = delete;
        auto operator=(const Engine12381Impl &) -> Engine12381Impl & = delete;

        void upload_bases(rust::Slice<const RawArkAffine> bases) {
            if (bases.size() < m_element_count) throw std::invalid_argument("not enough bases given");

            auto converted = AlignedArray<FpgaAffine12381>(4096, m_elements_per_round);

#if ENABLE_TRACE
            std::cout << "converting and uploading bases to fpga\n";
            auto t1 = std::chrono::steady_clock::now();
#endif

            for (int round=0; round<m_round_count; ++round) {
                std::size_t first_element = m_elements_per_round * round;

                #pragma omp parallel for num_threads(8)
                for (size_t i=0; i<m_elements_per_round; ++i) {
                    unsigned element_index = first_element + i;
                    const auto & base = bases[element_index];

                    if (base.infinity) [[unlikely]] {
#if ENABLE_TRACE
                        std::cerr << "WARNING: Detected an INFINITY point at index " << element_index << "\n";
#endif
                        converted[i] = {};
                        m_zeros[element_index] = true;
                    }
                    else {
                        converted[i] = convert_to_fpga_input(base);
                    }
                }

                m_base_buffers_a[round].write(converted.ptr());
                m_base_buffers_b[round].write(converted.ptr());
            }

#if ENABLE_TRACE
            auto t2 = std::chrono::steady_clock::now();
            std::cout << "bases converted and uploaded TIME: " << (t2-t1).count() << "ns\n";
#endif
        }

        void run(std::array<rust::Slice<const RawArkScalar>, 4> scalars, std::array<RawArkAffine, 4> & results) {
            int vec_count = static_cast<int>(scalars.size());

            if (vec_count > 0) {
                copy_scalars(scalars, 0, 0);
            }

            for (int ia=0; ia<vec_count; ++ia) {
                std::thread t_agg;

                if (ia >= 1) {
                    t_agg = std::thread([&] {
                        aggregate(results, ia-1, false);
                    });
                }

                for (int ib=0; ib<m_round_count; ++ib) {
                    std::thread t_kernel {[&] {
                        run_kernel(ia, ib);
                    }};

                    if (ib < m_round_count-1) {
                        copy_scalars(scalars, ia, ib+1);
                    }
                    else if (ia < vec_count - 1) {
                        copy_scalars(scalars, ia+1, 0);
                    }

                    t_kernel.join();
                }

                if (t_agg.joinable()) t_agg.join();
            }

            if (vec_count > 0) {
                aggregate(results, vec_count-1, true);
            }
        }

      private:
        void copy_scalars(const std::array<rust::Slice<const RawArkScalar>, 4> & scalars, int vec_index, int round) {
            const auto & scalar_vec = scalars[vec_index];
            int scalar_buffer_index = round % SCALAR_BUFFER_COUNT;
            auto first_element = round * m_elements_per_round;

#if ENABLE_TRACE
            std::cout << "copying a chunk of scalars for vector " << vec_index << " round " << round << "\n";
            auto t1 = std::chrono::steady_clock::now();
#endif

            auto ptr = m_scalar_buffers[scalar_buffer_index].ptr;

            std::memcpy(ptr, scalar_vec.data()+first_element, m_elements_per_round*sizeof(RawArkScalar));
            for (std::size_t i=0; i<m_elements_per_round; ++i) {
                if (m_zeros[first_element + i]) {
#if ENABLE_TRACE
                    std::cout << "ZERO PATCH AT " << first_element << " + " << i << "\n";
#endif
                    ptr[i] = {};
                }
            }
            m_scalar_buffers[scalar_buffer_index].buffer_object.sync(XCL_BO_SYNC_BO_TO_DEVICE);

#if ENABLE_TRACE
            auto t2 = std::chrono::steady_clock::now();
            std::cout << "copying finished TIME: " << (t2-t1).count() << "ns\n";
#endif
        }

        void run_kernel(int vec_index, int round) {
            uint32_t mode = resolve_kernel_mode(vec_index, round, m_round_count);
            int bucket_buffer_index = vec_index % BUCKET_BUFFER_COUNT;
            int scalar_buffer_index = round % SCALAR_BUFFER_COUNT;

#if ENABLE_TRACE
            std::cout << "starting kernel for vector " << vec_index << " round " << round << " (mode=" << mode << ")\n";
            auto t1 = std::chrono::steady_clock::now();
#endif

            auto r =
                m_kernel(
                    m_scalar_buffers[scalar_buffer_index].buffer_object,
                    m_bucket_buffers[bucket_buffer_index].buffer_object,
                    m_elements_per_round * sizeof(RawArkScalar) / 64,
                    FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaProjective12381) / 64,

                    m_base_buffers_a[round],
                    nullptr,
                    m_elements_per_round * sizeof(FpgaAffine12381) / 64,
                    0,

                    m_base_buffers_b[round],
                    nullptr,
                    m_elements_per_round * sizeof(FpgaAffine12381) / 64,
                    0,

                    mode
                );

#if ENABLE_TRACE
            std::cout << "waiting for kernel\n";
#endif
            r.wait();

#if ENABLE_TRACE
            auto t2 = std::chrono::steady_clock::now();
            std::cout << "kernel finished TIME: " << (t2-t1).count() << "ns\n";
#endif
        }

        void aggregate(std::array<RawArkAffine, 4> & results, int vec_index, bool full_speed) {
            int bucket_buffer_index = vec_index % BUCKET_BUFFER_COUNT;

#if ENABLE_TRACE
            std::cout << "fetching and aggregating buckets for vector " << vec_index << "\n";
#endif
            m_bucket_buffers[bucket_buffer_index].buffer_object.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            std::array<Projective12381, WINDOW_COUNT> partial_sums;

#if ENABLE_TRACE
            auto t1 = std::chrono::steady_clock::now();
#endif

            if (full_speed) {
                #pragma omp parallel for num_threads(8)
                for (int iwindow=0; iwindow<WINDOW_COUNT; ++iwindow) {
                    std::ptrdiff_t offset = BUCKET_COUNT * iwindow;
                    partial_sums[iwindow] = aggregate(BUCKET_COUNT, m_bucket_buffers[bucket_buffer_index].ptr+offset);
                }
            }
            else {
                #pragma omp parallel for num_threads(2)
                for (int iwindow=0; iwindow<WINDOW_COUNT; ++iwindow) {
                    std::ptrdiff_t offset = BUCKET_COUNT * iwindow;
                    partial_sums[iwindow] = aggregate(BUCKET_COUNT, m_bucket_buffers[bucket_buffer_index].ptr+offset);
                }
            }

#if ENABLE_TRACE
            auto t2 = std::chrono::steady_clock::now();
#endif

            Projective12381 result = partial_sums[WINDOW_COUNT-1];
            for (int iwindow=WINDOW_COUNT-2; iwindow>=0; --iwindow) {
                for (int i=0; i<WINDOW_BITS; ++i) {
                    double_assign(result);
                }

                add_assign(result, partial_sums[iwindow]);
            }

#if ENABLE_TRACE
            auto t3 = std::chrono::steady_clock::now();
            std::cout << "aggregated TIME: " << (t2-t1).count() << "ns, " << (t3-t2).count() << "ns\n";
#endif

            results[vec_index] = convert_to_final_output_mont(result);
        }

        static auto aggregate(int n, const FpgaProjective12381 * a)
            -> Projective12381
        {
            Projective12381 res = ZERO_PROJECTIVE12381;
            Projective12381 tmp = res;

            for (int i=n-1; i>=0; --i) {
                add_assign(tmp, a[i]);
                add_assign(res, tmp);
            }

            return res;
        }

        static auto resolve_kernel_mode([[maybe_unused]] int vec, int round, int total_rounds) -> uint32_t {
            if (round == total_rounds-1) {
                return 4;
            }
            else if (round == 0 /*and vec == 0*/) {
                return 1;
            }
            else {
                return 2;
            }
        }

        const std::size_t m_elements_per_round;
        const std::size_t m_element_count;
        int m_round_count;

        xrt::device m_device;
        xrt::kernel m_kernel;

        std::vector<xrt::bo> m_base_buffers_a;
        std::vector<xrt::bo> m_base_buffers_b;

        std::array<XrtBoAndPtr<RawArkScalar>, SCALAR_BUFFER_COUNT> m_scalar_buffers;

        std::array<XrtBoAndPtr<FpgaProjective12381>, BUCKET_BUFFER_COUNT> m_bucket_buffers;

        std::vector<bool> m_zeros;
    };

    auto create_engine_bls12381(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
        -> std::unique_ptr<Engine12381>
    {
        return std::make_unique<Engine12381>(round_pow, element_count, xclbin_dir);
    }

    Engine12381::Engine12381(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
      : impl {std::make_shared<Engine12381Impl>(round_pow, element_count, xclbin_dir)}
    {}

    void Engine12381::upload_bases(rust::Slice<const RawArkAffine> bases) const {
        impl->upload_bases(bases);
    }

    void Engine12381::run(std::array<rust::Slice<const RawArkScalar>, 4> scalars, std::array<RawArkAffine, 4> & results) const {
        impl->run(scalars, results);
    }

}
