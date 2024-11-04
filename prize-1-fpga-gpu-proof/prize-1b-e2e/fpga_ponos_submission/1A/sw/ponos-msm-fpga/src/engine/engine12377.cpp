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

#define USE_3ACCU 0
#define USE_DUMMY_21ST_WINDOW 0

namespace ponos {

    struct Bls12377Field {
        static const mpz_class modulo;
        static constexpr std::size_t word_count = 6;
    };

    const mpz_class Bls12377Field::modulo {"258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177"};

    using Fp12377 = FieldNumber<Bls12377Field>;

    struct FpgaFp12377 {
        std::array<uint64_t, 6> value;
        std::array<uint64_t, 2> padding;
    };

    static const auto K    = Fp12377("244536567197351118976972678317271058193963773829754279159068307164067353570771581460084726682472071493849921806358");
    static const auto K_INV = Fp12377("68198071207146767261083197268129002219705812920018760480175054542343395781477516022463262399168662188757680277506");

    static const auto ZERO = Fp12377("0");
    static const auto ONE  = Fp12377("1");

    static const auto R   = mpz_class("39402006196394479212279040100143613805079739270465446667948293404245721771497210611414266254884915640806627990306816");
    static const auto R_INV = mpz_class("193639650212296069206809223070911922792886794295675586437897404907786751263178459621544185310416431420969770445288");

    static auto into_montgomery(const mpz_class & x) -> Fp12377 {
        return Fp12377((x * R) % Fp12377::field::modulo);
    }

    static auto from_montgomery(const mpz_class & x) -> Fp12377 {
        return Fp12377((x * R_INV) % Fp12377::field::modulo);
    }

    struct FpgaXTEAffine12377 {
        FpgaFp12377 m;
        FpgaFp12377 p;
        FpgaFp12377 u;
    };

    struct TEProjective12377 {
        Fp12377 X, Y, T, Z;
    };

    static auto ZERO_TE_PROJECTIVE12381 = TEProjective12377 {ZERO, ONE, ZERO, ONE};

    struct FpgaTEProjective12377 {
        FpgaFp12377 X, Y, T, Z;
    };

    inline const auto S = Fp12377("10189023633222963290707194929886294091415157242906428298294512798502806398782149227503530278436336312243746741931");
    inline const auto S_INV = Fp12377("30567070899668889872121584789658882274245471728719284894883538395508419196346447682510590835309008936731240225793");
    inline const auto SQRT_MIN_A = Fp12377("235104237478051516191809091778322087600408126435680774020954291067230236919576441851480900410358060820255406299421");
    inline const auto SQRT_MIN_A_INV = Fp12377("61082706790104031304766947624654675238315746286697524004599167931659450635955047494384403499843722987888403344438");

    static auto is_zero(const TEProjective12377 & p) -> bool {
        return p.X == ZERO and p.Y != ZERO and p.T == ZERO and p.Z == p.Y;
    }

    static auto convert_to_fpga_input(const RawArkAffine & src) -> FpgaXTEAffine12377 {
        FpgaXTEAffine12377 dst {};

        if (src.infinity) [[unlikely]] {
            ONE.copy_to(dst.m.value);
            ONE.copy_to(dst.p.value);
            ZERO.copy_to(dst.u.value);
        }
        else {
            auto p_x = from_montgomery(mpz_from(src.x));
            auto p_y = from_montgomery(mpz_from(src.y));

            Fp12377 xpo = p_x + ONE;
            Fp12377 sxpo = xpo * S;
            Fp12377 axpo = xpo * SQRT_MIN_A;
            Fp12377 syxpo = sxpo * p_y;

            Fp12377 X = (sxpo + ONE) * axpo;
            Fp12377 Y = syxpo - p_y;
            Fp12377 Z = syxpo + p_y;
            mpz_class Z_inv = modulo_inverse<mpz_class>(Z.value(), Fp12377::field::modulo);

            Fp12377 x = X * Z_inv;
            Fp12377 y = Y * Z_inv;
            Fp12377 u = x * y * K;

            (y - x).copy_to(dst.m.value);
            (y + x).copy_to(dst.p.value);
            u.copy_to(dst.u.value);
        }
        return dst;
    }

    static auto convert_to_final_output_mont(const TEProjective12377 & src) -> RawArkAffine {
        RawArkAffine dst {};

        if (is_zero(src)) [[unlikely]] {
            into_montgomery(ZERO.value()).copy_to(dst.x);
            into_montgomery(ONE.value()).copy_to(dst.y);
            dst.infinity = true;
        }
        else {
            mpz_class Z_inv = modulo_inverse<mpz_class>(src.Z.value(), Fp12377::field::modulo);

            Fp12377 aff_x = src.X * Z_inv;
            Fp12377 aff_y = src.Y * Z_inv;
            mpz_class aff_x_inv = modulo_inverse<mpz_class>(aff_x.value(), Fp12377::field::modulo);

            Fp12377 p = ONE + aff_y;
            Fp12377 m = ONE - aff_y;
            mpz_class m_inv = modulo_inverse<mpz_class>(m.value(), Fp12377::field::modulo);

            Fp12377 u = p * m_inv;
            Fp12377 v = u * aff_x_inv;

            Fp12377 x = (u * S_INV) - ONE;
            Fp12377 y = v * S_INV * SQRT_MIN_A;

            into_montgomery(x.value()).copy_to(dst.x);
            into_montgomery(y.value()).copy_to(dst.y);
            dst.infinity = false;
        }
        return dst;
    }

    static void add_assign(TEProjective12377 & r, const TEProjective12377 & q) {
        Fp12377 A = (r.Y-r.X)*(q.Y-q.X);
        Fp12377 B = (r.Y+r.X)*(q.Y+q.X);
        Fp12377 C = r.T*K*q.T;
        Fp12377 D = r.Z*2*q.Z;
        Fp12377 E = B-A;
        Fp12377 F = D-C;
        Fp12377 G = D+C;
        Fp12377 H = B+A;
        r.X = E*F;
        r.Y = G*H;
        r.T = E*H;
        r.Z = F*G;
    }

    class Engine12377Impl {
      public:
        static constexpr int SCALAR_BUFFER_COUNT = 2;
        static constexpr int BUCKET_BUFFER_COUNT = 2;

        static constexpr int WINDOW_BITS = 13;
        static constexpr int WINDOW_COUNT = 20;
#if USE_3ACCU
#  if USE_DUMMY_21ST_WINDOW
        static constexpr int FPGA_WINDOW_COUNT = 3*7;
#  else
        static constexpr int FPGA_WINDOW_COUNT = 20;
#  endif
#else
        static constexpr int FPGA_WINDOW_COUNT = 2*10;
#endif

        static constexpr int BUCKET_COUNT = 1 << (WINDOW_BITS-1);

        explicit Engine12377Impl(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
          : m_elements_per_round(1 << round_pow),
            m_element_count(element_count)
        {
#if ENABLE_TRACE
            std::cout << "initializing MSM BLS12-377 engine\n";
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
            auto xclbin_uuid = m_device.load_xclbin(get_xclbin_path(static_cast<std::string>(xclbin_dir), "12377.xclbin"));
            m_kernel = xrt::kernel(m_device, xclbin_uuid, "krnl_msm_377");

            {
                m_base_buffers_a.resize(m_round_count);
                m_base_buffers_b.resize(m_round_count);
#ifdef USE_3ACCU
                m_base_buffers_c.resize(m_round_count);
#endif

                std::size_t size_bytes = m_elements_per_round * sizeof(FpgaXTEAffine12377);
#if ENABLE_TRACE
                std::cout << "initializing base buffers (size=" << size_bytes << ")\n";
#endif
                for (int round=0; round<m_round_count; ++round) {
                    m_base_buffers_a[round] = xrt::bo(m_device, size_bytes, xrt::bo::flags::device_only, m_kernel.group_id(4));
                    m_base_buffers_b[round] = xrt::bo(m_device, size_bytes, xrt::bo::flags::device_only, m_kernel.group_id(8));
#ifdef USE_3ACCU
                    m_base_buffers_c[round] = xrt::bo(m_device, size_bytes, xrt::bo::flags::device_only, m_kernel.group_id(12));
#endif
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
                std::size_t size_bytes = FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaTEProjective12377);
#if ENABLE_TRACE
                std::cout << "initializing bucket buffers (size=" << size_bytes << ")\n";
#endif
                for (int i=0; i<BUCKET_BUFFER_COUNT; ++i) {
                    m_bucket_buffers[i].buffer_object  = xrt::bo(m_device, size_bytes, xrt::bo::flags::host_only, m_kernel.group_id(1));
                    m_bucket_buffers[i].ptr = m_bucket_buffers[i].buffer_object.map<FpgaTEProjective12377 *>();
                    memset(m_bucket_buffers[i].ptr, 0, size_bytes);
                }
            }
        }

        ~Engine12377Impl() noexcept = default;

        Engine12377Impl(const Engine12377Impl &) = delete;
        auto operator=(const Engine12377Impl &) -> Engine12377Impl & = delete;

        void upload_bases(rust::Slice<const RawArkAffine> bases) {
            if (bases.size() < m_element_count) throw std::invalid_argument("not enough bases given");

            auto converted = AlignedArray<FpgaXTEAffine12377>(4096, m_elements_per_round);

#if ENABLE_TRACE
            std::cout << "converting and uploading bases to fpga\n";
            auto t1 = std::chrono::steady_clock::now();
#endif

            for (int round=0; round<m_round_count; ++round) {
                std::size_t first_element = m_elements_per_round * round;

                #pragma omp parallel for num_threads(8)
                for (size_t i=0; i<m_elements_per_round; ++i) {
                    converted[i] = convert_to_fpga_input(bases[first_element + i]);
                }

                m_base_buffers_a[round].write(converted.ptr());
                m_base_buffers_b[round].write(converted.ptr());
#ifdef USE_3ACCU
                m_base_buffers_c[round].write(converted.ptr());
#endif
            }

#if ENABLE_TRACE
            auto t2 = std::chrono::steady_clock::now();
            std::cout << "bases converted and uploaded TIME: " << (t2-t1).count() << "ns\n";
#endif
        }

        //
        //  Parallelize
        //
        // -    copy scalars v0 r0
        // 0,0  copy scalars v0 r1      run accu kernel v0 r0
        //      ...                     ...
        // 0,n-2 copy scalars v0 rN-1    run accu kernel v0 rN-2
        // 0,n-1    copy scalars v1 r0      run accu kernel v0 rN-1
        // 1,0  copy scalars v1 r1      run accu kernel v1 r0       aggregate v0
        // 1,1  copy scalars v1 r2      run accu kernel v1 r1       aggregate v0
        //      ...                     ...
        // 1,n-2 copy scalars v1 rN-1    run accu kernel v1 rN-2
        // 1,n-1     copy scalars v2 r0      run accu kernel v1 rN-1
        // 2,0     copy scalars v2 r1      run accu kernel v2 r0       aggregate v1
        // 2,1     copy scalars v2 r2      run accu kernel v2 r1       aggregate v1
        //      ...                     ...
        // 2,n-2     copy scalars v2 rN-1    run accu kernel v2 rN-2
        // 2,n-1     copy scalars v3 r0      run accu kernel v2 rN-1
        // 3,0     copy scalars v3 r1      run accu kernel v3 r0       aggregate v2
        // 3,1     copy scalars v3 r2      run accu kernel v3 r1       aggregate v2
        //      ...                     ...
        // 3,n-2     copy scalars v3 rN-1    run accu kernel v3 rN-2
        // 3,n-1                             run accu kernel v3 rN-1
        // -                                                         aggregate v3
        //                                                          aggregate v3
        //

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

            std::memcpy(m_scalar_buffers[scalar_buffer_index].ptr, scalar_vec.data()+first_element, m_elements_per_round*sizeof(RawArkScalar));
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
                    FPGA_WINDOW_COUNT * BUCKET_COUNT * sizeof(FpgaTEProjective12377) / 64,

                    m_base_buffers_a[round],
                    nullptr,
                    m_elements_per_round * sizeof(FpgaXTEAffine12377) / 64,
                    0,

                    m_base_buffers_b[round],
                    nullptr,
                    m_elements_per_round * sizeof(FpgaXTEAffine12377) / 64,
                    0,

#ifdef USE_3ACCU
                    m_base_buffers_c[round],
                    nullptr,
                    m_elements_per_round * sizeof(FpgaXTEAffine12377) / 64,
                    0,
#else
                    nullptr,
                    nullptr,
                    0,
                    0,
#endif

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

            std::array<TEProjective12377, WINDOW_COUNT> partial_sums;

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

            TEProjective12377 result = partial_sums[WINDOW_COUNT-1];
            for (int iwindow=WINDOW_COUNT-2; iwindow>=0; --iwindow) {
                for (int i=0; i<WINDOW_BITS; ++i) {
                    add_assign(result, result);
                }

                add_assign(result, partial_sums[iwindow]);
            }

#if ENABLE_TRACE
            auto t3 = std::chrono::steady_clock::now();
            std::cout << "aggregated TIME: " << (t2-t1).count() << "ns, " << (t3-t2).count() << "ns\n";
#endif

            results[vec_index] = convert_to_final_output_mont(result);
        }

        static auto aggregate(int n, const FpgaTEProjective12377 * a)
            -> TEProjective12377
        {
            TEProjective12377 res = ZERO_TE_PROJECTIVE12381;
            TEProjective12377 tmp = res;

            for (int i=n-1; i>=0; --i) {
                add_assign(tmp, {Fp12377(a[i].X.value), Fp12377(a[i].Y.value), Fp12377(a[i].T.value), Fp12377(a[i].Z.value)});
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
#ifdef USE_3ACCU
        std::vector<xrt::bo> m_base_buffers_c;
#endif

        std::array<XrtBoAndPtr<RawArkScalar>, SCALAR_BUFFER_COUNT> m_scalar_buffers;

        std::array<XrtBoAndPtr<FpgaTEProjective12377>, BUCKET_BUFFER_COUNT> m_bucket_buffers;
    };

    auto create_engine_bls12377(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
        -> std::unique_ptr<Engine12377>
    {
        return std::make_unique<Engine12377>(round_pow, element_count, xclbin_dir);
    }

    Engine12377::Engine12377(std::size_t round_pow, std::size_t element_count, rust::Str xclbin_dir)
      : impl {std::make_shared<Engine12377Impl>(round_pow, element_count, xclbin_dir)}
    {}

    void Engine12377::upload_bases(rust::Slice<const RawArkAffine> bases) const {
        impl->upload_bases(bases);
    }

    void Engine12377::run(std::array<rust::Slice<const RawArkScalar>, 4> scalars, std::array<RawArkAffine, 4> & results) const {
        impl->run(scalars, results);
    }

}
