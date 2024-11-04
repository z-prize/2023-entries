use crate::cuda::*;
use ark_bls12_381::{Fq, G1Affine, G1Projective};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GPU_CUDA_CORES;
    use ark_bls12_381::Fr;
    use ark_ec::msm::VariableBaseMSM;
    use ark_ec::ProjectiveCurve;
    use ark_ff::{BigInteger, BigInteger256, PrimeField};
    use ark_std::{rand, UniformRand};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use serial_test::serial;
    use std::{ops::AddAssign, sync::Mutex};

    const INIT_LEN: usize = 1 << 10;

    fn generate_bases_projective(count: usize) -> Vec<G1Projective> {
        // let mut rng = ChaCha20Rng::from_entropy();
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        assert!(count >= INIT_LEN);
        let copy_count = count / INIT_LEN;

        let mut result = (0..INIT_LEN)
            .map(|_| G1Projective::rand(&mut rng))
            .collect::<Vec<_>>();

        let copy_result = result.clone();
        for _ in 1..copy_count {
            result.extend(copy_result.iter());
        }

        result
    }

    fn generate_bases_affine(count: usize) -> Vec<G1Affine> {
        let mut rng = ChaCha20Rng::from_entropy();
        // let mut rng = ChaCha20Rng::seed_from_u64(0);
        assert!(count >= INIT_LEN);
        let copy_count = count / INIT_LEN;

        let mut result = (0..INIT_LEN)
            .map(|_| G1Projective::rand(&mut rng))
            .map(|gp| gp.into_affine())
            .collect::<Vec<_>>();

        let copy_result = result.clone();
        for _ in 1..copy_count {
            result.extend(copy_result.iter());
        }

        result
    }

    fn generate_scalars(count: usize) -> HostMemory<BigInteger256> {
        let mut rng = ChaCha20Rng::from_entropy();
        // let mut rng = ChaCha20Rng::seed_from_u64(0);

        let mut scalars = HostMemory::<BigInteger256>::new(count).unwrap();

        // let ret = (0..count)
        //     .map(|_| Fr::rand(&mut rng).into_repr())
        //     .collect::<Vec<_>>();

        for index in 0..count {
            scalars.as_mut_slice()[index] = Fr::rand(&mut rng).into_repr();
        }

        scalars
    }

    fn precompute_bases_test(
        bases: Vec<G1Projective>,
        stride: u32,
        count: u32,
    ) -> Vec<G1Projective> {
        assert!(bases.len() >= INIT_LEN);
        let copy_count = bases.len() / INIT_LEN;

        println!("stride:{}, count:{}", stride, count);

        let mut accumulator = bases[0..INIT_LEN].to_vec();
        let mut result = bases.clone();

        for _ in 1..count {
            accumulator.iter_mut().for_each(|x| {
                for _ in 0..stride {
                    x.double_in_place();
                }
            });

            let mut accumulator_all = accumulator.clone();
            for _ in 1..copy_count {
                accumulator_all.extend(accumulator.iter());
            }
            result.extend(accumulator_all.iter());
        }

        result
    }
    pub fn precompute_bases(
        bases: Vec<G1Affine>,
        stride: u32,
        count: u32,
    ) -> Vec<G1Projective> {
        let mut projective_points: Vec<G1Projective> = vec![];
        for point in &bases {
            let projective_point = G1Projective::from(point.clone());
            projective_points.push(projective_point);
        }

        let mut result = projective_points.clone();
        let mut accumulator = projective_points.clone();
        for _ in 1..count {
            accumulator.iter_mut().for_each(|x| {
                for _ in 0..stride {
                    x.double_in_place();
                }
            });
            result.extend(accumulator.iter());
        }
        result
    }

    //cargo test init_test --release  -- --nocapture
    #[test]
    fn init_test() {
        const GPU_DEV_IDX: usize = 0;
        const LOG_COUNT: u32 = 10;
        const COUNT: usize = 1 << LOG_COUNT;
        const WINDOW_BITS_COUNT: u32 = 23;
        const PRECOMPUTE_FACTOR: u32 = 4;
        const WINDOWS_COUNT: u32 = (255 - 1) / WINDOW_BITS_COUNT + 1;
        const PRECOMPUTE_WINDOWS_STRIDE: u32 =
            (WINDOWS_COUNT - 1) / PRECOMPUTE_FACTOR + 1;
        const RESULT_BITS_COUNT: u32 =
            WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;

        let bases_projective = generate_bases_projective(COUNT);
        let bases_affine =
            G1Projective::batch_normalization_into_affine(&bases_projective);
        let precalc_container = MsmPrecalcContainer::create_with_core(
            &GPU_CUDA_CORES[GPU_DEV_IDX],
            true,
        )
        .unwrap();
        let kern =
            MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container)
                .unwrap();

        let msm_ctx = kern.multi_scalar_mult_init(&bases_affine);

        let mut bases_gpu = vec![
            G1AffineNoInfinity::zero();
            (PRECOMPUTE_FACTOR as usize) * COUNT
        ];
        msm_ctx.read_bases_async(&kern.stream, &mut bases_gpu);
        kern.stream.sync().unwrap();

        let bases_cpu = precompute_bases_test(
            bases_projective,
            RESULT_BITS_COUNT,
            PRECOMPUTE_FACTOR,
        );
        let bases_cpu =
            G1Projective::batch_normalization_into_affine(&bases_cpu);
        let bases_cpu = bases_cpu
            .iter()
            .map(G1AffineNoInfinity::from)
            .collect::<Vec<_>>();

        assert_eq!(bases_cpu.len(), bases_gpu.len());
        assert_eq!(bases_cpu, bases_gpu);
    }

    // cargo test msm_bench --release  -- --nocapture
    #[test]
    fn msm_bench() {
        const GPU_DEV_IDX: usize = 0;
        for log_count in 22..23 {
            let count: usize = 1 << log_count;
            let bases = generate_bases_affine(count);

            let precalc_container = MsmPrecalcContainer::create_with_core(
                &GPU_CUDA_CORES[GPU_DEV_IDX],
                true,
            )
            .unwrap();
            let kern = MultiexpKernel::<G1Affine>::create(
                GPU_DEV_IDX,
                &precalc_container,
            )
            .unwrap();

            let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
            for _i in 0..1 {
                let scalars = generate_scalars(count);
                let scalars_gpu = DeviceMemory::<BigInteger256>::new(
                    msm_ctx.msm_kern.core.get_context(),
                    count,
                )
                .unwrap();
                scalars_gpu.read_from(&scalars, count).unwrap();

                // to_mont(&msm_ctx, scalars_gpu.get_inner(), scalars.len());
                let not_mont = true;

                let results_gpu = gpu_msm(&[], Some((scalars_gpu.get_inner(), count)), &mut msm_ctx, not_mont);
                let results_ark =
                    VariableBaseMSM::multi_scalar_mul(&bases, &scalars);
                assert_eq!(results_gpu, results_ark);
                println!("check gpu&&cpu passed");

                let mut scalars_2 = vec![BigInteger256::new([0;4]);scalars.len()];
                scalars_gpu.write_to(&mut scalars_2, scalars.len()).unwrap();
                assert_eq!(scalars.as_slice(), scalars_2.as_slice());
            }

            let scalars = generate_scalars(count);
            let scalars_gpu = DeviceMemory::<BigInteger256>::new(
                msm_ctx.msm_kern.core.get_context(),
                1,
            )
            .unwrap();
            //warm up
            for _i in 0..10 {
                let _ = gpu_msm(&scalars, None, &mut msm_ctx, true);
            }

            let start = std::time::Instant::now();
            for _i in 0..10 {
                let _ = gpu_msm(&scalars, None, &mut msm_ctx, true);
            }
            println!(
                "msm size:2^{}, spent:{:?}",
                log_count,
                start.elapsed() / 100
            );
        }
    }

    // cargo test msm_bench_copy_scalar --release  -- --nocapture
    // #[test]
    // fn msm_bench_copy_scalar() {
    //     const GPU_DEV_IDX: usize = 0;
    //     for log_count in 22..23 {
    //         let count: usize = 1 << log_count;
    //         let bases = generate_bases_affine(count);

    //         let precalc_container = MsmPrecalcContainer::create_with_core(
    //             &GPU_CUDA_CORES[GPU_DEV_IDX],
    //             true,
    //         )
    //         .unwrap();
    //         let kern = MultiexpKernel::<G1Affine>::create(
    //             GPU_DEV_IDX,
    //             &precalc_container,
    //         )
    //         .unwrap();

    //         let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
    //         for _i in 0..1 {
    //             let scalars = generate_scalars(count);
    //             let results_gpu = gpu_msm(&scalars, &mut msm_ctx);
    //             let results_ark =
    //                 VariableBaseMSM::multi_scalar_mul(&bases, &scalars);
    //             assert_eq!(results_gpu, results_ark);
    //             println!("check gpu&&cpu passed");
    //         }

    //         let scalars = generate_scalars(count);
    //         //warm up
    //         for _i in 0..10 {
    //             let _ = gpu_msm(&scalars, &mut msm_ctx);
    //         }

    //         let start = std::time::Instant::now();
    //         for _i in 0..10 {
    //             let _ = gpu_msm(&scalars, &mut msm_ctx);
    //         }
    //         println!(
    //             "msm size:2^{}, spent:{:?}",
    //             log_count,
    //             start.elapsed() / 100
    //         );
    //     }
    // }

    fn split_biguint_bits_from_low(
        num: &BigInteger256,
        low_bit_number: usize,
    ) -> (BigInteger256, BigInteger256) {
        let mut bits = num.to_bits_be();

        let len = bits.len();
        // println!("bits len:{}", len);

        let suffix = bits.drain(len - low_bit_number..).collect::<Vec<_>>();
        let mut prefix = bits;

        let extend = vec![false; low_bit_number];
        prefix.extend(extend);

        (
            BigInteger256::from_bits_be(&prefix),
            BigInteger256::from_bits_be(&suffix),
        )
    }
    #[test]
    fn test_split() {
        let b = BigInteger256([
            0x43f5fffffffcaaae,
            0x32b7fff2ed47fffd,
            0x7e83a49a2e99d69,
            0xeca8f3318332bb7a,
        ]);

        let (prefix, suffix) = split_biguint_bits_from_low(&b, 64);
        let mut total = prefix.clone();
        total.add_nocarry(&suffix);
        println!("p:{}, s:{}, b:{} new total:{}", prefix, suffix, b, total);
        assert_eq!(b, total);
    }

    //cargo test test_split_msm --release -- --nocapture
    #[test]
    fn test_cpu_split_msm() {
        const LOG_COUNT: u32 = 24;
        const COUNT: usize = 1 << LOG_COUNT;
        let scalars = generate_scalars(COUNT);
        let bases = generate_bases_affine(COUNT);

        let total_results_ark =
            VariableBaseMSM::multi_scalar_mul(&bases, &scalars);

        let mut scalars_high = vec![];
        let mut scalars_low = vec![];

        for s in scalars.iter() {
            let (h, l) = split_biguint_bits_from_low(s, 253);
            scalars_high.push(h);
            scalars_low.push(l);
        }

        let mut h_results_ark =
            VariableBaseMSM::multi_scalar_mul(&bases, &scalars_high);
        let l_results_ark =
            VariableBaseMSM::multi_scalar_mul(&bases, &scalars_low);
        h_results_ark.add_assign(&l_results_ark);
        assert_eq!(total_results_ark, h_results_ark);
    }
    // cargo test msm_simple --release -- --nocapture
    // #[test]
    // // #[serial]
    // fn msm_simple() {
    //     const GPU_DEV_IDX: usize = 0;
    //     const LOG_COUNT: u32 = 24;
    //     const COUNT: usize = 1 << LOG_COUNT;
    //     const WINDOW_BITS_COUNT: u32 = 23;
    //     const PRECOMPUTE_FACTOR: u32 = 4;
    //     const WINDOWS_COUNT: u32 = (255 - 1) / WINDOW_BITS_COUNT + 1;
    //     const PRECOMPUTE_WINDOWS_STRIDE: u32 = (WINDOWS_COUNT - 1) /
    // PRECOMPUTE_FACTOR + 1;     const RESULT_BITS_COUNT: u32 =
    // WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;     let bases =
    // generate_bases_projective(COUNT);     println!(
    //         "count:{}, windows_count:{}, stride:{}, RESULT_BITS_COUNT:{},
    // base len:{}",         COUNT,
    //         WINDOWS_COUNT,
    //         PRECOMPUTE_WINDOWS_STRIDE,
    //         RESULT_BITS_COUNT,
    //         bases.len()
    //     );

    //     let bases_2 = precompute_bases_test(bases.clone(), RESULT_BITS_COUNT,
    // PRECOMPUTE_FACTOR);     println!("pre compute end");
    //     // let bases_2_del = precompute_bases_old(bases, RESULT_BITS_COUNT,
    // PRECOMPUTE_FACTOR);     // assert_eq!(bases_2, bases_2_del);
    //     // dbg!(&bases_2);
    //     let bases_2 =
    // G1Projective::batch_normalization_into_affine(&bases_2);     println!
    // ("batch_normalization_into_affine end");

    //     let bases_no_infinity = bases_2
    //         .iter()
    //         .map(G1AffineNoInfinity::from)
    //         .collect::<Vec<_>>();

    //     println!("bases_2 to affine no infinity end");
    //     // for b in bases_no_infinity.clone().iter() {
    //     //     println!("b.x:{}, b.y:{}", b.x, b.y);
    //     // }

    //     let precalc_container =
    //         MsmPrecalcContainer::create_with_core(&
    // GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();     let kern =
    // MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX,
    // &precalc_container).unwrap();

    //     println!(
    //         "g1affine no infinity len:{} MB, g1Affine len:{},bases_2 len:{},
    // COUNT:{}",         std::mem::size_of::<G1AffineNoInfinity>() *
    // bases_no_infinity.len() / 1024 / 1024,
    //         std::mem::size_of::<G1Affine>(),
    //         bases_2.len(),
    //         COUNT
    //     );
    //     let bases_device =
    //         DeviceMemory::<G1AffineNoInfinity>::new(&kern.core.context,
    // bases_no_infinity.len())             .unwrap();

    //     bases_device
    //         .read_from_async(
    //             bases_no_infinity.as_slice(),
    //             bases_no_infinity.len(),
    //             &kern.stream,
    //         )
    //         .unwrap();

    //     let scalars = generate_scalars(COUNT);
    //     let mut scalars_high = vec![];
    //     let mut scalars_low = vec![];

    //     for s in scalars.iter() {
    //         let (h, l) = split_biguint_bits_from_low(s, 253);
    //         scalars_high.push(h);
    //         scalars_low.push(l);
    //     }

    //     let mut results = [G1Projective::zero(); RESULT_BITS_COUNT as usize];
    //     let mut unused_results = [G1Projective::zero(); 4];
    //     // let results_mutex = Mutex::new(results);
    //     // let mut results_gpu = vec![G1Projective::zero(); 1 as usize];
    //     // let results_gpu = Mutex::new(results_gpu);

    //     let reduce_callback = HostFn::new(|| {
    //         // todo
    //         println!("call back hostFn");
    //     });

    //     let mut cfg = MsmConfiguration {
    //         bases: bases_device.get_inner(),
    //         scalars: &scalars,
    //         results: &mut results,
    //         log_scalars_count: LOG_COUNT,
    //         h2d_copy_finished: None,
    //         h2d_copy_finished_callback: None,
    //         d2h_copy_finished: None,
    //         d2h_copy_finished_callback: Some(&reduce_callback),
    //         force_min_chunk_size: false,
    //         log_min_chunk_size: 0,
    //         force_max_chunk_size: true,
    //         log_max_chunk_size: LOG_COUNT as u32,
    //         window_bits_count: WINDOW_BITS_COUNT,
    //         precomputed_windows_stride: PRECOMPUTE_WINDOWS_STRIDE,
    //         precomputed_bases_stride: 1u32 << LOG_COUNT,
    //         scalars_not_montgomery: true,
    //         top_win_results: &mut unused_results,
    //     };

    //     let start = std::time::Instant::now();
    //     execute_async_test(&mut cfg, &kern);

    //     // drop(guard);
    //     kern.stream.sync().unwrap();
    //     let mut results_gpu = reduce_results(&results);
    //     let results_gpu_top_win = reduce_top_win_results(
    //         &unused_results,
    //         WINDOWS_COUNT,
    //         WINDOW_BITS_COUNT,
    //         PRECOMPUTE_WINDOWS_STRIDE,
    //     );

    //     let mut h_results_ark = VariableBaseMSM::multi_scalar_mul(&bases_2,
    // &scalars_high);     let l_results_ark =
    // VariableBaseMSM::multi_scalar_mul(&bases_2, &scalars_low);
    //     assert_eq!(l_results_ark, results_gpu);
    //     assert_eq!(h_results_ark, results_gpu_top_win);
    //     results_gpu.add_assign(results_gpu_top_win);
    //     // println!("results gpu ok , spent:{:?}", start.elapsed());

    //     let start = std::time::Instant::now();
    //     let results_ark = VariableBaseMSM::multi_scalar_mul(&bases_2,
    // &scalars);     // println!("results cpu ok , spent:{:?}",
    // start.elapsed());     h_results_ark.add_assign(l_results_ark);
    //     assert_eq!(h_results_ark, results_ark);
    //     assert_eq!(results_gpu, results_ark);
    // }

    // cargo test msm_top_window --release -- --nocapture
    // #[test]
    // // #[serial]
    // fn msm_top_window() {
    //     const GPU_DEV_IDX: usize = 0;
    //     const LOG_COUNT: u32 = 24;
    //     const COUNT: usize = 1 << LOG_COUNT;
    //     const WINDOW_BITS_COUNT: u32 = 23;
    //     const PRECOMPUTE_FACTOR: u32 = 4;
    //     const WINDOWS_COUNT: u32 = (255 - 1) / WINDOW_BITS_COUNT + 1;
    //     const PRECOMPUTE_WINDOWS_STRIDE: u32 = (WINDOWS_COUNT - 1) /
    // PRECOMPUTE_FACTOR + 1;     const RESULT_BITS_COUNT: u32 =
    // WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;     let bases =
    // generate_bases_projective(COUNT);     println!(
    //         "count:{}, windows_count:{}, stride:{}, RESULT_BITS_COUNT:{},
    // base len:{}",         COUNT,
    //         WINDOWS_COUNT,
    //         PRECOMPUTE_WINDOWS_STRIDE,
    //         RESULT_BITS_COUNT,
    //         bases.len()
    //     );

    //     let bases_2 = precompute_bases_test(bases.clone(), RESULT_BITS_COUNT,
    // PRECOMPUTE_FACTOR);     println!("pre compute end");
    //     // let bases_2_del = precompute_bases_old(bases, RESULT_BITS_COUNT,
    // PRECOMPUTE_FACTOR);     // assert_eq!(bases_2, bases_2_del);
    //     // dbg!(&bases_2);
    //     let bases_2 =
    // G1Projective::batch_normalization_into_affine(&bases_2);     println!
    // ("batch_normalization_into_affine end");

    //     let bases_no_infinity = bases_2
    //         .iter()
    //         .map(G1AffineNoInfinity::from)
    //         .collect::<Vec<_>>();

    //     println!("bases_2 to affine no infinity end");
    //     // for b in bases_no_infinity.clone().iter() {
    //     //     println!("b.x:{}, b.y:{}", b.x, b.y);
    //     // }

    //     let precalc_container =
    //         MsmPrecalcContainer::create_with_core(&
    // GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();     let kern =
    // MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX,
    // &precalc_container).unwrap();

    //     println!(
    //         "g1affine no infinity len:{} MB, g1Affine len:{},bases_2 len:{},
    // COUNT:{}",         std::mem::size_of::<G1AffineNoInfinity>() *
    // bases_no_infinity.len() / 1024 / 1024,
    //         std::mem::size_of::<G1Affine>(),
    //         bases_2.len(),
    //         COUNT
    //     );
    //     let bases_device =
    //         DeviceMemory::<G1AffineNoInfinity>::new(&kern.core.context,
    // bases_no_infinity.len())             .unwrap();

    //     bases_device
    //         .read_from_async(
    //             bases_no_infinity.as_slice(),
    //             bases_no_infinity.len(),
    //             &kern.stream,
    //         )
    //         .unwrap();

    //     let scalars = generate_scalars(COUNT);

    //     let total_windows_count = get_windows_count(WINDOW_BITS_COUNT);
    //     let top_window_unused_bits =
    //         total_windows_count * WINDOW_BITS_COUNT -
    // BLS_381_FR_MODULUS_BITS;     let top_win_used_bits =
    // WINDOW_BITS_COUNT - top_window_unused_bits;

    //     let mut results = HostMemory::<G1Projective>::new(RESULT_BITS_COUNT
    // as usize).unwrap();     let mut top_win_results =
    //         HostMemory::<G1Projective>::new(1 << top_win_used_bits as
    // usize).unwrap();

    //     // let mut results = [G1Projective::zero(); RESULT_BITS_COUNT as
    // usize];     // let mut unused_results = [G1Projective::zero(); 4];
    //     // let results_mutex = Mutex::new(results);
    //     // let mut results_gpu = vec![G1Projective::zero(); 1 as usize];
    //     // let results_gpu = Mutex::new(results_gpu);

    //     let reduce_callback = HostFn::new(|| {
    //         // todo
    //         println!("call back hostFn");
    //     });

    //     let mut cfg = MsmConfiguration {
    //         bases: bases_device.get_inner(),
    //         scalars: &scalars,
    //         // results: &mut results,
    //         log_scalars_count: LOG_COUNT,
    //         h2d_copy_finished: None,
    //         h2d_copy_finished_callback: None,
    //         d2h_copy_finished: None,
    //         d2h_copy_finished_callback: Some(&reduce_callback),
    //         force_min_chunk_size: false,
    //         log_min_chunk_size: 0,
    //         force_max_chunk_size: true,
    //         log_max_chunk_size: LOG_COUNT as u32,
    //         window_bits_count: WINDOW_BITS_COUNT,
    //         precomputed_windows_stride: PRECOMPUTE_WINDOWS_STRIDE,
    //         precomputed_bases_stride: 1u32 << LOG_COUNT,
    //         scalars_not_montgomery: true,
    //         // top_win_results: &mut top_win_results,
    //     };

    //     execute_async_test(&mut cfg, &kern);

    //     kern.stream.sync().unwrap();
    //     let results_gpu_top_win = reduce_top_win_results(
    //         &top_win_results,
    //         WINDOWS_COUNT,
    //         WINDOW_BITS_COUNT,
    //         PRECOMPUTE_WINDOWS_STRIDE,
    //     );

    //     let mut scalars_high = vec![];
    //     let mut scalars_low = vec![];

    //     for s in scalars.iter() {
    //         let (h, l) = split_biguint_bits_from_low(s, 253);
    //         scalars_high.push(h);
    //         scalars_low.push(l);
    //     }

    //     let h_results_ark = VariableBaseMSM::multi_scalar_mul(&bases_2,
    // &scalars_high);

    //     assert_eq!(h_results_ark, results_gpu_top_win);
    // }
}
