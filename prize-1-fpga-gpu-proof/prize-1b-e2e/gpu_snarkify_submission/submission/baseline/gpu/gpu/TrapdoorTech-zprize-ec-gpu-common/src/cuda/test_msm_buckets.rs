use crate::cuda::*;
use ark_bls12_381::{Fq, G1Affine, G1Projective};
use std::fmt::Debug;

#[derive(Copy, Clone, Debug)]
pub struct G1AffineNoInfinity {
    pub x: Fq,
    pub y: Fq,
}

impl From<&G1Affine> for G1AffineNoInfinity {
    fn from(point: &G1Affine) -> Self {
        assert!(!point.infinity);
        G1AffineNoInfinity {
            x: point.x,
            y: point.y,
        }
    }
}

#[allow(dead_code)]
pub fn execute_async_test<'a, 'b>(
    configuration: &mut MsmConfiguration<'_>,
    kern: &MultiexpKernel<'a, 'b, G1Affine>,
) {
    let h2d_fn_and_data = configuration
        .h2d_copy_finished_callback
        .map(get_raw_fn_and_data);

    let d2h_fn_and_data = configuration
        .d2h_copy_finished_callback
        .map(get_raw_fn_and_data);

    let conf = unsafe {
        ExecuteConfiguration {
            bases: configuration.bases,
            scalars: configuration.scalars,
            results: configuration.results,
            log_scalars_count: configuration.log_scalars_count,
            h2d_copy_finished: configuration
                .h2d_copy_finished
                .map_or(std::mem::zeroed(), |e| e.into()),
            h2d_copy_finished_callback: h2d_fn_and_data.and_then(|f| f.0),
            h2d_copy_finished_callback_data: h2d_fn_and_data.map_or(std::mem::zeroed(), |f| f.1),
            d2h_copy_finished: configuration
                .d2h_copy_finished
                .map_or(std::mem::zeroed(), |e| e.into()),
            d2h_copy_finished_callback: d2h_fn_and_data.and_then(|f| f.0),
            d2h_copy_finished_callback_data: d2h_fn_and_data.map_or(std::mem::zeroed(), |f| f.1),
            force_min_chunk_size: configuration.force_min_chunk_size,
            log_min_chunk_size: configuration.log_min_chunk_size,
            force_max_chunk_size: configuration.force_max_chunk_size,
            log_max_chunk_size: configuration.log_max_chunk_size,
            window_bits_count: configuration.window_bits_count,
            precomputed_windows_stride: configuration.precomputed_windows_stride,
            precomputed_bases_stride: configuration.precomputed_bases_stride,
            scalars_not_montgomery: true,
        }
    };
    kern.execute_msm(conf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GPU_CUDA_CORES;
    use ark_bls12_381::Fr;
    use ark_ec::msm::VariableBaseMSM;
    use ark_ec::ProjectiveCurve;
    use ark_ff::{BigInteger256, PrimeField, Zero};
    use ark_std::{rand, UniformRand};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use serial_test::serial;
    use std::sync::Mutex;

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

    fn generate_scalars(count: usize) -> Vec<BigInteger256> {
        let mut rng = ChaCha20Rng::from_entropy();

        (0..count)
            .map(|_| Fr::rand(&mut rng).into_repr())
            .collect::<Vec<_>>()
    }

    fn precompute_bases_test(
        bases: Vec<G1Projective>,
        stride: u32,
        count: u32,
    ) -> Vec<G1Projective> {
        assert!(bases.len() >= INIT_LEN);
        let copy_count = bases.len() / INIT_LEN;

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
    pub fn precompute_bases(bases: Vec<G1Affine>, stride: u32, count: u32) -> Vec<G1Projective> {
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

  
    // cargo test msm_bench_test --release  -- --nocapture
    #[test]
    fn msm_bench_test() {
        const GPU_DEV_IDX: usize = 0;
        const LOG_COUNT: u32 = 24;
        const COUNT: usize = 1 << LOG_COUNT;
        let bases = generate_bases_affine(COUNT);

        let precalc_container =
            MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
        let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();

        let msm_ctx = kern.multi_scalar_mult_init(bases.as_slice(), &kern);
        let scalars = generate_scalars(COUNT);
        //warm up
        for _i in 0..5 {
            let _ = gpu_msm(&scalars, &msm_ctx);
        }

        let start = std::time::Instant::now();

        for _i in 0..10 {
            let _ = gpu_msm(&scalars, &msm_ctx);
        }
        println!("msm size:2^{}, spent:{:?}", LOG_COUNT, start.elapsed() / 10);
    }

}
