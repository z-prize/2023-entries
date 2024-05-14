use crate::cuda::*;
#[cfg(feature = "bls12_381")]
use ark_bls12_381::{Fq, Fr, G1Affine, G1Projective};
use std::fmt::Debug;

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
#[allow(dead_code)]
pub fn execute_async_test<'a, 'b, 'c, G: ark_ec::AffineCurve>(
    configuration: &mut MsmConfiguration<'a, 'b, 'c, G>,
    kern: &MultiexpKernel<'a, 'b, G>,
) {
    let h2d_fn_and_data = configuration
        .h2d_copy_finished_callback
        .map(get_raw_fn_and_data);

    let d2h_fn_and_data = configuration
        .d2h_copy_finished_callback
        .map(get_raw_fn_and_data);

    let conf = unsafe {
        ExecuteConfiguration {
            scalars: configuration.scalars,
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
            scalars_not_montgomery: true,
            msm_context: configuration.msm_context,
        }
    };
    kern.execute_msm(conf);
}

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::GPU_CUDA_CORES;
    use ark_ec::msm::VariableBaseMSM;
    use ark_ec::ProjectiveCurve;
    use ark_ff::{ BigInteger256, PrimeField};
    use ark_std::{rand, UniformRand};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const INIT_LEN: usize = 1 << 10;

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

        for index in 0..count {
            scalars.as_mut_slice()[index] = Fr::rand(&mut rng).into_repr();
        }

        scalars
    }

    // cargo test msm_bench --features=bls12_381  --release  -- --nocapture
    #[test]
    fn msm_bench() {
        #[cfg(feature = "bls12_377")]
        println!("using curve bls12_377");

        #[cfg(feature = "bls12_381")]
        println!("using curve bls12_381");

        const GPU_DEV_IDX: usize = 0;
        for log_count in 24..25 {
            let count: usize = 1 << log_count;
            let bases = generate_bases_affine(count);

            let precalc_container =
                MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
            let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();

            let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
            //check
            for _i in 0..10 {
                let scalars = generate_scalars(count);

                let results_gpu = gpu_msm(&scalars, &mut msm_ctx);
                let results_ark = VariableBaseMSM::multi_scalar_mul(&bases, &scalars);
                assert_eq!(results_gpu, results_ark);
                println!("Check for correctness cpu && gpu results, passed");
            }
            for _ in 0..10 {
                // warm up
                let scalars = generate_scalars(count);

                for _i in 0..10 {
                    let _ = gpu_msm(&scalars, &mut msm_ctx);
                }

                let start = std::time::Instant::now();
                for _i in 0..50 {
                    let _ = gpu_msm(&scalars, &mut msm_ctx);
                }
                println!(
                    "msm size:2^{}, average spent:{:?}",
                    log_count,
                    start.elapsed().div_f64(50.0)
                );
            }
        }
    }

}
