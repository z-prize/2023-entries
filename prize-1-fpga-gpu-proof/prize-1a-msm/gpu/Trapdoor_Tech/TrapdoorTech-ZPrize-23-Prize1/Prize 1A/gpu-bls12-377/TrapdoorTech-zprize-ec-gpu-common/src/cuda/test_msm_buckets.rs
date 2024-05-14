use crate::cuda::*;
#[cfg(feature = "bls12_377")]
use ark_bls12_377::{Fr, G1Affine, G1Projective};
#[cfg(feature = "bls12_381")]
use ark_bls12_381::{Fq, Fr, G1Affine, G1Projective};
use std::fmt::Debug;

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuAffine, GpuEdProjective, GPU_CUDA_CORES};
    use ark_bls12_377::G1Affine;
    use ark_ec::msm::VariableBaseMSM;
    use ark_ec::ProjectiveCurve;
    use ark_ff::{BigInteger256, PrimeField};
    use ark_std::{rand, UniformRand};
    use crossbeam::thread;
    use rand_chacha::ChaCha20Rng;
    use std::sync::{Arc, Mutex};
    use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
    use ark_std::{
        rand::{RngCore, SeedableRng},
        test_rng,
    };

    fn generate_bases_affine(count: usize) -> Vec<G1Affine> {
        let mut rng = test_rng();
        let num_threads = 1<<6;
        let per_thread = count / num_threads;
        let mut rngs = (0..num_threads)
            .map(|i| {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                seed[0] = i as u8;
                ChaCha20Rng::from_seed(seed)
            })
            .collect::<Vec<_>>();

        let bases = rngs
            .par_iter_mut()
            .map(|mut rng| {
                let proj_bases = (0..per_thread)
                    .map(move |_| G1Projective::rand(&mut rng))
                    .collect::<Vec<_>>();
                G1Projective::batch_normalization_into_affine(&proj_bases)
            })
            .flatten()
            .collect::<Vec<_>>();
        bases
    }

    fn generate_scalars(count: usize) -> HostMemory<BigInteger256> {
        let mut rng = ChaCha20Rng::from_entropy();

        let mut scalars = HostMemory::<BigInteger256>::new(count).unwrap();

        for index in 0..count {
            scalars.as_mut_slice()[index] = Fr::rand(&mut rng).into_repr();
        }

        scalars
    }

    // cargo test msm_bench --features=bls12_377  --release  -- --nocapture
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
            println!("generate base ok");

            let precalc_container =
                MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
            let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();

            let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
            //check
            for _i in 0..2 {
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
                for _i in 0..100 {
                    let _ = gpu_msm(&scalars, &mut msm_ctx);
                }
                println!(
                    "msm size:2^{}, average spent:{:?}",
                    log_count,
                    start.elapsed().div_f64(100.0)
                );
            }
        }
    }
}
