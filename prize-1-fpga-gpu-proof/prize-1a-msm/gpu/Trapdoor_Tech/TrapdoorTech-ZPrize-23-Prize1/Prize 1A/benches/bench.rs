use ark_ec::msm::VariableBaseMSM;
use ark_ec::AffineCurve;
use ark_ec::ProjectiveCurve;
use ark_ff::BigInteger;
use ark_ff::BigInteger256;
use ark_ff::PrimeField;
use ark_ff::UniformRand;
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng,
};
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::time::Instant;

const LOG_SIZE: usize = 24;
const BATCHES: usize = 4;
const GPU_DEV_IDX: usize = 0;

fn gen_base_and_scalars<G: AffineCurve, R: RngCore>(
    rng: &mut R,
) -> [Vec<(G, G::ScalarField)>; BATCHES] {
    let num_threads = rayon::max_num_threads();
    let per_thread = (1 << LOG_SIZE) / num_threads + 1;
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
                .map(move |_| G::Projective::rand(&mut rng))
                .collect::<Vec<_>>();
            G::Projective::batch_normalization_into_affine(&proj_bases)
        })
        .flatten()
        .collect::<Vec<_>>();

    let scalars1 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(rng))
        .collect::<Vec<_>>();
    let scalars2 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(rng))
        .collect::<Vec<_>>();
    let scalars3 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(rng))
        .collect::<Vec<_>>();
    let scalars4 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(rng))
        .collect::<Vec<_>>();

    [
        bases.iter().cloned().zip(scalars1).collect::<Vec<_>>(),
        bases.iter().cloned().zip(scalars2).collect::<Vec<_>>(),
        bases.iter().cloned().zip(scalars3).collect::<Vec<_>>(),
        bases.iter().cloned().zip(scalars4).collect::<Vec<_>>(),
    ]
}

fn bench_curve_377<G: AffineCurve>(c: &mut Criterion)
where
    G::BaseField: PrimeField,
{
    use ec_gpu_common_377::{MsmPrecalcContainer, MultiexpKernel, GPU_CUDA_CORES, gpu_msm};
    use ark_bls12_377::{G1Affine, G1Projective, Fr};
    use crypto_cuda_377::HostMemory;
    let mut rng = test_rng();
    let num_threads = 64;
    let per_thread = (1 << LOG_SIZE) / num_threads;
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

    //Prepare GPU
    let precalc_container =
        MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
    let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();
    let scalar_count = (1 << LOG_SIZE) * BATCHES;
    let mut scalars_list = HostMemory::<BigInteger256>::new(scalar_count).unwrap();
    for item in scalars_list.iter_mut() {
        *item = Fr::rand(&mut rng).into_repr();
    }
    let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
    let scalars = &scalars_list[0..(1<<LOG_SIZE)];
    let warm_up_gpu_res = gpu_msm(scalars, &mut msm_ctx).into_affine();
    //let cpu_res = VariableBaseMSM::multi_scalar_mul(&bases, &scalars).into_affine();
    //assert_eq!(warm_up_gpu_res, cpu_res);

    let mut group = c.benchmark_group("Zprize 23, prize 1A");
    group.sample_size(10);
    let name = format!(
        "bench {} batches of {} msm over {} curve",
        BATCHES,
        1 << LOG_SIZE,
        curve_name::<G>()
    );
    group.bench_function(name, |b| {
        b.iter(|| {
            let _ = black_box(
                scalars_list.chunks(1<<LOG_SIZE)
                        .map(|scalars| {
                        let _ = gpu_msm(&scalars, &mut msm_ctx);
                    }).collect::<Vec<_>>()
            );
        })
    });
    group.finish();
}

fn bench_curve_381<G: AffineCurve>(c: &mut Criterion)
where
    G::BaseField: PrimeField,
{
    use ec_gpu_common_381::{MsmPrecalcContainer, MultiexpKernel, GPU_CUDA_CORES, gpu_msm};
    use ark_bls12_381::{G1Affine, G1Projective, Fr};
    use crypto_cuda_381::HostMemory;
    let mut rng = test_rng();
    let num_threads = 64;
    let per_thread = (1 << LOG_SIZE) / num_threads;
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

    //Prepare GPU
    let precalc_container =
        MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
    let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();
    let scalar_count = (1 << LOG_SIZE) * BATCHES;
    let mut scalars_list = HostMemory::<BigInteger256>::new(scalar_count).unwrap();
    for item in scalars_list.iter_mut() {
        *item = Fr::rand(&mut rng).into_repr();
    }
    let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
    let scalars = &scalars_list[0..(1<<LOG_SIZE)];
    let warm_up_gpu_res = gpu_msm(scalars, &mut msm_ctx).into_affine();
    //let cpu_res = VariableBaseMSM::multi_scalar_mul(&bases, &scalars).into_affine();
    //assert_eq!(warm_up_gpu_res, cpu_res);

    let mut group = c.benchmark_group("Zprize 23, prize 1A");
    group.sample_size(10);
    let name = format!(
        "bench {} batches of {} msm over {} curve",
        BATCHES,
        1 << LOG_SIZE,
        curve_name::<G>()
    );
    group.bench_function(name, |b| {
        b.iter(|| {
            let _ = black_box(
                scalars_list.chunks(1 << LOG_SIZE)
                    .map(|scalars| {
                        let _ = gpu_msm(&scalars, &mut msm_ctx);
                    }).collect::<Vec<_>>(),
            );
        })
    });
    group.finish();
}

fn bench_curves(c: &mut Criterion) {
    bench_curve_377::<ark_bls12_377::G1Affine>(c);
    bench_curve_381::<ark_bls12_381::G1Affine>(c);
}

fn curve_name<G: AffineCurve>() -> String
where
    G::BaseField: PrimeField,
{
    match <G::BaseField as PrimeField>::size_in_bits() {
        381 => "BLS12-381".to_string(),
        377 => "BLS12-377".to_string(),
        _ => panic!("not supported"),
    }
}

// A dummy wrapper of Arkwork's MSM interface
fn dummy_msm<G: AffineCurve>(bases_and_scalars: &[(G, G::ScalarField)]) -> G {
    let (bases, scalars): (Vec<G>, Vec<G::ScalarField>) = bases_and_scalars.iter().cloned().unzip();
    let scalars = scalars.iter().map(|s| s.into_repr()).collect::<Vec<_>>();
    VariableBaseMSM::multi_scalar_mul(&bases, &scalars).into_affine()
}

criterion_group!(benches, bench_curves);
criterion_main!(benches);
