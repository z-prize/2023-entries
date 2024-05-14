use std::os::raw::c_void;
use ark_bls12_377::{Fr, G1Affine};
use ark_ec::msm::VariableBaseMSM;
use ark_ec::AffineCurve;
use ark_ec::ProjectiveCurve;
use ark_ec::short_weierstrass_jacobian::GroupAffine;
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

const LOG_SIZE: usize = 24;
const BATCHES: usize = 4;

// common device context for CPU/GPU/FPGA
#[cfg_attr(feature = "quiet", allow(improper_ctypes), allow(dead_code))]
extern "C" {
    fn createContext(curve: u32, maxBatchCount: u32, maxPointCount: u32) -> *mut c_void;
    fn destroyContext(context: *mut c_void) -> i32;
    fn preprocessPoints(context: *mut c_void, pointData: *const G1Affine, pointCount: u32) -> i32;
    fn processBatches(context: *mut c_void, resultData: *mut u64, scalarData: *const *const u64, batchCount: u32, pointCount: u32) -> i32;
}

fn gen_bases<G: AffineCurve, R: RngCore>(rng: &mut R) -> Vec<G> {
    let num_threads = rayon::current_num_threads();
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
        .take_any(1 << LOG_SIZE)
        .collect::<Vec<_>>()
        .to_vec();

    bases
}

fn bench_curve<G: AffineCurve>(c: &mut Criterion)
    where
        G::BaseField: PrimeField,
{
    let mut rng = test_rng();
    let bases = gen_bases::<G, _>(&mut rng);
    let scalars1 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(&mut rng).into_repr())
        .collect::<Vec<_>>();
    let scalars2 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(&mut rng).into_repr())
        .collect::<Vec<_>>();
    let scalars3 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(&mut rng).into_repr())
        .collect::<Vec<_>>();
    let scalars4 = (0..1 << LOG_SIZE)
        .map(|_| G::ScalarField::rand(&mut rng).into_repr())
        .collect::<Vec<_>>();

    // -- [device setup] ---------------------------------------------------------------------------
    let context = unsafe {
        println!("Creating device context for {} bit base field", G::BaseField::size_in_bits());
        createContext(G::BaseField::size_in_bits() as u32, BATCHES as u32, 1 << LOG_SIZE)
    };
    let err = unsafe {
        preprocessPoints(context, bases.as_slice() as *const _ as *const G1Affine, bases.len() as u32)
    };
    if err != 0 {
        panic!("Error {} occurred in C code", err);
    }
    let mut msm_results = vec![G::default(); BATCHES];
    let element_size = match msm_results.get(0) {
        Some(elem) => std::mem::size_of_val(elem),
        None => 0, // Handle the case where the vector is empty
    };
    let num_bytes = msm_results.len() * element_size;
    // assumes batches == 4, the given code explicitly generates exactly 4 scalar vectors.
    assert_eq!(BATCHES, 4);
    let mut scalar_ptrs: [*const u64; BATCHES] = [
        scalars1.as_ptr() as *const u64,
        scalars2.as_ptr() as *const u64,
        scalars3.as_ptr() as *const u64,
        scalars4.as_ptr() as *const u64,
    ];
    // ---------------------------------------------------------------------------------------------

    let mut group = c.benchmark_group("Zprize 23, prize 1");
    group.sample_size(10);

    // benchmark for splitted form inputs
    {
        let name = format!(
            "bench {} batches of {} msm over {} curve; splitted form",
            BATCHES,
            1 << LOG_SIZE,
            curve_name::<G>()
        );
        group.bench_function(name, |b| {
            b.iter(|| {
                let err = unsafe {
                    processBatches(context, msm_results.as_mut_ptr() as *mut u64, scalar_ptrs.as_ptr(), BATCHES as u32, bases.len() as u32)
                };
                if err != 0 {
                    panic!("Error {} occurred in C code", err);
                }
            })
        });
    }

    // -- [destroy device context] -----------------------------------------------------------------
    unsafe { destroyContext(context); }
    // ---------------------------------------------------------------------------------------------

    group.finish();
}

fn bench_curves(c: &mut Criterion) {
    bench_curve::<ark_bls12_377::G1Affine>(c);
    bench_curve::<ark_bls12_381::G1Affine>(c);
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
fn dummy_msm_zipped<G: AffineCurve>(
    bases_and_scalars: &[(G, <G::ScalarField as PrimeField>::BigInt)],
) -> G {
    let (bases, scalars): (Vec<G>, Vec<<G::ScalarField as PrimeField>::BigInt>) =
        bases_and_scalars.iter().cloned().unzip();
    VariableBaseMSM::multi_scalar_mul(&bases, &scalars).into_affine()
}

// A dummy wrapper of Arkwork's MSM interface
fn dummy_msm_splitted<G: AffineCurve>(
    bases: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> G {
    VariableBaseMSM::multi_scalar_mul(&bases, &scalars).into_affine()
}

criterion_group!(benches, bench_curves);
criterion_main!(benches);