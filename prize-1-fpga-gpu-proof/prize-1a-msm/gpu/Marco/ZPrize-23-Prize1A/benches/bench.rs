use std::convert::TryInto;
use std::fs::File;
use std::io::{Read, Write};
use std::ptr;
use ark_bls12_381::G1Affine;
use ark_ec::msm::VariableBaseMSM;
use ark_ec::AffineCurve;
use ark_ec::ProjectiveCurve;
use ark_ff::{BigInteger256, FromBytes, PrimeField};
use ark_ff::UniformRand;
use ark_std::{
    rand::{RngCore, SeedableRng},
    test_rng,
};
use blst::print_bytes;
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use blst_msm::{multi_scalar_mult, multi_scalar_mult_free, multi_scalar_mult_init};
use ark_ff::ToBytes;
use rayon::prelude::ParallelSlice;
use rayon::iter::IndexedParallelIterator;

const LOG_SIZE: usize = 24;
const BATCHES: usize = 4;

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

fn bench_curve<G: AffineCurve>(c: &mut Criterion)
where
    G::BaseField: PrimeField,
{
    let mut rng = test_rng();
    let is381 = match <G::BaseField as PrimeField>::size_in_bits() {
        381 => 1,
        377 => 0,
        _ => panic!("not supported"),
    };

    let bases_and_scalars_lists = gen_base_and_scalars::<G, _>(&mut rng);

    let mut group = c.benchmark_group("Zprize 23, prize 1");
    group.sample_size(10);
    let name = format!(
        "bench {} batches of {} msm over {} curve",
        BATCHES,
        1 << LOG_SIZE,
        curve_name::<G>()
    );

    let points: Vec<_> = bases_and_scalars_lists[0].iter().map(|(g, _)| (*g).clone()).collect();
    let scalars: Vec<_> = bases_and_scalars_lists.iter().flat_map(|svec|
        svec.iter().map(|(_, s)|
                            (*s).into_repr()
        ).collect::<Vec<_>>()
    ).collect::<Vec<
        <<G as AffineCurve>::ScalarField as PrimeField>::BigInt
    >>();

    let mut context = multi_scalar_mult_init(points.as_slice(), BATCHES as u32, is381);
    group.bench_function(name, |b| {
        b.iter(|| {
            let msm_results = multi_scalar_mult(&mut context, points.as_slice(),
                & scalars
            );

            // let ark_res = black_box(dummy_msm::<G>(bases_and_scalars_lists[0].as_ref()));
            // assert_eq!(msm_results[0].into_affine(), ark_res);
            //
            // let ark_res = black_box(dummy_msm::<G>(bases_and_scalars_lists[1].as_ref()));
            // assert_eq!(msm_results[1].into_affine(), ark_res);
            //
            // let ark_res = black_box(dummy_msm::<G>(bases_and_scalars_lists[2].as_ref()));
            // assert_eq!(msm_results[2].into_affine(), ark_res);
            //
            // let ark_res = black_box(dummy_msm::<G>(bases_and_scalars_lists[3].as_ref()));
            // assert_eq!(msm_results[3].into_affine(), ark_res);

        })
    });
    group.finish();

    multi_scalar_mult_free(&mut context);
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
fn dummy_msm<G: AffineCurve>(bases_and_scalars: &[(G, G::ScalarField)]) -> G {
    let (bases, scalars): (Vec<G>, Vec<G::ScalarField>) = bases_and_scalars.iter().cloned().unzip();
    let scalars = scalars.iter().map(|s| s.into_repr()).collect::<Vec<_>>();
    VariableBaseMSM::multi_scalar_mul(&bases, &scalars).into_affine()
}

criterion_group!(benches, bench_curves);
criterion_main!(benches);
