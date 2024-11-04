use ark_ec::msm::VariableBaseMSM;
use ark_ec::AffineCurve;
use ark_ec::ProjectiveCurve;
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
    // It is best to create the Ponos MSM Engine before data is generated
    // to catch early any potential problem with FPGA initialization.
    let engine = ponos_msm_fpga::create_engine(1 << LOG_SIZE, "");

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

    let mut group = c.benchmark_group("Zprize 23, prize 1");
    group.sample_size(10);

    // benchmark for zipped form inputs
    {
        let bases_and_scalars_1 = bases
            .iter()
            .cloned()
            .zip(scalars1.iter().cloned())
            .collect::<Vec<_>>();
        let bases_and_scalars_2 = bases
            .iter()
            .cloned()
            .zip(scalars2.iter().cloned())
            .collect::<Vec<_>>();
        let bases_and_scalars_3 = bases
            .iter()
            .cloned()
            .zip(scalars3.iter().cloned())
            .collect::<Vec<_>>();
        let bases_and_scalars_4 = bases
            .iter()
            .cloned()
            .zip(scalars4.iter().cloned())
            .collect::<Vec<_>>();

        let name = format!(
            "bench {} batches of {} msm over {} curve; zipped form",
            BATCHES,
            1 << LOG_SIZE,
            curve_name::<G>()
        );
        //group.bench_function(name, |b| {
        //    b.iter(|| {
        //        let _ = black_box(dummy_msm_zipped::<G>(bases_and_scalars_1.as_ref()));
        //        let _ = black_box(dummy_msm_zipped::<G>(bases_and_scalars_2.as_ref()));
        //        let _ = black_box(dummy_msm_zipped::<G>(bases_and_scalars_3.as_ref()));
        //        let _ = black_box(dummy_msm_zipped::<G>(bases_and_scalars_4.as_ref()));
        //    })
        //});
    }

    // Upload bases to FPGA. Bases are common for all 4 vectors.
    // According to the project spec, this step is not timed.
    engine.upload_bases(&bases);

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
                // Run the Ponos MSM Engine over an array of 4 (references to) vectors.
                // It returns an array of 4 MSM results.
                let _ = black_box(engine.run([scalars1.as_ref(), scalars2.as_ref(), scalars3.as_ref(), scalars4.as_ref()]));
            })
        });
    }

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
