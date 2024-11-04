use std::array;

use ark_ff::PrimeField;
use ark_ff::UniformRand;
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ec::msm::VariableBaseMSM;
use ark_std::rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{ParallelIterator, IntoParallelIterator, IndexedParallelIterator};
use rayon::slice::ParallelSliceMut;
use clap::Parser;

const BATCHES: usize = 4;

/// ...
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Curve id
    #[arg(short, long)]
    curve: usize,

    /// Seed
    #[arg(short, long, default_value_t = 0)]
    seed: u64,

    /// Degree of the polynomial
    #[arg(short, long, default_value_t = 1<<13)]
    degree: usize,

    /// Round size in bits
    #[arg(short, long, default_value_t = 12)]
    round_bits: usize,

    /// XCLBIN directory
    #[arg(long, default_value = "")]
    xclbin_dir: String,
}

fn main() {
    let args = Args::parse();

    match args.curve {
        377 => run_curve::<ark_bls12_377::G1Affine>(args),
        381 => run_curve::<ark_bls12_381::G1Affine>(args),
        _ => unreachable!(),
    }
}

fn run_curve<G: AffineCurve>(args: Args)
where
    G::BaseField: PrimeField
{
    let engine = ponos_msm_fpga::create_engine_ext(args.round_bits, args.degree, args.xclbin_dir.as_str());

    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);
    let (bases, scalars_list) = gen_base_and_scalars::<G, _>(&mut rng, args.degree);

    engine.upload_bases(bases.as_slice());
    let results = engine.run(each_slice_ref(&scalars_list));

    let expected =
        (&scalars_list).into_par_iter()
            .map(|scalars| {
                VariableBaseMSM::multi_scalar_mul(&bases, &scalars).into_affine()
            })
            .collect::<Vec<_>>();

    (&results).into_iter().zip(&expected).enumerate()
        .for_each(|(i, (r, e))| {
            println!("Expected result for vector {}: {}", i, e);
            println!("                        FPGA: {}", r);
        });
}

fn gen_base_and_scalars<G: AffineCurve, R: RngCore>(
    rng: &mut R,
    degree: usize
)
    -> (Vec<G>, [Vec<<G::ScalarField as PrimeField>::BigInt>; BATCHES])
{
    let bases = gen_bases::<G, R>(rng, degree);
    let scalars_list = gen_scalars_list::<G, R>(rng, degree);
    (bases, scalars_list)
}

fn gen_bases<G: AffineCurve, R: RngCore>(
    rng: &mut R,
    degree: usize
)
    -> Vec<G>
{
    let nthreads = rayon::current_num_threads();

    let mut rngs = (0..nthreads).map(|_| {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        ChaCha20Rng::from_seed(seed)
    })
    .collect::<Vec<_>>();

    println!("generating rnd bases");
    let mut bases = vec!(G::default(); degree);
    let chunk_size = (bases.len() + nthreads-1) / nthreads;
    bases.par_chunks_mut(chunk_size).zip(&mut rngs)
        .for_each(|(c, rng)| {
            c.iter_mut()
                .for_each(|p| { *p = G::Projective::rand(rng).into_affine(); })
        });

    bases
}

fn gen_scalars_list<G: AffineCurve, R: RngCore>(
    rng: &mut R,
    degree: usize
)
    -> [Vec<<G::ScalarField as PrimeField>::BigInt>; BATCHES]
{
    array::from_fn(|_| {
        (0..degree).into_iter()
            .map(|_| G::ScalarField::rand(rng).into_repr())
            .collect::<Vec<_>>()
    })
}

fn each_slice_ref<T, const N: usize>(src: &[Vec<T>; N]) -> [&[T]; N] {
    array::from_fn(|i| src[i].as_slice())
}
