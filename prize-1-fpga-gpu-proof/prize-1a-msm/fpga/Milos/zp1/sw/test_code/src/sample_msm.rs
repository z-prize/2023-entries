use std::io::Write;
use std::{fs::File, mem::transmute};

use ark_ec::msm::VariableBaseMSM;
use ark_ec::short_weierstrass_jacobian::{GroupAffine, GroupProjective};
use ark_ec::{ProjectiveCurve, SWModelParameters};
use ark_ff::UniformRand;
use ark_ff::{Field, PrimeField};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use crate::Args;

pub(crate) fn sample_msm<P: SWModelParameters<BaseField = F>, F: PrimeField>(args: &Args) {
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);

    let scalars: Vec<P::ScalarField> = (0..args.degree)
        .map(|_| P::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();
    let scalars_bigint = scalars.iter().map(|x| x.into_repr()).collect::<Vec<_>>();

    let bases: Vec<GroupAffine<P>> = (0..args.degree)
        .map(|_| GroupProjective::<P>::rand(&mut rng).into_affine())
        .collect::<Vec<_>>();

    let res = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_bigint).into_affine();

    // for (i, e) in scalars.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    // for (i, e) in bases.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    // scalars
    {
        let filename = format!(
            "zprize_msm_curve_{}_scalars_dim_{}_seed_{}.csv",
            F::size_in_bits(),
            scalars.len(),
            args.seed,
        );
        let mut file = File::create(filename).unwrap();

        for (i, s) in scalars.iter().enumerate() {
            file.write_all(format!("{}, {}\n", i, printer_256(s)).as_ref())
                .unwrap();
        }
    }

    // base
    {
        let filename = format!(
            "zprize_msm_curve_{}_bases_dim_{}_seed_{}.csv",
            F::size_in_bits(),
            scalars.len(),
            args.seed,
        );
        let mut file = File::create(filename).unwrap();

        for (i, s) in bases.iter().enumerate() {
            file.write_all(
                format!(
                    "{}, x, {}, y, {}\n",
                    i,
                    printer_384(&s.x),
                    printer_384(&s.y)
                )
                .as_ref(),
            )
            .unwrap();
        }
    }

    // result
    {
        let filename = format!(
            "zprize_msm_curve_{}_res_dim_{}_seed_{}.csv",
            F::size_in_bits(),
            scalars.len(),
            args.seed,
        );
        let mut file = File::create(filename).unwrap();

        file.write_all(
            format!("x, {}, y, {}\n", printer_384(&res.x), printer_384(&res.y)).as_ref(),
        )
        .unwrap();
    }
}

fn printer_256<F: Field>(f: &F) -> String {
    let mont = unsafe { transmute::<_, &[u64; 4]>(f) };
    format!(
        "{:0>16x?}, {:0>16x?}, {:0>16x?}, {:0>16x?}, zz: {}",
        mont[3], mont[2], mont[1], mont[0], f
    )
}

fn printer_384<F: Field>(f: &F) -> String {
    let mont = unsafe { transmute::<_, &[u64; 6]>(f) };
    format!(
        "{:0>16x?}, {:0>16x?}, {:0>16x?}, {:0>16x?}, {:0>16x?}, {:0>16x?}, zz: {}",
        mont[5], mont[4], mont[3], mont[2], mont[1], mont[0], f
    )
}
