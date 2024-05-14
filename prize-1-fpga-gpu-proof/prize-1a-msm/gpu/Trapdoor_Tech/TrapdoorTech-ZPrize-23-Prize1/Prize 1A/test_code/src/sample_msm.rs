use std::io::Write;
use std::{fs::File, mem::transmute};

use ark_ec::msm::VariableBaseMSM;
use ark_ec::short_weierstrass_jacobian::{GroupAffine, GroupProjective};
use ark_ec::{ProjectiveCurve, SWModelParameters};
use ark_ff::UniformRand;
use ark_ff::{Field, PrimeField};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use crate::Args;

const GPU_DEV_IDX: usize = 0;

pub(crate) fn sample_msm_377<P: SWModelParameters<BaseField = F>, F: PrimeField>(args: &Args) {
    use ec_gpu_common_377::{MsmPrecalcContainer, MultiexpKernel, GPU_CUDA_CORES, gpu_msm};
    use ark_bls12_377::{G1Affine, G1Projective, Fr};
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);

    let mut scalars_bigint = (0..(1<<10))
        .map(|_| Fr::rand(&mut rng).into_repr())
        .collect::<Vec<_>>();
    
    for _ in 0..(args.degree - 10) {
        scalars_bigint.extend(scalars_bigint.clone());
    }

    let mut bases: Vec<G1Affine> = (0..(1<<10))
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect::<Vec<_>>();
    for _ in 0..(args.degree - 10) {
        bases.extend(bases.clone());
    }

    let res = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_bigint).into_affine();
    let precalc_container =
        MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
    let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();
    let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
    let warm_up_gpu_res = gpu_msm(&scalars_bigint, &mut msm_ctx).into_affine();

    assert_eq!(res, warm_up_gpu_res);
    println!("bls12-377 result correct!");

    // for (i, e) in scalars.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    // for (i, e) in bases.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    //// scalars
    //{
    //    let filename = format!(
    //        "zprize_msm_curve_{}_scalars_dim_{}_seed_{}.csv",
    //        F::size_in_bits(),
    //        scalars.len(),
    //        args.seed,
    //    );
    //    let mut file = File::create(filename).unwrap();

    //    for (i, s) in scalars.iter().enumerate() {
    //        file.write_all(format!("{}, {}\n", i, printer_256(s)).as_ref())
    //            .unwrap();
    //    }
    //}

    //// base
    //{
    //    let filename = format!(
    //        "zprize_msm_curve_{}_bases_dim_{}_seed_{}.csv",
    //        F::size_in_bits(),
    //        scalars.len(),
    //        args.seed,
    //    );
    //    let mut file = File::create(filename).unwrap();

    //    for (i, s) in bases.iter().enumerate() {
    //        file.write_all(
    //            format!(
    //                "{}, x, {}, y, {}\n",
    //                i,
    //                printer_384(&s.x),
    //                printer_384(&s.y)
    //            )
    //            .as_ref(),
    //        )
    //        .unwrap();
    //    }
    //}

    //// result
    //{
    //    let filename = format!(
    //        "zprize_msm_curve_{}_res_dim_{}_seed_{}.csv",
    //        F::size_in_bits(),
    //        scalars.len(),
    //        args.seed,
    //    );
    //    let mut file = File::create(filename).unwrap();

    //    file.write_all(
    //        format!("x, {}, y, {}\n", printer_384(&res.x), printer_384(&res.y)).as_ref(),
    //    )
    //    .unwrap();
    //}
}

pub(crate) fn sample_msm_381<P: SWModelParameters<BaseField = F>, F: PrimeField>(args: &Args) {
    use ec_gpu_common_381::{MsmPrecalcContainer, MultiexpKernel, GPU_CUDA_CORES, gpu_msm};
    use ark_bls12_381::{G1Affine, G1Projective, Fr};
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);

    let mut scalars_bigint = (0..(1<<10))
        .map(|_| Fr::rand(&mut rng).into_repr())
        .collect::<Vec<_>>();
    
    for _ in 0..(args.degree - 10) {
        scalars_bigint.extend(scalars_bigint.clone());
    }
    //let scalars_bigint = scalars.iter().map(|x| x.into_repr()).collect::<Vec<_>>();

    let mut bases: Vec<G1Affine> = (0..(1<<10))
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect::<Vec<_>>();
    for _ in 0..(args.degree - 10) {
        bases.extend(bases.clone());
    }

    let res = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_bigint).into_affine();
    let precalc_container =
        MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
    let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();
    let mut msm_ctx = kern.multi_scalar_mult_init(bases.as_slice());
    let warm_up_gpu_res = gpu_msm(&scalars_bigint, &mut msm_ctx).into_affine();

    assert_eq!(res, warm_up_gpu_res);
    println!("bls12-381 result correct!");

    // for (i, e) in scalars.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    // for (i, e) in bases.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    //// scalars
    //{
    //    let filename = format!(
    //        "zprize_msm_curve_{}_scalars_dim_{}_seed_{}.csv",
    //        F::size_in_bits(),
    //        scalars.len(),
    //        args.seed,
    //    );
    //    let mut file = File::create(filename).unwrap();

    //    for (i, s) in scalars.iter().enumerate() {
    //        file.write_all(format!("{}, {}\n", i, printer_256(s)).as_ref())
    //            .unwrap();
    //    }
    //}

    //// base
    //{
    //    let filename = format!(
    //        "zprize_msm_curve_{}_bases_dim_{}_seed_{}.csv",
    //        F::size_in_bits(),
    //        scalars.len(),
    //        args.seed,
    //    );
    //    let mut file = File::create(filename).unwrap();

    //    for (i, s) in bases.iter().enumerate() {
    //        file.write_all(
    //            format!(
    //                "{}, x, {}, y, {}\n",
    //                i,
    //                printer_384(&s.x),
    //                printer_384(&s.y)
    //            )
    //            .as_ref(),
    //        )
    //        .unwrap();
    //    }
    //}

    //// result
    //{
    //    let filename = format!(
    //        "zprize_msm_curve_{}_res_dim_{}_seed_{}.csv",
    //        F::size_in_bits(),
    //        scalars.len(),
    //        args.seed,
    //    );
    //    let mut file = File::create(filename).unwrap();

    //    file.write_all(
    //        format!("x, {}, y, {}\n", printer_384(&res.x), printer_384(&res.y)).as_ref(),
    //    )
    //    .unwrap();
    //}
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
