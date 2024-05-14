use std::io::Write;
use std::{fs::File, mem::transmute};
use std::os::raw::c_void;

use ark_ec::msm::VariableBaseMSM;
use ark_ec::short_weierstrass_jacobian::{GroupAffine, GroupProjective};
use ark_ec::{ProjectiveCurve, SWModelParameters};
use ark_ff::UniformRand;
use ark_ff::{Field, PrimeField};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use ark_bls12_377::{Fr, G1Affine};

use crate::Args;

// common device context for CPU/GPU/FPGA
#[cfg_attr(feature = "quiet", allow(improper_ctypes), allow(dead_code))]
extern "C" {
    fn createContext(curve: u32, maxBatchCount: u32, maxPointCount: u32) -> *mut c_void;
    fn destroyContext(context: *mut c_void) -> i32;
    fn preprocessPoints(context: *mut c_void, pointData: *const G1Affine, pointCount: u32) -> i32;
    fn processBatches(context: *mut c_void, resultData: *mut u64, scalarData: *const *const u64, batchCount: u32, pointCount: u32) -> i32;
}

pub(crate) fn sample_msm<P: SWModelParameters<BaseField=F>, F: PrimeField>(args: &Args) {
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed);

    let context = unsafe {
        println!("{}", F::size_in_bits());
        createContext(F::size_in_bits() as u32, 4, 16777216)
    };

    println!("Generating points");

    let bases: Vec<GroupAffine<P>> = (0..args.degree)
        .map(|_| GroupProjective::<P>::rand(&mut rng).into_affine())
        .collect::<Vec<_>>();

    let err = unsafe {
        preprocessPoints(context, bases.as_slice() as *const _ as *const G1Affine, bases.len() as u32)
    };
    if err != 0 {
        panic!("Error {} occurred in C code", err);
    }

    println!("Generating scalars");

    let scalars_0 = (0..args.degree)
        .map(|_| P::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>().iter().map(|x| x.into_repr()).collect::<Vec<_>>();
    let scalars_1 = (0..args.degree)
        .map(|_| P::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>().iter().map(|x| x.into_repr()).collect::<Vec<_>>();
    let scalars_2 = (0..args.degree)
        .map(|_| P::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>().iter().map(|x| x.into_repr()).collect::<Vec<_>>();
    let scalars_3 = (0..args.degree)
        .map(|_| P::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>().iter().map(|x| x.into_repr()).collect::<Vec<_>>();

    let batches: usize = 4;
    let mut scalar_ptrs: [*const u64; 4] = [
        scalars_0.as_ptr() as *const u64,
        scalars_1.as_ptr() as *const u64,
        scalars_2.as_ptr() as *const u64,
        scalars_3.as_ptr() as *const u64,
    ];

    println!("Running MSM");

    let mut msm_results = vec![GroupAffine::<P>::default(); batches];

    let err = unsafe {
        processBatches(context, msm_results.as_mut_ptr() as *mut u64, scalar_ptrs.as_ptr(), batches as u32, bases.len() as u32)
    };
    if err != 0 {
        panic!("Error {} occurred in C code", err);
    }

    println!("Returning results from ark msm!");
    let golden_0 = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_0).into_affine();
    let golden_1 = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_1).into_affine();
    let golden_2 = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_2).into_affine();
    let golden_3 = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_3).into_affine();
    assert_eq!(golden_0, msm_results[0]);
    assert_eq!(golden_1, msm_results[1]);
    assert_eq!(golden_2, msm_results[2]);
    assert_eq!(golden_3, msm_results[3]);

    unsafe {
        destroyContext(context);
    }

    // for (i, e) in scalars.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    // for (i, e) in bases.iter().enumerate() {
    //     println!("{} {:?}", i, e)
    // }

    // // scalars
    // {
    //     let filename = format!(
    //         "zprize_msm_curve_{}_scalars_dim_{}_seed_{}.csv",
    //         F::size_in_bits(),
    //         scalars.len(),
    //         args.seed,
    //     );
    //     let mut file = File::create(filename).unwrap();
    //
    //     for (i, s) in scalars.iter().enumerate() {
    //         file.write_all(format!("{}, {}\n", i, printer_256(s)).as_ref())
    //             .unwrap();
    //     }
    // }
    //
    // // base
    // {
    //     let filename = format!(
    //         "zprize_msm_curve_{}_bases_dim_{}_seed_{}.csv",
    //         F::size_in_bits(),
    //         scalars.len(),
    //         args.seed,
    //     );
    //     let mut file = File::create(filename).unwrap();
    //
    //     for (i, s) in bases.iter().enumerate() {
    //         file.write_all(
    //             format!(
    //                 "{}, x, {}, y, {}\n",
    //                 i,
    //                 printer_384(&s.x),
    //                 printer_384(&s.y)
    //             )
    //                 .as_ref(),
    //         )
    //             .unwrap();
    //     }
    // }
    //
    // // result
    // {
    //     let filename = format!(
    //         "zprize_msm_curve_{}_res_dim_{}_seed_{}.csv",
    //         F::size_in_bits(),
    //         scalars.len(),
    //         args.seed,
    //     );
    //     let mut file = File::create(filename).unwrap();
    //
    //     file.write_all(
    //         format!("x, {}, y, {}\n", printer_384(&res.x), printer_384(&res.y)).as_ref(),
    //     )
    //         .unwrap();
    // }
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