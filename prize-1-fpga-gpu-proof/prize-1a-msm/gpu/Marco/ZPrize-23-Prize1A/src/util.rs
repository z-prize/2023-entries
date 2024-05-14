// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use ark_bls12_381::G1Affine as G1Affine381;
use ark_bls12_377::G1Affine as G1Affine377;

use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{FromBytes, PrimeField};
use ark_std::UniformRand;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use rayon::iter::ParallelIterator;

pub fn generate_points_scalars<G: AffineCurve>(
    len: usize,
    batches: usize,
    is381: u32
) -> (Vec<G>, Vec<G::ScalarField>) {
    let mut fname_ext = "";
    if is381 == 1 {
        fname_ext = "381"
    }

    let bases_and_scalars_lists = (0..4).into_par_iter().map(|i | {
        (0..2).into_par_iter().flat_map(|j | {
            let file = File::open(format!("data{}-{}.bin", fname_ext, i*2 + j)).unwrap();
            (0..(1<<(24-1))).into_iter().map(|_| {
                let g = G::read(&file).unwrap();
                let s = G::ScalarField::read(&file).unwrap();
                (g, s)
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();
    let bases: Vec<_> = bases_and_scalars_lists[0].iter().map(|(g, _)| (*g).clone()).collect();
    let scalars: Vec<_> = bases_and_scalars_lists.iter().flat_map(|svec|
        svec.iter().map(|(_, s)|
            (*s).into_repr()
            // (*s)
        ).collect::<Vec<_>>()
    ).collect::<Vec<
        <<G as AffineCurve>::ScalarField as PrimeField>::BigInt
        // _
    >>();

    //
    // let rand_gen: usize = 1 << 15;
    // // let mut rng = ChaCha20Rng::from_entropy();
    // let mut rng = ChaCha20Rng::seed_from_u64(0x456231455);
    //
    // let mut bases =
    //     <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(
    //         &(0..rand_gen).into_iter()
    //             // .into_par_iter().map(|_| G::Projective::rand(&mut rng))
    //             .map(|_| G::Projective::rand(&mut rng))
    //             .collect::<Vec<_>>(),
    //     );
    // // Sprinkle in some infinity points
    // // points[3] = G::zero();
    // while bases.len() < len {
    //     bases.append(&mut bases.clone());
    // }

    //
    // let scalars = (0..len*batches)
    //     .into_iter()
    //     // .into_par_iter().map(|_| G::ScalarField::rand(&mut rng))
    //     // .map(|_| <G1Affine377 as AffineCurve>::ScalarField::rand(&mut rng))
    //     .map(|_| G::ScalarField::rand(&mut rng))
    //     .collect::<Vec<_>>();
    //
    let scalars =  unsafe {
        std::mem::transmute::<Vec<_>, Vec<<G as AffineCurve>::ScalarField>>(scalars)
    };
    (bases, scalars)
}

