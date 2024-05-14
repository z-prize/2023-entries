// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0


// const PRECOMPUTE_CNT :usize= 11;
// const WINDOW_BITS :usize = 23;
//
// use std::fs::File;
// use ark_bls12_377::G1Affine;

use std::ptr;
use ark_bls12_381::G1Affine;
const PRECOMPUTE_CNT: usize = 12;
const WINDOW_BITS :usize = 22;

use ark_bls12_381::G1Affine as G1Affine381;
use ark_bls12_381::G1Affine as G1Affine377;
use ark_ec::msm::VariableBaseMSM;
use ark_ff::{BigInteger256, PrimeField};
use ark_ec::{AffineCurve, ProjectiveCurve};

use ark_ff::prelude::*;
use ark_std::vec::Vec;
use num_cpus;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use ark_ec::twisted_edwards_extended::GroupAffine;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelSlice;

use blst_msm::*;
use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;
pub struct VariableBaseMSM2;
fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}

impl VariableBaseMSM2 {
    pub fn preprocess_points<G: AffineCurve>(
        bases: &[G],
    ) -> Vec<Vec<G>> {
        let num = num_cpus::get();
        let res = bases.par_chunks(bases.len()/(num*32)).map(|chunks| {
            let chunk_res = chunks.iter().map(|base| {
                let mut precompute_points = Vec::with_capacity(PRECOMPUTE_CNT);
                let mut base = base.clone();
                precompute_points.push(base);

                for i in 1..PRECOMPUTE_CNT {
                    // let mut proj_base = base.into_projective();
                    // for _ in 0..WINDOW_BITS {
                    //     proj_base.double_in_place();
                    // }
                    // base = proj_base.into_affine();
                    precompute_points.push(base);
                }
                precompute_points
            }).collect::<Vec<Vec<_>>>();
            chunk_res
        }).flatten().collect::<Vec<Vec<_>>>();

        res
    }

    pub fn multi_scalar_mul<G: AffineCurve>(
        bases: &[G],
        precompute_points: &Vec<Vec<G>>,
        scalars: &[BigInteger256],
        batch: usize,
    ) -> G::Projective {
        let size = ark_std::cmp::min(bases.len(), scalars.len());
        let scalars = &scalars[..size];
        let bases = &bases[..size];
        let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

        // let c = if size < 32 {
        //     3
        // } else {
        //     ln_without_floats(size) + 2
        // };
        let win_bits = WINDOW_BITS;

        // let num_bits : usize = 253;
        let num_bits = <G::ScalarField as PrimeField>::Params::MODULUS_BITS as usize;
        // let modulus = <<G1Affine377 as AffineCurve>::ScalarField as PrimeField>::Params::MODULUS;
        // let modulus = <<G as AffineCurve>::ScalarField as PrimeField>::Params::MODULUS;

        // let fr_one = ark_ff::BigInteger256([
        //     9015221291577245683u64,
        //     8239323489949974514u64,
        //     1646089257421115374u64,
        //     958099254763297437u64,
        // ]);

        let fr_one = G::ScalarField::one().into_repr();

        let zero = G::Projective::zero();
        let window_starts: Vec<_> = (0..num_bits).step_by(win_bits).collect();
        // println!("window_starts len: {}", window_starts.len());

        // // let precompute_scalars = Vec::with_capacity(scalars.len());
        // let precompute_scalars = scalars.par_chunks(scalars.len()/(num_cpus::get()*32))
        //     .enumerate()
        //     .map(|(sid, chunks)| {
        //     let chunk_res = chunks.iter().enumerate().map(|(cid, scalar0)| {
        //         let mut precompute_scalars = Vec::with_capacity(11);
        //         let mut is_neg = false;
        //         let mut scalar0 = *scalar0;
        //         // if sid ==0 && cid == 0 {
        //         //     let val : [u32; 8] = unsafe {std::mem::transmute(scalar0.0)};
        //         //     for (i, v) in val.iter().enumerate() {
        //         //         println!("v{}, {:x}", i, v);
        //         //     }
        //         //
        //         // }
        //         if scalar0.get_bit(num_bits-1) {
        //             let mut modulus1 = modulus.clone();
        //             modulus1.sub_noborrow(&scalar0);
        //             scalar0 = modulus1;
        //             is_neg = true;
        //         }
        //         // if sid ==0 && cid == 0 {
        //         //     let val : [u32; 8] = unsafe {std::mem::transmute(scalar0.0)};
        //         //     for (i, v) in val.iter().enumerate() {
        //         //         println!("2v{}, {:x}", i, v);
        //         //     }
        //         //
        //         // }
        //
        //         // if sid==0&&cid==0 {println!("is neg: {}", is_neg); }
        //
        //         for i in 0..11 {
        //             let mut scalar = scalar0.clone();
        //             scalar.divn(i * win_bits as u32);
        //
        //             // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
        //             let scalar = scalar.as_ref()[0] % (1 << win_bits);
        //             precompute_scalars.push(scalar);
        //         }
        //
        //         let slen = precompute_scalars.len();
        //         let sliced = &mut precompute_scalars;
        //
        //         for i in 0..slen {
        //             // if sid==0&&cid==0 {println!("scalar{}: {}, len: {}", i, sliced[i], slen); }
        //             // for scalar in precompute_scalars {
        //             if sliced[i] < (1<<(win_bits-1)) {
        //                 if is_neg && sliced[i] != 0 {
        //                     sliced[i] += (1<<(win_bits));
        //                 }
        //             } else if sliced[i] < (1<<(win_bits)) {
        //                 sliced[i]=(sliced[i] ^ ((1<<win_bits)-1)) + 1;
        //                 if(!is_neg) {
        //                     sliced[i] += 1 << (win_bits);
        //                 }
        //                 sliced[i+1] += 1;
        //             }
        //             else {
        //                 sliced[i]=0;
        //                 sliced[i+1] += 1;
        //             }
        //         }
        //         precompute_scalars
        //     }).collect::<Vec<Vec<_>>>();
        //     chunk_res
        // }).flatten().collect::<Vec<Vec<_>>>();

        // Each window is of size `c`.
        // We divide up the bits 0..num_bits into windows of size `c`, and
        // in parallel process each such window.
        // let window_sums: Vec<_> = ark_std::cfg_into_iter!(window_starts)

        let total_bucket0_cnt = Arc::new(Mutex::new(0));

        let windows_sums: Vec<_> = window_starts.into_par_iter()
            .enumerate()
            .map(|(win, w_start)| {
                let mut res = zero;
                // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
                let mut buckets = vec![zero; (1 << (win_bits)) - 1];
                // This clone is cheap, because the iterator contains just a
                // pointer and an index into the original vectors.
                // scalars.iter().zip(bases).for_each(|(&scalar, base) | {
                // scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {

                let mut bucket0_cnt = 0;
                for i in 0..scalars.len() {
                    let scalar = scalars[i];
                    let base = &bases[i];
                    // if scalar == fr_one {
                    //     // We only process unit scalars once in the first window.
                    //     if w_start == 0 {
                    //         res.add_assign_mixed(&bases[i]);
                    //     }
                    // } else
                    {
                        let mut scalar = scalar;
                        // let mut is_neg = false;
                        // if scalar.get_bit(<G::ScalarField as PrimeField>::Params::MODULUS_BITS as usize) {
                        //     scalar = -scalar;
                        //     is_neg = true;
                        // }
                        // We right-shift by w_start, thus getting rid of the
                        // lower bits.
                        scalar.divn(w_start as u32);

                        // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                        let scalar = scalar.as_ref()[0] % (1 << win_bits);
                        let mut base = precompute_points[i][win];

                        // let mut scalar = precompute_scalars[i][win];
                        // if i==0 {println!("scalar2, {}: {}", win, scalar); }
                        // if scalar & (1<<(win_bits)) != 0 {
                        //     scalar &= (1<<(win_bits))-1;
                        //     base = base.neg();
                        // }
                        if i==0 {println!("scalar3, batch: {}, {}: {}", batch, win, scalar); }

                        // If the scalar is non-zero, we update the corresponding
                        // bucket.
                        // (Recall that `buckets` doesn't have a zero bucket.)
                        // if i == 191925 {
                        //     println!("{}, 4pointIndex, win: {}, scalar: {}", win*(1<<24) + i, win, scalar);
                        // }

                        if scalar != 0 {
                            if scalar == 1 {
                                bucket0_cnt += 1;

                                {
                                    let res = base;
                                    let bytes_ptr :*const G = &res as *const G;
                                    let addr = bytes_ptr as usize;
                                    let bytes = unsafe {ptr::read(addr as *const [u8; 96])};

                                    // let bytes = ark_res.to_bytes();
                                    print!("point1: ");
                                    unsafe  {
                                        for byte in bytes {
                                            print!("{:02x}", byte);
                                        }
                                        println!();
                                    }
                                }

                                println!("{}, 1pointIndex", win*(1<<24) + i);
                            }

                            buckets[(scalar - 1) as usize].add_assign_mixed(&base);
                        }
                    }
                }
                {println!("bucket0, batch: {}, win: {}, count: {}", batch, win, bucket0_cnt); }
                let mut num = total_bucket0_cnt.lock().unwrap();
                *num += bucket0_cnt;
                // );

                // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
                // This is computed below for b buckets, using 2b curve additions.
                //
                // We could first normalize `buckets` and then use mixed-addition
                // here, but that's slower for the kinds of groups we care about
                // (Short Weierstrass curves and Twisted Edwards curves).
                // In the case of Short Weierstrass curves,
                // mixed addition saves ~4 field multiplications per addition.
                // However normalization (with the inversion batched) takes ~6
                // field multiplications per element,
                // hence batch normalization is a slowdown.

                // `running_sum` = sum_{j in i..num_buckets} bucket[j],
                // where we iterate backward from i = num_buckets to 0.
                // let mut running_sum = G::Projective::zero();
                // buckets.into_iter().rev().for_each(|b| {
                //     running_sum += &b;
                //     res += &running_sum;
                // });
                // res
                buckets
            })
            .collect();

        println!("bucket0: total, batch: {}, count: {}", batch, *total_bucket0_cnt.lock().unwrap());
        assert_eq!(windows_sums.len(), PRECOMPUTE_CNT);

        let bucket_sums = vec![zero; windows_sums[0].len()];
        let chunk_sz = bucket_sums.len()/(num_cpus::get()*32);
        let bucket_sums = (0..bucket_sums.len()).into_par_iter()
            .chunks(chunk_sz)
            .enumerate()
            .map(|(sid, chunks)| {
            let chunk_res = chunks.iter().enumerate().map(|(cid, _)| {
                let mut acc = zero;
                for i in 0..windows_sums.len() {
                    acc += &windows_sums[i][sid*chunk_sz+cid];
                }
                acc
            }).collect::<Vec<_>>();
            chunk_res
        }).flatten().collect::<Vec<_>>();

        {
            let res = bucket_sums[0].into_affine();
            let bytes_ptr :*const G = &res as *const G;
            let addr = bytes_ptr as usize;
            let bytes = unsafe {ptr::read(addr as *const [u8; 104])};

            // let bytes = ark_res.to_bytes();
            print!("in bucket0: ");
            unsafe  {
                for byte in bytes {
                    print!("{:02x}", byte);
                }
                println!();
            }
        }
        let mut res = zero;
        let mut running_sum = G::Projective::zero();
        bucket_sums.into_iter().rev().for_each(|b| {
            running_sum += &b;
            res += &running_sum;
        });
        res

        // // We store the sum for the lowest window.
        // let lowest = *window_sums.first().unwrap();
        //
        // // We're traversing windows from high to low.
        // lowest
        //     + &window_sums[1..]
        //     .iter()
        //     .rev()
        //     .fold(zero, |mut total, sum_i| {
        //         total += sum_i;
        //         // for _ in 0..c {
        //         //     total.double_in_place();
        //         // }
        //         total
        //     })
    }
    pub fn multi_scalar_mul3<G: AffineCurve>(
        bases: &[G],
        precompute_points: &Vec<Vec<G>>,
        scalars: &[BigInteger256],
    ) -> G::Projective {
        let size = ark_std::cmp::min(bases.len(), scalars.len());
        let scalars = &scalars[..size];
        let bases = &bases[..size];
        let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

        // let c = if size < 32 {
        //     3
        // } else {
        //     ln_without_floats(size) + 2
        // };
        let win_bits = 23;

        // let num_bits : usize = 253;
        let num_bits = <G::ScalarField as PrimeField>::Params::MODULUS_BITS as usize;
        // let modulus = <<G1Affine377 as AffineCurve>::ScalarField as PrimeField>::Params::MODULUS;
        // let modulus = <<G as AffineCurve>::ScalarField as PrimeField>::Params::MODULUS;

        // let fr_one = ark_ff::BigInteger256([
        //     9015221291577245683u64,
        //     8239323489949974514u64,
        //     1646089257421115374u64,
        //     958099254763297437u64,
        // ]);

        let fr_one = G::ScalarField::one().into_repr();

        let zero = G::Projective::zero();
        let window_starts: Vec<_> = (0..num_bits).step_by(win_bits).collect();

        // // let precompute_scalars = Vec::with_capacity(scalars.len());
        // let precompute_scalars = scalars.par_chunks(scalars.len()/(num_cpus::get()*32))
        //     .enumerate()
        //     .map(|(sid, chunks)| {
        //     let chunk_res = chunks.iter().enumerate().map(|(cid, scalar0)| {
        //         let mut precompute_scalars = Vec::with_capacity(11);
        //         let mut is_neg = false;
        //         let mut scalar0 = *scalar0;
        //         // if sid ==0 && cid == 0 {
        //         //     let val : [u32; 8] = unsafe {std::mem::transmute(scalar0.0)};
        //         //     for (i, v) in val.iter().enumerate() {
        //         //         println!("v{}, {:x}", i, v);
        //         //     }
        //         //
        //         // }
        //         if scalar0.get_bit(num_bits-1) {
        //             let mut modulus1 = modulus.clone();
        //             modulus1.sub_noborrow(&scalar0);
        //             scalar0 = modulus1;
        //             is_neg = true;
        //         }
        //         // if sid ==0 && cid == 0 {
        //         //     let val : [u32; 8] = unsafe {std::mem::transmute(scalar0.0)};
        //         //     for (i, v) in val.iter().enumerate() {
        //         //         println!("2v{}, {:x}", i, v);
        //         //     }
        //         //
        //         // }
        //
        //         // if sid==0&&cid==0 {println!("is neg: {}", is_neg); }
        //
        //         for i in 0..11 {
        //             let mut scalar = scalar0.clone();
        //             scalar.divn(i * win_bits as u32);
        //
        //             // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
        //             let scalar = scalar.as_ref()[0] % (1 << win_bits);
        //             precompute_scalars.push(scalar);
        //         }
        //
        //         let slen = precompute_scalars.len();
        //         let sliced = &mut precompute_scalars;
        //
        //         for i in 0..slen {
        //             // if sid==0&&cid==0 {println!("scalar{}: {}, len: {}", i, sliced[i], slen); }
        //             // for scalar in precompute_scalars {
        //             if sliced[i] < (1<<(win_bits-1)) {
        //                 if is_neg && sliced[i] != 0 {
        //                     sliced[i] += (1<<(win_bits));
        //                 }
        //             } else if sliced[i] < (1<<(win_bits)) {
        //                 sliced[i]=(sliced[i] ^ ((1<<win_bits)-1)) + 1;
        //                 if(!is_neg) {
        //                     sliced[i] += 1 << (win_bits);
        //                 }
        //                 sliced[i+1] += 1;
        //             }
        //             else {
        //                 sliced[i]=0;
        //                 sliced[i+1] += 1;
        //             }
        //         }
        //         precompute_scalars
        //     }).collect::<Vec<Vec<_>>>();
        //     chunk_res
        // }).flatten().collect::<Vec<Vec<_>>>();

        // Each window is of size `c`.
        // We divide up the bits 0..num_bits into windows of size `c`, and
        // in parallel process each such window.
        // let window_sums: Vec<_> = ark_std::cfg_into_iter!(window_starts)

        let window_sums: Vec<_> = window_starts.into_par_iter()
            .enumerate()
            .map(|(win, w_start)| {
                let mut res = zero;
                // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
                let mut buckets = vec![zero; (1 << (win_bits)) - 1];
                // This clone is cheap, because the iterator contains just a
                // pointer and an index into the original vectors.
                // scalars.iter().zip(bases).for_each(|(&scalar, base) | {
                // scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
                for i in 0..scalars.len() {
                    let scalar = scalars[i];
                    let base = &bases[i];
                    // if scalar == fr_one {
                    //     // We only process unit scalars once in the first window.
                    //     if w_start == 0 {
                    //         res.add_assign_mixed(&bases[i]);
                    //     }
                    // } else
                    {
                        let mut scalar = scalar;
                        // let mut is_neg = false;
                        // if scalar.get_bit(<G::ScalarField as PrimeField>::Params::MODULUS_BITS as usize) {
                        //     scalar = -scalar;
                        //     is_neg = true;
                        // }
                        // We right-shift by w_start, thus getting rid of the
                        // lower bits.
                        scalar.divn(w_start as u32);

                        // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                        let scalar = scalar.as_ref()[0] % (1 << win_bits);
                        let mut base = precompute_points[i][win];

                        // let mut scalar = precompute_scalars[i][win];
                        // if i==0 {println!("scalar2, {}: {}", win, scalar); }
                        // if scalar & (1<<(win_bits)) != 0 {
                        //     scalar &= (1<<(win_bits))-1;
                        //     base = base.neg();
                        // }
                        // if i==0 {println!("scalar3, {}: {}", win, scalar); }

                        // If the scalar is non-zero, we update the corresponding
                        // bucket.
                        // (Recall that `buckets` doesn't have a zero bucket.)
                        if scalar != 0 {
                            buckets[(scalar - 1) as usize].add_assign_mixed(&base);
                        }
                    }
                }
                // );

                // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
                // This is computed below for b buckets, using 2b curve additions.
                //
                // We could first normalize `buckets` and then use mixed-addition
                // here, but that's slower for the kinds of groups we care about
                // (Short Weierstrass curves and Twisted Edwards curves).
                // In the case of Short Weierstrass curves,
                // mixed addition saves ~4 field multiplications per addition.
                // However normalization (with the inversion batched) takes ~6
                // field multiplications per element,
                // hence batch normalization is a slowdown.

                // `running_sum` = sum_{j in i..num_buckets} bucket[j],
                // where we iterate backward from i = num_buckets to 0.
                let mut running_sum = G::Projective::zero();
                buckets.into_iter().rev().for_each(|b| {
                    running_sum += &b;
                    res += &running_sum;
                });
                res
            })
            .collect();

        assert_eq!(window_sums.len(), 11);
        // We store the sum for the lowest window.
        let lowest = *window_sums.first().unwrap();

        // We're traversing windows from high to low.
        lowest
            + &window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
                total += sum_i;
                // for _ in 0..c {
                //     total.double_in_place();
                // }
                total
            })
    }

    pub fn multi_scalar_mul2<G: AffineCurve>(
        bases: &[G],
        precompute_points: &Vec<Vec<G>>,
        scalars: &[BigInteger256],
    ) -> G::Projective {
        let size = ark_std::cmp::min(bases.len(), scalars.len());
        let scalars = &scalars[..size];
        let bases = &bases[..size];
        let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _)| !s.is_zero());

        // let c = if size < 32 {
        //     3
        // } else {
        //     ln_without_floats(size) + 2
        // };
        let win_bits = 23;
        // let num_bits = <G::ScalarField as PrimeField>::Params::MODULUS_BITS as usize;
        let modulus = <<G1Affine377 as AffineCurve>::ScalarField as PrimeField>::Params::MODULUS;

        let num_bits : usize = 253;
        let fr_one = ark_ff::BigInteger256([
            9015221291577245683u64,
            8239323489949974514u64,
            1646089257421115374u64,
            958099254763297437u64,
        ]);

        // let fr_one = G::ScalarField::from_repr(&fr_one).unwrap().into_repr();

        let zero = G::Projective::zero();
        let window_starts: Vec<_> = (0..num_bits).step_by(win_bits).collect();

        // let precompute_scalars = Vec::with_capacity(scalars.len());
        let precompute_scalars = scalars.par_chunks(scalars.len()/(num_cpus::get()*32))
            .enumerate()
            .map(|(sid, chunks)| {
                let chunk_res = chunks.iter().enumerate().map(|(cid, scalar0)| {
                    let mut precompute_scalars = Vec::with_capacity(11);
                    let mut is_neg = false;
                    let mut scalar0 = *scalar0;
                    // if sid ==0 && cid == 0 {
                    //     let val : [u32; 8] = unsafe {std::mem::transmute(scalar0.0)};
                    //     for (i, v) in val.iter().enumerate() {
                    //         println!("v{}, {:x}", i, v);
                    //     }
                    //
                    // }
                    if scalar0.get_bit(num_bits-1) {
                        let mut modulus1 = modulus.clone();
                        modulus1.sub_noborrow(&scalar0);
                        scalar0 = modulus1;
                        is_neg = true;
                    }
                    // if sid ==0 && cid == 0 {
                    //     let val : [u32; 8] = unsafe {std::mem::transmute(scalar0.0)};
                    //     for (i, v) in val.iter().enumerate() {
                    //         println!("2v{}, {:x}", i, v);
                    //     }
                    //
                    // }

                    // if sid==0&&cid==0 {println!("is neg: {}", is_neg); }

                    for i in 0..11 {
                        let mut scalar = scalar0.clone();
                        scalar.divn(i * win_bits as u32);

                        // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                        let scalar = scalar.as_ref()[0] % (1 << win_bits);
                        precompute_scalars.push(scalar);
                    }

                    let slen = precompute_scalars.len();
                    let sliced = &mut precompute_scalars;

                    for i in 0..slen {
                        // if sid==0&&cid==0 {println!("scalar{}: {}, len: {}", i, sliced[i], slen); }
                        // for scalar in precompute_scalars {
                        if sliced[i] < (1<<(win_bits-1)) {
                            if is_neg && sliced[i] != 0 {
                                sliced[i] += (1<<(win_bits));
                            }
                        } else if sliced[i] < (1<<(win_bits)) {
                            sliced[i]=(sliced[i] ^ ((1<<win_bits)-1)) + 1;
                            if(!is_neg) {
                                sliced[i] += 1 << (win_bits);
                            }
                            sliced[i+1] += 1;
                        }
                        else {
                            sliced[i]=0;
                            sliced[i+1] += 1;
                        }
                    }
                    precompute_scalars
                }).collect::<Vec<Vec<_>>>();
                chunk_res
            }).flatten().collect::<Vec<Vec<_>>>();

        // Each window is of size `c`.
        // We divide up the bits 0..num_bits into windows of size `c`, and
        // in parallel process each such window.
        // let window_sums: Vec<_> = ark_std::cfg_into_iter!(window_starts)
        let window_sums: Vec<_> = window_starts.into_par_iter()
            .enumerate()
            .map(|(win, w_start)| {
                let mut res = zero;
                // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
                let mut buckets = vec![zero; (1 << (win_bits)) - 1];
                // This clone is cheap, because the iterator contains just a
                // pointer and an index into the original vectors.
                // scalars.iter().zip(bases).for_each(|(&scalar, base) | {
                // scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
                for i in 0..scalars.len() {
                    let scalar = scalars[i];
                    let base = &bases[i];
                    // if scalar == fr_one {
                    //     // We only process unit scalars once in the first window.
                    //     if w_start == 0 {
                    //         res.add_assign_mixed(&bases[i]);
                    //     }
                    // } else
                    {
                        let mut scalar = scalar;
                        // let mut is_neg = false;
                        // if scalar.get_bit(<G::ScalarField as PrimeField>::Params::MODULUS_BITS as usize) {
                        //     scalar = -scalar;
                        //     is_neg = true;
                        // }
                        // We right-shift by w_start, thus getting rid of the
                        // lower bits.
                        scalar.divn(w_start as u32);

                        // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
                        let scalar = scalar.as_ref()[0] % (1 << win_bits);
                        let mut base = precompute_points[i][win];

                        // let mut scalar = precompute_scalars[i][win];
                        // If the scalar is non-zero, we update the corresponding
                        // bucket.
                        // (Recall that `buckets` doesn't have a zero bucket.)
                        // assert!(scalar < (1<<win_bits));

                        // if i==0 {println!("scalar2, {}: {}", win, scalar); }
                        // if scalar & (1<<(win_bits)) != 0 {
                        //     scalar &= (1<<(win_bits))-1;
                        //     base = base.neg();
                        // }
                        // if i==0 {println!("scalar3, {}: {}", win, scalar); }

                        if scalar != 0 {
                            buckets[(scalar - 1) as usize].add_assign_mixed(&base);
                        }
                    }
                }
                // );

                // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
                // This is computed below for b buckets, using 2b curve additions.
                //
                // We could first normalize `buckets` and then use mixed-addition
                // here, but that's slower for the kinds of groups we care about
                // (Short Weierstrass curves and Twisted Edwards curves).
                // In the case of Short Weierstrass curves,
                // mixed addition saves ~4 field multiplications per addition.
                // However normalization (with the inversion batched) takes ~6
                // field multiplications per element,
                // hence batch normalization is a slowdown.

                // `running_sum` = sum_{j in i..num_buckets} bucket[j],
                // where we iterate backward from i = num_buckets to 0.
                let mut running_sum = G::Projective::zero();
                buckets.into_iter().rev().for_each(|b| {
                    running_sum += &b;
                    res += &running_sum;
                });
                res
            })
            .collect();

        assert_eq!(window_sums.len(), 11);
        // We store the sum for the lowest window.
        let lowest = *window_sums.first().unwrap();

        // We're traversing windows from high to low.
        lowest
            + &window_sums[1..]
            .iter()
            .rev()
            .fold(zero, |mut total, sum_i| {
                total += sum_i;
                // for _ in 0..c {
                //     total.double_in_place();
                // }
                total
            })
    }
}

#[test]
fn msm_correctness() {
    let test_npow = std::env::var("TEST_NPOW").unwrap_or("24".to_string());
    let npoints_npow = i32::from_str(&test_npow).unwrap();
    let is381 = match <<G1Affine as AffineCurve>::BaseField as PrimeField>::size_in_bits() {
        381 => 1,
        377 => 0,
        _ => panic!("not supported"),
    };

    let batches = 4;
    let (points, scalars) =
        util::generate_points_scalars::<G1Affine>(1usize << npoints_npow, batches, is381);

    println!("bases0: {}, scalar0: {}", points[0], scalars[0]);
    println!("bases1: {}, scalar1: {}", points[1], scalars[1]);

    let bytes: [u8; std::mem::size_of::<<G1Affine as AffineCurve>::ScalarField>()] = unsafe { std::mem::transmute(scalars[0]) };
    print!("in scalar0: ");
    for byte in bytes.iter() {
        print!("{:02x}", byte);
    }
    println!();

    let mut context = multi_scalar_mult_init(points.as_slice(), batches as u32, is381);
    let msm_results = multi_scalar_mult(&mut context, points.as_slice(), unsafe {
        std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
    });
    for b in 0..batches {
        let bytes: [u8; std::mem::size_of::<G1Affine>()] = unsafe { std::mem::transmute(msm_results[b].into_affine()) };
        print!("in msm: {}, ", b);
        for byte in bytes.iter() {
            print!("{:02x}", byte);
        }
        println!();
    }

    // let bytes: [u8; std::mem::size_of::<G1Affine>()] = unsafe { std::mem::transmute(points[0]) };
    // print!("in rust: ");
    // for byte in bytes.iter() {
    //     print!("{:02x}", byte);
    // }
    // println!();
    //
    // let mut aff = points[0];
    // for i in 1..11 {
    //     for j in 0..23 {
    //         aff = aff.into_projective().double().into_affine();
    //     }
    //
    //     let bytes: [u8; std::mem::size_of::<G1Affine>()] = unsafe { std::mem::transmute(aff) };
    //     print!("in aff: {}, ", i);
    //     for byte in bytes.iter() {
    //         print!("{:02x}", byte);
    //     }
    //     println!();
    // }

    let precompute_points = VariableBaseMSM2::preprocess_points(points.as_slice());
    precompute_points[0].iter().enumerate().for_each(|(i, &point)| {
    let bytes: [u8; std::mem::size_of::<G1Affine>()] = unsafe { std::mem::transmute(point) };
        print!("in aff: {}, ", i);
        for byte in bytes.iter() {
            print!("{:02x}", byte);
        }
        println!();
    });

    // precompute_points.iter().map()
    for b in 0..batches {
        let start = b * points.len();
        let end = (b + 1) * points.len();

        let arkworks_result =
            VariableBaseMSM2::multi_scalar_mul(points.as_slice(), &precompute_points, unsafe {
                std::mem::transmute::<&[_], &[BigInteger256]>(&scalars[start..end])
        }, b)
        .into_affine();

        // let arkworks2_result =
        //     VariableBaseMSM::multi_scalar_mul(points.as_slice(), unsafe {
        //         std::mem::transmute::<&[_], &[BigInteger256]>(&scalars[start..end])
        //     })
        //         .into_affine();
        // assert_eq!(arkworks2_result, arkworks_result);


        let bytes: [u8; std::mem::size_of::<G1Affine>()] = unsafe { std::mem::transmute(arkworks_result) };
        print!("in ark: {}, ", b);
        for byte in bytes.iter() {
            print!("{:02x}", byte);
        }
        println!();

        // if (b > 0)
        {
            assert_eq!(msm_results[b].into_affine(), arkworks_result);

        }
    }
}
