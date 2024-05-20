// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use ark_bls12_377::{Fr, G1Affine};
use ark_ec::AffineCurve;
use ark_ff::PrimeField;
use ark_std::Zero;

use std::os::raw::c_void;
use std::time::SystemTime;

#[allow(unused_imports)]
use blst::*;

pub mod util;

#[repr(C)]
pub struct MultiScalarMultContext {
    context: *mut c_void,
}

#[link(name = "driver_ss")]
extern "C" {
    fn msm_init_for_ss(
        xclbin: *const u8,
        xclbin_len: usize,
        points: *const c_void,
        npoints: usize,
    ) -> *mut c_void;

    fn msm_mult_for_ss(
        context: *mut c_void,
        out: *mut c_void,
        batch_size: usize,
        scalars: *const c_void,
    ) -> ();
}

pub fn multi_scalar_mult_init_for_ss<G: AffineCurve>(
    points: &[G]
) -> MultiScalarMultContext {

    println!(" ffi multi_scalar_mult_init_for_ss points len {}", points.len());
    let now = SystemTime::now();
    let xclbin = std::env::var("XCLBIN").expect("Need an XCLBIN file");
    let ret = unsafe {
        let context = msm_init_for_ss(
            xclbin.as_ptr() as *const u8,
            xclbin.len(),
            points as *const _ as *const c_void,
            points.len(),
        );
        MultiScalarMultContext { context }
    };
    println!("ffi multi_scalar_mult_init took {:?}", now.elapsed());

    ret
}
use std::ffi::c_char;
pub fn multi_scalar_mult_for_ss<G: AffineCurve>(
    context: &mut MultiScalarMultContext,
    points: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> Vec<G::Projective> {
    let npoints = points.len();
    if scalars.len() % npoints != 0 {
        panic!("length mismatch")
    }

    let batch_size = scalars.len() / npoints;
    let mut ret = vec![G::Projective::zero(); batch_size];

    unsafe {
        let result_ptr = ret.as_mut_ptr();
        msm_mult_for_ss(
            context.context,
            result_ptr as *mut c_void,
            batch_size,
            scalars as *const _ as *const c_void,
        );
    };

    ret
}
