// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use ark_bls12_377::{Fr, G1Affine, G1Projective};
use ark_ec::ProjectiveCurve;
use ark_ff::{BigInteger256, PrimeField};
use ark_std::{rand, UniformRand};
use criterion::{criterion_group, criterion_main, Criterion};
use crypto_cuda::HostMemory;
use ec_gpu_common::{gpu_msm, MsmPrecalcContainer, MultiexpKernel, GPU_CUDA_CORES};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::str::FromStr;

const GPU_DEV_IDX: usize = 0;

fn generate_bases_affine(count: usize) -> Vec<G1Affine> {
    const INIT_LEN: usize = 1 << 16;

    let mut rng = ChaCha20Rng::from_entropy();
    assert!(count >= INIT_LEN);
    let copy_count = count / INIT_LEN;

    let mut result = (0..INIT_LEN)
        .map(|_| G1Projective::rand(&mut rng))
        .map(|gp| gp.into_affine())
        .collect::<Vec<_>>();

    let copy_result = result.clone();
    for _ in 1..copy_count {
        result.extend(copy_result.iter());
    }

    result
}

fn generate_scalars(count: usize) -> HostMemory<BigInteger256> {
    let mut rng = ChaCha20Rng::from_entropy();

    let mut scalars = HostMemory::<BigInteger256>::new(count).unwrap();

    for index in 0..count {
        scalars.as_mut_slice()[index] = Fr::rand(&mut rng).into_repr();
    }

    scalars
}

fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("24".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();

    let batches = 4;

    let precalc_container =
        MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX], true).unwrap();
    let kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();

    let points = generate_bases_affine(1 << npoints_npow);
    let scalars = generate_scalars(1 << npoints_npow);
    let mut msm_ctx = kern.multi_scalar_mult_init(points.as_slice());

    let mut group = c.benchmark_group("GPU-MSM");
    group.sample_size(10);

    let name = format!("2**{}x{}", npoints_npow, batches);
    group.bench_function(name, |b| {
        b.iter(|| {
            let _ = gpu_msm(&scalars, &mut msm_ctx);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
