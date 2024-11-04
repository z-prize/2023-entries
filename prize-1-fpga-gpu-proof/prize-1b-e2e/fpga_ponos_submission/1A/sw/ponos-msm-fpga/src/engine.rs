use ark_ec::AffineCurve;
use cxx;
use ark_ff::{BigInteger256, PrimeField};

type ArkScalar = BigInteger256;
type ArkBase12377 = ark_bls12_377::G1Affine;
type ArkBase12381 = ark_bls12_381::G1Affine;

#[cxx::bridge(namespace="ponos")]
pub mod ffi {
    struct RawArkScalar {
        pub repr: [u64; 4]
    }

    struct RawArkAffine {
        pub x: [u64; 6],
        pub y: [u64; 6],
        pub infinity: bool,
    }

    unsafe extern "C++" {
        include!("ponos-msm-fpga/src/engine/engine.hpp");

        type Engine12377;
        fn create_engine_bls12377(round_pow: usize, element_count: usize, xclbin_dir: &str) -> UniquePtr<Engine12377>;
        fn upload_bases(self: &Engine12377, bases: &[RawArkAffine]);
        fn run(self: &Engine12377, scalars: [&[RawArkScalar]; 4], result: &mut [RawArkAffine; 4]);

        type Engine12381;
        fn create_engine_bls12381(round_pow: usize, element_count: usize, xclbin_dir: &str) -> UniquePtr<Engine12381>;
        fn upload_bases(self: &Engine12381, bases: &[RawArkAffine]);
        fn run(self: &Engine12381, scalars: [&[RawArkScalar]; 4], result: &mut [RawArkAffine; 4]);
    }
}

pub trait Engine<G: AffineCurve> {
    fn upload_bases(&self, bases: &[G]);

    fn run(&self, scalars: [&[<<G as AffineCurve>::ScalarField as PrimeField>::BigInt]; 4]) -> [G; 4];
}

pub struct Engine12377 {
    ll: cxx::UniquePtr<ffi::Engine12377>,
}

pub struct Engine12381 {
    ll: cxx::UniquePtr<ffi::Engine12381>,
}

impl Engine12377 {
    pub fn new(round_pow: usize, element_count: usize, xclbin_dir: &str) -> Self {
        Self {ll: ffi::create_engine_bls12377(round_pow, element_count, xclbin_dir)}
    }
}

impl Engine<ark_bls12_377::G1Affine> for Engine12377 {
    fn upload_bases(&self, bases: &[ArkBase12377]) {
        unsafe {
            self.ll.upload_bases(std::mem::transmute(bases));
        }
    }

    fn run(&self, scalars: [&[ArkScalar]; 4]) -> [ArkBase12377; 4] {
        let mut results = [ArkBase12377::default(); 4];
        unsafe {
            self.ll.run(std::mem::transmute(scalars), std::mem::transmute(&mut results));
        }
        results
    }
}

impl Engine12381 {
    pub fn new(round_pow: usize, element_count: usize, xclbin_dir: &str) -> Self {
        Self {ll: ffi::create_engine_bls12381(round_pow, element_count, xclbin_dir)}
    }
}

impl Engine<ark_bls12_381::G1Affine> for Engine12381 {
    fn upload_bases(&self, bases: &[ArkBase12381]) {
        unsafe {
            self.ll.upload_bases(std::mem::transmute(bases));
        }
    }

    fn run(&self, scalars: [&[ArkScalar]; 4]) -> [ArkBase12381; 4] {
        let mut results = [ArkBase12381::default(); 4];
        unsafe {
            self.ll.run(std::mem::transmute(scalars), std::mem::transmute(&mut results));
        }
        results
    }
}
