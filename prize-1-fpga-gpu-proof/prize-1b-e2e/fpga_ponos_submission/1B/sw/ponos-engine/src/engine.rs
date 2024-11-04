use std::cmp;

use crate::utils::{transmute_ref, transmute_ref_mut, transmute_slice};
use ark_bls12_381::G1Affine;
pub use ark_ec::{msm::VariableBaseMSM, AffineCurve, ProjectiveCurve};
use ark_ff::BigInteger384;
pub use ark_ff::{BigInteger256, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, Write, SerializationError};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub type ArkBase12381 = ark_bls12_381::G1Affine;
pub type ArkProjectiveBase12381 = ark_bls12_381::G1Projective;

const MIN_FPGA_ELEMENT_COUNT: usize = 8192;

// Define PAGE_SIZE as 4096 bytes for 4KB alignment
const G1_381_TYPE_NAME: &str = "ark_ec::models::short_weierstrass_jacobian::GroupAffine<ark_bls12_381::curves::g1::Parameters>";

#[cxx::bridge(namespace = "ponos")]
pub mod ffi {
    #[derive(Default, Copy, Clone, PartialEq, Hash, Debug)]
    struct RawScalar256 {
        pub repr: [u64; 4],
    }

    #[derive(Default, Copy, Clone, PartialEq, Hash, Debug)]
    struct RawScalar384 {
        pub repr: [u64; 6],
    }

    #[derive(Default, Copy, Clone, PartialEq, Hash, Debug)]
    struct RawScalar384Padded {
        pub value: RawScalar384,
        pub padding: [u64; 2],
    }

    #[derive(Default, Copy, Clone, PartialEq, Hash, Debug)]
    struct RawArkAffine {
        pub x: [u64; 6],
        pub y: [u64; 6],
        pub infinity: bool,
    }

    #[derive(Default, Copy, Clone, PartialEq, Hash, Debug)]
    struct NBase384 {
        pub x: RawScalar384,
        pub y: RawScalar384,
    }

    #[derive(Default, Copy, Clone, PartialEq, Hash, Debug)]
    struct NBase384Padded {
        pub x: RawScalar384Padded,
        pub y: RawScalar384Padded,
    }

    #[derive(Default, Copy, Clone, PartialEq, Hash, Debug)]
    struct RawArkCombo12381 {
        pub scalar: RawScalar256,
        pub base: NBase384,
    }

    unsafe extern "C++" {
        include!("ponos-engine/src/engine/engine.hpp");

        fn run_fpga_msm_vitis_mont(scalars: &[RawScalar256], bases: &[RawArkAffine], output: &mut RawArkAffine);
        fn run_fpga_msm_vitis_normal(scalars: &[RawScalar256], bases: &[NBase384], output: &mut RawArkAffine);
    }
}

pub use ffi::RawScalar256;
pub use ffi::RawScalar384;
pub use ffi::RawScalar384Padded;
pub use ffi::NBase384;
pub use ffi::NBase384Padded;
pub use ffi::RawArkCombo12381;

impl NBase384 {
    fn infinity() -> Self {
        INFINITY_NBASE384
    }
}

impl CanonicalSerialize for NBase384 {
    fn serialized_size(&self) -> usize {
        2*48
    }

    fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
        unsafe {
            writer.write_all(std::mem::transmute(self.x.repr.as_slice()))?;
            writer.write_all(std::mem::transmute(self.y.repr.as_slice()))?;
            Ok(())
        }
    }
}

impl CanonicalDeserialize for NBase384 {
    fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
        let mut p = Self::infinity();
        unsafe {
            reader.read_exact(std::mem::transmute(p.x.repr.as_mut_slice()))?;
            reader.read_exact(std::mem::transmute(p.y.repr.as_mut_slice()))?;
        }
        Ok(p)
    }
}

pub const INFINITY_NBASE384: NBase384 =
    NBase384 {
        x: RawScalar384 { repr: [0xffffffffffffffffu64, 0xffffffffffffffffu64, 0xffffffffffffffffu64, 0xffffffffffffffffu64, 0xffffffffffffffffu64, 0x1fffffffffffffffu64] },
        y: RawScalar384 { repr: [0xffffffffffffffffu64, 0xffffffffffffffffu64, 0xffffffffffffffffu64, 0xffffffffffffffffu64, 0xffffffffffffffffu64, 0x1fffffffffffffffu64] },
    };

fn to_normal_381(base: &G1Affine) -> NBase384 {
    if !base.infinity {
        NBase384 {
            x: RawScalar384 { repr: Into::<BigInteger384>::into(base.x).0 },
            y: RawScalar384 { repr: Into::<BigInteger384>::into(base.y).0 },
        }
    }
    else {
        INFINITY_NBASE384
    }
}

pub fn to_normal<G: AffineCurve>(base: &G) -> NBase384 {
    assert!(std::any::type_name::<G>() == G1_381_TYPE_NAME);
    let base = unsafe { transmute_ref::<G1Affine, G>(base) };

    to_normal_381(base)
}

pub fn to_normal_vec<G: AffineCurve>(bases: &[G]) -> Vec<NBase384> {
    assert!(std::any::type_name::<G>() == G1_381_TYPE_NAME);
    let bases = unsafe { transmute_slice::<G1Affine, G>(bases) };

    bases.into_par_iter().map(to_normal_381).collect()
}

pub fn run_msm<G: AffineCurve>(
    bases: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> G::Projective
{
    let size = cmp::min(bases.len(), scalars.len());
    if size == 0 {
        return G::Projective::default();
    }
    if size < MIN_FPGA_ELEMENT_COUNT {
        return VariableBaseMSM::multi_scalar_mul(bases, scalars);
    }

    assert!(std::any::type_name::<G>() == G1_381_TYPE_NAME);

    let mut result = G::default();
    unsafe {
        ffi::run_fpga_msm_vitis_mont(
            transmute_slice::<ffi::RawScalar256, <G::ScalarField as PrimeField>::BigInt>(scalars),
            transmute_slice::<ffi::RawArkAffine, G>(bases),
            transmute_ref_mut::<ffi::RawArkAffine, G>(&mut result),
        );
    }
    result.into_projective()
}

pub fn run_msm_opt<G: AffineCurve>(
    bases: &[G],
    n_bases: &[NBase384],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> G::Projective
{
    let size = cmp::min(bases.len(), scalars.len());
    if size == 0 {
        return G::Projective::default();
    }
    if size < MIN_FPGA_ELEMENT_COUNT {
        return VariableBaseMSM::multi_scalar_mul(bases, scalars);
    }

    assert!(std::any::type_name::<G>() == G1_381_TYPE_NAME);

    let mut result = G::default();
    unsafe {
        ffi::run_fpga_msm_vitis_normal(
            transmute_slice::<ffi::RawScalar256, <G::ScalarField as PrimeField>::BigInt>(scalars),
            n_bases,
            transmute_ref_mut::<ffi::RawArkAffine, G>(&mut result),
        );
    }
    result.into_projective()
}
