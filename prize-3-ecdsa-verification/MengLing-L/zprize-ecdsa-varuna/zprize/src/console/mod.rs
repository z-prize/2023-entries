// Copyright (C) 2024 Mengling LIU, The Hong Kong Polytechnic University
//
// Apache License 2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// MIT License
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the “Software”),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
use self::nonnative::{
    basic::{mul::Mul, Polynomial, K},
    group::{
        add::ConsoleNonNativePointAdd, double::ConsoleNonNativePointDouble,
        mul::ConsoleNonNativePointMul, ConsoleNonNativePoint,
    },
};
use self::traits::ToPolynomial;
use crate::circuit::types::field::{
    NUMBER_OF_PADDED_ZERO, PADDED_CHUNK_SIZE, PADDED_FIELD_SIZE, SMALL_FIELD_SIZE_IN_BITS,
};
use crate::circuit::{keccak::Keccak256 as CircuitKeccak256, F};
use crate::console::traits::ToMul;
use ecdsa::{
    elliptic_curve::{ops::Reduce, sec1::ToEncodedPoint, PrimeField},
    hazmat::bits2field,
    signature::DigestVerifier,
};
use indexmap::IndexMap;
use k256::elliptic_curve::ops::Invert;
use k256::{
    ecdsa::{Signature, SigningKey, VerifyingKey},
    FieldElement, ProjectivePoint, Scalar, Secp256k1,
};
use lazy_static::lazy_static;
use num_bigint::{BigInt, Sign};
use rand::rngs::OsRng;
use sha3::{Digest, Keccak256};
use snarkvm_circuit_environment::{prelude::num_traits::FromPrimitive, Inject, Mode, Pow};
use snarkvm_console::{network::Testnet3, types::Field};
use snarkvm_console_algorithms::FromBits;
use snarkvm_console_network::{prelude::Uniform, Network};
use snarkvm_utilities::ToBits;
use std::marker::PhantomData;

pub mod keccak;
pub mod lookup_table;
pub mod nonnative;
pub mod table;
pub mod traits;
pub mod types;

pub type CF = Field<Testnet3>;

pub static ROTATED_VEC_SIZE: usize = 5;

lazy_static! {
    static ref SCALAR_MODULUS: [u8; 32] = [
        65, 65, 54, 208, 140, 94, 210, 191, 59, 160, 72, 175, 230, 220, 174, 186, 254, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
    ];
    static ref BASE_MODULUS: [u8; 32] = [
        47, 252, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
    ];
    pub static ref TWO_POW_K: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0
    ];
}
pub struct Console<N: Network> {
    _phantom: PhantomData<N>,
}

impl<N: Network> Console<N> {
    /// Return 2^K, K = 96 in our setting.
    pub(crate) fn chunk_size() -> Field<N> {
        let two: Field<N> = Field::from_u8(2);
        let k: Field<N> = Field::from_u32(K as u32);
        two.pow(k)
    }

    /// Given k, return 2^k,
    pub(crate) fn two_pow_k(k: u32) -> Field<N> {
        let two: Field<N> = Field::from_u8(2);
        let k_: Field<N> = Field::from_u32(k);
        two.pow(k_)
    }

    /// Return [0, 2^16, (2^16)^2, (2^16)^3, (2^16)^4, (2^16)^5, (2^16)^6]
    pub(crate) fn small_field_bases() -> Vec<Field<N>> {
        let two: Field<N> = Field::from_u8(2);
        let k: Field<N> = Field::from_u32(SMALL_FIELD_SIZE_IN_BITS as u32);
        let base = two.pow(k);
        let mut bases = Vec::with_capacity(K / SMALL_FIELD_SIZE_IN_BITS);
        for i in 0..(K / SMALL_FIELD_SIZE_IN_BITS + 1) {
            let exp = Field::<N>::from_u8((i) as u8);
            bases.push(base.pow(exp));
        }
        bases
    }

    /// Return [0, 2^48, (2^48)^2, (2^48)^3]
    pub(crate) fn padded_field_bases() -> Vec<Field<N>> {
        let two: Field<N> = Field::from_u8(2);
        let k: Field<N> = Field::from_u32(PADDED_FIELD_SIZE as u32);
        let base = two.pow(k);
        let mut bases = Vec::with_capacity(4);
        for i in 0..4 {
            let exp = Field::<N>::from_u8((i) as u8);
            bases.push(base.pow(exp));
        }
        bases
    }

    /// This function returns 100100100...etc which has bit length of PADDED_FIELD_SIZE * 4
    pub(crate) fn padded_one() -> Field<N> {
        let mut bits = Vec::with_capacity(PADDED_FIELD_SIZE * 4);
        for index in 0..(PADDED_FIELD_SIZE * 4) {
            match index % PADDED_CHUNK_SIZE {
                0 => bits.push(true),
                _ => bits.push(false),
            }
        }
        Field::from_bits_le(&bits).unwrap()
    }

    /// Return [g, g^2, g^4, g^8,...] used in non-native genertor multiplication.
    pub(crate) fn new_bases() -> Vec<ConsoleNonNativePointDouble<N>> {
        let mut g = ProjectivePoint::GENERATOR;

        // Compute the bases up to the size of the scalar field (in bits).
        let mut nonnative_g_bases: Vec<ConsoleNonNativePointDouble<N>> =
            Vec::with_capacity(Scalar::NUM_BITS as usize);

        for _ in 0..Scalar::NUM_BITS {
            let x_ = g;
            // Double 'g' for the next iteration to avoid conversion into AffinePoint.
            g = g.double();

            let g_base: ConsoleNonNativePointDouble<N> = ConsoleNonNativePointDouble::new(&x_, &g);
            nonnative_g_bases.push(g_base);
        }

        nonnative_g_bases
    }

    /// Return non-native zero -> presented as [0,0,0]
    pub(crate) fn zero() -> Polynomial<N> {
        Polynomial::from_nonnative_field(&FieldElement::ZERO)
    }

    /// Return non-native base field modulus -> presented as [q1,q2,q3]
    pub(crate) fn nonnative_base_modulus() -> Polynomial<N> {
        let q_ = BigInt::from_bytes_le(Sign::Plus, &*BASE_MODULUS);
        Polynomial::from_bigint(&q_)
    }

    /// Return non-native base field modulus - 1 in bits
    pub(crate) fn nonnative_base_modulus_minus_one_in_bits() -> Vec<bool> {
        let mut q_ = BigInt::from_bytes_le(Sign::Plus, &*BASE_MODULUS);
        let one = BigInt::from_u16(1).unwrap();
        q_ = q_ - one;
        let (_, le) = q_.to_bytes_le();
        le.to_bits_le()
    }

    /// Return non-native scalar field modulus -> presented as [q1,q2,q3]
    pub(crate) fn nonnative_scalar_modulus() -> Polynomial<N> {
        let q_ = BigInt::from_bytes_le(Sign::Plus, &*SCALAR_MODULUS);
        Polynomial::from_bigint(&q_)
    }

    /// Return non-native scalar field modulus - 1 in bits
    pub(crate) fn nonnative_scalar_modulus_minus_one_in_bits() -> Vec<bool> {
        let mut q_ = BigInt::from_bytes_le(Sign::Plus, &*SCALAR_MODULUS);
        let one = BigInt::from_u16(1).unwrap();
        q_ = q_ - one;
        let (_, le) = q_.to_bytes_le();
        le.to_bits_le()
    }

    /// Return non-native two -> presented as [2,0,0]
    pub(crate) fn nonnative_two() -> Polynomial<N> {
        let two = FieldElement::from_u64(2);
        Polynomial::from_nonnative_field(&two)
    }

    /// Return non-native three -> presented as [3,0,0]
    pub(crate) fn nonnative_three() -> Polynomial<N> {
        let three = FieldElement::from_u64(3);
        Polynomial::from_nonnative_field(&three)
    }

    /// Return non-native seven -> presented as [7,0,0]
    pub(crate) fn nonnative_seven() -> Polynomial<N> {
        let seven = FieldElement::from_u64(7);
        Polynomial::from_nonnative_field(&seven)
    }

    /// Return non-native scalar field identity
    pub(crate) fn nonnative_scalar_field_identity() -> Polynomial<N> {
        let ident: Scalar = Scalar::ONE;
        Polynomial::from_nonnative_scalar(&ident)
    }

    /// Return non-native identity point
    pub(crate) fn nonnative_point_identity() -> ConsoleNonNativePoint<N> {
        ConsoleNonNativePoint::new(&ProjectivePoint::IDENTITY.to_affine())
    }

    /// Return non-native generator point
    pub(crate) fn nonnative_point_generator() -> ConsoleNonNativePoint<N> {
        ConsoleNonNativePoint::new(&ProjectivePoint::GENERATOR.to_affine())
    }

    /// Return possible two^exp used in rotation.
    pub(crate) fn rotl_two_pow_exps() -> IndexMap<usize, F> {
        let rotls = CircuitKeccak256::rotl_offsets();
        let mut two_pow_exps: IndexMap<usize, F> = IndexMap::new();

        for n in rotls.iter() {
            let t = if n % SMALL_FIELD_SIZE_IN_BITS == 0 {
                n / SMALL_FIELD_SIZE_IN_BITS
            } else {
                n / SMALL_FIELD_SIZE_IN_BITS + 1
            };
            let mut exp = 0;
            let mut exp_rotated = 0;
            let last_n_t = ROTATED_VEC_SIZE.saturating_sub(t);
            for i in 0..ROTATED_VEC_SIZE {
                let base = F::new(Mode::Constant, Console::two_pow_k(exp));
                if let Some(value) = two_pow_exps.get_mut(&(exp as usize)) {
                    *value = base;
                } else {
                    two_pow_exps.insert(exp as usize, base);
                }

                if i == last_n_t - 1 && last_n_t > 0 {
                    let rest_bits = PADDED_FIELD_SIZE * 4
                        - (n * (NUMBER_OF_PADDED_ZERO + 1))
                        - i * PADDED_FIELD_SIZE;
                    exp += rest_bits as u32;
                    exp_rotated = (NUMBER_OF_PADDED_ZERO + 1)
                        * (SMALL_FIELD_SIZE_IN_BITS - rest_bits / (NUMBER_OF_PADDED_ZERO + 1));
                } else if i < ROTATED_VEC_SIZE - 1 {
                    exp += PADDED_FIELD_SIZE as u32;
                    exp_rotated = (NUMBER_OF_PADDED_ZERO + 1)
                        * (SMALL_FIELD_SIZE_IN_BITS - SMALL_FIELD_SIZE_IN_BITS);
                }

                let base = F::new(Mode::Constant, Console::two_pow_k(exp_rotated as u32));
                if let Some(value) = two_pow_exps.get_mut(&(exp_rotated as usize)) {
                    *value = base;
                } else {
                    two_pow_exps.insert(exp as usize, base);
                }
            }

            for i in 0..ROTATED_VEC_SIZE {
                let base = F::new(Mode::Constant, Console::two_pow_k(exp));
                if let Some(value) = two_pow_exps.get_mut(&(exp as usize)) {
                    *value = base;
                } else {
                    two_pow_exps.insert(exp as usize, base);
                }

                if t > 0 && i == t - 1 {
                    let rest_bits = (n * (NUMBER_OF_PADDED_ZERO + 1)) - i * PADDED_FIELD_SIZE;
                    exp += rest_bits as u32;
                    exp_rotated = (NUMBER_OF_PADDED_ZERO + 1)
                        * (SMALL_FIELD_SIZE_IN_BITS - rest_bits / (NUMBER_OF_PADDED_ZERO + 1));
                } else if i < ROTATED_VEC_SIZE - 1 {
                    exp += PADDED_FIELD_SIZE as u32;
                    exp_rotated = (NUMBER_OF_PADDED_ZERO + 1)
                        * (SMALL_FIELD_SIZE_IN_BITS - SMALL_FIELD_SIZE_IN_BITS);
                }

                let base = F::new(Mode::Constant, Console::two_pow_k(exp_rotated as u32));
                if let Some(value) = two_pow_exps.get_mut(&(exp_rotated as usize)) {
                    *value = base;
                } else {
                    two_pow_exps.insert(exp as usize, base);
                }
            }
        }
        two_pow_exps
    }
}

//Precompute
pub struct ECDSASignature<N: Network> {
    // signature.s
    pub s: Polynomial<N>,
    // signature.s^-1
    pub sinv: Polynomial<N>,
    // signature.r
    pub r: Polynomial<N>,
    // s^-1 * Hash(message)
    pub sinv_mul_hm: Mul<N>,
    // s^-1 * r
    pub sinv_mul_r: Mul<N>,
    // s^-1 * s
    pub sinv_mul_s: Mul<N>,
    // s^-1 * Hash(message) \cdot P (Generator)
    pub sinv_mul_hm_p: ConsoleNonNativePointMul<N>,
    // s^-1 * r\cdot PubKey
    pub sinv_mul_r_pub_key: ConsoleNonNativePointMul<N>,
    // Point hm_mul_p + r_mul_pub_key
    pub sinv_mul_hm_p_add_sinv_mul_r_pub_key: ConsoleNonNativePointAdd<N>,
}

pub struct ECDSAPubKey<N: Network> {
    pub public_key: ConsoleNonNativePoint<N>,
}

#[derive(Clone)]
pub struct Message {
    pub msg: Vec<bool>,
}

pub fn sample_msg(size: usize) -> Vec<u8> {
    let rng = &mut OsRng;
    let mut res = Vec::with_capacity(size);
    for _ in 0..size {
        res.push(u8::rand(rng));
    }
    res
}

pub fn msg_to_bits(msg: &[u8]) -> Vec<bool> {
    msg.iter()
        //.rev() // Reverse the bytes for little-endian bit order
        .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1 == 1)) // Convert bytes to bits
        .collect::<Vec<_>>()
}

/// Samples a public key and signature based on a message
pub fn sample_pubkey_sig<N: Network>(msg: &[u8]) -> (ECDSAPubKey<N>, ECDSASignature<N>, Message) {
    // generate keypair
    let rng = &mut OsRng;
    let secret_key = k256::ecdsa::SigningKey::random(rng);
    let public_key = *secret_key.verifying_key();

    // hash bytes
    let mut hasher = sha3::Keccak256::new();
    hasher.update(msg);

    // sign
    let (signature, _) = secret_key.sign_digest_recoverable(hasher.clone()).unwrap();

    // get scalar from hash of the message
    let z = bits2field::<Secp256k1>(&hasher.clone().finalize()).unwrap();
    let hm: Scalar =
        <Scalar as Reduce<<Secp256k1 as k256::elliptic_curve::Curve>::Uint>>::reduce_bytes(&z);

    // Convert public key to projective and non native type once.
    let pubkey_project: ProjectivePoint = public_key.as_affine().into();
    let non_native_public_key = ConsoleNonNativePoint::new(public_key.as_affine());

    // Convert signature to the non native type once.
    let sig = compute_non_native_signature(&hm, &signature, &pubkey_project);

    // verify the signature
    assert!(public_key.verify_digest(hasher, &signature).is_ok());

    (
        ECDSAPubKey {
            public_key: non_native_public_key,
        },
        sig,
        Message {
            msg: msg_to_bits(msg),
        },
    )
}

/// Generates `num` tuples of (public key, msg, signature) with messages of length `msg_len`.
pub fn generate_signatures(msg_len: usize, num: usize) -> Vec<(VerifyingKey, Vec<u8>, Signature)> {
    let mut res = vec![];

    for _ in 0..num {
        // generate keypair
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = secret_key.verifying_key();

        // generate msg_len random bytes
        let msg = sample_msg(msg_len);

        // hash msg
        let mut hasher = Keccak256::new();
        hasher.update(&msg);

        // generate signature
        let (signature, _) = secret_key.sign_digest_recoverable(hasher.clone()).unwrap();

        res.push((*public_key, msg, signature));
    }

    res
}

/// Helper functions to abstract away repeated logic.
pub fn hash_message(msg: &[u8]) -> Vec<u8> {
    // Hash message once and return the result.
    let mut hasher = sha3::Keccak256::new();
    hasher.update(msg);
    hasher.finalize().to_vec()
}

/// Compute scalar from hash of the message.
pub fn compute_scalar_from_hash_message(msg_hash: &[u8]) -> Scalar {
    let z = bits2field::<Secp256k1>(&msg_hash).expect("Failed to compute message scalar");
    let hm: Scalar =
        <Scalar as Reduce<<Secp256k1 as k256::elliptic_curve::Curve>::Uint>>::reduce_bytes(&z);
    hm
}

/// Compute non native signature from ecdsa signature
/// It includes:
/// 1. the precomputed value of s^-1 * hash(m)
/// 2. the precomputed value of s^-1 * sig_r
/// 3. the precomputed value of s^-1 * sig_s
/// 4. the precomputed value of (s^-1 * hash(m)) mul GENERATOR
/// 5. the precomputed value of (s^-1 * sig_r) mul Public Key
/// 6. the precomputed value for the process of adding point (s^-1 * hash(m)) mul GENERATOR and (s^-1 * sig_r) mul Public Key
pub fn compute_non_native_signature<N: Network>(
    hm: &Scalar,
    signature: &Signature,
    pubkey_project: &ProjectivePoint,
) -> ECDSASignature<N> {
    let (sig_r, sig_s) = signature.split_scalars();
    let mut sinv = *sig_s.invert_vartime();
    sinv = <Scalar as Reduce<<Secp256k1 as k256::elliptic_curve::Curve>::Uint>>::reduce_bytes(
        &sinv.to_bytes(),
    );
    let sinv_mul_hm = sinv * hm;
    let sinv_mul_r = sinv * sig_r.as_ref();
    let sinv_mul_s = sinv * sig_s.as_ref();

    let sinv_mul_hm_p = ProjectivePoint::GENERATOR * sinv_mul_hm;
    let sinv_mul_r_pub_key = *pubkey_project * sinv_mul_r;
    let sinv_mul_hm_p_add_sinv_mul_r_pub_key = sinv_mul_hm_p + sinv_mul_r_pub_key;

    assert_eq!(
        *sinv_mul_hm_p_add_sinv_mul_r_pub_key
            .to_affine()
            .to_encoded_point(false)
            .x()
            .unwrap(),
        sig_r.to_bytes()
    );

    ECDSASignature {
        s: Polynomial::from_nonnative_scalar(&sig_s),
        sinv: Polynomial::from_nonnative_scalar(&sinv),
        r: Polynomial::from_nonnative_scalar(&sig_r),
        sinv_mul_hm: Mul::scalar_to_mul(&sinv, &hm, &sinv_mul_hm),
        sinv_mul_r: Mul::scalar_to_mul(&sinv, &sig_r.as_ref(), &sinv_mul_r),
        sinv_mul_s: Mul::scalar_to_mul(&sinv, &sig_s.as_ref(), &sinv_mul_s),
        sinv_mul_hm_p: ConsoleNonNativePointMul::new(
            &sinv_mul_hm,
            &ProjectivePoint::GENERATOR,
            &sinv_mul_hm_p,
            true,
        ),
        sinv_mul_r_pub_key: ConsoleNonNativePointMul::new(
            &sinv_mul_r,
            &pubkey_project,
            &sinv_mul_r_pub_key,
            false,
        ),
        sinv_mul_hm_p_add_sinv_mul_r_pub_key: ConsoleNonNativePointAdd::new(
            &sinv_mul_hm_p,
            &sinv_mul_r_pub_key,
            &sinv_mul_hm_p_add_sinv_mul_r_pub_key,
        ),
    }
}
