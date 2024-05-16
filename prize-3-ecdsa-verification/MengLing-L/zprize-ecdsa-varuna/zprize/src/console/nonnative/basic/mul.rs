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

use super::*;

/// present non-native multiplication process by native fields polynomial.
/// given a, b ,c
/// such that a * b = c mod q_256 //over non-native field
/// -> a * b = c + \alpha * q_256 //over integer
/// -> a(x) * b(x) = c(x) + \alpha(x) * q_256(x) + (2^K - x)e(x) mod native_q //over native field
/// where f(X) = f_1 + f_2 * X + f_3 * X^2 and f = f_1 + f_2 * 2^K + f_3 * (2^K)^2
/// and f_1, f_2 and f_3 \in native field
///
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mul<N: Network> {
    pub(crate) c: Polynomial<N>,
    pub(crate) alpha: Polynomial<N>,
    pub(crate) e: Polynomial<N>,
}

impl<N: Network> ToMul for Mul<N> {
    type Input = FieldElement;

    /// a(x) * b(x) = c(x) + \alpha(x) * q_256(x) + (2^K - x)e(x) mod native_q //over native field
    ///
    /// a -> presented as [a1, a2, a3]
    /// b -> presented as [b1, b2, b3]
    /// c -> presented as [c1, c2, c3]
    /// alpha -> presented as [alpha1, alpha2, alpha3]
    /// q_256 -> presented as [q1, q2, q3]
    /// ab = a * b ->
    ///   presented as [ ab_1 = a1 * b1, //around 192 bit length
    ///                  ab_2 = a2 * b1 + a1 * b2, //around 193 bit length
    ///                  ab_3 = a3 * b1 + a2 * b2 + a1* b3, //around 194 bit length
    ///                  ab_4 = a2 * b3 + a3 * b2, //around 193 bit length
    ///                  ab_5 = a3 * b3 ] //around 192 bit length
    /// e_1 is calculated as [e1, e2, e3, e4, e5]
    /// such that, K = 96 in our setting
    /// ab_1 = k1 + e1 * 2^K , ab_2 = k2 + e2 * 2^K, ...etc
    /// where k1, k2, k3, k4, k5 have K bit length, and e1, e2, e3, e4, e5 have most K + 2 bit length
    ///
    /// alphaq_c = alpha * q_256 + c ->
    ///   presented as [ alphaq_c_1 = alpha1 * q1 + c1, //around 192 bit length
    ///                  alphaq_c_2 = alpha2 * q1 + alpha1 * q2 + c2, //around 193 bit length
    ///                  alphaq_c_3 = alpha3 * q1 + alpha2 * q2 + alpha1* q3 + c3, //around 194 bit length
    ///                  alphaq_c_4 = alpha2 * q3 + alpha3 * q2, //around 193 bit length
    ///                  alphaq_c_5 = alpha3 * q3 ] //around 192 bit length
    /// e_2 is calculated as [e1', e2', e3', e4', e5']
    /// such that, K = 96 in our setting
    /// alphaq_c_1 = k1' + e1' * 2^K , alphaq_c_2 = k2' + e2' * 2^K, ...etc
    /// where k1', k2', k3', k4', k5' have K bit length, and e1', e2', e3', e4', e5' have most K + 2 bit length
    ///
    /// To prove a * b = c mod q_256 in native field
    /// This function enforce following constriant:
    /// ab_1 = alphaq_c_1 + e * 2^K where e = e1' - e1
    /// ab_2  = alphaq_c_2 + e' - e * 2^K where e'= e (computed previous step), e = e2' - e2  
    /// ab_3  = alphaq_c_3 + e' - e * 2^K where e'= e (computed previous step), e = e3' - e3  
    /// ....
    /// Besides, we also need to prove that
    ///         alpha1, alpha2, alpha3 are 96, 96, 64 bit length, respectively,
    ///         c1, c2, c3 are 96, 96, 64 bit length, respectively,
    ///         and e1' - e1, e2' - e2, e3' - e3, e4' - e4, e5' - e5 are 96, 98, 98, 98, 96 bit length, respectively.
    ///
    /// Note: if e = ei' - ei < 0 over integer
    ///       given native modulus q_253
    ///       let e_positive = ei - ei' >= 0,  then q_253 - e_positive = e
    ///       Therefore
    ///       ab_i  = alphaq_c_i + e' - e * 2^K where e'= e{i-1}' - e{i-1}, e = ei' - ei
    ///       = alphaq_c_i + e' - (q_253 - e_positive) * 2^K mod q_253
    ///       = alphaq_c_i + e' - (q_253 - ei + ei') * 2^K mod q_253
    ///       = alphaq_c_i + e' - q_253 * 2^K - (ei' - ei) * 2^K mod q_253
    ///       = alphaq_c_i + e' - (ei' - ei) * 2^K mod q_253
    ///       = alphaq_c_i + e' - e * 2^K mod q_253
    ///       The constriant is correct.
    fn to_mul(field_a: &Self::Input, field_b: &Self::Input, field_c: &Self::Input) -> Self {
        let field_ab = field_a * field_b;
        assert_eq!(field_ab.normalize(), (*field_c).normalize());
        let a_ = BigInt::from_bytes_be(Sign::Plus, &field_a.to_bytes());
        let b_ = BigInt::from_bytes_be(Sign::Plus, &field_b.to_bytes());
        let c_ = BigInt::from_bytes_be(Sign::Plus, &field_c.to_bytes());
        let q_ = BigInt::from_bytes_le(Sign::Plus, &*BASE_MODULUS);

        assert_eq!((&a_ * &b_) % &q_, c_);

        let alpha_ = (&a_ * &b_) / &q_;

        assert_eq!((&a_ * &b_), &c_ + &alpha_ * &q_);

        let a: Polynomial<N> = Polynomial::from_nonnative_field(field_a);
        let b: Polynomial<N> = Polynomial::from_nonnative_field(field_b);
        let c: Polynomial<N> = Polynomial::from_nonnative_field(field_c);
        let q: Polynomial<N> = Polynomial::from_bigint(&q_);
        let alpha: Polynomial<N> = Polynomial::from_bigint(&alpha_);

        let ab: Polynomial<N> = a.raw_mul(&b);
        let e_1: Polynomial<N> = ab.calculate_e();

        let alphaq_c: Polynomial<N> = alpha.raw_mul(&q).raw_add(&c);
        let e_2: Polynomial<N> = alphaq_c.calculate_e();

        let e: Vec<Field<N>> = e_2
            .0
            .iter()
            .zip(e_1.0.iter())
            .map(|(e2, e1)| *e2 - *e1)
            .collect_vec();
        Mul {
            c,
            alpha,
            e: Polynomial(e),
        }
    }

    type ScalarInput = Scalar;

    fn scalar_to_mul(
        field_a: &Self::ScalarInput,
        field_b: &Self::ScalarInput,
        field_c: &Self::ScalarInput,
    ) -> Self {
        let field_ab = field_a * field_b;
        assert_eq!(field_ab, *field_c);
        let a_ = BigInt::from_bytes_be(Sign::Plus, &field_a.to_bytes());
        let b_ = BigInt::from_bytes_be(Sign::Plus, &field_b.to_bytes());
        let c_ = BigInt::from_bytes_be(Sign::Plus, &field_c.to_bytes());
        let q_ = BigInt::from_bytes_le(Sign::Plus, &*SCALAR_MODULUS);

        assert_eq!((&a_ * &b_) % &q_, c_);

        let alpha_ = (&a_ * &b_) / &q_;

        assert_eq!((&a_ * &b_), &c_ + &alpha_ * &q_);

        let a: Polynomial<N> = Polynomial::from_nonnative_scalar(field_a);
        let b: Polynomial<N> = Polynomial::from_nonnative_scalar(field_b);
        let c: Polynomial<N> = Polynomial::from_nonnative_scalar(field_c);
        let q: Polynomial<N> = Polynomial::from_bigint(&q_);
        let alpha: Polynomial<N> = Polynomial::from_bigint(&alpha_);

        let ab: Polynomial<N> = a.raw_mul(&b);
        let e_1: Polynomial<N> = ab.calculate_e();

        let alphaq_c: Polynomial<N> = alpha.raw_mul(&q).raw_add(&c);
        let e_2: Polynomial<N> = alphaq_c.calculate_e();

        let e: Vec<Field<N>> = e_2
            .0
            .iter()
            .zip(e_1.0.iter())
            .map(|(e2, e1)| *e2 - *e1)
            .collect_vec();
        Mul {
            c,
            alpha,
            e: Polynomial(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use ecdsa::elliptic_curve::ops::{Invert, Reduce};
    use k256::Secp256k1;
    use sha3::Keccak256;

    use crate::console::{compute_scalar_from_hash_message, hash_message, sample_msg};

    use sha3::Digest;

    use super::*;

    const ITERATIONS: u64 = 100;

    #[test]
    fn test_non_native_mul_for_secp256_base_field_on_vm_consule() {
        for _ in 0..ITERATIONS {
            let a: FieldElement = FieldElement::random(&mut OsRng);
            let b: FieldElement = FieldElement::random(&mut OsRng);
            let c = (a * b).normalize();

            let unit: Mul<Testnet3> = Mul::to_mul(&a, &b, &c);

            let ab: Polynomial<Testnet3> =
                Polynomial::from_nonnative_field(&a).raw_mul(&Polynomial::from_nonnative_field(&b));
            let e_1: Polynomial<Testnet3> = ab.calculate_e();
            let alphaq_c: Polynomial<Testnet3> = unit
                .alpha
                .raw_mul(&console::Console::nonnative_base_modulus())
                .raw_add(&unit.c);
            let e_2: Polynomial<Testnet3> = alphaq_c.calculate_e();

            let k_field: Field<Testnet3> = Field::from_u8(K as u8);
            let two: Field<Testnet3> = Field::from_u8(2);
            let four = two + two;
            let two_k = two.pow(k_field);
            let mut e_hat = Field::from_u8(0);
            for i in 0..ab.0.len() {
                let poly_tmp = ab.0[i];
                let poly_tmp_c = alphaq_c.0[i];
                let e = e_2.0[i] - e_1.0[i];

                if (e).cmp(&(two_k * four)) == Ordering::Less {
                    assert_eq!(e.cmp(&(two_k * four)), Ordering::Less);
                } else {
                    assert_eq!(e.cmp(&(-(two_k * four))), Ordering::Greater);
                }

                assert_eq!(poly_tmp_c + e_hat - two_k * e, poly_tmp);
                e_hat = e.clone();
            }
            assert_eq!(e_1.0[ab.0.len() - 1], e_2.0[ab.0.len() - 1]);
        }
    }

    #[test]
    fn test_non_native_mul_for_secp256_base_field_constant_on_vm_consule() {
        let a: FieldElement = FieldElement::from_u64(2);
        let b: FieldElement = FieldElement::random(&mut OsRng);
        let c = (a * b).normalize();

        let unit: Mul<Testnet3> = Mul::to_mul(&a, &b, &c);

        let ab: Polynomial<Testnet3> =
            Polynomial::from_nonnative_field(&a).raw_mul(&Polynomial::from_nonnative_field(&b));
        let e_1: Polynomial<Testnet3> = ab.calculate_e();
        let alphaq_c: Polynomial<Testnet3> = unit
            .alpha
            .raw_mul(&console::Console::nonnative_base_modulus())
            .raw_add(&unit.c);
        let e_2: Polynomial<Testnet3> = alphaq_c.calculate_e();

        let k_field: Field<Testnet3> = Field::from_u8(K as u8);
        let two: Field<Testnet3> = Field::from_u8(2);
        let two_k = two.pow(k_field);

        for i in 0..alphaq_c.0.len() {
            let mut poly_tmp = ab.0[i];
            let mut poly_tmp_c = alphaq_c.0[i];
            //let exp: vmField<Testnet3> = vmField::from_u8((i + 1) as u8);
            //let B = two_K.pow(exp);
            if i > 0 {
                poly_tmp = poly_tmp + e_1.0[i - 1];
                poly_tmp_c = poly_tmp_c + e_2.0[i - 1];
            }
            let mut ab_ = poly_tmp.to_bits_le();
            ab_.truncate(K);
            let ab_mod: Field<Testnet3> = Field::from_bits_le(&ab_).unwrap();

            let mut alphap_c_ = poly_tmp_c.to_bits_le();
            alphap_c_.truncate(K);
            let alphap_c_mod: Field<Testnet3> = Field::from_bits_le(&alphap_c_).unwrap();

            assert_eq!(poly_tmp, two_k * &e_1.0[i] + ab_mod);
            assert_eq!(alphap_c_mod, poly_tmp - two_k * &e_1.0[i]);

            let _ = e_1.0[i] - e_2.0[i];

            assert_eq!(poly_tmp_c - two_k * e_2.0[i], poly_tmp - two_k * e_1.0[i]);
        }
        assert_eq!(e_1.0[ab.0.len() - 1], e_2.0[ab.0.len() - 1]);
    }

    #[test]
    fn test_non_native_zero_mul_for_secp256_base_field_constant_on_vm_consule() {
        let a: FieldElement = FieldElement::ZERO;
        let b: FieldElement = FieldElement::ZERO;
        let c = (a * b).normalize();

        let unit: Mul<Testnet3> = Mul::to_mul(&a, &b, &c);

        let ab: Polynomial<Testnet3> =
            Polynomial::from_nonnative_field(&a).raw_mul(&Polynomial::from_nonnative_field(&b));
        let e_1: Polynomial<Testnet3> = ab.calculate_e();
        assert_eq!(e_1.0.len(), 5);
        let alphaq_c: Polynomial<Testnet3> = unit
            .alpha
            .raw_mul(&console::Console::nonnative_base_modulus())
            .raw_add(&unit.c);
        let e_2: Polynomial<Testnet3> = alphaq_c.calculate_e();
        assert_eq!(e_2.0.len(), 5);

        let k_field: Field<Testnet3> = Field::from_u8(K as u8);
        let two: Field<Testnet3> = Field::from_u8(2);
        let two_k = two.pow(k_field);

        for i in 0..alphaq_c.0.len() {
            let mut poly_tmp = ab.0[i];
            let mut poly_tmp_c = alphaq_c.0[i];
            //let exp: vmField<Testnet3> = vmField::from_u8((i + 1) as u8);
            //let B = two_K.pow(exp);
            if i > 0 {
                poly_tmp = poly_tmp + e_1.0[i - 1];
                poly_tmp_c = poly_tmp_c + e_2.0[i - 1];
            }
            let mut ab_ = poly_tmp.to_bits_le();
            ab_.truncate(K);
            let ab_mod: Field<Testnet3> = Field::from_bits_le(&ab_).unwrap();

            let mut alphap_c_ = poly_tmp_c.to_bits_le();
            alphap_c_.truncate(K);
            let alphap_c_mod: Field<Testnet3> = Field::from_bits_le(&alphap_c_).unwrap();

            assert_eq!(poly_tmp, two_k * &e_1.0[i] + ab_mod);
            assert_eq!(alphap_c_mod, poly_tmp - two_k * &e_1.0[i]);

            let e = e_1.0[i] - e_2.0[i];

            assert_eq!(poly_tmp_c + two_k * e, poly_tmp);
        }
        assert_eq!(e_1.0[ab.0.len() - 1], e_2.0[ab.0.len() - 1]);
    }

    #[test]
    fn test_non_native_mul_for_secp256_scalar_field_on_vm_consule() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);

        // generate msg_len random bytes
        let msg = sample_msg(100);

        // hash msg
        let mut hasher = Keccak256::new();
        hasher.update(&msg);

        // generate signature
        let (signature, _) = secret_key.sign_digest_recoverable(hasher.clone()).unwrap();

        let (_, s) = signature.split_scalars();
        let s_inv = *s.invert_vartime();
        let a: Scalar =
            <Scalar as Reduce<<Secp256k1 as k256::elliptic_curve::Curve>::Uint>>::reduce_bytes(
                &s_inv.to_bytes(),
            );
        let b: Scalar = Scalar::random(&mut OsRng);
        let c = a * b;

        let unit: Mul<Testnet3> = Mul::scalar_to_mul(&a, &b, &c);

        let ab: Polynomial<Testnet3> =
            Polynomial::from_nonnative_scalar(&a).raw_mul(&Polynomial::from_nonnative_scalar(&b));

        let e_1: Polynomial<Testnet3> = ab.calculate_e();
        let alphaq_c: Polynomial<Testnet3> = unit
            .alpha
            .raw_mul(&console::Console::nonnative_scalar_modulus())
            .raw_add(&unit.c);
        let e_2: Polynomial<Testnet3> = alphaq_c.calculate_e();

        let k_field: Field<Testnet3> = Field::from_u8(K as u8);
        let two: Field<Testnet3> = Field::from_u8(2);
        let four = two + two;
        let two_k = two.pow(k_field);
        let mut e_hat = Field::from_u8(0);
        for i in 0..ab.0.len() {
            let poly_tmp = ab.0[i];
            let poly_tmp_c = alphaq_c.0[i];
            let e = e_2.0[i] - e_1.0[i];

            if (e).cmp(&(two_k * four)) == Ordering::Less {
                assert_eq!(e.cmp(&(two_k * four)), Ordering::Less);
            } else {
                assert_eq!(e.cmp(&(two_k * four)), Ordering::Greater);
            }

            assert_eq!(poly_tmp_c + e_hat - two_k * e, poly_tmp);
            e_hat = e.clone();
        }
        assert_eq!(e_1.0[ab.0.len() - 1], e_2.0[ab.0.len() - 1]);
    }

    #[test]
    fn test_non_native_mul_a_mul_inv_a_for_secp256_scalar_field_on_vm_consule() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);

        // generate msg_len random bytes
        let msg = sample_msg(100);

        // hash msg
        let mut hasher = Keccak256::new();
        hasher.update(&msg);

        // Perform hashing only once per tuple.
        let msg_hash = hash_message(&msg);

        // Get scalar from hash of the message
        let _ = compute_scalar_from_hash_message(&msg_hash);

        // generate signature
        let (signature, _) = secret_key.sign_digest_recoverable(hasher.clone()).unwrap();

        let (_, s) = signature.split_scalars();
        let s_inv = *s.invert_vartime();
        let a: Scalar =
            <Scalar as Reduce<<Secp256k1 as k256::elliptic_curve::Curve>::Uint>>::reduce_bytes(
                &s_inv.to_bytes(),
            );
        let b: Scalar = *signature.s();
        let c = a * b;

        assert_eq!(c, Scalar::ONE);

        let unit: Mul<Testnet3> = Mul::scalar_to_mul(&a, &b, &c);

        let ab: Polynomial<Testnet3> =
            Polynomial::from_nonnative_scalar(&a).raw_mul(&Polynomial::from_nonnative_scalar(&b));

        let e_1: Polynomial<Testnet3> = ab.calculate_e();
        let alphaq_c: Polynomial<Testnet3> = unit
            .alpha
            .raw_mul(&console::Console::nonnative_scalar_modulus())
            .raw_add(&unit.c);
        let e_2: Polynomial<Testnet3> = alphaq_c.calculate_e();

        let k_field: Field<Testnet3> = Field::from_u8(K as u8);
        let two: Field<Testnet3> = Field::from_u8(2);
        let four = two + two;
        let two_k = two.pow(k_field);
        let mut e_hat = Field::from_u8(0);
        for i in 0..ab.0.len() {
            let poly_tmp = ab.0[i];
            let poly_tmp_c = alphaq_c.0[i];
            let e = e_2.0[i] - e_1.0[i];

            if (e).cmp(&(two_k * four)) == Ordering::Less {
                assert_eq!(e.cmp(&(two_k * four)), Ordering::Less);
            } else {
                assert_eq!(e.cmp(&(-(two_k * four))), Ordering::Greater);
            }

            assert_eq!(poly_tmp_c + e_hat - two_k * e, poly_tmp);
            e_hat = e.clone();
        }
        assert_eq!(e_1.0[ab.0.len() - 1], e_2.0[ab.0.len() - 1]);
    }

    #[test]
    fn test_signature_verify() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = *secret_key.verifying_key();

        // Convert public key to projective and non native type once.
        let pubkey_project: ProjectivePoint = public_key.as_affine().into();

        // generate msg_len random bytes
        let msg = sample_msg(100);

        // hash msg
        let mut hasher = Keccak256::new();
        hasher.update(&msg);

        // Perform hashing only once per tuple.
        let msg_hash = hash_message(&msg);

        // Get scalar from hash of the message
        let hm = compute_scalar_from_hash_message(&msg_hash);

        // generate signature
        let (signature, _) = secret_key.sign_digest_recoverable(hasher.clone()).unwrap();

        let (sig_r, sig_s) = signature.split_scalars();
        let mut sinv = *sig_s.invert_vartime();
        sinv = <Scalar as Reduce<<Secp256k1 as k256::elliptic_curve::Curve>::Uint>>::reduce_bytes(
            &sinv.to_bytes(),
        );
        let sinv_mul_hm = sinv * hm;
        let sinv_mul_r = sinv * sig_r.as_ref();
        let _ = sinv * sig_s.as_ref();

        let sinv_mul_hm_p = ProjectivePoint::GENERATOR * sinv_mul_hm;

        let sinv_mul_r_pub_key = pubkey_project * sinv_mul_r;

        let sinv_mul_hm_p_add_sinv_mul_r_pub_key = sinv_mul_hm_p + sinv_mul_r_pub_key;

        assert_eq!(
            *sinv_mul_hm_p_add_sinv_mul_r_pub_key
                .to_affine()
                .to_encoded_point(false)
                .x()
                .unwrap(),
            sig_r.to_bytes()
        );
    }
}
