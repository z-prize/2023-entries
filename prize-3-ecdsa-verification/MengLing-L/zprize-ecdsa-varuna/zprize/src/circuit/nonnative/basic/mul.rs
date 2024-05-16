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
#[derive(Clone, Debug)]
pub struct CircuitNonNativeMul {
    pub(crate) c: Vec<F>,
    pub(crate) alpha: Vec<F>,
    pub(crate) e: Vec<F>,
}

impl Inject for CircuitNonNativeMul {
    type Primitive = Mul<Testnet3>;

    fn new(mode: Mode, console_mul_unit: Self::Primitive) -> Self {
        let cir_c_fields = console_mul_unit
            .c
            .0
            .iter()
            .map(|b| F::new(mode, *b))
            .collect_vec();
        let cir_alpha_fields = console_mul_unit
            .alpha
            .0
            .iter()
            .map(|b| F::new(mode, *b))
            .collect_vec();
        let cir_e_fields = console_mul_unit
            .e
            .0
            .iter()
            .map(|b| F::new(mode, *b))
            .collect_vec();
        Self {
            c: cir_c_fields,
            alpha: cir_alpha_fields,
            e: cir_e_fields,
        }
    }
}

impl Eject for CircuitNonNativeMul {
    type Primitive = Mul<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.c.eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        let console_c_fields = self.c.iter().map(|b| b.eject_value()).collect_vec();
        let console_alpha_fields = self.alpha.iter().map(|b| b.eject_value()).collect_vec();
        let console_e_fields = self.e.iter().map(|b| b.eject_value()).collect_vec();
        Mul {
            c: Polynomial(console_c_fields),
            alpha: Polynomial(console_alpha_fields),
            e: Polynomial(console_e_fields),
        }
    }
}

impl NonNativeMul for CircuitNonNativeMul {
    type Size = F;

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
    fn poly_mul_wellform(&self, chunk_size: &Self::Size, a: &Vec<F>, b: &Vec<F>) {
        NONNATIVE_BASE_MODULUS.with(|modulus| {
            self.alpha.range_prove(NUM_BITS);
            self.c.range_prove(NUM_BITS);
            let ab = a.poly_mul(b);
            let mut alphaq_c = modulus.poly_mul(&self.alpha);
            alphaq_c = alphaq_c.poly_add(&self.c);
            let mut e_hat = F::zero();
            for i in 0..min(ab.len(), alphaq_c.len()) {
                let bound_bit_length = if i == 0 { K } else { K + 2 };
                let poly_tmp = ab[i].clone();
                let poly_tmp_c = alphaq_c[i].clone();

                self.e[i].range_prove_include_negtive(bound_bit_length);

                Env::assert_eq(&poly_tmp_c + &e_hat - chunk_size * &self.e[i], &poly_tmp);

                e_hat = self.e[i].clone();
            }
        })
    }

    fn scalar_poly_mul_wellform(&self, chunk_size: &Self::Size, a: &Vec<F>, b: &Vec<F>) {
        NONNATIVE_SCALAR_MODULUS.with(|modulus| {
            self.alpha.range_prove(NUM_BITS);
            self.c.range_prove(NUM_BITS);
            let ab = a.poly_mul(b);
            let mut alphaq_c = modulus.poly_mul(&self.alpha);
            alphaq_c = alphaq_c.poly_add(&self.c);
            let mut e_hat = F::zero();
            for i in 0..min(ab.len(), alphaq_c.len()) {
                let bound_bit_length = if i == 0 { K } else { K + 2 };
                let poly_tmp = ab[i].clone();
                let poly_tmp_c = alphaq_c[i].clone();

                self.e[i].range_prove_include_negtive(bound_bit_length);

                Env::assert_eq(&poly_tmp_c + &e_hat - chunk_size * &self.e[i], &poly_tmp);

                e_hat = self.e[i].clone();
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use ecdsa::elliptic_curve::ops::{Invert, Reduce};
    use k256::Scalar;
    use k256::Secp256k1;
    use sha3::Digest;
    use sha3::Keccak256;

    use crate::console::sample_msg;

    use super::*;

    const ITERATIONS: u64 = 1;
    #[test]
    fn test_secp256_base_field_to_console_mul_unit() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = secret_key.verifying_key();

        let public_key_point = public_key.to_encoded_point(false);
        let x: FieldElement = FieldElement::from_bytes(public_key_point.x().unwrap()).unwrap();
        let mut x_2 = x.clone();
        x_2 = x_2.square().normalize();
        let _: Mul<Testnet3> = Mul::to_mul(&x, &x, &x_2);
    }

    #[test]
    fn test_secp256_base_field_from_console_mul_unit_circuit_mul_unit() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = secret_key.verifying_key();

        let public_key_point = public_key.to_encoded_point(false);
        let x: FieldElement = FieldElement::from_bytes(public_key_point.x().unwrap()).unwrap();
        let mut x_2 = x.clone();
        x_2 = x_2.square();
        let mul_unit: Mul<Testnet3> = Mul::to_mul(&x, &x, &x_2);
        let cir_mul_unit = CircuitNonNativeMul::new(Mode::Private, mul_unit.clone());
        let recover_console_mul = cir_mul_unit.eject_value();

        assert_eq!(mul_unit.c, recover_console_mul.c);
        assert_eq!(mul_unit.alpha, recover_console_mul.alpha);
        assert_eq!(mul_unit.e, recover_console_mul.e);
    }

    fn check_mul(
        mode: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
    ) {
        for i in 0..ITERATIONS {
            let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
            let public_key = secret_key.verifying_key();

            let public_key_point = public_key.to_encoded_point(false);
            let x: FieldElement = FieldElement::from_bytes(public_key_point.x().unwrap()).unwrap();
            let mut x_2 = x.clone();
            x_2 = x_2.square();
            let mul_unit: Mul<Testnet3> = Mul::to_mul(&x, &x, &x_2);
            let a: Polynomial<Testnet3> = Polynomial::from_nonnative_field(&x);
            let b: Polynomial<Testnet3> = Polynomial::from_nonnative_field(&x);
            let a = a.0.iter().map(|f| F::new(mode, *f)).collect_vec();
            let b = b.0.iter().map(|f| F::new(mode, *f)).collect_vec();
            let chunk_size = F::new(
                Mode::Constant,
                crate::console::Console::<Testnet3>::chunk_size(),
            );
            let cir_mul_unit = CircuitNonNativeMul::new(mode, mul_unit.clone());
            Circuit::scope(format!("{mode} {i}"), || {
                cir_mul_unit.poly_mul_wellform(&chunk_size, &a, &b);
                assert_scope!(num_constants, num_public, num_private, num_constraints);
            });
        }
    }

    fn check_scalar_mul(
        mode: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
    ) {
        for i in 0..ITERATIONS {
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

            let mul_unit: Mul<Testnet3> = Mul::scalar_to_mul(&a, &b, &c);
            let a: Polynomial<Testnet3> = Polynomial::from_nonnative_scalar(&a);
            let b: Polynomial<Testnet3> = Polynomial::from_nonnative_scalar(&b);
            let a = a.0.iter().map(|f| F::new(mode, *f)).collect_vec();
            let b = b.0.iter().map(|f| F::new(mode, *f)).collect_vec();
            let chunk_size = F::new(
                Mode::Constant,
                crate::console::Console::<Testnet3>::chunk_size(),
            );
            let cir_mul_unit = CircuitNonNativeMul::new(mode, mul_unit.clone());
            Circuit::scope(format!("{mode} {i}"), || {
                cir_mul_unit.scalar_poly_mul_wellform(&chunk_size, &a, &b);
                assert_scope!(num_constants, num_public, num_private, num_constraints);
            });
        }
    }

    #[test]
    fn test_mul_constant() {
        check_mul(Mode::Constant, 786509, 0, 0, 0);
    }

    #[test]
    fn test_mul_public() {
        check_mul(Mode::Public, 786443, 66, 9, 43);
    }

    #[test]
    fn test_mul_private() {
        check_mul(Mode::Private, 786443, 0, 75, 43);
    }

    #[test]
    fn test_scalar_mul_constant() {
        check_scalar_mul(Mode::Constant, 786509, 0, 0, 0);
    }

    #[test]
    fn test_scalar_mul_public() {
        check_scalar_mul(Mode::Public, 786443, 66, 9, 43);
    }

    #[test]
    fn test_scalar_mul_private() {
        check_scalar_mul(Mode::Private, 786443, 0, 75, 43);
    }
}
