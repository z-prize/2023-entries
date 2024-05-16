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

impl RangeProof for F {
    /// Given the native field f
    /// Chunks it in every 16 bits, since we set the range as [-2^16 + 1, 2^16 - 1] in the lookup table for range checking
    ///
    /// For example,
    /// To prove f has 96-bit length,
    /// we view f as f1 + f2 * 2^16 + f3 * 2^32 + f4 * 2^48 + f5 * 2^64 + f6 * 2^80
    /// then check f1,f2,f3,f4,f5,f6 are in range [-2^16 + 1, 2^16 - 1]
    ///
    /// This function is used to prove that f has bit length of bound_bit_length
    ///
    /// Note: if f is negative,
    ///       we chunk q_253 - f mod q_253 in every 16 bits instead.
    ///       now f = (-f1) + (-f2) * 2^16 + (-f3) * 2^32 + (-f4) * 2^48 + (-f5) * 2^64 + (-f6) * 2^80
    ///       then we check -f1,-f2,-f3,-f4,-f5,-f6 are in range [-2^16 + 1, 2^16 - 1]
    ///     
    fn range_prove_include_negtive(&self, bound_bit_length: usize) {
        let two_pow_b: Field<<Env as Environment>::Network> =
            Console::two_pow_k(bound_bit_length as u32);
        TABLES.with(|tables| {
            let mut flag = false;
            let mut value = self.eject_value();

            // To check whether f is negtive.
            // If it is, set f = q_253 - f = - f mod q_253
            if self.eject_value().cmp(&two_pow_b) == Ordering::Greater {
                flag = true;
                value = -value;
            }

            let mut bits = value.to_bits_le();
            if bits.len() != bound_bit_length {
                bits.truncate(bound_bit_length);
            }

            // Chunks it in every 16 bits
            // Prove every 16-bit length element in the range [-2^16 + 1, 2^16 - 1]
            // And enforce constriant f = f1 + f2 * 2^16 + f3 * 2^32 + f4 * 2^48 + f5 * 2^64 + f6 * 2^80
            SMALL_FIELD_BASES.with(|bases| {
                let mut acc = F::zero();
                for (i, chunks) in bits.chunks(SMALL_FIELD_SIZE_IN_BITS).enumerate() {
                    let mut console_f =
                        Field::<<Env as Environment>::Network>::from_bits_le(&chunks).unwrap();

                    // If f is negtive.
                    // f1,f2,f3,f4,f5,f6 are also negtive.
                    if flag {
                        console_f = -console_f;
                    }
                    let f = F::new(self.eject_mode(), console_f);
                    tables.range_check_include_negtive(&f);

                    // When the bit length of f is not the multiple 16
                    // the bit length k of last element in chunks is less than 16
                    // we need to enforce one more constriant
                    // f_last_element * 2^(16 - k) is in the range [-2^16 + 1, 2^16 - 1]
                    // since we have prove f_last_element is in the range [-2^16 + 1, 2^16 - 1] above
                    // now we can prove that f_last_element is in the range [-2^k + 1, 2^k - 1]
                    if chunks.len() < SMALL_FIELD_SIZE_IN_BITS {
                        let rest_k = SMALL_FIELD_SIZE_IN_BITS - chunks.len();
                        let two_pow_rest_k =
                            F::new(Mode::Constant, Console::two_pow_k(rest_k as u32));
                        let right_shift_rest_bits_f = &f * &two_pow_rest_k;
                        tables.range_check_include_negtive(&right_shift_rest_bits_f);
                    }
                    acc = &acc + &f * &bases[i];
                }
                assert_eq!(
                    acc.eject_value(),
                    self.eject_value(),
                    "Field range check faild. Bound bits length: {})",
                    bound_bit_length
                );
                // enforce constriant f = f1 + f2 * 2^16 + f3 * 2^32 + f4 * 2^48 + f5 * 2^64 + f6 * 2^80
                Env::assert_eq(acc, self);
            });
        });
    }

    /// Given the native field f
    /// Chunks it in every 16 bits, since we set the range as [0, 2^16 - 1] in the lookup table for range checking
    ///
    /// For example,
    /// To prove f has 96-bit length,
    /// we view f as f1 + f2 * 2^16 + f3 * 2^32 + f4 * 2^48 + f5 * 2^64 + f6 * 2^80
    /// then check f1,f2,f3,f4,f5,f6 are in range [0, 2^16 - 1]
    ///
    /// This function is used to prove that f has bit length of bound_bit_length
    fn range_prove(&self, bound_bit_length: usize) {
        TABLES.with(|tables| {
            let value = self.eject_value();

            let mut bits = value.to_bits_le();
            if bits.len() != bound_bit_length {
                bits.truncate(bound_bit_length);
            }

            // Chunks it in every 16 bits
            // Prove every 16-bit length element in the range [0, 2^16 - 1]
            // And enforce constriant f = f1 + f2 * 2^16 + f3 * 2^32 + f4 * 2^48 + f5 * 2^64 + f6 * 2^80
            SMALL_FIELD_BASES.with(|bases| {
                let mut acc = F::zero();
                for (i, chunks) in bits.chunks(SMALL_FIELD_SIZE_IN_BITS).enumerate() {
                    let console_f =
                        Field::<<Env as Environment>::Network>::from_bits_le(&chunks).unwrap();

                    let f = F::new(self.eject_mode(), console_f);
                    tables.range_check(&f);

                    // When the bit length of f is not the multiple 16
                    // the bit length k of last element in chunks is less than 16
                    // we need to enforce one more constriant
                    // f_last_element * 2^(16 - k) is in the range [0, 2^16 - 1]
                    // since we have prove f_last_element is in the range [0 2^16 - 1] above
                    // now we can prove that f_last_element is in the range [0, 2^k - 1]
                    if chunks.len() < SMALL_FIELD_SIZE_IN_BITS {
                        let rest_k = SMALL_FIELD_SIZE_IN_BITS - chunks.len();
                        let two_pow_rest_k =
                            F::new(Mode::Constant, Console::two_pow_k(rest_k as u32));
                        let right_shift_rest_bits_f = &f * &two_pow_rest_k;
                        tables.range_check(&right_shift_rest_bits_f);
                    }
                    acc = &acc + &f * &bases[i];
                }
                assert_eq!(
                    acc.eject_value(),
                    self.eject_value(),
                    "Field range check faild. Bound bits length: {})",
                    bound_bit_length
                );
                // enforce constriant f = f1 + f2 * 2^16 + f3 * 2^32 + f4 * 2^48 + f5 * 2^64 + f6 * 2^80
                Env::assert_eq(acc, self);
            });
        });
    }

    fn less_than_non_native_base_field_modulus(&self) {
        todo!()
    }

    fn less_than_non_native_scalar_field_modulus(&self) -> Vec<B> {
        todo!()
    }
}

impl RangeProof for Vec<F> {
    /// Given non-native a in the secp256k1 base field.
    /// non_native_a -> presented as [a1, a2, a3]
    /// such that non_native_a = a1 + a2 * 2^K + a3 * (2^K)^2
    /// where a1, a2, a3 are in the native field.
    ///
    /// This function is used to prove that the bit length of non_native_a is less than bound_bit_length (256 bits for non-native a)
    /// In other words, it's equivalent to prove that the bit length of a1, a2, a3 are less than  96, 96, 64 respectively.
    fn range_prove(&self, bound_bit_length: usize) {
        // In our setting, K = 96, bound_bit_length = 256
        // q = 3 which means that non_native_a is presented by three elements in the native field.
        let q = if bound_bit_length % K == 0 {
            bound_bit_length / K
        } else {
            bound_bit_length / K + 1
        };
        // calculte r = 256 % 96 = 64
        // which presents the bit length of the last element
        let r = bound_bit_length % K;
        if self.len() != q {
            Env::halt("The length non-native polynomial is wrong.")
        }
        for (i, f) in self.iter().enumerate() {
            if i == (q - 1) {
                // prove that a3 has 64 bit length.
                f.range_prove(r);
            } else {
                // prove that a1, a2 has 96 bit length.
                f.range_prove(K);
            }
        }
    }

    /// Given non-native a in the secp256k1 base field.
    /// non_native_a -> presented as [a1, a2, a3]
    /// such that non_native_a = a1 + a2 * 2^K + a3 * (2^K)^2
    /// where a1, a2, a3 are in the native field.
    ///
    /// This function is used to prove that non_native_a is in the range [1, q_256-1]
    fn less_than_non_native_base_field_modulus(&self) {
        let non_native_q_bit_le =
            Console::<<Env as Environment>::Network>::nonnative_base_modulus_minus_one_in_bits();
        let q = if NUM_BITS % K == 0 {
            NUM_BITS / K
        } else {
            NUM_BITS / K + 1
        };
        let r = NUM_BITS % K;

        let mut circuit_bits = Vec::with_capacity(0);

        // obtain 256 bits for non_native_a
        // it obtains lowest 96, 96, 64 bits of a1, a2, a3 respectively.
        // then ensure highest 157, 157, 198 bits of of a1, a2, a3 are zero.
        // finally, check non_native_a <= non_native_q in bits.
        for (i, chunk) in self.iter().enumerate() {
            let bits = chunk.to_bits_le();
            let num_bits = bits.len();
            let mut bound_in_bits = 0;
            if i < q - 1 {
                bound_in_bits = K;
            } else {
                bound_in_bits = r;
            }
            // ensure highest 157, 157, 198 bits of of a1, a2, a3 are zero.
            if num_bits > bound_in_bits {
                // Check if all excess bits are zero.
                for bit in bits[bound_in_bits..].iter() {
                    Env::assert_eq(Env::zero(), bit);
                }
            }
            circuit_bits.extend_from_slice(&bits[..bound_in_bits]);
        }
        // After ensure highest 157, 157, 198 bits of of a1, a2, a3 are zero, 
        // we can ensure that if a1+a2+a3 != 0 in q_253, then a not equal zero.
        Env::assert_neq(self.iter().fold(F::zero(), |acc, f| acc + f),  F::zero());
        // Asserts that `circuit_bits <= non_native_q_bit_le`.
        B::assert_less_than_or_equal_constant(&circuit_bits, &non_native_q_bit_le);
    }

    /// Given non-native a in the secp256k1 scalar field.
    /// non_native_a -> presented as [a1, a2, a3]
    /// such that non_native_a = a1 + a2 * 2^K + a3 * (2^K)^2
    /// where a1, a2, a3 are in the native field.
    ///
    /// This function is used to prove that non_native_a is in the range [1, p_256-1]
    fn less_than_non_native_scalar_field_modulus(&self) -> Vec<B> {
        let non_native_p_bit_le =
            Console::<<Env as Environment>::Network>::nonnative_scalar_modulus_minus_one_in_bits();
        let q = if NUM_BITS % K == 0 {
            NUM_BITS / K
        } else {
            NUM_BITS / K + 1
        };
        let r = NUM_BITS % K;

        let mut circuit_bits = Vec::with_capacity(0);

        // obtain 256 bits for non_native_a
        // it obtains lowest 96, 96, 64 bits of a1, a2, a3 respectively.
        // then ensure highest 157, 157, 198 bits of of a1, a2, a3 are zero.
        // finally, check non_native_a <= non_native_q in bits.
        for (i, chunk) in self.iter().enumerate() {
            let bits = chunk.to_bits_le();
            let num_bits = bits.len();
            let mut bound_in_bits = 0;
            if i < q - 1 {
                bound_in_bits = K;
            } else {
                bound_in_bits = r;
            }
            // ensure highest 157, 157, 198 bits of of a1, a2, a3 are zero.
            if num_bits > bound_in_bits {
                // Check if all excess bits are zero.
                for bit in bits[bound_in_bits..].iter() {
                    Env::assert_eq(Env::zero(), bit);
                }
            }
            circuit_bits.extend_from_slice(&bits[..bound_in_bits]);
        }
        // After ensure highest 157, 157, 198 bits of of a1, a2, a3 are zero, 
        // we can ensure that if a1+a2+a3 != 0 in q_253, then a not equal zero.
        Env::assert_neq(self.iter().fold(F::zero(), |acc, f| acc + f),  F::zero());
        // Asserts that `circuit_bits <= non_native_p_bit_le`.
        B::assert_less_than_or_equal_constant(&circuit_bits, &non_native_p_bit_le);
        circuit_bits
    }

    fn range_prove_include_negtive(&self, _: usize) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ITERATIONS: u64 = 1;

    fn check_range(
        mode: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
        num_lookup_constraints: u64,
    ) {
        for i in 0..ITERATIONS {
            let field = FieldElement::random(&mut OsRng);
            let poly: Polynomial<<Circuit as Environment>::Network> =
                Polynomial::from_nonnative_field(&field);
            let cir_fields = poly.0.iter().map(|b| F::new(mode, *b)).collect_vec();
            Circuit::scope(format!("{mode} {i}"), || {
                cir_fields.range_prove_include_negtive(NUM_BITS);
                assert_scope!(num_constants, num_public, num_private, num_constraints);
                assert_scope_with_lookup!(
                    num_constants,
                    num_public,
                    num_private,
                    num_constraints,
                    num_lookup_constraints
                );
            });
        }
    }

    #[test]
    fn test_and_constant() {
        check_range(Mode::Constant, 786452, 0, 0, 0, 0);
    }

    #[test]
    fn test_and_public() {
        check_range(Mode::Public, 786436, 16, 0, 3, 16);
    }

    #[test]
    fn test_and_private() {
        check_range(Mode::Private, 786436, 0, 16, 3, 16);
    }

    fn check_less_than_or_equal(
        mode: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
        num_lookup_constraints: u64,
    ) {
        for i in 0..ITERATIONS {
            let field = FieldElement::random(&mut OsRng);
            let poly: Polynomial<<Circuit as Environment>::Network> =
                Polynomial::from_nonnative_field(&field);
            let cir_fields = poly.0.iter().map(|b| F::new(mode, *b)).collect_vec();

            Circuit::scope(format!("{mode} {i}"), || {
                cir_fields.less_than_non_native_base_field_modulus();
                assert_scope!(num_constants, num_public, num_private, num_constraints);
                assert_scope_with_lookup!(
                    num_constants,
                    num_public,
                    num_private,
                    num_constraints,
                    num_lookup_constraints
                );
            });
        }
    }

    #[test]
    fn test_less_than_or_equal_constant() {
        check_less_than_or_equal(Mode::Constant, 759, 0, 0, 0, 0);
    }

    #[test]
    fn test_less_than_or_equal_public() {
        check_less_than_or_equal(Mode::Public, 0, 0, 1766, 2276, 0);
    }

    #[test]
    fn test_less_than_or_equal_private() {
        check_less_than_or_equal(Mode::Private, 0, 0, 1766, 2276, 0);
    }

    #[test]
    fn test_not_less_than_or_equal() {
        for i in 0..ITERATIONS {
            NONNATIVE_BASE_MODULUS.with(|modulus| {
                let mut cir_fields = modulus.clone();
                cir_fields[0] = &cir_fields[0] + &F::one();
                Circuit::scope(format!(" {i}"), || {
                    // cir_fields.less_than_or_equal_non_native_base_field_modulus();
                    //assert!(!Circuit::is_satisfied_in_scope(), "(should not satisfy)");
                });
            })
        }
    }
}
