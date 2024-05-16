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

pub mod linear;
pub mod mul;

use self::linear::Signed;

pub static K: usize = 96;

pub static NUM_BITS: usize = 256;

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Polynomial<N: Network>(pub(crate) Vec<Field<N>>);

/// Convert non native field x 256 bits into three native fields x1, x2, x3 in our setting
/// such that x = x1 + x2\codt 2^96 + x3\cdot 2^192 (polynomial)
/// x1,x2 and x3 have 96,96, and 64 bit length, respectively.
impl<N: Network> ToPolynomial for Polynomial<N> {
    type FieldInput = FieldElement;

    /// Convert secp256k1 base field into non native fields.
    fn from_nonnative_field(input: &Self::FieldInput) -> Self {
        let mut le = input.to_bytes().to_vec();
        le.reverse();
        Polynomial(
            le.to_bits_le()
                .chunks(K)
                .map(|chunks| {
                    Field::from_bits_le(chunks).unwrap_or_else(|_| {
                        panic!("Failed covert to console field from secp256 base field {chunks:?}")
                    })
                })
                .collect::<Vec<_>>(),
        )
    }

    type ScalarInput = Scalar;

    /// Convert secp256k1 scalar field into non native fields.
    fn from_nonnative_scalar(input: &Self::ScalarInput) -> Self {
        let mut le = input.to_bytes().to_vec();
        le.reverse();
        Polynomial(
            le.to_bits_le()
                .chunks(K)
                .map(|chunks| {
                    Field::from_bits_le(chunks).unwrap_or_else(|_| {
                        panic!("Failed covert to console field from secp256 base field {chunks:?}")
                    })
                })
                .collect::<Vec<_>>(),
        )
    }

    type BigInput = BigInt;

    /// Convert intermediate bigint during operation into non native fields.
    fn from_bigint(input: &Self::BigInput) -> Self {
        let (_, le) = input.to_bytes_le();
        let k = NUM_BITS as usize / K + 1;
        let mut native_field_vec = le
            .to_bits_le()
            .chunks(K)
            .map(|chunks| {
                Field::from_bits_le(chunks).unwrap_or_else(|_| {
                    panic!("Failed covert to console field from bigint {chunks:?}")
                })
            })
            .collect::<Vec<_>>();
        while native_field_vec.len() < k {
            native_field_vec.extend([Field::zero()]);
        }
        Polynomial(native_field_vec)
    }
}

/// Recover non native field from native fields
impl<N: Network> FromPolynomial for Polynomial<N> {
    type FieldOutput = FieldElement;

    fn to_nonnative_field(&self) -> Self::FieldOutput
    where
        Self: Sized,
    {
        let mut bit_le = self
            .0
            .clone()
            .into_iter()
            .flat_map(|x| {
                let mut a = x.to_bits_le();
                a.truncate(K);
                a
            })
            .collect::<Vec<_>>();

        bit_le.truncate(NUM_BITS);

        let mut bytes = bit_le
            .chunks(8)
            .map(|chunks| {
                u8::from_bits_le(chunks)
                    .unwrap_or_else(|_| panic!("Failed covert to bytes {chunks:?}"))
            })
            .collect_vec();
        bytes.reverse();

        while bytes.len() < 32 {
            bytes.insert(0, 0);
        }

        let concat: &GenericArray<u8, U32> = GenericArray::from_slice(&bytes);

        FieldElement::from_bytes(concat).unwrap()
    }

    type BigOutput = BigInt;

    fn to_bigint(&self) -> Self::BigOutput
    where
        Self: Sized,
    {
        let mut bit_le = self
            .0
            .clone()
            .into_iter()
            .flat_map(|x| {
                let mut a = x.to_bits_le();
                a.truncate(K);
                a
            })
            .collect::<Vec<_>>();

        bit_le.truncate(NUM_BITS);

        let bytes = bit_le
            .chunks(8)
            .map(|chunks| {
                u8::from_bits_le(chunks)
                    .unwrap_or_else(|_| panic!("Failed covert to bytes {chunks:?}"))
            })
            .collect_vec();

        BigInt::from_bytes_le(Sign::Plus, &bytes)
    }

    type ScalarOutput = Scalar;

    fn to_nonnative_scalar(&self) -> Self::ScalarOutput
    where
        Self: Sized,
    {
        let mut bit_le = self
            .0
            .clone()
            .into_iter()
            .flat_map(|x| {
                let mut a = x.to_bits_le();
                a.truncate(K);
                a
            })
            .collect::<Vec<_>>();

        bit_le.truncate(NUM_BITS);

        let mut bytes = bit_le
            .chunks(8)
            .map(|chunks| {
                u8::from_bits_le(chunks)
                    .unwrap_or_else(|_| panic!("Failed covert to bytes {chunks:?}"))
            })
            .collect_vec();
        bytes.reverse();

        while bytes.len() < 32 {
            bytes.insert(0, 0);
        }

        let concat: GenericArray<u8, U32> = *GenericArray::from_slice(&bytes);

        Scalar::from_repr(concat).unwrap()
    }
}

/// The Arithmetic operation for native fields
/// which converted from the non-native field
impl<N: Network> Arithmetic for Polynomial<N> {
    /// Add two native polynomials which converted from two non-native fields respectively.
    /// e.g
    /// non_native_a -> presented as [native_x1, native_x2, native_x3]
    /// non_native_b -> presented as [native_x1', native_x2', native_x3']
    /// non_native_a + non_native_b ->
    ///     presented as [native_x1 + native_x1', native_x2 + native_x2', native_x3 + native_x3']
    /// Note: Addition in this operation will not access the range of the native field.
    fn raw_add(&self, other: &Self) -> Self {
        let degree1 = self.0.len();
        let degree2 = other.0.len();
        let result_degree = degree1.max(degree2);
        let mut result: Vec<Field<N>> = vec![Field::zero(); result_degree];

        for i in 0..degree1 {
            result[i] += self.0[i];
        }

        for i in 0..degree2 {
            result[i] += other.0[i];
        }
        Polynomial(result)
    }

    /// Multiply two native polynomials which converted from two non-native fields respectively.
    /// e.g
    /// non_native_a -> presented as [native_x1, native_x2, native_x3]
    /// non_native_b -> presented as [native_x1', native_x2', native_x3']
    /// non_native_a * non_native_b ->
    ///     presented as [native_x1 * native_x1',
    ///                   native_x2 * native_x1' + native_x1 * native_x2',
    ///                   native_x3 * native_x1' + native_x2 * native_x2' + native_x1* native_x3',
    ///                   native_x2 * native_x3' + native_x3 * native_x2'.
    ///                   native_x3 * native_x3' ]
    /// Note: Multipilication in this operation will not access the range of the native field.
    fn raw_mul(&self, other: &Self) -> Self {
        let degree1 = self.0.len();
        let degree2 = other.0.len();
        let result_degree = degree1 + degree2 - 1;
        let mut result: Vec<Field<N>> = vec![Field::zero(); result_degree];

        for i in 0..degree1 {
            for j in 0..degree2 {
                result[i + j] += self.0[i] * other.0[j];
            }
        }
        Polynomial(result)
    }

    /// Calculate e for the Multipilication result from two native polynomials.
    /// e.g
    /// c is the multiplication result
    ///   presented as [ c1 = native_x1 * native_x1', //around 192 bit length
    ///                  c2 = native_x2 * native_x1' + native_x1 * native_x2', //around 193 bit length
    ///                  c3 = native_x3 * native_x1' + native_x2 * native_x2' + native_x1* native_x3', //around 194 bit length
    ///                  c4 = native_x2 * native_x3' + native_x3 * native_x2'. //around 193 bit length
    ///                  c5 = native_x3 * native_x3' ] //around 192 bit length
    /// e is calculated as [e1, e2, e3, e4, e5]
    /// such that, K = 96 in our setting
    /// c1 = k1 + e1 * 2^K , c2 = k2 + e2 * 2^K, ...etc
    /// where k1, k2, k3, k4, k5 have K bit length, and e1, e2, e3, e4, e5 have most K + 2 bit length
    fn calculate_e(&self) -> Self {
        let degree = self.0.len();
        let mut e: Vec<Field<N>> = vec![Field::zero(); degree];

        for i in 0..degree {
            let mut poly_tmp = self.0[i];
            if i > 0 {
                poly_tmp = poly_tmp + e[i - 1]
            }
            let mut _e = poly_tmp.to_bits_be();
            _e.truncate(_e.len() - K);
            e[i] = Field::from_bits_be(&_e).unwrap();
        }
        Polynomial(e)
    }

    /// linear combination of several native polynomials which converted from non-native fields respectively.
    /// e.g
    /// non_native_a -> presented as [a1, a2, a3]
    /// non_native_b -> presented as [b1, b2, b3]
    /// non_native_c -> presented as [c1, c2, c3]
    /// non_native_q_256 -> presented as [q1, q2, q3]
    /// Given non_native_a + non_native_b - non_native_c \mod non_native_q_256 = non_native_d \mod non_native_q_256
    /// non_native_d -> presented as [d1, d2, d3]
    /// non_native_a + non_native_b - non_native_c ->
    ///     presented as [a1 + b1 - c1 + q1, a2 + b2 - c2 + q2, a3 + b3 - c3 + q3]
    /// Note: Since the substruction result of two native field over native q may be negtive.
    ///       then a1 + b1 - c1 != d1 over native q
    ///       Therefore, it needs add non_native_q_256 with the substruction result to make the value positive.
    ///       It's correct since non_native_a + non_native_b - non_native_c +  non_native_q_256 \mod non_native_q_256
    ///                                     = non_native_d +  non_native_q_256 \mod non_native_q_256
    fn raw_linear_combination(terms: &Vec<(Signed, Self)>, modulo: &Self) -> Self
    where
        Self: Sized,
    {
        let result_degree = terms.iter().map(|(_, b)| b.0.len()).max().unwrap();
        let mut result: Vec<Field<N>> = vec![Field::zero(); result_degree];

        for (s, item) in terms.iter() {
            for i in 0..item.0.len() {
                match s {
                    Signed::ADD => {
                        result[i] += item.0[i];
                    }
                    Signed::SUB => {
                        result[i] = result[i] - item.0[i] + modulo.0[i];
                    }
                }
            }
        }
        Polynomial(result)
    }
}

#[cfg(test)]
mod tests {
    use k256::elliptic_curve::Field;

    use super::*;

    const ITERATIONS: u64 = 10_000;

    #[test]
    fn test_secp256_to_console_field() {
        for _ in 0..ITERATIONS {
            let expected = FieldElement::random(&mut OsRng);

            let candidate: Polynomial<Testnet3> = Polynomial::from_nonnative_field(&expected);

            assert_eq!(candidate.0.len(), 3);

            assert_eq!(expected, candidate.to_nonnative_field());
        }
    }

    #[test]
    fn test_secp256_to_console_field_constant() {
        for _ in 0..ITERATIONS {
            let expected = FieldElement::from_u64(2);

            let candidate: Polynomial<Testnet3> = Polynomial::from_nonnative_field(&expected);
            assert_eq!(candidate.0.len(), 3);

            assert_eq!(expected, candidate.to_nonnative_field());
        }
    }

    #[test]
    fn test_secp256_to_console_scalar() {
        for _ in 0..ITERATIONS {
            let expected: Scalar = Scalar::random(&mut OsRng);

            let candidate: Polynomial<Testnet3> = Polynomial::from_nonnative_scalar(&expected);
            assert_eq!(candidate.0.len(), 3);

            assert_eq!(expected, candidate.to_nonnative_scalar());
        }
    }

    #[test]
    fn test_bigint_to_console_field() {
        for _ in 0..ITERATIONS {
            let randfield: FieldElement = FieldElement::random(&mut OsRng);
            let expected = BigInt::from_bytes_be(Sign::Plus, &randfield.to_bytes().to_vec());

            let candidate: Polynomial<Testnet3> = Polynomial::from_bigint(&expected.clone());
            assert_eq!(candidate.0.len(), 3);

            assert_eq!(expected, candidate.to_bigint());
        }
    }

    #[test]
    fn test_bigintsmall_to_console_field() {
        for _ in 0..ITERATIONS {
            let expected = BigInt::from_u64(23).unwrap();

            let candidate: Polynomial<Testnet3> = Polynomial::from_bigint(&expected.clone());
            assert_eq!(candidate.0.len(), 3);

            assert_eq!(expected, candidate.to_bigint());
        }
    }
}
