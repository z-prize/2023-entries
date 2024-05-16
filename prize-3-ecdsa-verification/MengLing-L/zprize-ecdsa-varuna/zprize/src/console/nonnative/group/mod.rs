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

pub mod add;
pub mod double;
pub mod mul;

/// Given the point P on secp256k1 curve.
/// it's x,y corrdinates are in the q_256 field.
/// The challenge is how to verify point P is on secp256k1 curve in native field q_253.
///
/// To prove one point is on secp256k1 curve,
/// The corrdinates of the point should satisfy:
/// 1. x^3 + 7 = y^2 mod q_256
/// 2. x, y are in the q_256 field
///
/// In other words, we should ensure that
/// x_2 = x * x mod q_256
/// x_3 = x_2 * x mod q_256
/// y_2 = y * y mod q_256
/// x_3_add_7 = x_3 + 7 mod q_256
/// x_3_add_7 = y_2 mod q_256
///
/// Therefore, we can precompute x_2, x_3, x_3_add_7, y_2 over q_256,
/// and view the above conditions as the non-native mulitiplication
/// or non-native linear combination.
///
/// For example:
/// given non-native multiplication
/// x_2 = x * x mod q_256 //over non-native field
/// -> x * x = x_2 + alpha * q_256 //over integer
///
/// Therefore if the following condition is satisfy, we can belive that x_2 indeed equal x * x mod q_256.
/// -> x(X) * x(X) = x_2(X) + alpha(X) * q_256(X) + (2^K - X)e(X) mod native_q //over native field
/// where f(X) = f_1 + f_2 * X + f_3 * X^2 and f = f_1 + f_2 * 2^K + f_3 * (2^K)^2
/// and f_1, f_2 and f_3 \in native field
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsoleNonNativePoint<N: Network> {
    pub(crate) x_: FieldElement,
    pub(crate) y_: FieldElement,

    // the x corrdinate of the point
    pub(crate) x: Polynomial<N>,
    // the y corrdinate of the point
    pub(crate) y: Polynomial<N>,

    // non-nvative multiplication x_2 = x * x mod q_256
    pub(crate) x_2: Mul<N>,
    // non-nvative multiplication x_3 = x_2 * x mod q_256
    pub(crate) x_3: Mul<N>,
    // non-nvative multiplication y_2 = y * y mod q_256
    pub(crate) y_2: Mul<N>,
    // non-nvative linear combination x_3_add_7 = x_3 + 7 mod q_256
    pub(crate) x_3_add_7: LinearCombination<N>,

    // the flag to show whether this point is infinity point or not
    // if the point is not infinity , the flag equals true
    // otherwise, it equals false
    pub(crate) noninfinity: bool,
}

impl<N: Network> ConsoleNonNativePoint<N> {
    /// Create a non-native point unit from secp256k1 point.
    /// Namely, it includes two main steps:
    /// 1. Precompute x_2, x_3, y_2, x_3_add_7 over q_256
    /// 2. Construct the corresponding non-nvative multiplication
    ///         and non-nvative linear combination for those values.
    ///
    /// Note: If the point is the point at infinity,
    ///         x, y, x_2, x_3, y_2, x_3_add_7 should be zero;
    pub fn new(input: &AffinePoint) -> Self {
        let projective_point: ProjectivePoint = input.into();
        if projective_point == ProjectivePoint::IDENTITY {
            return Self {
                x_: FieldElement::ZERO,
                y_: FieldElement::ZERO,
                x: Polynomial::from_nonnative_field(&FieldElement::ZERO),
                y: Polynomial::from_nonnative_field(&FieldElement::ZERO),
                x_2: Mul::to_mul(
                    &FieldElement::ZERO,
                    &FieldElement::ZERO,
                    &FieldElement::ZERO,
                ),
                x_3: Mul::to_mul(
                    &FieldElement::ZERO,
                    &FieldElement::ZERO,
                    &FieldElement::ZERO,
                ),
                y_2: Mul::to_mul(
                    &FieldElement::ZERO,
                    &FieldElement::ZERO,
                    &FieldElement::ZERO,
                ),
                x_3_add_7: LinearCombination::to_linear_combination(
                    &[
                        (Signed::ADD, FieldElement::ZERO),
                        (Signed::ADD, FieldElement::ZERO),
                    ],
                    &FieldElement::ZERO,
                ),
                noninfinity: false,
            };
        }

        let encoded_point = input.to_encoded_point(false);
        let x = encoded_point
            .x()
            .map_or(FieldElement::ZERO, |b| FieldElement::from_bytes(b).unwrap());
        let y = encoded_point
            .y()
            .map_or(FieldElement::ZERO, |b| FieldElement::from_bytes(b).unwrap());

        let seven = FieldElement::from_u64(7);

        let x_2 = x.square().normalize();
        let x_3 = (x_2 * &x).normalize();
        let y_2 = y.square().normalize();

        let x_3_add_seven = x_3 + &seven;

        assert_eq!(x_3_add_seven.normalize(), y_2);

        Self {
            x_: x,
            y_: y,
            x: Polynomial::from_nonnative_field(&x),
            y: Polynomial::from_nonnative_field(&y),
            x_2: Mul::to_mul(&x, &x, &x_2),
            x_3: Mul::to_mul(&x_2, &x, &x_3),
            y_2: Mul::to_mul(&y, &y, &y_2),
            x_3_add_7: LinearCombination::to_linear_combination(
                &[(Signed::ADD, x_3), (Signed::ADD, seven)],
                &x_3_add_seven,
            ),
            noninfinity: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const ITERATIONS: u64 = 100;
    #[test]
    fn test_secp256_point() {
        for _ in 0..ITERATIONS {
            let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
            let public_key = secret_key.verifying_key();

            let _: ConsoleNonNativePoint<Testnet3> =
                ConsoleNonNativePoint::new(&public_key.as_affine());
        }
    }
}
