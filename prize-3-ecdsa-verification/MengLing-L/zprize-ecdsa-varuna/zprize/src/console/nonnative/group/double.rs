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

/// Given two points A (x,y) and C (x_c, y_c) on secp256k1 curve.
/// it's they corrdinates are in the q_256 field.
/// The challenge is how to prove C = A + A on native field q_253.
///
/// To prove C equal A + A,
/// they corrdinates should satisfy:
/// lambda = 3x^2 / 2y mod q_256
/// x_c = lambda^2 - x - x mod q_256
/// y_c = lambda * (x - x_c) - y mod q_256
///
/// In other words, we should ensure that
/// x_sub_x_c = x - x_c mod q_256
/// y_c_add_y = y_c + y mod q_256
/// lambda_2 = lambda * lambda mod q_256
/// two_mul_y = two * y mod q_256
/// x_2 = x * x mod q_256
/// three_mul_x_2 = three * x_2 mod q_256
/// lambda_mul_two_mul_y = lambda * two_mul_y mod q_256
/// lambda_mul_x_sub_x_c = lambda * x_sub_x_c mod q_256
/// lambda_2_sub_x_sub_x = lambda_2 - x -x  mod q_256
///
/// Therefore, we can precompute lambda_2, x_sub_x_c, y_c_add_y, lambda_2
///             , two_mul_y, x_2, three_mul_x_2, lambda_mul_x_sub_x_c
///                 lambda_mul_two_mul_y, lambda_2_sub_x_sub_x over q_256,
/// and view the above conditions as the non-native mulitiplication
/// or non-native linear combination.
///
/// Note: This struct emit storing the non-native multiplication unit for x_2 = x * x mod q_256,
///       since this multiplication unit has been proven in the non-native point A unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsoleNonNativePointDouble<N: Network> {
    pub(crate) point_c: ConsoleNonNativePoint<N>,

    pub(crate) lambda: Polynomial<N>,

    pub(crate) lambda_2: Mul<N>,
    pub(crate) two_mul_y: Mul<N>,
    pub(crate) three_mul_x_2: Mul<N>,
    pub(crate) lambda_mul_x_sub_x_c: Mul<N>,
    pub(crate) lambda_mul_two_mul_y: Mul<N>,

    pub(crate) x_sub_x_c: LinearCombination<N>,
    pub(crate) y_c_add_y: LinearCombination<N>,
    pub(crate) lambda_2_sub_x_sub_x: LinearCombination<N>,
}

impl<N: Network> ConsoleNonNativePointDouble<N> {
    /// Create a non-native point double unit from tow secp256k1 points C = A + A.
    /// Namely, it includes two main steps:
    /// 1. Precompute and record lambda_2, x_sub_x_c, y_c_add_y, lambda_2
    ///             , two_mul_y, x_2, three_mul_x_2, lambda_mul_x_sub_x_c
    ///                 lambda_mul_two_mul_y, lambda_2_sub_x_sub_x over q_256,
    /// 2. Construct the corresponding non-nvative multiplication
    ///         and non-nvative linear combination for those values.
    pub fn new(nonnative_point_a: &ProjectivePoint, nonnative_point_c: &ProjectivePoint) -> Self {
        assert_eq!(
            (nonnative_point_a.double()).to_bytes(),
            nonnative_point_c.to_bytes()
        );

        let point_a: ConsoleNonNativePoint<N> =
            ConsoleNonNativePoint::new(&nonnative_point_a.into());
        let point_c: ConsoleNonNativePoint<N> =
            ConsoleNonNativePoint::new(&nonnative_point_c.into());
        let two = FieldElement::from_u64(2);
        let three = FieldElement::from_u64(3);

        let mut x_sub_x_c_: Vec<(Signed, FieldElement)> = Vec::with_capacity(2);
        let mut y_c_add_y_: Vec<(Signed, FieldElement)> = Vec::with_capacity(2);
        let mut lambda_2_sub_x_sub_x_: Vec<(Signed, FieldElement)> = Vec::with_capacity(3);

        let x_sub_x_c = (point_a.x_ - point_c.x_).normalize();
        x_sub_x_c_.push((Signed::ADD, point_a.x_));
        x_sub_x_c_.push((Signed::SUB, point_c.x_));
        let y_c_add_y = (point_c.y_ + point_a.y_).normalize();
        y_c_add_y_.push((Signed::ADD, point_c.y_));
        y_c_add_y_.push((Signed::ADD, point_a.y_));

        let x_2 = (point_a.x_ * point_a.x_).normalize();
        let three_mul_x_2 = (three * x_2).normalize();
        let two_mul_y = (two * point_a.y_).normalize();

        let invert = two_mul_y.invert().unwrap_or(FieldElement::ZERO);

        let lambda = (three_mul_x_2 * invert).normalize();
        let lambda_mul_x_sub_x_c = (lambda * x_sub_x_c).normalize();
        let lambda_mul_two_mul_y = (lambda * two_mul_y).normalize();
        let lambda_2 = lambda.square().normalize();
        let lambda_2_sub_x_sub_x = (lambda_2 - point_a.x_ - point_a.x_).normalize();
        lambda_2_sub_x_sub_x_.push((Signed::ADD, lambda_2));
        lambda_2_sub_x_sub_x_.push((Signed::SUB, point_a.x_));
        lambda_2_sub_x_sub_x_.push((Signed::SUB, point_a.x_));

        assert_eq!(point_c.x_, lambda_2_sub_x_sub_x);
        assert_eq!(y_c_add_y, lambda_mul_x_sub_x_c);
        assert_eq!(lambda_mul_two_mul_y, three_mul_x_2);

        Self {
            point_c,
            lambda: Polynomial::from_nonnative_field(&lambda),
            lambda_2: Mul::to_mul(&lambda, &lambda, &lambda_2),
            two_mul_y: Mul::to_mul(&two, &point_a.y_, &two_mul_y),
            three_mul_x_2: Mul::to_mul(&three, &x_2, &three_mul_x_2),
            lambda_mul_x_sub_x_c: Mul::to_mul(&lambda, &x_sub_x_c, &lambda_mul_x_sub_x_c),
            lambda_mul_two_mul_y: Mul::to_mul(&lambda, &two_mul_y, &lambda_mul_two_mul_y),
            x_sub_x_c: LinearCombination::to_linear_combination(
                &x_sub_x_c_,
                &x_sub_x_c_
                    .iter()
                    .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                        Signed::ADD => acc + x,
                        Signed::SUB => acc - x,
                    })
                    .normalize(),
            ),
            y_c_add_y: LinearCombination::to_linear_combination(
                &y_c_add_y_,
                &y_c_add_y_
                    .iter()
                    .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                        Signed::ADD => acc + x,
                        Signed::SUB => acc - x,
                    })
                    .normalize(),
            ),
            lambda_2_sub_x_sub_x: LinearCombination::to_linear_combination(
                &lambda_2_sub_x_sub_x_,
                &lambda_2_sub_x_sub_x_
                    .iter()
                    .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                        Signed::ADD => acc + x,
                        Signed::SUB => acc - x,
                    })
                    .normalize(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secp256_point_add() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = secret_key.verifying_key();

        let affine_point = public_key.as_affine();
        let projective_point: ProjectivePoint = affine_point.into();

        let projective_base: ProjectivePoint = ProjectivePoint::GENERATOR;
        let _: AffinePoint = projective_base.into();

        let add = projective_point.double();

        let _: ConsoleNonNativePointDouble<Testnet3> =
            ConsoleNonNativePointDouble::new(&projective_point, &add);
    }
}
