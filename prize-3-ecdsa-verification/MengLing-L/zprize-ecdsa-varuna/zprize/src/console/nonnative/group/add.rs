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

/// Given three points A, B and C on secp256k1 curve.
/// it's they corrdinates are in the q_256 field.
/// The challenge is how to prove C = A + B on native field q_253.
///
/// To prove C equal A + B,
/// they corrdinates should satisfy:
/// lambda = (y_a - y_b)/(x_a - x_b)
/// x_c = lambda^2 - x_b - x_a
/// y_c = lambda * (x_b - x_c) - y_b
///
/// In other words, we should ensure that
/// lambda_2 = lambda * lambda mod q_256
/// x_a_sub_x_b = x_a - x_b mod q_256
/// y_a_sub_y_b = y_a - y_b mod q_256
/// x_b_sub_x_c = x_b - x_c mod q_256
/// lambda_mul_x_a_sub_x_b = lambda * x_a_sub_x_b mod q_256
/// lambda_2_sub_x_b_sub_x_a = lambda_2 - x_b - x_a mod q_256
/// lambda_mul_x_b_sub_x_c = lambda * x_b_sub_x_c mod q_256
/// lambda_mul_x_b_sub_x_c_sub_y_b = lambda_mul_x_b_sub_x_c - y_b mod q_256
/// x_c = lambda_2_sub_x_b_sub_x_a mod q_256
/// y_c = lambda_mul_x_b_sub_x_c_sub_y_b mod q_256
///
/// Therefore, we can precompute lambda_2, x_a_sub_x_b, y_a_sub_y_b, x_b_sub_x_c
///             , lambda_mul_x_a_sub_x_b, lambda_2_sub_x_b_sub_x_a, lambda_mul_x_b_sub_x_c
///                 lambda_mul_x_b_sub_x_c_sub_y_b  over q_256,
/// and view the above conditions as the non-native mulitiplication
/// or non-native linear combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsoleNonNativePointAdd<N: Network> {
    pub(crate) point_a: ConsoleNonNativePoint<N>,
    pub(crate) point_c: ConsoleNonNativePoint<N>,

    pub(crate) lambda: Polynomial<N>,

    pub(crate) lambda_2: Mul<N>,
    pub(crate) lambda_mul_x_a_sub_x_b: Mul<N>,
    pub(crate) lambda_mul_x_b_sub_x_c: Mul<N>,

    pub(crate) x_a_sub_x_b: LinearCombination<N>,
    pub(crate) y_a_sub_y_b: LinearCombination<N>,
    pub(crate) x_b_sub_x_c: LinearCombination<N>,

    pub(crate) lambda_2_sub_x_b_sub_x_a: LinearCombination<N>,
    pub(crate) lambda_mul_x_b_sub_x_c_sub_y_b: LinearCombination<N>,
}

impl<N: Network> ConsoleNonNativePointAdd<N> {
    /// Create a non-native point add unit from three secp256k1 points C = A + B.
    /// Namely, it includes two main steps:
    /// 1. Precompute and record lambda_2, x_a_sub_x_b, y_a_sub_y_b, x_b_sub_x_c, lambda_mul_x_a_sub_x_b,
    ///               lambda_2_sub_x_b_sub_x_a, lambda_mul_x_b_sub_x_c
    ///                lambda_mul_x_b_sub_x_c_sub_y_b  over q_256,
    /// 2. Construct the corresponding non-nvative multiplication
    ///         and non-nvative linear combination for those values.
    ///
    /// Note: If A is at infinity, C = B
    ///       If B is at infinity, C = A
    ///       Besides, if any point of A,B is at infinity
    ///       lambda_mul_x_a_sub_x_b, lambda_mul_x_b_sub_x_c
    ///       lambda_2_sub_x_b_sub_x_a, lambda_mul_x_b_sub_x_c_sub_y_b should be zero.
    pub fn new(
        nonnative_point_a: &ProjectivePoint,
        nonnative_point_b: &ProjectivePoint,
        nonnative_point_c: &ProjectivePoint,
    ) -> Self {
        assert_eq!(
            (nonnative_point_a + nonnative_point_b).to_bytes(),
            nonnative_point_c.to_bytes()
        );

        let point_a: ConsoleNonNativePoint<N> =
            ConsoleNonNativePoint::new(&nonnative_point_a.into());
        let point_b: ConsoleNonNativePoint<N> =
            ConsoleNonNativePoint::new(&nonnative_point_b.into());
        let point_c: ConsoleNonNativePoint<N> =
            ConsoleNonNativePoint::new(&nonnative_point_c.into());

        let mut y_a_sub_y_b_: Vec<(Signed, FieldElement)> = Vec::with_capacity(2);
        let mut x_a_sub_x_b_: Vec<(Signed, FieldElement)> = Vec::with_capacity(2);
        let mut x_b_sub_x_c_: Vec<(Signed, FieldElement)> = Vec::with_capacity(2);
        let mut y_c_add_y_b_: Vec<(Signed, FieldElement)> = Vec::with_capacity(2);
        let mut lambda_2_sub_x_b_sub_x_a_: Vec<(Signed, FieldElement)> = Vec::with_capacity(3);
        let mut lambda_mul_x_b_sub_x_c_sub_y_b_: Vec<(Signed, FieldElement)> =
            Vec::with_capacity(2);
        let mut lambda_mul_x_a_sub_x_b_: Mul<N> = Mul::to_mul(
            &FieldElement::ZERO,
            &FieldElement::ZERO,
            &FieldElement::ZERO,
        );
        let mut lambda_mul_x_b_sub_x_c_: Mul<N> = Mul::to_mul(
            &FieldElement::ZERO,
            &FieldElement::ZERO,
            &FieldElement::ZERO,
        );

        let y_a_sub_y_b = (point_a.y_ - point_b.y_).normalize();
        y_a_sub_y_b_.push((Signed::ADD, point_a.y_));
        y_a_sub_y_b_.push((Signed::SUB, point_b.y_));
        let x_a_sub_x_b = (point_a.x_ - point_b.x_).normalize();
        x_a_sub_x_b_.push((Signed::ADD, point_a.x_));
        x_a_sub_x_b_.push((Signed::SUB, point_b.x_));
        let x_b_sub_x_c = (point_b.x_ - point_c.x_).normalize();
        x_b_sub_x_c_.push((Signed::ADD, point_b.x_));
        x_b_sub_x_c_.push((Signed::SUB, point_c.x_));
        let _y_c_add_y_b = (point_c.y_ + point_b.y_).normalize();
        y_c_add_y_b_.push((Signed::ADD, point_c.y_));
        y_c_add_y_b_.push((Signed::ADD, point_b.y_));

        let invert = x_a_sub_x_b.invert().unwrap_or(FieldElement::ZERO);

        let lambda = (y_a_sub_y_b * invert).normalize();
        let lambda_2 = lambda.square().normalize();
        let lambda_mul_x_b_sub_x_c = (lambda * x_b_sub_x_c).normalize();
        let lambda_mul_x_a_sub_x_b = (lambda * x_a_sub_x_b).normalize();
        let lambda_mul_x_b_sub_x_c_sub_y_b = (lambda_mul_x_b_sub_x_c - point_b.y_).normalize();
        let lambda_2_sub_x_b_sub_x_a = (lambda_2 - point_b.x_ - point_a.x_).normalize();

        if !point_a.noninfinity || !point_b.noninfinity {
            lambda_2_sub_x_b_sub_x_a_.push((Signed::ADD, FieldElement::ZERO));
            lambda_2_sub_x_b_sub_x_a_.push((Signed::SUB, FieldElement::ZERO));
            lambda_2_sub_x_b_sub_x_a_.push((Signed::SUB, FieldElement::ZERO));

            lambda_mul_x_b_sub_x_c_sub_y_b_.push((Signed::ADD, FieldElement::ZERO));
            lambda_mul_x_b_sub_x_c_sub_y_b_.push((Signed::SUB, FieldElement::ZERO));
        } else {
            lambda_mul_x_b_sub_x_c_ = Mul::to_mul(&lambda, &x_b_sub_x_c, &lambda_mul_x_b_sub_x_c);
            lambda_mul_x_a_sub_x_b_ = Mul::to_mul(&lambda, &x_a_sub_x_b, &lambda_mul_x_a_sub_x_b);
            lambda_2_sub_x_b_sub_x_a_.push((Signed::ADD, lambda_2));
            lambda_2_sub_x_b_sub_x_a_.push((Signed::SUB, point_b.x_));
            lambda_2_sub_x_b_sub_x_a_.push((Signed::SUB, point_a.x_));
            lambda_mul_x_b_sub_x_c_sub_y_b_.push((Signed::ADD, lambda_mul_x_b_sub_x_c));
            lambda_mul_x_b_sub_x_c_sub_y_b_.push((Signed::SUB, point_b.y_));
            assert_eq!(
                point_c.x_, lambda_2_sub_x_b_sub_x_a,
                "Point Addtion: point C != A + B"
            );
            assert_eq!(
                point_c.y_, lambda_mul_x_b_sub_x_c_sub_y_b,
                "Point Addtion: point C != A + B"
            );
        }

        Self {
            point_a: point_a.clone(),
            point_c: point_c.clone(),
            lambda: Polynomial::from_nonnative_field(&lambda),
            lambda_2: Mul::to_mul(&lambda, &lambda, &lambda_2),
            lambda_mul_x_a_sub_x_b: lambda_mul_x_a_sub_x_b_,
            lambda_mul_x_b_sub_x_c: lambda_mul_x_b_sub_x_c_,
            x_a_sub_x_b: LinearCombination::to_linear_combination(
                &x_a_sub_x_b_,
                &x_a_sub_x_b_
                    .iter()
                    .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                        Signed::ADD => acc + x,
                        Signed::SUB => acc - x,
                    })
                    .normalize(),
            ),
            y_a_sub_y_b: LinearCombination::to_linear_combination(
                &y_a_sub_y_b_,
                &y_a_sub_y_b_
                    .iter()
                    .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                        Signed::ADD => acc + x,
                        Signed::SUB => acc - x,
                    })
                    .normalize(),
            ),
            x_b_sub_x_c: LinearCombination::to_linear_combination(
                &x_b_sub_x_c_,
                &x_b_sub_x_c_
                    .iter()
                    .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                        Signed::ADD => acc + x,
                        Signed::SUB => acc - x,
                    })
                    .normalize(),
            ),
            lambda_2_sub_x_b_sub_x_a: LinearCombination::to_linear_combination(
                &lambda_2_sub_x_b_sub_x_a_,
                &lambda_2_sub_x_b_sub_x_a_
                    .iter()
                    .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                        Signed::ADD => acc + x,
                        Signed::SUB => acc - x,
                    })
                    .normalize(),
            ),
            lambda_mul_x_b_sub_x_c_sub_y_b: LinearCombination::to_linear_combination(
                &lambda_mul_x_b_sub_x_c_sub_y_b_,
                &lambda_mul_x_b_sub_x_c_sub_y_b_
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
        let _: ProjectivePoint = affine_point.into();

        let projective_base: ProjectivePoint = ProjectivePoint::GENERATOR;
        let _: AffinePoint = projective_base.into();

        let add = ProjectivePoint::IDENTITY + projective_base;

        let _: ConsoleNonNativePointAdd<Testnet3> =
            ConsoleNonNativePointAdd::new(&ProjectivePoint::IDENTITY, &projective_base, &add);

        // let console_add: ConsoleNonNativePointAdd<Testnet3> =
        // ConsoleNonNativePointAdd::new(&ProjectivePoint::IDENTITY, &ProjectivePoint::IDENTITY, &add);
    }

    #[test]
    fn test_secp256_point_add_two_identy() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = secret_key.verifying_key();

        let affine_point = public_key.as_affine();
        let _: ProjectivePoint = affine_point.into();

        let projective_base: ProjectivePoint = ProjectivePoint::GENERATOR;
        let _: AffinePoint = projective_base.into();

        let add = ProjectivePoint::IDENTITY + ProjectivePoint::IDENTITY;

        let _: ConsoleNonNativePointAdd<Testnet3> = ConsoleNonNativePointAdd::new(
            &ProjectivePoint::IDENTITY,
            &ProjectivePoint::IDENTITY,
            &add,
        );
    }

    #[test]
    fn test_secp256_point_add_one_identy() {
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = secret_key.verifying_key();

        let affine_point = public_key.as_affine();
        let projective_point: ProjectivePoint = affine_point.into();

        let projective_base: ProjectivePoint = ProjectivePoint::GENERATOR;
        let _: AffinePoint = projective_base.into();

        let add = projective_point + ProjectivePoint::IDENTITY;

        let _: ConsoleNonNativePointAdd<Testnet3> =
            ConsoleNonNativePointAdd::new(&projective_point, &ProjectivePoint::IDENTITY, &add);
    }
}
