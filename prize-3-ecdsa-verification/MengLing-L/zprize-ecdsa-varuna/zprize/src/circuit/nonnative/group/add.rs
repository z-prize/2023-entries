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
#[derive(Clone, Debug)]
pub struct CircuitNonNativePointAdd {
    pub(crate) point_a: CircuitNonNativePoint,
    pub(crate) point_c: CircuitNonNativePoint,

    pub(crate) lambda: Vec<F>,

    pub(crate) lambda_2: CircuitNonNativeMul,
    pub(crate) lambda_mul_x_a_sub_x_b: CircuitNonNativeMul,
    pub(crate) lambda_mul_x_b_sub_x_c: CircuitNonNativeMul,

    pub(crate) x_a_sub_x_b: CircuitNonNativeLinearComb,
    pub(crate) y_a_sub_y_b: CircuitNonNativeLinearComb,
    pub(crate) x_b_sub_x_c: CircuitNonNativeLinearComb,

    pub(crate) lambda_2_sub_x_b_sub_x_a: CircuitNonNativeLinearComb,
    pub(crate) lambda_mul_x_b_sub_x_c_sub_y_b: CircuitNonNativeLinearComb,
}

impl Inject for CircuitNonNativePointAdd {
    type Primitive = ConsoleNonNativePointAdd<Testnet3>;

    fn new(mode: Mode, console_point_add: Self::Primitive) -> Self {
        Self {
            point_a: CircuitNonNativePoint::new(mode, console_point_add.point_a),
            point_c: CircuitNonNativePoint::new(mode, console_point_add.point_c),

            lambda: console_point_add
                .lambda
                .0
                .iter()
                .map(|b| F::new(mode, *b))
                .collect_vec(),
            lambda_2: CircuitNonNativeMul::new(mode, console_point_add.lambda_2),
            lambda_mul_x_a_sub_x_b: CircuitNonNativeMul::new(
                mode,
                console_point_add.lambda_mul_x_a_sub_x_b,
            ),
            lambda_mul_x_b_sub_x_c: CircuitNonNativeMul::new(
                mode,
                console_point_add.lambda_mul_x_b_sub_x_c,
            ),
            x_a_sub_x_b: CircuitNonNativeLinearComb::new(mode, console_point_add.x_a_sub_x_b),
            y_a_sub_y_b: CircuitNonNativeLinearComb::new(mode, console_point_add.y_a_sub_y_b),
            x_b_sub_x_c: CircuitNonNativeLinearComb::new(mode, console_point_add.x_b_sub_x_c),
            lambda_2_sub_x_b_sub_x_a: CircuitNonNativeLinearComb::new(
                mode,
                console_point_add.lambda_2_sub_x_b_sub_x_a,
            ),
            lambda_mul_x_b_sub_x_c_sub_y_b: CircuitNonNativeLinearComb::new(
                mode,
                console_point_add.lambda_mul_x_b_sub_x_c_sub_y_b,
            ),
        }
    }
}

impl Eject for CircuitNonNativePointAdd {
    type Primitive = ConsoleNonNativePointAdd<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.lambda.eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        ConsoleNonNativePointAdd {
            point_a: self.point_a.eject_value(),
            point_c: self.point_c.eject_value(),
            lambda: Polynomial(self.lambda.eject_value()),
            lambda_2: self.lambda_2.eject_value(),
            lambda_mul_x_a_sub_x_b: self.lambda_mul_x_a_sub_x_b.eject_value(),
            x_a_sub_x_b: self.x_a_sub_x_b.eject_value(),
            y_a_sub_y_b: self.y_a_sub_y_b.eject_value(),
            x_b_sub_x_c: self.x_b_sub_x_c.eject_value(),
            lambda_2_sub_x_b_sub_x_a: self.lambda_2_sub_x_b_sub_x_a.eject_value(),
            lambda_mul_x_b_sub_x_c_sub_y_b: self.lambda_mul_x_b_sub_x_c_sub_y_b.eject_value(),
            lambda_mul_x_b_sub_x_c: self.lambda_mul_x_b_sub_x_c.eject_value(),
        }
    }
}

impl NonNativePointAdd for CircuitNonNativePointAdd {
    type Size = F;

    fn points_add(&self, point_b: &CircuitNonNativePoint) {
        self.point_c.on_curve();
        CHUNK_SIZE.with(|chunk_size| {
            NONNATIVE_ZERO.with(|zeros| {
                let noninfinity = &(&self.point_a.noninfinity & &point_b.noninfinity);

                self.lambda_2
                    .poly_mul_wellform(&chunk_size, &self.lambda, &self.lambda);

                self.lambda_mul_x_a_sub_x_b.poly_mul_wellform(
                    &chunk_size,
                    &Vec::<F>::ternary(noninfinity, &self.lambda, zeros),
                    &Vec::<F>::ternary(noninfinity, &self.x_a_sub_x_b.value, zeros),
                );

                self.lambda_mul_x_b_sub_x_c.poly_mul_wellform(
                    &chunk_size,
                    &Vec::<F>::ternary(noninfinity, &self.lambda, zeros),
                    &Vec::<F>::ternary(noninfinity, &self.x_b_sub_x_c.value, zeros),
                );

                self.y_a_sub_y_b.poly_linear_combination_wellform(
                    &chunk_size,
                    &vec![
                        (Signed::ADD, self.point_a.y.clone()),
                        (Signed::SUB, point_b.y.clone()),
                    ],
                );

                self.x_a_sub_x_b.poly_linear_combination_wellform(
                    &chunk_size,
                    &vec![
                        (Signed::ADD, self.point_a.x.clone()),
                        (Signed::SUB, point_b.x.clone()),
                    ],
                );

                self.x_b_sub_x_c.poly_linear_combination_wellform(
                    &chunk_size,
                    &vec![
                        (Signed::ADD, point_b.x.clone()),
                        (Signed::SUB, self.point_c.x.clone()),
                    ],
                );

                self.lambda_2_sub_x_b_sub_x_a
                    .poly_linear_combination_wellform(
                        &chunk_size,
                        &vec![
                            (
                                Signed::ADD,
                                Vec::<F>::ternary(noninfinity, &self.lambda_2.c, zeros),
                            ),
                            (
                                Signed::SUB,
                                Vec::<F>::ternary(noninfinity, &point_b.x, zeros),
                            ),
                            (
                                Signed::SUB,
                                Vec::<F>::ternary(noninfinity, &self.point_a.x, zeros),
                            ),
                        ],
                    );

                // lambda(x_a - x_b) = y_a - y_b
                self.lambda_mul_x_a_sub_x_b
                    .c
                    .poly_eq(noninfinity, &self.y_a_sub_y_b.value, zeros);

                // x == lambda^2 - x_b - x_a
                self.point_c.x.poly_eq(
                    noninfinity,
                    &self.lambda_2_sub_x_b_sub_x_a.value,
                    &Vec::<F>::ternary(&self.point_a.noninfinity, &self.point_a.x, &point_b.x),
                );

                // y == lambda*(x_b - x) - y_b
                self.point_c.y.poly_eq(
                    noninfinity,
                    &self.lambda_mul_x_b_sub_x_c_sub_y_b.value,
                    &Vec::<F>::ternary(&self.point_a.noninfinity, &self.point_a.y, &point_b.y),
                );
            });
        });
    }
}
