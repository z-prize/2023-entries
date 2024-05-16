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
#[derive(Clone, Debug)]
pub struct CircuitNonNativePointDouble {
    pub(crate) point_c: CircuitNonNativePoint,

    pub(crate) lambda: Vec<F>,

    pub(crate) lambda_2: CircuitNonNativeMul,
    pub(crate) two_mul_y: CircuitNonNativeMul,
    pub(crate) three_mul_x_2: CircuitNonNativeMul,
    pub(crate) lambda_mul_x_sub_x_c: CircuitNonNativeMul,
    pub(crate) lambda_mul_two_mul_y: CircuitNonNativeMul,

    pub(crate) x_sub_x_c: CircuitNonNativeLinearComb,
    pub(crate) y_c_add_y: CircuitNonNativeLinearComb,
    pub(crate) lambda_2_sub_x_sub_x: CircuitNonNativeLinearComb,
}

impl Inject for CircuitNonNativePointDouble {
    type Primitive = ConsoleNonNativePointDouble<Testnet3>;

    fn new(mode: Mode, console_point_double: Self::Primitive) -> Self {
        Self {
            point_c: CircuitNonNativePoint::new(mode, console_point_double.point_c),

            lambda: console_point_double
                .lambda
                .0
                .iter()
                .map(|b| F::new(mode, *b))
                .collect_vec(),
            lambda_2: CircuitNonNativeMul::new(mode, console_point_double.lambda_2),
            two_mul_y: CircuitNonNativeMul::new(mode, console_point_double.two_mul_y),
            three_mul_x_2: CircuitNonNativeMul::new(mode, console_point_double.three_mul_x_2),
            lambda_mul_x_sub_x_c: CircuitNonNativeMul::new(
                mode,
                console_point_double.lambda_mul_x_sub_x_c,
            ),
            lambda_mul_two_mul_y: CircuitNonNativeMul::new(
                mode,
                console_point_double.lambda_mul_two_mul_y,
            ),
            x_sub_x_c: CircuitNonNativeLinearComb::new(mode, console_point_double.x_sub_x_c),
            y_c_add_y: CircuitNonNativeLinearComb::new(mode, console_point_double.y_c_add_y),
            lambda_2_sub_x_sub_x: CircuitNonNativeLinearComb::new(
                mode,
                console_point_double.lambda_2_sub_x_sub_x,
            ),
        }
    }
}

impl Eject for CircuitNonNativePointDouble {
    type Primitive = ConsoleNonNativePointDouble<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.lambda.eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        ConsoleNonNativePointDouble {
            point_c: self.point_c.eject_value(),
            lambda: Polynomial(self.lambda.eject_value()),
            lambda_2: self.lambda_2.eject_value(),
            two_mul_y: self.two_mul_y.eject_value(),
            three_mul_x_2: self.three_mul_x_2.eject_value(),
            lambda_mul_x_sub_x_c: self.lambda_mul_x_sub_x_c.eject_value(),
            lambda_mul_two_mul_y: self.lambda_mul_two_mul_y.eject_value(),
            x_sub_x_c: self.x_sub_x_c.eject_value(),
            y_c_add_y: self.y_c_add_y.eject_value(),
            lambda_2_sub_x_sub_x: self.lambda_2_sub_x_sub_x.eject_value(),
        }
    }
}

impl NonNativePointDouble for CircuitNonNativePointDouble {
    type Size = F;

    fn points_double(&self, point_a: &CircuitNonNativePoint) {
        CHUNK_SIZE.with(|chunk_size| {
            self.point_c.on_curve();

            self.lambda_2
                .poly_mul_wellform(&chunk_size, &self.lambda, &self.lambda);
            NONNATIVE_TWO.with(|two| {
                self.two_mul_y
                    .poly_mul_wellform(&chunk_size, two, &point_a.y);
            });
            NONNATIVE_THREE.with(|three| {
                self.three_mul_x_2
                    .poly_mul_wellform(&chunk_size, three, &point_a.x_2.c);
            });
            self.lambda_mul_x_sub_x_c.poly_mul_wellform(
                &chunk_size,
                &self.lambda,
                &self.x_sub_x_c.value,
            );

            self.x_sub_x_c.poly_linear_combination_wellform(
                &chunk_size,
                &vec![
                    (Signed::ADD, point_a.x.clone()),
                    (Signed::SUB, self.point_c.x.clone()),
                ],
            );

            self.y_c_add_y.poly_linear_combination_wellform(
                &chunk_size,
                &vec![
                    (Signed::ADD, self.point_c.y.clone()),
                    (Signed::ADD, point_a.y.clone()),
                ],
            );

            self.lambda_2_sub_x_sub_x.poly_linear_combination_wellform(
                &chunk_size,
                &vec![
                    (Signed::ADD, self.lambda_2.c.clone()),
                    (Signed::SUB, point_a.x.clone()),
                    (Signed::SUB, point_a.x.clone()),
                ],
            );

            //x' == lambda^2 - x - x
            self.point_c.x.poly_eq(
                &point_a.noninfinity,
                &self.lambda_2_sub_x_sub_x.value,
                &self.lambda_2_sub_x_sub_x.value,
            );

            //y == lambda(x - x') - y
            self.y_c_add_y.value.poly_eq(
                &point_a.noninfinity,
                &self.lambda_mul_x_sub_x_c.c,
                &self.lambda_mul_x_sub_x_c.c,
            );

            // lambda*2y = 3x^2
            self.lambda_mul_two_mul_y.c.poly_eq(
                &point_a.noninfinity,
                &self.three_mul_x_2.c,
                &self.three_mul_x_2.c,
            );
        });
    }
}
