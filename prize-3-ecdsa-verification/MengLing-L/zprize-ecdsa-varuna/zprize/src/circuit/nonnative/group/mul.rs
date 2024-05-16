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
/// Consider following Point multiplication algorithm over secp256k1 curve base field
/// It also records all intermediate points of point additions and doublings.
/// output_point = scalar \cdot P where P is the base point
/// ====================================================
/// let bits = bit_representation(s) # the vector of bits (from LSB to MSB) representing scalar
/// let output = IDENTITY (point at infinity)
/// let temp = base point P
/// let point_add = Vec<NonNativePointAdd>
/// let point_double = Vec<NonNativePointDouble>
///
/// for bit in bits:
///     let old_output = output
///     if bit == 1:  
///         output =  temp + old_output # point add
///         point_add.push(NonNativePointAdd(temp, old_output, output))
///     else bit == 0:
///         output = IDENTITY + old_output # point add
///         point_add.push(NonNativePointAdd(IDENTITY, old_output, output))
///
///     let old_temp = temp
///     temp = temp + temp # point double
///     point_double.push(NonNativePointDouble(old_temp, temp))
/// return output
/// ====================================================
///
/// When given the scalar bits, intermedate points, the base point, and the multiplication result point over secp256k1 curve,
/// if we can prove that all intermedate points are correct computed from the base point in native field,
/// we can prove that the output_point = scalar \cdot P over secp256k1 curve.
///
/// Therefore, we can view all intermedate points as the NonNativePointAdd unit or the NonNativePointDouble unit.
/// Please go to ./add.rs ./double.rs to see how to prove all intermedate points are correct computed from the base point in native field
///
#[derive(Clone, Debug)]
pub struct CircuitNonNativePointMul {
    //pub(crate) scalar: Option<Vec<B>>,
    pub(crate) point_add: Vec<CircuitNonNativePointAdd>,
    pub(crate) point_double: Option<Vec<CircuitNonNativePointDouble>>,
}

impl Inject for CircuitNonNativePointMul {
    type Primitive = ConsoleNonNativePointMul<Testnet3>;

    fn new(mode: Mode, value: Self::Primitive) -> Self {
        // let scalar = value
        //     .scalar
        //     .map(|bits| bits.iter().map(|b| Boolean::new(mode, *b)).collect_vec());
        let point_double = value.point_double.map(|doubles| {
            doubles
                .iter()
                .map(|b| CircuitNonNativePointDouble::new(mode, b.clone()))
                .collect_vec()
        });
        Self {
            //scalar,
            point_add: value
                .point_add
                .iter()
                .map(|b| CircuitNonNativePointAdd::new(mode, b.clone()))
                .collect_vec(),
            point_double,
        }
    }
}

impl Eject for CircuitNonNativePointMul {
    type Primitive = ConsoleNonNativePointMul<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.point_add[0].eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        // let scalar = self
        //     .scalar
        //     .as_ref()
        //     .map(|bits| bits.iter().map(|b| b.eject_value()).collect_vec());
        let point_double = self
            .point_double
            .as_ref()
            .map(|doubles| doubles.iter().map(|b| b.eject_value()).collect_vec());
        ConsoleNonNativePointMul {
            // scalar,
            point_add: self.point_add.iter().map(|b| b.eject_value()).collect_vec(),
            point_double,
        }
    }
}

impl NonNativePointMul for CircuitNonNativePointMul {
    type Size = F;

    fn point_mul(&self, scalar: &Vec<B>, base_point: &CircuitNonNativePoint) {
        NONNATIVE_IDENTITY.with(|identity| {
            NONNATIVE_ZERO.with(|zeros| {
                scalar.iter().enumerate().for_each(|(i, bit)| {
                    if i == 0 {
                        self.point_double.as_ref().unwrap()[i].points_double(base_point);
                        self.point_add[i].points_add(identity);
                        self.point_add[i].point_c.x.poly_eq(
                            bit,
                            &self.point_add[i].point_c.x,
                            &identity.x,
                        );
                        self.point_add[i].point_c.y.poly_eq(
                            bit,
                            &self.point_add[i].point_c.y,
                            &identity.y,
                        );

                        self.point_add[i]
                            .point_a
                            .x
                            .poly_eq(bit, &base_point.x, zeros);
                        self.point_add[i]
                            .point_a
                            .y
                            .poly_eq(bit, &base_point.y, zeros);
                    } else {
                        self.point_double.as_ref().unwrap()[i]
                            .points_double(&self.point_double.as_ref().unwrap()[i - 1].point_c);
                        self.point_add[i].points_add(&self.point_add[i - 1].point_c);
                        self.point_add[i].point_c.x.poly_eq(
                            bit,
                            &self.point_add[i].point_c.x,
                            &self.point_add[i - 1].point_c.x,
                        );
                        self.point_add[i].point_c.y.poly_eq(
                            bit,
                            &self.point_add[i].point_c.y,
                            &self.point_add[i - 1].point_c.y,
                        );
                        self.point_add[i].point_a.x.poly_eq(
                            bit,
                            &self.point_double.as_ref().unwrap()[i - 1].point_c.x,
                            zeros,
                        );
                        self.point_add[i].point_a.y.poly_eq(
                            bit,
                            &self.point_double.as_ref().unwrap()[i - 1].point_c.y,
                            zeros,
                        );
                    }
                });
            });
        });
    }

    fn generator_mul(&self, scalar: &Vec<B>) {
        NONNATIVE_GENERATOR_G.with(|bases| {
            NONNATIVE_GENERATOR.with(|generator| {
                NONNATIVE_IDENTITY.with(|identity| {
                    scalar.iter().enumerate().for_each(|(i, bit)| {
                        if i == 0 {
                            bases[i].points_double(&generator);
                            self.point_add[i].points_add(identity);
                            self.point_add[i].point_c.x.poly_eq(
                                bit,
                                &self.point_add[i].point_c.x,
                                &identity.x,
                            );
                            self.point_add[i].point_c.y.poly_eq(
                                bit,
                                &self.point_add[i].point_c.y,
                                &identity.y,
                            );

                            NONNATIVE_ZERO.with(|zeros| {
                                self.point_add[i]
                                    .point_a
                                    .x
                                    .poly_eq(bit, &generator.x, zeros);
                                self.point_add[i]
                                    .point_a
                                    .y
                                    .poly_eq(bit, &generator.y, zeros);
                            });
                        } else {
                            bases[i].points_double(&bases[i - 1].point_c);
                            self.point_add[i].points_add(&self.point_add[i - 1].point_c);
                            self.point_add[i].point_c.x.poly_eq(
                                bit,
                                &self.point_add[i].point_c.x,
                                &self.point_add[i - 1].point_c.x,
                            );
                            self.point_add[i].point_c.y.poly_eq(
                                bit,
                                &self.point_add[i].point_c.y,
                                &self.point_add[i - 1].point_c.y,
                            );
                            NONNATIVE_ZERO.with(|zeros| {
                                self.point_add[i].point_a.x.poly_eq(
                                    bit,
                                    &bases[i - 1].point_c.x,
                                    zeros,
                                );
                                self.point_add[i].point_a.y.poly_eq(
                                    bit,
                                    &bases[i - 1].point_c.y,
                                    zeros,
                                );
                            });
                        }
                    });
                });
            });
        });
    }

    fn chunk_size(&self) -> Self::Size {
        F::new(
            Mode::Constant,
            crate::console::Console::<Testnet3>::chunk_size(),
        )
    }
}
