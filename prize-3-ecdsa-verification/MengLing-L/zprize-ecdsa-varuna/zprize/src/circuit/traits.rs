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

use super::{nonnative::group::CircuitNonNativePoint, ECDSAPubKey, B, F};
use crate::console::nonnative::basic::linear::Signed;
use snarkvm_circuit_environment::{Eject, Inject};

pub trait NonNativePolynomial {
    //condition == 1 : first , else second
    fn poly_eq(&self, condition: &B, first: &Self, second: &Self);

    fn poly_mul(&self, input: &Self) -> Self;

    fn poly_add(&self, input: &Self) -> Self;
}

pub trait NonNativePoint {
    type Size;

    fn on_curve(&self);
}

pub trait NonNativeMul {
    type Size;

    fn poly_mul_wellform(&self, chunk_size: &Self::Size, a: &Vec<F>, b: &Vec<F>);

    fn scalar_poly_mul_wellform(&self, chunk_size: &Self::Size, a: &Vec<F>, b: &Vec<F>);
}

pub trait NonNativeLinearCombination {
    type Size;

    type Poly;

    fn poly_linear_combination_wellform(
        &self,
        chunk_size: &Self::Size,
        terms: &Vec<(Signed, Vec<F>)>,
    );

    fn poly_linear_combination(&self, terms: &Vec<(Signed, Vec<F>)>) -> Self::Poly;
}

pub trait NonNativePointAdd {
    type Size;

    fn points_add(&self, point_b: &CircuitNonNativePoint);
}

pub trait NonNativePointDouble {
    type Size;

    fn points_double(&self, point_a: &CircuitNonNativePoint);
}

pub trait NonNativePointMul {
    type Size;

    fn point_mul(&self, scalar: &Vec<B>, base_point: &CircuitNonNativePoint);

    fn generator_mul(&self, scalar: &Vec<B>);

    fn chunk_size(&self) -> Self::Size;
}

pub trait ECDSAVerify {
    fn verify(&self, hash_massge: &Vec<B>, public_key: &ECDSAPubKey);
}

pub trait Ternary {
    type Boolean;
    type Output;

    /// Returns `first` if `condition` is `true`, otherwise returns `second`.
    fn ternary(condition: &Self::Boolean, first: &Self, second: &Self) -> Self::Output
    where
        Self: Sized;
}

/// A trait for a hash function.
pub trait Hash {
    type Input: Inject + Eject + Clone;
    type BitsInput: Inject + Eject + Clone;
    type Output: Inject + Eject + Clone;

    /// Returns the hash of the given input.
    fn hash(&self, input: &[Self::Input], bits_input: &[Self::BitsInput]) -> Self::Output;
}

pub trait Intermediate {
    type Output;
    type InterOutput;

    fn calculate_intermediate_values(
        &self,
    ) -> (Self::InterOutput, Self::InterOutput, Self::InterOutput)
    where
        Self: Sized;

    fn calculate_padded_intermidate_values(&self) -> (Self::Output, Self::Output, Self::Output)
    where
        Self: Sized;

    fn recover_value_from_f(&self) -> Self;
}

pub trait Not<Rhs = Self> {
    type Output;

    #[must_use]
    fn not(&self) -> Self::Output;
}

pub trait RangeProof {
    fn range_prove(&self, bound_bit_length: usize);

    fn range_prove_include_negtive(&self, bound_bit_length: usize);

    fn less_than_non_native_base_field_modulus(&self);

    fn less_than_non_native_scalar_field_modulus(&self) -> Vec<B>;
}
