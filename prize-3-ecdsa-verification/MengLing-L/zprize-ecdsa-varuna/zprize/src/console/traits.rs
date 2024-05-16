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

use super::nonnative::basic::linear::Signed;

/// Convert non native field x 256 bits into three native fields x1, x2, x3
/// such that x = x1 + x2\codt 2^96 + x3\cdot 2^192
/// x1,x2 and x3 have 96,96, and 64 bit length, respectively.
pub trait ToPolynomial {
    type FieldInput;

    type BigInput;

    type ScalarInput;

    /// Convert secp256k1 base field into non native fields.
    fn from_nonnative_field(input: &Self::FieldInput) -> Self;

    /// Convert secp256k1 scalar field into non native fields.
    fn from_nonnative_scalar(input: &Self::ScalarInput) -> Self;

    /// Convert intermediate bigint during operation into non native fields.
    fn from_bigint(input: &Self::BigInput) -> Self;
}

/// Recover non native field from native fields
pub trait FromPolynomial {
    type FieldOutput;

    type BigOutput;

    type ScalarOutput;

    fn to_nonnative_field(&self) -> Self::FieldOutput
    where
        Self: Sized;

    fn to_nonnative_scalar(&self) -> Self::ScalarOutput
    where
        Self: Sized;

    fn to_bigint(&self) -> Self::BigOutput
    where
        Self: Sized;
}

pub trait Arithmetic {
    fn raw_linear_combination(terms: &Vec<(Signed, Self)>, modulo: &Self) -> Self
    where
        Self: Sized;

    fn raw_add(&self, other: &Self) -> Self;

    fn raw_mul(&self, other: &Self) -> Self;

    fn calculate_e(&self) -> Self;
}

pub trait ToMul {
    type Input;

    type ScalarInput;

    fn to_mul(a: &Self::Input, b: &Self::Input, c: &Self::Input) -> Self;

    fn scalar_to_mul(a: &Self::ScalarInput, b: &Self::ScalarInput, c: &Self::ScalarInput) -> Self;
}

pub trait ToLinearCombination {
    type Input;

    fn to_linear_combination(terms: &[(Signed, Self::Input)], value: &Self::Input) -> Self;
}

/// Unary operator for instantiating from bits.
pub trait SmallField {
    fn insert_intermidate_zero(&self, num_of_zeros: usize) -> Self;
}
