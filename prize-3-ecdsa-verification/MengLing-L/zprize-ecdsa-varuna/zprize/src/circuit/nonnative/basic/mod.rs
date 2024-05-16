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

pub mod linear;
pub mod mul;
pub mod ternary;

impl NonNativePolynomial for Vec<F> {
    fn poly_eq(&self, condition: &B, first: &Self, second: &Self) {
        for i in 0..self.len() {
            Env::enforce(|| ((&first[i] - &second[i]), condition, (&self[i] - &second[i])));
        }
    }

    fn poly_mul(&self, input: &Self) -> Self {
        let degree1 = self.len();
        let degree2 = input.len();
        let result_degree = degree1 + degree2 - 1;
        let mut result: Vec<F> = vec![F::zero(); result_degree];

        for i in 0..degree1 {
            for j in 0..degree2 {
                let tmp_prodect = &self[i] * &input[j];

                Env::enforce(|| (&self[i], &input[j], &tmp_prodect));

                result[i + j] += &tmp_prodect;
            }
        }
        result
    }

    fn poly_add(&self, input: &Self) -> Self {
        let degree1 = self.len();
        let degree2 = input.len();
        let result_degree = degree1.max(degree2);
        let mut result: Vec<F> = vec![F::zero(); result_degree];

        for i in 0..degree1 {
            result[i] += &self[i];
        }

        for i in 0..degree2 {
            result[i] += &input[i];
        }
        result
    }
}
