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

impl Ternary for Vec<F> {
    type Boolean = B;
    type Output = Self;

    /// Returns `first` if `condition` is `true`, otherwise returns `second`.
    fn ternary(condition: &Self::Boolean, first: &Self, second: &Self) -> Self::Output {
        let mut output: Vec<F> = Vec::with_capacity(max(first.len(), second.len()));

        // Compute the ternary over the field representation (for efficiency).
        for (first, second) in first.iter().zip(second.iter()) {
            output.push(Field::ternary(condition, first, second));
        }
        // Return the result.
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ITERATIONS: u64 = 32;

    fn check_ternary(
        name: &str,
        flag: bool,
        first: Polynomial<Testnet3>,
        second: Polynomial<Testnet3>,
        mode_condition: Mode,
        mode_a: Mode,
        mode_b: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
    ) {
        let expected = if flag { first.clone() } else { second.clone() };
        let condition = Boolean::<Circuit>::new(mode_condition, flag);
        let a = first.0.iter().map(|b| F::new(mode_a, *b)).collect_vec();
        let b = second.0.iter().map(|b| F::new(mode_b, *b)).collect_vec();

        Circuit::scope(name, || {
            let mut candidate: Vec<conField<Testnet3>> = Vec::with_capacity(expected.0.len());
            for (first, second) in a.iter().zip(b.iter()) {
                let temp = Field::ternary(&condition, &first, &second);
                candidate.push(temp.eject_value());
            }
            assert_eq!(expected, Polynomial(candidate));
            assert_scope!(num_constants, num_public, num_private, num_constraints);
        });
    }

    fn run_test(
        mode_condition: Mode,
        mode_a: Mode,
        mode_b: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
    ) {
        let check_ternary = |name: &str, flag, first, second| {
            check_ternary(
                name,
                flag,
                first,
                second,
                mode_condition,
                mode_a,
                mode_b,
                num_constants,
                num_public,
                num_private,
                num_constraints,
            )
        };

        for i in 0..ITERATIONS {
            for flag in [true, false] {
                let name = format!("{flag} ? {mode_a} : {mode_b}, {i}");

                let first = Polynomial::from_nonnative_field(&FieldElement::random(&mut OsRng));
                let second = Polynomial::from_nonnative_field(&FieldElement::random(&mut OsRng));

                check_ternary(&name, flag, first, second);
            }
        }

        let zero = Polynomial::from_nonnative_field(&FieldElement::ZERO);
        let one = Polynomial::from_nonnative_field(&FieldElement::from_u64(1));

        check_ternary("true ? zero : zero", true, zero.clone(), zero.clone());
        check_ternary("true ? zero : one", true, zero.clone(), one.clone());
        check_ternary("true ? one : zero", true, one.clone(), zero.clone());
        check_ternary("true ? one : one", true, one.clone(), one.clone());

        check_ternary("false ? zero : zero", false, zero.clone(), zero.clone());
        check_ternary("false ? zero : one", false, zero.clone(), one.clone());
        check_ternary("false ? one : zero", false, one.clone(), zero.clone());
        check_ternary("false ? one : one", false, one.clone(), one.clone());
    }

    #[test]
    fn test_if_constant_then_constant_else_constant() {
        run_test(Mode::Constant, Mode::Constant, Mode::Constant, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_constant_else_public() {
        run_test(Mode::Constant, Mode::Public, Mode::Constant, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_constant_else_private() {
        run_test(Mode::Constant, Mode::Private, Mode::Constant, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_public_else_constant() {
        run_test(Mode::Constant, Mode::Constant, Mode::Public, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_public_else_public() {
        run_test(Mode::Constant, Mode::Public, Mode::Public, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_public_else_private() {
        run_test(Mode::Constant, Mode::Private, Mode::Public, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_private_else_constant() {
        run_test(Mode::Constant, Mode::Constant, Mode::Private, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_private_else_public() {
        run_test(Mode::Constant, Mode::Public, Mode::Private, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_constant_then_private_else_private() {
        run_test(Mode::Constant, Mode::Private, Mode::Private, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_public_then_constant_else_constant() {
        run_test(Mode::Public, Mode::Constant, Mode::Constant, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_public_then_constant_else_public() {
        run_test(Mode::Public, Mode::Constant, Mode::Public, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_public_then_constant_else_private() {
        run_test(Mode::Public, Mode::Constant, Mode::Private, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_public_then_public_else_constant() {
        run_test(Mode::Public, Mode::Public, Mode::Constant, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_public_then_public_else_public() {
        run_test(Mode::Public, Mode::Public, Mode::Public, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_public_then_public_else_private() {
        run_test(Mode::Public, Mode::Public, Mode::Private, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_public_then_private_else_constant() {
        run_test(Mode::Public, Mode::Private, Mode::Constant, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_public_then_private_else_public() {
        run_test(Mode::Public, Mode::Private, Mode::Public, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_public_then_private_else_private() {
        run_test(Mode::Public, Mode::Private, Mode::Private, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_constant_else_constant() {
        run_test(Mode::Private, Mode::Constant, Mode::Constant, 0, 0, 0, 0);
    }

    #[test]
    fn test_if_private_then_constant_else_public() {
        run_test(Mode::Private, Mode::Constant, Mode::Public, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_constant_else_private() {
        run_test(Mode::Private, Mode::Constant, Mode::Private, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_public_else_constant() {
        run_test(Mode::Private, Mode::Public, Mode::Constant, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_public_else_public() {
        run_test(Mode::Private, Mode::Public, Mode::Public, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_public_else_private() {
        run_test(Mode::Private, Mode::Public, Mode::Private, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_private_else_constant() {
        run_test(Mode::Private, Mode::Private, Mode::Constant, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_private_else_public() {
        run_test(Mode::Private, Mode::Private, Mode::Public, 0, 0, 3, 3);
    }

    #[test]
    fn test_if_private_then_private_else_private() {
        run_test(Mode::Private, Mode::Private, Mode::Private, 0, 0, 3, 3);
    }
}
