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

/// present non-native linear combination process by native fields polynomial.
/// given a, b ,c, d
/// such that a + b - c = d mod q_256 //over non-native field
/// -> a + b - c = d + \alpha * q_256 //over integer
/// -> a(x) + b(x) - c(x) = d(x) + \alpha(x) * q_256(x) + (2^K - x)e(x) mod native_q //over native field
/// where f(X) = f_1 + f_2 * X + f_3 * X^2 and f = f_1 + f_2 * 2^K + f_3 * (2^K)^2
/// and f_1, f_2 and f_3 \in native field
///
#[derive(Clone, Debug)]
pub struct CircuitNonNativeLinearComb {
    pub(crate) value: Vec<F>,
    pub(crate) alpha: Vec<F>,
    pub(crate) e: Vec<F>,
}

impl Inject for CircuitNonNativeLinearComb {
    type Primitive = LinearCombination<Testnet3>;

    fn new(mode: Mode, console_linear_unit: Self::Primitive) -> Self {
        let cir_value_fields = console_linear_unit
            .value
            .0
            .iter()
            .map(|b| F::new(mode, *b))
            .collect_vec();
        let cir_alpha_fields = console_linear_unit
            .alpha
            .0
            .iter()
            .map(|b| F::new(mode, *b))
            .collect_vec();
        let cir_e_fields = console_linear_unit
            .e
            .0
            .iter()
            .map(|b| F::new(mode, *b))
            .collect_vec();
        Self {
            value: cir_value_fields,
            alpha: cir_alpha_fields,
            e: cir_e_fields,
        }
    }
}

impl Eject for CircuitNonNativeLinearComb {
    type Primitive = LinearCombination<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.value.eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        let console_value_fields = self.value.iter().map(|b| b.eject_value()).collect_vec();
        let console_alpha_fields = self.alpha.iter().map(|b| b.eject_value()).collect_vec();
        let console_e_fields = self.e.iter().map(|b| b.eject_value()).collect_vec();

        LinearCombination {
            value: Polynomial(console_value_fields),
            alpha: Polynomial(console_alpha_fields),
            e: Polynomial(console_e_fields),
        }
    }
}

impl NonNativeLinearCombination for CircuitNonNativeLinearComb {
    type Size = F;
    type Poly = Vec<F>;

    /// a(x) + b(x) - c(x) = d(x) + \alpha(x) * q_256(x) + (2^K - x)e(x) mod native_q //over native field
    ///
    /// a -> presented as [a1, a2, a3]
    /// b -> presented as [b1, b2, b3]
    /// c -> presented as [c1, c2, c3]
    /// d -> presented as [d1, d2, d3]
    /// alpha -> presented as [alpha1, alpha2, alpha3]
    /// q_256 -> presented as [q1, q2, q3]
    /// abc = a + b - c ->
    ///   presented as [ abc_1 = a1 + b1 - c1 + q1, //around 98 bit length
    ///                  abc_2 = a2 + b2 - c2 + q2, //around 98 bit length
    ///                  abc_3 = a3 + b3 - c3 + q3, //around 98 bit length
    ///                ]
    /// e_1 is calculated as [e1, e2, e3]
    /// such that, K = 96 in our setting
    /// abc_1 = k1 + e1 * 2^K , abc_2 = k2 + e2 * 2^K, ...etc
    /// where k1, k2, k3 have K bit length, and e1, e2, e3 have most K + 1 bit length
    ///
    /// alphaq_v = alpha * q_256 + d ->
    ///   presented as [ alphaq_v_1 = alpha1 * q1 + d1, //around 192 bit length
    ///                  alphaq_v_2 = alpha2 * q1 + alpha1 * q2 + d2, //around 193 bit length
    ///                  alphaq_v_3 = alpha3 * q1 + alpha2 * q2 + alpha1* q3 + d3, //around 194 bit length
    ///                  alphaq_v_4 = alpha2 * q3 + alpha3 * q2, //around 193 bit length
    ///                  alphaq_v_5 = alpha3 * q3 ] //around 192 bit length
    /// e_2 is calculated as [e1', e2', e3', e4', e5']
    /// such that, K = 96 in our setting
    /// alphaq_c_1 = k1' + e1' * 2^K , alphaq_c_2 = k2' + e2' * 2^K, ...etc
    /// where k1', k2', k3', k4', k5' have K bit length, and e1', e2', e3', e4', e5' have most K + 2 bit length
    ///
    /// To prove a + b - c = d mod q_256 in native field
    /// This function enforce following constriant:
    /// abc_1 = alphaq_v_1 + e * 2^K where e = e1' - e1
    /// abc_2 = alphaq_v_2 + e' - e * 2^K where e'= e (computed previous step), e = e2' - e2  
    /// abc_3 = alphaq_v_3 + e' - e * 2^K where e'= e (computed previous step), e = e3' - e3  
    /// 0     = alphaq_v_4 + e' - e * 2^K where e'= e (computed previous step), e = e4'  
    /// 0     = alphaq_v_5 + e' - e * 2^K where e'= e (computed previous step), e = e5'
    ///
    /// Besides, we also need to prove that
    ///         alpha1, alpha2, alpha3 are 96, 96, 64 bit length, respectively,
    ///         d1, d2, d3 are 96, 96, 64 bit length, respectively,
    ///         and e1' - e1, e2' - e2, e3' - e3, e4', e5' are 96, 98, 98, 98, 96 bit length, respectively.
    ///
    /// Note: if e = ei' - ei < 0 over integer
    ///       given native modulus q_253
    ///       let e_positive = ei - ei' >= 0,  then q_253 - e_positive = e
    ///       Therefore
    ///       abc_i  = alphaq_v_i + e' - e * 2^K where e'= e{i-1}' - e{i-1}, e = ei' - ei
    ///       = alphaq_v_i + e' - (q_253 - e_positive) * 2^K mod q_253
    ///       = alphaq_v_i + e' - (q_253 - ei + ei') * 2^K mod q_253
    ///       = alphaq_v_i + e' - q_253 * 2^K - (ei' - ei) * 2^K mod q_253
    ///       = alphaq_v_i + e' - (ei' - ei) * 2^K mod q_253
    ///       = alphaq_v_i + e' - e * 2^K mod q_253
    ///       The constriant is correct.
    fn poly_linear_combination_wellform(
        &self,
        chunk_size: &Self::Size,
        terms: &Vec<(Signed, Vec<F>)>,
    ) {
        NONNATIVE_BASE_MODULUS.with(|modulus| {
            self.alpha.range_prove(NUM_BITS);
            self.value.range_prove(NUM_BITS);
            let ab = self.poly_linear_combination(terms);
            let mut alphap_v = modulus.poly_mul(&self.alpha);
            alphap_v = alphap_v.poly_add(&self.value);
            let mut e_hat = F::zero();
            for i in 0..ab.len() {
                let bound_bit_length = if i == 0 { K } else { K + 2 };
                let poly_tmp = ab[i].clone();
                let poly_tmp_c = alphap_v[i].clone();

                self.e[i].range_prove_include_negtive(bound_bit_length);
                Env::assert_eq(&poly_tmp_c + &e_hat - chunk_size * &self.e[i], &poly_tmp);

                e_hat = self.e[i].clone();
            }

            for i in ab.len()..alphap_v.len() {
                let bound_bit_length = if i == 0 { K } else { K + 2 };
                let poly_tmp_c = alphap_v[i].clone();

                self.e[i].range_prove_include_negtive(bound_bit_length);
                Env::assert_eq(&poly_tmp_c + &e_hat - chunk_size * &self.e[i], &F::zero());

                e_hat = self.e[i].clone();
            }
        })
    }

    fn poly_linear_combination(&self, terms: &Vec<(Signed, Vec<F>)>) -> Self::Poly {
        NONNATIVE_BASE_MODULUS.with(|modulus| {
            let result_degree = terms.iter().map(|(_s, b)| b.len()).max().unwrap();
            let mut result: Vec<F> = Vec::with_capacity(result_degree);
            for _ in 0..result_degree {
                result.push(F::zero());
            }
            for (s, item) in terms.iter() {
                for i in 0..item.len() {
                    match s {
                        Signed::ADD => {
                            result[i] += &item[i];
                        }
                        Signed::SUB => {
                            result[i] = &result[i] - &item[i] + &modulus[i];
                        }
                    }
                }
            }
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const ITERATIONS: u64 = 1;
    #[test]
    fn test_secp256_base_field_from_console_mul_unit_circuit_linear_unit() {
        let mut value = FieldElement::from_u64(0);
        let mut terms_field: Vec<(Signed, FieldElement)> = Vec::with_capacity(10);

        for _ in 0..10 {
            let tmp = FieldElement::random(&mut OsRng);
            terms_field.push((Signed::ADD, tmp));
            value += tmp;
        }

        let linear_unit: LinearCombination<Testnet3> =
            LinearCombination::to_linear_combination(&terms_field, &value);

        let cir_linear_unit = CircuitNonNativeLinearComb::new(Mode::Private, linear_unit.clone());
        let recover_console_linear = cir_linear_unit.eject_value();

        assert_eq!(linear_unit.value, recover_console_linear.value);
        assert_eq!(linear_unit.alpha, recover_console_linear.alpha);
        assert_eq!(linear_unit.e, recover_console_linear.e);
    }

    fn check_linear(
        mode: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
    ) {
        for i in 0..ITERATIONS {
            let mut value = FieldElement::from_u64(0);
            let mut terms_field: Vec<(Signed, FieldElement)> = Vec::with_capacity(10);

            for _ in 0..3 {
                let tmp = FieldElement::random(&mut OsRng);
                terms_field.push((Signed::ADD, tmp));
                value += tmp;
            }

            let terms = terms_field
                .iter()
                .map(|(sign, f)| {
                    let temp: Polynomial<Testnet3> = Polynomial::from_nonnative_field(&f);
                    (*sign, temp.0.iter().map(|f| F::new(mode, *f)).collect_vec())
                })
                .collect_vec();

            let linear_unit: LinearCombination<Testnet3> =
                LinearCombination::to_linear_combination(&terms_field, &value);

            let cir_linear_unit = CircuitNonNativeLinearComb::new(mode, linear_unit.clone());

            let chunk_size = F::new(
                Mode::Constant,
                crate::console::Console::<Testnet3>::chunk_size(),
            );
            Circuit::scope(format!("{mode} {i}"), || {
                cir_linear_unit.poly_linear_combination_wellform(&chunk_size, &terms);
                assert_scope!(num_constants, num_public, num_private, num_constraints);
            });
        }
    }

    #[test]
    fn test_linear_constant() {
        check_linear(Mode::Constant, 786509, 0, 0, 0);
    }

    #[test]
    fn test_linear_public() {
        check_linear(Mode::Public, 786443, 66, 0, 25);
    }

    #[test]
    fn test_linear_private() {
        check_linear(Mode::Private, 786443, 0, 66, 25);
    }
}
