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
/// e.g
/// a + b - c = d mod q_256 //over non-native field
/// -> a + b - c = d + \alpha * q_256 //over integer
/// -> a(x) + b(x) - c(x) = d(x) + \alpha(x) * q_256(x) + (2^K - x)e(x) mod native_q //over native field
/// where f(X) = f_1 + f_2 * X + f_3 * X^2 and f = f_1 + f_2 * 2^K + f_3 * (2^K)^2
/// and f_1, f_2 and f_3 \in native field
///

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum Signed {
    ADD,
    SUB,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearCombination<N: Network> {
    pub(crate) value: Polynomial<N>,
    pub(crate) alpha: Polynomial<N>,
    pub(crate) e: Polynomial<N>,
}

impl<N: Network> ToLinearCombination for LinearCombination<N> {
    type Input = FieldElement;

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
    fn to_linear_combination(
        field_terms: &[(Signed, Self::Input)],
        field_value: &Self::Input,
    ) -> Self {
        let check_sum = field_terms
            .iter()
            .fold(FieldElement::ZERO, |acc, (s, x)| match s {
                Signed::ADD => acc + x,
                Signed::SUB => acc - x,
            });
        assert_eq!(check_sum.normalize(), (*field_value).normalize());
        let q_ = BigInt::from_bytes_le(Sign::Plus, &*BASE_MODULUS);
        let value_ = BigInt::from_bytes_be(Sign::Plus, &field_value.to_bytes());
        let sum_ = field_terms
            .iter()
            .fold(BigInt::from_u64(0).unwrap(), |acc, (s, x)| match s {
                Signed::ADD => acc + BigInt::from_bytes_be(Sign::Plus, &x.to_bytes()),
                Signed::SUB => acc - BigInt::from_bytes_be(Sign::Plus, &x.to_bytes()) + &q_,
            });

        assert_eq!(&sum_ % &q_, value_);

        let alpha_ = &sum_ / &q_;

        assert_eq!(sum_, &value_ + &alpha_ * &q_);
        let terms = field_terms
            .iter()
            .map(|(s, x)| (*s, Polynomial::from_nonnative_field(x)))
            .collect();
        let value: Polynomial<N> = Polynomial::from_nonnative_field(field_value);
        let q: Polynomial<N> = Polynomial::from_bigint(&q_);
        let alpha: Polynomial<N> = Polynomial::from_bigint(&alpha_);

        let poly_sum = Polynomial::raw_linear_combination(&terms, &q);
        let e_1: Polynomial<N> = poly_sum.calculate_e();

        let alphap_v = alpha.raw_mul(&q).raw_add(&value);
        let e_2: Polynomial<N> = alphap_v.calculate_e();

        let mut e_1_vec = e_1.0;
        while e_1_vec.len() < e_2.0.len() {
            e_1_vec.push(Field::from_u8(0));
        }
        let e: Vec<Field<N>> = e_2
            .0
            .iter()
            .zip(e_1_vec.iter())
            .map(|(e2, e1)| *e2 - *e1)
            .collect_vec();
        LinearCombination {
            value,
            alpha,
            e: Polynomial(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use ecdsa::elliptic_curve::Field as BaseField;

    use super::*;

    #[test]
    fn test_non_native_subadd_linear_combination_base_field() {
        let mut value = FieldElement::from_u64(0);
        let mut field_terms: Vec<(Signed, FieldElement)> = Vec::with_capacity(0);
        for _ in 0..10 {
            let tmp: FieldElement = FieldElement::random(&mut OsRng);
            field_terms.push((Signed::SUB, tmp));
            value -= tmp;
        }

        for _ in 0..10 {
            let tmp: FieldElement = FieldElement::random(&mut OsRng);
            field_terms.push((Signed::ADD, tmp));
            value += tmp;
        }

        let linear: LinearCombination<Testnet3> =
            LinearCombination::to_linear_combination(&field_terms, &value.normalize());

        let poly_terms = field_terms
            .iter()
            .map(|(signed, term)| (*signed, Polynomial::from_nonnative_field(term)))
            .collect_vec();

        let poly_sum = Polynomial::raw_linear_combination(
            &poly_terms,
            &console::Console::nonnative_base_modulus(),
        );
        let e_1: Polynomial<Testnet3> = poly_sum.calculate_e();
        let alphap_v = linear
            .alpha
            .raw_mul(&console::Console::nonnative_base_modulus())
            .raw_add(&linear.value);
        let e_2: Polynomial<Testnet3> = alphap_v.calculate_e();

        let k_field: Field<Testnet3> = Field::from_u8(K as u8);
        let two: Field<Testnet3> = Field::from_u8(2);
        let two_k = two.pow(k_field);

        for i in 0..poly_sum.0.len() {
            let mut poly_tmp = poly_sum.0[i];
            let mut poly_tmp_c = alphap_v.0[i];
            if i > 0 {
                poly_tmp = poly_tmp + e_1.0[i - 1];
                poly_tmp_c = poly_tmp_c + e_2.0[i - 1];
            }
            let mut ab_ = poly_tmp.to_bits_le();
            ab_.truncate(K);
            let ab_mod: Field<Testnet3> = Field::from_bits_le(&ab_).unwrap();

            let mut alphap_c_ = poly_tmp_c.to_bits_le();
            alphap_c_.truncate(K);
            let alphap_c_mod: Field<Testnet3> = Field::from_bits_le(&alphap_c_).unwrap();

            assert_eq!(poly_tmp, two_k * &e_1.0[i] + ab_mod);
            assert_eq!(alphap_c_mod, poly_tmp - two_k * &e_1.0[i]);

            let e = e_1.0[i] - e_2.0[i];

            assert_eq!(poly_tmp_c + two_k * e, poly_tmp);
        }
        assert_eq!(e_1.0[poly_sum.0.len() - 1], e_2.0[poly_sum.0.len() - 1]);
    }
}
