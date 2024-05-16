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

#[derive(Clone, Debug)]
pub struct U64([F; 4]);

impl U64 {
    pub fn get(&self) -> Vec<F> {
        self.0.to_vec()
    }

    pub fn from_fields(fields: &Vec<F>) -> Self {
        if fields.len() != 4 {
            Env::halt("U64 must be constructed from four fields.")
        }
        Self([
            fields[0].clone(),
            fields[1].clone(),
            fields[2].clone(),
            fields[3].clone(),
        ])
    }

    pub fn zero() -> Self {
        Self([F::zero(), F::zero(), F::zero(), F::zero()])
    }

    /// Given [f1, f2, f3, f4]
    /// Calculate f = f1 + f2 * PADDED_FIELD_BASE + f3 * PADDED_FIELD_BASE^2  + f4 * PADDED_FIELD_BASE^3
    pub fn padded_sum(&self) -> F {
        PADDED_FIELD_BASES.with(|bases| {
            self.0
                .iter()
                .enumerate()
                .fold(F::zero(), |acc, (i, b)| b * &bases[i] + acc)
        })
    }

    ///
    /// In the hash function, it will chunk input into 64 bits, and rotate it left n bits.
    /// We view 64 bits as 4 fields, each with a 16-bit length, instead of viewing them as 64 independent bits.
    ///
    /// However, when rotating n bits, n may be not the multiple of 16
    /// Therefore, when viewing 64 bits as 4 fields, it's hard to rotate.
    ///
    /// We reconstruct 64 bits as five field for easy rotating.
    /// For example , if n = 23
    /// We reconstruct 64 bits as
    /// fields = [f1 (16 bits), f2 (7 bits), f3 (16 bits), f4 (16 bits), f5 (9 bits)]
    /// Now when rotating 23 bits, it equal rotating left t fields where t = n/16 + 1 = 2
    /// rotated_fields = [f3 (16 bits), f4 (16 bits), f5 (9 bits), f1 (16 bits), f2 (7 bits)]
    ///
    ///
    /// In this function
    /// for example, given rotated_fields [f3 (16 bits), f4 (16 bits), f5 (9 bits), f1 (16 bits), f2 (7 bits)]
    ///     where t = n/16 + 1, n is the rotated bits.
    ///
    /// This function calculates and returns
    ///      f = f3 + f4 * 2^16 + f5 * 2^32  + f1 * 2^41 + f2 * 2^57
    pub fn rotated_padded_sum(rotated_value: &[F], n: usize) -> F {
        ROTL_TWO_POW_EXPS.with(|two_pow_exps| {
            // Calculate the number of chunks using ceiling division
            let t = if n % SMALL_FIELD_SIZE_IN_BITS == 0 {
                n / SMALL_FIELD_SIZE_IN_BITS
            } else {
                n / SMALL_FIELD_SIZE_IN_BITS + 1
            };

            let last_n_t = rotated_value.len().saturating_sub(t);
            let mut exp = 0;
            let mut res = F::zero();

            for (i, item) in rotated_value.iter().enumerate() {
                // Calculate the base once for each iteration
                let mut base = F::zero();
                if let Some(value) = two_pow_exps.get(&(exp as usize)) {
                    base = value.clone();
                } else {
                    base = F::constant(console::Console::two_pow_k(exp));
                }

                // Accumulate the result
                res += base * item;

                // In the example,
                // rotated_fields = [f3 (16 bits), f4 (16 bits), f5 (9 bits), f1 (16 bits), f2 (7 bits)]
                // last_n_t = 3
                // the following step is to calculate the bit length (rest_bits) of f5 when index = last_n_t - 1
                if i == last_n_t - 1 && last_n_t > 0 {
                    let rest_bits = PADDED_FIELD_SIZE * 4
                        - (n * (NUMBER_OF_PADDED_ZERO + 1))
                        - i * PADDED_FIELD_SIZE;
                    exp += rest_bits as u32;
                } else if i < rotated_value.len() - 1 {
                    exp += PADDED_FIELD_SIZE as u32;
                }
            }
            res
        })
    }

    ///
    /// In the hash function, it will chunk input into 64 bits, and rotate it left n bits.
    /// We view 64 bits as 4 fields, each with a 16-bit length, instead of viewing them as 64 independent bits.
    ///
    /// However, when rotating n bits, n may not the multiple of 16
    /// Therefore, when view 64 bits as 4 fields, it's hard to rotate.
    ///
    /// We reconstruct 64 bits as five field for easy rotating.
    /// For example , if n = 23
    /// We reconstruct 64 bits as
    /// fields = [f1 (16 bits), f2 (7 bits), f3 (16 bits), f4 (16 bits), f5 (9 bits)]
    ///
    ///
    /// In this function
    /// for example, given fields [f1 (16 bits), f2 (7 bits), f3 (16 bits), f4 (16 bits), f5 (9 bits)]
    ///     where t = n/16 + 1, n is the rotated bits.
    ///
    /// This function calculate and returen
    ///      f = f1 + f2 * 2^16 + f3 * 2^23  + f4 * 2^39 + f5 * 2^55
    pub fn before_rotated_padded_sum(rotated_value: &[F], n: usize) -> F {
        ROTL_TWO_POW_EXPS.with(|two_pow_exps| {
            let t = if n % SMALL_FIELD_SIZE_IN_BITS == 0 {
                n / SMALL_FIELD_SIZE_IN_BITS
            } else {
                n / SMALL_FIELD_SIZE_IN_BITS + 1
            };

            let mut exp = 0;
            let mut res = F::zero();
            for (i, item) in rotated_value.iter().enumerate() {
                // Calculate the base once for each iteration
                let mut base = F::zero();
                if let Some(value) = two_pow_exps.get(&(exp as usize)) {
                    base = value.clone();
                } else {
                    base = F::constant(console::Console::two_pow_k(exp));
                }

                // Accumulate the result
                res += base * item;

                // In the example,
                // fields = [f1 (16 bits), f2 (7 bits), f3 (16 bits), f4 (16 bits), f5 (9 bits)]
                // t = 2
                // the following step is to calculate the bit length (rest_bits) of f2 when index = t - 1
                if t > 0 && i == t - 1 {
                    let rest_bits = (n * (NUMBER_OF_PADDED_ZERO + 1)) - i * PADDED_FIELD_SIZE;
                    exp += rest_bits as u32;
                } else if i < rotated_value.len() - 1 {
                    exp += PADDED_FIELD_SIZE as u32;
                }
            }

            res
        })
    }
}

impl Inject for U64 {
    type Primitive = CU64;

    fn new(_mode: Mode, value: Self::Primitive) -> Self {
        let fields = value
            .get()
            .iter()
            .map(|f| F::new(Mode::Private, *f))
            .collect_vec();
        Self::from_fields(&fields)
    }
}

impl Eject for U64 {
    type Primitive = CU64;

    fn eject_mode(&self) -> Mode {
        self.0[0].eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        CU64::from_fields(&self.0.iter().map(|f| f.eject_value()).collect_vec())
    }
}

#[cfg(test)]
mod tests {

    use snarkvm_circuit_environment::FromBits;
    use snarkvm_console_network::FromBits as OtherFromBits;

    use super::*;

    const MODULO: usize = 5;

    fn get_random_u64_from_bits() -> CU64 {
        let mut rng = TestRng::default();
        let len = 64;
        let bits = (0..len)
            .map(|_| Uniform::rand(&mut rng))
            .collect::<Vec<bool>>();
        let fields = bits
            .chunks(SMALL_FIELD_SIZE_IN_BITS)
            .map(|b| CF::from_bits_le(&b).unwrap())
            .collect_vec();
        CU64::from_fields(&fields)
    }

    fn rotl_offsets() -> [usize; MODULO * MODULO] {
        let mut rotl = [0; MODULO * MODULO];
        let mut x: usize = 1;
        let mut y: usize = 0;

        for t in 0..=23 {
            let rotr = ((t + 1) * (t + 2) / 2) % 64;
            rotl[x + (y * MODULO)] = (64 - rotr) % 64;

            // Update x and y.
            let x_old = x;
            x = y;
            y = (2 * x_old + 3 * y) % MODULO;
        }
        rotl
    }

    #[test]
    fn test_padded_intermediate_values() {
        TABLES.with(|tables| {
            let mut a = Vec::new();
            let mut fothers = Vec::new();
            for _ in 0..5 {
                let fa = tables.lookup(&get_random_u64_from_bits());
                a.push(fa);
            }
            for i in 1..5 {
                fothers.push(a[i].clone());
            }
            let fsum = a[0].padded_sum()
                + &fothers
                    .iter()
                    .fold(F::zero(), |acc, b| b.padded_sum() + acc);
            let (y, w1, w2) = fsum.calculate_padded_intermidate_values();
            let fy = tables.lookup(&y);
            let f_w1 = tables.lookup(&w1);
            let f_w2 = tables.lookup(&w2);
            let two = F::one() + F::one();
            let four = &two + &two;
            assert_eq!(
                fsum.eject_value(),
                (fy.padded_sum() + two * f_w1.padded_sum() + four * f_w2.padded_sum())
                    .eject_value()
            );
        })
    }

    #[test]
    fn test_padded_before_rotated_sum() {
        TABLES.with(|tables| {
            for n in rotl_offsets() {
                let mut a = Vec::new();
                let mut fothers = Vec::new();
                for _ in 0..5 {
                    let fa = tables.lookup(&get_random_u64_from_bits());
                    a.push(fa);
                }
                for i in 1..5 {
                    fothers.push(a[i].clone());
                }
                let fsum = a[0].padded_sum()
                    + &fothers
                        .iter()
                        .fold(F::zero(), |acc, b| b.padded_sum() + acc);
                let (y, _, _) = fsum.calculate_padded_intermidate_values();
                let rotated_fx4 = tables.rotated_lookup(&y, n);
                let fy = tables.lookup(&y);

                assert_eq!(
                    U64::before_rotated_padded_sum(&rotated_fx4, n).eject_value(),
                    fy.padded_sum().eject_value()
                );
            }
        })
    }

    #[test]
    fn test_padded_rotated_sum() {
        TABLES.with(|tables| {
            for n in rotl_offsets() {
                let mut a = Vec::new();
                let mut fothers = Vec::new();
                for _ in 0..5 {
                    let fa = tables.lookup(&get_random_u64_from_bits());
                    a.push(fa);
                }
                for i in 1..5 {
                    fothers.push(a[i].clone());
                }
                let fsum = a[0].padded_sum()
                    + &fothers
                        .iter()
                        .fold(F::zero(), |acc, b| b.padded_sum() + acc);
                let (y, _, _) = fsum.calculate_padded_intermidate_values();

                let mut rotated_fx4 = tables.rotated_lookup(&y, n as usize);
                let fy = tables.lookup(&y);

                let t = if n as usize % SMALL_FIELD_SIZE_IN_BITS == 0 {
                    n as usize / SMALL_FIELD_SIZE_IN_BITS
                } else {
                    n as usize / SMALL_FIELD_SIZE_IN_BITS + 1
                };

                rotated_fx4.rotate_left(t);

                let mut rotated_bits = fy
                    .get()
                    .iter()
                    .flat_map(|f| {
                        let mut bits = f.to_bits_le();
                        bits.truncate(PADDED_FIELD_SIZE);
                        bits
                    })
                    .collect_vec();

                rotated_bits.rotate_left(n as usize * PADDED_CHUNK_SIZE);

                let fy = U64::from_fields(
                    &rotated_bits
                        .chunks(PADDED_FIELD_SIZE)
                        .map(|chunks| F::from_bits_le(chunks))
                        .collect_vec(),
                );

                assert_eq!(
                    U64::rotated_padded_sum(&rotated_fx4, n as usize).eject_value(),
                    fy.padded_sum().eject_value()
                );
            }
        })
    }
}
