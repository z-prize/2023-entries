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

/// Funcitons here for circuit field
/// DONOT create any new variable and increase any new constriant.
///
/// Just used of calculating intermediate values for hash process.
///
impl Intermediate for F {
    type InterOutput = CF;

    type Output = CU64;

    /// Given circuit field f,
    /// This function calculates and return y, w1, w2
    ///     such that f = y + two * w1 + four * w2
    fn calculate_intermediate_values(
        &self,
    ) -> (Self::InterOutput, Self::InterOutput, Self::InterOutput) {
        let mut bits = self.eject_value().to_bits_le();
        bits.truncate(PADDED_FIELD_SIZE);
        let mut y = Vec::with_capacity(SMALL_FIELD_SIZE_IN_BITS);
        let mut w1 = Vec::with_capacity(SMALL_FIELD_SIZE_IN_BITS);
        let mut w2 = Vec::with_capacity(SMALL_FIELD_SIZE_IN_BITS);
        for (index, b) in bits.iter().enumerate() {
            match index % PADDED_CHUNK_SIZE {
                0 => y.push(*b),
                1 => w1.push(*b),
                2 => w2.push(*b),
                _ => {}
            }
        }
        let y_ = CF::from_bits_le(&y).unwrap();
        let w1_ = CF::from_bits_le(&w1).unwrap();
        let w2_ = CF::from_bits_le(&w2).unwrap();
        (y_, w1_, w2_)
    }

    /// Given circuit field f,
    /// This function calculates and return y, w1, w2
    ///     such that f = y + two * w1 + four * w2
    fn calculate_padded_intermidate_values(&self) -> (Self::Output, Self::Output, Self::Output) {
        let mut bits = self.eject_value().to_bits_le();
        bits.truncate(PADDED_FIELD_SIZE * 4);
        let mut y = Vec::with_capacity(SMALL_FIELD_SIZE_IN_BITS * 4);
        let mut w1 = Vec::with_capacity(SMALL_FIELD_SIZE_IN_BITS * 4);
        let mut w2 = Vec::with_capacity(SMALL_FIELD_SIZE_IN_BITS * 4);
        for (index, b) in bits.iter().enumerate() {
            match index % PADDED_CHUNK_SIZE {
                0 => y.push(*b),
                1 => w1.push(*b),
                2 => w2.push(*b),
                _ => {}
            }
        }
        let y_ = CU64::from_fields(
            &y.chunks(SMALL_FIELD_SIZE_IN_BITS)
                .map(|f| CF::from_bits_le(f).unwrap())
                .collect_vec(),
        );
        let w1_ = CU64::from_fields(
            &w1.chunks(SMALL_FIELD_SIZE_IN_BITS)
                .map(|f| CF::from_bits_le(f).unwrap())
                .collect_vec(),
        );
        let w2_ = CU64::from_fields(
            &w2.chunks(SMALL_FIELD_SIZE_IN_BITS)
                .map(|f| CF::from_bits_le(f).unwrap())
                .collect_vec(),
        );
        (y_, w1_, w2_)
    }

    /// Given circuit field f,
    /// This function calculates y, w1, w2, and return y
    ///     such that f = y + two * w1 + four * w2
    fn recover_value_from_f(&self) -> Self {
        let mut bits = self.eject_value().to_bits_le();
        bits.truncate(PADDED_FIELD_SIZE);
        let mut y = Vec::with_capacity(SMALL_FIELD_SIZE_IN_BITS);
        for (index, b) in bits.iter().enumerate() {
            match index % PADDED_CHUNK_SIZE {
                0 => y.push(*b),
                _ => {}
            }
        }
        let y_ = F::new(Mode::Private, CF::from_bits_le(&y).unwrap());
        y_
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     const ITERATIONS: u64 = 100;

//     fn check_from_bits_le(
//         mode: Mode,
//         num_constants: u64,
//         num_public: u64,
//         num_private: u64,
//         num_constraints: u64,
//     ) {
//         for i in 0..ITERATIONS {
//             let mut rng = TestRng::default();
//             let len = 16;
//             let bits = (0..len)
//                 .map(|_| Uniform::rand(&mut rng))
//                 .collect::<Vec<bool>>();
//             let expected_size_in_bits = bits.len();
//             let expected = vmField::from_bits_le(&bits).unwrap();
//             let given_bits = bits.iter().map(|b| Boolean::new(mode, *b)).collect_vec();

//             Circuit::scope(format!("{mode} {i}"), || {
//                 let candidate = F::from_sixteen_bits_le(&given_bits[..16]);
//                 assert_eq!(expected, candidate.eject_value());
//                 assert_eq!(
//                     expected_size_in_bits,
//                     candidate.bits_le.get().expect("Caching failed").len()
//                 );
//                 assert_scope!(num_constants, num_public, num_private, num_constraints);

//                 // Ensure a subsequent call to `to_bits_le` does not incur additional costs.
//                 let candidate_bits = candidate.to_bits_le();
//                 assert_eq!(expected_size_in_bits, candidate_bits.len());
//                 assert_scope!(num_constants, num_public, num_private, num_constraints);
//             });

//             // Add excess zero bits.
//             let candidate = [given_bits, vec![Boolean::new(mode, false); i as usize]].concat();

//             Circuit::scope(&format!("Excess {mode} {i}"), || {
//                 let range = 16 + i as usize;
//                 let candidate = F::from_sixteen_bits_le(&candidate[..range]);
//                 assert_eq!(expected, candidate.eject_value());
//                 assert_eq!(
//                     expected_size_in_bits,
//                     candidate.bits_le.get().expect("Caching failed").len()
//                 );
//                 match mode.is_constant() {
//                     true => assert_scope!(num_constants, num_public, num_private, num_constraints),
//                     // `num_private` gets 1 free excess bit, then is incremented by one for each excess bit.
//                     // `num_constraints` is incremented by one for each excess bit.
//                     false => {
//                         assert_scope!(num_constants, num_public, num_private, num_constraints + i)
//                     }
//                 };
//             });
//         }
//     }

//     #[test]
//     fn test_from_bits_le_constant() {
//         check_from_bits_le(Mode::Constant, 0, 0, 0, 0);
//     }

//     #[test]
//     fn test_from_bits_le_public() {
//         check_from_bits_le(Mode::Public, 0, 0, 0, 0);
//     }

//     #[test]
//     fn test_from_bits_le_private() {
//         check_from_bits_le(Mode::Private, 0, 0, 0, 0);
//     }
// }
