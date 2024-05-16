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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsoleNonNativePointMul<N: Network> {
    // pub(crate) scalar: Option<Vec<bool>>,
    pub(crate) point_add: Vec<ConsoleNonNativePointAdd<N>>,
    pub(crate) point_double: Option<Vec<ConsoleNonNativePointDouble<N>>>,
}

impl<N: Network> ConsoleNonNativePointMul<N> {
    /// Create a non-native point multiplication unit from two secp256k1 points output_point, base point P, and scalar,
    ///     where output_point  = scalar \cdot P over secp256k1 curve.
    /// Namely, it includes two main steps:
    /// 1. Precompute and record all intermediate points of point additions and doubling during multiplication.
    /// 2. Construct the corresponding non-nvative point additions
    ///         and non-nvative point doubling for those point.
    ///
    /// Note: If with_scalar = ture, records scalar_bits
    ///       if is_generator_mul = ture, not record intermediate points of point doubling
    ///
    pub fn new(
        scalar: &Scalar,
        base: &ProjectivePoint,
        res: &ProjectivePoint,
        // with_scalar: bool,
        is_generator_mul: bool,
    ) -> Self {
        let mut scalar_bits_le = scalar.to_bytes().to_vec();
        scalar_bits_le.reverse();
        let scalar_bits_le = scalar_bits_le.to_bits_le();

        let mut point_add = Vec::with_capacity(scalar_bits_le.len());
        let mut point_double = Vec::with_capacity(scalar_bits_le.len());

        let (mut temp, mut output) = (*base, ProjectivePoint::IDENTITY);

        scalar_bits_le.iter().for_each(|&bit| {
            let old_output = output;
            if bit {
                output += temp;
                point_add.push(ConsoleNonNativePointAdd::new(&temp, &old_output, &output));
            } else {
                point_add.push(ConsoleNonNativePointAdd::new(
                    &ProjectivePoint::IDENTITY,
                    &old_output,
                    &output,
                ));
            }
            let old_temp = temp;
            temp = temp.double();
            point_double.push(ConsoleNonNativePointDouble::new(&old_temp, &temp));
        });

        assert_eq!(res.to_bytes(), output.to_bytes());

        Self {
            // scalar: with_scalar.then(|| scalar_bits_le),
            point_add,
            point_double: (!is_generator_mul).then(|| point_double),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ITERATIONS: u64 = 20;

    #[test]
    fn test_secp256_point_mul() {
        for _ in 0..ITERATIONS {
            let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
            let public_key = secret_key.verifying_key();

            let affine_point = public_key.as_affine();

            let projective_point: ProjectivePoint = affine_point.into();

            let projective_base: ProjectivePoint = ProjectivePoint::GENERATOR;

            let scalar: Scalar = **secret_key.as_nonzero_scalar();

            let _ = ConsoleNonNativePointMul::<Testnet3>::new(
                &scalar,
                &projective_base,
                &projective_point,
                true,
            );
        }
    }
}
