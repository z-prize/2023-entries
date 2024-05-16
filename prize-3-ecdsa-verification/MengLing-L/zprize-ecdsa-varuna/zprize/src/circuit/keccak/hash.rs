// Copyright (C) 2024 Mengling LIU, Yang HENG, The Hong Kong Polytechnic University
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

use std::vec;
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use super::*;
use crate::circuit::types::field::SMALL_FIELD_SIZE_IN_BITS;
use crate::circuit::{traits::Hash, B};
use crate::circuit::{Env, Intermediate, Not, F, TABLES};
use crate::console::types::integers::integer::CU64;

impl<const TYPE: u8, const VARIANT: usize> Hash for Keccak<TYPE, VARIANT> {
    type Input = F;
    type BitsInput = B;
    type Output = Vec<B>;

    // Returns the Keccak hash of the given input as bits.
    #[inline]
    fn hash(&self, input: &[Self::Input], rest_bits_input: &[Self::BitsInput]) -> Self::Output {
        // The bitrate `r`.
        // The capacity is twice the digest length (i.e. twice the variant, where the variant is in {224, 256, 384, 512}),
        // and the bit rate is the width (1600 in our case) minus the capacity.
        let bitrate = PERMUTATION_WIDTH - 2 * VARIANT;
        debug_assert!(
            bitrate < PERMUTATION_WIDTH,
            "The bitrate must be less than the permutation width"
        );
        debug_assert!(bitrate % 8 == 0, "The bitrate must be a multiple of 8");
        TABLES.with(|tables| {
            // Ensure the input is not empty.
            if input.is_empty() && rest_bits_input.is_empty() {
                Env::halt("The input to the hash function must not be empty")
            }

            // The root state `s` is defined as `0^b`.
            let mut s = vec![F::zero(); PERMUTATION_WIDTH / SMALL_FIELD_SIZE_IN_BITS];

            // The padded blocks `P`.
            let padded_blocks = Self::pad_keccak(input, rest_bits_input, bitrate);
            let two = F::one() + F::one();

            /* The first part of the sponge construction (the absorbing phase):
             *
             * for i = 0 to |P| − 1 do
             *   s = s ⊕ (P_i || 0^c) # Note: |P_i| + c == b, since |P_i| == r
             *   s = f(s)
             * end for
             */
            for block in padded_blocks {
                // s = s ⊕ (P_i || 0^c)
                for (j, f) in block.into_iter().enumerate() {
                    /* Here, we use the symbol f to represent a function
                     * that pads two zeros in front of each bit of x in binary representation.
                     * For example f(101)=001 000 001.
                     * We employ a variant of the better XOR method to compute ⊕.
                     * When calculating a ⊕ b, we compute f(x) + f(y).
                     * We can observe that the positions in f(x) + f(y)
                     * where the modulo 3 is 0 precisely record
                     * the result of a ⊕ b, while the positions where the modulo 3 is 1
                     * represent the carries, and the positions where the modulo 3 is 2 are always 0.
                     * Therefore, we can use two values, v and w, to respectively capture
                     * the result of ⊕ and the carry values.
                     * Finally, we enforce the constraint f(x) + f(y) = f(v) + 2*f(w)
                     * to express that v and w extract their respective positions.
                     */

                    /* The variable fsum holds the result of adding two f values,
                     * while y and w1 represent the corresponding ⊕ result and carry value, respectively.
                     * It is worth noting that both y and w1 are 16-bit values.
                     * For the f results of 16-bit values, we create a table in advance
                     * to record the mapping between all 16-bit numbers
                     * and their corresponding f values.
                     * We can then use this table to retrieve the corresponding f values.
                     */
                    let fsum = tables.enforce_insert_intermidate_zero_lookup(&f) + &s[j];
                    let (y, w1, _) = fsum.calculate_intermediate_values();
                    s[j] = tables.lookup_by_field(&y);

                    // Constraint: fsum = f(y) + 2*f(w1)
                    Env::assert_eq(
                        fsum,
                        s[j].clone() + two.clone() * tables.lookup_by_field(&w1),
                    );
                }
                // Through the aforementioned process, we can observe that
                // the input s in the permutation_f function is a vector of f values.
                s = Self::permutation_f::<PERMUTATION_WIDTH, NUM_ROUNDS>(
                    s,
                    &self.round_constants,
                    &self.rotl,
                );
            }

            /* The second part of the sponge construction (the squeezing phase):
             *
             * Z = s[0..r-1]
             * while |Z| < d do // d is the digest length
             *   s = f(s)
             *   Z = Z || s[0..r-1]
             * end while
             * return Z[0..d-1]
             */
            // Z = s[0..r-1]
            let mut z = s[..(bitrate / SMALL_FIELD_SIZE_IN_BITS)].to_vec();
            // while |Z| < l do
            while z.len() < VARIANT {
                // s = f(s)
                s = Self::permutation_f::<PERMUTATION_WIDTH, NUM_ROUNDS>(
                    s,
                    &self.round_constants,
                    &self.rotl,
                );
                // Z = Z || s[0..r-1]
                z.extend(s.iter().take(bitrate / SMALL_FIELD_SIZE_IN_BITS).cloned());
            }
            // return Z[0..d-1]
            // recover_value_from_f function turns the f value into the original value.
            z.truncate(VARIANT / SMALL_FIELD_SIZE_IN_BITS);
            z.iter()
                .flat_map(|f| {
                    let remove_zero_f = f.recover_value_from_f();
                    tables.enforce_insert_intermidate_zero_lookup_by_key_and_value(
                        &remove_zero_f,
                        &f,
                    );
                    let mut bits = remove_zero_f.to_bits_le();
                    bits.truncate(SMALL_FIELD_SIZE_IN_BITS);
                    bits
                })
                .collect_vec()
        })
    }
}

impl<const TYPE: u8, const VARIANT: usize> Keccak<TYPE, VARIANT> {
    /*
     * In Keccak, `pad` is a multi-rate padding, defined as `pad(M) = M || 0x01 || 0x00…0x00 || 0x80`,
     * where `M` is the input data, and `0x01 || 0x00…0x00 || 0x80` is the padding.
     * The padding extends the input data to a multiple of the bitrate `r`, defined as `r = b - c`,
     * where `b` is the width of the permutation, and `c` is the capacity.
     */
    fn pad_keccak(input: &[F], bits_input: &[B], bitrate: usize) -> Vec<Vec<F>> {
        debug_assert!(bitrate > 0, "The bitrate must be positive");

        // Resize the input to a multiple of 8.
        let mut padded_input = bits_input.to_vec();
        let padded_lengths =
            (input.len() * 16 + padded_input.len() + 7) / 8 * 8 - (input.len() * 16);
        padded_input.resize(padded_lengths, Boolean::constant(false));

        // Step 1: Append the bit "1" to the message.
        padded_input.push(Boolean::constant(true));

        // Step 2: Append "0" bits until the length of the message is congruent to r-1 mod r.
        while ((input.len() * 16 + padded_input.len()) % bitrate) != (bitrate - 1) {
            padded_input.push(Boolean::constant(false));
        }

        // Step 3: Append the bit "1" to the message.
        padded_input.push(Boolean::constant(true));

        // Step 4: Convert every 16 bits in padded boolean to field
        let padded_fields = padded_input
            .chunks(SMALL_FIELD_SIZE_IN_BITS)
            .map(|b| F::from_bits_le(b))
            .collect_vec();

        // Construct the padded blocks.
        let mut result = Vec::new();
        for block in input
            .to_vec()
            .iter()
            .chain(padded_fields.iter())
            .cloned()
            .collect_vec()
            .chunks(bitrate / SMALL_FIELD_SIZE_IN_BITS)
        {
            result.push(block.to_vec());
        }
        result
    }

    /* The permutation `f` is a function that takes a fixed-length input and produces a fixed-length output,
     * defined as `f = Keccak-f[b]`, where `b := 25 * 2^l` is the width of the permutation,
     * and `l` is the log width of the permutation.
     *
     * The round function `Rnd` is applied `12 + 2l` times, where `l` is the log width of the permutation.
     */
    fn permutation_f<const WIDTH: usize, const NUM_ROUNDS: usize>(
        input: Vec<F>,
        round_constants: &[CU64],
        rotl: &[usize],
    ) -> Vec<F> {
        debug_assert_eq!(
            input.len(),
            WIDTH / SMALL_FIELD_SIZE_IN_BITS,
            "The input vector must have {WIDTH} bits"
        );
        debug_assert_eq!(
            round_constants.len(),
            NUM_ROUNDS,
            "The round constants vector must have {NUM_ROUNDS} elements"
        );
        // A 64-bit value can be divided into four blocks of 16 bits each
        // and represented using a linear combination.
        // Similarly, the f value (192 bits) of a 64-bit value can also be represented
        // using a linear combination of four f values of 16 bits each
        // (each f value being 48 bits).
        // Now we will use vector a to represent the f values of 64-bit values,
        // where each 64-bit value is formed
        // by aggregating four f values together from the input.
        let mut a = input
            .chunks(4)
            .map(|chunks| U64::from_fields(&chunks.to_vec()))
            .collect::<Vec<_>>();
        // Permute the input.
        for round_constant in round_constants.iter().take(NUM_ROUNDS) {
            a = Self::round(a, round_constant, rotl);
        }
        // Return the permuted input.
        a.iter().flat_map(|e| e.get()).collect_vec()
    }

    /// The round function `Rnd` consists of several parts:
    /// Rnd = ι ◦ χ ◦ π ◦ ρ ◦ θ(a)
    /// where `◦` denotes function composition.
    /// Each part operating on a minimum unit of 64 bits.
    /// In our implementation, we use the defined type U64
    /// to represent a 64-bit number.
    /// Since the f transformation (i.e., padding 2 zeros in front of each bit)
    /// is reversible, all variables in our constraints are f values of 64-bit numbers.
    /// U64bit contains four f values of 16-bit numbers.
    /// As mentioned earlier, the f value of the corresponding 64-bit number can be
    /// computed through a linear combination. The CU64 data type
    /// used in the round function is defined for convenient computation.
    /// It represents a number of 64 bits, calculated by
    /// performing linear combinations on the four 16-bit numbers storing inside it.
    /// We define CU64 to easily obtain the actual variables
    /// (that is, the f value, by simply padding zeros) needed
    /// when establishing constraints. This process does not actually increase variables
    /// or constraints but essentially allows the prover to compute
    /// the corresponding circuit witness.
    /// Additionally, since the values in U64 are all f values of 16-bit numbers,
    /// we need to ensure that each f value in a new U64 comes from a 16-bit number.
    /// Therefore, we introduce a lookup table T that records the f values of
    /// all integers in the range [0, 2^16-1]. When creating a new U64,
    /// each newly generated f value can be obtained by looking up the new table T
    /// to ensure that the value is within the appropriate range.
    fn round(a: Vec<U64>, round_constant: &CU64, rotl: &[usize]) -> Vec<U64> {
        debug_assert_eq!(
            a.len(),
            MODULO * MODULO,
            "The input vector 'a' must have {} elements",
            MODULO * MODULO
        );
        /* The total number of operation bits is 1600,
         * which is divided into 5 * 5 * 64 (5 is the value of MODULE in the input).
         * This means there are a total of 25 blocks of 64 bits each.
         * The positions of these bit blocks are typically
         * described using (x, y) coordinates,
         * where indexing with (x, y) will locate the (x + y * 5)th block. */
        TABLES.with(|tables| {
            let two = F::one() + F::one();
            let four = two.clone() + two.clone();

            // The first algorithm is θ, consisting 2 parts,
            /* the first part is:
             *
             * for x = 0 to 4 do
             *   C[x] = a[x, 0]
             *   for y = 1 to 4 do
             *     C[x] = C[x] ⊕ a[x, y]
             *   end for
             * end for
             */

            /* Here a[x, y] is the original 64-bit operating unit with index (x,y).
             * C[x] is the xor sum of a[x, 0],a[x, 1],a[x, 2],a[x, 3] and a[x, 4].
             * We can use their f value to build a constraint:
             * f(C[x])+2*f(w1)+4*f(w2) = f(a[x,0])+f(a[x,1])+f(a[x,2])+f(a[x,3])+f(a[x,4])
             * In binary, the result of the XOR operation, denoted as C[x],
             * is preserved in the bits where the modulo 3 is 0
             * on the right side of the constraint. We can extract it
             * using a similar method as before, using f(C[x]).
             * The variables w1 and w2 in this case are both 64-bit numbers,
             * corresponding to the carries where the modulo 3 is 1 and 2, respectively,
             * on the right side of the constraint.
             * Therefore, we can extract the value where the modulo 3 is 1 using 2 * f(w1),
             * and extract the value where the modulo 3 is 2 using 4 * f(w2).
             */

            let mut c = Vec::with_capacity(MODULO);
            for x in 0..MODULO {
                let fothers = vec![
                    a[x + MODULO].clone(),
                    a[x + (2 * MODULO)].clone(),
                    a[x + (3 * MODULO)].clone(),
                    a[x + (4 * MODULO)].clone(),
                ];
                // fsum here means f(a[x,0])+f(a[x,1])+f(a[x,2])+f(a[x,3])+f(a[x,4])
                let fsum = a[x].padded_sum()
                    + &fothers
                        .iter()
                        .fold(F::zero(), |acc, b| b.padded_sum() + acc);

                // Here xor is C[x]
                // rotated_fx4 contains 5 small f values to express rotation,
                // and will be explained latter.
                let (xor, w1, w2) = fsum.calculate_padded_intermidate_values();
                let rotated_fx4 = tables.rotated_lookup(&xor, 63);
                let fw_1 = tables.lookup(&w1);
                let fw_2 = tables.lookup(&w2);
                c.push(rotated_fx4.clone());
                // U64::before_rotated_padded_sum(&rotated_fx4, 63) is the actual f(C[x]).
                // The constraint below is fsum = f(C[x])+2*f(w1)+4*f(w2),
                // which is the same as we prescribed above.
                Env::assert_eq(
                    fsum,
                    U64::before_rotated_padded_sum(&rotated_fx4, 63)
                        + &two * fw_1.padded_sum()
                        + &four * fw_2.padded_sum(),
                );
            }

            /* The second part of algorithm θ is:
             *
             * for x = 0 to 4 do
             *   D[x] = C[x−1] ⊕ ROT(C[x+1],1)
             *   for y = 0 to 4 do
             *     A[x, y] = a[x, y] ⊕ D[x]
             *   end for
             * end for
             */

            /*
             * It is worth noting that we are now implementing
             * a new operation called ROT, which stands for rotation.
             * When we apply ROT to the original 64-bit value x,
             * if we need to rotate by a multiple of 16 bits, then
             * for the new value x', when representing f values using U64,
             * the internal representation of the four 16-bit f values
             * within f(x) and f(x') remains the same.
             * The only difference is the way they are linearly combined.
             * In our implementation, we can think of it as a change
             * in the positions of these four smaller f values.
             * However, when the rotation amount for ROT is not a multiple of 16,
             * we cannot use the previous method. To avoid adding
             * too many constraints and variables, our approach is to
             * represent the object of ROT using five smaller f values.
             * Among these five f values, three f values represent
             * consecutive 16-bit numbers, and the other two represent
             * two numbers that are less than 16 bits. Together, these two numbers
             * add up to 16 bits. For example, when the rotation amount is +1,
             * which means rotating by 63 bits, we divide the 64-bit value
             * into five blocks from the least significant bit to the most significant bit:
             * 16 bit, 16 bits, 16 bits, 15 bits, and 1 bits.
             * When rotating by 45 bits, it would be divided into five blocks:
             * 16 bits, 16 bits, 13 bits, 16 bits, and 3 bits. When dividing into
             * five blocks, we can represent the f values before and after
             * the ROT operation using different linear combinations
             * (which is by exchanging the order of the five blocks in our implementation).
             * It is important to note that these five smaller blocks are also obtained
             * through lookup constraints to obtain their corresponding f values.
             * For the 16-bit blocks, it needs to be proven that their corresponding f values
             * exist in table T. For the smaller block with less than 16 bits (e.g., k bits)
             * at the front position, it needs to be proven that
             * the product of its f value and 2^(3*(16-k)) exists in table T.
             * As for the smaller block with less than 16 bits at the end (with (16-k) bits),
             * it needs to be proven that its corresponding f value exists in table T. 
             * If the rotation amount is less than to 16, additionally, 
             * it needs to be proven that the product of the end block's f value
             *  and 2^(3k) exists in table T; Otherwise, it needs to be proven that 
             * first small block's f value exists in table T.
             */

            // Now we can explain how we represent f(C[x]) here.
            // Since we need to calculate ROT(C[x+1],1),
            // we use the above method to respectively
            // represent f(C[x]) and f(ROT(C[x+1],1)).
            // The function before_rotated_padded_sum(&x, y) can be used to
            // get the f value of a 64bit number before rotating by y.
            // The rotated f value directly comes from the function rotate_left.
            // Now we see that the rotated f value f(ROT(C[x+1],1)) is now stored
            // in the mutable vector c.

            // Vector d stores the value of f(C[x]) + f(ROT(C[x+1], 1)).
            let mut d = Vec::with_capacity(MODULO);
            for x in 0..MODULO {
                let rot = Self::rotate_left(&c[(x + 1) % MODULO], 63);
                let fsum = U64::before_rotated_padded_sum(&c[(x + 4) % MODULO], 63)
                    + U64::rotated_padded_sum(&rot, 63);
                d.push(fsum);
            }

            let mut f_rot_a_1: Vec<Vec<F>> = Vec::with_capacity(MODULO * MODULO);
            for y in 0..MODULO {
                for x in 0..MODULO {
                    // Here fsum is f(a[x,y]) + f(D[x]) = f(a[x,y]) + f(C[x]) + f(ROT(C[x+1], 1))
                    let fsum = a[x + (y * MODULO)].padded_sum() + d[x].clone();
                    // xor is the fanal A[x,y].
                    let (xor, w1, _) = fsum.calculate_padded_intermidate_values();
                    // rotated_fx_xor consists 5 small f values to express f(A[x,y]) in rotation,
                    // since there are ROT operations used in the next algorithms.
                    let rotated_fx_xor = tables.rotated_lookup(&xor, rotl[x + (y * MODULO)]);
                    let fw_1 = tables.lookup(&w1);

                    f_rot_a_1.push(rotated_fx_xor.to_vec());
                    // fsum = f(A[x,y]) + 2 * f(w1)
                    // Since fsum is the sum of three f values, the carry will
                    // only propagate to the bits where the modulo 3 is equal to 1.
                    // In this case, we can use w1 to represent the carries
                    // in those positions.
                    Env::assert_eq(
                        fsum,
                        U64::before_rotated_padded_sum(&rotated_fx_xor, rotl[x + (y * MODULO)])
                            + &two * fw_1.padded_sum(),
                    );
                }
            }

            /* Algorithm 3, π:
             *
             * for x = 0 to 4 do
             *   for y = 0 to 4 do
             *     (X, Y) = (y, (2*x + 3*y) mod 5)
             *     A[X, Y] = a[x, y]
             *   end for
             * end for
             *
             * Algorithm 2, ρ:
             *
             * A[0, 0] = a[0, 0]
             * (x, y) = (1, 0)
             * for t = 0 to 23 do
             *   A[x, y] = ROT(a[x, y], (t + 1)(t + 2)/2)
             *   (x, y) = (y, (2*x + 3*y) mod 5)
             * end for
             */

            /* Algorithm 3 involves rearranging the current 25 f values of 64-bit numbers.
             * On the other hand, the results of Algorithm 2 can be directly extracted
             * using the rotate_left function.
             */

            let mut rotl_1 = rotl.to_vec().clone();
            let mut a_2_temp = f_rot_a_1.clone();
            for y in 0..MODULO {
                for x in 0..MODULO {
                    rotl_1[y + ((((2 * x) + (3 * y)) % MODULO) * MODULO)] = rotl[x + (y * MODULO)];
                    a_2_temp[y + ((((2 * x) + (3 * y)) % MODULO) * MODULO)] =
                        Self::rotate_left(&f_rot_a_1[x + (y * MODULO)], rotl[x + (y * MODULO)]);
                }
            }

            /* Algorithm 4, χ:
             *
             * for y = 0 to 4 do
             *   for x = 0 to 4 do
             *     A[x, y] = a[x, y] ⊕ ((¬a[x+1, y]) ∧ a[x+2, y])
             *   end for
             * end for
             */

            /* Given a 64 bit value x we have f(¬x) = f(2^64-1) - f(x).
             * Given 2 numbers x and y, we have f(x) + f(y) = 2 * f(x∧y) + f(x⊕y)
             * So we now build the constraint like this:
             * f(w0) + 2*f(A[x,y]) + 4*f(w2) = 2*f(a[x,y]) + f(¬a[x+1, y]) + f(a[x+2, y])
             * Since we need to compute the bitwise AND of one value and the XOR of another value,
             * we can simplify the process by multiplying the other value by 2.
             * This way, we can directly obtain the result using the bits on the right-hand side
             * of the constraint where the modulo 3 is equal to 1.
             * Then, we can use w0 and w1 to extract the values
             * from the bits where the modulo 3 is equal to 0 and 2, respectively.
             */
            let mut f_a_3 = Vec::with_capacity(MODULO * MODULO);
            for y in 0..MODULO {
                for x in 0..MODULO {
                    let a = &a_2_temp[x + (y * MODULO)];
                    let b = &a_2_temp[((x + 1) % MODULO) + (y * MODULO)];
                    let c = &a_2_temp[((x + 2) % MODULO) + (y * MODULO)];
                    let f_b_value =
                        U64::rotated_padded_sum(b, rotl_1[((x + 1) % MODULO) + (y * MODULO)]);
                    let f_not_b = f_b_value.not();

                    // fsum = 2*f(a[x,y]) + f(¬a[x+1, y]) + f(a[x+2, y]).
                    let mut fsum = &two * U64::rotated_padded_sum(&a, rotl_1[x + (y * MODULO)])
                        + f_not_b
                        + U64::rotated_padded_sum(&c, rotl_1[((x + 2) % MODULO) + (y * MODULO)]);

                    if x == 0 && y == 0 {
                        /* Algorithm 5, ι:
                         *
                         * A[0, 0] = A[0, 0] ⊕ RC
                         */
                        // In Algorithm 5, we can extract RC and add 2f(RC) to fsum.
                        // This way, we can still obtain the result from the bits in fsum
                        // where the modulo 3 is equal to 1.
                        fsum += &two * tables.lookup(&round_constant).padded_sum();
                    }

                    let (w0, val, w2) = fsum.calculate_padded_intermidate_values();
                    let fy = tables.lookup(&val);
                    let fw_0 = tables.lookup(&w0);
                    let fw_2 = tables.lookup(&w2);
                    // Enforce constraint: fsum = f(w0) + 2*f(A[x,y]) + 4*f(w2)
                    Env::assert_eq(
                        fsum,
                        fw_0.padded_sum() + &two * fy.padded_sum() + &four * fw_2.padded_sum(),
                    );
                    f_a_3.push(fy);
                }
            }

            f_a_3
        })
    }

    fn rotate_left(value: &[F], n: usize) -> Vec<F> {
        let mut rotated_value = value.to_vec().clone();

        let t = if n % SMALL_FIELD_SIZE_IN_BITS == 0 {
            n / SMALL_FIELD_SIZE_IN_BITS
        } else {
            n / SMALL_FIELD_SIZE_IN_BITS + 1
        };

        rotated_value.rotate_left(t);

        rotated_value
    }
}

#[cfg(all(test))]
mod tests {
    use crate::console;

    use super::*;
    use snarkvm_circuit::environment::Circuit;
    use snarkvm_circuit::Environment;
    use snarkvm_circuit_environment::{assert_scope, assert_scope_with_lookup};
    use snarkvm_console_network::prelude::*;

    const ITERATIONS: usize = 1;

    macro_rules! check_equivalence {
        ($console:expr, $circuit:expr) => {
            use snarkvm_console::prelude::Hash as H;

            let rng = &mut TestRng::default();

            let mut input_sizes = vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024,
            ];
            input_sizes.extend((0..5).map(|_| rng.gen_range(1..1024)));

            for num_inputs in input_sizes {
                println!("Checking equivalence for {num_inputs} inputs");

                // Prepare the preimage.
                let native_input = (0..num_inputs)
                    .map(|_| Uniform::rand(rng))
                    .collect::<Vec<bool>>();

                // Convert into circuit type
                let mut input = Vec::new();
                let mut rest_bits = Vec::new();
                for chunks in native_input.chunks(SMALL_FIELD_SIZE_IN_BITS) {
                    if chunks.len() == SMALL_FIELD_SIZE_IN_BITS {
                        input.push(F::new(Mode::Private, CF::from_bits_le(chunks).unwrap()));
                    } else {
                        for b in chunks.iter() {
                            rest_bits.push(B::new(Mode::Private, *b));
                        }
                    }
                }

                // Compute the console hash.
                let expected = $console
                    .hash(&native_input)
                    .expect("Failed to hash console input");

                // Compute the circuit hash.
                let candidate = $circuit.hash(&input, &rest_bits);
                assert_eq!(expected, candidate.eject_value());
                Circuit::reset();
            }
        };
    }

    fn check_hash(
        mode: Mode,
        num_inputs: usize,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
        num_lookup_constraints: u64,
        rng: &mut TestRng,
    ) {
        use snarkvm_console::prelude::Hash as H;

        let native = console::keccak::Keccak256::default();
        let keccak = Keccak256::new();

        for i in 0..ITERATIONS {
            // Prepare the preimage.
            let native_input = (0..num_inputs)
                .map(|_| Uniform::rand(rng))
                .collect::<Vec<bool>>();

            // Convert into circuit type
            let mut input = Vec::new();
            let mut rest_bits = Vec::new();
            for chunks in native_input.chunks(SMALL_FIELD_SIZE_IN_BITS) {
                if chunks.len() == SMALL_FIELD_SIZE_IN_BITS {
                    input.push(F::new(mode, CF::from_bits_le(chunks).unwrap()));
                } else {
                    for b in chunks.iter() {
                        rest_bits.push(B::new(mode, *b));
                    }
                }
            }

            // Compute the native hash.
            let expected = native
                .hash(&native_input)
                .expect("Failed to hash native input");

            // Compute the circuit hash.
            Circuit::scope(format!("Keccak {mode} {i}"), || {
                let candidate = keccak.hash(&input, &rest_bits);
                assert_eq!(expected, candidate.eject_value());
                let case = format!("(mode = {mode}, num_inputs = {num_inputs})");
                assert_scope!(
                    case,
                    num_constants,
                    num_public,
                    num_private,
                    num_constraints
                );
                assert_scope_with_lookup!(
                    num_constants,
                    num_public,
                    num_private,
                    num_constraints,
                    num_lookup_constraints
                );
            });
            Circuit::reset();
        }
    }

    #[test]
    fn test_keccak_256_hash_constant() {   
        let mut rng = TestRng::default();

        check_hash(Mode::Constant, 1, 985146, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 2, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 3, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 4, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 5, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 6, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 7, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 8, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 16, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 32, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 64, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 128, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 256, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 511, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 512, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 513, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 1023, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 1024, 1604, 0, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Constant, 1025, 1604, 0, 65160, 13460, 59932, &mut rng);
    }

    #[test]
    fn test_keccak_256_hash_public() {
        let mut rng = TestRng::default();

        check_hash(Mode::Public, 1, 985145, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 2, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 3, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 4, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 5, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 6, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 7, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 8, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 16, 1603, 1, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 32, 1602, 2, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 64, 1600, 4, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 128, 1596, 8, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 256, 1588, 16, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 512, 1572, 32, 65160, 13460, 59932, &mut rng);
        check_hash(Mode::Public, 1024, 1540, 64, 65160, 13460, 59932, &mut rng);
    }

    #[test]
    fn test_keccak_256_hash_private() {
        let mut rng = TestRng::default();
        // check_hash(
        //     Mode::Private,
        //     400000,
        //     1126030,
        //     0,
        //     5363216,
        //     522856,
        //     5613376,
        //     &mut rng,
        // );
        check_hash(Mode::Private, 1, 985145, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 2, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 3, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 4, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 5, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 6, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 7, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 8, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 16, 1603, 0, 65161, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 32, 1602, 0, 65162, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 64, 1600, 0, 65164, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 128, 1596, 0, 65168, 13460, 59932, &mut rng);
        check_hash(Mode::Private, 256, 1588, 0, 65176, 13460, 59932,&mut rng);
        check_hash(Mode::Private, 512, 1572, 0, 65192, 13460, 59932,&mut rng);
        check_hash(Mode::Private, 1024, 1540, 0, 65224, 13460, 59932,&mut rng);
    }

    #[test]
    fn test_keccak_256_equivalence() {
        check_equivalence!(console::keccak::Keccak256::default(), Keccak256::new());
    }
}
