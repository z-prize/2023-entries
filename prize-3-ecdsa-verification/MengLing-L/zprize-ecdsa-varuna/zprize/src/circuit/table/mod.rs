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

use super::types::field::{NUMBER_OF_PADDED_ZERO, SMALL_FIELD_SIZE_IN_BITS};
use super::types::integers::integer::U64;
use super::{F, ROTL_TWO_POW_EXPS};
use crate::circuit::Env;
use crate::console::traits::SmallField;
use crate::console::types::integers::integer::CU64;
use crate::console::{self, CF};
use snarkvm_algorithms::r1cs::LookupTable;
use snarkvm_circuit::prelude::*;

#[derive(Clone)]
pub struct Table {
    tables: Vec<LookupTable<<Env as Environment>::BaseField>>,
}

impl Table {
    /// Performs a lookup for each field of a key in a table and constructs a U64 from the results.
    /// Panics if the lookup fails.
    pub fn rotated_lookup(&self, key: &CU64, rotl: usize) -> Vec<F> {
        ROTL_TWO_POW_EXPS.with(|two_pow_exps| {
            let one = F::one();
            let one_lc = LinearCombination::from(one);

            // Collect the results of the lookups into a vector using iterator transformation.
            let bits_le = key.to_bits_le();

            // Obatin first t bits of input
            let pre_t = bits_le.iter().take(rotl).cloned().collect_vec();

            // Obatin last n-t bits of input
            let last_n_t = bits_le.iter().skip(rotl).cloned().collect_vec();
            // 'flag' whether wo have done the first small block.
            let mut flag = false;

            let results: Vec<F> = pre_t
                .chunks(SMALL_FIELD_SIZE_IN_BITS)
                .chain(last_n_t.chunks(SMALL_FIELD_SIZE_IN_BITS))
                .map(|bits| {
                    let field = CF::from_bits_le(bits).unwrap();
                    let cir_field = F::new(
                        Mode::Private,
                        field.insert_intermidate_zero(NUMBER_OF_PADDED_ZERO),
                    );
                    let field_lc = LinearCombination::from(cir_field.clone());

                    // Perform the table lookup with the current field and `one` as the key.
                    let (_, _, _) = self.tables[0]
                        .lookup(&[field_lc.value(), one_lc.value()])
                        .expect("Lookup failed: value is missing");

                    let bitslen = bits.len();

                    if bitslen == 16 {
                        // check it's a f value from a 16-bit number
                        Env::enforce_lookup(|| (cir_field.clone(), F::one(), cir_field.clone(), 0));
                    } else if flag == false {
                        // first small block
                        let exp =
                            (NUMBER_OF_PADDED_ZERO + 1) * (SMALL_FIELD_SIZE_IN_BITS - bitslen);
                        let mut base = F::zero();
                        if let Some(value) = two_pow_exps.get(&(exp as usize)) {
                            base = value.clone();
                        } else {
                            base = F::constant(console::Console::two_pow_k(exp as u32));
                        }
                        // Check it's a f value from a 16 bit-number even after multiplying by 8^(16-block_length)
                        Env::enforce_lookup(|| {
                            (&base * &cir_field, F::one(), &base * &cir_field, 0)
                        });
                        if rotl > SMALL_FIELD_SIZE_IN_BITS {
                            // Ensuring the it's a f value from a block_length-bit number.
                            Env::enforce_lookup(|| (&cir_field, F::one(), &cir_field, 0));
                        }
                        flag = true;
                    } else {
                        // end block
                        Env::enforce_lookup(|| (cir_field.clone(), F::one(), cir_field.clone(), 0));
                        if rotl < SMALL_FIELD_SIZE_IN_BITS {
                            let exp =
                                (NUMBER_OF_PADDED_ZERO + 1) * (SMALL_FIELD_SIZE_IN_BITS - bitslen);
                            let mut base = F::zero();
                            if let Some(value) = two_pow_exps.get(&(exp as usize)) {
                                base = value.clone();
                            } else {
                                base = F::constant(console::Console::two_pow_k(exp as u32));
                            }
                            // Check it's a f value from a 16 bit-number even after multiplying by 8^(16-block_length)
                            Env::enforce_lookup(|| {
                                (&base * &cir_field, F::one(), &base * &cir_field, 0)
                            });
                        }
                    }

                    cir_field
                })
                .collect();

            // Construct a U64 from the lookup results and return it.
            results
        })
    }

    /// Performs a lookup for each field of a key in a table and constructs a U64 from the results.
    /// Enforce the lookup constriant for four field elements in U64 <f(x) , one> -> f(x)  in table 0
    /// Panics if the lookup fails.
    pub fn lookup(&self, key: &CU64) -> U64 {
        let one = F::one();
        let one_lc = LinearCombination::from(one);

        // Collect the results of the lookups into a vector using iterator transformation.
        let results: Vec<F> = key
            .get()
            .iter()
            .map(|field| {
                let cir_field = F::new(
                    Mode::Private,
                    field.insert_intermidate_zero(NUMBER_OF_PADDED_ZERO),
                );
                let field_lc = LinearCombination::from(cir_field.clone());

                // Perform the table lookup with the current field and `one` as the key.
                let (_, _, _) = self.tables[0]
                    .lookup(&[field_lc.value(), one_lc.value()])
                    .expect("Lookup failed: value is missing");

                Env::enforce_lookup(|| (cir_field.clone(), F::one(), cir_field.clone(), 0));

                cir_field
            })
            .collect();

        // Construct a U64 from the lookup results and return it.
        U64::from_fields(&results)
    }

    /// Enforce the lookup constriant for the field element <f(x) , one> -> f(x)  in table 0
    pub fn lookup_by_field(&self, key: &CF) -> F {
        let one = F::one();
        let one_lc = LinearCombination::from(one);

        let cir_field = F::new(
            Mode::Private,
            key.insert_intermidate_zero(NUMBER_OF_PADDED_ZERO),
        );
        // Collect the results of the lookups into a vector using iterator transformation.
        let field_lc = LinearCombination::from(cir_field.clone());

        // Perform the table lookup with the current field and `one` as the key.
        let (_, _, _) = self.tables[0]
            .lookup(&[field_lc.value(), one_lc.value()])
            .expect("Lookup failed: value is missing");

        Env::enforce_lookup(|| (cir_field.clone(), F::one(), cir_field.clone(), 0));

        cir_field
    }

    ///  enforce the lookup constriant for field element f is in table 1 [-2^16 + 1, 2^16 - 1]
    pub fn range_check_include_negtive(&self, key: &F) {
        let one = F::one();
        let one_lc = LinearCombination::from(one);

        // Collect the results of the lookups into a vector using iterator transformation.
        let field_lc = LinearCombination::from(key.clone());

        // Perform the table lookup with the current field and `one` as the key.
        let (_, _, _) = self.tables[1]
            .lookup(&[field_lc.value(), one_lc.value()])
            .expect("Lookup failed: value is missing");

        Env::enforce_lookup(|| (key.clone(), F::one(), key.clone(), 1));
    }

    ///  enforce the lookup constriant for field element f is in table 3  [0, 2^16 - 1]
    pub fn range_check(&self, key: &F) {
        let one = F::one();
        let one_lc = LinearCombination::from(one);

        // Collect the results of the lookups into a vector using iterator transformation.
        let field_lc = LinearCombination::from(key.clone());

        // Perform the table lookup with the current field and `one` as the key.
        let (_, _, _) = self.tables[1]
            .lookup(&[field_lc.value(), one_lc.value()])
            .expect("Lookup failed: value is missing");

        Env::enforce_lookup(|| (key.clone(), F::one(), key.clone(), 3));
    }

    ///  enforce the lookup constriant for the maping from x to f(x) is in table 2 <x , one> -> f(x)
    pub fn enforce_insert_intermidate_zero_lookup(&self, key: &F) -> F {
        let one = F::one();
        let one_lc = LinearCombination::from(one);

        let key_lc = LinearCombination::from(key.clone());

        // Perform the table lookup with the current field and `one` as the key.
        let (_, _, value) = self.tables[2]
            .lookup(&[key_lc.value(), one_lc.value()])
            .expect("Lookup failed: value is missing");

        let term: F = LinearCombination::from(Env::new_variable(key.eject_mode(), *value)).into();

        Env::enforce_lookup(|| (key.clone(), F::one(), term.clone(), 2));

        term
    }

    ///  enforce the lookup constriant for the maping from x to f(x) is in table 2 <x , one> -> f(x)
    pub fn enforce_insert_intermidate_zero_lookup_by_key_and_value(&self, key: &F, value: &F) {
        let one = F::one();
        let one_lc = LinearCombination::from(one);

        let key_lc = LinearCombination::from(key.clone());
        let value_lc = LinearCombination::from(value.clone());

        // Perform the table lookup with the current field and `one` as the key.
        let (_, _, look_up_value) = self.tables[2]
            .lookup(&[key_lc.value(), one_lc.value()])
            .expect("Lookup failed: value is missing");

        assert_eq!(*look_up_value, value_lc.value());
        Env::enforce_lookup(|| (key.clone(), F::one(), value.clone(), 2));
    }
}

impl Inject for Table {
    type Primitive = console::table::Table<<Env as Environment>::Network>;

    /// Initializes a new instance of a TableExample circuit
    fn new(_mode: Mode, table: Self::Primitive) -> Self {
        let mut tables: Vec<LookupTable<<Env as Environment>::BaseField>> = vec![];
        for console_table in table.tables().iter() {
            let mut lookup_table = LookupTable::default();
            for (key, value) in &console_table.table {
                let cir_keys: Vec<F> = Inject::new(_mode, key.to_vec());
                let cir_value: F = Inject::new(_mode, *value);
                let key_0: LinearCombination<<Env as Environment>::BaseField> =
                    cir_keys[0].clone().into();
                let key_1: LinearCombination<<Env as Environment>::BaseField> =
                    cir_keys[1].clone().into();
                let lookup_value = [key_0.value(), key_1.value()];
                let value: LinearCombination<<Env as Environment>::BaseField> = cir_value.into();
                lookup_table.fill(lookup_value, value.value());
            }
            Env::add_lookup_table(lookup_table.clone());
            tables.push(lookup_table);
        }
        Self { tables }
    }
}

#[cfg(all(test))]
mod tests {
    use crate::{
        circuit::{self, types::field::NUMBER_OF_PADDED_ZERO, B, F},
        console::traits::SmallField as CSmallField,
    };

    use super::*;

    use snarkvm_algorithms::r1cs::SynthesisError;
    use snarkvm_circuit::environment::{assert_scope_with_lookup, Circuit};
    use snarkvm_console_network::Testnet3;
    use snarkvm_utilities::{TestRng, Uniform};

    use anyhow::Result;
    const ITERATIONS: u64 = 100;

    fn get_random_field_from_bits(mode: Mode) -> F {
        let mut rng = TestRng::default();
        let len = 16;
        let bits = (0..len)
            .map(|_| Uniform::rand(&mut rng))
            .collect::<Vec<bool>>();
        let given_bits = bits.iter().map(|b| B::new(mode, *b)).collect_vec();
        F::from_bits_le(&given_bits[..16])
    }

    fn check_verify(
        mode: Mode,
        num_constants: u64,
        num_public: u64,
        num_private: u64,
        num_constraints: u64,
        num_lookup_constraints: u64,
    ) -> Result<()> {
        for _ in 0..ITERATIONS {
            let native_table: console::table::Table<Testnet3> = console::table::Table::setup()?;
            let circuit_table = circuit::table::Table::new(Mode::Constant, native_table);
            let a = get_random_field_from_bits(mode);
            let b = F::one();
            let c = a
                .eject_value()
                .insert_intermidate_zero(NUMBER_OF_PADDED_ZERO);

            Circuit::scope(format!("TableExample {mode}"), || {
                let a: LinearCombination<<Circuit as Environment>::BaseField> = a.clone().into();
                let b: LinearCombination<<Circuit as Environment>::BaseField> = b.clone().into();
                let expected: LinearCombination<<Circuit as Environment>::BaseField> =
                    F::new(mode, c).into();

                let (_, _, candidate) = circuit_table.tables[2]
                    .lookup(&[a.value(), b.value()])
                    .ok_or(SynthesisError::LookupValueMissing)
                    .unwrap();
                assert_scope_with_lookup!(
                    num_constants,
                    num_public,
                    num_private,
                    num_constraints,
                    num_lookup_constraints
                );
                assert_eq!(expected.value(), *candidate);
            });
        }
        Ok(())
    }

    #[test]
    fn test_verify_constant() -> Result<()> {
        check_verify(Mode::Constant, 1, 0, 0, 0, 0)
    }

    #[test]
    fn test_verify_public() -> Result<()> {
        check_verify(Mode::Public, 0, 1, 0, 0, 0)
    }

    #[test]
    fn test_verify_private() -> Result<()> {
        check_verify(Mode::Private, 0, 0, 1, 0, 0)
    }
}
