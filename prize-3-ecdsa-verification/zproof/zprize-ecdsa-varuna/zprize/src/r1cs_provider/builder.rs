use std::collections::HashMap;
use std::convert::From;
use std::path::Path;
use std::path::PathBuf;

use aleo_std_profiler::{end_timer, start_timer};
use anyhow::Result;
use scopeguard::defer;
use snarkvm_algorithms::r1cs::LookupTable;
use snarkvm_circuit::Circuit as Env;
use snarkvm_circuit::Field;
use snarkvm_circuit_environment::prelude::snarkvm_fields::Fp256;
use snarkvm_circuit_environment::prelude::PrimeField;
use snarkvm_circuit_environment::{Environment as _, Inject as _, LinearCombination, Mode};
use snarkvm_console_network::{Environment, Testnet3};
use snarkvm_curves::bls12_377::FrParameters;
use snarkvm_utilities::BigInteger256;

use super::deserialize;
use super::deserialize::BigInt;

type EF = <Testnet3 as Environment>::Field;
type F = Field<Env>;

impl From<&BigInt> for Fp256<FrParameters> {
    fn from(value: &BigInt) -> Self {
        // Self(BigInteger256(value.0), PhantomData)
        Self::from_bigint(BigInteger256(value.0)).unwrap()
    }
}

pub(crate) fn construct_r1cs_from_file(
    r1cs_file: impl AsRef<Path>,
    assignment_file: impl AsRef<Path>,
    lookup_file: Option<PathBuf>,
) -> Result<()> {
    let construct_time = start_timer!(|| "builder::construct_r1cs_from_file()");
    defer! {
        end_timer!(construct_time);
    }

    let (r1cs, assignment, lookup) =
        deserialize::parse_file(r1cs_file, assignment_file, lookup_file)?;

    let fields = assignment
        .variables
        .iter()
        .enumerate()
        .map(|(id, variable)| {
            if id == 0 {
                // Insert the first element `1`
                F::from(Env::one())
            } else {
                F::new(
                    if id < assignment.num_public_inputs {
                        Mode::Public
                    } else {
                        Mode::Private
                    },
                    snarkvm_console::types::Field::new(EF::from(variable)),
                )
            }
        })
        .collect::<Vec<_>>();

    let func_convert_lc = |lc: &HashMap<usize, BigInt>| -> Result<_> {
        // create Field<Env> from libsnark's linear_combination
        let mut f: Field<Env> = F::from(Env::zero());
        for term in lc {
            let coeff = EF::from(term.1);
            f += &F::from(LinearCombination::from(&fields[*term.0]) * (&coeff));
        }
        Ok(f)
    };

    let (mut count_non_zero_a, mut count_non_zero_b) = (0usize, 0usize);

    /* Count nun zeros on lookup constrains */
    if let Some(lookup) = &lookup {
        /* We only need to count for A, since B and C are all zeros */
        let count_num_zero_lookup_a = lookup
            .constraints
            .iter()
            .fold(0, |acc, constraint| acc + constraint.a.len());
        count_non_zero_a += count_num_zero_lookup_a;
    }

    r1cs.0.iter().try_for_each(|constraint| -> Result<_> {
        let mut a = func_convert_lc(&constraint.a)?;
        let mut b = func_convert_lc(&constraint.b)?;
        let c = func_convert_lc(&constraint.c)?;

        let mut len_a = constraint.a.len();
        let mut len_b = constraint.b.len();

        if (len_a < len_b && count_non_zero_a < count_non_zero_b)
            || (len_a > len_b && count_non_zero_a > count_non_zero_b)
        {
            /* Swap a and b to make non zeros values of A and B more balance. */
            let t = b;
            b = a;
            a = t;

            let t = len_b;
            len_b = len_a;
            len_a = t;
        }

        count_non_zero_a += len_a;
        count_non_zero_b += len_b;

        Env::enforce(|| (a, b, c));
        Ok(())
    })?;

    if let Some(lookup) = lookup {
        let mut table = LookupTable::default();
        lookup.table.0.iter().try_for_each(|item| -> Result<_> {
            let a = EF::from(item[0]);
            let b = EF::from(item[1]);
            let c = EF::from(item[2]);

            table.fill([a, b], c);
            Ok(())
        })?;
        Env::add_lookup_table(table);

        let table_index = 0; /* Currently we only have one table */
        lookup
            .constraints
            .iter()
            .try_for_each(|constraint| -> Result<_> {
                let a = func_convert_lc(&constraint.a)?;
                let b = func_convert_lc(&constraint.b)?;
                let c = func_convert_lc(&constraint.c)?;

                Env::enforce_lookup(|| (a, b, c, table_index));
                Ok(())
            })?;
    }

    Ok(())
}
