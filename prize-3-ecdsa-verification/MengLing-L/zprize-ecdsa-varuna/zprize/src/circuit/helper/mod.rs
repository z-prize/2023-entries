pub mod range_proof;

#[cfg(test)]
use crate::circuit::NONNATIVE_BASE_MODULUS;
use crate::circuit::{types::field::SMALL_FIELD_SIZE_IN_BITS, RangeProof, B, F};
use crate::circuit::{Env, SMALL_FIELD_BASES, TABLES};
use crate::console::nonnative::basic::NUM_BITS;
use crate::console::{nonnative::basic::K, Console};
use snarkvm_circuit::Environment;

use snarkvm_circuit_environment::ToBits;
use snarkvm_circuit_environment::{Eject, Inject, Mode, Zero};
use snarkvm_console_algorithms::{Field, FromBits, Ordering};
use snarkvm_utilities::ToBits as OtherToBits;

#[cfg(test)]
use snarkvm_circuit_environment::One;

#[cfg(test)]
use crate::console::{nonnative::basic::Polynomial, traits::ToPolynomial};
#[cfg(test)]
use k256::elliptic_curve::Field as SecpField;
#[cfg(test)]
use k256::FieldElement;
#[cfg(test)]
use rand::rngs::OsRng;
#[cfg(test)]
use snarkvm_circuit::Circuit;
#[cfg(test)]
use snarkvm_circuit_environment::{assert_scope, assert_scope_with_lookup};
#[cfg(test)]
use snarkvm_console_algorithms::Itertools;
