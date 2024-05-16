pub mod field;
pub mod integers;

use crate::circuit::types::field::*;
use crate::circuit::*;
use crate::console::types::integers::integer::CU64;
use crate::console::CF;
use snarkvm_circuit_environment::Eject;

use snarkvm_circuit_environment::Mode;
use snarkvm_circuit_environment::{Environment, Inject};
use snarkvm_console_algorithms::FromBits;
use snarkvm_console_algorithms::Itertools;
use snarkvm_utilities::ToBits as OtherToBits;

#[cfg(test)]
use snarkvm_circuit_environment::One;
#[cfg(test)]
use snarkvm_circuit_environment::ToBits;
#[cfg(test)]
use snarkvm_utilities::{TestRng, Uniform};
