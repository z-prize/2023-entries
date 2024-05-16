pub mod basic;
pub mod group;

use crate::circuit::Env;
use crate::circuit::{NonNativePolynomial, B, F};
use snarkvm_circuit_environment::{Eject, Environment, Inject, Mode, Zero};

use crate::{
    circuit::traits::*,
    circuit::*,
    console::nonnative::{
        basic::{linear::Signed, Polynomial},
        group::{
            add::ConsoleNonNativePointAdd, double::ConsoleNonNativePointDouble,
            mul::ConsoleNonNativePointMul, ConsoleNonNativePoint,
        },
    },
};
use basic::{linear::CircuitNonNativeLinearComb, mul::CircuitNonNativeMul};
use snarkvm_console_network::{prelude::Itertools, Testnet3};

use crate::circuit::NonNativeMul;
use crate::circuit::{traits::RangeProof, NonNativeLinearCombination, NONNATIVE_BASE_MODULUS};
use crate::console::nonnative::basic::linear::LinearCombination;
use crate::console::nonnative::basic::mul::Mul;
use crate::console::nonnative::basic::{K, NUM_BITS};
use std::cmp::min;

use std::cmp::max;

use crate::circuit::Ternary;
use snarkvm_circuit::Field;
use snarkvm_console_network::Ternary as snarkVMTernary;

#[cfg(test)]
use crate::console::traits::*;
#[cfg(test)]
use k256::{ecdsa::SigningKey, FieldElement};

#[cfg(test)]
use ecdsa::elliptic_curve::Field as secp256Field;
#[cfg(test)]
use rand::rngs::OsRng;
#[cfg(test)]
use snarkvm_circuit_environment::{assert_scope, Circuit};
#[cfg(test)]
use snarkvm_console::types::Field as conField;
