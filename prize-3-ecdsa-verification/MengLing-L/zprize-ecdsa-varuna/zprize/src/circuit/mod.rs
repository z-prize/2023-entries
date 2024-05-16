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

use self::{
    keccak::Keccak256,
    nonnative::{
        basic::mul::CircuitNonNativeMul,
        group::{
            add::CircuitNonNativePointAdd, double::CircuitNonNativePointDouble,
            mul::CircuitNonNativePointMul, *,
        },
    },
    table::Table,
};
use crate::console::nonnative::basic::Polynomial;
use crate::console::table::Table as ConsoleTable;
use crate::console::{self, Console};
use indexmap::IndexMap;
use snarkvm_circuit::{Boolean, Circuit as Env, Field};
use snarkvm_circuit_environment::{Environment, Inject, Mode, Zero};
use snarkvm_console_algorithms::Itertools;

pub mod helper;
pub mod keccak;
pub mod message;
pub mod nonnative;
pub mod pubkey;
pub mod signature;
pub mod table;
pub mod traits;
pub mod types;

pub use traits::*;

pub type F = Field<Env>;

pub type B = Boolean<Env>;

thread_local! {
    static NONNATIVE_GENERATOR_G: Vec<CircuitNonNativePointDouble> = Vec::constant(Console::new_bases().to_vec());
    static NONNATIVE_ZERO: Vec<F> = Vec::constant(Console::zero().0);
    static NONNATIVE_TWO: Vec<F> = Vec::constant(Console::nonnative_two().0);
    static NONNATIVE_THREE: Vec<F> = Vec::constant(Console::nonnative_three().0);
    static NONNATIVE_SEVEN: Vec<F> = Vec::constant(Console::nonnative_seven().0);
    static NONNATIVE_SCALAR_FIELD_IDENTITY: Vec<F> = Vec::constant(Console::nonnative_scalar_field_identity().0);
    static NONNATIVE_BASE_MODULUS: Vec<F> = Vec::constant(Console::nonnative_base_modulus().0);
    static NONNATIVE_SCALAR_MODULUS: Vec<F> = Vec::constant(Console::nonnative_scalar_modulus().0);
    static NONNATIVE_BASE_MODULUS_IN_BITS: Vec<B> = Vec::constant(Console::<<Env as Environment>::Network>::nonnative_base_modulus_minus_one_in_bits());
    static NONNATIVE_IDENTITY: CircuitNonNativePoint = CircuitNonNativePoint::constant(Console::nonnative_point_identity());
    static NONNATIVE_GENERATOR: CircuitNonNativePoint = CircuitNonNativePoint::constant(Console::nonnative_point_generator());
    static SMALL_FIELD_BASES: Vec<F> = Vec::constant(Console::small_field_bases());
    static PADDED_FIELD_BASES: Vec<F> = Vec::constant(Console::padded_field_bases());
    static TABLES: Table = Table::new(Mode::Constant, ConsoleTable::setup().unwrap());
    static ROTL_TWO_POW_EXPS: IndexMap<usize, F> = Console::<<Env as Environment>::Network>::rotl_two_pow_exps();
    static PADDED_ONE: F = F::constant(Console::padded_one());
    static CHUNK_SIZE: F = F::constant(Console::chunk_size());
}

pub struct ECDSASignature {
    // signature.s
    pub s: Vec<F>,
    // signature.s^-1
    pub sinv: Vec<F>,
    // signature.r
    pub r: Vec<F>,
    // s^-1 * Hash(message)
    pub sinv_mul_hm: CircuitNonNativeMul,
    // s^-1 * r
    pub sinv_mul_r: CircuitNonNativeMul,
    // s^-1 * s
    pub sinv_mul_s: CircuitNonNativeMul,
    // s^-1 * Hash(message) \cdot P (Generator)
    pub sinv_mul_hm_p: CircuitNonNativePointMul,
    // s^-1 * r\cdot PubKey
    pub sinv_mul_r_pub_key: CircuitNonNativePointMul,
    // Point hm_mul_p + r_mul_pub_key
    pub sinv_mul_hm_p_add_sinv_mul_r_pub_key: CircuitNonNativePointAdd,
}

pub struct ECDSAPubKey {
    pub public_key: CircuitNonNativePoint,
}

pub struct Message {
    msg: Vec<F>, // every 16bits one field
}

/// Verifies a single ECDSA signature on a message.
pub fn verify_one(msg: Message, sig: ECDSASignature, pubkey: ECDSAPubKey) {
    let keccak = Keccak256::new();
    // If we don't have constraints Varuna will panic.
    // So here are some dummy constraints to make the code compile & run.
    let rest_bits = Vec::new();
    let mut hash_massge = keccak.hash(&msg.msg, &rest_bits);
    hash_massge = hash_massge
        .chunks(8)
        .rev() // Reverse the bytes for little-endian bit order
        .flat_map(|byte| byte.to_vec())
        .collect::<Vec<_>>();

    pubkey.public_key.on_curve();
    pubkey
        .public_key
        .x
        .less_than_non_native_base_field_modulus();
    pubkey
        .public_key
        .y
        .less_than_non_native_base_field_modulus();

    sig.verify(&hash_massge, &pubkey);
}
