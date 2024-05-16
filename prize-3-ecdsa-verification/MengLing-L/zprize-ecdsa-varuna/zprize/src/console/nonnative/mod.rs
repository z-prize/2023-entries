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

pub mod basic;
pub mod group;

use crate::console::traits::*;
use crate::console::BASE_MODULUS;
use crate::console::SCALAR_MODULUS;
use basic::linear::*;
use basic::mul::*;
use basic::*;
use ecdsa::elliptic_curve::{consts::U32, generic_array::GenericArray, PrimeField};
use group::add::*;
use group::double::*;
use k256::elliptic_curve::group::GroupEncoding;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{AffinePoint, FieldElement, ProjectivePoint, Scalar};
use num_bigint::{BigInt, Sign};
use snarkvm_circuit_environment::prelude::num_traits::FromPrimitive;
use snarkvm_console::types::Field;
use snarkvm_console_algorithms::Itertools;
use snarkvm_console_algorithms::Zero;
use snarkvm_console_network::FromBits;
use snarkvm_console_network::Network;
use snarkvm_utilities::FromBits as OtherFromBits;
use snarkvm_utilities::ToBits;

#[cfg(test)]
use crate::console;
#[cfg(test)]
use k256::ecdsa::SigningKey;
#[cfg(test)]
use k256::elliptic_curve::Field as BaseField;
#[cfg(test)]
use rand::rngs::OsRng;
#[cfg(test)]
use snarkvm_circuit_environment::Pow;
#[cfg(test)]
use snarkvm_console_algorithms::Ordering;
#[cfg(test)]
use snarkvm_console_network::Testnet3;
