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
use super::{nonnative::group::CircuitNonNativePoint, ECDSAPubKey};
use snarkvm_circuit_environment::{Eject, Inject, Mode};
use snarkvm_console_network::Testnet3;

//use snarkvm_circuit::{AleoV0 as Env, Field};

//
// Aliases
// =======
//

//
// Data structures
// ===============
//
// We use the Bls12_377 curve to instantiate Varuna.
// This means that the circuit field (Fr) is modulo the following prime:
//
// 8444461749428370424248824938781546531375899335154063827935233455917409239041
//
// To verify ECDSA signatures with the secp256k1 curve,
// we have to handle scalar field and base field elements:
//
// - scalar field: 115792089237316195423570985008687907852837564279074904382605163141518161494337
// - base field: 115792089237316195423570985008687907853269984665640564039457584007908834671663
//
// Both of these are larger than our circuit field.
// While both the secp256k1 fields are 256 bits, the Bls12_377 circuit field is 253 bits.
// So this means we'll have to cut things up.
//
// Since the job is really on you to figure out how to optimize sutff,
// we use a super naive way to encode things: 1 field = 1 byte.
//

//
// Trait implementations
// =====================
//

impl Inject for ECDSAPubKey {
    type Primitive = crate::console::ECDSAPubKey<Testnet3>;

    fn new(mode: Mode, input: Self::Primitive) -> Self {
        Self {
            public_key: CircuitNonNativePoint::new(mode, input.public_key),
        }
    }
}

impl Eject for ECDSAPubKey {
    type Primitive = crate::console::ECDSAPubKey<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.public_key.eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        Self::Primitive {
            public_key: self.public_key.eject_value(),
        }
    }
}
