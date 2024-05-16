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

use snarkvm_circuit_environment::{Eject, FromBits, Inject, Mode};
use snarkvm_console_network::Testnet3;

// TODO: better to use AleoV0 or Circuit?

use crate::console::nonnative::basic::K;

use super::*;

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

impl Inject for ECDSASignature {
    type Primitive = crate::console::ECDSASignature<Testnet3>;

    fn new(mode: Mode, input: Self::Primitive) -> Self {
        Self {
            s: input.s.0.iter().map(|b| F::new(mode, *b)).collect_vec(),
            sinv: input.sinv.0.iter().map(|b| F::new(mode, *b)).collect_vec(),
            r: input.r.0.iter().map(|b| F::new(mode, *b)).collect_vec(),
            sinv_mul_hm: CircuitNonNativeMul::new(mode, input.sinv_mul_hm),
            sinv_mul_r: CircuitNonNativeMul::new(mode, input.sinv_mul_r),
            sinv_mul_s: CircuitNonNativeMul::new(mode, input.sinv_mul_s),
            sinv_mul_hm_p: CircuitNonNativePointMul::new(mode, input.sinv_mul_hm_p),
            sinv_mul_r_pub_key: CircuitNonNativePointMul::new(mode, input.sinv_mul_r_pub_key),
            sinv_mul_hm_p_add_sinv_mul_r_pub_key: CircuitNonNativePointAdd::new(
                mode,
                input.sinv_mul_hm_p_add_sinv_mul_r_pub_key,
            ),
        }
    }
}

impl Eject for ECDSASignature {
    type Primitive = crate::console::ECDSASignature<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.r.eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        Self::Primitive {
            s: Polynomial(self.s.eject_value()),
            sinv: Polynomial(self.sinv.eject_value()),
            r: Polynomial(self.r.eject_value()),
            sinv_mul_hm: self.sinv_mul_hm.eject_value(),
            sinv_mul_r: self.sinv_mul_r.eject_value(),
            sinv_mul_s: self.sinv_mul_s.eject_value(),
            sinv_mul_hm_p: self.sinv_mul_hm_p.eject_value(),
            sinv_mul_r_pub_key: self.sinv_mul_r_pub_key.eject_value(),
            sinv_mul_hm_p_add_sinv_mul_r_pub_key: self
                .sinv_mul_hm_p_add_sinv_mul_r_pub_key
                .eject_value(),
        }
    }
}

impl ECDSAVerify for ECDSASignature {
    fn verify(&self, hash_massge: &Vec<B>, public_key: &ECDSAPubKey) {
        let noninfinity = &self
            .sinv_mul_hm_p_add_sinv_mul_r_pub_key
            .point_c
            .noninfinity;

        let hm = hash_massge
            .chunks(K)
            .map(|chunks| F::from_bits_le(chunks))
            .collect_vec();

        NONNATIVE_SCALAR_FIELD_IDENTITY.with(|ident| {
            self.sinv_mul_s.c.poly_eq(noninfinity, &ident, &ident);
        });

        CHUNK_SIZE.with(|chunk_size| {
            self.sinv_mul_hm
                .scalar_poly_mul_wellform(chunk_size, &self.sinv, &hm);
            self.sinv_mul_r
                .scalar_poly_mul_wellform(chunk_size, &self.sinv, &self.r);
            self.sinv_mul_s
                .scalar_poly_mul_wellform(chunk_size, &self.sinv, &self.s);
        });

        let sinv_mul_hm_in_bits = self
            .sinv_mul_hm
            .c
            .less_than_non_native_scalar_field_modulus();

        let sinv_mul_r_in_bits = self
            .sinv_mul_r
            .c
            .less_than_non_native_scalar_field_modulus();

        self.sinv_mul_hm_p.generator_mul(&sinv_mul_hm_in_bits);
        self.sinv_mul_r_pub_key
            .point_mul(&sinv_mul_r_in_bits, &public_key.public_key);
        self.sinv_mul_hm_p_add_sinv_mul_r_pub_key
            .point_c
            .x
            .poly_eq(noninfinity, &self.r, &self.r);
    }
}

//
// Functions
// =========
//
