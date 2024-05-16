// Copyright (C) 2019-2023 Aleo Systems Inc.
// This file is part of the snarkVM library.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use k256::ecdsa::{Signature, VerifyingKey};
use snarkvm_algorithms::{
    polycommit::kzg10::UniversalParams,
    snark::varuna::{CircuitProvingKey, CircuitVerifyingKey, VarunaHidingMode},
};
use snarkvm_curves::bls12_377::Bls12_377;

pub mod api;
pub mod circuit;
pub mod console;

/// A (public key, msg, signature) tuple.
pub type Tuples = Vec<(VerifyingKey, Vec<u8>, Signature)>;

pub fn prove_and_verify(
    urs: &UniversalParams<Bls12_377>,
    pk: &CircuitProvingKey<Bls12_377, VarunaHidingMode>,
    vk: &CircuitVerifyingKey<Bls12_377>,
    tuples: &Tuples,
) {
    // TODO: test could be adjusted to pass references and clone less
    let proof = api::prove(urs, pk, tuples.clone());

    // Note: proof verification should take negligible time,
    api::verify_proof(urs, vk, tuples, &proof);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // generate `num` (pubkey, msg, signature)
        // with messages of length `msg_len`
        let num = 1;
        let msg_len = 50000;
        let tuples = console::generate_signatures(msg_len, num);

        // setup
        let urs = api::setup(1000, 1000, 1000);
        let (pk, vk) = api::compile(&urs, msg_len);

        // prove and verify
        prove_and_verify(&urs, &pk, &vk, &tuples);
    }
}
