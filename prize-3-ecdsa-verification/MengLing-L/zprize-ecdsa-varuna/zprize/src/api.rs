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

use k256::ProjectivePoint;
use rand::rngs::OsRng;
use snarkvm_algorithms::{
    crypto_hash::PoseidonSponge,
    polycommit::kzg10::UniversalParams,
    snark::varuna::{self, AHPForR1CS, CircuitProvingKey, CircuitVerifyingKey, VarunaHidingMode},
    traits::SNARK,
};
use snarkvm_circuit::{
    environment::{Assignment, Circuit},
    Environment as _,
};
use snarkvm_circuit_environment::{Inject, Mode};
use snarkvm_console::network::Testnet3 as Network;
use snarkvm_console_network::Network as _;
use snarkvm_curves::bls12_377::{Bls12_377, Fq, Fr};

use std::collections::BTreeMap;

use crate::console::{
    self, compute_non_native_signature, compute_scalar_from_hash_message, hash_message,
    nonnative::group::ConsoleNonNativePoint, ECDSAPubKey,
};
use crate::Tuples;
use crate::{
    circuit,
    console::{msg_to_bits, Message},
};

//
// Aliases
// =======
//

type FS = PoseidonSponge<Fq, 2, 1>;
type VarunaInst = varuna::VarunaSNARK<Bls12_377, FS, VarunaHidingMode>;

//
// Functions
// =========
//

/// Our circuit synthesizer.
pub fn run_circuit(
    tables: console::table::Table<Network>,
    pub_key: console::ECDSAPubKey<Network>,
    sig: console::ECDSASignature<Network>,
    msg: console::Message,
) -> Assignment<Fr> {
    // reset circuit writer
    Circuit::reset();

    // sample msg and witness as public input
    let msg = circuit::Message::new(Mode::Public, msg.clone());

    let pub_key = circuit::ECDSAPubKey::new(Mode::Public, pub_key);

    let sig = circuit::ECDSASignature::new(Mode::Private, sig);

    let _: circuit::table::Table = circuit::table::Table::new(Mode::Constant, tables);

    // run circuit
    circuit::verify_one(msg, sig, pub_key);

    assert!(Circuit::is_satisfied());
    // return circuit
    Circuit::eject_assignment_and_reset()
}

/// Setup the parameters.
pub fn setup(
    num_constraints: usize,
    num_variables: usize,
    num_non_zero: usize,
) -> UniversalParams<Bls12_377> {
    // Note: you can change this to increase the size of the circuit.
    // Of course, the higher these values, the slower the prover...
    let max_degree = AHPForR1CS::<Fr, VarunaHidingMode>::max_degree(
        num_constraints,
        num_variables,
        num_non_zero,
    )
    .unwrap();
    VarunaInst::universal_setup(max_degree).unwrap()
}

/// Compile the circuit.
pub fn compile(
    urs: &UniversalParams<Bls12_377>,
    msg_len: usize,
) -> (
    CircuitProvingKey<Bls12_377, VarunaHidingMode>,
    CircuitVerifyingKey<Bls12_377>,
) {
    let msg = console::sample_msg(msg_len);
    let (pubkey, sig, msg) = console::sample_pubkey_sig(&msg);
    let tables: console::table::Table<Network> = console::table::Table::setup().unwrap();
    let circuit = run_circuit(tables, pubkey, sig, msg);
    println!("num constraints: {}", circuit.num_constraints());
    println!("num public: {}", circuit.num_public());
    println!("num private: {}", circuit.num_private());
    println!("num non-zeros: {:?}", circuit.num_nonzeros());
    VarunaInst::circuit_setup(urs, &circuit).unwrap()
}

/// Run and prove the circuit.
pub fn prove(
    urs: &UniversalParams<Bls12_377>,
    pk: &CircuitProvingKey<Bls12_377, VarunaHidingMode>,
    tuples: Tuples,
) -> varuna::Proof<Bls12_377> {
    // Prepare the instances.
    let mut instances = BTreeMap::new();
    let mut assignments = Vec::with_capacity(tuples.len());
    for tuple in tuples {
        // Note: we use a naive encoding here,
        // you can modify it as long as a verifier can still pass tuples `(public key, msg, signature)`.
        let (public_key, msg, signature) = tuple;

        // Perform hashing only once per tuple.
        let msg_hash = hash_message(&msg);

        // Get scalar from hash of the message
        let hm = compute_scalar_from_hash_message(&msg_hash);

        // Convert public key to projective and non native type once.
        let pubkey_project: ProjectivePoint = public_key.as_affine().into();
        let non_native_public_key = ConsoleNonNativePoint::new(public_key.as_affine());

        // Convert signature to the non native type once.
        let sig = compute_non_native_signature(&hm, &signature, &pubkey_project);

        // Initialize the lookup table
        let tables: console::table::Table<Network> = console::table::Table::setup().unwrap();

        // Construct circuit.
        let assignment = run_circuit(
            tables,
            ECDSAPubKey {
                public_key: non_native_public_key,
            },
            sig,
            Message {
                msg: msg_to_bits(&msg),
            },
        );
        assignments.push(assignment);
    }

    instances.insert(pk, &assignments[..]);

    // Compute the proof.
    let rng = &mut OsRng;
    let universal_prover = urs.to_universal_prover().unwrap();
    let fiat_shamir = Network::varuna_fs_parameters();

    VarunaInst::prove_batch(&universal_prover, fiat_shamir, &instances, rng).unwrap()
}

/// Verify a proof.
pub fn verify_proof(
    urs: &UniversalParams<Bls12_377>,
    vk: &CircuitVerifyingKey<Bls12_377>,
    tuples: &Tuples,
    proof: &varuna::Proof<Bls12_377>,
) {
    // Note: this is a hacky way of formatting public inputs,
    // we shouldn't have to run the circuit to do that.
    let mut inputs = Vec::with_capacity(tuples.len());
    for tuple in tuples {
        let (public_key, msg, signature) = tuple.clone();

        // Perform hashing only once per tuple.
        let msg_hash = hash_message(&msg);

        // Get scalar from hash of the message
        let hm = compute_scalar_from_hash_message(&msg_hash);

        // Convert public key to projective and non native type once.
        let pubkey_project: ProjectivePoint = public_key.as_affine().into();
        let non_native_public_key = ConsoleNonNativePoint::new(public_key.as_affine());

        // Convert signature to the non native type once.
        let sig = compute_non_native_signature(&hm, &signature, &pubkey_project);

        // Initialize the lookup table
        let tables: console::table::Table<Network> = console::table::Table::setup().unwrap();

        // Construct circuit.
        let assignment = run_circuit(
            tables,
            ECDSAPubKey {
                public_key: non_native_public_key,
            },
            sig,
            Message {
                msg: msg_to_bits(&msg),
            },
        );

        // prepare input
        let mut inputs_i = vec![];
        for (_, input) in assignment.public_inputs().iter() {
            inputs_i.push(*input);
        }
        inputs.push(inputs_i);
    }

    // verify
    let mut keys_to_inputs = BTreeMap::new();
    keys_to_inputs.insert(vk, &inputs[..]);
    let fiat_shamir = Network::varuna_fs_parameters();
    let universal_verifier = urs.to_universal_verifier().unwrap();

    // Note: same comment here, verify_batch could verify several proofs instead of one ;)
    VarunaInst::verify_batch(&universal_verifier, fiat_shamir, &keys_to_inputs, proof).unwrap();
}
