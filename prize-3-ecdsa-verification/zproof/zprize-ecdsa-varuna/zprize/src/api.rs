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

use aleo_std_profiler::{end_timer, start_timer};
use anyhow::{bail, Context};
use rand::rngs::OsRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use sha3::{Digest, Keccak256};
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
use snarkvm_circuit_environment::SameCircuitAssignment;
use snarkvm_console::{network::Testnet3 as Network, program::Itertools};
use snarkvm_console_network::Network as _;
use snarkvm_curves::bls12_377::{Bls12_377, Fq, Fr};
use std::{collections::BTreeMap, sync::Arc};

use crate::r1cs_provider;
use crate::Tuples;
use crate::{console, CircuitRunType};

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

pub fn run_circuit_keccak(msgs: Vec<Vec<u8>>) -> Assignment<Fr> {
    // reset circuit writer
    Circuit::reset();

    r1cs_provider::gnark::build_r1cs_for_verify_plonky2(msgs)
        .context("failed to build keccak circuit")
        .unwrap();

    // return circuit
    Circuit::eject_assignment_and_reset()
}

/// Our circuit synthesizer for ecdsa.
///
pub fn run_circuit_ecdsa(
    public_key: &console::ECDSAPublicKey,
    signature: &console::ECDSASignature,
    msg: &[u8],
) -> Assignment<Fr> {
    // reset circuit writer
    Circuit::reset();

    let mut hasher = Keccak256::new();
    hasher.update(&msg);
    let hash = hasher
        .finalize()
        .as_slice()
        .try_into()
        .expect("Wrong length");

    r1cs_provider::gnark::build_r1cs_for_verify_ecdsa(public_key, signature, &hash)
        .context("failed to build ecdsa circuit")
        .unwrap();

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
    run_type: CircuitRunType,
    urs: &UniversalParams<Bls12_377>,
    tuple_num: usize,
    msg_len: usize,
) -> Vec<(
    CircuitProvingKey<Bls12_377, VarunaHidingMode>,
    CircuitVerifyingKey<Bls12_377>,
)> {
    println!("compile({run_type:?}) tuple_num: {tuple_num} msg_len: {msg_len}");

    /* Sample some tuples */
    let msg = console::sample_msg(msg_len);
    let (public_key, signature) = console::sample_pubkey_sig(&msg);

    let mut unique_circuits = vec![];

    if matches!(
        run_type,
        CircuitRunType::RunKeccakOnly
            | CircuitRunType::RunKeccakAndEcdsa
            | CircuitRunType::RunKeccakThenEcdsa
    ) {
        // Let's get a keccak circuit
        let keccak_circuit = run_circuit_keccak(
            // TODO: optimize this alloc
            (0..tuple_num)
                .into_iter()
                .map(|_| msg.clone())
                .collect_vec(),
        );
        println!(
            "keccak_circuit: num constraints: {}",
            keccak_circuit.num_constraints()
        );
        println!(
            "keccak_circuit: num lookup tables: {}",
            keccak_circuit.num_lookup_tables()
        );
        println!(
            "keccak_circuit: num lookup constraints: {}",
            keccak_circuit.num_lookup_constraints()
        );
        println!(
            "keccak_circuit: num public: {}",
            keccak_circuit.num_public()
        );
        println!(
            "keccak_circuit: num private: {}",
            keccak_circuit.num_private()
        );
        println!(
            "keccak_circuit: num non-zeros(both non-lookup and lookup): {:?}",
            keccak_circuit.num_nonzeros()
        );
        unique_circuits.push(keccak_circuit);
    }

    if matches!(
        run_type,
        CircuitRunType::RunEcdsaOnly
            | CircuitRunType::RunKeccakAndEcdsa
            | CircuitRunType::RunKeccakThenEcdsa
    ) {
        // Let's get a ecdsa circuit
        let ecdsa_circuit = run_circuit_ecdsa(&public_key, &signature, &msg);
        println!(
            "ecdsa_circuit: num constraints: {}",
            ecdsa_circuit.num_constraints()
        );
        println!(
            "ecdsa_circuit: num lookup tables: {}",
            ecdsa_circuit.num_lookup_tables()
        );
        println!(
            "ecdsa_circuit: num lookup constraints: {}",
            ecdsa_circuit.num_lookup_constraints()
        );
        println!("ecdsa_circuit: num public: {}", ecdsa_circuit.num_public());
        println!(
            "ecdsa_circuit: num private: {}",
            ecdsa_circuit.num_private()
        );
        println!(
            "ecdsa_circuit: num non-zeros(both non-lookup and lookup): {:?}",
            ecdsa_circuit.num_nonzeros()
        );
        unique_circuits.push(ecdsa_circuit);
    }

    if matches!(run_type, CircuitRunType::RunKeccakThenEcdsa) {
        /* If it is RunKeccakThenEcdsa, we have to run batch_circuit_setup() for keccak and ecdsa separately. */
        let mut r0 = VarunaInst::batch_circuit_setup(&urs, &[&unique_circuits[0]])
            .with_context(|| format!("VarunaInst::batch_circuit_setup() for circuit: keccak"))
            .unwrap();
        let r1 = VarunaInst::batch_circuit_setup(&urs, &[&unique_circuits[1]])
            .with_context(|| format!("VarunaInst::batch_circuit_setup() for circuit: ecdsa"))
            .unwrap();
        r0.extend(r1);
        r0
    } else {
        /* If it is RunKeccaOnly or RunEcdsaOnly or RunKeccakAndEcdsa, just run batch_circuit_setup() once. */
        let unique_circuits = unique_circuits.iter().map(|c| c).collect_vec();
        VarunaInst::batch_circuit_setup(&urs, &unique_circuits).unwrap()
    }
}

/// Run and prove the circuit.
pub fn prove(
    run_type: CircuitRunType,
    urs: &UniversalParams<Bls12_377>,
    pks: &[&CircuitProvingKey<Bls12_377, VarunaHidingMode>],
    tuples: Tuples,
) -> (varuna::Proof<Bls12_377>, Vec<Vec<Vec<Fr>>>) {
    if matches!(run_type, CircuitRunType::RunKeccakThenEcdsa) {
        /* RunKeccakThenEcdsa should be handled by prove_and_verify() */
        panic!("CircuitRunType::RunKeccakThenEcdsa should not be here !");
    }

    let mut pks_to_constraints = BTreeMap::new();

    let mut keccak_assignments = None;
    if matches!(
        run_type,
        CircuitRunType::RunKeccakOnly | CircuitRunType::RunKeccakAndEcdsa
    ) {
        let keccak_pk = pks[0];
        let keccak_assignment = run_circuit_keccak(
            // TODO: optimize this alloc
            tuples.iter().map(|t| t.1.clone()).collect_vec(),
        );
        keccak_assignments = Some([SameCircuitAssignment::single_one(keccak_assignment)]);
        pks_to_constraints.insert(keccak_pk, &keccak_assignments.as_ref().unwrap()[..]);
    }

    let mut ecdsa_assignments = None;
    if matches!(
        run_type,
        CircuitRunType::RunEcdsaOnly | CircuitRunType::RunKeccakAndEcdsa
    ) {
        let ecdsa_pk = match run_type {
            CircuitRunType::RunEcdsaOnly => pks[0],
            CircuitRunType::RunKeccakAndEcdsa => pks[1],
            _ => unreachable!(),
        };

        let base_assignment = run_circuit_ecdsa(
            &console::ECDSAPublicKey {
                public_key: tuples[0].0.clone(),
            },
            &console::ECDSASignature {
                signature: tuples[0].2.clone(),
            },
            &tuples[0].1,
        );

        if tuples.len() == 1 {
            ecdsa_assignments = Some(vec![SameCircuitAssignment::single_one(base_assignment)]);
            pks_to_constraints.insert(ecdsa_pk, &ecdsa_assignments.as_ref().unwrap()[..]);
        } else {
            let base_assignment = Arc::new(base_assignment);

            /* limit num of parallel tasks here for saving memory */
            let num_parallel_tasks = 5;
            ecdsa_assignments = Some(
                tuples
                    .into_par_iter()
                    .with_min_len(tuples.len() / num_parallel_tasks)
                    .map(|tuple| {
                        // Note: we use a naive encoding here,
                        // you can modify it as long as a verifier can still pass tuples `(public key, msg, signature)`.
                        let (public_key, msg, signature) = tuple;
                        let assignment = run_circuit_ecdsa(
                            &console::ECDSAPublicKey {
                                public_key: public_key.clone(),
                            },
                            &console::ECDSASignature {
                                signature: signature.clone(),
                            },
                            msg,
                        );
                        SameCircuitAssignment::create_with_base(base_assignment.clone(), assignment)
                    })
                    .collect::<Vec<_>>(),
            );

            pks_to_constraints.insert(ecdsa_pk, &ecdsa_assignments.as_ref().unwrap()[..]);
        }
    }

    // Compute the proof.
    let rng = &mut OsRng::default();
    let universal_prover = urs.to_universal_prover().unwrap();
    let fiat_shamir = Network::varuna_fs_parameters();
    let proof =
        VarunaInst::prove_batch(&universal_prover, fiat_shamir, &pks_to_constraints, rng).unwrap();

    /* Prepare inputs for verifier, this should be verify fast since it is just memory copy ... */
    let mut all_inputs = vec![];

    if matches!(
        run_type,
        CircuitRunType::RunKeccakOnly | CircuitRunType::RunKeccakAndEcdsa
    ) {
        let keccak_inputs = keccak_assignments
            .as_ref()
            .unwrap()
            .iter()
            .map(|assignment| {
                assignment
                    .public_inputs()
                    .iter()
                    .map(|(_, input)| *input)
                    .collect_vec()
            })
            .collect_vec();
        all_inputs.push(keccak_inputs);
    }

    if matches!(
        run_type,
        CircuitRunType::RunEcdsaOnly | CircuitRunType::RunKeccakAndEcdsa
    ) {
        let ecdsa_inputs = ecdsa_assignments
            .as_ref()
            .unwrap()
            .iter()
            .map(|assignment| {
                assignment
                    .public_inputs()
                    .iter()
                    .map(|(_, input)| *input)
                    .collect_vec()
            })
            .collect_vec();
        all_inputs.push(ecdsa_inputs);
    }

    (proof, all_inputs)
}

/// Verify a proof.
pub fn verify_proof(
    urs: &UniversalParams<Bls12_377>,
    proof: &varuna::Proof<Bls12_377>,
    vks_to_inputs: &BTreeMap<&CircuitVerifyingKey<Bls12_377>, &[Vec<Fr>]>,
) {
    // verify
    let fiat_shamir = Network::varuna_fs_parameters();
    let universal_verifier = urs.to_universal_verifier().unwrap();

    // Note: same comment here, verify_batch could verify several proofs instead of one ;)
    let time = start_timer!(|| "Run VarunaInst::verify_batch()");
    VarunaInst::verify_batch(&universal_verifier, fiat_shamir, vks_to_inputs, proof).unwrap();
    end_timer!(time);
}
