use ark_bls12_381::{Bls12_381, Fr, G1Affine};
use ark_ed_on_bls12_381::EdwardsParameters;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use merkle_tree::HEIGHT;
use merkle_tree::{MerkleTree, MerkleTreeCircuit};
use plonk_core::commitment::KZG10;
use plonk_core::prelude::{verify_proof, Circuit, VerifierData};
use plonk_core::proof_system::Prover;
use plonk_hashing::poseidon::constants::PoseidonConstants;
use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;

use ark_poly_commit::sonic_pc::{CommitterKey, UniversalParams};
use ec_gpu_common::api::*;
use ec_gpu_common::G1AffineNoInfinity;
use std::os::raw::c_void;

// repeating the prover and verifier for REPEAT times
const REPEAT: usize = 4;

fn build_tree<R: RngCore>(
    rng: &mut R,
    param: &PoseidonConstants<Fr>,
) -> MerkleTree<NativeSpecRef<Fr>> {
    let leaf_nodes = (0..1 << (HEIGHT - 1))
        .map(|_| Fr::rand(rng))
        .collect::<Vec<_>>();
    MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(param, &leaf_nodes)
}

fn send_poseidon_constants(poseidon_constants: &PoseidonConstants<Fr>) {
    let mut matrix = Vec::with_capacity(9);
    let tag = vec![poseidon_constants.domain_tag];

    for row_idx in 0..3 {
        for column_idx in 0..3 {
            matrix
                .push(poseidon_constants.mds_matrices.m.0[row_idx][column_idx]);
        }
    }

    unsafe {
        sync_hash_consts(
            poseidon_constants.full_rounds as u64,
            poseidon_constants.half_full_rounds as u64,
            poseidon_constants.partial_rounds as u64,
            poseidon_constants.round_constants.len() as u64,
            poseidon_constants.round_constants.as_ptr() as *const c_void,
            tag.as_ptr() as *const c_void,
            matrix.as_ptr() as *const c_void,
        );
    }
}

fn send_merkle_tree(circuit: &MerkleTreeCircuit) {
    unsafe {
        sync_mt(
            HEIGHT as i32,
            circuit.merkle_tree.non_leaf_nodes.as_ptr() as *const c_void,
            circuit.merkle_tree.leaf_nodes.as_ptr() as *const c_void,
        );
    }
}

fn send_bases(coeffs_count: i32, ck: &CommitterKey<Bls12_381>) {
    let bases = ck
        .powers_of_g
        .iter()
        .map(G1AffineNoInfinity::from)
        .collect::<Vec<_>>();
    unsafe {
        init_bases(bases.as_ptr() as *const c_void, coeffs_count);
    }
}

fn allocate_data(coeffs_count: i32) {
    unsafe {
        init_data(coeffs_count);
    }
}

fn run_composer(
    prover: &mut Prover<Fr, EdwardsParameters, KZG10<Bls12_381>>,
) {
    let cs = prover.mut_cs();

    let hash_vars = cs.get_vars();
    let mut vars = Vec::with_capacity(hash_vars.len());
    for i in 0..hash_vars.len() {
        vars.push(hash_vars[&cs.get_varib(i as usize)]);
    }

    unsafe {
        sync_composer(
            HEIGHT as i32,
            cs.get_wl().as_ptr() as *const c_void,
            cs.get_wr().as_ptr() as *const c_void,
            cs.get_wo().as_ptr() as *const c_void,
            cs.get_w4().as_ptr() as *const c_void,
            cs.get_wl().len() as u64,
            vars.as_ptr() as *const c_void,
            cs.get_vars().len() as u64,
        );
    }
}

fn main() {
    let mut rng = test_rng();
    let param = PoseidonConstants::<Fr>::generate::<3>();

    send_poseidon_constants(&param);

    // public parameters
    let size = 1 << (HEIGHT + 9);
    let pp = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();

    let dummy_tree = build_tree(&mut rng, &param);
    let mut dummy_circuit = MerkleTreeCircuit {
        param: param.clone(),
        merkle_tree: dummy_tree,
    };

    // pre-processing
    let (pk, (vk, _pi_pos)) =
        dummy_circuit.compile::<KZG10<Bls12_381>>(&pp).unwrap();

    // trace generation
    let mut real_circuits = Vec::with_capacity(REPEAT);
    for _ in 0..REPEAT {
        let tree = build_tree(&mut rng, &param);
        let real_circuit = MerkleTreeCircuit {
            param: param.clone(),
            merkle_tree: tree,
        };
        real_circuits.push(real_circuit);
    }

    let coeffs_count = 1 << (HEIGHT + 7);
    let padded_circuit_size = real_circuits[0].padded_circuit_size();
    let (ck, _) = <KZG10<Bls12_381>>::trim(
        &pp,
        padded_circuit_size,
        0,
        None,
    )
        .unwrap();

    let mut prover = Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new(
        b"Merkle tree",
    );
    prover.prover_key = Some(pk);
    prover.register_data(coeffs_count);
    prover.copy_data();
        
    send_bases(coeffs_count, &ck);

    // proof generation
    let now = std::time::Instant::now();
    println!("==============================");
    println!("Start generating {} proofs", REPEAT);
    let mut proof_and_pi_s = Vec::with_capacity(4);
    let mut overall_real_prove_time = std::time::Duration::ZERO;
    for i in 0..REPEAT {
        let mut circuit = &mut real_circuits[i];

        allocate_data(coeffs_count);
        send_merkle_tree(circuit);
        run_composer(&mut prover);

        let real_prove_time = std::time::Instant::now();

        let result = circuit
            .gen_proof::<KZG10<Bls12_381>, G1Affine>(&mut prover, &ck, None)
            .unwrap();
        proof_and_pi_s.push(result);
        overall_real_prove_time += real_prove_time.elapsed();

        println!("Proof {} is generated", i);
        println!("Time elapsed: {:?}", now.elapsed());
    }
    println!("The total prove generation time is {:?}", now.elapsed());
    println!(
        "Amortized cost for each proof is {:?}",
        now.elapsed() / REPEAT as u32
    );
    println!(
        "Real amortized cost for each proof is {:?} msec",
        overall_real_prove_time.as_millis() / REPEAT as u128
    );
    println!("==============================");

    // proof verification
    let now = std::time::Instant::now();
    println!("Start verifying {} proofs", REPEAT);
    for i in 0..REPEAT {
        let (proof, pi) = &proof_and_pi_s[i];
        let verifier_data = VerifierData::new(vk.clone(), pi.clone());
        let res = verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
            &pp,
            verifier_data.key.clone(),
            &proof,
            &verifier_data.pi,
            b"Merkle tree",
        );
        println!("Proof {} is verified: {}", i, res.is_ok());
        println!("Time elapsed: {:?}", now.elapsed());
    }
    println!("The prove verification time is {:?}", now.elapsed());
    println!(
        "Amortized cost for each proof is {:?}",
        now.elapsed() / REPEAT as u32
    );
    println!("==============================");
}

