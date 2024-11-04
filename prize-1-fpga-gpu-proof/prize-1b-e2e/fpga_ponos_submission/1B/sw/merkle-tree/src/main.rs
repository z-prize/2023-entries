use ark_bls12_381::{Bls12_381, Fr};
use ark_ed_on_bls12_381::EdwardsParameters;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use merkle_tree::HEIGHT;
use merkle_tree::{MerkleTree, MerkleTreeCircuit};
use plonk_core::commitment::KZG10;
use plonk_core::prelude::{
    verify_proof, Circuit, StandardComposer, VerifierData,
};
use plonk_core::proof_system::Prover;
use plonk_hashing::poseidon::constants::PoseidonConstants;
use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;

fn main() {
    let total_time = std::time::Instant::now();
    let mut rng = test_rng();

    let param = PoseidonConstants::<Fr>::generate::<3>();

    // ==============================
    // first we build a merkle tree
    // ==============================
    let build_merkle_tree_time = std::time::Instant::now();

    let leaf_nodes = (0..1 << (HEIGHT - 1))
        .map(|_| Fr::rand(&mut rng))
        .collect::<Vec<_>>();
    let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
        &param,
        &leaf_nodes,
    );

    let index = rng.next_u32() % (1 << (HEIGHT - 1));
    let proof = tree.gen_proof(index as usize);
    let res = proof.verify(&param, &tree.root());

    // ==============================
    // next we generate the constraints for the tree
    // ==============================
    let gen_mtree_constraints_time = std::time::Instant::now();
    let mut composer = StandardComposer::<Fr, EdwardsParameters>::new();
    tree.gen_constraints(&mut composer, &param);

    composer.check_circuit_satisfied();
    // ==============================
    // last we generate the plonk proof
    // ==============================
    {
        let now = std::time::Instant::now();

        // public parameters
        let size = 1 << (HEIGHT + 9);
        let pp = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();

        let mut dummy_circuit = MerkleTreeCircuit {
            param: param.clone(),
            merkle_tree: tree,
        };

        // preprocessing
        let preprocess_time = std::time::Instant::now();
        let (pk, (vk, _pi_pos)) =
            dummy_circuit.compile::<KZG10<Bls12_381>>(&pp).unwrap();

        // proof generation
        let leaf_nodes_mt_hashing = std::time::Instant::now();
        let leaf_nodes = (0..1 << (HEIGHT - 1))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
            &param,
            &leaf_nodes,
        );
        let mut real_circuit = MerkleTreeCircuit {
            param: param.clone(),
            merkle_tree: tree,
        };
        let proof_cal = std::time::Instant::now();
        let (proof, pi) = {
            let time1 = std::time::Instant::now();
            let mut prover =
                Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new(
                    b"Merkle tree",
                );
            let gadget_time = std::time::Instant::now();
            real_circuit.gadget(prover.mut_cs()).unwrap();
            let proof_gen_time = std::time::Instant::now();
            real_circuit
                .gen_proof::<KZG10<Bls12_381>>(&pp, pk, b"Merkle tree")
                .unwrap()
        };

        let varification_time = std::time::Instant::now();
        let verifier_data = VerifierData::new(vk, pi.clone());
        let res = verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
            &pp,
            verifier_data.key.clone(),
            &proof,
            &verifier_data.pi,
            b"Merkle tree",
        );
    }
}
