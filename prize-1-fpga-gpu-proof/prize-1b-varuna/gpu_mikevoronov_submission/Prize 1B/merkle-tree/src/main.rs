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
    let mut rng = test_rng();

    let param = PoseidonConstants::<Fr>::generate::<3>();

    // ==============================
    // first we build a merkle tree
    // ==============================

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

    // omitted: parameters too large
    // println!("generating merkle tree with parameter {:?}", param);

    //println!("merkle tree with height: {}:\n{}\n", HEIGHT, tree);
    //println!(
    //    "merkle proof for {}-th leaf: {}\n{}\n",
    //    index, leaf_nodes[index as usize], proof
    //);
    //println!("proof is valid: {}", res);

    // ==============================
    // next we generate the constraints for the tree
    // ==============================

    let mut composer = StandardComposer::<Fr, EdwardsParameters>::new();
    tree.gen_constraints(&mut composer, &param);

    composer.check_circuit_satisfied();

    // ==============================
    // last we generate the plonk proof
    // ==============================
    {
        // public parameters
        let size = 1 << (HEIGHT + 9);
        let pp = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();

        let mut dummy_circuit = MerkleTreeCircuit {
            param: param.clone(),
            merkle_tree: tree,
        };

        // proprocessing
        let (pk, (vk, _pi_pos)) =
            dummy_circuit.compile::<KZG10<Bls12_381>>(&pp).unwrap();

        // proof generation
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
        let now = std::time::Instant::now();
        let (proof, pi) = {
            let mut prover =
                Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new(
                    b"Merkle tree",
                );
            real_circuit.gadget(prover.mut_cs()).unwrap();

            real_circuit
                .gen_proof::<KZG10<Bls12_381>>(&pp, pk, b"Merkle tree")
                .unwrap()
        };
        println!("The prove generation time is {:?}", now.elapsed());

        let verifier_data = VerifierData::new(vk, pi.clone());
        let res = verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
            &pp,
            verifier_data.key.clone(),
            &proof,
            &verifier_data.pi,
            b"Merkle tree",
        );

        //println!("proof: {:?}", proof);
        //println!("public input: {:?}", pi);
        //println!("verifier data: {:?}", verifier_data);
        println!("proof is verified: {}", res.is_ok());
    }
}
