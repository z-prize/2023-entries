use ark_bls12_381::Fr;
use ark_ed_on_bls12_381::EdwardsParameters;
use plonk_core::prelude::{Circuit, Error, StandardComposer};
use plonk_hashing::poseidon::{
    constants::PoseidonConstants, poseidon_ref::NativeSpecRef,
};

use crate::{MerkleTree, HEIGHT};
use ark_serialize::*;
use ark_std::sync::Arc;
use ark_std::sync::Mutex;
use plonk_core::permutation::Permutation;

#[derive(CanonicalDeserialize, CanonicalSerialize)]
pub struct MerkleTreeCircuit {
    pub param: PoseidonConstants<Fr>,
    pub merkle_tree: MerkleTree<NativeSpecRef<Fr>>,
}

impl Circuit<Fr, EdwardsParameters> for MerkleTreeCircuit {
    const CIRCUIT_ID: [u8; 32] = [0xff; 32];

    fn gadget(
        &mut self,
        composer: &mut StandardComposer<Fr, EdwardsParameters>,
    ) -> Result<(), Error> {
        self.merkle_tree.gen_constraints(composer, &self.param);
        Ok(())
    }

    fn gadget_with_permutation(
        &mut self,
        composer: &mut StandardComposer<Fr, EdwardsParameters>,
        with_permutation: bool,
    ) -> Result<(), Error> {
        self.merkle_tree
            .gen_constraints_4(composer, &self.param, with_permutation);
        Ok(())
    }

    fn padded_circuit_size(&self) -> usize {
        1 << (HEIGHT + 9)
    }
}

#[cfg(test)]
mod test {
    use crate::HEIGHT;

    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly_commit::PolynomialCommitment;
    use ark_std::{test_rng, UniformRand};
    use plonk_core::{
        commitment::KZG10,
        prelude::{verify_proof, VerifierData},
        proof_system::Prover,
    };
    use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;

    #[test]
    fn test_e2e() {
        let mut rng = test_rng();

        // Generate CRS
        let size = 1 << (HEIGHT + 9);
        let pp = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();

        let hash_param = PoseidonConstants::<Fr>::generate::<3>();

        let leaf_nodes = (0..1 << (HEIGHT - 1))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let merkle_tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
            &hash_param,
            &leaf_nodes,
        );
        let mut circuit = MerkleTreeCircuit {
            param: hash_param,
            merkle_tree,
        };

        let (pk, (vk, _pi_pos)) =
            circuit.compile::<KZG10<Bls12_381>>(&pp).unwrap();

        let (proof, pi) = {
            let mut prover =
                Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new(
                    b"Merkle tree",
                );
            circuit.gadget(prover.mut_cs()).unwrap();
            prover.mut_cs().check_circuit_satisfied();
            circuit
                .gen_proof::<KZG10<Bls12_381>>(&pp, pk, b"Merkle tree")
                .unwrap()
        };

        let verifier_data = VerifierData::new(vk, pi);
        assert!(verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
            &pp,
            verifier_data.key,
            &proof,
            &verifier_data.pi,
            b"Merkle tree",
        )
        .is_ok());
    }
}
