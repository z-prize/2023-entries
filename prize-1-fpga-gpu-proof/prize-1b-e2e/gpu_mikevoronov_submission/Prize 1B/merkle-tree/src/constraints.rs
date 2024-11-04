use ark_bls12_381::Fr;
use ark_ed_on_bls12_381::EdwardsParameters;
use ark_std::One;
use ark_std::Zero;
use plonk_core::prelude::StandardComposer;
use plonk_hashing::poseidon::{
    constants::PoseidonConstants, poseidon_ref::NativeSpecRef,
};

use crate::{
    assert_hash_constraints,
    util::{left_child_index, right_child_index},
    MerkleTree, HEIGHT,
};

// Hard coded with poseidon width = 3.
impl MerkleTree<NativeSpecRef<Fr>> {
    /// Generate constraints asserting a merkle tree is generated correctly
    /// This is hard coded for BLS12-381 scalar field
    pub fn gen_constraints(
        &self,
        composer: &mut StandardComposer<Fr, EdwardsParameters>,
        hash_param: &PoseidonConstants<Fr>,
    ) {
        let leaf_node_vars = self
            .leaf_nodes
            .iter()
            .map(|node| composer.add_input(*node))
            .collect::<Vec<_>>();

        let non_leaf_node_vars = self
            .non_leaf_nodes
            .iter()
            .map(|node| composer.add_input(*node))
            .collect::<Vec<_>>();

        let root_node = &non_leaf_node_vars[0];

        // Compute the starting indices for each non-leaf level of the tree
        let mut index = 0;
        let mut level_indices = Vec::with_capacity(HEIGHT - 1);
        for _ in 0..(HEIGHT - 1) {
            level_indices.push(index);
            index = left_child_index(index);
        }
        // compute the hash values for the non-leaf bottom layer
        {
            let start_index = level_indices.pop().unwrap();
            let upper_bound = left_child_index(start_index);

            non_leaf_node_vars
                .iter()
                .enumerate()
                .take(upper_bound)
                .skip(start_index)
                .for_each(|(current_index, e)| {
                    // `left_child_index(current_index)` and
                    // `right_child_index(current_index) returns the position of
                    // leaf in the whole tree (represented as a list in level
                    // order). We need to shift it
                    // by `-upper_bound` to get the index in `leaf_nodes` list.
                    let left_leaf_index =
                        left_child_index(current_index) - upper_bound;
                    let right_leaf_index =
                        right_child_index(current_index) - upper_bound;
                    // assert hash is valid
                    assert_hash_constraints(
                        composer,
                        hash_param,
                        &leaf_node_vars[left_leaf_index],
                        &leaf_node_vars[right_leaf_index],
                        e,
                    )
                });
        }

        // compute the hash values for nodes in every other layer in the tree
        level_indices.reverse();

        for &start_index in &level_indices {
            // The layer beginning `start_index` ends at `upper_bound`
            // (exclusive).
            let upper_bound = left_child_index(start_index);
            non_leaf_node_vars[start_index..upper_bound]
                .iter()
                .enumerate()
                .for_each(|(index, node)| {
                    assert_hash_constraints(
                        composer,
                        hash_param,
                        &non_leaf_node_vars
                            [left_child_index(index + start_index)],
                        &non_leaf_node_vars
                            [right_child_index(index + start_index)],
                        node,
                    )
                });
        }

        // constrain the root is correct w.r.t. public input
        let zero = composer.zero_var();
        composer.arithmetic_gate(|gate| {
            gate.witness(*root_node, zero, Some(zero))
                .add(Fr::one(), Fr::zero())
                .pi(-self.non_leaf_nodes[0])
        });
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::{test_rng, UniformRand};
    use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;

    #[test]
    fn test_tree_constraints() {
        let mut rng = test_rng();
        let param = PoseidonConstants::<Fr>::generate::<3>();
        for _ in 0..10 {
            let leafs: Vec<Fr> = (0..(1 << (HEIGHT - 1)))
                .map(|_| Fr::rand(&mut rng))
                .collect();
            let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
                &param, &leafs,
            );

            let mut composer = StandardComposer::<Fr, EdwardsParameters>::new();
            tree.gen_constraints(&mut composer, &param);

            composer.check_circuit_satisfied();
        }
    }
}
