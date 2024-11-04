//! This file implements the tree construction.
//! This file is adopted from <https://github.com/arkworks-rs/crypto-primitives/blob/main/src/merkle_tree/mod.rs>
//! with substantial changes.

use core::fmt;
use std::fmt::Display;

use ark_std::{end_timer, start_timer};
use plonk_hashing::poseidon::{
    constants::PoseidonConstants, poseidon_ref::PoseidonRefSpec,
};

use crate::{
    hash,
    path::Path,
    util::{
        convert_index_to_last_level, is_left_child, left_child_index,
        parent_index, right_child_index, sibling_index,
    },
    HEIGHT,
};

/// A merkle tree. Associated type: poseidon spec with a hard coded width = 3
#[derive(Clone, Debug, Default)]
pub struct MerkleTree<S: PoseidonRefSpec<(), 3>> {
    /// stores the non-leaf nodes in level order. The first element is the root
    /// node. The ith nodes (starting at 1st) children are at indices
    /// `2*i`, `2*i+1`
    pub non_leaf_nodes: Vec<S::Field>,

    /// store the hash of leaf nodes from left to right
    pub leaf_nodes: Vec<S::Field>,
}

impl<S: PoseidonRefSpec<(), 3>> Display for MerkleTree<S>
where
    S::Field: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "non leaf nodes:")?;
        for (i, e) in self.non_leaf_nodes.iter().enumerate() {
            writeln!(f, "{}: {}", i, e)?;
        }
        writeln!(f, "leaf nodes:")?;
        for (i, e) in self.leaf_nodes.iter().enumerate() {
            writeln!(f, "{}: {}", i, e)?;
        }
        Ok(())
    }
}

// Hard coded with poseidon width = 3.
impl<S: PoseidonRefSpec<(), 3>> MerkleTree<S>
where
    S::Field: Default + Copy,
{
    /// create an empty tree
    pub fn init(param: &PoseidonConstants<S::ParameterField>) -> Self {
        let leaf_nodes = vec![S::Field::default(); 1 << (HEIGHT - 1)];
        Self::new_with_leaf_nodes(param, &leaf_nodes)
    }

    /// create a new tree with leaf nodes
    pub fn new_with_leaf_nodes(
        param: &PoseidonConstants<S::ParameterField>,
        leaf_nodes: &[S::Field],
    ) -> Self {
        let timer = start_timer!(|| format!(
            "generate new tree with {} leaves",
            leaf_nodes.len()
        ));

        let len = leaf_nodes.len();
        assert_eq!(len, 1 << (HEIGHT - 1), "incorrect leaf size");

        let mut non_leaf_nodes =
            vec![S::Field::default(); (1 << (HEIGHT - 1)) - 1];

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

            non_leaf_nodes
                .iter_mut()
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
                    // compute hash
                    *e = hash::<S>(
                        param,
                        &leaf_nodes[left_leaf_index],
                        &leaf_nodes[right_leaf_index],
                    );
                });
        }

        // compute the hash values for nodes in every other layer in the tree
        level_indices.reverse();

        for &start_index in &level_indices {
            // The layer beginning `start_index` ends at `upper_bound`
            // (exclusive).
            let upper_bound = left_child_index(start_index);
            let mut buf = non_leaf_nodes[start_index..upper_bound].to_vec();
            buf.iter_mut().enumerate().for_each(|(index, node)| {
                *node = hash::<S>(
                    param,
                    &non_leaf_nodes[left_child_index(index + start_index)],
                    &non_leaf_nodes[right_child_index(index + start_index)],
                );
            });
            non_leaf_nodes[start_index..upper_bound]
                .clone_from_slice(buf.as_ref());
        }
        end_timer!(timer);
        Self {
            non_leaf_nodes,
            leaf_nodes: leaf_nodes.to_vec(),
        }
    }
    pub fn root(&self) -> S::Field {
        self.non_leaf_nodes[0]
    }

    // generate a membership proof for the given index
    pub fn gen_proof(&self, index: usize) -> Path<S> {
        // Get Leaf hash, and leaf sibling hash,
        let leaf_index_in_tree = convert_index_to_last_level(index, HEIGHT);

        // path.len() = `tree height - 1`, the missing elements being the
        // root
        let mut nodes = Vec::with_capacity(HEIGHT - 1);
        if index % 2 == 0 {
            nodes.push((self.leaf_nodes[index], self.leaf_nodes[index + 1]))
        } else {
            nodes.push((self.leaf_nodes[index - 1], self.leaf_nodes[index]))
        }

        // Iterate from the bottom layer after the leaves, to the
        // top, storing all nodes and their siblings.
        let mut current_node = parent_index(leaf_index_in_tree).unwrap();
        while current_node != 0 {
            let sibling_node = sibling_index(current_node).unwrap();
            if is_left_child(current_node) {
                nodes.push((
                    self.non_leaf_nodes[current_node],
                    self.non_leaf_nodes[sibling_node],
                ));
            } else {
                nodes.push((
                    self.non_leaf_nodes[sibling_node],
                    self.non_leaf_nodes[current_node],
                ));
            }
            current_node = parent_index(current_node).unwrap();
        }

        // we want to make path from root to bottom
        nodes.reverse();

        Path { index, nodes }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_std::{rand::RngCore, test_rng, UniformRand};
    use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;

    #[test]
    fn test_tree() {
        let mut rng = test_rng();
        let param = PoseidonConstants::<Fr>::generate::<3>();

        let leafs: Vec<Fr> = (0..(1 << (HEIGHT - 1)))
            .map(|_| Fr::rand(&mut rng))
            .collect();
        let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
            &param, &leafs,
        );
        for _ in 0..100 {
            let index = rng.next_u32() % (1 << (HEIGHT - 1));
            let proof = tree.gen_proof(index as usize);
            assert!(proof.verify(&param, &tree.non_leaf_nodes[0],));
        }
    }
}
