use ark_bls12_381::Fr;
use ark_ed_on_bls12_381::EdwardsParameters;
use ark_std::One;
use ark_std::Zero;
use plonk_core::permutation::Permutation;
use plonk_core::prelude::StandardComposer;
use plonk_hashing::poseidon::{
    constants::PoseidonConstants, poseidon_ref::NativeSpecRef,
};

use crate::Variable;
use itertools::Itertools;
use std::thread;

use crate::{
    assert_hash_constraints, assert_hash_constraints_2,
    util::{left_child_index, right_child_index},
    MerkleTree, HEIGHT,
};
use std::sync::{Arc, Mutex};

lazy_static::lazy_static! {
    pub static ref LEAF_NUM: usize = {
        std::env::var("LEAF_NUM")
            .and_then(|v| match v.parse() {
                Ok(val) => Ok(val),
                Err(_) => {
                    println!("Invalid env GPU_IDX! Defaulting to 0...");
                    Ok(8)
                }
            })
            .unwrap_or(8)
    };

    pub static ref PATH_NUM: usize = {
        std::env::var("PATH_NUM")
            .and_then(|v| match v.parse() {
                Ok(val) => Ok(val),
                Err(_) => {
                    println!("Invalid env GPU_IDX! Defaulting to 0...");
                    Ok(8)
                }
            })
            .unwrap_or(8)
    };

}

// Hard coded with poseidon width = 3.
impl MerkleTree<NativeSpecRef<Fr>> {
    /// Generate constraints asserting a merkle tree is generated correctly
    /// This is hard coded for BLS12-381 scalar field
    pub fn gen_constraints(
        &self,
        composer: &mut StandardComposer<Fr, EdwardsParameters>,
        hash_param: &PoseidonConstants<Fr>,
    ) {
        let start = std::time::Instant::now();
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
                    // println!(
                    //     "1-1 index:{}, upper_bound:{}",
                    //     current_index, upper_bound
                    // );

                    // `left_child_index(current_index)` and
                    // `right_child_index(current_index) returns the position of
                    // leaf in the whole tree (represented as a list in level
                    // order). We need to shift it
                    // by `-upper_bound` to get the index in `leaf_nodes` list.
                    let left_leaf_index =
                        left_child_index(current_index) - upper_bound;
                    let right_leaf_index =
                        right_child_index(current_index) - upper_bound;

                    // let start = std::time::Instant::now();

                    // assert hash is valid
                    assert_hash_constraints(
                        composer,
                        hash_param,
                        &leaf_node_vars[left_leaf_index],
                        &leaf_node_vars[right_leaf_index],
                        e,
                    );
                    // println!(
                    //     "assert_hash_constraints1-2, spent:{:?}",
                    //     start.elapsed()
                    // );
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
                    // println!(
                    //     "1-2 index:{}, upper_bound:{}",
                    //     index, upper_bound
                    // );
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
        //composer.perm.sort_wire();

        //println!("gen_constraints 1 spent:{:?}", start.elapsed());
    }

    pub fn gen_constraints_4(
        &self,
        composer: &mut StandardComposer<Fr, EdwardsParameters>,
        hash_param: &PoseidonConstants<Fr>,
        with_permutation: bool,
    ) {
        // let start_all = std::time::Instant::now();
        let reuse_perm = with_permutation;
        // let mut perms = vec![];

        // let mut variables_count = composer.variables.len();
        let mut variables_count = composer.variable_number;

        let leaf_node_vars = self
            .leaf_nodes
            .iter()
            .map(|node| {
                if reuse_perm {
                    composer.add_input_reuse_perm(*node, &mut variables_count)
                } else {
                    composer.add_input(*node)
                }
            })
            .collect::<Vec<_>>();

        let non_leaf_node_vars = self
            .non_leaf_nodes
            .iter()
            .map(|node| {
                if reuse_perm {
                    composer.add_input_reuse_perm(*node, &mut variables_count)
                } else {
                    composer.add_input(*node)
                }
            })
            .collect::<Vec<_>>();

        let composer: &StandardComposer<Fr, EdwardsParameters> = composer;
        let root_node = &non_leaf_node_vars[0];

        // Compute the starting indices for each non-leaf level of the tree
        let mut index = 0;
        let mut level_indices = Vec::with_capacity(HEIGHT - 1);
        for _ in 0..(HEIGHT - 1) {
            level_indices.push(index);
            index = left_child_index(index);
        }

        let start_index = level_indices.pop().unwrap();
        let upper_bound = left_child_index(start_index);
        let mut init_variables_size = composer.variable_number;

        // compute the hash values for the non-leaf bottom layer
        let cpu_kernel_number = *LEAF_NUM;
        let binding = non_leaf_node_vars
            .iter()
            .enumerate()
            .take(upper_bound)
            .skip(start_index)
            .collect_vec();

        assert!(binding.len() >= cpu_kernel_number);
        assert!((binding.len() % cpu_kernel_number) == 0);
        let chunk_size = binding.len() / cpu_kernel_number;
        let chunks = binding.chunks(chunk_size);

        thread::scope(|s| {
            for (chunk_seq, chunk) in chunks.enumerate() {
                let chunk = chunk.clone();
                let leaf_node_vars = &leaf_node_vars;
                // let perm = Arc::new(Mutex::new(Permutation::new()));
                // let perm_clone = perm.clone();
                s.spawn(move || {
                    // let mut perm_lock = perm_clone.lock().unwrap();
                    let mut hash_count = chunk_seq * chunk_size;

                    for (current_index, e) in chunk {
                        let left_leaf_index =
                            left_child_index(*current_index) - upper_bound;
                        let right_leaf_index =
                            right_child_index(*current_index) - upper_bound;

                        let left = &leaf_node_vars[left_leaf_index];
                        let right = &leaf_node_vars[right_leaf_index];

                        // assert hash is valid
                        assert_hash_constraints_2(
                            composer,
                            hash_param,
                            left,
                            right,
                            e,
                            hash_count,
                            init_variables_size,
                            // &mut perm_lock,
                            reuse_perm,
                        );

                        hash_count += 1;
                    }
                });

                //     if !reuse_perm {
                //         perms.push(perm.clone());
                //     }
            }
        });

        // compute the hash values for nodes in every other layer in the tree
        level_indices.reverse();

        let cpu_kernel_number = *PATH_NUM;
        let mut hash_count = binding.len();
        for &start_index in &level_indices {
            // The layer beginning `start_index` ends at `upper_bound`
            // (exclusive).
            let upper_bound = left_child_index(start_index);

            let nodes = non_leaf_node_vars[start_index..upper_bound]
                .iter()
                .enumerate()
                .take(upper_bound)
                .collect_vec();

            let nodes_len = nodes.len();
            let cpu_kernel_number = std::cmp::min(cpu_kernel_number, nodes_len);
            let chunk_size = nodes_len / cpu_kernel_number;
            assert!((nodes_len % cpu_kernel_number) == 0);
            let chunks = nodes.chunks(chunk_size);

            thread::scope(|s| {
                for (chunk_seq, chunk) in chunks.enumerate() {
                    let chunk = chunk.clone();
                    // let perm = Arc::new(Mutex::new(Permutation::new()));
                    // let perm_clone = perm.clone();
                    let non_leaf_node_vars = non_leaf_node_vars.clone();

                    let handle = s.spawn(move || {
                        let mut hash_count_in =
                            hash_count + chunk_seq * chunk_size;

                        // let mut perm_lock = perm_clone.lock().unwrap();
                        for (index, node) in chunk {
                            assert_hash_constraints_2(
                                composer,
                                hash_param,
                                &non_leaf_node_vars
                                    [left_child_index(*index + start_index)],
                                &non_leaf_node_vars
                                    [right_child_index(*index + start_index)],
                                node,
                                hash_count_in,
                                init_variables_size,
                                // &mut perm_lock,
                                reuse_perm,
                            );

                            hash_count_in += 1;
                        }
                    });
                    // if !reuse_perm {
                    //     perms.push(perm.clone());
                    // }
                }
            });
            hash_count += nodes_len;
        }

        let c: *const StandardComposer<Fr, EdwardsParameters> = composer;
        let composer = c as *mut StandardComposer<Fr, EdwardsParameters>;
        let composer = {
            unsafe {
                let mut_ref: &mut StandardComposer<Fr, EdwardsParameters> =
                    &mut *composer;
                mut_ref
            }
        };

        // if !reuse_perm {
        //     for p in (&perms).into_iter() {
        //         let perm_lock = p.lock().unwrap();
        //         for (key, p) in &perm_lock.variable_map {
        //             composer.perm.add_wire_datas(*key, p.clone());
        //         }
        //     }
        // }

        // println!("hash_count:{}", hash_count);sad
        // println!("len:{}", composer.w_l.len());
        // println!("l:{:?}", &composer.w_l);

        let gate_index = 4;
        let mut gate_index = hash_count * 193 + gate_index;

        let variable_index = init_variables_size;
        let mut variable_index = hash_count * 193 + variable_index;
        // constrain the root is correct w.r.t. public input

        let zero = composer.zero_var();
        composer.arithmetic_gate_end(
            |gate| {
                gate.witness(*root_node, zero, Some(zero))
                    .add(Fr::one(), Fr::zero())
                    .pi(-self.non_leaf_nodes[0])
            },
            &mut gate_index,
            &mut variable_index,
            reuse_perm,
        );

        // if !reuse_perm {
        //     composer.perm.sort_wire();
        // }
        // drop(leaf_node_vars);
        // drop(non_leaf_node_vars);
        // mem::drop(v);
        // drop(perms);
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
