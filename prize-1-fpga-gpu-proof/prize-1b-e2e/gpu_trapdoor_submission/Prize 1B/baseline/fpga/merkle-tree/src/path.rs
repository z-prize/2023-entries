use core::fmt;
use std::fmt::Display;

use ark_std::{end_timer, start_timer};
use plonk_hashing::poseidon::{
    constants::PoseidonConstants, poseidon_ref::PoseidonRefSpec,
};

use crate::{hash, HEIGHT};

#[derive(Clone, Debug, PartialEq)]
pub struct Path<S: PoseidonRefSpec<(), 3>> {
    pub(crate) nodes: Vec<(S::Field, S::Field)>, /* left and right nodes */
    pub(crate) index: usize,
}

impl<S: PoseidonRefSpec<(), 3>> Default for Path<S>
where
    S::Field: Default + Copy,
{
    fn default() -> Self {
        Self {
            nodes: [(S::Field::default(), S::Field::default()); HEIGHT - 1]
                .to_vec(),
            index: 0,
        }
    }
}

impl<S: PoseidonRefSpec<(), 3>> Display for Path<S>
where
    S::Field: Display + Default + Copy + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "merkle proof, from root to leaf")?;
        let position_list = self.position_list();
        for ((i, (left, right)), is_right_node) in
            self.nodes.iter().enumerate().zip(position_list)
        {
            if is_right_node {
                writeln!(f, "{}-th node is right node", i)?;
            } else {
                writeln!(f, "{}-th node is left node", i)?;
            }
            writeln!(f, "left    {}\nright   {}", left, right,)?;
        }

        Ok(())
    }
}

impl<S: PoseidonRefSpec<(), 3>> Path<S>
where
    S::Field: Default + Copy + PartialEq,
{
    /// The position of on_path node in `leaf_and_sibling_hash` and
    /// `non_leaf_and_sibling_hash_path`. `position[i]` is 0 (false) iff
    /// `i`th on-path node from top to bottom is on the left.
    ///
    /// This function simply converts `self.leaf_index` to boolean array in big
    /// endian form.
    fn position_list(&'_ self) -> impl '_ + Iterator<Item = bool> {
        (0..self.nodes.len() + 1)
            .map(move |i| ((self.index >> i) & 1) != 0)
            .rev()
    }

    /// verifies the path against a root
    pub fn verify(
        &self,
        param: &PoseidonConstants<S::ParameterField>,
        root: &S::Field,
    ) -> bool {
        let timer = start_timer!(|| "hvc path verify");
        // check that the first two elements hashes to root
        if hash::<S>(param, &self.nodes[0].0, &self.nodes[0].1) != *root {
            println!("hash digest does not match the root");
            return false;
        }
        let position_list: Vec<_> = self.position_list().collect();

        let checks = self
            .nodes
            .clone()
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, (left, right))| {
                if position_list[i] {
                    hash::<S>(param, left, right) == self.nodes[i - 1].1
                } else {
                    hash::<S>(param, left, right) == self.nodes[i - 1].0
                }
            })
            .collect::<Vec<_>>()
            .iter()
            .all(|mk| *mk);

        end_timer!(timer);
        checks
    }
}
