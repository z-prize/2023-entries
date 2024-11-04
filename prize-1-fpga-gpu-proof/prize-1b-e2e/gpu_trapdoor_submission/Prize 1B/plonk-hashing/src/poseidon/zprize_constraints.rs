//! Submission for ZPrize hash track
//! Author: Zhenfei Zhang
//! All rights reserved for now.
//! Correct, Naive, reference implementation of Poseidon hash function.

use crate::poseidon::PoseidonError;

use crate::poseidon::constants::PoseidonConstants;
use ark_ec::TEModelParameters;
use ark_ff::PrimeField;
use core::marker::PhantomData;
use plonk_core::permutation::Permutation;
use plonk_core::{
    constraint_system::StandardComposer,
    prelude::{self as plonk, Variable},
};

use super::poseidon_ref::PoseidonRefSpec;
use derivative::Derivative;

#[derive(Derivative)]
#[derivative(Debug(bound = ""))]
pub struct PoseidonZZRef<
    COM,
    S: PoseidonRefSpec<COM, WIDTH>,
    const WIDTH: usize,
> where
    S: ?Sized,
{
    pub(crate) constants_offset: usize,
    pub(crate) current_round: usize,
    pub elements: [S::Field; WIDTH],
    pos: usize,
    pub(crate) constants: PoseidonConstants<S::ParameterField>,
}

impl<COM, S: PoseidonRefSpec<COM, WIDTH>, const WIDTH: usize>
    PoseidonZZRef<COM, S, WIDTH>
{
    pub fn new(
        c: &mut COM,
        constants: PoseidonConstants<S::ParameterField>,
    ) -> Self {
        let mut elements = S::zeros(c);
        elements[0] = S::alloc(c, constants.domain_tag);
        PoseidonZZRef {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: 1,
            constants,
        }
    }

    pub fn new_2(
        c: &mut COM,
        constants: PoseidonConstants<S::ParameterField>,
        gate_index: &mut usize,
    ) -> Self {
        let mut elements: [<S as PoseidonRefSpec<COM, WIDTH>>::Field; WIDTH] =
            S::zeros(c);
        elements[0] = S::alloc_2(c, constants.domain_tag, gate_index);
        PoseidonZZRef {
            constants_offset: 0,
            current_round: 0,
            elements,
            pos: 1,
            constants,
        }
    }

    pub fn arity(&self) -> usize {
        WIDTH - 1
    }

    pub fn reset(&mut self, c: &mut COM) {
        self.constants_offset = 0;
        self.current_round = 0;
        self.elements[1..].iter_mut().for_each(|l| *l = S::zero(c));
        self.elements[0] = S::alloc(c, self.constants.domain_tag);
        self.pos = 1;
    }

    /// input one field element to Poseidon. Return the position of the element
    /// in state.
    pub fn input(&mut self, input: S::Field) -> Result<usize, PoseidonError> {
        // Cannot input more elements than the defined constant width
        if self.pos >= WIDTH {
            return Err(PoseidonError::FullBuffer);
        }

        // Set current element, and increase the pointer
        self.elements[self.pos] = input;
        self.pos += 1;

        Ok(self.pos - 1)
    }

    pub fn output_hash(&mut self, c: &mut COM) -> S::Field {
        S::full_round(
            c,
            &self.constants,
            &mut self.constants_offset,
            &mut self.elements,
        );

        // println!(
        //     "1-1 output-hash spent :{:?}, id:{:?}",
        //     start.elapsed(), main_thread_id
        // );

        for _ in 1..self.constants.half_full_rounds {
            S::full_round(
                c,
                &self.constants,
                &mut self.constants_offset,
                &mut self.elements,
            );
        }

        // println!("1-2 self.elements[1] :{:?}, id:{:?}",  self.elements[1],
        // main_thread_id);

        S::partial_round(
            c,
            &self.constants,
            &mut self.constants_offset,
            &mut self.elements,
        );

        // println!("1-3 self.elements[1] :{:?}, id:{:?}",  self.elements[1],
        // main_thread_id);

        for _ in 1..self.constants.partial_rounds {
            S::partial_round(
                c,
                &self.constants,
                &mut self.constants_offset,
                &mut self.elements,
            );
        }

        // println!("1-4 self.elements[1] :{:?}, id:{:?}",  self.elements[1],
        // main_thread_id);

        for _ in 0..self.constants.half_full_rounds {
            S::full_round(
                c,
                &self.constants,
                &mut self.constants_offset,
                &mut self.elements,
            )
        }
        // println!("1-5 self.elements[1] :{:?}, id:{:?}",  self.elements[1],
        // main_thread_id);

        self.elements[1].clone()
    }

    /// Output the hash
    pub fn output_hash_2(
        &mut self,
        c: &mut COM,
        gate_index: &mut usize,
        variable_index: &mut usize,
        // perm: &mut Permutation,
        reuse_perm: bool,
    ) -> S::Field {
        S::full_round_2(
            c,
            &self.constants,
            &mut self.constants_offset,
            &mut self.elements,
            gate_index,
            variable_index,
            // perm,
            reuse_perm,
        );

        for _ in 1..self.constants.half_full_rounds {
            S::full_round_2(
                c,
                &self.constants,
                &mut self.constants_offset,
                &mut self.elements,
                gate_index,
                variable_index,
                // perm,
                reuse_perm,
            );
        }

        S::partial_round_2(
            c,
            &self.constants,
            &mut self.constants_offset,
            &mut self.elements,
            gate_index,
            variable_index,
            // perm,
            reuse_perm,
        );

        for _ in 1..self.constants.partial_rounds {
            S::partial_round_2(
                c,
                &self.constants,
                &mut self.constants_offset,
                &mut self.elements,
                gate_index,
                variable_index,
                // perm,
                reuse_perm,
            );
        }

        for _ in 0..self.constants.half_full_rounds {
            S::full_round_2(
                c,
                &self.constants,
                &mut self.constants_offset,
                &mut self.elements,
                gate_index,
                variable_index,
                // perm,
                reuse_perm,
            )
        }
        self.elements[1].clone()
    }
}
use ark_serialize::*;
#[derive(CanonicalDeserialize, CanonicalSerialize)]
pub struct PlonkSpecZZ<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<
        F: PrimeField,
        P: TEModelParameters<BaseField = F>,
        const WIDTH: usize,
    > PoseidonRefSpec<StandardComposer<F, P>, WIDTH> for PlonkSpecZZ<F>
{
    type Field = Variable;
    type ParameterField = F;

    fn full_round(
        c: &mut StandardComposer<F, P>,
        constants: &PoseidonConstants<Self::ParameterField>,
        constants_offset: &mut usize,
        state: &mut [Self::Field; WIDTH],
    ) {
        let pre_round_keys = constants
            .round_constants
            .iter()
            .skip(*constants_offset)
            .collect::<Vec<_>>();
        let mut res = *state;
        if *constants_offset == 0 {
            // first round
            res[0] = <Self as PoseidonRefSpec<_, WIDTH>>::addi(
                c,
                &res[0],
                pre_round_keys[0],
            );

            res[1] = <Self as PoseidonRefSpec<_, WIDTH>>::addi(
                c,
                &res[1],
                pre_round_keys[1],
            );
            res[2] = <Self as PoseidonRefSpec<_, WIDTH>>::addi(
                c,
                &res[2],
                pre_round_keys[2],
            );
        }

        let zero = F::zero();
        let current_round_key = if pre_round_keys.len() == 3 {
            // Last round
            (&zero, &zero, &zero)
        } else {
            (pre_round_keys[3], pre_round_keys[4], pre_round_keys[5])
        };

        let matrix = &constants.mds_matrices.m.iter_rows().collect::<Vec<_>>();

        state[0] = c.full_affine_transform_gate(
            &[res[0], res[1], res[2]],
            &[
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                *current_round_key.0,
                -F::one(),
            ],
        );
        state[1] = c.full_affine_transform_gate(
            &[res[0], res[1], res[2]],
            &[
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                *current_round_key.1,
                -F::one(),
            ],
        );

        state[2] = c.full_affine_transform_gate(
            &[res[0], res[1], res[2]],
            &[
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
                *current_round_key.2,
                -F::one(),
            ],
        );

        *constants_offset += WIDTH;
    }

    fn full_round_2(
        c: &mut StandardComposer<F, P>,
        constants: &PoseidonConstants<Self::ParameterField>,
        constants_offset: &mut usize,
        state: &mut [Self::Field; WIDTH],
        index: &mut usize,
        variable_index: &mut usize,
        // perm: &mut Permutation,
        reuse_perm: bool,
    ) {
        let pre_round_keys = constants
            .round_constants
            .iter()
            .skip(*constants_offset)
            .collect::<Vec<_>>();

        let mut res = *state;
        if *constants_offset == 0 {
            // first round
            res[0] = <Self as PoseidonRefSpec<_, WIDTH>>::addi_2(
                c,
                &res[0],
                pre_round_keys[0],
                index,
                variable_index,
                // perm,
                reuse_perm,
            );
            res[1] = <Self as PoseidonRefSpec<_, WIDTH>>::addi_2(
                c,
                &res[1],
                pre_round_keys[1],
                index,
                variable_index,
                // perm,
                reuse_perm,
            );
            res[2] = <Self as PoseidonRefSpec<_, WIDTH>>::addi_2(
                c,
                &res[2],
                pre_round_keys[2],
                index,
                variable_index,
                // perm,
                reuse_perm,
            );
        }

        let zero = F::zero();
        let current_round_key = if pre_round_keys.len() == 3 {
            // Last round
            (&zero, &zero, &zero)
        } else {
            (pre_round_keys[3], pre_round_keys[4], pre_round_keys[5])
        };

        let matrix = &constants.mds_matrices.m.iter_rows().collect::<Vec<_>>();

        state[0] = c.full_affine_transform_gate_2(
            &[res[0], res[1], res[2]],
            &[
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                *current_round_key.0,
                -F::one(),
            ],
            index,
            variable_index,
            // perm,
            reuse_perm,
        );

        state[1] = c.full_affine_transform_gate_2(
            &[res[0], res[1], res[2]],
            &[
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                *current_round_key.1,
                -F::one(),
            ],
            index,
            variable_index,
            // perm,
            reuse_perm,
        );

        state[2] = c.full_affine_transform_gate_2(
            &[res[0], res[1], res[2]],
            &[
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
                *current_round_key.2,
                -F::one(),
            ],
            index,
            variable_index,
            // perm,
            reuse_perm,
        );

        *constants_offset += WIDTH;
    }

    fn partial_round(
        c: &mut StandardComposer<F, P>,
        constants: &PoseidonConstants<Self::ParameterField>,
        constants_offset: &mut usize,
        state: &mut [Self::Field; WIDTH],
    ) {
        let pre_round_keys = constants
            .round_constants
            .iter()
            .skip(*constants_offset)
            .collect::<Vec<_>>();

        let res = *state;
        let matrix = &constants.mds_matrices.m.iter_rows().collect::<Vec<_>>();

        state[0] = c.partial_affine_transform_gate(
            &[res[0], res[1], res[2]],
            &[
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                *pre_round_keys[3],
                -F::one(),
            ],
        );
        state[1] = c.partial_affine_transform_gate(
            &[res[0], res[1], res[2]],
            &[
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                *pre_round_keys[4],
                -F::one(),
            ],
        );
        state[2] = c.partial_affine_transform_gate(
            &[res[0], res[1], res[2]],
            &[
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
                *pre_round_keys[5],
                -F::one(),
            ],
        );
        *constants_offset += WIDTH;
    }

    fn partial_round_2(
        c: &mut StandardComposer<F, P>,
        constants: &PoseidonConstants<Self::ParameterField>,
        constants_offset: &mut usize,
        state: &mut [Self::Field; WIDTH],
        index: &mut usize,
        variable_index: &mut usize,
        // perm: &mut Permutation,
        reuse_perm: bool,
    ) {
        let pre_round_keys = constants
            .round_constants
            .iter()
            .skip(*constants_offset)
            .collect::<Vec<_>>();

        let res = *state;
        let matrix = &constants.mds_matrices.m.iter_rows().collect::<Vec<_>>();

        state[0] = c.partial_affine_transform_gate_2(
            &[res[0], res[1], res[2]],
            &[
                matrix[0][0],
                matrix[0][1],
                matrix[0][2],
                *pre_round_keys[3],
                -F::one(),
            ],
            index,
            variable_index,
            // perm,
            reuse_perm,
        );

        state[1] = c.partial_affine_transform_gate_2(
            &[res[0], res[1], res[2]],
            &[
                matrix[1][0],
                matrix[1][1],
                matrix[1][2],
                *pre_round_keys[4],
                -F::one(),
            ],
            index,
            variable_index,
            // perm,
            reuse_perm,
        );

        state[2] = c.partial_affine_transform_gate_2(
            &[res[0], res[1], res[2]],
            &[
                matrix[2][0],
                matrix[2][1],
                matrix[2][2],
                *pre_round_keys[5],
                -F::one(),
            ],
            index,
            variable_index,
            // perm,
            reuse_perm,
        );
        *constants_offset += WIDTH;
    }

    fn alloc(
        c: &mut StandardComposer<F, P>,
        v: Self::ParameterField,
    ) -> Self::Field {
        c.add_input(v)
    }

    fn alloc_2(
        c: &mut StandardComposer<F, P>,
        v: Self::ParameterField,
        gate_index: &mut usize,
    ) -> Self::Field {
        c.add_input_2(v, gate_index, false)
    }

    fn zeros<const W: usize>(
        c: &mut StandardComposer<F, P>,
    ) -> [Self::Field; W] {
        [c.zero_var(); W]
    }

    fn add(
        c: &mut StandardComposer<F, P>,
        x: &Self::Field,
        y: &Self::Field,
    ) -> Self::Field {
        c.arithmetic_gate(|g| g.witness(*x, *y, None).add(F::one(), F::one()))
    }

    fn addi(
        c: &mut StandardComposer<F, P>,
        a: &Self::Field,
        b: &Self::ParameterField,
    ) -> Self::Field {
        let zero = c.zero_var();
        c.arithmetic_gate(|g| {
            g.witness(*a, zero, None)
                .add(F::one(), F::zero())
                .constant(*b)
        })
    }

    fn addi_2(
        c: &mut StandardComposer<F, P>,
        a: &Self::Field,
        b: &Self::ParameterField,
        gate_index: &mut usize,
        variable_index: &mut usize,
        // perm: &mut Permutation,
        reuse_perm: bool,
    ) -> Self::Field {
        let zero = c.zero_var();
        c.arithmetic_gate_2(
            |g| {
                g.witness(*a, zero, None)
                    .add(F::one(), F::zero())
                    .constant(*b)
            },
            gate_index,
            variable_index,
            // perm,
            reuse_perm,
        )
    }

    fn mul(
        _c: &mut StandardComposer<F, P>,
        _x: &Self::Field,
        _y: &Self::Field,
    ) -> Self::Field {
        unimplemented!()
    }

    fn muli(
        _c: &mut StandardComposer<F, P>,
        _x: &Self::Field,
        _y: &Self::ParameterField,
    ) -> Self::Field {
        unimplemented!()
    }
}
use ark_serialize::*;
#[derive(CanonicalDeserialize, CanonicalSerialize)]
pub struct PlonkSpecRef;

impl<F, P, const WIDTH: usize>
    PoseidonRefSpec<plonk::StandardComposer<F, P>, WIDTH> for PlonkSpecRef
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
{
    type Field = plonk::Variable;
    type ParameterField = F;

    fn alloc(
        c: &mut StandardComposer<F, P>,
        v: Self::ParameterField,
    ) -> Self::Field {
        c.add_input(v)
    }

    fn alloc_2(
        c: &mut StandardComposer<F, P>,
        v: Self::ParameterField,
        _: &mut usize,
    ) -> Self::Field {
        unimplemented!()
    }

    fn zeros<const W: usize>(
        c: &mut StandardComposer<F, P>,
    ) -> [Self::Field; W] {
        [c.zero_var(); W]
    }

    fn add(
        _c: &mut StandardComposer<F, P>,
        _x: &Self::Field,
        _y: &Self::Field,
    ) -> Self::Field {
        unimplemented!()
    }

    fn addi(
        _c: &mut StandardComposer<F, P>,
        _a: &Self::Field,
        _b: &Self::ParameterField,
    ) -> Self::Field {
        unimplemented!()
    }
    fn addi_2(
        _c: &mut StandardComposer<F, P>,
        _a: &Self::Field,
        _b: &Self::ParameterField,
        _index: &mut usize,
        _variable_index: &mut usize,
        // _perm: &mut Permutation,
        _reuse_perm: bool,
    ) -> Self::Field {
        unimplemented!()
    }

    fn mul(
        _c: &mut StandardComposer<F, P>,
        _x: &Self::Field,
        _y: &Self::Field,
    ) -> Self::Field {
        unimplemented!()
    }

    fn muli(
        _c: &mut StandardComposer<F, P>,
        _x: &Self::Field,
        _y: &Self::ParameterField,
    ) -> Self::Field {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use ark_ed_on_bls12_381::EdwardsParameters;
    use ark_ff::UniformRand;
    use ark_std::test_rng;
    use plonk_core::prelude::StandardComposer;

    use crate::poseidon::{
        constants::PoseidonConstants,
        poseidon_ref::{NativeSpecRef, PoseidonRef},
        zprize_constraints::{PlonkSpecZZ, PoseidonZZRef},
    };

    type Composer = StandardComposer<Fr, EdwardsParameters>;

    #[test]
    fn test_poseidon_constraints() {
        let mut rng = test_rng();

        let param = PoseidonConstants::generate::<3>();
        let mut poseidon = PoseidonRef::<(), NativeSpecRef<Fr>, 3>::new(
            &mut (),
            param.clone(),
        );
        let inputs = (0..2).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        inputs.iter().for_each(|x| {
            let _ = poseidon.input(*x).unwrap();
        });
        let mut poseidon = PoseidonRef::<(), NativeSpecRef<Fr>, 3>::new(
            &mut (),
            param.clone(),
        );

        inputs.iter().for_each(|x| {
            let _ = poseidon.input(*x).unwrap();
        });
        let native_hash: Fr = poseidon.output_hash(&mut ());

        let mut composer = Composer::new();
        let mut hasher = PoseidonZZRef::<_, PlonkSpecZZ<Fr>, 3>::new(
            &mut composer,
            param.clone(),
        );

        inputs.iter().for_each(|x| {
            let var = composer.add_input(*x);
            let _ = hasher.input(var).unwrap();
        });

        for e in hasher.elements {
            println!("{:?} {}", e, composer.value_of_var(e));
        }

        let output = hasher.output_hash(&mut composer);

        assert_eq!(native_hash, composer.value_of_var(output));

        println!("{} {}", param.partial_rounds, param.full_rounds);
        println!("#constraints: {}", composer.total_size());
    }
}
