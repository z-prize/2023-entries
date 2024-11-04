use ark_bls12_381::Fr;
use ark_ed_on_bls12_381::EdwardsParameters;
use plonk_core::prelude::{StandardComposer, Variable};
use plonk_hashing::poseidon::{
    constants::PoseidonConstants,
    poseidon_ref::{NativeSpecRef, PoseidonRef, PoseidonRefSpec},
    zprize_constraints::{PlonkSpecZZ, PoseidonZZRef},
};

mod circuit;
mod constraints;
mod path;
mod tree;
mod util;

pub use circuit::MerkleTreeCircuit;
pub use path::Path;
pub use tree::MerkleTree;

pub const HEIGHT: usize = 15;

pub type PoseidonHash = PoseidonRef<(), NativeSpecRef<Fr>, 3>;

#[inline]
pub(crate) fn hash<S: PoseidonRefSpec<(), 3>>(
    param: &PoseidonConstants<S::ParameterField>,
    left: &S::Field,
    right: &S::Field,
) -> S::Field
where
    S::Field: Default + Copy,
{
    let mut poseidon = PoseidonRef::<(), S, 3>::new(&mut (), param.clone());
    let _ = poseidon.input(*left);
    let _ = poseidon.input(*right);
    // pad the last input to stop extension attacks
    let _ = poseidon.input(S::Field::default());
    poseidon.output_hash(&mut ())
}

#[inline]
pub(crate) fn assert_hash_constraints(
    composer: &mut StandardComposer<Fr, EdwardsParameters>,
    param: &PoseidonConstants<Fr>,
    left: &Variable,
    right: &Variable,
    output: &Variable,
) {
    let mut poseidon =
        PoseidonZZRef::<_, PlonkSpecZZ<Fr>, 3>::new(composer, param.clone());

    let _ = poseidon.input(*left);
    let _ = poseidon.input(*right);
    // pad the last input to stop extension attacks
    let _ = poseidon.input(composer.zero_var());

    let output_rec = poseidon.output_hash(composer);

    composer.assert_equal(*output, output_rec);
}

