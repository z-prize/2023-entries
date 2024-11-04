mod engine;

use ark_ec::AffineCurve;
use ark_ff::PrimeField;

pub use engine::Engine;
pub use engine::Engine12377;
pub use engine::Engine12381;

pub fn create_engine<G: AffineCurve>(element_count: usize, xclbin_dir: &str) -> Box<dyn Engine<G>>
    where
        G::BaseField: PrimeField
{
    return create_engine_ext::<G>(20, element_count, xclbin_dir);
}

pub fn create_engine_ext<G: AffineCurve>(round_pow: usize, element_count: usize, xclbin_dir: &str) -> Box<dyn Engine<G>>
    where
        G::BaseField: PrimeField
{
    match <G::BaseField as PrimeField>::size_in_bits() {
        377 => {
            let engine = Box::new(Engine12377::new(round_pow, element_count, xclbin_dir));
            let e2: Box<dyn Engine<ark_bls12_377::G1Affine>> = Box::new(*engine);
            unsafe {Box::from_raw(Box::into_raw(e2) as *mut dyn Engine<G>)}
        },
        381 => {
            let engine = Box::new(Engine12381::new(round_pow, element_count, xclbin_dir));
            let e2: Box<dyn Engine<ark_bls12_381::G1Affine>> = Box::new(*engine);
            unsafe {Box::from_raw(Box::into_raw(e2) as *mut dyn Engine<G>)}
        }
        _ => panic!("not supported"),
    }
}
