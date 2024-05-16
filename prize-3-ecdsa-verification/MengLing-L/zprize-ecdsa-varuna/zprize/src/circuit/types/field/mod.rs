pub mod field;
pub mod not;

use super::*;

pub static SMALL_FIELD_MODULUS_MINUS_ONE: u16 = 65535;
pub static SMALL_FIELD_SIZE_IN_BITS: usize = 16;
pub static NUMBER_OF_PADDED_ZERO: usize = 2;
pub static PADDED_CHUNK_SIZE: usize = NUMBER_OF_PADDED_ZERO + 1;
pub static PADDED_FIELD_SIZE: usize = SMALL_FIELD_SIZE_IN_BITS * PADDED_CHUNK_SIZE;
