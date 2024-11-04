// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Permutations

pub(crate) mod constants;

use ark_poly::{
    domain::{EvaluationDomain, GeneralEvaluationDomain},
    univariate::DensePolynomial,
    UVPolynomial,
};
use constants::*;
use hashbrown::HashMap;
use itertools::izip;

use crate::{
    constraint_system::{Variable, WireData},
    util::EvaluationDomainExt,
};
use ark_bls12_381::G1Affine;
use ark_ec::AffineCurve;
use ark_ff::{BigInteger256, FftField};

use ec_gpu_common::DeviceMemory;
use ec_gpu_common::{
    log2_floor, product_argument_test, wires_to_single_gate, Fr as GpuFr,
    GPUSourceCore, GpuPoly, GpuPolyContainer, MSMContext, MsmPrecalcContainer,
    MultiexpKernel, PolyKernel, GPU_CUDA_CORES,
};

use std::{any::type_name, sync::Mutex};
// use ref_thread_local::RefThreadLocal;

lazy_static::lazy_static! {
    pub static ref GPU_KERN_IDX: usize = {
        std::env::var("GPU_IDX")
            .and_then(|v| match v.parse() {
                Ok(val) => Ok(val),
                Err(_) => {
                    println!("Invalid env GPU_IDX! Defaulting to 0...");
                    Ok(0)
                }
            })
            .unwrap_or(0)
    };

    pub static ref PRECALC_CONTAINER:MsmPrecalcContainer = {
        MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[*GPU_KERN_IDX], true).unwrap()
    };

    pub static ref  GPU_POLY_KERNEL: PolyKernel<'static, GpuFr> = {
        PolyKernel::<GpuFr>::create_with_core(&GPU_CUDA_CORES[*GPU_KERN_IDX]).unwrap()
    };

    pub static ref GPU_POLY_CONTAINER: Mutex<GpuPolyContainer<GpuFr>> = {
        Mutex::new(GpuPolyContainer::<GpuFr>::create().unwrap())
    };

    pub static ref MSM_KERN : MultiexpKernel<'static, 'static, G1Affine> = {
        MultiexpKernel::<G1Affine>::create(*GPU_KERN_IDX, &PRECALC_CONTAINER)
            .unwrap()
    };
}

// ref_thread_local::ref_thread_local! {
//     /// thread
//     // pub static managed GPU_POLY_KERNEL: PolyKernel<'static, GpuFr> = {
//     //
// PolyKernel::<GpuFr>::create_with_core(&GPU_CUDA_CORES[*GPU_KERN_IDX]).
// unwrap()     // };

//     /// thread
//     pub static managed GPU_POLY_CONTAINER: GpuPolyContainer<GpuFr> = {
//         GpuPolyContainer::<GpuFr>::create().unwrap()
//     };
// }

fn print_type_of<T>(_: &T) {
    println!(" type: {:#?}", std::any::type_name::<T>())
}

pub fn gpu_ifft_without_back_orig_data<F: FftField>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    scalar: &Vec<F>,
    name: &str,
    n: usize,
    size_inv: F,
) {
    let lgn = log2_floor(n);
    let gpu_poly = gpu_container.find(&kern, name);
    let mut gpu_poly = match gpu_poly {
        Ok(poly) => poly,
        Err(_) => {
            let mut ret = gpu_container.ask_for(&kern, n).unwrap();
            ret
        }
    };

    let orig_name = name.to_owned() + "_orig";
    let orig_name = orig_name.as_str();
    let gpu_poly_orig = gpu_container.find(&kern, orig_name);
    let mut gpu_poly_orig = match gpu_poly_orig {
        Ok(poly) => poly,
        Err(_) => {
            let mut ret = gpu_container.ask_for(&kern, n).unwrap();
            ret
        }
    };

    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    let ipq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq_ifft"))
        .unwrap();
    let iomegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas_ifft"))
        .unwrap();

    gpu_poly.fill_with_fe(&F::zero()).unwrap();
    tmp_buf.fill_with_fe(&F::zero()).unwrap();
    gpu_poly.read_from(scalar).unwrap();

    gpu_poly_orig.copy_from_gpu(&gpu_poly).unwrap();
    //gpu_poly.ifft_full(&size_inv).unwrap();
    gpu_poly
        .ifft(&mut tmp_buf, &ipq_buf, &iomegas_buf, &size_inv, lgn)
        .unwrap();

    gpu_container.save(name, gpu_poly).unwrap();
    gpu_container
        .save(&format!("domain_{n}_pq_ifft"), ipq_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_omegas_ifft"), iomegas_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
        .unwrap();

    gpu_container.save(orig_name, gpu_poly_orig).unwrap();
}

pub fn gpu_ifft_without_back<F: FftField>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    scalar: &Vec<F>,
    name: &str,
    n: usize,
    size_inv: F,
) {
    let lgn = log2_floor(n);
    let gpu_poly = gpu_container.find(&kern, name);
    let mut gpu_poly = match gpu_poly {
        Ok(poly) => poly,
        Err(_) => {
            let mut ret = gpu_container.ask_for(&kern, n).unwrap();
            ret
        }
    };
    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    let ipq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq_ifft"))
        .unwrap();
    let iomegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas_ifft"))
        .unwrap();

    gpu_poly.fill_with_fe(&F::zero()).unwrap();
    tmp_buf.fill_with_fe(&F::zero()).unwrap();
    gpu_poly.read_from(scalar).unwrap();
    //gpu_poly.ifft_full(&size_inv).unwrap();
    gpu_poly
        .ifft(&mut tmp_buf, &ipq_buf, &iomegas_buf, &size_inv, lgn)
        .unwrap();

    gpu_container.save(name, gpu_poly).unwrap();
    gpu_container
        .save(&format!("domain_{n}_pq_ifft"), ipq_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_omegas_ifft"), iomegas_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
        .unwrap();
}

pub fn gpu_ifft_without_back_gscalars<F: FftField>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    gpu_poly: &mut GpuPoly<GpuFr>,
    name: &str,
    n: usize,
    size_inv: F,
) {
    let lgn = log2_floor(n);
    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    let ipq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq_ifft"))
        .unwrap();
    let iomegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas_ifft"))
        .unwrap();

    tmp_buf.fill_with_fe(&F::zero()).unwrap();
    gpu_poly
        .ifft(&mut tmp_buf, &ipq_buf, &iomegas_buf, &size_inv, lgn)
        .unwrap();

    gpu_container
        .save(&format!("domain_{n}_pq_ifft"), ipq_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_omegas_ifft"), iomegas_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
        .unwrap();
}

pub fn gpu_ifft_without_back_noread<F: FftField>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    name: &str,
    n: usize,
    size_inv: F,
) {
    let lgn = log2_floor(n);
    let mut gpu_poly = gpu_container.find(&kern, name).unwrap();

    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    let ipq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq_ifft"))
        .unwrap();
    let iomegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas_ifft"))
        .unwrap();

    tmp_buf.fill_with_fe(&F::zero()).unwrap();
    //gpu_poly.ifft_full(&size_inv).unwrap();
    gpu_poly
        .ifft(&mut tmp_buf, &ipq_buf, &iomegas_buf, &size_inv, lgn)
        .unwrap();

    gpu_container.save(name, gpu_poly).unwrap();
    gpu_container
        .save(&format!("domain_{n}_pq_ifft"), ipq_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_omegas_ifft"), iomegas_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
        .unwrap();
}

/// gpu_ifft
// pub fn gpu_fft<F: FftField>(
//     kern: &PolyKernel<GpuFr>,
//     gpu_container: &mut GpuPolyContainer<GpuFr>,
//     scalar: &Vec<F>,
//     name: &str,
//     n: usize,
//     _omega: F,
// ) -> DensePolynomial<F> {
//     let start = std::time::Instant::now();

//     let lgn = log2_floor(n);
//     let gpu_poly = gpu_container.find(&kern, name);
//     let mut gpu_poly = match gpu_poly {
//         Ok(poly) => poly,
//         Err(_) => {
//             let mut ret = gpu_container.ask_for(&kern, n).unwrap();
//             ret.fill_with_zero().unwrap();
//             ret
//         }
//     };

//     let mut tmp_buf = gpu_container
//         .find(&kern, &format!("domain_{n}_tmp_buf"))
//         .unwrap();
//     let pq_buf = gpu_container
//         .find(&kern, &format!("domain_{n}_pq"))
//         .unwrap();
//     let omegas_buf = gpu_container
//         .find(&kern, &format!("domain_{n}_omegas"))
//         .unwrap();

//     tmp_buf.fill_with_fe(&F::zero()).unwrap();
//     gpu_poly.read_from(scalar).unwrap();
//     //gpu_poly.fft_full(&omega).unwrap();
//     gpu_poly
//         .fft(&mut tmp_buf, &pq_buf, &omegas_buf, lgn)
//         .unwrap();

//     let mut gpu_fft_out = vec![F::zero(); n];

//     gpu_poly.write_to(&mut gpu_fft_out).unwrap();

//     let gpu_fft_out = DensePolynomial::from_coefficients_vec(gpu_fft_out);

//     gpu_container.save(name, gpu_poly).unwrap();
//     gpu_container
//         .save(&format!("domain_{n}_pq"), pq_buf)
//         .unwrap();
//     gpu_container
//         .save(&format!("domain_{n}_omegas"), omegas_buf)
//         .unwrap();
//     gpu_container
//         .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
//         .unwrap();
//     // println!("fft cpy spent:{:?}", start.elapsed());

//     gpu_fft_out
// }

// gpu coset fft
pub fn gpu_coset_fft<F: FftField>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    scalar: &Vec<F>,
    name: &str,
    n: usize,
) {
    let start = std::time::Instant::now();

    let lgn = log2_floor(n);
    let gpu_poly = gpu_container.find(&kern, name);
    let mut gpu_poly = match gpu_poly {
        Ok(poly) => poly,
        Err(_) => {
            let mut ret = gpu_container.ask_for(&kern, n).unwrap();
            ret.fill_with_zero().unwrap();
            ret
        }
    };

    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    let pq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq"))
        .unwrap();
    let omegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas"))
        .unwrap();
    let coset_powers_buf = gpu_container
        .find(&kern, &format!("domain_{n}_coset_powers"))
        .unwrap();

    tmp_buf.fill_with_fe(&F::zero()).unwrap();
    gpu_poly.read_from(scalar).unwrap();
    //gpu_poly.fft_full(&omega).unwrap();
    gpu_poly.mul_assign(&coset_powers_buf).unwrap();
    gpu_poly
        .fft(&mut tmp_buf, &pq_buf, &omegas_buf, lgn)
        .unwrap();

    gpu_container.save(name, gpu_poly).unwrap();
    gpu_container
        .save(&format!("domain_{n}_coset_powers"), coset_powers_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_pq"), pq_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_omegas"), omegas_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
        .unwrap();

    // println!("coset fft spent:{:?}", start.elapsed());
}

// gpu coset fft
pub fn gpu_coset_fft_gscalar(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    scalar: &GpuPoly<GpuFr>,
    name: &str,
    n: usize,
) {
    let start = std::time::Instant::now();

    let lgn = log2_floor(n);
    let gpu_poly = gpu_container.find(&kern, name);
    let mut gpu_poly = match gpu_poly {
        Ok(poly) => poly,
        Err(_) => {
            let mut ret = gpu_container.ask_for(&kern, n).unwrap();
            ret
        }
    };

    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    let pq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq"))
        .unwrap();
    let omegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas"))
        .unwrap();
    let coset_powers_buf = gpu_container
        .find(&kern, &format!("domain_{n}_coset_powers"))
        .unwrap();

    tmp_buf.fill_with_zero().unwrap();
    gpu_poly.fill_with_zero().unwrap();
    gpu_poly.copy_from_gpu(scalar).unwrap();
    //gpu_poly.fft_full(&omega).unwrap();
    gpu_poly.mul_assign(&coset_powers_buf).unwrap();
    gpu_poly
        .fft(&mut tmp_buf, &pq_buf, &omegas_buf, lgn)
        .unwrap();

    gpu_container.save(name, gpu_poly).unwrap();
    gpu_container
        .save(&format!("domain_{n}_coset_powers"), coset_powers_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_pq"), pq_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_omegas"), omegas_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
        .unwrap();

    // println!("coset fft spent:{:?}", start.elapsed());
}

// gpu coset ifft
pub fn gpu_coset_ifft_gscalar<F: FftField>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    scalar: &GpuPoly<GpuFr>,
    name: &str,
    n: usize,
    size_inv: F,
) {
    let start = std::time::Instant::now();
    let lgn = log2_floor(n);
    let gpu_poly = gpu_container.find(&kern, name);
    let mut gpu_poly = match gpu_poly {
        Ok(poly) => poly,
        Err(_) => {
            let mut ret = gpu_container.ask_for(&kern, n).unwrap();
            ret.fill_with_zero().unwrap();
            ret
        }
    };

    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    let ipq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq_ifft"))
        .unwrap();
    let iomegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas_ifft"))
        .unwrap();
    let coset_powers_buf = gpu_container
        .find(&kern, &format!("domain_{n}_coset_powers_ifft"))
        .unwrap();

    tmp_buf.fill_with_fe(&F::zero()).unwrap();
    gpu_poly.copy_from_gpu(scalar).unwrap();
    gpu_poly
        .ifft(&mut tmp_buf, &ipq_buf, &iomegas_buf, &size_inv, lgn)
        .unwrap();
    gpu_poly.mul_assign(&coset_powers_buf).unwrap();

    // let mut gpu_fft_out = vec![F::zero(); n];

    // gpu_poly.write_to(&mut gpu_fft_out).unwrap();

    gpu_container.save(name, gpu_poly).unwrap();
    gpu_container
        .save(&format!("domain_{n}_coset_powers_ifft"), coset_powers_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_pq_ifft"), ipq_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_omegas_ifft"), iomegas_buf)
        .unwrap();
    gpu_container
        .save(&format!("domain_{n}_tmp_buf"), tmp_buf)
        .unwrap();
    // println!("coset i fft spent:{:?}", start.elapsed());
}

/// Permutation provides the necessary state information and functions
/// to create the permutation polynomial. In the literature, Z(X) is the
/// "accumulator", this is what this codebase calls the permutation polynomial.
#[derive(derivative::Derivative)]
#[derivative(Clone, Debug)]
pub struct Permutation {
    /// Maps a variable to the wires that it is associated to.
    pub variable_map: HashMap<Variable, Vec<WireData>>,
}

impl Permutation {
    /// Creates a Permutation struct with an expected capacity of zero.
    pub fn new() -> Self {
        let ret = Permutation::with_capacity(0);
        ret
    }

    /// Creates a Permutation struct with an expected capacity of `n`.
    pub fn with_capacity(expected_size: usize) -> Self {
        Self {
            variable_map: HashMap::with_capacity(expected_size),
        }
    }

    /// Creates a new [`Variable`] by incrementing the index of the
    /// `variable_map`. This is correct as whenever we add a new [`Variable`]
    /// into the system It is always allocated in the `variable_map`.
    pub fn new_variable(&mut self) -> Variable {
        // Generate the Variable
        let var = Variable(self.variable_map.keys().len());

        // Allocate space for the Variable on the variable_map
        // Each vector is initialised with a capacity of 16.
        // This number is a best guess estimate.
        self.variable_map.insert(var, Vec::with_capacity(16usize));

        var
    }

    pub fn new_variable_2(&mut self, gate_index: &mut usize) -> Variable {
        // Generate the Variable
        // let var = Variable(self.variable_map.keys().len());
        let var = Variable(*gate_index);
        *gate_index += 1;
        // Allocate space for the Variable on the variable_map
        // Each vector is initialised with a capacity of 16.
        // This number is a best guess estimate.
        self.variable_map.insert(var, Vec::with_capacity(16usize));

        var
    }

    /// Checks that the [`Variable`]s are valid by determining if they have been
    /// added to the system.
    fn valid_variables(&self, variables: &[Variable]) -> bool {
        variables
            .iter()
            .all(|var| self.variable_map.contains_key(var))
    }

    /// Maps a set of [`Variable`]s (a,b,c,d) to a set of [`Wire`](WireData)s
    /// (left, right, out, fourth) with the corresponding gate index
    pub fn add_variables_to_map(
        &mut self,
        a: Variable,
        b: Variable,
        c: Variable,
        d: Variable,
        gate_index: usize,
    ) {
        let left: WireData = WireData::Left(gate_index);
        let right: WireData = WireData::Right(gate_index);
        let output: WireData = WireData::Output(gate_index);
        let fourth: WireData = WireData::Fourth(gate_index);

        // Map each variable to the wire it is associated with
        // This essentially tells us that:
        self.add_variable_to_map(a, left);
        self.add_variable_to_map(b, right);
        self.add_variable_to_map(c, output);
        self.add_variable_to_map(d, fourth);
    }

    pub fn add_variable_to_map(&mut self, var: Variable, wire_data: WireData) {
        // assert!(self.valid_variables(&[var]));

        // NOTE: Since we always allocate space for the Vec of WireData when a
        // `Variable` is added to the variable_map, this should never fail.
        match self.variable_map.get_mut(&var) {
            Some(vec_wire_data) => {
                vec_wire_data.push(wire_data);
            }
            None => {
                self.variable_map.insert(var, vec![wire_data]);
            }
        }

        // let vec_wire_data = self.variable_map.get_mut(&var).unwrap();
        // vec_wire_data.push(wire_data);
        // vec_wire_data.sort();
    }

    pub fn add_wire_datas(&mut self, var: Variable, wire_datas: Vec<WireData>) {
        // assert!(self.valid_variables(&[var]));

        // NOTE: Since we always allocate space for the Vec of WireData when a
        // `Variable` is added to the variable_map, this should never fail.
        match self.variable_map.get_mut(&var) {
            Some(v) => {
                // let v = self.variable_map.get_mut(&var).unwrap();
                v.extend_from_slice(&wire_datas.as_slice());
            }
            None => {
                self.variable_map.insert(var, wire_datas);
            }
        }

        // let vec_wire_data = self.variable_map.get_mut(&var).unwrap();
        // vec_wire_data.push(wire_data);
        // vec_wire_data.sort();
    }

    pub fn sort_wire(&mut self) {
        self.variable_map
            .iter_mut()
            .for_each(|(_, items)| items.sort());
    }
    /// Performs shift by one permutation and computes `sigma_1`, `sigma_2` and
    /// `sigma_3`, `sigma_4` permutations from the variable maps.
    pub(super) fn compute_sigma_permutations(
        &mut self,
        n: usize,
    ) -> [Vec<WireData>; 4] {
        let sigma_1 = (0..n).map(WireData::Left).collect::<Vec<_>>();
        let sigma_2 = (0..n).map(WireData::Right).collect::<Vec<_>>();
        let sigma_3 = (0..n).map(WireData::Output).collect::<Vec<_>>();
        let sigma_4 = (0..n).map(WireData::Fourth).collect::<Vec<_>>();

        let mut sigmas = [sigma_1, sigma_2, sigma_3, sigma_4];

        for (_, wire_data) in self.variable_map.iter() {
            // Gets the data for each wire assosciated with this variable
            for (wire_index, current_wire) in wire_data.iter().enumerate() {
                // Fetch index of the next wire, if it is the last element
                // We loop back around to the beginning
                let next_index = match wire_index == wire_data.len() - 1 {
                    true => 0,
                    false => wire_index + 1,
                };

                // Fetch the next wire
                let next_wire = &wire_data[next_index];

                // Map current wire to next wire
                match current_wire {
                    WireData::Left(index) => sigmas[0][*index] = *next_wire,
                    WireData::Right(index) => sigmas[1][*index] = *next_wire,
                    WireData::Output(index) => sigmas[2][*index] = *next_wire,
                    WireData::Fourth(index) => sigmas[3][*index] = *next_wire,
                };
            }
        }

        sigmas
    }
}
impl Permutation {
    fn compute_permutation_lagrange<F: FftField>(
        &self,
        sigma_mapping: &[WireData],
        domain: &GeneralEvaluationDomain<F>,
    ) -> Vec<F> {
        let roots: Vec<_> = domain.elements().collect();

        let lagrange_poly: Vec<F> = sigma_mapping
            .iter()
            .map(|x| match x {
                WireData::Left(index) => {
                    let root = &roots[*index];
                    *root
                }
                WireData::Right(index) => {
                    let root = &roots[*index];
                    K1::<F>() * root
                }
                WireData::Output(index) => {
                    let root = &roots[*index];
                    K2::<F>() * root
                }
                WireData::Fourth(index) => {
                    let root = &roots[*index];
                    K3::<F>() * root
                }
            })
            .collect();

        lagrange_poly
    }

    /// Computes the sigma polynomials which are used to build the permutation
    /// polynomial.
    pub fn compute_sigma_polynomials<F: FftField>(
        &mut self,
        n: usize,
        domain: &GeneralEvaluationDomain<F>,
    ) -> (
        DensePolynomial<F>,
        DensePolynomial<F>,
        DensePolynomial<F>,
        DensePolynomial<F>,
    ) {
        // Compute sigma mappings
        let sigmas = self.compute_sigma_permutations(n);

        assert_eq!(sigmas[0].len(), n);
        assert_eq!(sigmas[1].len(), n);
        assert_eq!(sigmas[2].len(), n);
        assert_eq!(sigmas[3].len(), n);

        // define the sigma permutations using two non quadratic residues
        let left_sigma = self.compute_permutation_lagrange(&sigmas[0], domain);
        let right_sigma = self.compute_permutation_lagrange(&sigmas[1], domain);
        let out_sigma = self.compute_permutation_lagrange(&sigmas[2], domain);
        let fourth_sigma =
            self.compute_permutation_lagrange(&sigmas[3], domain);

        let left_sigma_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(&left_sigma));
        let right_sigma_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(&right_sigma));
        let out_sigma_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(&out_sigma));
        let fourth_sigma_poly =
            DensePolynomial::from_coefficients_vec(domain.ifft(&fourth_sigma));

        (
            left_sigma_poly,
            right_sigma_poly,
            out_sigma_poly,
            fourth_sigma_poly,
        )
    }

    #[allow(dead_code)]
    fn compute_slow_permutation_poly<I, F: FftField>(
        &self,
        domain: &GeneralEvaluationDomain<F>,
        w_l: I,
        w_r: I,
        w_o: I,
        w_4: I,
        beta: &F,
        gamma: &F,
        (left_sigma_poly, right_sigma_poly, out_sigma_poly, fourth_sigma_poly): (
            &DensePolynomial<F>,
            &DensePolynomial<F>,
            &DensePolynomial<F>,
            &DensePolynomial<F>,
        ),
    ) -> (Vec<F>, Vec<F>, Vec<F>)
    where
        I: Iterator<Item = F>,
    {
        let n = domain.size();

        let left_sigma_mapping = domain.fft(left_sigma_poly);
        let right_sigma_mapping = domain.fft(right_sigma_poly);
        let out_sigma_mapping = domain.fft(out_sigma_poly);
        let fourth_sigma_mapping = domain.fft(fourth_sigma_poly);

        // Compute beta * sigma polynomials
        let beta_left_sigma_iter =
            left_sigma_mapping.iter().map(|sigma| *sigma * beta);
        let beta_right_sigma_iter =
            right_sigma_mapping.iter().map(|sigma| *sigma * beta);
        let beta_out_sigma_iter =
            out_sigma_mapping.iter().map(|sigma| *sigma * beta);
        let beta_fourth_sigma_iter =
            fourth_sigma_mapping.iter().map(|sigma| *sigma * beta);

        // Compute beta * roots
        let beta_roots_iter = domain.elements().map(|root| root * beta);

        // Compute beta * roots * K1
        let beta_roots_k1_iter =
            domain.elements().map(|root| K1::<F>() * beta * root);

        // Compute beta * roots * K2
        let beta_roots_k2_iter =
            domain.elements().map(|root| K2::<F>() * beta * root);

        // Compute beta * roots * K3
        let beta_roots_k3_iter =
            domain.elements().map(|root| K3::<F>() * beta * root);

        // Compute left_wire + gamma
        let w_l_gamma: Vec<_> = w_l.map(|w| w + gamma).collect();

        // Compute right_wire + gamma
        let w_r_gamma: Vec<_> = w_r.map(|w| w + gamma).collect();

        // Compute out_wire + gamma
        let w_o_gamma: Vec<_> = w_o.map(|w| w + gamma).collect();

        // Compute fourth_wire + gamma
        let w_4_gamma: Vec<_> = w_4.map(|w| w + gamma).collect();

        let mut numerator_partial_components: Vec<F> = Vec::with_capacity(n);
        let mut denominator_partial_components: Vec<F> = Vec::with_capacity(n);

        let mut numerator_coefficients: Vec<F> = Vec::with_capacity(n);
        let mut denominator_coefficients: Vec<F> = Vec::with_capacity(n);

        // First element in both of them is one
        numerator_coefficients.push(F::one());
        denominator_coefficients.push(F::one());

        // Compute numerator coefficients
        for (
            w_l_gamma,
            w_r_gamma,
            w_o_gamma,
            w_4_gamma,
            beta_root,
            beta_root_k1,
            beta_root_k2,
            beta_root_k3,
        ) in izip!(
            w_l_gamma.iter(),
            w_r_gamma.iter(),
            w_o_gamma.iter(),
            w_4_gamma.iter(),
            beta_roots_iter,
            beta_roots_k1_iter,
            beta_roots_k2_iter,
            beta_roots_k3_iter,
        ) {
            // (w_l + beta * root + gamma)
            let prod_a = beta_root + w_l_gamma;

            // (w_r + beta * root * k_1 + gamma)
            let prod_b = beta_root_k1 + w_r_gamma;

            // (w_o + beta * root * k_2 + gamma)
            let prod_c = beta_root_k2 + w_o_gamma;

            // (w_4 + beta * root * k_3 + gamma)
            let prod_d = beta_root_k3 + w_4_gamma;

            let mut prod = prod_a * prod_b * prod_c * prod_d;

            numerator_partial_components.push(prod);

            prod *= numerator_coefficients.last().unwrap();

            numerator_coefficients.push(prod);
        }

        // Compute denominator coefficients
        for (
            w_l_gamma,
            w_r_gamma,
            w_o_gamma,
            w_4_gamma,
            beta_left_sigma,
            beta_right_sigma,
            beta_out_sigma,
            beta_fourth_sigma,
        ) in izip!(
            w_l_gamma,
            w_r_gamma,
            w_o_gamma,
            w_4_gamma,
            beta_left_sigma_iter,
            beta_right_sigma_iter,
            beta_out_sigma_iter,
            beta_fourth_sigma_iter,
        ) {
            // (w_l + beta * left_sigma + gamma)
            let prod_a = beta_left_sigma + w_l_gamma;

            // (w_r + beta * right_sigma + gamma)
            let prod_b = beta_right_sigma + w_r_gamma;

            // (w_o + beta * out_sigma + gamma)
            let prod_c = beta_out_sigma + w_o_gamma;

            // (w_4 + beta * fourth_sigma + gamma)
            let prod_d = beta_fourth_sigma + w_4_gamma;

            let mut prod = prod_a * prod_b * prod_c * prod_d;

            denominator_partial_components.push(prod);

            let last_element = denominator_coefficients.last().unwrap();

            prod *= last_element;

            denominator_coefficients.push(prod);
        }

        assert_eq!(denominator_coefficients.len(), n + 1);
        assert_eq!(numerator_coefficients.len(), n + 1);

        // Check that n+1'th elements are equal (taken from proof)
        let a = numerator_coefficients.last().unwrap();
        assert_ne!(a, &F::zero());
        let b = denominator_coefficients.last().unwrap();
        assert_ne!(b, &F::zero());
        assert_eq!(*a * b.inverse().unwrap(), F::one());

        // Remove those extra elements
        numerator_coefficients.remove(n);
        denominator_coefficients.remove(n);

        // Combine numerator and denominator

        let mut z_coefficients: Vec<F> = Vec::with_capacity(n);
        for (numerator, denominator) in numerator_coefficients
            .iter()
            .zip(denominator_coefficients.iter())
        {
            z_coefficients.push(*numerator * denominator.inverse().unwrap());
        }
        assert_eq!(z_coefficients.len(), n);

        (
            z_coefficients,
            numerator_partial_components,
            denominator_partial_components,
        )
    }

    #[allow(dead_code)]
    fn compute_fast_permutation_poly<F: FftField>(
        &self,
        domain: &GeneralEvaluationDomain<F>,
        w_l: &[F],
        w_r: &[F],
        w_o: &[F],
        w_4: &[F],
        beta: F,
        gamma: F,
        (left_sigma_poly, right_sigma_poly, out_sigma_poly, fourth_sigma_poly): (
            &DensePolynomial<F>,
            &DensePolynomial<F>,
            &DensePolynomial<F>,
            &DensePolynomial<F>,
        ),
    ) -> Vec<F> {
        let n = domain.size();

        // Compute beta * roots
        let common_roots: Vec<F> =
            domain.elements().map(|root| root * beta).collect();

        let left_sigma_mapping = domain.fft(left_sigma_poly);
        let right_sigma_mapping = domain.fft(right_sigma_poly);
        let out_sigma_mapping = domain.fft(out_sigma_poly);
        let fourth_sigma_mapping = domain.fft(fourth_sigma_poly);

        // Compute beta * sigma polynomials
        let beta_left_sigmas: Vec<_> = left_sigma_mapping
            .iter()
            .copied()
            .map(|sigma| sigma * beta)
            .collect();
        let beta_right_sigmas: Vec<_> = right_sigma_mapping
            .iter()
            .copied()
            .map(|sigma| sigma * beta)
            .collect();
        let beta_out_sigmas: Vec<_> = out_sigma_mapping
            .iter()
            .copied()
            .map(|sigma| sigma * beta)
            .collect();
        let beta_fourth_sigmas: Vec<_> = fourth_sigma_mapping
            .iter()
            .copied()
            .map(|sigma| sigma * beta)
            .collect();

        // Compute beta * roots * K1
        let beta_roots_k1: Vec<_> = common_roots
            .iter()
            .copied()
            .map(|x| x * K1::<F>())
            .collect();

        // Compute beta * roots * K2
        let beta_roots_k2: Vec<_> = common_roots
            .iter()
            .copied()
            .map(|x| x * K2::<F>())
            .collect();

        // Compute beta * roots * K3
        let beta_roots_k3: Vec<_> = common_roots
            .iter()
            .copied()
            .map(|x| x * K3::<F>())
            .collect();

        // Compute left_wire + gamma
        let w_l_gamma: Vec<_> =
            w_l.iter().copied().map(|w_l| w_l + gamma).collect();

        // Compute right_wire + gamma
        let w_r_gamma: Vec<_> =
            w_r.iter().copied().map(|w_r| w_r + gamma).collect();

        // Compute out_wire + gamma
        let w_o_gamma: Vec<_> =
            w_o.iter().copied().map(|w_o| w_o + gamma).collect();

        // Compute fourth_wire + gamma
        let w_4_gamma: Vec<_> =
            w_4.iter().copied().map(|w_4| w_4 + gamma).collect();

        // Compute 6 accumulator components
        // Parallisable
        let accumulator_components_without_l1: Vec<_> = izip!(
            w_l_gamma,
            w_r_gamma,
            w_o_gamma,
            w_4_gamma,
            common_roots,
            beta_roots_k1,
            beta_roots_k2,
            beta_roots_k3,
            beta_left_sigmas,
            beta_right_sigmas,
            beta_out_sigmas,
            beta_fourth_sigmas,
        )
        .map(
            |(
                w_l_gamma,
                w_r_gamma,
                w_o_gamma,
                w_4_gamma,
                beta_root,
                beta_root_k1,
                beta_root_k2,
                beta_root_k3,
                beta_left_sigma,
                beta_right_sigma,
                beta_out_sigma,
                beta_fourth_sigma,
            )| {
                // w_j + beta * root^j-1 + gamma
                let ac1 = w_l_gamma + beta_root;

                // w_{n+j} + beta * K1 * root^j-1 + gamma
                let ac2 = w_r_gamma + beta_root_k1;

                // w_{2n+j} + beta * K2 * root^j-1 + gamma
                let ac3 = w_o_gamma + beta_root_k2;

                // w_{3n+j} + beta * K3 * root^j-1 + gamma
                let ac4 = w_4_gamma + beta_root_k3;

                // 1 / w_j + beta * sigma(j) + gamma
                let ac5 = (w_l_gamma + beta_left_sigma).inverse().unwrap();

                // 1 / w_{n+j} + beta * sigma(n+j) + gamma
                let ac6 = (w_r_gamma + beta_right_sigma).inverse().unwrap();

                // 1 / w_{2n+j} + beta * sigma(2n+j) + gamma
                let ac7 = (w_o_gamma + beta_out_sigma).inverse().unwrap();

                // 1 / w_{3n+j} + beta * sigma(3n+j) + gamma
                let ac8 = (w_4_gamma + beta_fourth_sigma).inverse().unwrap();

                (ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8)
            },
        )
        .collect();

        // Prepend ones to the beginning of each accumulator to signify L_1(x)
        let accumulator_components = core::iter::once((
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
        ))
        .chain(accumulator_components_without_l1);

        // Multiply each component of the accumulators
        // A simplified example is the following:
        // A1 = [1,2,3,4]
        // result = [1, 1*2, 1*2*3, 1*2*3*4]
        // Non Parallelisable
        let mut prev = (
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
            F::one(),
        );
        let product_acumulated_components: Vec<_> = accumulator_components
            .map(move |current_component| {
                prev.0 *= current_component.0;
                prev.1 *= current_component.1;
                prev.2 *= current_component.2;
                prev.3 *= current_component.3;
                prev.4 *= current_component.4;
                prev.5 *= current_component.5;
                prev.6 *= current_component.6;
                prev.7 *= current_component.7;

                prev
            })
            .collect();

        // Right now we basically have 6 acumulators of the form:
        // A1 = [a1, a1 * a2, a1*a2*a3,...]
        // A2 = [b1, b1 * b2, b1*b2*b3,...]
        // A3 = [c1, c1 * c2, c1*c2*c3,...]
        // ... and so on
        // We want:
        // [a1*b1*c1, a1 * a2 *b1 * b2 * c1 * c2,...]
        // Parallisable
        let mut z: Vec<_> = product_acumulated_components
            .iter()
            .map(move |current_component| {
                let mut prev = F::one();
                prev *= current_component.0;
                prev *= current_component.1;
                prev *= current_component.2;
                prev *= current_component.3;
                prev *= current_component.4;
                prev *= current_component.5;
                prev *= current_component.6;
                prev *= current_component.7;

                prev
            })
            .collect();
        // Remove the last(n+1'th) element
        z.remove(n);

        assert_eq!(n, z.len());

        z
    }

    // These are the formulas for the irreducible factors used in the product
    // argument
    fn numerator_irreducible<F: FftField>(
        root: F,
        w: F,
        k: F,
        beta: F,
        gamma: F,
    ) -> F {
        w + beta * k * root + gamma
    }

    fn denominator_irreducible<F: FftField>(
        _root: F,
        w: F,
        sigma: F,
        beta: F,
        gamma: F,
    ) -> F {
        w + beta * sigma + gamma
    }

    // This can be adapted into a general product argument
    // for any number of wires, with specific formulas defined
    // in the numerator_irreducible and denominator_irreducible functions
    pub fn compute_permutation_poly<'a, 'b, G, F>(
        &self,
        domain: &GeneralEvaluationDomain<F>,
        gate_roots: &DeviceMemory<BigInteger256>,
        // wires: (&[F], &[F], &[F], &[F]),
        beta: F,
        gamma: F,
        // sigma_polys: (&Vec<F>, &Vec<F>, &Vec<F>, &Vec<F>),
        sigma: &DeviceMemory<BigInteger256>,
        kern: &PolyKernel<GpuFr>,
        gpu_container: &mut GpuPolyContainer<GpuFr>,
        msm_context: &MSMContext<'a, 'b, G>,
    ) where
        G: AffineCurve,
        F: FftField,
    {
        let n = domain.size();
        let start = std::time::Instant::now();

        // Constants defining cosets H, k1H, k2H, etc
        let ks = vec![F::one(), K1::<F>(), K2::<F>(), K3::<F>()];

        // let omega = domain.group_gen();
        // let omega_inv = domain.group_gen_inv();
        let size_inv = domain.size_inv();

        let start = std::time::Instant::now();
        // Transpose wires and sigma values to get "rows" in the form [wl_i,
        // wr_i, wo_i, ... ] where each row contains the wire and sigma
        // values for a single gate

        let mut w_l_gpu_orig =
            gpu_container.find(kern, "w_l_poly_orig").unwrap();
        let mut w_r_gpu_orig =
            gpu_container.find(kern, "w_r_poly_orig").unwrap();
        let mut w_o_gpu_orig =
            gpu_container.find(kern, "w_o_poly_orig").unwrap();
        let mut w_4_gpu_orig =
            gpu_container.find(kern, "w_4_poly_orig").unwrap();

        let mut gatewise_wires_poly = gpu_container
            .ask_for(kern, w_l_gpu_orig.size() * 4)
            .unwrap();
        gatewise_wires_poly.fill_with_fe(&F::zero()).unwrap();
        kern.sync().unwrap();

        let w_l_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                w_l_gpu_orig.get_memory(),
            )
        };
        let w_r_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                w_r_gpu_orig.get_memory(),
            )
        };
        let w_o_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                w_o_gpu_orig.get_memory(),
            )
        };
        let w_4_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                w_4_gpu_orig.get_memory(),
            )
        };
        let gatewise_wires_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                gatewise_wires_poly.get_memory(),
            )
        };
        wires_to_single_gate(
            w_l_gpu,
            w_r_gpu,
            w_o_gpu,
            w_4_gpu,
            gatewise_wires_gpu,
            msm_context,
        );

        // w_l_gpu_orig.fill_with_zero().unwrap();
        // gpu_container.recycle(w_l_gpu_orig).unwrap();
        // w_r_gpu_orig.fill_with_zero().unwrap();
        // gpu_container.recycle(w_r_gpu_orig).unwrap();
        // w_o_gpu_orig.fill_with_zero().unwrap();
        // gpu_container.recycle(w_o_gpu_orig).unwrap();
        // w_4_gpu_orig.fill_with_zero().unwrap();
        // gpu_container.recycle(w_4_gpu_orig).unwrap();
        // gatewise_wires_poly.fill_with_zero().unwrap();

        // gpu_container.recycle(gatewise_wires_poly).unwrap();
        // gpu_container.save("w_l_poly_orig", w_l_gpu_orig).unwrap();
        // gpu_container.save("w_r_poly_orig", w_r_gpu_orig).unwrap();
        // gpu_container.save("w_o_poly_orig", w_o_gpu_orig).unwrap();
        // gpu_container.save("w_4_poly_orig", w_4_gpu_orig).unwrap();
        // gpu_container.recycle(gatewise_wires_gpu).unwrap();

        let ks = unsafe { std::mem::transmute::<_, &Vec<BigInteger256>>(&ks) };
        let beta = unsafe { std::mem::transmute::<_, &BigInteger256>(&beta) };
        let gamma = unsafe { std::mem::transmute::<_, &BigInteger256>(&gamma) };

        /* find z_value_poly */
        let z_poly = gpu_container.find(&kern, "z_poly");
        let mut z_poly = match z_poly {
            Ok(poly) => poly,
            Err(_) => {
                let z = gpu_container.ask_for(&kern, n).unwrap();
                z
            }
        };

        // z_poly.fill_with_fe(&F::zero()).unwrap();
        // z_poly.sync().unwrap();
        // kern.sync().unwrap();

        let count = gate_roots.size();
        let mut dest_gpu = gpu_container.ask_for(&kern, count).unwrap();
        // dest_gpu.fill_with_fe(&F::zero()).unwrap();

        let mut d_dest_gpu = gpu_container.ask_for(&kern, count).unwrap();
        // d_dest_gpu.fill_with_fe(&F::zero()).unwrap();

        let mut ks_gpu = gpu_container.ask_for(&kern, ks.len()).unwrap();
        // ks_gpu.fill_with_fe(&F::zero()).unwrap();
        // ks_gpu.fill_with_zero().unwrap(); //???
        let mut beta_gpu = gpu_container.ask_for(&kern, 1).unwrap();
        let mut gamma_gpu = gpu_container.ask_for(&kern, 1).unwrap();

        let chunk_size = 512;
        let chunks = count / chunk_size;

        let mut z_iv = gpu_container.ask_for(&kern, chunks).unwrap();
        let mut z_piv = gpu_container.ask_for(&kern, chunks).unwrap();

        let _dest_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                dest_gpu.get_memory(),
            )
        };

        let _d_dest_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                d_dest_gpu.get_memory(),
            )
        };

        let _ks_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                ks_gpu.get_memory(),
            )
        };

        let _beta_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                beta_gpu.get_memory(),
            )
        };

        let _gamma_gpu = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                gamma_gpu.get_memory(),
            )
        };
        let _z_iv = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                z_iv.get_memory(),
            )
        };
        let _z_piv = unsafe {
            std::mem::transmute::<_, &DeviceMemory<BigInteger256>>(
                z_piv.get_memory(),
            )
        };

        _ks_gpu.read_from(&ks, ks.len()).unwrap();
        _beta_gpu.read_from(vec![*beta].as_slice(), 1).unwrap();
        _gamma_gpu.read_from(vec![*gamma].as_slice(), 1).unwrap();

        let start = std::time::Instant::now();
        product_argument_test(
            gate_roots,
            sigma,
            gatewise_wires_gpu,
            z_poly.get_memory().get_inner(),
            msm_context,
            &_dest_gpu,
            &_d_dest_gpu,
            &_ks_gpu,
            &_beta_gpu,
            &_gamma_gpu,
            _z_iv,
            _z_piv,
        );

        gpu_container.save("z_poly", z_poly).unwrap();

        // DensePolynomial::<F>::from_coefficients_vec(domain.ifft(&z))
        gpu_ifft_without_back_noread(
            kern,
            gpu_container,
            "z_poly",
            n,
            size_inv,
        );
        kern.sync().unwrap();

        gpu_container.recycle(ks_gpu).unwrap();
        gpu_container.recycle(dest_gpu).unwrap();
        gpu_container.recycle(d_dest_gpu).unwrap();
        gpu_container.recycle(beta_gpu).unwrap();
        gpu_container.recycle(gamma_gpu).unwrap();
        gpu_container.recycle(z_iv).unwrap();
        gpu_container.recycle(z_piv).unwrap();

        w_l_gpu_orig.fill_with_zero().unwrap();
        gpu_container.recycle(w_l_gpu_orig).unwrap();
        w_r_gpu_orig.fill_with_zero().unwrap();
        gpu_container.recycle(w_r_gpu_orig).unwrap();
        w_o_gpu_orig.fill_with_zero().unwrap();
        gpu_container.recycle(w_o_gpu_orig).unwrap();
        w_4_gpu_orig.fill_with_zero().unwrap();
        gpu_container.recycle(w_4_gpu_orig).unwrap();
        gatewise_wires_poly.fill_with_zero().unwrap();
        gpu_container.recycle(gatewise_wires_poly).unwrap();
    }

    pub(crate) fn compute_lookup_permutation_poly<F: FftField>(
        &self,
        domain: &GeneralEvaluationDomain<F>,
        f: &[F],
        t: &[F],
        h_1: &[F],
        h_2: &[F],
        delta: F,
        epsilon: F,
    ) -> DensePolynomial<F> {
        let n = domain.size();
        /*
        assert_eq!(f.len(), n);
        assert_eq!(t.len(), n);
        assert_eq!(h_1.len(), n);
        assert_eq!(h_2.len(), n);

        let t_next: Vec<F> = [&t[1..], &[t[0]]].concat();
        let h_1_next: Vec<F> = [&h_1[1..], &[h_1[0]]].concat();

        let product_arguments: Vec<F> = f
            .iter()
            .zip(t)
            .zip(t_next)
            .zip(h_1)
            .zip(h_1_next)
            .zip(h_2)
            // Derive the numerator and denominator for each gate plonkup
            // gate and pair the results
            .map(|(((((f, t), t_next), h_1), h_1_next), h_2)| {
                Self::lookup_ratio(
                    delta, epsilon, *f, *t, t_next, *h_1, h_1_next, *h_2,
                )
            })
            .collect();

        let mut state = F::one();
        let mut p = Vec::with_capacity(n + 1);
        p.push(state);
        for s in product_arguments {
            state *= s;
            p.push(state);
        }
        p.pop();
        assert_eq!(n, p.len());

        DensePolynomial::from_coefficients_vec(domain.ifft(&p))

        println!(
                "compute_lookup_permutation_poly, 1 spent:{:?}",
                start.elapsed()
            );
            let start = std::time::Instant::now();

            let ret = DensePolynomial::from_coefficients_vec(domain.ifft(&p));
            println!(
                "compute_lookup_permutation_poly, 2 spent:{:?}",
                start.elapsed()
            );

            ret
         */
        let coeff = Self::lookup_ratio(
            delta,
            epsilon,
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
            F::zero(),
        );
        let mut coeff_v = vec![F::zero(); n];
        coeff_v[0] = coeff;
        DensePolynomial::from_coefficients_vec(coeff_v)
    }

    fn lookup_ratio<F: FftField>(
        delta: F,
        epsilon: F,
        f: F,
        t: F,
        t_next: F,
        h_1: F,
        h_1_next: F,
        h_2: F,
    ) -> F {
        let one_plus_delta = F::one() + delta;
        let epsilon_one_plus_delta = epsilon * one_plus_delta;
        one_plus_delta
            * (epsilon + f)
            * (epsilon_one_plus_delta + t + (delta * t_next))
            * ((epsilon_one_plus_delta + h_1 + (h_2 * delta))
                * (epsilon_one_plus_delta + h_2 + (h_1_next * delta)))
                .inverse()
                .unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{batch_test_field, batch_test_field_params};
    use crate::{
        constraint_system::StandardComposer, util::EvaluationDomainExt,
    };
    use ark_bls12_377::Bls12_377;
    use ark_bls12_381::Bls12_381;
    use ark_ec::TEModelParameters;
    use ark_ff::{Field, PrimeField};
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::Polynomial;
    use rand_core::OsRng;

    fn test_multizip_permutation_poly<F, P>()
    where
        F: PrimeField,
        P: TEModelParameters<BaseField = F>,
    {
        let mut cs: StandardComposer<F, P> =
            StandardComposer::<F, P>::with_expected_size(4);

        let zero = F::zero();
        let one = F::one();
        let two = one + one;

        let x1 = cs.add_input(F::from(4u64));
        let x2 = cs.add_input(F::from(12u64));
        let x3 = cs.add_input(F::from(8u64));
        let x4 = cs.add_input(F::from(3u64));

        // x1 * x4 = x2
        cs.poly_gate(x1, x4, x2, one, zero, zero, -one, zero, None);

        // x1 + x3 = x2
        cs.poly_gate(x1, x3, x2, zero, one, one, -one, zero, None);

        // x1 + x2 = 2*x3
        cs.poly_gate(x1, x2, x3, zero, one, one, -two, zero, None);

        // x3 * x4 = 2*x2
        cs.poly_gate(x3, x4, x2, one, zero, zero, -two, zero, None);

        let domain =
            GeneralEvaluationDomain::<F>::new(cs.circuit_bound()).unwrap();

        let pad = vec![F::zero(); domain.size() - cs.w_l.len()];
        let mut w_l_scalar: Vec<F> =
            cs.w_l.iter().map(|v| cs.variable_vec[v.0]).collect();
        let mut w_r_scalar: Vec<F> =
            cs.w_r.iter().map(|v| cs.variables_vec[v.0]).collect();
        let mut w_o_scalar: Vec<F> =
            cs.w_o.iter().map(|v| cs.variables_vec[v.0]).collect();
        let mut w_4_scalar: Vec<F> =
            cs.w_4.iter().map(|v| cs.variables_vec[v.0]).collect();

        w_l_scalar.extend(&pad);
        w_r_scalar.extend(&pad);
        w_o_scalar.extend(&pad);
        w_4_scalar.extend(&pad);

        let sigmas: Vec<Vec<F>> = cs
            .perm
            .compute_sigma_permutations(cs.circuit_bound())
            .iter()
            .map(|wd| cs.perm.compute_permutation_lagrange(wd, &domain))
            .collect();

        let beta = F::rand(&mut OsRng);
        let gamma = F::rand(&mut OsRng);

        let sigma_polys: Vec<DensePolynomial<F>> = sigmas
            .iter()
            .map(|v| DensePolynomial::from_coefficients_vec(domain.ifft(v)))
            .collect();

        let mz = cs.perm.compute_permutation_poly(
            &domain,
            (&w_l_scalar, &w_r_scalar, &w_o_scalar, &w_4_scalar),
            beta,
            gamma,
            (
                &sigma_polys[0],
                &sigma_polys[1],
                &sigma_polys[2],
                &sigma_polys[3],
            ),
        );

        let old_z = DensePolynomial::from_coefficients_vec(domain.ifft(
            &cs.perm.compute_fast_permutation_poly(
                &domain,
                &w_l_scalar,
                &w_r_scalar,
                &w_o_scalar,
                &w_4_scalar,
                beta,
                gamma,
                (
                    &sigma_polys[0],
                    &sigma_polys[1],
                    &sigma_polys[2],
                    &sigma_polys[3],
                ),
            ),
        ));

        assert_eq!(mz, old_z);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_permutation_format() {
        let mut perm: Permutation = Permutation::new();

        let num_variables = 10u8;
        for i in 0..num_variables {
            let var = perm.new_variable();
            assert_eq!(var.0, i as usize);
            assert_eq!(perm.variable_map.len(), (i as usize) + 1);
        }

        let var_one = perm.new_variable();
        let var_two = perm.new_variable();
        let var_three = perm.new_variable();

        let gate_size = 100;
        for i in 0..gate_size {
            perm.add_variables_to_map(var_one, var_one, var_two, var_three, i);
        }

        // Check all gate_indices are valid
        for (_, wire_data) in perm.variable_map.iter() {
            for wire in wire_data.iter() {
                match wire {
                    WireData::Left(index)
                    | WireData::Right(index)
                    | WireData::Output(index)
                    | WireData::Fourth(index) => assert!(*index < gate_size),
                };
            }
        }
    }

    fn test_permutation_compute_sigmas_only_left_wires<F: FftField>() {
        let mut perm = Permutation::new();

        let var_zero = perm.new_variable();
        let var_two = perm.new_variable();
        let var_three = perm.new_variable();
        let var_four = perm.new_variable();
        let var_five = perm.new_variable();
        let var_six = perm.new_variable();
        let var_seven = perm.new_variable();
        let var_eight = perm.new_variable();
        let var_nine = perm.new_variable();

        let num_wire_mappings = 4;

        // Add four wire mappings
        perm.add_variables_to_map(var_zero, var_zero, var_five, var_nine, 0);
        perm.add_variables_to_map(var_zero, var_two, var_six, var_nine, 1);
        perm.add_variables_to_map(var_zero, var_three, var_seven, var_nine, 2);
        perm.add_variables_to_map(var_zero, var_four, var_eight, var_nine, 3);

        /*
        var_zero = {L0, R0, L1, L2, L3}
        var_two = {R1}
        var_three = {R2}
        var_four = {R3}
        var_five = {O0}
        var_six = {O1}
        var_seven = {O2}
        var_eight = {O3}
        var_nine = {F0, F1, F2, F3}
        Left_sigma = {R0, L2, L3, L0}
        Right_sigma = {L1, R1, R2, R3}
        Out_sigma = {O0, O1, O2, O3}
        Fourth_sigma = {F1, F2, F3, F0}
        */
        let sigmas = perm.compute_sigma_permutations(num_wire_mappings);
        let left_sigma = &sigmas[0];
        let right_sigma = &sigmas[1];
        let out_sigma = &sigmas[2];
        let fourth_sigma = &sigmas[3];

        // Check the left sigma polynomial
        assert_eq!(left_sigma[0], WireData::Right(0));
        assert_eq!(left_sigma[1], WireData::Left(2));
        assert_eq!(left_sigma[2], WireData::Left(3));
        assert_eq!(left_sigma[3], WireData::Left(0));

        // Check the right sigma polynomial
        assert_eq!(right_sigma[0], WireData::Left(1));
        assert_eq!(right_sigma[1], WireData::Right(1));
        assert_eq!(right_sigma[2], WireData::Right(2));
        assert_eq!(right_sigma[3], WireData::Right(3));

        // Check the output sigma polynomial
        assert_eq!(out_sigma[0], WireData::Output(0));
        assert_eq!(out_sigma[1], WireData::Output(1));
        assert_eq!(out_sigma[2], WireData::Output(2));
        assert_eq!(out_sigma[3], WireData::Output(3));

        // Check the output sigma polynomial
        assert_eq!(fourth_sigma[0], WireData::Fourth(1));
        assert_eq!(fourth_sigma[1], WireData::Fourth(2));
        assert_eq!(fourth_sigma[2], WireData::Fourth(3));
        assert_eq!(fourth_sigma[3], WireData::Fourth(0));

        let domain =
            GeneralEvaluationDomain::<F>::new(num_wire_mappings).unwrap();
        let w = domain.group_gen();
        let w_squared = w.pow([2, 0, 0, 0]);
        let w_cubed = w.pow([3, 0, 0, 0]);

        // Check the left sigmas have been encoded properly
        // Left_sigma = {R0, L2, L3, L0}
        // Should turn into {1 * K1, w^2, w^3, 1}
        let encoded_left_sigma =
            perm.compute_permutation_lagrange(left_sigma, &domain);
        assert_eq!(encoded_left_sigma[0], F::one() * K1::<F>());
        assert_eq!(encoded_left_sigma[1], w_squared);
        assert_eq!(encoded_left_sigma[2], w_cubed);
        assert_eq!(encoded_left_sigma[3], F::one());

        // Check the right sigmas have been encoded properly
        // Right_sigma = {L1, R1, R2, R3}
        // Should turn into {w, w * K1, w^2 * K1, w^3 * K1}
        let encoded_right_sigma =
            perm.compute_permutation_lagrange(right_sigma, &domain);
        assert_eq!(encoded_right_sigma[0], w);
        assert_eq!(encoded_right_sigma[1], w * K1::<F>());
        assert_eq!(encoded_right_sigma[2], w_squared * K1::<F>());
        assert_eq!(encoded_right_sigma[3], w_cubed * K1::<F>());

        // Check the output sigmas have been encoded properly
        // Out_sigma = {O0, O1, O2, O3}
        // Should turn into {1 * K2, w * K2, w^2 * K2, w^3 * K2}

        let encoded_output_sigma =
            perm.compute_permutation_lagrange(out_sigma, &domain);
        assert_eq!(encoded_output_sigma[0], F::one() * K2::<F>());
        assert_eq!(encoded_output_sigma[1], w * K2::<F>());
        assert_eq!(encoded_output_sigma[2], w_squared * K2::<F>());
        assert_eq!(encoded_output_sigma[3], w_cubed * K2::<F>());

        // Check the fourth sigmas have been encoded properly
        // Out_sigma = {F1, F2, F3, F0}
        // Should turn into {w * K3, w^2 * K3, w^3 * K3, 1 * K3}
        let encoded_fourth_sigma =
            perm.compute_permutation_lagrange(fourth_sigma, &domain);
        assert_eq!(encoded_fourth_sigma[0], w * K3::<F>());
        assert_eq!(encoded_fourth_sigma[1], w_squared * K3::<F>());
        assert_eq!(encoded_fourth_sigma[2], w_cubed * K3::<F>());
        assert_eq!(encoded_fourth_sigma[3], K3());

        let w_l =
            vec![F::from(2u64), F::from(2u64), F::from(2u64), F::from(2u64)];
        let w_r = vec![F::from(2_u64), F::one(), F::one(), F::one()];
        let w_o = vec![F::one(), F::one(), F::one(), F::one()];
        let w_4 = vec![F::one(), F::one(), F::one(), F::one()];

        test_correct_permutation_poly(
            num_wire_mappings,
            perm,
            &domain,
            w_l,
            w_r,
            w_o,
            w_4,
        );
    }
    fn test_permutation_compute_sigmas<F: FftField>() {
        let mut perm: Permutation = Permutation::new();

        let var_one = perm.new_variable();
        let var_two = perm.new_variable();
        let var_three = perm.new_variable();
        let var_four = perm.new_variable();

        let num_wire_mappings = 4;

        // Add four wire mappings
        perm.add_variables_to_map(var_one, var_one, var_two, var_four, 0);
        perm.add_variables_to_map(var_two, var_one, var_two, var_four, 1);
        perm.add_variables_to_map(var_three, var_three, var_one, var_four, 2);
        perm.add_variables_to_map(var_two, var_one, var_three, var_four, 3);

        /*
        Below is a sketch of the map created by adding the specific variables into the map
        var_one : {L0,R0, R1, O2, R3 }
        var_two : {O0, L1, O1, L3}
        var_three : {L2, R2, O3}
        var_four : {F0, F1, F2, F3}
        Left_Sigma : {0,1,2,3} -> {R0,O1,R2,O0}
        Right_Sigma : {0,1,2,3} -> {R1, O2, O3, L0}
        Out_Sigma : {0,1,2,3} -> {L1, L3, R3, L2}
        Fourth_Sigma : {0,1,2,3} -> {F1, F2, F3, F0}
        */
        let sigmas = perm.compute_sigma_permutations(num_wire_mappings);
        let left_sigma = &sigmas[0];
        let right_sigma = &sigmas[1];
        let out_sigma = &sigmas[2];
        let fourth_sigma = &sigmas[3];

        // Check the left sigma polynomial
        assert_eq!(left_sigma[0], WireData::Right(0));
        assert_eq!(left_sigma[1], WireData::Output(1));
        assert_eq!(left_sigma[2], WireData::Right(2));
        assert_eq!(left_sigma[3], WireData::Output(0));

        // Check the right sigma polynomial
        assert_eq!(right_sigma[0], WireData::Right(1));
        assert_eq!(right_sigma[1], WireData::Output(2));
        assert_eq!(right_sigma[2], WireData::Output(3));
        assert_eq!(right_sigma[3], WireData::Left(0));

        // Check the output sigma polynomial
        assert_eq!(out_sigma[0], WireData::Left(1));
        assert_eq!(out_sigma[1], WireData::Left(3));
        assert_eq!(out_sigma[2], WireData::Right(3));
        assert_eq!(out_sigma[3], WireData::Left(2));

        // Check the fourth sigma polynomial
        assert_eq!(fourth_sigma[0], WireData::Fourth(1));
        assert_eq!(fourth_sigma[1], WireData::Fourth(2));
        assert_eq!(fourth_sigma[2], WireData::Fourth(3));
        assert_eq!(fourth_sigma[3], WireData::Fourth(0));

        /*
        Check that the unique encodings of the sigma polynomials have been computed properly
        Left_Sigma : {R0,O1,R2,O0}
            When encoded using w, K1,K2,K3 we have {1 * K1, w * K2, w^2 * K1, 1 * K2}
        Right_Sigma : {R1, O2, O3, L0}
            When encoded using w, K1,K2,K3 we have {w * K1, w^2 * K2, w^3 * K2, 1}
        Out_Sigma : {L1, L3, R3, L2}
            When encoded using w, K1, K2,K3 we have {w, w^3 , w^3 * K1, w^2}
        Fourth_Sigma : {0,1,2,3} -> {F1, F2, F3, F0}
            When encoded using w, K1, K2,K3 we have {w * K3, w^2 * K3, w^3 * K3, 1 * K3}
        */
        let domain =
            GeneralEvaluationDomain::<F>::new(num_wire_mappings).unwrap();
        let w = domain.group_gen();
        let w_squared = w.pow([2, 0, 0, 0]);
        let w_cubed = w.pow([3, 0, 0, 0]);
        // check the left sigmas have been encoded properly
        let encoded_left_sigma =
            perm.compute_permutation_lagrange(left_sigma, &domain);
        assert_eq!(encoded_left_sigma[0], K1());
        assert_eq!(encoded_left_sigma[1], w * K2::<F>());
        assert_eq!(encoded_left_sigma[2], w_squared * K1::<F>());
        assert_eq!(encoded_left_sigma[3], F::one() * K2::<F>());

        // check the right sigmas have been encoded properly
        let encoded_right_sigma =
            perm.compute_permutation_lagrange(right_sigma, &domain);
        assert_eq!(encoded_right_sigma[0], w * K1::<F>());
        assert_eq!(encoded_right_sigma[1], w_squared * K2::<F>());
        assert_eq!(encoded_right_sigma[2], w_cubed * K2::<F>());
        assert_eq!(encoded_right_sigma[3], F::one());

        // check the output sigmas have been encoded properly
        let encoded_output_sigma =
            perm.compute_permutation_lagrange(out_sigma, &domain);
        assert_eq!(encoded_output_sigma[0], w);
        assert_eq!(encoded_output_sigma[1], w_cubed);
        assert_eq!(encoded_output_sigma[2], w_cubed * K1::<F>());
        assert_eq!(encoded_output_sigma[3], w_squared);

        // check the fourth sigmas have been encoded properly
        let encoded_fourth_sigma =
            perm.compute_permutation_lagrange(fourth_sigma, &domain);
        assert_eq!(encoded_fourth_sigma[0], w * K3::<F>());
        assert_eq!(encoded_fourth_sigma[1], w_squared * K3::<F>());
        assert_eq!(encoded_fourth_sigma[2], w_cubed * K3::<F>());
        assert_eq!(encoded_fourth_sigma[3], K3());
    }

    fn test_basic_slow_permutation_poly<F: FftField>() {
        let num_wire_mappings = 2;
        let mut perm = Permutation::new();
        let domain =
            GeneralEvaluationDomain::<F>::new(num_wire_mappings).unwrap();

        let var_one = perm.new_variable();
        let var_two = perm.new_variable();
        let var_three = perm.new_variable();
        let var_four = perm.new_variable();

        perm.add_variables_to_map(var_one, var_two, var_three, var_four, 0);
        perm.add_variables_to_map(var_three, var_two, var_one, var_four, 1);

        let w_l = vec![F::one(), F::from(3u64)];
        let w_r = vec![F::from(2u64), F::from(2u64)];
        let w_o = vec![F::from(3u64), F::one()];
        let w_4 = vec![F::one(), F::one()];

        test_correct_permutation_poly(
            num_wire_mappings,
            perm,
            &domain,
            w_l,
            w_r,
            w_o,
            w_4,
        );
    }

    // shifts the polynomials by one root of unity
    fn shift_poly_by_one<F: Field>(z_coefficients: Vec<F>) -> Vec<F> {
        let mut shifted_z_coefficients = z_coefficients;
        shifted_z_coefficients.push(shifted_z_coefficients[0]);
        shifted_z_coefficients.remove(0);
        shifted_z_coefficients
    }

    fn test_correct_permutation_poly<F: FftField>(
        n: usize,
        mut perm: Permutation,
        domain: &GeneralEvaluationDomain<F>,
        w_l: Vec<F>,
        w_r: Vec<F>,
        w_o: Vec<F>,
        w_4: Vec<F>,
    ) {
        // 0. Generate beta and gamma challenges
        //
        let beta = F::rand(&mut OsRng);
        let gamma = F::rand(&mut OsRng);
        assert_ne!(gamma, beta);

        // 1. Compute the permutation polynomial using both methods
        //
        let (
            left_sigma_poly,
            right_sigma_poly,
            out_sigma_poly,
            fourth_sigma_poly,
        ) = perm.compute_sigma_polynomials(n, domain);
        let (z_vec, numerator_components, denominator_components) = perm
            .compute_slow_permutation_poly(
                domain,
                w_l.clone().into_iter(),
                w_r.clone().into_iter(),
                w_o.clone().into_iter(),
                w_4.clone().into_iter(),
                &beta,
                &gamma,
                (
                    &left_sigma_poly,
                    &right_sigma_poly,
                    &out_sigma_poly,
                    &fourth_sigma_poly,
                ),
            );

        let fast_z_vec = perm.compute_fast_permutation_poly(
            domain,
            &w_l,
            &w_r,
            &w_o,
            &w_4,
            beta,
            gamma,
            (
                &left_sigma_poly,
                &right_sigma_poly,
                &out_sigma_poly,
                &fourth_sigma_poly,
            ),
        );
        assert_eq!(fast_z_vec, z_vec);

        // 2. First we perform basic tests on the permutation vector
        //
        // Check that the vector has length `n` and that the first element is
        // `1`
        assert_eq!(z_vec.len(), n);
        assert_eq!(&z_vec[0], &F::one());
        //
        // Check that the \prod{f_i} / \prod{g_i} = 1
        // Where f_i and g_i are the numerator and denominator components in the
        // permutation polynomial
        let (mut a_0, mut b_0) = (F::one(), F::one());
        for n in numerator_components.iter() {
            a_0 *= n;
        }
        for n in denominator_components.iter() {
            b_0 *= n;
        }
        assert_eq!(a_0 * b_0.inverse().unwrap(), F::one());

        // 3. Now we perform the two checks that need to be done on the
        // permutation polynomial (z)
        let z_poly =
            DensePolynomial::<F>::from_coefficients_vec(domain.ifft(&z_vec));
        //
        // Check that z(w^{n+1}) == z(1) == 1
        // This is the first check in the protocol
        assert_eq!(z_poly.evaluate(&F::one()), F::one());
        let n_plus_one = domain.elements().last().unwrap() * domain.group_gen();
        assert_eq!(z_poly.evaluate(&n_plus_one), F::one());
        //
        // Check that when z is unblinded, it has the correct degree
        assert_eq!(z_poly.degree(), n - 1);
        //
        // Check relationship between z(X) and z(Xw)
        // This is the second check in the protocol
        let roots: Vec<_> = domain.elements().collect();

        for i in 1..roots.len() {
            let current_root = roots[i];
            let next_root = current_root * domain.group_gen();

            let current_identity_perm_product = &numerator_components[i];
            assert_ne!(current_identity_perm_product, &F::zero());

            let current_copy_perm_product = &denominator_components[i];
            assert_ne!(current_copy_perm_product, &F::zero());

            assert_ne!(
                current_copy_perm_product,
                current_identity_perm_product
            );

            let z_eval = z_poly.evaluate(&current_root);
            assert_ne!(z_eval, F::zero());

            let z_eval_shifted = z_poly.evaluate(&next_root);
            assert_ne!(z_eval_shifted, F::zero());

            // Z(Xw) * copy_perm
            let lhs = z_eval_shifted * current_copy_perm_product;
            // Z(X) * iden_perm
            let rhs = z_eval * current_identity_perm_product;
            assert_eq!(
                lhs, rhs,
                "check failed at index: {}\'n lhs is : {:?} \n rhs is :{:?}",
                i, lhs, rhs
            );
        }

        // Test that the shifted polynomial is correct
        let shifted_z = shift_poly_by_one(fast_z_vec);
        let shifted_z_poly = DensePolynomial::<F>::from_coefficients_vec(
            domain.ifft(&shifted_z),
        );
        for element in domain.elements() {
            let z_eval = z_poly.evaluate(&(element * domain.group_gen()));
            let shifted_z_eval = shifted_z_poly.evaluate(&element);

            assert_eq!(z_eval, shifted_z_eval)
        }
    }

    // Test on Bls12-381
    batch_test_field!(
        [test_permutation_compute_sigmas_only_left_wires,
        test_permutation_compute_sigmas,
        test_basic_slow_permutation_poly
        ],
        []
        => (
            Bls12_381
        )
    );

    // Test on Bls12-377
    batch_test_field!(
        [test_permutation_compute_sigmas_only_left_wires,
        test_permutation_compute_sigmas,
        test_basic_slow_permutation_poly
        ],
        []
        => (
            Bls12_377
        )
    );

    // Test on Bls12-381
    batch_test_field_params!(
        [test_multizip_permutation_poly
        ],
        []
        => (
            Bls12_381,
            ark_ed_on_bls12_381::EdwardsParameters
        )
    );

    // Test on Bls12-377
    batch_test_field_params!(
        [test_multizip_permutation_poly
        ],
        []
        => (
            Bls12_377,
            ark_ed_on_bls12_377::EdwardsParameters
        )
    );
}
