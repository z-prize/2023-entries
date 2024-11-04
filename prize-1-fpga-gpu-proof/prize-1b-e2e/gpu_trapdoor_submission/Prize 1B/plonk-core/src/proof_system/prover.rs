// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Prover-side of the PLONK Proving System

use crate::permutation::{
    gpu_ifft_without_back, gpu_ifft_without_back_gscalars,
    gpu_ifft_without_back_orig_data,
};
use crate::proof_system::gpu_prover::{
    compute_linear_gpu, compute_quotient_gpu, open_gpu, prepare_domain,
    prepare_domain_8n_coset,
};
use crate::{
    commitment::HomomorphicCommitment,
    constraint_system::{StandardComposer, Variable},
    error::{to_pc_error, Error},
    label_polynomial,
    proof_system::{
        linearisation_poly, proof::Proof, quotient_poly, ProverKey,
    },
    transcript::TranscriptProtocol,
};
use ark_ec::{AffineCurve, ModelParameters, PairingEngine, TEModelParameters};
use ark_ff::{PrimeField, Zero};
use ark_poly::Polynomial;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain,
    UVPolynomial,
};
use ark_poly_commit::{PCCommitment, PolynomialCommitment};
use core::marker::PhantomData;
use merlin::Transcript;

use crate::permutation::GPU_POLY_KERNEL;
use crate::permutation::MSM_KERN;
use crate::util::EvaluationDomainExt;
use ark_ff::BigInteger256;
use ark_poly_commit::LabeledCommitment;
use ec_gpu_common::{
    CUdeviceptr, CudaContext, DeviceMemory, Fr as GpuFr, GpuPolyContainer,
    MSMContext, PolyKernel,
};
use itertools::izip;
use rayon::prelude::*;
use std::sync::Mutex;
use std::thread;
use std::time::Instant;
use crate::permutation::Permutation;

pub const HEIGHT: usize = 15;

lazy_static::lazy_static! {
    pub static ref DOMAIN:GeneralEvaluationDomain<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr> = GeneralEvaluationDomain::<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>::new(1<<(HEIGHT+7)).unwrap();
    pub static ref DOMAIN_8N:GeneralEvaluationDomain<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr> = GeneralEvaluationDomain::new(8 * (1<<(HEIGHT+7))).unwrap();
    pub static ref W_L_VEC: Mutex<Vec<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>> = Mutex::new(vec![<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr::zero(); (1<<(HEIGHT + 7))]);
    pub static ref W_R_VEC: Mutex<Vec<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>> = Mutex::new(vec![<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr::zero(); (1<<(HEIGHT + 7))]);
    pub static ref W_O_VEC: Mutex<Vec<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>> = Mutex::new(vec![<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr::zero(); (1<<(HEIGHT + 7))]);
    pub static ref W_4_VEC: Mutex<Vec<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>> = Mutex::new(vec![<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr::zero(); (1<<(HEIGHT + 7))]);
}
/// Abstraction structure designed to construct a circuit and generate
/// [`Proof`]s for it.
pub struct Prover<F, P, PC>
where
    F: PrimeField,
    P: ModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
{
    /// Proving Key which is used to create proofs about a specific PLONK
    /// circuit.
    pub prover_key: Option<ProverKey<F>>,

    /// Circuit Description
    pub(crate) cs: StandardComposer<F, P>,

    /// Store the messages exchanged during the preprocessing stage.
    ///
    /// This is copied each time, we make a proof.
    pub preprocessed_transcript: Transcript,

    pub gate_roots: DeviceMemory<BigInteger256>,
    pub gatewise_sigmas: DeviceMemory<BigInteger256>,

    _phantom: PhantomData<PC>,
}
impl<F, P, PC> Prover<F, P, PC>
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
{
    /// Creates a new `Prover` instance.
    pub fn new(label: &'static [u8], ctx: &CudaContext) -> Self {
        Self {
            prover_key: None,
            cs: StandardComposer::new(),
            preprocessed_transcript: Transcript::new(label),
            _phantom: PhantomData::<PC>,
            gate_roots: DeviceMemory::<BigInteger256>::new(ctx, 1).unwrap(),
            gatewise_sigmas: DeviceMemory::<BigInteger256>::new(ctx, 1)
                .unwrap(),
        }
    }

    pub fn new_2(
        label: &'static [u8],
        ctx: &CudaContext,
        tree_height: u32,
        prover_key: Option<&ProverKey<F>>,
        domain: Option<&GeneralEvaluationDomain<F>>,
        kern: Option<&PolyKernel<GpuFr>>,
        gpu_container: Option<&mut GpuPolyContainer<GpuFr>>,
    ) -> Self {
        if prover_key.is_some() {
            let prover_key = prover_key.unwrap();
            let kern = kern.unwrap();
            let gpu_container = gpu_container.unwrap();
            let domain = domain.unwrap();

            let n = domain.size();
            let mut gpu_left_sigma = gpu_container.ask_for(kern, n).unwrap();
            gpu_left_sigma.fill_with_zero().unwrap();

            gpu_left_sigma
                .read_from(&prover_key.permutation.left_sigma.0)
                .unwrap();
            gpu_container
                .save("prover_key.permutation.left_sigma", gpu_left_sigma)
                .unwrap();
            let mut gpu_right_sigma = gpu_container.ask_for(kern, n).unwrap();
            gpu_right_sigma.fill_with_zero().unwrap();
            gpu_right_sigma
                .read_from(&prover_key.permutation.right_sigma.0)
                .unwrap();
            gpu_container
                .save("prover_key.permutation.right_sigma", gpu_right_sigma)
                .unwrap();
            let mut gpu_out_sigma = gpu_container.ask_for(kern, n).unwrap();
            gpu_out_sigma.fill_with_zero().unwrap();
            gpu_out_sigma
                .read_from(&prover_key.permutation.out_sigma.0)
                .unwrap();
            gpu_container
                .save("prover_key.permutation.out_sigma", gpu_out_sigma)
                .unwrap();

            //arithmetic
            let n8 = n * 8;
            let mut q_l_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_l_1_evals
                .read_from(&prover_key.arithmetic.q_l.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_l_1_evals", q_l_1_evals)
                .unwrap();

            let mut q_r_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_r_1_evals
                .read_from(&prover_key.arithmetic.q_r.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_r_1_evals", q_r_1_evals)
                .unwrap();

            let mut q_o_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_o_1_evals
                .read_from(&prover_key.arithmetic.q_o.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_o_1_evals", q_o_1_evals)
                .unwrap();

            let mut q_4_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_4_1_evals
                .read_from(&prover_key.arithmetic.q_4.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_4_1_evals", q_4_1_evals)
                .unwrap();

            let mut q_m_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_m_1_evals
                .read_from(&prover_key.arithmetic.q_m.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_m_1_evals", q_m_1_evals)
                .unwrap();

            let mut q_hl_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_hl_1_evals
                .read_from(&prover_key.arithmetic.q_hl.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_hl_1_evals", q_hl_1_evals)
                .unwrap();

            let mut q_hr_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_hr_1_evals
                .read_from(&prover_key.arithmetic.q_hr.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_hr_1_evals", q_hr_1_evals)
                .unwrap();

            let mut q_h4_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_h4_1_evals
                .read_from(&prover_key.arithmetic.q_h4.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_h4_1_evals", q_h4_1_evals)
                .unwrap();

            let mut q_c_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_c_1_evals
                .read_from(&prover_key.arithmetic.q_c.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_c_1_evals", q_c_1_evals)
                .unwrap();

            let mut q_arith_1_evals = gpu_container.ask_for(kern, n8).unwrap();
            q_arith_1_evals
                .read_from(&prover_key.arithmetic.q_arith.1.evals)
                .unwrap();
            gpu_container
                .save("prover_key.arithmetic.q_arith_1_evals", q_arith_1_evals)
                .unwrap();

            let mut linear_evaluations_evals =
                gpu_container.ask_for(kern, n8).unwrap();
            linear_evaluations_evals
                .read_from(&prover_key.permutation.linear_evaluations.evals)
                .unwrap();
            gpu_container
                .save(
                    "prover_key.permutation.linear_evaluations_evals",
                    linear_evaluations_evals,
                )
                .unwrap();

            let mut left_sigma_1_evals =
                gpu_container.ask_for(kern, n8).unwrap();
            left_sigma_1_evals
                .read_from(&prover_key.permutation.left_sigma.1.evals)
                .unwrap();
            gpu_container
                .save(
                    "prover_key.permutation.left_sigma_1_evals",
                    left_sigma_1_evals,
                )
                .unwrap();

            let mut right_sigma_1_evals =
                gpu_container.ask_for(kern, n8).unwrap();
            right_sigma_1_evals
                .read_from(&prover_key.permutation.right_sigma.1.evals)
                .unwrap();
            gpu_container
                .save(
                    "prover_key.permutation.right_sigma_1_evals",
                    right_sigma_1_evals,
                )
                .unwrap();

            let mut out_sigma_1_evals =
                gpu_container.ask_for(kern, n8).unwrap();
            out_sigma_1_evals
                .read_from(&prover_key.permutation.out_sigma.1.evals)
                .unwrap();
            gpu_container
                .save(
                    "prover_key.permutation.out_sigma_1_evals",
                    out_sigma_1_evals,
                )
                .unwrap();

            let mut fourth_sigma_1_evals =
                gpu_container.ask_for(kern, n8).unwrap();
            fourth_sigma_1_evals
                .read_from(&prover_key.permutation.fourth_sigma.1.evals)
                .unwrap();
            gpu_container
                .save(
                    "prover_key.permutation.fourth_sigma_1_evals",
                    fourth_sigma_1_evals,
                )
                .unwrap();

            let mut l1_eval_8n_gpu = gpu_container.ask_for(kern, n8).unwrap();
            let l1_eval_8n = prover_key.l1_poly_coset_8n();
            l1_eval_8n_gpu.read_from(&l1_eval_8n).unwrap();
            gpu_container
                .save("prover_key.arithmetic.l1_eval_8n_gpu", l1_eval_8n_gpu)
                .unwrap();

            let mut v_h_coset_8n_inv_evals =
                gpu_container.ask_for(kern, n8).unwrap();
            v_h_coset_8n_inv_evals
                .read_from(&prover_key.v_h_coset_8n_inv().evals)
                .unwrap();
            gpu_container
                .save(
                    "prover_key.arithmetic.v_h_coset_8n_inv_evals",
                    v_h_coset_8n_inv_evals,
                )
                .unwrap();

            let sigma_polys = (
                &prover_key.left_sigma_eval_n,
                &prover_key.right_sigma_eval_n,
                &prover_key.out_sigma_eval_n,
                &prover_key.fourth_sigma_eval_n,
            );

            let sigma_mappings =
                (sigma_polys.0, sigma_polys.1, sigma_polys.2, sigma_polys.3);
            let gatewise_sigmas = izip!(
                sigma_mappings.0,
                sigma_mappings.1,
                sigma_mappings.2,
                sigma_mappings.3
            )
            .map(|(s0, s1, s2, s3)| vec![s0, s1, s2, s3]);

            let gatewise_sigmas: Vec<F> = gatewise_sigmas
                .clone()
                .into_iter()
                .flatten()
                .cloned()
                .collect();

            let gatewise_sigmas = unsafe {
                std::mem::transmute::<_, &Vec<BigInteger256>>(&gatewise_sigmas)
            };

            let roots: Vec<F> = domain.elements().collect();
            let roots: &Vec<BigInteger256> = unsafe {
                std::mem::transmute::<_, &Vec<BigInteger256>>(&roots)
            };

            let gate_roots_gpu =
                DeviceMemory::<BigInteger256>::new(ctx, roots.len()).unwrap();
            let gatewise_sigmas_gpu =
                DeviceMemory::<BigInteger256>::new(ctx, gatewise_sigmas.len())
                    .unwrap();

            gate_roots_gpu.read_from(roots, roots.len()).unwrap();
            gatewise_sigmas_gpu
                .read_from(gatewise_sigmas.as_slice(), gatewise_sigmas.len())
                .unwrap();

            Self {
                prover_key: None,
                cs: StandardComposer::new_2(tree_height),
                preprocessed_transcript: Transcript::new(label),
                _phantom: PhantomData::<PC>,
                gate_roots: gate_roots_gpu,
                gatewise_sigmas: gatewise_sigmas_gpu,
            }
        } else {
            Self {
                prover_key: None,
                cs: StandardComposer::new_2(tree_height),
                preprocessed_transcript: Transcript::new(label),
                _phantom: PhantomData::<PC>,
                gate_roots: DeviceMemory::<BigInteger256>::new(ctx, 1).unwrap(),
                gatewise_sigmas: DeviceMemory::<BigInteger256>::new(ctx, 1)
                    .unwrap(),
            }
        }
    }

    /// Creates a new `Prover` object with some expected size.
    pub fn with_expected_size(label: &'static [u8], size: usize) -> Self {
        let ctx = &MSM_KERN.core.context;
        Self {
            prover_key: None,
            cs: StandardComposer::with_expected_size(size),
            preprocessed_transcript: Transcript::new(label),
            _phantom: PhantomData::<PC>,
            gate_roots: DeviceMemory::<BigInteger256>::new(ctx, 1).unwrap(),
            gatewise_sigmas: DeviceMemory::<BigInteger256>::new(ctx, 1)
                .unwrap(),
        }
    }

    /// Returns a mutable copy of the underlying [`StandardComposer`].
    pub fn mut_cs(&mut self) -> &mut StandardComposer<F, P> {
        &mut self.cs
    }

    /// Returns the smallest power of two needed for the curcuit.
    pub fn circuit_bound(&self) -> usize {
        self.cs.circuit_bound()
    }

    /// Preprocesses the underlying constraint system.
    pub fn preprocess(
        &mut self,
        commit_key: &PC::CommitterKey,
    ) -> Result<(), Error> {
        if self.prover_key.is_some() {
            return Err(Error::CircuitAlreadyPreprocessed);
        }
        let pk = self.cs.preprocess_prover(
            commit_key,
            &mut self.preprocessed_transcript,
            PhantomData::<PC>,
        )?;
        self.prover_key = Some(pk);
        Ok(())
    }

    #[allow(clippy::type_complexity)] // NOTE: This is an ok type for internal use.
    fn split_tx_poly(
        &self,
        n: usize,
        base_gpu_ptr: CUdeviceptr,
    ) -> Vec<CUdeviceptr> {
        let mut ret = vec![];
        let mut base_gpu_ptr = base_gpu_ptr;

        for _ in 0..8 {
            ret.push(base_gpu_ptr);
            base_gpu_ptr += (n * std::mem::size_of::<F>()) as u64;
        }
        ret
    }

    /// Convert variables to their actual witness values.
    fn to_scalars(&self, vars: &[Variable]) -> Vec<F> {
        vars.iter()
            .map(|var| self.cs.variables_vec[var.0])
            .collect()
    }

    /// Resets the witnesses in the prover object.
    ///
    /// This function is used when the user wants to make multiple proofs with
    /// the same circuit.
    pub fn clear_witness(&mut self, height: u32) {
        self.cs.reset(height);
    }

    /// Clears all data in the [`Prover`] instance.
    ///
    /// This function is used when the user wants to use the same `Prover` to
    /// make a [`Proof`] regarding a different circuit.
    pub fn clear(&mut self) {
        self.clear_witness(HEIGHT as u32);
        // self.prover_key = None;
        self.preprocessed_transcript = Transcript::new(b"plonk");
    }

    /// Keys the [`Transcript`] with additional seed information
    /// Wrapper around [`Transcript::append_message`].
    ///
    /// [`Transcript`]: merlin::Transcript
    /// [`Transcript::append_message`]: merlin::Transcript::append_message
    pub fn key_transcript(&mut self, label: &'static [u8], message: &[u8]) {
        self.preprocessed_transcript.append_message(label, message);
    }

    /// Creates a [`Proof]` that demonstrates that a circuit is satisfied.
    /// # Note
    /// If you intend to construct multiple [`Proof`]s with different witnesses,
    /// after calling this method, the user should then call
    /// [`Prover::clear_witness`].
    /// This is automatically done when [`Prover::prove`] is called.
    pub fn prove_with_preprocessed<'a, 'b, G>(
        &self,
        commit_key: &PC::CommitterKey,
        prover_key: &ProverKey<F>,
        msm_context: Option<&mut MSMContext<'a, 'b, G>>,
        gpu_container: &mut GpuPolyContainer<GpuFr>,
        _data: PhantomData<PC>,
    ) -> Result<Proof<F, PC>, Error>
    where
        G: AffineCurve,
    {
        let kern = &GPU_POLY_KERNEL;
        let msm_context = msm_context.unwrap();

        let domain = &DOMAIN;
        let n = domain.size();

        let domain = unsafe {
            std::mem::transmute::<&GeneralEvaluationDomain<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>, &GeneralEvaluationDomain<F>>(domain,)
        };

        let domain_8n = &DOMAIN_8N;
        let domain_8n = unsafe {
            std::mem::transmute::<&GeneralEvaluationDomain<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>, &GeneralEvaluationDomain<F>>(domain_8n,)
        };

        // Since the caller is passing a pre-processed circuit
        // We assume that the Transcript has been seeded with the preprocessed
        // Commitments
        let mut transcript = self.preprocessed_transcript.clone();

        // Append Public Inputs to the transcript
        transcript.append(b"pi", self.cs.get_pi());

        // 1. Compute witness Polynomials
        //
        // Convert Variables to scalars padding them to the
        // correct domain size.
        let w_l = &self.cs.w_l;
        let w_r = &self.cs.w_r;
        let w_o = &self.cs.w_o;
        let w_4 = &self.cs.w_4;
        let variables = &self.cs.variables_vec;

        let chunk_size = w_l.len() / 2;
        let (w_l_vec, w_r_vec, w_o_vec, w_4_vec) = thread::scope(|s| {
            let w_l_handler = s.spawn(|| -> &Vec<F> {
                let mut w_l_tmp = &mut *W_L_VEC.lock().unwrap();
                let mut w_l_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_l_tmp)
                };
                // let start = Instant::now();
                w_l.par_chunks(chunk_size)
                    .zip(w_l_scalars[0..w_l.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_l_scalars
            });
            let w_r_handler = s.spawn(|| -> &Vec<F> {
                let mut w_r_tmp = &mut *W_R_VEC.lock().unwrap();
                let mut w_r_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_r_tmp)
                };

                w_r.par_chunks(chunk_size)
                    .zip(w_r_scalars[0..w_r.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_r_scalars
            });
            let w_o_handler = s.spawn(|| -> &Vec<F> {
                // let start = Instant::now();
                let mut w_o_tmp = &mut *W_O_VEC.lock().unwrap();
                let mut w_o_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_o_tmp)
                };
                w_o.par_chunks(chunk_size)
                    .zip(w_o_scalars[0..w_o.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_o_scalars
            });
            let w_4_handler = s.spawn(|| -> &Vec<F> {
                // let start = Instant::now();
                let mut w_4_tmp = &mut *W_4_VEC.lock().unwrap();
                let mut w_4_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_4_tmp)
                };
                // let start = Instant::now();
                w_4.par_chunks(chunk_size)
                    .zip(w_4_scalars[0..w_4.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_4_scalars
            });
            (
                w_l_handler.join().unwrap(),
                w_r_handler.join().unwrap(),
                w_o_handler.join().unwrap(),
                w_4_handler.join().unwrap(),
            )
        });
        let w_l_scalar = w_l_vec;
        let w_r_scalar = w_r_vec;
        let w_o_scalar = w_o_vec;
        let w_4_scalar = w_4_vec;

        // Witnesses are now in evaluation form, convert them to coefficients
        // so that we may commit to them.
        let size_inv = domain.size_inv();

        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_l_scalar,
            "w_l_poly",
            n,
            size_inv,
        );
        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_r_scalar,
            "w_r_poly",
            n,
            size_inv,
        );
        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_o_scalar,
            "w_o_poly",
            n,
            size_inv,
        );
        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_4_scalar,
            "w_4_poly",
            n,
            size_inv,
        );

        // kern.sync().unwrap();

        let w_l_gpu = gpu_container.find(kern, "w_l_poly").unwrap();
        let w_r_gpu = gpu_container.find(kern, "w_r_poly").unwrap();
        let w_o_gpu = gpu_container.find(kern, "w_o_poly").unwrap();
        let w_4_gpu = gpu_container.find(kern, "w_4_poly").unwrap();

        let w_polys = vec![
            w_l_gpu.get_memory().get_inner(),
            w_r_gpu.get_memory().get_inner(),
            w_o_gpu.get_memory().get_inner(),
            w_4_gpu.get_memory().get_inner(),
        ];
        let w_poly_labels = vec![
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_o_poly".to_string(),
            "w_4_poly".to_string(),
        ];
        let (w_commits, _) = PC::commit_gpu_scalars(
            commit_key,
            w_polys,
            n,
            w_poly_labels,
            None,
            msm_context,
        )
        .map_err(to_pc_error::<F, PC>)?;

        // Add witness polynomial commitments to transcript.
        transcript.append(b"w_l", w_commits[0].commitment());
        transcript.append(b"w_r", w_commits[1].commitment());
        transcript.append(b"w_o", w_commits[2].commitment());
        transcript.append(b"w_4", w_commits[3].commitment());

        // 2. Derive lookup polynomials

        // Generate table compression factor
        let zeta = transcript.challenge_scalar(b"zeta");
        transcript.append(b"zeta", &zeta);

        // Compress lookup table into vector of single elements
        let table_poly = DensePolynomial::zero();
        let f_poly = DensePolynomial::zero();

        // Commit to query polynomial
        let f_poly_commit = <PC as PolynomialCommitment<
            F,
            DensePolynomial<F>,
        >>::Commitment::empty();
        // Add f_poly commitment to transcript
        transcript.append(b"f", &f_poly_commit);

        let h_1_poly = DensePolynomial::zero();
        let h_2_poly = DensePolynomial::zero();

        // Add blinders to h polynomials
        // let h_1_poly = Self::add_blinder(&h_1_poly, n, 1);
        // let h_2_poly = Self::add_blinder(&h_2_poly, n, 1);

        // Commit to h polys
        let h_1_poly_commit = <PC as PolynomialCommitment<
            F,
            DensePolynomial<F>,
        >>::Commitment::empty();
        let h_2_poly_commit = <PC as PolynomialCommitment<
            F,
            DensePolynomial<F>,
        >>::Commitment::empty();

        // Add h polynomials to transcript
        transcript.append(b"h1", &h_1_poly_commit);
        transcript.append(b"h2", &h_2_poly_commit);

        // 3. Compute permutation polynomial
        //
        // Compute permutation challenge `beta`.
        let beta = transcript.challenge_scalar(b"beta");
        transcript.append(b"beta", &beta);
        // Compute permutation challenge `gamma`.
        let gamma = transcript.challenge_scalar(b"gamma");
        transcript.append(b"gamma", &gamma);
        // Compute permutation challenge `delta`.
        let delta = transcript.challenge_scalar(b"delta");
        transcript.append(b"delta", &delta);

        // Compute permutation challenge `epsilon`.
        let epsilon = transcript.challenge_scalar(b"epsilon");
        transcript.append(b"epsilon", &epsilon);

        // Challenges must be different
        assert!(beta != gamma, "challenges must be different");
        assert!(beta != delta, "challenges must be different");
        assert!(beta != epsilon, "challenges must be different");
        assert!(gamma != delta, "challenges must be different");
        assert!(gamma != epsilon, "challenges must be different");
        assert!(delta != epsilon, "challenges must be different");

        self.cs.perm.compute_permutation_poly(
            &domain,
            &self.gate_roots,
            // (w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar),
            beta,
            gamma,
            // (
            //     &prover_key.left_sigma_eval_n,
            //     &prover_key.right_sigma_eval_n,
            //     &prover_key.out_sigma_eval_n,
            //     &prover_key.fourth_sigma_eval_n,
            // ),
            &self.gatewise_sigmas,
            kern,
            gpu_container,
            msm_context,
        );

        // Commit to permutation polynomial.
        let z_poly_gpu = gpu_container.find(kern, "z_poly").unwrap();

        let z_poly_label = vec!["z_poly".to_string()];
        let (z_poly_commit, _) = PC::commit_gpu_scalars(
            commit_key,
            vec![z_poly_gpu.get_memory().get_inner()],
            n,
            z_poly_label,
            None,
            msm_context,
        )
        .map_err(to_pc_error::<F, PC>)?;

        // Add permutation polynomial commitment to transcript.
        transcript.append(b"z", z_poly_commit[0].commitment());

        // Compute mega permutation polynomial.
        // Compute lookup permutation poly
        let z_2_poly = DensePolynomial::from_coefficients_slice(&[F::one()]);

        // TODO: Find strategy for blinding lookups
        // Add blinder for lookup permutation poly
        // Commit to lookup permutation polynomial.
        let (z_2_poly_commit, _) = PC::commit(
            commit_key,
            &[label_polynomial!(z_2_poly)],
            None,
            Some(msm_context),
        )
        .map_err(to_pc_error::<F, PC>)?;

        // 3. Compute public inputs polynomial.

        let pi_idx: usize = 3161923;

        // let mut pi_gpu = gpu_container.ask_for(kern, n).unwrap();
        let pi_gpu = gpu_container.find(kern, "pi_poly");
        let mut pi_gpu = match pi_gpu {
            Ok(old_pi_gpu) => old_pi_gpu,
            Err(_) => {
                let pi_gpu = gpu_container.ask_for(kern, n).unwrap();
                pi_gpu
            }
        };
        pi_gpu.fill_with_zero().unwrap();
        pi_gpu
            .add_at_offset(
                self.cs.public_inputs.get_val(pi_idx).unwrap(),
                pi_idx,
            )
            .unwrap();
        gpu_ifft_without_back_gscalars(
            kern,
            gpu_container,
            &mut pi_gpu,
            "pi_poly",
            n,
            size_inv,
        );
        // kern.sync().unwrap();
        gpu_container.save("pi_poly", pi_gpu).unwrap();

        // 4. Compute quotient polynomial
        //
        // Compute quotient challenge; `alpha`, and gate-specific separation
        // challenges.
        let alpha = transcript.challenge_scalar(b"alpha");
        transcript.append(b"alpha", &alpha);

        let range_sep_challenge: F =
            transcript.challenge_scalar(b"range separation challenge");
        transcript.append(b"range seperation challenge", &range_sep_challenge);

        let logic_sep_challenge: F =
            transcript.challenge_scalar(b"logic separation challenge");
        transcript.append(b"logic seperation challenge", &logic_sep_challenge);

        let fixed_base_sep_challenge: F =
            transcript.challenge_scalar(b"fixed base separation challenge");
        transcript.append(
            b"fixed base separation challenge",
            &fixed_base_sep_challenge,
        );

        let var_base_sep_challenge: F =
            transcript.challenge_scalar(b"variable base separation challenge");
        transcript.append(
            b"variable base separation challenge",
            &var_base_sep_challenge,
        );

        let lookup_sep_challenge =
            transcript.challenge_scalar(b"lookup separation challenge");
        transcript
            .append(b"lookup separation challenge", &lookup_sep_challenge);

        compute_quotient_gpu::<F>(
            kern,
            gpu_container,
            &domain,
            &domain_8n,
            prover_key,
            &z_poly_gpu,
            &z_2_poly,
            &w_l_gpu,
            &w_r_gpu,
            &w_o_gpu,
            &w_4_gpu,
            &f_poly,
            &table_poly,
            &h_1_poly,
            &h_2_poly,
            &alpha,
            &beta,
            &gamma,
            &delta,
            &epsilon,
            &lookup_sep_challenge,
        );
        kern.sync().unwrap();

        let mut labels = vec![];
        for i in 0..8 {
            let label = format!("t_i_polys[{}]", i);
            labels.push(label);
        }

        let t_i_polys_gpu = gpu_container.find(kern, "quotient").unwrap();
        let t_i_gpu = t_i_polys_gpu.get_memory();
        let t_i_gpu = self.split_tx_poly(n, t_i_gpu.get_inner());

        // Commit to splitted quotient polynomial
        let (mut t_commits, _) = PC::commit_gpu_scalars(
            commit_key,
            t_i_gpu.as_slice()[0..6].to_vec(),
            n,
            labels,
            None,
            msm_context,
        )
        .map_err(to_pc_error::<F, PC>)?;
        t_commits.push(LabeledCommitment::new(
            "t_i_polys[6]".to_string(),
            PCCommitment::empty(),
            None,
        ));
        t_commits.push(LabeledCommitment::new(
            "t_i_polys[7]".to_string(),
            PCCommitment::empty(),
            None,
        ));

        // gpu_container.save("quotient", t_i_polys_gpu).unwrap();
        gpu_container.recycle(t_i_polys_gpu).unwrap();

        // Add quotient polynomial commitments to transcript
        transcript.append(b"t_1", t_commits[0].commitment());
        transcript.append(b"t_2", t_commits[1].commitment());
        transcript.append(b"t_3", t_commits[2].commitment());
        transcript.append(b"t_4", t_commits[3].commitment());
        transcript.append(b"t_5", t_commits[4].commitment());
        transcript.append(b"t_6", t_commits[5].commitment());
        transcript.append(b"t_7", t_commits[6].commitment());
        transcript.append(b"t_8", t_commits[7].commitment());

        // 4. Compute linearisation polynomial
        //
        // Compute evaluation challenge; `z`.
        let z_challenge = transcript.challenge_scalar(b"z");
        transcript.append(b"z", &z_challenge);
        // todo
        let (lin_eval, evaluations) = compute_linear_gpu::<F, P>(
            kern,
            gpu_container,
            &domain,
            prover_key,
            &alpha,
            &beta,
            &gamma,
            &delta,
            &epsilon,
            &zeta,
            &lookup_sep_challenge,
            &z_challenge,
            &w_l_gpu,
            &w_r_gpu,
            &w_o_gpu,
            &w_4_gpu,
            t_i_gpu[0],
            t_i_gpu[1],
            t_i_gpu[2],
            t_i_gpu[3],
            t_i_gpu[4],
            t_i_gpu[5],
            &z_poly_gpu,
            &z_2_poly,
            &f_poly,
            &h_1_poly,
            &h_2_poly,
            &table_poly,
        );
        gpu_container.save("w_l_poly", w_l_gpu).unwrap();
        gpu_container.save("w_r_poly", w_r_gpu).unwrap();
        gpu_container.save("w_o_poly", w_o_gpu).unwrap();
        gpu_container.save("w_4_poly", w_4_gpu).unwrap();
        gpu_container.save("z_poly", z_poly_gpu).unwrap();

        // Add evaluations to transcript.
        // First wire evals
        transcript.append(b"a_eval", &evaluations.wire_evals.a_eval);
        transcript.append(b"b_eval", &evaluations.wire_evals.b_eval);
        transcript.append(b"c_eval", &evaluations.wire_evals.c_eval);
        transcript.append(b"d_eval", &evaluations.wire_evals.d_eval);

        // Second permutation evals
        transcript
            .append(b"left_sig_eval", &evaluations.perm_evals.left_sigma_eval);
        transcript.append(
            b"right_sig_eval",
            &evaluations.perm_evals.right_sigma_eval,
        );
        transcript
            .append(b"out_sig_eval", &evaluations.perm_evals.out_sigma_eval);
        transcript
            .append(b"perm_eval", &evaluations.perm_evals.permutation_eval);

        // Third lookup evals
        transcript.append(b"f_eval", &evaluations.lookup_evals.f_eval);
        transcript
            .append(b"q_lookup_eval", &evaluations.lookup_evals.q_lookup_eval);
        transcript.append(
            b"lookup_perm_eval",
            &evaluations.lookup_evals.z2_next_eval,
        );
        transcript.append(b"h_1_eval", &evaluations.lookup_evals.h1_eval);
        transcript
            .append(b"h_1_next_eval", &evaluations.lookup_evals.h1_next_eval);
        transcript.append(b"h_2_eval", &evaluations.lookup_evals.h2_eval);

        // Third, all evals needed for custom gates
        evaluations
            .custom_evals
            .vals
            .iter()
            .for_each(|(label, eval)| {
                let static_label = Box::leak(label.to_owned().into_boxed_str());
                transcript.append(static_label.as_bytes(), eval);
            });

        // 5. Compute Openings using KZG10
        //
        // We merge the quotient polynomial using the `z_challenge` so the SRS
        // is linear in the circuit size `n`

        // Compute aggregate witness to polynomials evaluated at the evaluation
        // challenge `z`
        let aw_challenge: F = transcript.challenge_scalar(b"aggregate_witness");

        // XXX: The quotient polynomials is used here and then in the
        // opening poly. It is being left in for now but it may not
        // be necessary. Warrants further investigation.
        // Ditto with the out_sigma poly.

        let aw_poly_labels = vec![
            "lin_poly".to_string(),
            "prover_key.permutation.left_sigma".to_string(),
            "prover_key.permutation.right_sigma".to_string(),
            "prover_key.permutation.out_sigma".to_string(),
            "f_poly".to_string(),
            "h_2_poly".to_string(),
            "table_poly".to_string(),
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_o_poly".to_string(),
            "w_4_poly".to_string(),
        ];
        let aw_poly_evals = vec![
            lin_eval.neg(),
            evaluations.perm_evals.left_sigma_eval.neg(),
            evaluations.perm_evals.right_sigma_eval.neg(),
            evaluations.perm_evals.out_sigma_eval.neg(),
            F::zero(),
            F::zero(),
            F::zero(),
            evaluations.wire_evals.a_eval.neg(),
            evaluations.wire_evals.b_eval.neg(),
            evaluations.wire_evals.c_eval.neg(),
            evaluations.wire_evals.d_eval.neg(),
        ];
        let aw_opening = open_gpu::<G, F, PC>(
            kern,
            gpu_container,
            &commit_key,
            &aw_poly_labels,
            &aw_poly_evals,
            z_challenge.clone(),
            aw_challenge,
            &size_inv,
            n,
            msm_context,
            false,
        )
        .unwrap();

        let saw_challenge: F =
            transcript.challenge_scalar(b"aggregate_witness");

        let saw_poly_labels = vec![
            "z_poly".to_string(),
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_4_poly".to_string(),
            "h_1_poly".to_string(),
            "z_2_poly".to_string(),
            "table_poly".to_string(),
        ];
        let saw_poly_evals = vec![
            evaluations.perm_evals.permutation_eval.neg(),
            evaluations.custom_evals.vals[7].1.neg(),
            evaluations.custom_evals.vals[8].1.neg(),
            evaluations.custom_evals.vals[9].1.neg(),
            F::zero(),
            F::one(),
            F::zero(),
        ];
        let saw_opening = open_gpu::<G, F, PC>(
            kern,
            gpu_container,
            &commit_key,
            &saw_poly_labels,
            &saw_poly_evals,
            z_challenge * domain.element(1),
            saw_challenge,
            &size_inv,
            n,
            msm_context,
            true,
        )
        .unwrap();

        let recycle_labels = vec![
            "z_poly".to_string(),
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_4_poly".to_string(),
            "lin_poly".to_string(),
            "w_o_poly".to_string(),
        ];

        for label in recycle_labels.iter() {
            let gpu_poly = gpu_container.find(kern, label.as_str()).unwrap();
            gpu_container.recycle(gpu_poly).unwrap();
        }

        let ret = Proof {
            a_comm: w_commits[0].commitment().clone(),
            b_comm: w_commits[1].commitment().clone(),
            c_comm: w_commits[2].commitment().clone(),
            d_comm: w_commits[3].commitment().clone(),
            z_comm: z_poly_commit[0].commitment().clone(),
            f_comm: f_poly_commit,
            h_1_comm: h_1_poly_commit,
            h_2_comm: h_2_poly_commit,
            z_2_comm: z_2_poly_commit[0].commitment().clone(),
            t_1_comm: t_commits[0].commitment().clone(),
            t_2_comm: t_commits[1].commitment().clone(),
            t_3_comm: t_commits[2].commitment().clone(),
            t_4_comm: t_commits[3].commitment().clone(),
            t_5_comm: t_commits[4].commitment().clone(),
            t_6_comm: t_commits[5].commitment().clone(),
            t_7_comm: t_commits[6].commitment().clone(),
            t_8_comm: t_commits[7].commitment().clone(),
            aw_opening,
            saw_opening,
            evaluations,
        };

        Ok(ret)
    }

    pub fn prove_with_preprocessed_with_permutation<'a, 'b, G>(
        &self,
        commit_key: &PC::CommitterKey,
        prover_key: &ProverKey<F>,
        msm_context: Option<&mut MSMContext<'a, 'b, G>>,
        gpu_container: &mut GpuPolyContainer<GpuFr>,
        permutation: &Permutation,
        _data: PhantomData<PC>,
    ) -> Result<Proof<F, PC>, Error>
    where
        G: AffineCurve,
    {
        let kern = &GPU_POLY_KERNEL;
        let msm_context = msm_context.unwrap();

        let domain = &DOMAIN;
        let n = domain.size();

        let domain = unsafe {
            std::mem::transmute::<&GeneralEvaluationDomain<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>, &GeneralEvaluationDomain<F>>(domain,)
        };

        let domain_8n = &DOMAIN_8N;
        let domain_8n = unsafe {
            std::mem::transmute::<&GeneralEvaluationDomain<<ark_ec::models::bls12::Bls12<ark_bls12_381::Parameters> as PairingEngine>::Fr>, &GeneralEvaluationDomain<F>>(domain_8n,)
        };

        // Since the caller is passing a pre-processed circuit
        // We assume that the Transcript has been seeded with the preprocessed
        // Commitments
        let mut transcript = self.preprocessed_transcript.clone();

        // Append Public Inputs to the transcript
        transcript.append(b"pi", self.cs.get_pi());

        // 1. Compute witness Polynomials
        //
        // Convert Variables to scalars padding them to the
        // correct domain size.
        let w_l = &self.cs.w_l;
        let w_r = &self.cs.w_r;
        let w_o = &self.cs.w_o;
        let w_4 = &self.cs.w_4;
        let variables = &self.cs.variables_vec;

        let chunk_size = w_l.len() / 2;
        let (w_l_vec, w_r_vec, w_o_vec, w_4_vec) = thread::scope(|s| {
            let w_l_handler = s.spawn(|| -> &Vec<F> {
                let mut w_l_tmp = &mut *W_L_VEC.lock().unwrap();
                let mut w_l_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_l_tmp)
                };
                // let start = Instant::now();
                w_l.par_chunks(chunk_size)
                    .zip(w_l_scalars[0..w_l.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_l_scalars
            });
            let w_r_handler = s.spawn(|| -> &Vec<F> {
                let mut w_r_tmp = &mut *W_R_VEC.lock().unwrap();
                let mut w_r_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_r_tmp)
                };

                w_r.par_chunks(chunk_size)
                    .zip(w_r_scalars[0..w_r.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_r_scalars
            });
            let w_o_handler = s.spawn(|| -> &Vec<F> {
                // let start = Instant::now();
                let mut w_o_tmp = &mut *W_O_VEC.lock().unwrap();
                let mut w_o_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_o_tmp)
                };
                w_o.par_chunks(chunk_size)
                    .zip(w_o_scalars[0..w_o.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_o_scalars
            });
            let w_4_handler = s.spawn(|| -> &Vec<F> {
                // let start = Instant::now();
                let mut w_4_tmp = &mut *W_4_VEC.lock().unwrap();
                let mut w_4_scalars = unsafe {
                    std::mem::transmute::<
                        &mut Vec<
                            <ark_ec::models::bls12::Bls12<
                                ark_bls12_381::Parameters,
                            > as PairingEngine>::Fr,
                        >,
                        &mut Vec<F>,
                    >(w_4_tmp)
                };
                // let start = Instant::now();
                w_4.par_chunks(chunk_size)
                    .zip(w_4_scalars[0..w_4.len()].par_chunks_mut(chunk_size))
                    .for_each(|(idx_slice, dst_slice)| {
                        idx_slice.iter().zip(dst_slice.iter_mut()).for_each(
                            |(idx, item)| {
                                *item = variables.as_slice()[idx.0];
                            },
                        );
                    });
                w_4_scalars
            });
            (
                w_l_handler.join().unwrap(),
                w_r_handler.join().unwrap(),
                w_o_handler.join().unwrap(),
                w_4_handler.join().unwrap(),
            )
        });
        let w_l_scalar = w_l_vec;
        let w_r_scalar = w_r_vec;
        let w_o_scalar = w_o_vec;
        let w_4_scalar = w_4_vec;

        // Witnesses are now in evaluation form, convert them to coefficients
        // so that we may commit to them.
        let size_inv = domain.size_inv();

        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_l_scalar,
            "w_l_poly",
            n,
            size_inv,
        );
        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_r_scalar,
            "w_r_poly",
            n,
            size_inv,
        );
        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_o_scalar,
            "w_o_poly",
            n,
            size_inv,
        );
        gpu_ifft_without_back_orig_data(
            kern,
            gpu_container,
            w_4_scalar,
            "w_4_poly",
            n,
            size_inv,
        );

        // kern.sync().unwrap();

        let w_l_gpu = gpu_container.find(kern, "w_l_poly").unwrap();
        let w_r_gpu = gpu_container.find(kern, "w_r_poly").unwrap();
        let w_o_gpu = gpu_container.find(kern, "w_o_poly").unwrap();
        let w_4_gpu = gpu_container.find(kern, "w_4_poly").unwrap();

        let w_polys = vec![
            w_l_gpu.get_memory().get_inner(),
            w_r_gpu.get_memory().get_inner(),
            w_o_gpu.get_memory().get_inner(),
            w_4_gpu.get_memory().get_inner(),
        ];
        let w_poly_labels = vec![
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_o_poly".to_string(),
            "w_4_poly".to_string(),
        ];
        let (w_commits, _) = PC::commit_gpu_scalars(
            commit_key,
            w_polys,
            n,
            w_poly_labels,
            None,
            msm_context,
        )
        .map_err(to_pc_error::<F, PC>)?;

        // Add witness polynomial commitments to transcript.
        transcript.append(b"w_l", w_commits[0].commitment());
        transcript.append(b"w_r", w_commits[1].commitment());
        transcript.append(b"w_o", w_commits[2].commitment());
        transcript.append(b"w_4", w_commits[3].commitment());

        // 2. Derive lookup polynomials

        // Generate table compression factor
        let zeta = transcript.challenge_scalar(b"zeta");
        transcript.append(b"zeta", &zeta);

        // Compress lookup table into vector of single elements
        let table_poly = DensePolynomial::zero();
        let f_poly = DensePolynomial::zero();

        // Commit to query polynomial
        let f_poly_commit = <PC as PolynomialCommitment<
            F,
            DensePolynomial<F>,
        >>::Commitment::empty();
        // Add f_poly commitment to transcript
        transcript.append(b"f", &f_poly_commit);

        let h_1_poly = DensePolynomial::zero();
        let h_2_poly = DensePolynomial::zero();

        // Add blinders to h polynomials
        // let h_1_poly = Self::add_blinder(&h_1_poly, n, 1);
        // let h_2_poly = Self::add_blinder(&h_2_poly, n, 1);

        // Commit to h polys
        let h_1_poly_commit = <PC as PolynomialCommitment<
            F,
            DensePolynomial<F>,
        >>::Commitment::empty();
        let h_2_poly_commit = <PC as PolynomialCommitment<
            F,
            DensePolynomial<F>,
        >>::Commitment::empty();

        // Add h polynomials to transcript
        transcript.append(b"h1", &h_1_poly_commit);
        transcript.append(b"h2", &h_2_poly_commit);

        // 3. Compute permutation polynomial
        //
        // Compute permutation challenge `beta`.
        let beta = transcript.challenge_scalar(b"beta");
        transcript.append(b"beta", &beta);
        // Compute permutation challenge `gamma`.
        let gamma = transcript.challenge_scalar(b"gamma");
        transcript.append(b"gamma", &gamma);
        // Compute permutation challenge `delta`.
        let delta = transcript.challenge_scalar(b"delta");
        transcript.append(b"delta", &delta);

        // Compute permutation challenge `epsilon`.
        let epsilon = transcript.challenge_scalar(b"epsilon");
        transcript.append(b"epsilon", &epsilon);

        // Challenges must be different
        assert!(beta != gamma, "challenges must be different");
        assert!(beta != delta, "challenges must be different");
        assert!(beta != epsilon, "challenges must be different");
        assert!(gamma != delta, "challenges must be different");
        assert!(gamma != epsilon, "challenges must be different");
        assert!(delta != epsilon, "challenges must be different");

        //self.cs.perm.compute_permutation_poly(
        permutation.compute_permutation_poly(
            &domain,
            &self.gate_roots,
            beta,
            gamma,
            &self.gatewise_sigmas,
            kern,
            gpu_container,
            msm_context,
        );

        // Commit to permutation polynomial.
        let z_poly_gpu = gpu_container.find(kern, "z_poly").unwrap();

        let z_poly_label = vec!["z_poly".to_string()];
        let (z_poly_commit, _) = PC::commit_gpu_scalars(
            commit_key,
            vec![z_poly_gpu.get_memory().get_inner()],
            n,
            z_poly_label,
            None,
            msm_context,
        )
        .map_err(to_pc_error::<F, PC>)?;

        // Add permutation polynomial commitment to transcript.
        transcript.append(b"z", z_poly_commit[0].commitment());

        // Compute mega permutation polynomial.
        // Compute lookup permutation poly
        let z_2_poly = DensePolynomial::from_coefficients_slice(&[F::one()]);

        // TODO: Find strategy for blinding lookups
        // Add blinder for lookup permutation poly
        // Commit to lookup permutation polynomial.
        let (z_2_poly_commit, _) = PC::commit(
            commit_key,
            &[label_polynomial!(z_2_poly)],
            None,
            Some(msm_context),
        )
        .map_err(to_pc_error::<F, PC>)?;

        // 3. Compute public inputs polynomial.
        let pi_idx: usize = 3161923;

        // let mut pi_gpu = gpu_container.ask_for(kern, n).unwrap();
        let pi_gpu = gpu_container.find(kern, "pi_poly");
        let mut pi_gpu = match pi_gpu {
            Ok(old_pi_gpu) => old_pi_gpu,
            Err(_) => {
                let pi_gpu = gpu_container.ask_for(kern, n).unwrap();
                pi_gpu
            }
        };
        pi_gpu.fill_with_zero().unwrap();
        pi_gpu
            .add_at_offset(
                self.cs.public_inputs.get_val(pi_idx).unwrap(),
                pi_idx,
            )
            .unwrap();
        gpu_ifft_without_back_gscalars(
            kern,
            gpu_container,
            &mut pi_gpu,
            "pi_poly",
            n,
            size_inv,
        );
        // kern.sync().unwrap();
        gpu_container.save("pi_poly", pi_gpu).unwrap();

        // 4. Compute quotient polynomial
        //
        // Compute quotient challenge; `alpha`, and gate-specific separation
        // challenges.
        let alpha = transcript.challenge_scalar(b"alpha");
        transcript.append(b"alpha", &alpha);

        let range_sep_challenge: F =
            transcript.challenge_scalar(b"range separation challenge");
        transcript.append(b"range seperation challenge", &range_sep_challenge);

        let logic_sep_challenge: F =
            transcript.challenge_scalar(b"logic separation challenge");
        transcript.append(b"logic seperation challenge", &logic_sep_challenge);

        let fixed_base_sep_challenge: F =
            transcript.challenge_scalar(b"fixed base separation challenge");
        transcript.append(
            b"fixed base separation challenge",
            &fixed_base_sep_challenge,
        );

        let var_base_sep_challenge: F =
            transcript.challenge_scalar(b"variable base separation challenge");
        transcript.append(
            b"variable base separation challenge",
            &var_base_sep_challenge,
        );

        let lookup_sep_challenge =
            transcript.challenge_scalar(b"lookup separation challenge");
        transcript
            .append(b"lookup separation challenge", &lookup_sep_challenge);

        let start = std::time::Instant::now();
        compute_quotient_gpu::<F>(
            kern,
            gpu_container,
            &domain,
            &domain_8n,
            prover_key,
            &z_poly_gpu,
            &z_2_poly,
            &w_l_gpu,
            &w_r_gpu,
            &w_o_gpu,
            &w_4_gpu,
            &f_poly,
            &table_poly,
            &h_1_poly,
            &h_2_poly,
            &alpha,
            &beta,
            &gamma,
            &delta,
            &epsilon,
            &lookup_sep_challenge,
        );
        kern.sync().unwrap();

        let mut labels = vec![];
        for i in 0..8 {
            let label = format!("t_i_polys[{}]", i);
            labels.push(label);
        }

        let t_i_polys_gpu = gpu_container.find(kern, "quotient").unwrap();
        let t_i_gpu = t_i_polys_gpu.get_memory();
        let t_i_gpu = self.split_tx_poly(n, t_i_gpu.get_inner());

        // Commit to splitted quotient polynomial
        let (mut t_commits, _) = PC::commit_gpu_scalars(
            commit_key,
            t_i_gpu.as_slice()[0..6].to_vec(),
            n,
            labels,
            None,
            msm_context,
        )
        .map_err(to_pc_error::<F, PC>)?;
        t_commits.push(LabeledCommitment::new(
            "t_i_polys[6]".to_string(),
            PCCommitment::empty(),
            None,
        ));
        t_commits.push(LabeledCommitment::new(
            "t_i_polys[7]".to_string(),
            PCCommitment::empty(),
            None,
        ));

        // gpu_container.save("quotient", t_i_polys_gpu).unwrap();
        gpu_container.recycle(t_i_polys_gpu).unwrap();

        // Add quotient polynomial commitments to transcript
        transcript.append(b"t_1", t_commits[0].commitment());
        transcript.append(b"t_2", t_commits[1].commitment());
        transcript.append(b"t_3", t_commits[2].commitment());
        transcript.append(b"t_4", t_commits[3].commitment());
        transcript.append(b"t_5", t_commits[4].commitment());
        transcript.append(b"t_6", t_commits[5].commitment());
        transcript.append(b"t_7", t_commits[6].commitment());
        transcript.append(b"t_8", t_commits[7].commitment());

        // 4. Compute linearisation polynomial
        //
        // Compute evaluation challenge; `z`.
        let z_challenge = transcript.challenge_scalar(b"z");
        transcript.append(b"z", &z_challenge);
        // todo
        let (lin_eval, evaluations) = compute_linear_gpu::<F, P>(
            kern,
            gpu_container,
            &domain,
            prover_key,
            &alpha,
            &beta,
            &gamma,
            &delta,
            &epsilon,
            &zeta,
            &lookup_sep_challenge,
            &z_challenge,
            &w_l_gpu,
            &w_r_gpu,
            &w_o_gpu,
            &w_4_gpu,
            t_i_gpu[0],
            t_i_gpu[1],
            t_i_gpu[2],
            t_i_gpu[3],
            t_i_gpu[4],
            t_i_gpu[5],
            &z_poly_gpu,
            &z_2_poly,
            &f_poly,
            &h_1_poly,
            &h_2_poly,
            &table_poly,
        );
        gpu_container.save("w_l_poly", w_l_gpu).unwrap();
        gpu_container.save("w_r_poly", w_r_gpu).unwrap();
        gpu_container.save("w_o_poly", w_o_gpu).unwrap();
        gpu_container.save("w_4_poly", w_4_gpu).unwrap();
        gpu_container.save("z_poly", z_poly_gpu).unwrap();

        // Add evaluations to transcript.
        // First wire evals
        transcript.append(b"a_eval", &evaluations.wire_evals.a_eval);
        transcript.append(b"b_eval", &evaluations.wire_evals.b_eval);
        transcript.append(b"c_eval", &evaluations.wire_evals.c_eval);
        transcript.append(b"d_eval", &evaluations.wire_evals.d_eval);

        // Second permutation evals
        transcript
            .append(b"left_sig_eval", &evaluations.perm_evals.left_sigma_eval);
        transcript.append(
            b"right_sig_eval",
            &evaluations.perm_evals.right_sigma_eval,
        );
        transcript
            .append(b"out_sig_eval", &evaluations.perm_evals.out_sigma_eval);
        transcript
            .append(b"perm_eval", &evaluations.perm_evals.permutation_eval);

        // Third lookup evals
        transcript.append(b"f_eval", &evaluations.lookup_evals.f_eval);
        transcript
            .append(b"q_lookup_eval", &evaluations.lookup_evals.q_lookup_eval);
        transcript.append(
            b"lookup_perm_eval",
            &evaluations.lookup_evals.z2_next_eval,
        );
        transcript.append(b"h_1_eval", &evaluations.lookup_evals.h1_eval);
        transcript
            .append(b"h_1_next_eval", &evaluations.lookup_evals.h1_next_eval);
        transcript.append(b"h_2_eval", &evaluations.lookup_evals.h2_eval);

        // Third, all evals needed for custom gates
        evaluations
            .custom_evals
            .vals
            .iter()
            .for_each(|(label, eval)| {
                let static_label = Box::leak(label.to_owned().into_boxed_str());
                transcript.append(static_label.as_bytes(), eval);
            });

        // 5. Compute Openings using KZG10
        //
        // We merge the quotient polynomial using the `z_challenge` so the SRS
        // is linear in the circuit size `n`

        // Compute aggregate witness to polynomials evaluated at the evaluation
        // challenge `z`
        let aw_challenge: F = transcript.challenge_scalar(b"aggregate_witness");

        // XXX: The quotient polynomials is used here and then in the
        // opening poly. It is being left in for now but it may not
        // be necessary. Warrants further investigation.
        // Ditto with the out_sigma poly.

        let aw_poly_labels = vec![
            "lin_poly".to_string(),
            "prover_key.permutation.left_sigma".to_string(),
            "prover_key.permutation.right_sigma".to_string(),
            "prover_key.permutation.out_sigma".to_string(),
            "f_poly".to_string(),
            "h_2_poly".to_string(),
            "table_poly".to_string(),
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_o_poly".to_string(),
            "w_4_poly".to_string(),
        ];
        let aw_poly_evals = vec![
            lin_eval.neg(),
            evaluations.perm_evals.left_sigma_eval.neg(),
            evaluations.perm_evals.right_sigma_eval.neg(),
            evaluations.perm_evals.out_sigma_eval.neg(),
            F::zero(),
            F::zero(),
            F::zero(),
            evaluations.wire_evals.a_eval.neg(),
            evaluations.wire_evals.b_eval.neg(),
            evaluations.wire_evals.c_eval.neg(),
            evaluations.wire_evals.d_eval.neg(),
        ];
        let aw_opening = open_gpu::<G, F, PC>(
            kern,
            gpu_container,
            &commit_key,
            &aw_poly_labels,
            &aw_poly_evals,
            z_challenge.clone(),
            aw_challenge,
            &size_inv,
            n,
            msm_context,
            false,
        )
        .unwrap();

        let saw_challenge: F =
            transcript.challenge_scalar(b"aggregate_witness");

        let saw_poly_labels = vec![
            "z_poly".to_string(),
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_4_poly".to_string(),
            "h_1_poly".to_string(),
            "z_2_poly".to_string(),
            "table_poly".to_string(),
        ];
        let saw_poly_evals = vec![
            evaluations.perm_evals.permutation_eval.neg(),
            evaluations.custom_evals.vals[7].1.neg(),
            evaluations.custom_evals.vals[8].1.neg(),
            evaluations.custom_evals.vals[9].1.neg(),
            F::zero(),
            F::one(),
            F::zero(),
        ];
        let saw_opening = open_gpu::<G, F, PC>(
            kern,
            gpu_container,
            &commit_key,
            &saw_poly_labels,
            &saw_poly_evals,
            z_challenge * domain.element(1),
            saw_challenge,
            &size_inv,
            n,
            msm_context,
            true,
        )
        .unwrap();

        let recycle_labels = vec![
            "z_poly".to_string(),
            "w_l_poly".to_string(),
            "w_r_poly".to_string(),
            "w_4_poly".to_string(),
            "lin_poly".to_string(),
            "w_o_poly".to_string(),
        ];

        for label in recycle_labels.iter() {
            let gpu_poly = gpu_container.find(kern, label.as_str()).unwrap();
            gpu_container.recycle(gpu_poly).unwrap();
        }

        let ret = Proof {
            a_comm: w_commits[0].commitment().clone(),
            b_comm: w_commits[1].commitment().clone(),
            c_comm: w_commits[2].commitment().clone(),
            d_comm: w_commits[3].commitment().clone(),
            z_comm: z_poly_commit[0].commitment().clone(),
            f_comm: f_poly_commit,
            h_1_comm: h_1_poly_commit,
            h_2_comm: h_2_poly_commit,
            z_2_comm: z_2_poly_commit[0].commitment().clone(),
            t_1_comm: t_commits[0].commitment().clone(),
            t_2_comm: t_commits[1].commitment().clone(),
            t_3_comm: t_commits[2].commitment().clone(),
            t_4_comm: t_commits[3].commitment().clone(),
            t_5_comm: t_commits[4].commitment().clone(),
            t_6_comm: t_commits[5].commitment().clone(),
            t_7_comm: t_commits[6].commitment().clone(),
            t_8_comm: t_commits[7].commitment().clone(),
            aw_opening,
            saw_opening,
            evaluations,
        };

        Ok(ret)
    }

    /// Proves a circuit is satisfied, then clears the witness variables
    /// If the circuit is not pre-processed, then the preprocessed circuit will
    /// also be computed.
    pub fn prove_with_permutation<'a, 'b, G>(
        &mut self,
        commit_key: &PC::CommitterKey,
        prover_key: &ProverKey<F>,

        msm_context: Option<&mut MSMContext<'a, 'b, G>>,
        gpu_container: &mut GpuPolyContainer<GpuFr>,
        permutation: &Permutation,
    ) -> Result<Proof<F, PC>, Error>
    where
        G: AffineCurve,
    {
        // if self.prover_key.is_none() {
        //     // Preprocess circuit and store preprocessed circuit and
        // transcript     // in the Prover.
        //     self.prover_key = Some(self.cs.preprocess_prover(
        //         commit_key,
        //         &mut self.preprocessed_transcript,
        //         PhantomData::<PC>,
        //     )?);
        // }

        // let prover_key = self.prover_key.as_ref().unwrap();

        let proof = self.prove_with_preprocessed_with_permutation(
            commit_key,
            prover_key,
            msm_context,
            gpu_container,
            permutation,
            PhantomData::<PC>,
        )?;

        // Clear witness and reset composer variables
        self.clear_witness(HEIGHT as u32);
        Ok(proof)
    }

    pub fn prove<'a, 'b, G>(
        &mut self,
        commit_key: &PC::CommitterKey,
        prover_key: &ProverKey<F>,

        msm_context: Option<&mut MSMContext<'a, 'b, G>>,
        gpu_container: &mut GpuPolyContainer<GpuFr>,
    ) -> Result<Proof<F, PC>, Error>
    where
        G: AffineCurve,
    {
        // if self.prover_key.is_none() {
        //     // Preprocess circuit and store preprocessed circuit and
        // transcript     // in the Prover.
        //     self.prover_key = Some(self.cs.preprocess_prover(
        //         commit_key,
        //         &mut self.preprocessed_transcript,
        //         PhantomData::<PC>,
        //     )?);
        // }

        // let prover_key = self.prover_key.as_ref().unwrap();

        let proof = self.prove_with_preprocessed(
            commit_key,
            prover_key,
            msm_context,
            gpu_container,
            PhantomData::<PC>,
        )?;

        // Clear witness and reset composer variables
        self.clear_witness(HEIGHT as u32);
        Ok(proof)
    }
}

impl<F, P, PC> Default for Prover<F, P, PC>
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
{
    #[inline]
    fn default() -> Self {
        let kern = &MSM_KERN;
        Prover::new_2(
            b"plonk",
            &kern.core.context,
            HEIGHT as u32,
            None,
            None,
            None,
            None,
        )
    }
}
