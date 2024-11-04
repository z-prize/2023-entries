// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Prover-side of the PLONK Proving System

use crate::lookup::MultiSet;
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

use crate::permutation::gpu_ifft;
use ark_ec::{AffineCurve, ModelParameters, PairingEngine, TEModelParameters};
use ark_ff::PrimeField;
use ark_ff::Zero;

use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain,
    UVPolynomial,
};
use core::marker::PhantomData;
use itertools::izip;
use merlin::Transcript;

use crate::util::EvaluationDomainExt;
use std::time::Instant;

use ec_gpu_common::{
    Fr as GpuFr, GPUSourceCore, GpuPolyContainer, MSMContext,
    MsmPrecalcContainer, MultiexpKernel, PolyKernel, GPU_CUDA_CORES,
};

use ec_gpu_common::api::{*};
use ark_bls12_381::Bls12_381;
use ark_serialize::CanonicalSerialize;
use ark_ff::ToBytes;
use ark_poly_commit::kzg10::Commitment;
use ark_poly_commit::PolynomialCommitment;
use ark_poly_commit::LabeledCommitment;

use std::os::raw::c_void;
use std::os::raw::c_int;
use std::mem;

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

    _phantom: PhantomData<PC>,
}
impl<F, P, PC> Prover<F, P, PC>
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
{
    /// Creates a new `Prover` instance.
    pub fn new(label: &'static [u8]) -> Self {
        Self {
            prover_key: None,
            cs: StandardComposer::new(),
            preprocessed_transcript: Transcript::new(label),
            _phantom: PhantomData::<PC>,
        }
    }

    /// Creates a new `Prover` object with some expected size.
    pub fn with_expected_size(label: &'static [u8], size: usize) -> Self {
        Self {
            prover_key: None,
            cs: StandardComposer::with_expected_size(size),
            preprocessed_transcript: Transcript::new(label),
            _phantom: PhantomData::<PC>,
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

    /// Split `t(X)` poly into 8 n-sized polynomials.
    #[allow(clippy::type_complexity)] // NOTE: This is an ok type for internal use.
    fn split_tx_poly(
        &self,
        n: usize,
        t_x: &DensePolynomial<F>,
    ) -> [DensePolynomial<F>; 8] {
        let mut buf = t_x.coeffs.to_vec();
        buf.resize(n << 3, F::zero());

        [
            DensePolynomial::from_coefficients_vec(buf[0..n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[n..2 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[2 * n..3 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[3 * n..4 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[4 * n..5 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[5 * n..6 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[6 * n..7 * n].to_vec()),
            DensePolynomial::from_coefficients_vec(buf[7 * n..].to_vec()),
        ]
    }

    /// Convert variables to their actual witness values.
    fn to_scalars(&self, vars: &[Variable]) -> Vec<F> {
        vars.iter().map(|var| self.cs.variables[var]).collect()
    }

    /// Resets the witnesses in the prover object.
    ///
    /// This function is used when the user wants to make multiple proofs with
    /// the same circuit.
    pub fn clear_witness(&mut self) {
        self.cs = StandardComposer::new();
    }

    /// Clears all data in the [`Prover`] instance.
    ///
    /// This function is used when the user wants to use the same `Prover` to
    /// make a [`Proof`] regarding a different circuit.
    pub fn clear(&mut self) {
        self.clear_witness();
        self.prover_key = None;
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
        msm_context: Option<&MSMContext<'a, 'b, G>>,
        _data: PhantomData<PC>,
    ) -> Result<Proof<F, PC>, Error>
    where
        G: AffineCurve,
    {
        let start = std::time::Instant::now();
        
        let domain =
            GeneralEvaluationDomain::new(self.cs.circuit_bound()).ok_or(Error::InvalidEvalDomainSize {
                log_size_of_group: self.cs.circuit_bound().trailing_zeros(),
                adicity: <<F as ark_ff::FftField>::FftParams as ark_ff::FftParameters>::TWO_ADICITY,
            })?;

        let domain_8n = GeneralEvaluationDomain::new(8 * domain.size())
            .ok_or(Error::InvalidEvalDomainSize {
            log_size_of_group: (8 * domain.size()).trailing_zeros(),
            adicity: <<F as ark_ff::FftField>::FftParams as ark_ff::FftParameters>::TWO_ADICITY,
        })?;
        
        /*unsafe { 
            registerAllArrays(domain.size() as i32, 
                              prover_key.v_h_coset_8n().evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_m.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_l.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_r.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_o.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_4.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_hl.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_hr.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_h4.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_c.1.evals.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_arith.1.evals.as_ptr() as *const c_void,
                              prover_key.permutation.linear_evaluations.evals.as_ptr() as *const c_void,
                              prover_key.permutation.left_sigma.1.evals.as_ptr() as *const c_void,
                              prover_key.permutation.right_sigma.1.evals.as_ptr() as *const c_void,
                              prover_key.permutation.out_sigma.1.evals.as_ptr() as *const c_void,
                              prover_key.permutation.fourth_sigma.1.evals.as_ptr() as *const c_void,
                              
                              prover_key.arithmetic.q_arith.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_c.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_l.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_r.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_hl.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_hr.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_h4.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_m.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_4.0.coeffs.as_ptr() as *const c_void,
                              prover_key.arithmetic.q_o.0.coeffs.as_ptr() as *const c_void);
        }*/

        let n = domain.size();

        let omega = domain.group_gen();
        let omega_8n = domain_8n.group_gen();
        let omega_inv_8n = domain_8n.group_gen_inv();
        let omega_inv = domain.group_gen_inv();

        let size_inv = unsafe { F::from_str(&format!("{}", domain.size())).unwrap_unchecked() };
        let size_inv = size_inv.inverse().expect("m must have inverse");

        let size_8inv = unsafe { F::from_str(&format!("{}", domain_8n.size())).unwrap_unchecked() };
        let size_8inv = size_8inv.inverse().expect("m must have inverse");

        let K1 = F::from(7_u64);
        let K2 = F::from(13_u64);
        let K3 = F::from(17_u64);

        let g = F::multiplicative_generator();
        let g_inv = F::multiplicative_generator().inverse().expect("m must have inverse");

        let mut g_vec = Vec::<F>::new(); g_vec.resize(2, F::zero()); g_vec[0] = g; g_vec[1] = g_inv;
        let mut size_inv_vec = Vec::<F>::new(); size_inv_vec.resize(2, F::zero()); size_inv_vec[0] = size_inv; size_inv_vec[1] = size_8inv;

        let mut omegas_vec = Vec::<F>::new(); omegas_vec.resize(2, F::zero()); omegas_vec[0] = omega; omegas_vec[1] = omega_8n;
        let mut omegas_inv_vec = Vec::<F>::new(); omegas_inv_vec.resize(2, F::zero()); omegas_inv_vec[0] = omega_inv; omegas_inv_vec[1] = omega_inv_8n;
        let empty_vec = Vec::<F>::new();
        let empty_poly = DensePolynomial::<F>::from_coefficients_vec(empty_vec);

        let emptyComm = PC::get_commitment::<G>(-1);
        println!("setup done = {:?}", start.elapsed());

        //TODO: move this to async section!!
        let z_2_poly = DensePolynomial::from_coefficients_vec(vec![F::from(1_u64); 1]);
        let (z_2_poly_commit, _) = PC::commit(commit_key, &[label_polynomial!(z_2_poly)], None, None::<&MSMContext<'_, '_, G>>).map_err(to_pc_error::<F, PC>)?;
        println!("commit done = {:?}", start.elapsed());
        
        //let pi_poly = self.cs.get_pi().into_dense_poly(n);
        let pi_poly = self.cs.get_pi().as_evals(n);
        println!("pi_poly done = {:?}", start.elapsed());
        
        // Since the caller is passing a pre-processed circuit
        // We assume that the Transcript has been seeded with the preprocessed
        // Commitments
        let mut transcript = self.preprocessed_transcript.clone();
       
        // Append Public Inputs to the transcript
        transcript.append(b"pi", self.cs.get_pi());
        println!("pi_poly done = {:?}", start.elapsed());

        // 1. Compute witness Polynomials
        //
        // Convert Variables to scalars padding them to the
        // correct domain size.

        let w_l_scalar = self.to_scalars(&self.cs.w_l);
        let w_r_scalar = self.to_scalars(&self.cs.w_r);
        let w_o_scalar = self.to_scalars(&self.cs.w_o);
        let w_4_scalar = self.to_scalars(&self.cs.w_4);

        unsafe { compute_w_polys (w_l_scalar.as_ptr() as *const c_void,
                                  w_r_scalar.as_ptr() as *const c_void,
                                  w_o_scalar.as_ptr() as *const c_void,
                                  w_4_scalar.as_ptr() as *const c_void,
                                  pi_poly.as_ptr() as *const c_void,
                                  
                                  omegas_vec.as_ptr() as *const c_void,
                                  omegas_inv_vec.as_ptr() as *const c_void,
                                  size_inv_vec.as_ptr() as *const c_void,
                                  domain.size() as i32, 
                                  self.cs.w_l.len() as i32);
               }

        let w_l = PC::get_commitment::<G>(0);
        let w_r = PC::get_commitment::<G>(1);
        let w_o = PC::get_commitment::<G>(2);
        let w_4 = PC::get_commitment::<G>(3);
        
        // Add witness polynomial commitments to transcript.
        transcript.append(b"w_l", &w_l);
        transcript.append(b"w_r", &w_r);
        transcript.append(b"w_o", &w_o);
        transcript.append(b"w_4", &w_4);
        
        // 2. Derive lookup polynomials
        //println!("first done = {:?}", start.elapsed());
        // Generate table compression factor
        let zeta : F = transcript.challenge_scalar(b"zeta");
        transcript.append(b"zeta", &zeta);

        // Compress lookup table into vector of single elements
        /*let compressed_t_multiset = MultiSet::compress(
            &[
                prover_key.lookup.table_1.clone(),
                prover_key.lookup.table_2.clone(),
                prover_key.lookup.table_3.clone(),
                prover_key.lookup.table_4.clone(),
            ],
            zeta,
        );*/

        let table_poly = empty_poly.clone(); //gpu_ifft(kern, gpu_container, &compressed_t_multiset.0, "table_poly", n, omega_inv);
        
        /*let q_lookup_pad = vec![F::zero(); n - self.cs.q_lookup.len()];
        let padded_q_lookup = &[self.cs.q_lookup.as_slice(), q_lookup_pad.as_slice()].concat();

        let mut f_scalars: Vec<MultiSet<F>> = vec![MultiSet::with_capacity(w_l_scalar.len()); 4];

        for (q_lookup, w_l, w_r, w_o, w_4) in izip!(
            padded_q_lookup,
            w_l_scalar,
            w_r_scalar,
            w_o_scalar,
            w_4_scalar,
        ) {
            if q_lookup.is_zero() {
                f_scalars[0].push(compressed_t_multiset.0[0]);
                f_scalars.iter_mut().skip(1).for_each(|f| f.push(F::zero()));
            } else {
                f_scalars[0].push(*w_l);
                f_scalars[1].push(*w_r);
                f_scalars[2].push(*w_o);
                f_scalars[3].push(*w_4);
            }
        }*/

        // Compress all wires into a single vector
        //let compressed_f_multiset = MultiSet::compress(&f_scalars, zeta);

        // Compute query poly
        let f_poly = empty_poly.clone();//gpu_ifft(kern, gpu_container, &compressed_f_multiset.0, "f_poly", n, omega_inv);
        
        // Add blinders to query polynomials
        //let f_poly = Self::add_blinder(&f_poly, n, 1);

        // Commit to query polynomial
        //let (f_poly_commit, _) = PC::commit(commit_key, &[label_polynomial!(f_poly)], None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;

        // Add f_poly commitment to transcript
        transcript.append(b"f", &emptyComm.clone());

        // Compute s, as the sorted and concatenated version of f and t
        //let (h_1, h_2) = compressed_t_multiset.combine_split(&compressed_f_multiset).unwrap();

        // Compute h polys
        let h_1_poly = empty_poly.clone();//gpu_ifft(kern, gpu_container, &h_1.0, "h_1_poly", n, omega_inv);
        let h_2_poly = empty_poly.clone();//gpu_ifft(kern, gpu_container, &h_2.0, "h_2_poly", n, omega_inv);
        
        // Add blinders to h polynomials
        // let h_1_poly = Self::add_blinder(&h_1_poly, n, 1);
        // let h_2_poly = Self::add_blinder(&h_2_poly, n, 1);

        // Commit to h polys
        //let (h_1_poly_commit, _) = PC::commit(commit_key, &[label_polynomial!(h_1_poly)], None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;
        //let (h_2_poly_commit, _) = PC::commit(commit_key, &[label_polynomial!(h_2_poly)], None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;

        // Add h polynomials to transcript
        transcript.append(b"h1", &emptyComm.clone());
        transcript.append(b"h2", &emptyComm.clone());
        //println!("second done = {:?}", start.elapsed());

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

        /*let z_poly = self.cs.perm.compute_permutation_poly(
            &domain,
            (w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar),
            beta,
            gamma,
            (
                &prover_key.permutation.left_sigma.0,
                &prover_key.permutation.right_sigma.0,
                &prover_key.permutation.out_sigma.0,
                &prover_key.permutation.fourth_sigma.0,
            ),
            kern,
            gpu_container,
        );*/

        ///////////START        
        let mut beta_vec = Vec::<F>::new(); beta_vec.resize(1, F::zero()); beta_vec[0] = beta;
        let mut gamma_vec = Vec::<F>::new(); gamma_vec.resize(1, F::zero()); gamma_vec[0] = gamma;

        let mut K1_vec = Vec::<F>::new(); K1_vec.resize(1, F::zero()); K1_vec[0] = K1 * beta;
        let mut K2_vec = Vec::<F>::new(); K2_vec.resize(1, F::zero()); K2_vec[0] = K2 * beta;
        let mut K3_vec = Vec::<F>::new(); K3_vec.resize(1, F::zero()); K3_vec[0] = K3 * beta;
        
        unsafe { compute_z_poly (beta_vec.as_ptr() as *const c_void,
                                 gamma_vec.as_ptr() as *const c_void,
                                 K1_vec.as_ptr() as *const c_void,
                                 K2_vec.as_ptr() as *const c_void,
                                 K3_vec.as_ptr() as *const c_void,
                                 size_inv_vec.as_ptr() as *const c_void,
                                 g_vec.as_ptr() as *const c_void,
                                 omegas_vec.as_ptr() as *const c_void,
                                 domain.size() as i32);
        }
        /////////// END
        
        //println!("z poly compute = {:?}", start.elapsed());

        // Commit to permutation polynomial.
        //let (z_poly_commit, _) = PC::commit(commit_key, &[label_polynomial!(z_poly)], None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;

        let z_p = PC::get_commitment::<G>(4);
        // Add permutation polynomial commitment to transcript.
        transcript.append(b"z", &z_p);

        // Compute mega permutation polynomial.
        // Compute lookup permutation poly
        //let z_2_poly = DensePolynomial::from_coefficients_vec(vec![F::from(1_u64); 1]);
        /*DensePolynomial::from_coefficients_slice(
            &self.cs.perm.compute_lookup_permutation_poly(
                &domain,
                &compressed_f_multiset.0,
                &compressed_t_multiset.0,
                &h_1.0,
                &h_2.0,
                delta,
                epsilon,
            ),
        );*/

        // TODO: Find strategy for blinding lookups
        // Add blinder for lookup permutation poly
        // z_2_poly = Self::add_blinder(&z_2_poly, n, 2);

        // Commit to lookup permutation polynomial.
        //let (z_2_poly_commit, _) = PC::commit(commit_key, &[label_polynomial!(z_2_poly)], None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;

        // 3. Compute public inputs polynomial. 
        // XXX:moved up
        //let pi_poly = self.cs.get_pi().into_dense_poly(n);
        //println!("third done = {:?}", start.elapsed());
        // 4. Compute quotient polynomial
        //
        // Compute quotient challenge; `alpha`, and gate-specific separation
        // challenges.
        let alpha = transcript.challenge_scalar(b"alpha");
        transcript.append(b"alpha", &alpha);

        let range_sep_challenge : F = transcript.challenge_scalar(b"range separation challenge");
        transcript.append(b"range seperation challenge", &range_sep_challenge);

        let logic_sep_challenge : F = transcript.challenge_scalar(b"logic separation challenge");
        transcript.append(b"logic seperation challenge", &logic_sep_challenge);

        let fixed_base_sep_challenge : F = transcript.challenge_scalar(b"fixed base separation challenge");
        transcript.append(b"fixed base separation challenge", &fixed_base_sep_challenge);

        let var_base_sep_challenge : F = transcript.challenge_scalar(b"variable base separation challenge");
        transcript.append(b"variable base separation challenge", &var_base_sep_challenge);

        let lookup_sep_challenge : F = transcript.challenge_scalar(b"lookup separation challenge");
        transcript.append(b"lookup separation challenge", &lookup_sep_challenge);

        /*let t_poly = quotient_poly::compute::<F, P>(
            &domain, prover_key, &z_poly, &z_2_poly, &w_l_poly, &w_r_poly, &w_o_poly,
            &w_4_poly, &pi_poly, &f_poly, &table_poly, &h_1_poly, &h_2_poly, &alpha, &beta,
            &gamma, &delta, &epsilon, &zeta, &range_sep_challenge, &logic_sep_challenge, &fixed_base_sep_challenge, 
            &var_base_sep_challenge, &lookup_sep_challenge, &kern, gpu_container,
        )?;*/

/////////////////////// START
        let mut alpha_vec = Vec::<F>::new(); alpha_vec.resize(1, F::zero()); alpha_vec[0] = alpha;
        let mut alpha_sq_vec = Vec::<F>::new(); alpha_sq_vec.resize(1, F::zero()); alpha_sq_vec[0] = alpha.square();

        unsafe {
                compute_quotient (domain.size() as i32, domain_8n.size() as i32,
                                   g_vec.as_ptr() as *const c_void,
                                   alpha_vec.as_ptr() as *const c_void,
                                   alpha_sq_vec.as_ptr() as *const c_void,
                                   beta_vec.as_ptr() as *const c_void,
                                   gamma_vec.as_ptr() as *const c_void,
                                   size_inv_vec.as_ptr() as *const c_void,
                                   K1_vec.as_ptr() as *const c_void,
                                   K2_vec.as_ptr() as *const c_void,
                                   K3_vec.as_ptr() as *const c_void,

                                   omegas_vec.as_ptr() as *const c_void,
                                   omegas_inv_vec.as_ptr() as *const c_void);
                }
/////////////////////// END

        // Commit to splitted quotient polynomial
        /*let (t_commits, _) = PC::commit(
            commit_key,
            &[
                label_polynomial!(t_polys[0]),
                label_polynomial!(t_polys[1]),
                label_polynomial!(t_polys[2]),
                label_polynomial!(t_polys[3]),
                label_polynomial!(t_polys[4]),
                label_polynomial!(t_polys[5]),
                label_polynomial!(t_polys[6]),
                label_polynomial!(t_polys[7]),
            ],
            None,
            Some(&msm_context),
        )
        .map_err(to_pc_error::<F, PC>)?;*/

        let q_0 = PC::get_commitment::<G>(5);
        let q_1 = PC::get_commitment::<G>(6);
        let q_2 = PC::get_commitment::<G>(7);
        let q_3 = PC::get_commitment::<G>(8);
        let q_4 = PC::get_commitment::<G>(9);
        let q_5 = PC::get_commitment::<G>(10);
        
        // Add quotient polynomial commitments to transcript
        transcript.append(b"t_1", &q_0);
        transcript.append(b"t_2", &q_1);
        transcript.append(b"t_3", &q_2);
        transcript.append(b"t_4", &q_3);
        transcript.append(b"t_5", &q_4);
        transcript.append(b"t_6", &q_5);
        transcript.append(b"t_7", &emptyComm.clone());
        transcript.append(b"t_8", &emptyComm.clone());
        // 4. Compute linearisation polynomial
        //
        // Compute evaluation challenge; `z`.
        let z_challenge : F = transcript.challenge_scalar(b"z");
        transcript.append(b"z", &z_challenge);

        /*let (lin_poly1, evaluations1) = linearisation_poly::compute::<F, P>(
            &domain, prover_key, &alpha, &beta, &gamma, &delta, &epsilon, &zeta,
            &range_sep_challenge, &logic_sep_challenge, &fixed_base_sep_challenge, &var_base_sep_challenge, &lookup_sep_challenge,
            &z_challenge, &w_l_poly, &w_r_poly, &w_o_poly, &w_4_poly, &t_polys[0], &t_polys[1], &t_polys[2], &t_polys[3], &t_polys[4], &t_polys[5], &t_polys[6], &t_polys[7],
            &z_poly, &z_2_poly, &f_poly, &h_1_poly, &h_2_poly, &table_poly,
        )?;*/
        
        ////////// START
        //TODO: may be need to move to async section or to GPU
        let vanishing_poly_eval = domain.evaluate_vanishing_polynomial(z_challenge);
        let z_challenge_to_n = vanishing_poly_eval + F::one();
    
        let mut values = Vec::<F>::new(); 
        values.resize(20, F::zero()); 
        values[0] = z_challenge; 
        values[1] = z_challenge * omega;
        values[2] = vanishing_poly_eval;
        values[3] = z_challenge_to_n;
        values[4] = alpha;
        values[5] = beta;
        values[6] = gamma;
        //values[7] = domain.evaluate_all_lagrange_coefficients(z_challenge)[0] * alpha.square(); //TODO: move to GPU!!

        let size : u64 = domain.size() as u64;
        let tau = z_challenge;

        let z_h_at_tau = tau.pow(&[size]) - F::one();
        let domain_offset = F::one();
        let mut u = F::zero();
        
        if z_h_at_tau.is_zero() {
            let mut omega_i = domain_offset;
            if omega_i == tau {
                u = F::one();
            }
        } else {
            
            // v_0_inv = m * h^(m-1)
            let v_0_inv = F::from(size as u64) * domain_offset.pow(&[(size - 1)]);
            let mut l_i = z_h_at_tau.inverse().unwrap() * v_0_inv;
            let mut negative_cur_elem = -domain_offset;            
            let r_i = tau + negative_cur_elem;
            u = l_i * r_i;
            u = u.inverse().unwrap();
        }
        values[7] = u* alpha.square();

        unsafe {    
            compute_linearisation_poly(values.as_ptr() as *mut c_void,
                                       domain.size() as i32);            
        }
        let evaluations = linearisation_poly::get_from_values::<F, P>(&values)?;
        ///////////END
        
        // Add evaluations to transcript.
        // First wire evals
        transcript.append(b"a_eval", &evaluations.wire_evals.a_eval);
        transcript.append(b"b_eval", &evaluations.wire_evals.b_eval);
        transcript.append(b"c_eval", &evaluations.wire_evals.c_eval);
        transcript.append(b"d_eval", &evaluations.wire_evals.d_eval);

        // Second permutation evals
        transcript.append(b"left_sig_eval", &evaluations.perm_evals.left_sigma_eval);
        transcript.append(b"right_sig_eval",&evaluations.perm_evals.right_sigma_eval);
        transcript.append(b"out_sig_eval", &evaluations.perm_evals.out_sigma_eval);
        transcript.append(b"perm_eval", &evaluations.perm_evals.permutation_eval);

        // Third lookup evals
        transcript.append(b"f_eval", &evaluations.lookup_evals.f_eval);
        transcript.append(b"q_lookup_eval", &evaluations.lookup_evals.q_lookup_eval);
        transcript.append(b"lookup_perm_eval",&evaluations.lookup_evals.z2_next_eval);
        transcript.append(b"h_1_eval", &evaluations.lookup_evals.h1_eval);
        transcript.append(b"h_1_next_eval", &evaluations.lookup_evals.h1_next_eval);
        transcript.append(b"h_2_eval", &evaluations.lookup_evals.h2_eval);

        // Third, all evals needed for custom gates
        evaluations.custom_evals.vals.iter().for_each(|(label, eval)| {
                let static_label = Box::leak(label.to_owned().into_boxed_str());
                transcript.append(static_label.as_bytes(), eval);
            });

        //println!("fourth done = {:?}", start.elapsed());
        // 5. Compute Openings using KZG10
        //
        // We merge the quotient polynomial using the `z_challenge` so the SRS
        // is linear in the circuit size `n`

        // Compute aggregate witness to polynomials evaluated at the evaluation
        // challenge `z`
        let aw_challenge: F = transcript.challenge_scalar(b"aggregate_witness");
        let saw_challenge: F = transcript.challenge_scalar(b"aggregate_witness");
        
        values[0] = z_challenge;
        values[1] = aw_challenge;
        values[2] = z_challenge * domain.element(1);
        values[3] = saw_challenge;

        /*let mut w_l_poly = vec![F::zero(); domain.size()];
        let mut w_r_poly = vec![F::zero(); domain.size()];
        let mut w_o_poly = vec![F::zero(); domain.size()];
        let mut w_4_poly = vec![F::zero(); domain.size()];
        
        let mut z_poly = vec![F::zero(); domain.size()];
        let mut lin_poly = empty_poly.clone();
        lin_poly.coeffs = vec![F::zero(); domain.size()];*/

        unsafe { 
            compute_open_commits(values.as_ptr() as *const c_void, domain.size() as i32);
            
            /*copy_to_host_tmp(w_l_poly.as_ptr() as *mut c_void,
                             w_r_poly.as_ptr() as *mut c_void,
                             w_o_poly.as_ptr() as *mut c_void,
                             w_4_poly.as_ptr() as *mut c_void,
                             z_poly.as_ptr() as *mut c_void,
                             lin_poly.coeffs.as_ptr() as *mut c_void,
                             domain.size() as i32);*/
        }
        
        let aw_proof = PC::get_proof::<G>(11);
        let sw_proof = PC::get_proof::<G>(12);
        
        /*
        let pad = vec![F::zero(); n - self.cs.w_l.len()];
        let w_l_scalar = &[&self.to_scalars(&self.cs.w_l)[..], &pad].concat();
        let w_r_scalar = &[&self.to_scalars(&self.cs.w_r)[..], &pad].concat();
        let w_o_scalar = &[&self.to_scalars(&self.cs.w_o)[..], &pad].concat();
        let w_4_scalar = &[&self.to_scalars(&self.cs.w_4)[..], &pad].concat();

        let msm_context = msm_context.unwrap();
        let w_l_poly = DensePolynomial::<F>::from_coefficients_vec(w_l_poly);
        let w_r_poly = DensePolynomial::<F>::from_coefficients_vec(w_r_poly);
        let w_o_poly = DensePolynomial::<F>::from_coefficients_vec(w_o_poly);
        let w_4_poly = DensePolynomial::<F>::from_coefficients_vec(w_4_poly);
        
        let w_polys = [
            label_polynomial!(w_l_poly),
            label_polynomial!(w_r_poly),
            label_polynomial!(w_o_poly),
            label_polynomial!(w_4_poly),
        ];

        // Commit to witness polynomials.
        let (w_commits, w_rands) = PC::commit(commit_key, w_polys.iter(), None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;
        let z_poly = DensePolynomial::<F>::from_coefficients_vec(z_poly);

        // XXX: The quotient polynomials is used here and then in the
        // opening poly. It is being left in for now but it may not
        // be necessary. Warrants further investigation.
        // Ditto with the out_sigma poly.
        let aw_polys = [
            label_polynomial!(lin_poly),
            label_polynomial!(prover_key.permutation.left_sigma.0.clone()),
            label_polynomial!(prover_key.permutation.right_sigma.0.clone()),
            label_polynomial!(prover_key.permutation.out_sigma.0.clone()),
            label_polynomial!(f_poly),
            label_polynomial!(h_2_poly),
            label_polynomial!(table_poly),
        ];

        let (aw_commits, aw_rands) = PC::commit(commit_key, &aw_polys, None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;
        let aw_opening = PC::open(commit_key, 
                                  aw_polys.iter().chain(w_polys.iter()), 
                                  None, 
                                  &z_challenge, 
                                  aw_challenge, 
                                  aw_rands.iter().chain(w_rands.iter()), 
                                  None, 
                                  Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;
        
        

        let saw_polys = [
            label_polynomial!(z_poly),
            label_polynomial!(w_l_poly),
            label_polynomial!(w_r_poly),
            label_polynomial!(w_4_poly),
            label_polynomial!(h_1_poly),
            label_polynomial!(z_2_poly),
            //label_polynomial!(table_poly),
        ];

        let (saw_commits, saw_rands) = PC::commit(commit_key, &saw_polys, None, Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;
        let saw_opening = PC::open(commit_key, 
                                   &saw_polys, 
                                   None, 
                                   &(z_challenge * domain.element(1)), 
                                   saw_challenge, 
                                   &saw_rands, 
                                   None, 
                                   Some(&msm_context)).map_err(to_pc_error::<F, PC>)?;*/

        println!("total prove = {:?}", start.elapsed());
        Ok(Proof {
            a_comm: w_l.clone(),
            b_comm: w_r.clone(),
            c_comm: w_o.clone(),
            d_comm: w_4.clone(),
            z_comm: z_p.clone(),
            f_comm: emptyComm.clone(),
            h_1_comm: emptyComm.clone(),
            h_2_comm: emptyComm.clone(),
            z_2_comm: z_2_poly_commit[0].commitment().clone(),
            t_1_comm: q_0.clone(),
            t_2_comm: q_1.clone(),
            t_3_comm: q_2.clone(),
            t_4_comm: q_3.clone(),
            t_5_comm: q_4.clone(),
            t_6_comm: q_5.clone(),
            t_7_comm: emptyComm.clone(),
            t_8_comm: emptyComm.clone(),
            aw_opening: aw_proof,
            saw_opening : sw_proof,
            evaluations,
        })
    }

    /// Proves a circuit is satisfied, then clears the witness variables
    /// If the circuit is not pre-processed, then the preprocessed circuit will
    /// also be computed.
    pub fn prove<'a, 'b, G>(
        &mut self,
        commit_key: &PC::CommitterKey,
        msm_context: Option<&MSMContext<'a, 'b, G>>,
    ) -> Result<Proof<F, PC>, Error>
    where
        G: AffineCurve,
    {
        if self.prover_key.is_none() {
            //println!("preprocess");
            // Preprocess circuit and store preprocessed circuit and transcript
            // in the Prover.
            self.prover_key = Some(self.cs.preprocess_prover(
                commit_key,
                &mut self.preprocessed_transcript,
                PhantomData::<PC>,
            )?);
        }
        /*else {
            println!("skip preprocessing");
        }*/

        let prover_key = self.prover_key.as_ref().unwrap();
        let proof = self.prove_with_preprocessed(
            commit_key,
            prover_key,
            msm_context,
            PhantomData::<PC>,
        )?;

        // Clear witness and reset composer variables
        self.clear_witness();

        Ok(proof)
    }
    
    pub fn copy_data<'a, 'b, 'c>(&mut self, coeffs_count: i32)
    {
        let prover_key = self.prover_key.as_ref().unwrap();
                      
        unsafe {
           copyArrays(coeffs_count, 
                      prover_key.v_h_coset_8n().evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_m.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_l.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_r.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_o.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_4.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_hl.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_hr.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_h4.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_c.1.evals.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_arith.1.evals.as_ptr() as *const c_void,

                      prover_key.permutation.linear_evaluations.evals.as_ptr() as *const c_void,
                      prover_key.permutation.left_sigma.1.evals.as_ptr() as *const c_void,
                      prover_key.permutation.right_sigma.1.evals.as_ptr() as *const c_void,
                      prover_key.permutation.out_sigma.1.evals.as_ptr() as *const c_void,
                      prover_key.permutation.fourth_sigma.1.evals.as_ptr() as *const c_void,
                      
                      prover_key.arithmetic.q_arith.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_c.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_l.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_r.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_hl.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_hr.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_h4.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_m.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_4.0.coeffs.as_ptr() as *const c_void,
                      prover_key.arithmetic.q_o.0.coeffs.as_ptr() as *const c_void,
                      
                      prover_key.permutation.left_sigma.0.coeffs.as_ptr() as *const c_void,
                      prover_key.permutation.right_sigma.0.coeffs.as_ptr() as *const c_void,
                      prover_key.permutation.out_sigma.0.coeffs.as_ptr() as *const c_void,
                      prover_key.permutation.fourth_sigma.0.coeffs.as_ptr() as *const c_void);
        }
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
        Prover::new(b"plonk")
    }
}
