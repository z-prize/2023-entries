// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Tools & traits for PLONK circuits

use crate::label_eval;
use crate::proof_system::linearisation_poly::{
    CustomEvaluations, LookupEvaluations, PermutationEvaluations,
    ProofEvaluations, WireEvaluations,
};
use crate::transcript::TranscriptProtocol;
use crate::{
    commitment::HomomorphicCommitment,
    error::{to_pc_error, Error},
    prelude::StandardComposer,
    proof_system::{
        pi::PublicInputs, Proof, Prover, ProverKey, Verifier, VerifierKey,
    },
};
use ark_bls12_381::G1Affine;
use ark_bls12_381::{Bls12_381, Fq};
use ark_ec::models::TEModelParameters;
use ark_ff::{BigInteger, BigInteger256, BigInteger384, PrimeField};
use ark_poly_commit::kzg10::{self, Commitment};
use ark_serialize::*;
use merkle_accel::{
    self, Round1Res, Round3Res, Round4ARes, Round4BRes, Round5Res,
};
use merlin::Transcript;
use num_traits::Zero;
use rand::prelude::StdRng;
use rand_core::SeedableRng;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Instant;

/// Collection of structs/objects that the Verifier will use in order to
/// de/serialize data needed for Circuit proof verification.
/// This structure can be seen as a link between the [`Circuit`] public input
/// positions and the [`VerifierKey`] that the Verifier needs to use.
#[derive(CanonicalDeserialize, CanonicalSerialize, derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = "VerifierKey<F,PC>: core::fmt::Debug"),
    Eq(bound = "VerifierKey<F,PC>: Eq"),
    PartialEq(bound = "VerifierKey<F,PC>: PartialEq")
)]
pub struct VerifierData<F, PC>
where
    F: PrimeField,
    PC: HomomorphicCommitment<F>,
{
    /// Verifier Key
    pub key: VerifierKey<F, PC>,
    /// Public Input
    pub pi: PublicInputs<F>,
}

impl<F, PC> VerifierData<F, PC>
where
    F: PrimeField,
    PC: HomomorphicCommitment<F>,
{
    /// Creates a new `VerifierData` from a [`VerifierKey`] and the public
    /// input of the circuit that it represents.
    pub fn new(key: VerifierKey<F, PC>, pi: PublicInputs<F>) -> Self {
        Self { key, pi }
    }

    /// Returns a reference to the contained [`VerifierKey`].
    pub fn key(&self) -> &VerifierKey<F, PC> {
        &self.key
    }

    /// Returns a reference to the contained Public Input .
    pub fn pi(&self) -> &PublicInputs<F> {
        &self.pi
    }
}

/// prover key as a static to be set at initialization
static KEY: OnceLock<([Vec<BigInteger256>; 28], Vec<BigInteger384>)> =
    OnceLock::new();
/// pointer to the device created at initialization
static CTX: OnceLock<usize> = OnceLock::new();

// Wrapper for println which can be disabled with a feature
#[cfg(feature = "show-print")]
macro_rules! println {
    () => {
        std::println!()
    };
    ($($arg:tt)*) => {{
        std::println!($($arg)*)
    }};
}
#[cfg(not(feature = "show-print"))]
macro_rules! println {
    () => {
        let _ = std::format!();
    };
    ($($arg:tt)*) => {
        // to prevent warnings due to unused values
        let _ = std::format!($($arg)*);
    };
}
/// Trait that should be implemented for any circuit function to provide to it
/// the capabilities of automatically being able to generate, and verify proofs
/// as well as compile the circuit.
///
/// # Example
///
/// ```rust,no_run
/// use ark_bls12_381::{Bls12_381, Fr as BlsScalar};
/// use ark_ec::PairingEngine;
/// use ark_ec::models::twisted_edwards_extended::GroupAffine;
/// use ark_ec::{TEModelParameters, AffineCurve, ProjectiveCurve};
/// use ark_ed_on_bls12_381::{
///     EdwardsAffine as JubJubAffine, EdwardsParameters as JubJubParameters,
///     EdwardsProjective as JubJubProjective, Fr as JubJubScalar,
/// };
/// use ark_ff::{FftField, PrimeField, BigInteger, ToConstraintField};
/// use plonk_core::circuit::{Circuit, verify_proof};
/// use plonk_core::constraint_system::StandardComposer;
/// use plonk_core::error::{to_pc_error,Error};
/// use ark_poly::polynomial::univariate::DensePolynomial;
/// use ark_poly_commit::{PolynomialCommitment, sonic_pc::SonicKZG10};
/// use plonk_core::prelude::*;
/// use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
/// use num_traits::{Zero, One};
/// use rand_core::OsRng;
///
/// fn main() -> Result<(), Error> {
/// // Implements a circuit that checks:
/// // 1) a + b = c where C is a PI
/// // 2) a <= 2^6
/// // 3) b <= 2^5
/// // 4) a * b = d where D is a PI
/// // 5) JubJub::GENERATOR * e(JubJubScalar) = f where F is a PI
/// #[derive(derivative::Derivative)]
///    #[derivative(Debug(bound = ""), Default(bound = ""))]
/// pub struct TestCircuit<F, P>
/// where
///     F: PrimeField,
///     P: TEModelParameters<BaseField = F>,
/// {
///        a: F,
///        b: F,
///        c: F,
///        d: F,
///        e: P::ScalarField,
///        f: GroupAffine<P>,
///    }
///
/// impl<F, P> Circuit<F, P> for TestCircuit<F, P>
/// where
///     F: PrimeField,
///     P: TEModelParameters<BaseField = F>,
///    {
///        const CIRCUIT_ID: [u8; 32] = [0xff; 32];
///
///        fn gadget(
///            &mut self,
///            composer: &mut StandardComposer<F, P>,
///        ) -> Result<(), Error> {
///            let a = composer.add_input(self.a);
///            let b = composer.add_input(self.b);
///            let zero = composer.zero_var();
///
///            // Make first constraint a + b = c (as public input)
///            composer.arithmetic_gate(|gate| {
///                gate.witness(a, b, Some(zero))
///                    .add(F::one(), F::one())
///                    .pi(-self.c)
///            });
///
///            // Check that a and b are in range
///            composer.range_gate(a, 1 << 6);
///            composer.range_gate(b, 1 << 5);
///            // Make second constraint a * b = d
///            composer.arithmetic_gate(|gate| {
///                gate.witness(a, b, Some(zero)).mul(F::one()).pi(-self.d)
///            });
///            let e = composer
///                .add_input(from_embedded_curve_scalar::<F, P>(self.e));
///            let (x, y) = P::AFFINE_GENERATOR_COEFFS;
///            let generator = GroupAffine::new(x, y);
///            let scalar_mul_result =
///                composer.fixed_base_scalar_mul(e, generator);
///            // Apply the constrain
///            composer.assert_equal_public_point(scalar_mul_result, self.f);
///            Ok(())
///        }
///
///        fn padded_circuit_size(&self) -> usize {
///            1 << 11
///        }
///    }
///
/// // Generate CRS
/// type PC = SonicKZG10::<Bls12_381,DensePolynomial<BlsScalar>>;
/// let pp = PC::setup(
///     1 << 10, None, &mut OsRng
///  )?;
///
/// let mut circuit = TestCircuit::<BlsScalar, JubJubParameters>::default();
/// // Compile the circuit
/// let (pk_p, (vk, _pi_pos)) = circuit.compile::<PC>(&pp)?;
///
/// let (x, y) = JubJubParameters::AFFINE_GENERATOR_COEFFS;
/// let generator: GroupAffine<JubJubParameters> = GroupAffine::new(x, y);
/// let point_f_pi: GroupAffine<JubJubParameters> = AffineCurve::mul(
///     &generator,
///     JubJubScalar::from(2u64).into_repr(),
/// )
/// .into_affine();
/// // Prover POV
/// let (proof, pi) = {
///     let mut circuit: TestCircuit<BlsScalar, JubJubParameters> = TestCircuit {
///         a: BlsScalar::from(20u64),
///         b: BlsScalar::from(5u64),
///         c: BlsScalar::from(25u64),
///         d: BlsScalar::from(100u64),
///         e: JubJubScalar::from(2u64),
///         f: point_f_pi,
///     };
///     circuit.gen_proof::<PC>(&pp, pk_p, b"Test")
/// }?;
///
/// let verifier_data = VerifierData::new(vk, pi);
/// // Test serialisation for verifier_data
/// let mut verifier_data_bytes = Vec::new();
/// verifier_data.serialize(&mut verifier_data_bytes).unwrap();
///
/// let deserialized_verifier_data: VerifierData<BlsScalar, PC> =
///     VerifierData::deserialize(verifier_data_bytes.as_slice()).unwrap();
///
/// assert!(deserialized_verifier_data == verifier_data);
///
/// // Verifier POV
/// verify_proof::<BlsScalar, JubJubParameters, PC>(
///     &pp,
///     verifier_data.key,
///     &proof,
///     &verifier_data.pi,
///     b"Test",
/// )
/// }
/// ```
pub trait Circuit<F, P>
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
{
    /// Circuit identifier associated constant.
    const CIRCUIT_ID: [u8; 32];

    /// Gadget implementation used to fill the composer.
    fn gadget(
        &mut self,
        composer: &mut StandardComposer<F, P>,
    ) -> Result<(), Error>;

    /// Compiles the circuit by using a function that returns a `Result`
    /// with the [`ProverKey`], [`VerifierKey`] and a vector of the intended
    /// positions for public inputs and the circuit size.
    #[allow(clippy::type_complexity)]
    fn compile<PC>(
        &mut self,
        u_params: &PC::UniversalParams,
    ) -> Result<(ProverKey<F>, (VerifierKey<F, PC>, Vec<usize>)), Error>
    where
        F: PrimeField,
        PC: HomomorphicCommitment<F>,
    {
        // Setup PublicParams
        let circuit_size = self.padded_circuit_size();
        let (ck, _) = PC::trim(u_params, circuit_size, 0, None)
            .map_err(to_pc_error::<F, PC>)?;

        //Generate & save `ProverKey` with some random values.
        let mut prover = Prover::<F, P, PC>::new(b"CircuitCompilation");
        self.gadget(prover.mut_cs())?;
        prover.preprocess(&ck)?;

        // Generate & save `VerifierKey` with some random values.
        let mut verifier = Verifier::new(b"CircuitCompilation");
        self.gadget(verifier.mut_cs())?;
        verifier.preprocess(&ck)?;
        Ok((
            prover
                .prover_key
                .expect("Unexpected error. Missing ProverKey in compilation"),
            (
                verifier.verifier_key.expect(
                    "Unexpected error. Missing VerifierKey in compilation",
                ),
                verifier.cs.intended_pi_pos,
            ),
        ))
    }

    /// Generates a proof using the provided [`ProverKey`] and
    /// [`ark_poly_commit::PCUniversalParams`]. Returns a
    /// [`crate::proof_system::Proof`] and the [`PublicInputs`].
    fn gen_proof<PC>(
        &mut self,
        u_params: &PC::UniversalParams,
        prover_key: ProverKey<F>,
        transcript_init: &'static [u8],
    ) -> Result<(Proof<F, PC>, PublicInputs<F>), Error>
    where
        F: PrimeField,
        P: TEModelParameters<BaseField = F>,
        PC: HomomorphicCommitment<F>,
    {
        let circuit_size = self.padded_circuit_size();
        let (ck, _) = PC::trim(u_params, circuit_size, 0, None)
            .map_err(to_pc_error::<F, PC>)?;
        // New Prover instance
        let mut prover = Prover::new(transcript_init);
        // Fill witnesses for Prover
        self.gadget(prover.mut_cs())?;
        // Add ProverKey to Prover
        prover.prover_key = Some(prover_key);
        let pi = prover.cs.get_pi().clone();

        Ok((prover.prove(&ck)?, pi))
    }

    /// Returns the Circuit size padded to the next power of two.
    fn padded_circuit_size(&self) -> usize;

    /// per circuit, witness universal initialization that precomputes and
    /// prepares the data which isn't tied to a particular witness. Also
    /// sends it to the device and initializes it.
    fn init(
        prover_key: &ProverKey<F>,
        commit_key: &Vec<ark_bls12_381::G1Affine>,
    ) where
        F: PrimeField<BigInt = BigInteger256>,
    {
        // here it's mostly just changing types
        let repr = |x: &Vec<F>| {
            x.iter().map(|x| x.into_repr()).collect::<Vec<F::BigInt>>()
        };
        let ProverKey {
            arithmetic,
            permutation,
            v_h_coset_8n,
            ..
        } = prover_key;

        let q_l_poly: Vec<F::BigInt> = repr(&arithmetic.q_l.0.coeffs);
        let q_r_poly: Vec<F::BigInt> = repr(&arithmetic.q_r.0.coeffs);
        let q_o_poly: Vec<F::BigInt> = repr(&arithmetic.q_o.0.coeffs);
        let q_4_poly: Vec<F::BigInt> = repr(&arithmetic.q_4.0.coeffs);
        let q_hl_poly: Vec<F::BigInt> = repr(&arithmetic.q_hl.0.coeffs);
        let q_hr_poly: Vec<F::BigInt> = repr(&arithmetic.q_hr.0.coeffs);
        let q_h4_poly: Vec<F::BigInt> = repr(&arithmetic.q_h4.0.coeffs);
        let q_c_poly: Vec<F::BigInt> = repr(&arithmetic.q_c.0.coeffs);
        let q_arith_poly = repr(&arithmetic.q_arith.0.coeffs);

        let q_l_evals: Vec<F::BigInt> = repr(&arithmetic.q_l.1.evals);
        let q_r_evals: Vec<F::BigInt> = repr(&arithmetic.q_r.1.evals);
        let q_o_evals: Vec<F::BigInt> = repr(&arithmetic.q_o.1.evals);
        let q_4_evals: Vec<F::BigInt> = repr(&arithmetic.q_4.1.evals);
        let q_hl_evals = repr(&arithmetic.q_hl.1.evals);
        let q_hr_evals = repr(&arithmetic.q_hr.1.evals);
        let q_h4_evals = repr(&arithmetic.q_h4.1.evals);
        let q_c_evals = repr(&arithmetic.q_c.1.evals);
        let q_arith_evals = repr(&arithmetic.q_arith.1.evals);

        let left_sigma_poly = repr(&permutation.left_sigma.0.coeffs);
        let right_sigma_poly = repr(&permutation.right_sigma.0.coeffs);
        let out_sigma_poly = repr(&permutation.out_sigma.0.coeffs);
        let fourth_sigma_poly = repr(&permutation.fourth_sigma.0.coeffs);

        let left_sigma_evals = repr(&permutation.left_sigma.1.evals);
        let right_sigma_evals = repr(&permutation.right_sigma.1.evals);
        let out_sigma_evals = repr(&permutation.out_sigma.1.evals);
        let fourth_sigma_evals = repr(&permutation.fourth_sigma.1.evals);

        let linear_evals = repr(&permutation.linear_evaluations.evals);

        // we store the already inverted evaluations to be used in the division
        // later, this is a general optimization which could be used in cpu.
        let v_h_evals_inv: Vec<F::BigInt> = v_h_coset_8n
            .evals
            .iter()
            .map(|x| x.inverse().unwrap().into_repr())
            .collect();
        let prover_key = [
            q_l_poly,
            q_r_poly,
            q_o_poly,
            q_4_poly,
            q_hl_poly,
            q_hr_poly,
            q_h4_poly,
            q_c_poly,
            q_arith_poly,
            q_l_evals,
            q_r_evals,
            q_o_evals,
            q_4_evals,
            q_hl_evals,
            q_hr_evals,
            q_h4_evals,
            q_c_evals,
            q_arith_evals,
            left_sigma_poly,
            right_sigma_poly,
            out_sigma_poly,
            fourth_sigma_poly,
            left_sigma_evals,
            right_sigma_evals,
            out_sigma_evals,
            fourth_sigma_evals,
            linear_evals,
            v_h_evals_inv,
        ];

        let commit_key_raw: Vec<BigInteger384> = commit_key
            .iter()
            .flat_map(|x| vec![x.x.into_repr(), x.y.into_repr()])
            .collect();

        let (prover_key, commit_key_raw) =
            KEY.get_or_init(|| (prover_key, commit_key_raw));

        let prover_key: [*const u64; 28] = prover_key
            .iter()
            .map(|x| x.as_slice().as_ptr() as *const u64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let [q_l_poly, q_r_poly, q_o_poly, q_4_poly, q_hl_poly, q_hr_poly, q_h4_poly, q_c_poly, q_arith_poly, q_l_evals, q_r_evals, q_o_evals, q_4_evals, q_hl_evals, q_hr_evals, q_h4_evals, q_c_evals, q_arith_evals, left_sigma_poly, right_sigma_poly, out_sigma_poly, fourth_sigma_poly, left_sigma_evals, right_sigma_evals, out_sigma_evals, fourth_sigma_evals, linear_evals, v_h_evals_inv] =
            prover_key;

        let prover_key_raw = merkle_accel::ProverKey {
            q_l_poly,
            q_r_poly,
            q_o_poly,
            q_4_poly,
            q_hl_poly,
            q_hr_poly,
            q_h4_poly,
            q_c_poly,
            q_arith_poly,
            q_l_evals,
            q_r_evals,
            q_o_evals,
            q_4_evals,
            q_hl_evals,
            q_hr_evals,
            q_h4_evals,
            q_c_evals,
            q_arith_evals,
            left_sigma_poly,
            right_sigma_poly,
            out_sigma_poly,
            fourth_sigma_poly,
            left_sigma_evals,
            right_sigma_evals,
            out_sigma_evals,
            fourth_sigma_evals,
            linear_evals,
            v_h_evals_inv,
        };

        // initializing the context with a reference to the keys
        let ctx = merkle_accel::init_ctx(
            commit_key_raw.as_slice().as_ptr() as *const u64,
            prover_key_raw,
        );
        CTX.get_or_init(|| ctx as usize);
    }
    /// provides nodes of the tree \[non_leaf, leaf\]
    fn tree_nodes(&self) -> [&Vec<F>; 2] {
        unimplemented!()
    }
    /// provides access to the context, assumes `init()` was called before
    fn ctx() -> *mut c_void {
        *CTX.get().unwrap() as *mut c_void
    }
    /// Generates proof with hardware acceleration
    fn gen_proof_accel<PC>(
        &mut self,
        transcript_init: &'static [u8],
        commit_key: &Vec<ark_bls12_381::G1Affine>,
        pi: PublicInputs<F>,
    ) -> Result<(Proof<F, PC>, PublicInputs<F>), Error>
    where
        F: PrimeField<BigInt = BigInteger256>,
        P: TEModelParameters<BaseField = F>,
        PC: HomomorphicCommitment<
            F,
            Commitment = Commitment<Bls12_381>,
            Proof = kzg10::Proof<Bls12_381>,
        >,
    {
        let mut transcript = Transcript::new(transcript_init);

        let [merkle_non_leaf_nodes, merkle_leaf_nodes] = self.tree_nodes();
        // TODO: Check if this is ok
        let seed = [42; 32];
        let mut rng = StdRng::from_seed(seed);
        let mut blinding_factors: Vec<F::BigInt> =
            vec![F::zero().into_repr(); 8];
        blinding_factors[0] = F::rand(&mut rng).into_repr();
        blinding_factors[1] = F::rand(&mut rng).into_repr();
        blinding_factors[3] = F::rand(&mut rng).into_repr();
        blinding_factors[2] = F::rand(&mut rng).into_repr();
        blinding_factors[4] = F::rand(&mut rng).into_repr();
        blinding_factors[5] = F::rand(&mut rng).into_repr();
        blinding_factors[7] = F::rand(&mut rng).into_repr();
        blinding_factors[6] = F::rand(&mut rng).into_repr();

        let merkle_non_leaf_raw: Vec<F::BigInt> = merkle_non_leaf_nodes
            .iter()
            .map(|x| x.into_repr())
            .collect();
        let merkle_leaf_raw: Vec<F::BigInt> =
            merkle_leaf_nodes.iter().map(|x| x.into_repr()).collect();

        println!("-- [INIT] -----------------------------------------------------------------------------------------");
        let mut start = Instant::now();
        let ctx = Self::ctx();
        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");

        println!("-- [ROUND_0] --------------------------------------------------------------------------------------");
        start = Instant::now();

        merkle_accel::round_0(
            ctx,
            blinding_factors.as_slice().as_ptr() as *const u64,
            merkle_non_leaf_raw.as_slice().as_ptr() as *const u64,
            merkle_leaf_raw.as_slice().as_ptr() as *const u64,
        );

        transcript.append(b"pi", &pi);

        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");

        println!("-- [ROUND_1] --------------------------------------------------------------------------------------");
        start = Instant::now();

        let round_1_res = merkle_accel::round_1(ctx);

        transcript.append(b"w_l", &BigInteger384(round_1_res.w_l_commit));
        transcript.append(b"w_r", &BigInteger384(round_1_res.w_r_commit));
        transcript.append(b"w_o", &BigInteger384(round_1_res.w_o_commit));
        transcript.append(b"w_4", &BigInteger384(round_1_res.w_4_commit));

        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");

        println!("-- [ROUND_2] --------------------------------------------------------------------------------------");
        start = Instant::now();

        let zeta: F = transcript.challenge_scalar(b"zeta");
        println!("zeta: {:?}", zeta.to_string());

        let f_commit = G1Affine::zero();
        let h_1_commit = G1Affine::zero();
        let h_2_commit = G1Affine::zero();

        transcript.append(b"zeta", &zeta);
        transcript.append(b"f", &f_commit);
        transcript.append(b"h1", &h_1_commit);
        transcript.append(b"h2", &h_2_commit);

        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");
        println!("-- [ROUND_3] --------------------------------------------------------------------------------------");
        start = Instant::now();

        let beta: F = transcript.challenge_scalar(b"beta");
        transcript.append(b"beta", &beta);
        let gamma: F = transcript.challenge_scalar(b"gamma");
        transcript.append(b"gamma", &gamma);
        let delta: F = transcript.challenge_scalar(b"delta");
        transcript.append(b"delta", &delta);
        let epsilon: F = transcript.challenge_scalar(b"epsilon");
        transcript.append(b"epsilon", &epsilon);

        println!("beta: {:?}", beta.to_string());
        println!("gamma: {:?}", gamma.to_string());
        println!("delta: {:?}", delta.to_string());
        println!("epsilon: {:?}", epsilon.to_string());

        let round_3_res = merkle_accel::round_3(
            ctx,
            beta.into_repr().as_ref().as_ptr(),
            gamma.into_repr().as_ref().as_ptr(),
        );

        transcript.append(b"z", &BigInteger384(round_3_res.z_commit));

        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");

        println!("-- [ROUND_4a] -------------------------------------------------------------------------------------");
        start = Instant::now();

        let alpha: F = transcript.challenge_scalar(b"alpha");
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
        let lookup_sep_challenge: F =
            transcript.challenge_scalar(b"lookup separation challenge");
        transcript
            .append(b"lookup separation challenge", &lookup_sep_challenge);

        println!("alpha: {:?}", alpha.to_string());

        let round_4a_res = merkle_accel::round_4a(
            ctx,
            alpha.into_repr().as_ref().as_ptr(),
            beta.into_repr().as_ref().as_ptr(),
            gamma.into_repr().as_ref().as_ptr(),
        );

        transcript.append(b"t_1", &BigInteger384(round_4a_res.t_0_commit));
        transcript.append(b"t_2", &BigInteger384(round_4a_res.t_1_commit));
        transcript.append(b"t_3", &BigInteger384(round_4a_res.t_2_commit));
        transcript.append(b"t_4", &BigInteger384(round_4a_res.t_3_commit));
        transcript.append(b"t_5", &BigInteger384(round_4a_res.t_4_commit));
        transcript.append(b"t_6", &BigInteger384(round_4a_res.t_5_commit));
        transcript.append(b"t_7", &BigInteger384(round_4a_res.t_6_commit));
        transcript.append(b"t_8", &BigInteger384(round_4a_res.t_7_commit));

        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");

        println!("-- [ROUND_4b] -------------------------------------------------------------------------------------");
        start = Instant::now();
        let z_challenge: F = transcript.challenge_scalar(b"z");
        transcript.append(b"z", &z_challenge);

        println!("z_challenge: {:?}", z_challenge.to_string());

        let round_4b_res = merkle_accel::round_4b(
            ctx,
            alpha.into_repr().as_ref().as_ptr(),
            beta.into_repr().as_ref().as_ptr(),
            gamma.into_repr().as_ref().as_ptr(),
            z_challenge.into_repr().as_ref().as_ptr(),
        );

        transcript.append(b"a_eval", &BigInteger256(round_4b_res.a_eval));
        transcript.append(b"b_eval", &BigInteger256(round_4b_res.b_eval));
        transcript.append(b"c_eval", &BigInteger256(round_4b_res.c_eval));
        transcript.append(b"d_eval", &BigInteger256(round_4b_res.d_eval));

        transcript.append(
            b"left_sig_eval",
            &BigInteger256(round_4b_res.left_sigma_eval),
        );
        transcript.append(
            b"right_sig_eval",
            &BigInteger256(round_4b_res.right_sigma_eval),
        );
        transcript.append(
            b"out_sig_eval",
            &BigInteger256(round_4b_res.out_sigma_eval),
        );
        transcript.append(
            b"perm_eval",
            &BigInteger256(round_4b_res.permutation_eval),
        );

        transcript.append(b"f_eval", &F::zero());
        transcript.append(b"q_lookup_eval", &F::zero());
        transcript.append(b"lookup_perm_eval", &F::one());
        transcript.append(b"h_1_eval", &F::zero());
        transcript.append(b"h_1_next_eval", &F::zero());
        transcript.append(b"h_2_eval", &F::zero());

        transcript
            .append(b"q_arith_eval", &BigInteger256(round_4b_res.q_arith_eval));
        transcript.append(b"q_c_eval", &BigInteger256(round_4b_res.q_c_eval));
        transcript.append(b"q_l_eval", &BigInteger256(round_4b_res.q_l_eval));
        transcript.append(b"q_r_eval", &BigInteger256(round_4b_res.q_r_eval));
        transcript.append(b"q_hl_eval", &BigInteger256(round_4b_res.q_hl_eval));
        transcript.append(b"q_hr_eval", &BigInteger256(round_4b_res.q_hr_eval));
        transcript.append(b"q_h4_eval", &BigInteger256(round_4b_res.q_h4_eval));
        transcript
            .append(b"a_next_eval", &BigInteger256(round_4b_res.a_next_eval));
        transcript
            .append(b"b_next_eval", &BigInteger256(round_4b_res.b_next_eval));
        transcript
            .append(b"d_next_eval", &BigInteger256(round_4b_res.d_next_eval));

        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");

        println!("-- [ROUND_5] --------------------------------------------------------------------------------------");
        start = Instant::now();

        let aw_challenge: F = transcript.challenge_scalar(b"aggregate_witness");
        let saw_challenge: F =
            transcript.challenge_scalar(b"aggregate_witness");

        println!("aw_challenge: {:?}", aw_challenge.to_string());
        println!("saw_challenge: {:?}", saw_challenge.to_string());

        let round_5_res = merkle_accel::round_5(
            ctx,
            z_challenge.into_repr().as_ref().as_ptr(),
            aw_challenge.into_repr().as_ref().as_ptr(),
            saw_challenge.into_repr().as_ref().as_ptr(),
        );

        println!("exec time: {:?}", start.elapsed());
        println!("---------------------------------------------------------------------------------------------------");

        println!("-- [PROOF] ----------------------------------------------------------------------------------------");
        let proof = assemble_proof::<F, PC>(
            round_1_res,
            round_3_res,
            round_4a_res,
            round_4b_res,
            round_5_res,
            commit_key[0],
        );
        Ok((proof, pi))
    }
}

/// Verifies a proof using the provided `CircuitInputs` & `VerifierKey`
/// instances.
pub fn verify_proof<F, P, PC>(
    u_params: &PC::UniversalParams,
    plonk_verifier_key: VerifierKey<F, PC>,
    proof: &Proof<F, PC>,
    public_inputs: &PublicInputs<F>,
    transcript_init: &'static [u8],
) -> Result<(), Error>
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
{
    let mut verifier: Verifier<F, P, PC> = Verifier::new(transcript_init);
    let padded_circuit_size = plonk_verifier_key.padded_circuit_size();
    verifier.verifier_key = Some(plonk_verifier_key);
    let (_, vk) = PC::trim(u_params, padded_circuit_size, 0, None)
        .map_err(to_pc_error::<F, PC>)?;

    verifier.verify(proof, &vk, public_inputs)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{constraint_system::StandardComposer, util};
    use ark_bls12_377::Bls12_377;
    use ark_bls12_381::Bls12_381;
    use ark_ec::{
        twisted_edwards_extended::GroupAffine, AffineCurve, PairingEngine,
        ProjectiveCurve,
    };
    use ark_ff::{FftField, PrimeField};
    use rand_core::OsRng;

    // Implements a circuit that checks:
    // 1) a + b = c where C is a PI
    // 2) a <= 2^6
    // 3) b <= 2^5
    // 4) a * b = d where D is a PI
    // 5) JubJub::GENERATOR * e(JubJubScalar) = f where F is a PI
    #[derive(derivative::Derivative)]
    #[derivative(Debug(bound = ""), Default(bound = ""))]
    pub struct TestCircuit<F: FftField, P: TEModelParameters<BaseField = F>> {
        a: F,
        b: F,
        c: F,
        d: F,
        e: P::ScalarField,
        f: GroupAffine<P>,
    }

    impl<F, P> Circuit<F, P> for TestCircuit<F, P>
    where
        F: PrimeField,
        P: TEModelParameters<BaseField = F>,
    {
        const CIRCUIT_ID: [u8; 32] = [0xff; 32];

        fn gadget(
            &mut self,
            composer: &mut StandardComposer<F, P>,
        ) -> Result<(), Error> {
            let a = composer.add_input(self.a);
            let b = composer.add_input(self.b);
            let zero = composer.zero_var;

            // Make first constraint a + b = c (as public input)
            composer.arithmetic_gate(|gate| {
                gate.witness(a, b, Some(zero))
                    .add(F::one(), F::one())
                    .pi(-self.c)
            });

            // Check that a and b are in range
            composer.range_gate(a, 1 << 6);
            composer.range_gate(b, 1 << 5);
            // Make second constraint a * b = d
            composer.arithmetic_gate(|gate| {
                gate.witness(a, b, Some(zero)).mul(F::one()).pi(-self.d)
            });
            let e = composer
                .add_input(util::from_embedded_curve_scalar::<F, P>(self.e));
            let (x, y) = P::AFFINE_GENERATOR_COEFFS;
            let generator = GroupAffine::new(x, y);
            let scalar_mul_result =
                composer.fixed_base_scalar_mul(e, generator);

            // Apply the constrain
            composer.assert_equal_public_point(scalar_mul_result, self.f);
            Ok(())
        }

        fn padded_circuit_size(&self) -> usize {
            1 << 9
        }
    }

    fn test_full<F, P, PC>() -> Result<(), Error>
    where
        F: PrimeField,
        P: TEModelParameters<BaseField = F>,
        PC: HomomorphicCommitment<F>,
        VerifierData<F, PC>: PartialEq,
    {
        // Generate CRS
        let pp = PC::setup(1 << 10, None, &mut OsRng)
            .map_err(to_pc_error::<F, PC>)?;

        let mut circuit = TestCircuit::<F, P>::default();

        // Compile the circuit
        let (pk, (vk, _pi_pos)) = circuit.compile::<PC>(&pp)?;

        let (x, y) = P::AFFINE_GENERATOR_COEFFS;
        let generator: GroupAffine<P> = GroupAffine::new(x, y);
        let point_f_pi: GroupAffine<P> = AffineCurve::mul(
            &generator,
            P::ScalarField::from(2u64).into_repr(),
        )
        .into_affine();

        // Prover POV
        let (proof, pi) = {
            let mut circuit: TestCircuit<F, P> = TestCircuit {
                a: F::from(20u64),
                b: F::from(5u64),
                c: F::from(25u64),
                d: F::from(100u64),
                e: P::ScalarField::from(2u64),
                f: point_f_pi,
            };

            cfg_if::cfg_if! {
                if #[cfg(feature = "trace")] {
                    // Test trace
                    let mut prover: Prover<F, P, PC> = Prover::new(b"Test");
                    circuit.gadget(prover.mut_cs())?;
                    prover.cs.check_circuit_satisfied();
                }
            }

            circuit.gen_proof::<PC>(&pp, pk, b"Test")?
        };

        let verifier_data = VerifierData::new(vk, pi);

        // Test serialisation for verifier_data
        let mut verifier_data_bytes = Vec::new();
        verifier_data.serialize(&mut verifier_data_bytes).unwrap();

        let deserialized_verifier_data: VerifierData<F, PC> =
            VerifierData::deserialize(verifier_data_bytes.as_slice()).unwrap();

        assert!(deserialized_verifier_data == verifier_data);

        // Verifier POV

        // TODO: non-ideal hack for a first functional version.
        assert!(verify_proof::<F, P, PC>(
            &pp,
            verifier_data.key,
            &proof,
            &verifier_data.pi,
            b"Test",
        )
        .is_ok());

        Ok(())
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_full_on_Bls12_381() -> Result<(), Error> {
        test_full::<
            <Bls12_381 as PairingEngine>::Fr,
            ark_ed_on_bls12_381::EdwardsParameters,
            crate::commitment::KZG10<Bls12_381>,
        >()
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_full_on_Bls12_381_ipa() -> Result<(), Error> {
        test_full::<
            <Bls12_381 as PairingEngine>::Fr,
            ark_ed_on_bls12_381::EdwardsParameters,
            crate::commitment::IPA<
                <Bls12_381 as PairingEngine>::G1Affine,
                blake2::Blake2b,
            >,
        >()
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_full_on_Bls12_377() -> Result<(), Error> {
        test_full::<
            <Bls12_377 as PairingEngine>::Fr,
            ark_ed_on_bls12_377::EdwardsParameters,
            crate::commitment::KZG10<Bls12_377>,
        >()
    }
    #[test]
    #[allow(non_snake_case)]
    fn test_full_on_Bls12_377_ipa() -> Result<(), Error> {
        test_full::<
            <Bls12_377 as PairingEngine>::Fr,
            ark_ed_on_bls12_377::EdwardsParameters,
            crate::commitment::IPA<
                <Bls12_377 as PairingEngine>::G1Affine,
                blake2::Blake2b,
            >,
        >()
    }
}

/// deserializes a g1 point, which is encoded as a single element,
/// the X coordinate, with the 2 higher bits as flags.
/// If the first bit (lowest) is set, the point is the infinity point.
/// Y can be recovered using the curve equation with X, the equation
/// will give 2 posible values, we decide which one to take based on
/// the second (highest) flag.
fn deserialize_g1(mut x: [u64; 6]) -> G1Affine {
    let mask = 0x3fffffff_ffffffff;
    let int = BigInteger384::new(x);
    let infinity = int.get_bit(382);
    let greatest = int.get_bit(383);
    x[5] = x[5] & mask;
    if infinity {
        G1Affine::zero()
    } else {
        let x = Fq::from_repr(BigInteger384::new(x)).unwrap();
        G1Affine::get_point_from_x(x, greatest).unwrap()
    }
}

/// deserialize evals from the results of round 4b
fn deserialize_evals<F>(round4b: Round4BRes) -> ProofEvaluations<F>
where
    F: PrimeField<BigInt = BigInteger256>,
{
    let f = |x| F::from_repr(BigInteger256::new(x)).unwrap();
    let Round4BRes {
        a_eval,
        b_eval,
        c_eval,
        d_eval,
        a_next_eval,
        b_next_eval,
        d_next_eval,
        left_sigma_eval,
        right_sigma_eval,
        out_sigma_eval,
        permutation_eval,
        q_arith_eval,
        q_c_eval,
        q_l_eval,
        q_r_eval,
        q_hl_eval,
        q_hr_eval,
        q_h4_eval,
    } = round4b;
    let wire_evals = WireEvaluations {
        a_eval: f(a_eval),
        b_eval: f(b_eval),
        c_eval: f(c_eval),
        d_eval: f(d_eval),
    };

    let perm_evals = PermutationEvaluations {
        left_sigma_eval: f(left_sigma_eval),
        right_sigma_eval: f(right_sigma_eval),
        out_sigma_eval: f(out_sigma_eval),
        permutation_eval: f(permutation_eval),
    };

    // zeros as lookups aren't used
    let lookup_evals = LookupEvaluations {
        q_lookup_eval: F::zero(),
        z2_next_eval: F::one(),
        h1_eval: F::zero(),
        h1_next_eval: F::zero(),
        h2_eval: F::zero(),
        f_eval: F::zero(),
        table_eval: F::zero(),
        table_next_eval: F::zero(),
    };
    let custom = [
        q_arith_eval,
        q_c_eval,
        q_l_eval,
        q_r_eval,
        q_hl_eval,
        q_hr_eval,
        q_h4_eval,
        a_next_eval,
        b_next_eval,
        d_next_eval,
    ];
    let [q_arith_eval, q_c_eval, q_l_eval, q_r_eval, q_hl_eval, q_hr_eval, q_h4_eval, a_next_eval, b_next_eval, d_next_eval] =
        custom.map(f);
    let custom_evals = CustomEvaluations {
        vals: vec![
            label_eval!(q_arith_eval),
            label_eval!(q_c_eval),
            label_eval!(q_l_eval),
            label_eval!(q_r_eval),
            label_eval!(q_hl_eval),
            label_eval!(q_hr_eval),
            label_eval!(q_h4_eval),
            label_eval!(a_next_eval),
            label_eval!(b_next_eval),
            label_eval!(d_next_eval),
        ],
    };
    let evals = ProofEvaluations {
        wire_evals,
        perm_evals,
        lookup_evals,
        custom_evals,
    };
    evals
}

/// deserializes the results from rounds and combines them into
/// the proof
fn assemble_proof<F, PC>(
    round1: Round1Res,
    round3: Round3Res,
    round4a: Round4ARes,
    round4b: Round4BRes,
    round5: Round5Res,
    first_g_power: G1Affine,
) -> Proof<F, PC>
where
    F: PrimeField<BigInt = BigInteger256>,
    PC: HomomorphicCommitment<
        F,
        Commitment = Commitment<Bls12_381>,
        Proof = kzg10::Proof<Bls12_381>,
    >,
{
    let Round1Res {
        w_l_commit,
        w_r_commit,
        w_o_commit,
        w_4_commit,
    } = round1;
    let g1 = |x| kzg10::Commitment(deserialize_g1(x));
    let a_comm = g1(w_l_commit);
    let b_comm = g1(w_r_commit);
    let c_comm = g1(w_o_commit);
    let d_comm = g1(w_4_commit);

    let Round3Res { z_commit } = round3;
    let z_comm = g1(z_commit);

    // for commitments known to be zero in this circuit
    let zero_comm = || kzg10::Commitment(G1Affine::zero());

    let f_comm = zero_comm();
    let h_1_comm = zero_comm();
    let h_2_comm = zero_comm();
    let z_2_comm = kzg10::Commitment(first_g_power);

    let Round4ARes {
        t_0_commit,
        t_1_commit,
        t_2_commit,
        t_3_commit,
        t_4_commit,
        t_5_commit,
        t_6_commit,
        t_7_commit,
    } = round4a;
    let t = [
        t_0_commit, t_1_commit, t_2_commit, t_3_commit, t_4_commit, t_5_commit,
        t_6_commit, t_7_commit,
    ];
    let [t_1_comm, t_2_comm, t_3_comm, t_4_comm, t_5_comm, t_6_comm, t_7_comm, t_8_comm] =
        t.map(|x| g1(x));
    let Round5Res {
        aw_opening,
        saw_opening,
    } = round5;
    let aw_opening = kzg10::Proof {
        w: deserialize_g1(aw_opening),
        random_v: None,
    };
    let saw_opening = kzg10::Proof {
        w: deserialize_g1(saw_opening),
        random_v: None,
    };

    let evaluations = deserialize_evals(round4b);
    Proof {
        a_comm,
        b_comm,
        c_comm,
        d_comm,
        z_comm,
        f_comm,
        h_1_comm,
        h_2_comm,
        z_2_comm,
        t_1_comm,
        t_2_comm,
        t_3_comm,
        t_4_comm,
        t_5_comm,
        t_6_comm,
        t_7_comm,
        t_8_comm,
        aw_opening,
        saw_opening,
        evaluations,
    }
}
