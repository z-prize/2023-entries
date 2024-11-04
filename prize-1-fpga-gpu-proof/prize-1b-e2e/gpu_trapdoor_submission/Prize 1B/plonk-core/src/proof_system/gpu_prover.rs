use super::{linearisation_poly::ProofEvaluations, ProverKey};

use crate::{
    commitment::HomomorphicCommitment,
    constraint_system::SBOX_ALPHA,
    label_eval,
    permutation::{
        self, gpu_coset_fft, gpu_coset_fft_gscalar, gpu_coset_ifft_gscalar,
    },
    proof_system::{
        linearisation_poly::{
            self, CustomEvaluations, LookupEvaluations, PermutationEvaluations,
            WireEvaluations,
        },
        proof, quotient_poly,
    },
    util::EvaluationDomainExt,
};
use ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve, TEModelParameters};
use ark_ff::{BigInteger256, PrimeField, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain,
    Polynomial, UVPolynomial,
};

use ark_poly_commit::{kzg10::Proof, PolynomialCommitment};
use ec_gpu_common::DeviceMemory;
use ec_gpu_common::{
    gpu_msm, log2_floor, to_mont, CUdeviceptr, Fr as GpuFr, GPUResult, GpuPoly,
    GpuPolyContainer, MSMContext, PolyKernel, PrimeField as GpuPrimeField,
    MAX_POLY_LEN_LOG,
};

use rayon::prelude::*;
use std::marker::PhantomData;
use std::mem::transmute;

pub struct GPUFunc<
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
    PC: HomomorphicCommitment<F>,
> {
    _phantomf: PhantomData<F>,
    _phantomp: PhantomData<P>,
    _phantompc: PhantomData<PC>,
}

pub fn prepare_domain<F: PrimeField, Fg: GpuPrimeField>(
    kern: &PolyKernel<Fg>,
    gpu_container: &mut GpuPolyContainer<Fg>,
    domain: &GeneralEvaluationDomain<F>,
    if_powers: bool,
) -> GPUResult<()> {
    let n = domain.size();
    let lg_n = log2_floor(n);
    let max_deg = std::cmp::min(ec_gpu_common::MAX_RADIX_DEGREE, lg_n);

    // tmp is dangerous
    let mut tmp_buf = gpu_container.ask_for(&kern, n)?;
    tmp_buf.fill_with_zero().unwrap();

    let mut gpu_pq = gpu_container.ask_for_pq(&kern)?;
    let mut gpu_omegas = gpu_container.ask_for_omegas_with_size(&kern, n)?;
    let mut gpu_pq_ifft = gpu_container.ask_for_pq(&kern)?;
    let mut gpu_omegas_ifft =
        gpu_container.ask_for_omegas_with_size(&kern, n)?;

    let omega = domain.group_gen();
    let omega_inv = domain.group_gen_inv();

    gpu_pq.setup_pq(&omega, n, max_deg)?;
    gpu_omegas.setup_omegas_with_size(&omega, n)?;
    gpu_pq_ifft.setup_pq(&omega_inv, n, max_deg)?;
    gpu_omegas_ifft.setup_omegas_with_size(&omega_inv, n)?;
    if if_powers == true {
        let mut gpu_powers = gpu_container.ask_for_powers(&kern)?;
        gpu_powers.fill_with_fe(&Fg::zero()).unwrap();

        let mut gpu_domain_powers_group_gen =
            gpu_container.ask_for(&kern, n)?;
        gpu_domain_powers_group_gen
            .fill_with_fe(&Fg::zero())
            .unwrap();
        gpu_domain_powers_group_gen.generate_powers(&mut gpu_powers, &omega)?;
        gpu_container.save(
            &format!("domain_{n}_powers_group_gen"),
            gpu_domain_powers_group_gen,
        )?;
        gpu_container.recycle(gpu_powers)?;
    }

    gpu_container.save(&format!("domain_{n}_pq"), gpu_pq)?;
    gpu_container.save(&format!("domain_{n}_omegas"), gpu_omegas)?;
    gpu_container.save(&format!("domain_{n}_pq_ifft"), gpu_pq_ifft)?;
    gpu_container.save(&format!("domain_{n}_omegas_ifft"), gpu_omegas_ifft)?;
    gpu_container.save(&format!("domain_{n}_tmp_buf"), tmp_buf)?;
    Ok(())
}

pub fn prepare_domain_8n_coset<F: PrimeField, Fg: GpuPrimeField>(
    kern: &PolyKernel<Fg>,
    gpu_container: &mut GpuPolyContainer<Fg>,
    prover_key: &ProverKey<F>,
    domain_8n: &GeneralEvaluationDomain<F>,
) -> GPUResult<()> {
    let n = domain_8n.size();
    let mut gpu_coset_powers = gpu_container.ask_for(kern, n).unwrap();
    gpu_coset_powers.fill_with_zero().unwrap();

    gpu_coset_powers
        .read_from(&prover_key.coset_powers_8n()[..])
        .unwrap();
    gpu_container
        .save(&format!("domain_{}_coset_powers", n), gpu_coset_powers)
        .unwrap();
    let mut gpu_coset_powers_ifft = gpu_container.ask_for(kern, n).unwrap();
    gpu_coset_powers_ifft.fill_with_zero().unwrap();
    gpu_coset_powers_ifft
        .read_from(&prover_key.coset_powers_8n_ifft()[..])
        .unwrap();
    gpu_container
        .save(
            &format!("domain_{}_coset_powers_ifft", n),
            gpu_coset_powers_ifft,
        )
        .unwrap();
    Ok(())
}

pub fn compute_quotient_gpu<F: PrimeField>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    domain: &GeneralEvaluationDomain<F>,
    domain_8n: &GeneralEvaluationDomain<F>,
    prover_key: &ProverKey<F>,
    z_poly: &GpuPoly<GpuFr>,
    z2_poly: &DensePolynomial<F>,
    w_l_poly: &GpuPoly<GpuFr>,
    w_r_poly: &GpuPoly<GpuFr>,
    w_o_poly: &GpuPoly<GpuFr>,
    w_4_poly: &GpuPoly<GpuFr>,
    f_poly: &DensePolynomial<F>,
    table_poly: &DensePolynomial<F>,
    h1_poly: &DensePolynomial<F>,
    h2_poly: &DensePolynomial<F>,
    alpha: &F,
    beta: &F,
    gamma: &F,
    delta: &F,
    epsilon: &F,
    lookup_challenge: &F,
) {
    let n = domain_8n.size();

    let l1_eval_8n = prover_key.l1_poly_coset_8n();

    gpu_coset_fft_gscalar(kern, gpu_container, z_poly, "z_eval_8n", n);
    // calc coset w_l
    gpu_coset_fft_gscalar(kern, gpu_container, w_l_poly, "wl_eval_8n", n);
    // calc coset w_r
    gpu_coset_fft_gscalar(kern, gpu_container, w_r_poly, "wr_eval_8n", n);
    // calc coset w_o
    gpu_coset_fft_gscalar(kern, gpu_container, w_o_poly, "wo_eval_8n", n);
    // calc coset w_4
    gpu_coset_fft_gscalar(kern, gpu_container, w_4_poly, "w4_eval_8n", n);
    // calc coset z2

    // Compute gate constraint
    // q_arith * (q_l * w_l + q_r * w_r + q_o * w_o + q_4 * w_4 + q_m * w_l *
    // w_r +            q_hl * w_l^5 + q_hr * w_r^5 + q_h4 * w_4^5 + q_c) +
    // PI q_l * w_l
    let mut gpu_w_l = gpu_container.find(kern, "wl_eval_8n").unwrap();
    let mut gpu_res_final = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_final.fill_with_zero().unwrap();

    // gpu_res_final
    //     .read_from(&prover_key.arithmetic.q_l.1.evals)
    //     .unwrap();

    let mut q_l_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_l_1_evals")
        .unwrap();
    gpu_res_final.copy_from_gpu(&q_l_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_l_1_evals", q_l_1_evals)
        .unwrap();
    gpu_res_final.mul_assign(&gpu_w_l).unwrap();

    // q_r * w_r
    let mut gpu_w_r = gpu_container.find(kern, "wr_eval_8n").unwrap();
    let mut gpu_res_tmp = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_tmp.fill_with_zero().unwrap();
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_r.1.evals)
    //     .unwrap();
    let mut q_r_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_r_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_r_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_r_1_evals", q_r_1_evals)
        .unwrap();
    gpu_res_tmp.mul_assign(&gpu_w_r).unwrap();
    // sum = q_l * w_l + q_r * w_r
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    // q_o * w_o
    let mut gpu_w_o = gpu_container.find(kern, "wo_eval_8n").unwrap();
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_o.1.evals)
    //     .unwrap();
    let mut q_o_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_o_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_o_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_o_1_evals", q_o_1_evals)
        .unwrap();
    gpu_res_tmp.mul_assign(&gpu_w_o).unwrap();
    // sum += q_o * w_o
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    // q_4 * w_4
    let mut gpu_w_4 = gpu_container.find(kern, "w4_eval_8n").unwrap();
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_4.1.evals)
    //     .unwrap();

    let mut q_4_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_4_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_4_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_4_1_evals", q_4_1_evals)
        .unwrap();

    gpu_res_tmp.mul_assign(&gpu_w_4).unwrap();
    // sum += q_4 * w_4
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    // q_m * w_l * w_r
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_m.1.evals)
    //     .unwrap();

    let mut q_m_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_m_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_m_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_m_1_evals", q_m_1_evals)
        .unwrap();

    gpu_res_tmp.mul_assign(&gpu_w_l).unwrap();
    gpu_res_tmp.mul_assign(&gpu_w_r).unwrap();
    // sum += q_m * w_l * w_r
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();

    // q_hl * w_l^5
    let mut gpu_res_tmph = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_tmph.fill_with_zero().unwrap();

    gpu_res_tmph.copy_from_gpu(&gpu_w_l).unwrap();
    gpu_res_tmph.square().unwrap();
    gpu_res_tmph.square().unwrap();
    gpu_res_tmph.mul_assign(&gpu_w_l).unwrap();
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_hl.1.evals)
    //     .unwrap();

    let mut q_hl_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_hl_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_hl_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_hl_1_evals", q_hl_1_evals)
        .unwrap();

    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    // sum += q_hl * w_l^5
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    // q_hr * w_r^5
    gpu_res_tmph.copy_from_gpu(&gpu_w_r).unwrap();
    gpu_res_tmph.square().unwrap();
    gpu_res_tmph.square().unwrap();
    gpu_res_tmph.mul_assign(&gpu_w_r).unwrap();
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_hr.1.evals)
    //     .unwrap();
    let mut q_hr_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_hr_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_hr_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_hr_1_evals", q_hr_1_evals)
        .unwrap();

    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    // sum += q_hr * w_r^5
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    // q_h4 * w_4^5
    gpu_res_tmph.copy_from_gpu(&gpu_w_4).unwrap();
    gpu_res_tmph.square().unwrap();
    gpu_res_tmph.square().unwrap();
    gpu_res_tmph.mul_assign(&gpu_w_4).unwrap();
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_h4.1.evals)
    //     .unwrap();
    let mut q_h4_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_h4_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_h4_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_h4_1_evals", q_h4_1_evals)
        .unwrap();

    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    // sum += q_h4 * w_4^5
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    // q_c & q_arith
    // gpu_res_tmp
    //     .read_from(&prover_key.arithmetic.q_c.1.evals)
    //     .unwrap();
    let mut q_c_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_c_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&q_c_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_c_1_evals", q_c_1_evals)
        .unwrap();

    // gpu_res_tmph
    //     .read_from(&prover_key.arithmetic.q_arith.1.evals)
    //     .unwrap();
    let mut q_arith_1_evals = gpu_container
        .find(kern, "prover_key.arithmetic.q_arith_1_evals")
        .unwrap();
    gpu_res_tmph.copy_from_gpu(&q_arith_1_evals).unwrap();
    gpu_container
        .save("prover_key.arithmetic.q_arith_1_evals", q_arith_1_evals)
        .unwrap();

    // sum += q_c
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    // sum *= q_arith
    gpu_res_final.mul_assign(&gpu_res_tmph).unwrap();

    // calc pi_poly
    let mut gpu_pi = gpu_container.find(kern, "pi_poly").unwrap();
    gpu_coset_fft_gscalar(kern, gpu_container, &gpu_pi, "pi_eval_8n", n);
    let mut gpu_pi_8n = gpu_container.find(kern, "pi_eval_8n").unwrap();
    gpu_res_final.add_assign(&gpu_pi_8n).unwrap();
    gpu_pi_8n.fill_with_zero().unwrap();
    gpu_container.recycle(gpu_pi_8n).unwrap();

    gpu_pi.fill_with_zero().unwrap();
    gpu_container.recycle(gpu_pi).unwrap();

    // permutation
    // 1.(w_l + beta * X + gamma) (w_r + beta * k1 * X + gamma) (w_o + beta *
    // k2 * X + gamma)(w_4 + beta * k3 * X + gamma)z(X) * alpha
    // (1). w_l + beta * X  + gamma
    let mut gpu_res_perm = gpu_container.ask_for(kern, n).unwrap();

    gpu_res_perm.fill_with_zero().unwrap();

    // gpu_res_tmph
    //     .read_from(&prover_key.permutation.linear_evaluations.evals)
    //     .unwrap();
    let mut linear_evaluations_evals = gpu_container
        .find(kern, "prover_key.permutation.linear_evaluations_evals")
        .unwrap();
    gpu_res_perm
        .copy_from_gpu(&linear_evaluations_evals)
        .unwrap();

    // gpu_res_perm.copy_from_gpu(&gpu_res_tmph).unwrap();

    gpu_res_perm.scale(beta).unwrap();
    gpu_res_perm.add_constant(gamma).unwrap();
    gpu_res_perm.add_assign(&gpu_w_l).unwrap();

    // (2). w_r + beta * K1 * X  + gamma
    gpu_res_tmp
        .copy_from_gpu(&linear_evaluations_evals)
        .unwrap();
    let k1_beta = crate::permutation::constants::K1::<F>() * beta;
    gpu_res_tmp.scale(&k1_beta).unwrap();
    gpu_res_tmp.add_constant(gamma).unwrap();
    gpu_res_tmp.add_assign(&gpu_w_r).unwrap();
    // product = (1) * (2)

    gpu_res_perm.mul_assign(&gpu_res_tmp).unwrap();

    // ==
    // (3). w_o + beta * K2 * X  + gamma
    gpu_res_tmp
        .copy_from_gpu(&linear_evaluations_evals)
        .unwrap();
    let k2_beta = crate::permutation::constants::K2::<F>() * beta;
    gpu_res_tmp.scale(&k2_beta).unwrap();
    gpu_res_tmp.add_constant(gamma).unwrap();
    gpu_res_tmp.add_assign(&gpu_w_o).unwrap();
    // product *= (3)

    gpu_res_perm.mul_assign(&gpu_res_tmp).unwrap();
    // (4). w_4 + beta * K3 * X  + gamma
    gpu_res_tmp
        .copy_from_gpu(&linear_evaluations_evals)
        .unwrap();
    let k3_beta = crate::permutation::constants::K3::<F>() * beta;
    gpu_res_tmp.scale(&k3_beta).unwrap();
    gpu_res_tmp.add_constant(gamma).unwrap();
    gpu_res_tmp.add_assign(&gpu_w_4).unwrap();
    // product *= (4)
    gpu_res_perm.mul_assign(&gpu_res_tmp).unwrap();

    // product *= z
    let mut gpu_z = gpu_container.find(kern, "z_eval_8n").unwrap();
    gpu_res_perm.mul_assign(&gpu_z).unwrap();
    gpu_res_perm.scale(alpha).unwrap();

    gpu_container
        .save(
            "prover_key.permutation.linear_evaluations_evals",
            linear_evaluations_evals,
        )
        .unwrap();

    //2. (w_l + beta* Sigma1(X) + gamma) (w_r + beta * Sigma2(X) + gamma) (w_o
    // + beta * Sigma3(X) + gamma)(w_4 + beta * Sigma4(X) + gamma) Z(X.omega) *
    // alpha
    //(1). w_l + beta* Sigma1(X) + gamma
    // gpu_res_tmph
    //     .read_from(&prover_key.permutation.left_sigma.1.evals)
    //     .unwrap();

    let mut left_sigma_1_evals = gpu_container
        .find(kern, "prover_key.permutation.left_sigma_1_evals")
        .unwrap();
    gpu_res_tmph.copy_from_gpu(&left_sigma_1_evals).unwrap();
    gpu_container
        .save(
            "prover_key.permutation.left_sigma_1_evals",
            left_sigma_1_evals,
        )
        .unwrap();
    gpu_res_tmph.scale(beta).unwrap();
    gpu_res_tmph.add_constant(gamma).unwrap();
    gpu_res_tmph.add_assign(&gpu_w_l).unwrap();
    //(2). w_r + beta* Sigma2(X) + gamma
    // gpu_res_tmp
    //     .read_from(&prover_key.permutation.right_sigma.1.evals)
    //     .unwrap();

    let mut right_sigma_1_evals = gpu_container
        .find(kern, "prover_key.permutation.right_sigma_1_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&right_sigma_1_evals).unwrap();
    gpu_container
        .save(
            "prover_key.permutation.right_sigma_1_evals",
            right_sigma_1_evals,
        )
        .unwrap();

    gpu_res_tmp.scale(beta).unwrap();
    gpu_res_tmp.add_constant(gamma).unwrap();
    gpu_res_tmp.add_assign(&gpu_w_r).unwrap();
    // product = (1) * (2)
    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    //(3). w_o + beta* Sigma3(X) + gamma
    // gpu_res_tmph
    //     .read_from(&prover_key.permutation.out_sigma.1.evals)
    //     .unwrap();
    let mut out_sigma_1_evals = gpu_container
        .find(kern, "prover_key.permutation.out_sigma_1_evals")
        .unwrap();
    gpu_res_tmph.copy_from_gpu(&out_sigma_1_evals).unwrap();
    gpu_container
        .save(
            "prover_key.permutation.out_sigma_1_evals",
            out_sigma_1_evals,
        )
        .unwrap();

    gpu_res_tmph.scale(beta).unwrap();
    gpu_res_tmph.add_constant(gamma).unwrap();
    gpu_res_tmph.add_assign(&gpu_w_o).unwrap();
    // product *= (3)
    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    //(4). w_4 + beta* Sigma4(X) + gamma
    // gpu_res_tmph
    //     .read_from(&prover_key.permutation.fourth_sigma.1.evals)
    //     .unwrap();

    let mut fourth_sigma_1_evals = gpu_container
        .find(kern, "prover_key.permutation.fourth_sigma_1_evals")
        .unwrap();
    gpu_res_tmph.copy_from_gpu(&fourth_sigma_1_evals).unwrap();
    gpu_container
        .save(
            "prover_key.permutation.fourth_sigma_1_evals",
            fourth_sigma_1_evals,
        )
        .unwrap();

    gpu_res_tmph.scale(beta).unwrap();
    gpu_res_tmph.add_constant(gamma).unwrap();
    gpu_res_tmph.add_assign(&gpu_w_4).unwrap();
    // product *= (4)
    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    // product *= z_next
    //gpu_res_tmph.read_from(&z_eval_8n[8..]).unwrap();
    gpu_res_tmph
        .copy_from_gpu_offset_with_len(&gpu_z, 8, n - 8)
        .unwrap();
    gpu_z
        .copy_to_gpu_offset_with_len(&mut gpu_res_tmph, n - 8, 8)
        .unwrap();

    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    // product *= alpha
    gpu_res_tmp.scale(alpha).unwrap();

    // sub
    gpu_res_perm.sub_assign(&gpu_res_tmp).unwrap();
    // 3. L_1(X)[Z(X) - 1] * alpha^2
    // gpu_res_tmp.read_from(&l1_eval_8n).unwrap();
    let mut l1_eval_8n_gpu = gpu_container
        .find(kern, "prover_key.arithmetic.l1_eval_8n_gpu")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&l1_eval_8n_gpu).unwrap();
    gpu_container
        .save("prover_key.arithmetic.l1_eval_8n_gpu", l1_eval_8n_gpu)
        .unwrap();

    gpu_res_tmph.copy_from_gpu(&gpu_z).unwrap();
    gpu_res_tmph.sub_constant(&F::one()).unwrap();
    let alpha_sqr = alpha.square();
    gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    gpu_res_tmp.scale(&alpha_sqr).unwrap();
    //add
    gpu_res_perm.add_assign(&gpu_res_tmp).unwrap();

    // gate + permutation
    gpu_res_final.add_assign(&gpu_res_perm).unwrap();

    // compute lookup
    //let lookup_sep_sq = lookup_challenge.square();
    //let lookup_sep_cu = lookup_sep_sq * lookup_challenge;
    //let one_plus_delta = *delta + F::one();
    //let epsilon_one_plus_delta = *epsilon * one_plus_delta;
    //// 1. z2(X) * (1+δ) * (ε+f(X)) * (ε*(1+δ) + t(X) + δt(Xω)) * lookup_sep^2
    //// (1). ε*(1+δ) + t(X) + δt(Xω)
    ////gpu_res_perm.read_from(&table_eval_8n[8..]).unwrap();
    ////let gpu_t_poly = gpu_container.find(kern, "table_eval_8n").unwrap();
    //gpu_res_tmp.fill_with_fe(&F::zero()).unwrap();
    //gpu_res_tmp.scale(delta).unwrap();
    ////gpu_res_tmp.add_assign(&gpu_t_poly).unwrap();
    //gpu_res_tmp.add_constant(&epsilon_one_plus_delta).unwrap();
    //// (2).(ε+f(X))
    ////let gpu_f = gpu_container.find(kern, "f_eval_8n").unwrap();
    ////gpu_res_perm.copy_from_gpu(&gpu_f).unwrap();
    //gpu_res_perm.fill_with_fe(&F::zero()).unwrap();
    //gpu_res_perm.add_constant(epsilon).unwrap();
    //// (3) (1) * (2)
    //gpu_res_perm.mul_assign(&gpu_res_tmp).unwrap();
    //// (4). (3) * z2(x)
    ////let gpu_z2 = gpu_container.find(kern, "z2_eval_8n").unwrap();
    //gpu_res_tmp.fill_with_fe(&F::one()).unwrap();
    //gpu_res_perm.mul_assign(&gpu_res_tmp).unwrap();
    //// (5). (4) * (1+δ) * lookup_seq^2
    //let part1_used = lookup_sep_sq * one_plus_delta;
    //gpu_res_perm.scale(&part1_used).unwrap();

    // 2. − z2(Xω) * (ε*(1+δ) + h1(X) + δ*h2(X)) * (ε*(1+δ) + h2(X) + δ*h1(Xω))
    // * lookup_sep^2
    // (1). (ε*(1+δ) + h1(X) + δ*h2(X))
    //let gpu_h1 = gpu_container.find(kern, "h1_eval_8n").unwrap();
    //let gpu_h2 = gpu_container.find(kern, "h2_eval_8n").unwrap();
    //gpu_res_tmp.fill_with_fe(&F::zero()).unwrap();
    //gpu_res_tmp.scale(delta).unwrap();
    ////gpu_res_tmp.add_assign(&gpu_h1).unwrap();
    //gpu_res_tmp.add_constant(&epsilon_one_plus_delta).unwrap();
    //// (2). (ε*(1+δ) + h2(X) + δ*h1(Xω))
    ////gpu_res_tmph.read_from(&h1_eval_8n[8..]).unwrap();
    //gpu_res_tmph.fill_with_fe(&F::zero()).unwrap();
    //gpu_res_tmph.scale(delta).unwrap();
    ////gpu_res_tmph.add_assign(&gpu_h2).unwrap();
    //gpu_res_tmph.add_constant(&epsilon_one_plus_delta).unwrap();
    ////(3). (1) * (2)
    //gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    ////(4). (3) * z2(wx)
    ////gpu_res_tmph.read_from(&z2_eval_8n[8..]).unwrap();
    //gpu_res_tmph.fill_with_fe(&F::one()).unwrap();
    //gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    //// (5). (4) * lookup_seq^2
    //gpu_res_tmp.scale(&lookup_sep_sq).unwrap();
    //// sub
    //gpu_res_perm.sub_assign(&gpu_res_tmp).unwrap();
    // 3. (z2 - 1) * l1 * lookup_seq^3
    //gpu_res_tmp.copy_from_gpu(&gpu_z2).unwrap();
    //gpu_res_tmp.fill_with_fe(&F::one()).unwrap();
    //gpu_res_tmp.sub_constant(&F::one()).unwrap();
    //gpu_res_tmp.scale(&lookup_sep_cu).unwrap();
    //gpu_res_tmph.read_from(&l1_eval_8n).unwrap();
    //gpu_res_tmp.mul_assign(&gpu_res_tmph).unwrap();
    //// add
    //gpu_res_perm.add_assign(&gpu_res_tmp).unwrap();

    // gate + perm + lookup
    //gpu_res_final.add_assign(&gpu_res_perm).unwrap();
    //let mut permutation = vec![F::zero(); n];
    //gpu_res_final.write_to(&mut permutation).unwrap();

    // gpu_res_tmp
    //     .read_from(&prover_key.v_h_coset_8n_inv().evals)
    //     .unwrap();

    let mut v_h_coset_8n_inv_evals = gpu_container
        .find(kern, "prover_key.arithmetic.v_h_coset_8n_inv_evals")
        .unwrap();
    gpu_res_tmp.copy_from_gpu(&v_h_coset_8n_inv_evals).unwrap();
    gpu_container
        .save(
            "prover_key.arithmetic.v_h_coset_8n_inv_evals",
            v_h_coset_8n_inv_evals,
        )
        .unwrap();

    gpu_res_final.mul_assign(&gpu_res_tmp).unwrap();

    gpu_container.recycle(gpu_res_tmph).unwrap();
    gpu_container.recycle(gpu_res_tmp).unwrap();
    gpu_container.recycle(gpu_res_perm).unwrap();
    gpu_container.recycle(gpu_z).unwrap();
    gpu_container.recycle(gpu_w_l).unwrap();
    gpu_container.recycle(gpu_w_r).unwrap();
    gpu_container.recycle(gpu_w_o).unwrap();
    gpu_container.recycle(gpu_w_4).unwrap();

    gpu_coset_ifft_gscalar(
        kern,
        gpu_container,
        &gpu_res_final,
        "quotient",
        n,
        domain_8n.size_inv(),
    );

    gpu_container.recycle(gpu_res_final).unwrap();
}

/// Compute the linearisation polynomial.
pub fn compute_linear_gpu<F, P>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    domain: &GeneralEvaluationDomain<F>,
    prover_key: &ProverKey<F>,
    alpha: &F,
    beta: &F,
    gamma: &F,
    delta: &F,
    epsilon: &F,
    zeta: &F,
    lookup_separation_challenge: &F,
    z_challenge: &F,
    w_l_poly: &GpuPoly<GpuFr>,
    w_r_poly: &GpuPoly<GpuFr>,
    w_o_poly: &GpuPoly<GpuFr>,
    w_4_poly: &GpuPoly<GpuFr>,
    t_1_poly: CUdeviceptr,
    t_2_poly: CUdeviceptr,
    t_3_poly: CUdeviceptr,
    t_4_poly: CUdeviceptr,
    t_5_poly: CUdeviceptr,
    t_6_poly: CUdeviceptr,
    z_poly: &GpuPoly<GpuFr>,
    z2_poly: &DensePolynomial<F>,
    f_poly: &DensePolynomial<F>,
    h1_poly: &DensePolynomial<F>,
    h2_poly: &DensePolynomial<F>,
    table_poly: &DensePolynomial<F>,
) -> (F, ProofEvaluations<F>)
where
    F: PrimeField,
    P: TEModelParameters<BaseField = F>,
{
    let n = domain.size();
    let omega = domain.group_gen();
    let shifted_z_challenge = *z_challenge * omega;

    // Wire evaluations
    let mut evaluate_powers =
        gpu_container.ask_for(kern, MAX_POLY_LEN_LOG).unwrap();
    evaluate_powers.fill_with_zero().unwrap();

    evaluate_powers.setup_powers(z_challenge).unwrap();
    let mut shifted_evaluate_powers =
        gpu_container.ask_for(kern, MAX_POLY_LEN_LOG).unwrap();
    shifted_evaluate_powers.fill_with_zero().unwrap();

    shifted_evaluate_powers
        .setup_powers(&shifted_z_challenge)
        .unwrap();
    let mut gpu_res_tmp = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_tmp.fill_with_zero().unwrap();

    let a_eval = w_l_poly
        .evaluate_at_naive(&mut evaluate_powers, &mut gpu_res_tmp, z_challenge)
        .unwrap();
    let b_eval = w_r_poly
        .evaluate_at_naive(&mut evaluate_powers, &mut gpu_res_tmp, z_challenge)
        .unwrap();
    let c_eval = w_o_poly
        .evaluate_at_naive(&mut evaluate_powers, &mut gpu_res_tmp, z_challenge)
        .unwrap();
    let d_eval = w_4_poly
        .evaluate_at_naive(&mut evaluate_powers, &mut gpu_res_tmp, z_challenge)
        .unwrap();
    gpu_container.recycle(evaluate_powers).unwrap();
    let a_next_eval = w_l_poly
        .evaluate_at_naive(
            &mut shifted_evaluate_powers,
            &mut gpu_res_tmp,
            &shifted_z_challenge,
        )
        .unwrap();
    let b_next_eval = w_r_poly
        .evaluate_at_naive(
            &mut shifted_evaluate_powers,
            &mut gpu_res_tmp,
            &shifted_z_challenge,
        )
        .unwrap();
    let c_next_eval = w_o_poly
        .evaluate_at_naive(
            &mut shifted_evaluate_powers,
            &mut gpu_res_tmp,
            &shifted_z_challenge,
        )
        .unwrap();
    let d_next_eval = w_4_poly
        .evaluate_at_naive(
            &mut shifted_evaluate_powers,
            &mut gpu_res_tmp,
            &shifted_z_challenge,
        )
        .unwrap();
    let wire_evals = WireEvaluations {
        a_eval,
        b_eval,
        c_eval,
        d_eval,
    };
    // Permutation evaluations
    let left_sigma_eval =
        prover_key.permutation.left_sigma.0.evaluate(z_challenge);
    let right_sigma_eval =
        prover_key.permutation.right_sigma.0.evaluate(z_challenge);
    let out_sigma_eval =
        prover_key.permutation.out_sigma.0.evaluate(z_challenge);
    //let permutation_eval = z_poly.evaluate(&shifted_z_challenge);
    let permutation_eval = z_poly
        .evaluate_at_naive(
            &mut shifted_evaluate_powers,
            &mut gpu_res_tmp,
            &shifted_z_challenge,
        )
        .unwrap();
    gpu_container.recycle(shifted_evaluate_powers).unwrap();
    let perm_evals = PermutationEvaluations {
        left_sigma_eval,
        right_sigma_eval,
        out_sigma_eval,
        permutation_eval,
    };

    // Arith selector evaluation
    let q_arith_eval = prover_key.arithmetic.q_arith.0.evaluate(z_challenge);

    // Lookup selector evaluation
    let q_lookup_eval = prover_key.lookup.q_lookup.0.evaluate(z_challenge);

    // Custom gate evaluations
    let q_c_eval = prover_key.arithmetic.q_c.0.evaluate(z_challenge);
    let q_l_eval = prover_key.arithmetic.q_l.0.evaluate(z_challenge);
    let q_r_eval = prover_key.arithmetic.q_r.0.evaluate(z_challenge);

    // High degree selector evaluations
    let q_hl_eval = prover_key.arithmetic.q_hl.0.evaluate(z_challenge);
    let q_hr_eval = prover_key.arithmetic.q_hr.0.evaluate(z_challenge);
    let q_h4_eval = prover_key.arithmetic.q_h4.0.evaluate(z_challenge);

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

    let z2_next_eval = z2_poly.evaluate(&shifted_z_challenge);
    let h1_eval = h1_poly.evaluate(z_challenge);
    let h1_next_eval = h1_poly.evaluate(&shifted_z_challenge);
    let h2_eval = h2_poly.evaluate(z_challenge);
    let f_eval = f_poly.evaluate(z_challenge);
    let table_eval = table_poly.evaluate(z_challenge);
    let table_next_eval = table_poly.evaluate(&shifted_z_challenge);

    // Compute the last term in the linearisation polynomial
    // (negative_quotient_term):
    // - Z_h(z_challenge) * [t_1(X) + z_challenge^n * t_2(X) + z_challenge^2n *
    //   t_3(X) + z_challenge^3n * t_4(X)]
    let vanishing_poly_eval =
        domain.evaluate_vanishing_polynomial(*z_challenge);
    let z_challenge_to_n = vanishing_poly_eval + F::one();
    let l1_eval = proof::compute_first_lagrange_evaluation(
        domain,
        &vanishing_poly_eval,
        z_challenge,
    );

    let lookup_evals = LookupEvaluations {
        q_lookup_eval,
        z2_next_eval,
        h1_eval,
        h1_next_eval,
        h2_eval,
        f_eval,
        table_eval,
        table_next_eval,
    };

    // gate constraints
    // 1. q_arith * (qm * w_l * w_r + q_l * w_l + q_r * w_r + q_o * w_o + q_4 *
    // w_ 4 + q_hl * w_l^5 +      q_hr * w_r^5 + q_h4 * w_4^5 + q_c)
    let mut gpu_res_final = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_final.fill_with_zero().unwrap();

    gpu_res_final
        .read_from(&prover_key.arithmetic.q_m.0)
        .unwrap();
    gpu_res_final.scale(&(a_eval * b_eval)).unwrap();
    gpu_res_tmp.read_from(&prover_key.arithmetic.q_l.0).unwrap();
    gpu_res_tmp.scale(&a_eval).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_tmp.read_from(&prover_key.arithmetic.q_r.0).unwrap();
    gpu_res_tmp.scale(&b_eval).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_tmp.read_from(&prover_key.arithmetic.q_o.0).unwrap();
    gpu_res_tmp.scale(&c_eval).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_tmp.read_from(&prover_key.arithmetic.q_4.0).unwrap();
    gpu_res_tmp.scale(&d_eval).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_tmp
        .read_from(&prover_key.arithmetic.q_hl.0)
        .unwrap();
    gpu_res_tmp.scale(&a_eval.pow([SBOX_ALPHA])).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_tmp
        .read_from(&prover_key.arithmetic.q_hr.0)
        .unwrap();
    gpu_res_tmp.scale(&b_eval.pow([SBOX_ALPHA])).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_tmp
        .read_from(&prover_key.arithmetic.q_h4.0)
        .unwrap();
    gpu_res_tmp.scale(&d_eval.pow([SBOX_ALPHA])).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_tmp.read_from(&prover_key.arithmetic.q_c.0).unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_final.scale(&q_arith_eval).unwrap();

    let lookup_1 = prover_key.lookup.compute_linearisation(
        l1_eval,
        a_eval,
        b_eval,
        c_eval,
        d_eval,
        f_eval,
        table_eval,
        table_next_eval,
        h1_next_eval,
        h2_eval,
        z2_next_eval,
        *delta,
        *epsilon,
        *zeta,
        z2_poly,
        h1_poly,
        *lookup_separation_challenge,
    );
    let ep_delta_one = *epsilon * (F::one() + delta);
    let lookup_sp_sq = lookup_separation_challenge.square();
    let b_1 = ep_delta_one
        * (ep_delta_one + table_eval + *delta * table_next_eval)
        * lookup_sp_sq;
    let b_2 = l1_eval * lookup_sp_sq * lookup_separation_challenge;
    let lookup = b_1 + b_2;

    // permutation
    // A = (a_eval + beta * z_challenge + gamma)(b_eval + beta * K1 *
    // z_challenge + gamma)(c_eval + beta * K2 * z_challenge + gamma)(d_eval
    // + beta * K3 * z_challenge + gamma) * alpha z(X) + l1_z * alpha^2 * z(X)
    let mut part1_val = a_eval + *beta * z_challenge + gamma;
    let mut tmp_part2_val = b_eval
        + *beta * crate::permutation::constants::K1::<F>() * z_challenge
        + gamma;
    let mut tmp_part3_val = c_eval
        + *beta * crate::permutation::constants::K2::<F>() * z_challenge
        + gamma;
    let tmp_part4_val = d_eval
        + *beta * crate::permutation::constants::K3::<F>() * z_challenge
        + gamma;
    tmp_part2_val *= tmp_part3_val * tmp_part4_val;
    part1_val *= tmp_part2_val;
    part1_val *= alpha;
    let mut l_1_z = domain.evaluate_all_lagrange_coefficients(*z_challenge)[0];
    l_1_z *= alpha.square();
    part1_val += l_1_z;
    gpu_res_tmp.copy_from_gpu(&z_poly).unwrap();
    gpu_res_tmp.scale(&part1_val).unwrap();
    // B =  -(a_eval + beta * sigma_1 + gamma)(b_eval + beta * sigma_2 + gamma)
    // (c_eval + beta * sigma_3 + gamma) * beta *z_eval * alpha * Sigma_4(X)
    part1_val = a_eval + *beta * left_sigma_eval + gamma;
    tmp_part2_val = b_eval + *beta * right_sigma_eval + gamma;
    tmp_part3_val = c_eval + *beta * out_sigma_eval + gamma;
    tmp_part2_val *= tmp_part3_val * *beta * permutation_eval * alpha;
    part1_val *= tmp_part2_val;
    let mut gpu_res_tmp2 = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_tmp2.fill_with_zero().unwrap();
    gpu_res_tmp2
        .read_from(&prover_key.permutation.fourth_sigma.0)
        .unwrap();
    gpu_res_tmp2.scale(&part1_val).unwrap();
    gpu_res_tmp.sub_assign(&gpu_res_tmp2).unwrap();

    // gate + permutation
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();

    // quotient part
    gpu_res_tmp.fill_with_fe(&F::zero()).unwrap();
    gpu_res_tmp2.fill_with_fe(&F::zero()).unwrap();
    // gpu_res_tmp2.read_from(&t_6_poly).unwrap();

    gpu_res_tmp.add_assign_gpu_ptr(t_6_poly, n).unwrap();
    gpu_res_tmp.scale(&z_challenge_to_n).unwrap();

    // gpu_res_tmp2.read_from(&t_5_poly).unwrap();
    gpu_res_tmp.add_assign_gpu_ptr(t_5_poly, n).unwrap();
    gpu_res_tmp.scale(&z_challenge_to_n).unwrap();
    // gpu_res_tmp2.read_from(&t_4_poly).unwrap();
    gpu_res_tmp.add_assign_gpu_ptr(t_4_poly, n).unwrap();
    gpu_res_tmp.scale(&z_challenge_to_n).unwrap();
    // gpu_res_tmp2.read_from(&t_3_poly).unwrap();
    gpu_res_tmp.add_assign_gpu_ptr(t_3_poly, n).unwrap();
    gpu_res_tmp.scale(&z_challenge_to_n).unwrap();
    // gpu_res_tmp2.read_from(&t_2_poly).unwrap();
    gpu_res_tmp.add_assign_gpu_ptr(t_2_poly, n).unwrap();
    gpu_res_tmp.scale(&z_challenge_to_n).unwrap();
    // gpu_res_tmp2.read_from(&t_1_poly).unwrap();
    gpu_res_tmp.add_assign_gpu_ptr(t_1_poly, n).unwrap();
    gpu_res_tmp
        .scale(&(vanishing_poly_eval * (-F::one())))
        .unwrap();
    gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
    gpu_res_final.add_at_offset(&lookup, 0).unwrap();

    //let linearisation_polynomial =
    //    gate_constraints + permutation + lookup + quotient_term;
    let mut powers = gpu_container.ask_for_powers(kern).unwrap();
    let lin_eval = gpu_res_final
        .evaluate_at(&mut powers, &mut gpu_res_tmp, z_challenge)
        .unwrap();

    gpu_container.save("lin_poly", gpu_res_final).unwrap();
    gpu_container.recycle(gpu_res_tmp).unwrap();
    gpu_container.recycle(powers).unwrap();
    gpu_container.recycle(gpu_res_tmp2).unwrap();

    (
        lin_eval,
        ProofEvaluations {
            wire_evals,
            perm_evals,
            lookup_evals,
            custom_evals,
        },
    )
}

pub fn open_gpu<'a, 'b, 'c, G, F, PC>(
    kern: &PolyKernel<GpuFr>,
    gpu_container: &mut GpuPolyContainer<GpuFr>,
    ck: &PC::CommitterKey,
    poly_labels: &Vec<String>,
    poly_evals: &Vec<F>,
    point: F,
    opening_challenge: F,
    size_inv: &F,
    n: usize,
    msm_context: &mut MSMContext<'b, 'c, G>,
    recycle: bool,
) -> GPUResult<(PC::Proof)>
where
    G: AffineCurve,
    F: PrimeField,
    'c: 'a,
    'b: 'a,
    PC: HomomorphicCommitment<F>,
{
    // cal label poly
    let lgn = log2_floor(n);
    let mut gpu_res_tmp = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_tmp.fill_with_zero().unwrap();

    let mut gpu_res_final = gpu_container.ask_for(kern, n).unwrap();
    gpu_res_final.fill_with_zero().unwrap();

    for (label, eval) in poly_labels.iter().rev().zip(poly_evals.iter().rev()) {
        gpu_res_final.scale(&opening_challenge).unwrap();
        if label == "f_poly"
            || label == "h_2_poly"
            || label == "table_poly"
            || label == "z_2_poly"
            || label == "h_1_poly"
        {
            // These are all zero
            gpu_res_tmp.fill_with_fe(&F::zero()).unwrap();
        } else {
            let gpu_poly = gpu_container.find(kern, label.as_str()).unwrap();
            gpu_res_tmp.copy_from_gpu(&gpu_poly).unwrap();
            gpu_res_tmp.add_at_offset(eval, 0).unwrap();
            gpu_res_final.add_assign(&gpu_res_tmp).unwrap();
            gpu_container.save(label.as_str(), gpu_poly).unwrap();
        }
    }

    //calc numerator
    let mut tmp_buf = gpu_container
        .find(&kern, &format!("domain_{n}_tmp_buf"))
        .unwrap();
    tmp_buf.fill_with_fe(&F::zero())?;
    let pq_buf = gpu_container
        .find(&kern, &format!("domain_{n}_pq"))
        .unwrap();
    let omegas_buf = gpu_container
        .find(&kern, &format!("domain_{n}_omegas"))
        .unwrap();

    gpu_res_final
        .fft(&mut tmp_buf, &pq_buf, &omegas_buf, lgn)
        .unwrap();

    //calc denumerator
    let mut gpu_result = gpu_container.ask_for_results(kern)?;
    gpu_result.fill_with_zero()?;
    let mut gpu_buckets = gpu_container.ask_for_buckets(kern)?;
    gpu_buckets.fill_with_zero()?;

    let gpu_saved_powers =
        gpu_container.find(kern, &format!("domain_{n}_powers_group_gen"))?;
    gpu_res_tmp.copy_from_gpu(&gpu_saved_powers)?;
    gpu_res_tmp.sub_constant(&point)?;
    gpu_res_tmp.batch_inversion(&mut gpu_result, &mut gpu_buckets)?;

    // calc val
    gpu_res_final.mul_assign(&gpu_res_tmp)?;

    // calc coeffs
    tmp_buf.fill_with_fe(&F::zero())?;
    let ipq_buf = gpu_container.find(kern, &format!("domain_{n}_pq_ifft"))?;
    let iomegas_buf =
        gpu_container.find(kern, &format!("domain_{n}_omegas_ifft"))?;

    gpu_res_final.ifft(&mut tmp_buf, &ipq_buf, &iomegas_buf, size_inv, lgn)?;

    gpu_container
        .save(&format!("domain_{n}_powers_group_gen"), gpu_saved_powers)?;
    gpu_container.save(&format!("domain_{n}_pq"), pq_buf)?;
    gpu_container.save(&format!("domain_{n}_omegas"), omegas_buf)?;
    gpu_container.save(&format!("domain_{n}_pq_ifft"), ipq_buf)?;
    gpu_container.save(&format!("domain_{n}_omegas_ifft"), iomegas_buf)?;
    gpu_container.save(&format!("domain_{n}_tmp_buf"), tmp_buf)?;

    gpu_container.recycle(gpu_buckets)?;
    gpu_container.recycle(gpu_result)?;
    gpu_container.recycle(gpu_res_tmp)?;

    // witness commit
    kern.sync().unwrap();
    let scalar_ptr = gpu_res_final.get_memory().get_inner();

    let proof =
        PC::compute_witness_proof_gpu(ck, scalar_ptr, n, Some(msm_context))
            .unwrap();

    gpu_container.recycle(gpu_res_final)?;

    Ok(proof)
}
