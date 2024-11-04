/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__launch_bounds__(512)
__global__ void compute_z_alpha_kernel(bls12381_fr* z_alpha, bls12381_fr* z, bls12381_fr* challenges, uint32_t count) {
  uint32_t    globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr alpha;

  alpha=challenges[ALPHA];

  for(uint32_t i=globalTID;i<count;i+=globalStride) 
    z_alpha[i]=alpha*z[i];
}

__launch_bounds__(256)
__global__ void compute_quotient_kernel(bls12381_fr* quotient, bls12381_fr* challenges, witness_t w, prover_key_arithmetic_t pka, prover_key_permutation_t pkp, z_t z, uint32_t count) {
  uint32_t    globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr alpha, beta, gamma;
  bls12381_fr one_beta_lin_eval, three_beta_lin_eval, seven_beta_lin_eval, thirteen_beta_lin_eval, seventeen_beta_lin_eval;
  bls12381_fr gate_constraint, identity_range, copy_range;
  bls12381_fr wload, qload, qhload, pload, wpow; 
  bls12381_fr q_arith, z_alpha, z_alpha_next, l1_alpha, pi, v_h_inv, q;

  alpha=challenges[ALPHA];
  beta=challenges[BETA];
  gamma=challenges[GAMMA];

  for(uint32_t i=globalTID;i<count;i+=globalStride) {
    wload=w.w_eval_l[i];
    qload=pka.q_eval_l[i];
    qhload=pka.q_eval_hl[i];
    pload=pkp.p_eval_l[i];
    one_beta_lin_eval=beta*pkp.lin_eval[i];
    wpow=wload.sqr().sqr() * wload;

    gate_constraint=wload*qload + wpow*qhload;
    identity_range=wload + one_beta_lin_eval + gamma;
    copy_range=wload + beta*pload + gamma;

    wload=w.w_eval_r[i];
    qload=pka.q_eval_r[i];
    qhload=pka.q_eval_hr[i];
    pload=pkp.p_eval_r[i];
    three_beta_lin_eval=one_beta_lin_eval + one_beta_lin_eval + one_beta_lin_eval;
    seven_beta_lin_eval=three_beta_lin_eval + three_beta_lin_eval + one_beta_lin_eval;
    wpow=wload.sqr().sqr() * wload;

    gate_constraint=gate_constraint + wload*qload + wpow*qhload;
    identity_range=identity_range * (wload + seven_beta_lin_eval + gamma);
    copy_range=copy_range * (wload + beta*pload + gamma);

    wload=w.w_eval_4[i];
    qload=pka.q_eval_4[i];
    qhload=pka.q_eval_h4[i];
    pload=pkp.p_eval_4[i];
    thirteen_beta_lin_eval=seven_beta_lin_eval + seven_beta_lin_eval - one_beta_lin_eval;
    seventeen_beta_lin_eval=thirteen_beta_lin_eval + three_beta_lin_eval + one_beta_lin_eval;
    wpow=wload.sqr().sqr() * wload;

    gate_constraint=gate_constraint + wload*qload + wpow*qhload;
    identity_range=identity_range * (wload + seventeen_beta_lin_eval + gamma);
    copy_range=copy_range * (wload + beta*pload + gamma);

    wload=w.w_eval_o[i];
    qload=pka.q_eval_o[i];
    pload=pkp.p_eval_o[i];

    gate_constraint=gate_constraint + wload*qload + pka.q_eval_c[i];
    identity_range=identity_range * (wload + thirteen_beta_lin_eval + gamma);
    copy_range=copy_range * (wload + beta*pload + gamma);

    q_arith=pka.q_eval_arith[i];
    z_alpha=z.z_alpha_eval[i];
    z_alpha_next=z.z_alpha_eval[i + 8 & count-1];
    l1_alpha=alpha*pkp.l1_eval[i];
    pi=w.pi_eval[i];
    v_h_inv=pkp.v_h_eval_inv[i];

    quotient[i]=(gate_constraint*q_arith + (identity_range + l1_alpha)*z_alpha - copy_range*z_alpha_next - alpha*l1_alpha + pi) * v_h_inv;
  }
}



round_4a_res_t round4a(Context* context) {
  uint32_t                            sm_count=context->sm_count;
  void*                               gpu_results;
  storage<ec::xy<ec::bls12381>,false> affine_results[8];
  round_4a_res_t                      res;

  compute_z_alpha_kernel<<<sm_count, 512>>>(context->zero_eval, context->z.z_poly, context->constants, POLY_SIZE);

  coset_25_to_montgomery(context->z.z_alpha_eval, context->z.z_alpha_eval, context->temps, context->zero_eval, context->ntt_roots);
  coset_25_to_montgomery(context->w.pi_eval, context->w.pi_eval, context->temps, context->w.pi_poly, context->ntt_roots);
  coset_25_to_montgomery(context->w.w_eval_l, context->w.w_eval_l, context->temps, context->w.w_poly_l, context->ntt_roots);
  coset_25_to_montgomery(context->w.w_eval_r, context->w.w_eval_r, context->temps, context->w.w_poly_r, context->ntt_roots);
  coset_25_to_montgomery(context->w.w_eval_4, context->w.w_eval_4, context->temps, context->w.w_poly_4, context->ntt_roots);
  coset_25_to_montgomery(context->w.w_eval_o, context->w.w_eval_o, context->temps, context->w.w_poly_o, context->ntt_roots);

  compute_quotient_kernel<<<sm_count, 256>>>(context->round_output, context->constants, context->w, context->pka, context->pkp, context->z, EVAL_SIZE);

  icoset_25_from_montgomery(context->round_output, context->round_output, context->temps, context->round_output, context->ntt_inv_roots);

  void* scalars[6]={context->round_output + 0*POLY_SIZE, context->round_output + 1*POLY_SIZE, context->round_output + 2*POLY_SIZE,
                    context->round_output + 3*POLY_SIZE, context->round_output + 4*POLY_SIZE, context->round_output + 5*POLY_SIZE};

  gpu_results=context->msm.msm(scalars, POLY_SIZE, 6);
  
  $CUDA(cudaMemcpy(affine_results, gpu_results, sizeof(affine_results), cudaMemcpyDeviceToHost));
  affine_results[6].x=storage<fp<ec::bls12381>,false>::zero();
  affine_results[6].y=storage<fp<ec::bls12381>,false>::zero();
  affine_results[7].x=storage<fp<ec::bls12381>,false>::zero();
  affine_results[7].y=storage<fp<ec::bls12381>,false>::zero();

  res.t_0_commit=compress_msm_result(affine_results[0]);
  res.t_1_commit=compress_msm_result(affine_results[1]);
  res.t_2_commit=compress_msm_result(affine_results[2]);
  res.t_3_commit=compress_msm_result(affine_results[3]);
  res.t_4_commit=compress_msm_result(affine_results[4]);
  res.t_5_commit=compress_msm_result(affine_results[5]);
  res.t_6_commit=compress_msm_result(affine_results[6]);
  res.t_7_commit=compress_msm_result(affine_results[7]);

  return res;  
}
