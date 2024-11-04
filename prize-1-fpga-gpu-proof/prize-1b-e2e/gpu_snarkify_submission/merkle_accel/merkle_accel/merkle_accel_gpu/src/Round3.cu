/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__launch_bounds__(384)
__global__ void compute_num_denom_kernel(bls12381_fr* num, bls12381_fr* denom, bls12381_fr* challenges, prover_key_permutation_t pkp, witness_t witness, bls12381_fr* root_table, uint32_t count) {
  uint32_t    globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr beta, gamma, advance;
  bls12381_fr one_beta_root, three_beta_root, seven_beta_root, thirteen_beta_root, seventeen_beta_root;
  bls12381_fr n, d, w_load, p_load;

  beta=challenges[BETA];
  gamma=challenges[GAMMA];

  one_beta_root=power(root_table, globalTID<<3, 25) * beta;
  three_beta_root=one_beta_root + one_beta_root + one_beta_root;
  seven_beta_root=three_beta_root + three_beta_root + one_beta_root;
  thirteen_beta_root=seven_beta_root + seven_beta_root - one_beta_root;
  seventeen_beta_root=thirteen_beta_root + three_beta_root + one_beta_root;

  advance=power(root_table, globalStride<<3, 22);

  for(uint32_t i=globalTID;i<count;i+=globalStride) {
    w_load=witness.w_scalar_l[i];
    p_load=pkp.p_scalar_l[i];
    n=(w_load + one_beta_root + gamma);
    d=(w_load + beta*p_load + gamma);
    one_beta_root=one_beta_root * advance;

    w_load=witness.w_scalar_r[i];
    p_load=pkp.p_scalar_r[i];
    n=n * (w_load + seven_beta_root + gamma);
    d=d * (w_load + beta*p_load + gamma);
    seven_beta_root=seven_beta_root * advance;

    w_load=witness.w_scalar_o[i];
    p_load=pkp.p_scalar_o[i];
    n=n * (w_load + thirteen_beta_root + gamma);
    d=d * (w_load + beta*p_load + gamma);
    thirteen_beta_root=thirteen_beta_root * advance;

    w_load=witness.w_scalar_4[i];
    p_load=pkp.p_scalar_4[i];
    n=n * (w_load + seventeen_beta_root + gamma);
    d=d * (w_load + beta*p_load + gamma);
    seventeen_beta_root=seventeen_beta_root * advance;

    num[i]=n;
    denom[i]=d;
  }
}

__global__ void compute_z_kernel(bls12381_fr* out, bls12381_fr* num, bls12381_fr* denom, uint32_t count) {
  uint32_t globalTID=blockIdx.x*blockDim.x+threadIdx.x;

  if(globalTID==0) 
    out[0]=bls12381_fr::r();

  if(globalTID>=count-1)
    return;

  out[globalTID+1]=num[globalTID] * denom[globalTID]; 
}

round_3_res_t round3(Context* context) {
  uint32_t                            sm_count=context->sm_count;
  void*                               gpu_results;
  storage<ec::xy<ec::bls12381>,false> affine_result;
  round_3_res_t                       res;

  compute_num_denom_kernel<<<sm_count, 384>>>(context->z.z_num, context->z.z_denom, context->constants, context->pkp, context->w, context->root_pow_table, POLY_SIZE);
  array_batch_invert(context->z.z_denom, context->z.z_denom, context->temps, POLY_SIZE);
  compute_z_kernel<<<(POLY_SIZE+255)/256, 256>>>(context->z.z_scalar, context->z.z_num, context->z.z_denom, POLY_SIZE);
  scan<prefix_scan_multiply>(context->z.z_scalar, context->z.z_scalar, context->temps, POLY_SIZE);
  intt_22_from_montgomery(context->z.z_poly, context->z.z_poly, context->temps, context->z.z_scalar, context->ntt_inv_roots);

  gpu_results=context->msm.msm(context->z.z_poly, POLY_SIZE);
  $CUDA(cudaMemcpy(&affine_result, gpu_results, sizeof(affine_result), cudaMemcpyDeviceToHost));
  res.z_commit=compress_msm_result(affine_result);

  return res;  
}