/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

round_1_res_t round1(Context* context) {
  bls12381_fr*                        temp=context->temps;
  void*                               scalars[4]={context->w.w_poly_l, context->w.w_poly_r, context->w.w_poly_o, context->w.w_poly_4};
  void*                               gpu_results;
  storage<ec::xy<ec::bls12381>,false> affine_results[4];
  round_1_res_t                       res;

  intt_22_from_montgomery(context->w.pi_poly, context->w.pi_poly, temp, context->w.pi_scalar, context->ntt_inv_roots);
  intt_22_from_montgomery(context->w.w_poly_l, context->w.w_poly_l, temp, context->w.w_scalar_l, context->ntt_inv_roots);
  intt_22_from_montgomery(context->w.w_poly_r, context->w.w_poly_r, temp, context->w.w_scalar_r, context->ntt_inv_roots);
  intt_22_from_montgomery(context->w.w_poly_o, context->w.w_poly_o, temp, context->w.w_scalar_o, context->ntt_inv_roots);
  intt_22_from_montgomery(context->w.w_poly_4, context->w.w_poly_4, temp, context->w.w_scalar_4, context->ntt_inv_roots);

  gpu_results=context->msm.msm(scalars, POLY_SIZE, 4);
  
  $CUDA(cudaMemcpy(affine_results, gpu_results, sizeof(affine_results), cudaMemcpyDeviceToHost));

  res.w_l_commit=compress_msm_result(affine_results[0]);
  res.w_r_commit=compress_msm_result(affine_results[1]);
  res.w_o_commit=compress_msm_result(affine_results[2]);
  res.w_4_commit=compress_msm_result(affine_results[3]);

  return res;
}
