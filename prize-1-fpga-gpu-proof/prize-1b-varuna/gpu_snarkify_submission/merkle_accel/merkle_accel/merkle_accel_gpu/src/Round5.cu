/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__device__ bls12381_fr pow(bls12381_fr value, uint32_t k) {
  bls12381_fr res=bls12381_fr::r(), sqr=value;

  while(k!=0) {
    if((k & 0x01)!=0)
      res=res * sqr;
    sqr=sqr * sqr;
    k=k>>1;
  }
  return res;
}
 
__launch_bounds__(512)
__global__ void compute_aw_commitment_kernel(bls12381_fr* commitment, bls12381_fr* linearization, bls12381_fr* constants, prover_key_permutation_t pkp, witness_t w, uint32_t count) {
  uint32_t               globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr            challenge, load, sum;

  __shared__ bls12381_fr challenge_pow[16];

  // Note, w_l_poly, w_r_poly, w_o_poly and w_4_poly are in normal space.  So we 
  // multiply the corresponding challenges by R to convert to Montgomery space.

  if(threadIdx.x<16) {
    challenge=pow(constants[AW_CHALLENGE], threadIdx.x);
    if(threadIdx.x>=7 && threadIdx.x<=10) 
      challenge=bls12381_fr::to_montgomery(challenge);
    challenge_pow[threadIdx.x]=challenge;
  }

  __syncthreads();

  for(uint32_t i=globalTID;i<count;i+=globalStride) {
    sum=linearization[i];

    challenge=challenge_pow[1];
    load=pkp.p_poly_l[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[2];
    load=pkp.p_poly_r[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[3];
    load=pkp.p_poly_o[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[7];
    load=w.w_poly_l[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[8];
    load=w.w_poly_r[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[9];
    load=w.w_poly_o[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[10];
    load=w.w_poly_4[i];
    commitment[i]=sum + challenge*load;
  }
}

__launch_bounds__(512)
__global__ void compute_saw_commitment_kernel(bls12381_fr* commitment, bls12381_fr* constants, witness_t w, z_t z, uint32_t count) {
  uint32_t               globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr            challenge, load, sum;

  __shared__ bls12381_fr challenge_pow[16];

  if(threadIdx.x<16) {
    challenge=pow(constants[SAW_CHALLENGE], threadIdx.x);
    challenge_pow[threadIdx.x]=challenge;
  }

  __syncthreads();

  for(uint32_t i=globalTID;i<count;i+=globalStride) {
    sum=z.z_poly[i];

    challenge=challenge_pow[1];
    load=w.w_poly_l[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[2];
    load=w.w_poly_r[i];
    sum=sum + challenge*load;

    challenge=challenge_pow[3];
    load=w.w_poly_4[i];
    sum=sum + challenge*load;

    if(i==0) {
      challenge=challenge_pow[5];
      load=bls12381_fr::one();
      sum=sum + challenge*load;
    }

    // note, all the polynomials are in normal space.  So we convert to Montgomery space here.
    commitment[i]=bls12381_fr::to_montgomery(sum);
  }
}

round_5_res_t round5(Context* context) {
  uint32_t                            sm_count=context->sm_count;
  bls12381_fr*                        round_output_0=context->round_output + 0*POLY_SIZE;
  bls12381_fr*                        round_output_1=context->round_output + 1*POLY_SIZE;
  void*                               gpu_results;
  storage<ec::xy<ec::bls12381>,false> affine_results[2];
  round_5_res_t                       res;

  compute_aw_commitment_kernel<<<sm_count, 512>>>(round_output_0, context->round_output, context->constants, context->pkp, context->w, POLY_SIZE);
  compute_saw_commitment_kernel<<<sm_count, 512>>>(round_output_1, context->constants, context->w, context->z, POLY_SIZE);

  polynomial_divide_from_montgomery(round_output_0, round_output_0, context->temps, context->z_ch_pow_table, context->z_ch_inv_pow_table, POLY_SIZE);
  polynomial_divide_from_montgomery(round_output_1, round_output_1, context->temps, context->z_ch_shifted_pow_table, context->z_ch_shifted_inv_pow_table, POLY_SIZE);

  void* scalars[2]={round_output_0, round_output_1};

  gpu_results=context->msm.msm(scalars, POLY_SIZE, 2);
  
  $CUDA(cudaMemcpy(affine_results, gpu_results, sizeof(affine_results), cudaMemcpyDeviceToHost));
  res.aw_opening=compress_msm_result(affine_results[0]);
  res.saw_opening=compress_msm_result(affine_results[1]);
  return res;
}
