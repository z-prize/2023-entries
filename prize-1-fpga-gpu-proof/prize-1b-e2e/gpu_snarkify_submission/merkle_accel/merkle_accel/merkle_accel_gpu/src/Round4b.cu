/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__global__ void round4b_convert(bls12381_fr* constants_normal, bls12381_fr* constants_montgomery) {
  bls12381_fr value;

  if(threadIdx.x>=CONST_COUNT)
    return;

  value=constants_montgomery[threadIdx.x];
  if((threadIdx.x>=W_L_EVAL && threadIdx.x<=W_4_NEXT_EVAL) || threadIdx.x==Z_NEXT_EVAL)
    value=bls12381_fr::to_montgomery(value);
  constants_montgomery[threadIdx.x]=value;

  constants_normal[threadIdx.x]=bls12381_fr::from_montgomery(value);
}

__launch_bounds__(512)
__global__ void compute_linearization_kernel(bls12381_fr* linearization, bls12381_fr* quotient, bls12381_fr* constants,
                                             prover_key_arithmetic_t pka, prover_key_permutation_t pkp, z_t z, uint32_t count) {
  uint32_t               globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr            q_arith_eval, z_challenge_pow, z_vanishing, a_const, b_const, constant, load, q_coeff, gate_constraint, poly;
  __shared__ bls12381_fr local_constants[CONST_COUNT];

  if(threadIdx.x<CONST_COUNT)
    local_constants[threadIdx.x]=constants[threadIdx.x];

  __syncthreads();

  // Note, quotient and z_poly are in normal space, not Montgomery space.
  // So we multiply z_vanishing and a_const by R, to convert to Montgomery

  z_challenge_pow=local_constants[Z_CHALLENGE_POW];
  z_vanishing=bls12381_fr::to_montgomery(local_constants[Z_VANISHING]);  
  a_const=bls12381_fr::to_montgomery(local_constants[A_CONST]);
  b_const=local_constants[B_CONST];

  for(uint32_t i=globalTID;i<count;i+=globalStride) {
    constant=local_constants[W_L_EVAL];
    load=pka.q_poly_l[i];
    q_coeff=quotient[i + 7*POLY_SIZE];
    gate_constraint=constant*load;    
    poly=q_coeff;

    constant=local_constants[W_R_EVAL];
    load=pka.q_poly_r[i];
    q_coeff=quotient[i + 6*POLY_SIZE];
    gate_constraint=gate_constraint + constant*load;    
    poly=poly*z_challenge_pow + q_coeff;

    constant=local_constants[W_O_EVAL];
    load=pka.q_poly_o[i];
    q_coeff=quotient[i + 5*POLY_SIZE];
    gate_constraint=gate_constraint + constant*load;    
    poly=poly*z_challenge_pow + q_coeff;

    constant=local_constants[W_4_EVAL];
    load=pka.q_poly_4[i];
    q_coeff=quotient[i + 4*POLY_SIZE];
    gate_constraint=gate_constraint + constant*load;    
    poly=poly*z_challenge_pow + q_coeff;

    constant=local_constants[W_L_EVAL_POW_5];
    load=pka.q_poly_hl[i];
    q_coeff=quotient[i + 3*POLY_SIZE];
    gate_constraint=gate_constraint + constant*load;    
    poly=poly*z_challenge_pow + q_coeff;

    constant=local_constants[W_R_EVAL_POW_5];
    load=pka.q_poly_hr[i];
    q_coeff=quotient[i + 2*POLY_SIZE];
    gate_constraint=gate_constraint + constant*load;    
    poly=poly*z_challenge_pow + q_coeff;

    constant=local_constants[W_4_EVAL_POW_5];
    load=pka.q_poly_h4[i];
    q_coeff=quotient[i + 1*POLY_SIZE];
    gate_constraint=gate_constraint + constant*load;    
    poly=poly*z_challenge_pow + q_coeff;

    constant=local_constants[Q_ARITH_EVAL];
    load=pka.q_poly_c[i];
    q_coeff=quotient[i + 0*POLY_SIZE];
    gate_constraint=(gate_constraint + load)*constant;    
    poly=poly*z_challenge_pow + q_coeff;

    linearization[i]=a_const*z.z_poly[i] - b_const*pkp.p_poly_4[i] - z_vanishing*poly + gate_constraint;
  }
}

round_4b_res_t round4b(Context* context, const uint64_t* ffi_z_challenge) {
  uint32_t       sm_count=context->sm_count;
  st_t           st_inv_size_22={0x069003FF00000401ull, 0xCC5D8EB4096E4FFFull, 0xBD44B73B2241B7DEull, 0x73EDA583730030A1ull};
  st_t           st_z_challenge={ffi_z_challenge[0], ffi_z_challenge[1], ffi_z_challenge[2], ffi_z_challenge[3]};
  impl_mont_t    inv_size_22, l1_eval, z_ch, z_ch_power, z_vanishing, one=impl_mont_t::one();
  st_t           local_constants[CONST_COUNT];
  impl_mont_t    alpha, beta, gamma;
  impl_mont_t    w_l_eval, w_r_eval, w_o_eval, w_4_eval, w_l_eval_pow5, w_r_eval_pow5, w_4_eval_pow5;
  impl_mont_t    p_l_eval, p_r_eval, p_o_eval, z_next_eval;
  impl_mont_t    beta_z_ch, three_beta_z_ch, seven_beta_z_ch, thirteen_beta_z_ch, seventeen_beta_z_ch;
  impl_mont_t    a, b;
  round_4b_res_t res;

  z_ch=(impl_mont_t)st_z_challenge;
  z_ch_power=z_ch.pow(1<<22);
  z_vanishing=z_ch_power - one;

  if(z_vanishing.is_zero()) 
    l1_eval=(z_ch==one) ? one : impl_mont_t::zero();
  else {
    inv_size_22=(impl_mont_t)st_inv_size_22;
    l1_eval=z_ch - one;
    l1_eval=l1_eval.inv() * inv_size_22 * z_vanishing; 
  }

  generate_power_table(context->z_ch_pow_table, &context->constants[Z_CHALLENGE]);
  generate_inv_power_table(context->z_ch_inv_pow_table, &context->constants[Z_CHALLENGE]);
  generate_power_table(context->z_ch_shifted_pow_table, &context->constants[Z_CHALLENGE_SHIFTED]);
  generate_inv_power_table(context->z_ch_shifted_inv_pow_table, &context->constants[Z_CHALLENGE_SHIFTED]);

  polynomial_evaluate(&context->constants[W_L_EVAL], context->w.w_poly_l, context->temps, context->z_ch_pow_table, POLY_SIZE); 
  polynomial_evaluate(&context->constants[W_R_EVAL], context->w.w_poly_r, context->temps, context->z_ch_pow_table, POLY_SIZE); 
  polynomial_evaluate(&context->constants[W_O_EVAL], context->w.w_poly_o, context->temps, context->z_ch_pow_table, POLY_SIZE); 
  polynomial_evaluate(&context->constants[W_4_EVAL], context->w.w_poly_4, context->temps, context->z_ch_pow_table, POLY_SIZE); 
  polynomial_evaluate(&context->constants[W_L_NEXT_EVAL], context->w.w_poly_l, context->temps, context->z_ch_shifted_pow_table, POLY_SIZE); 
  polynomial_evaluate(&context->constants[W_R_NEXT_EVAL], context->w.w_poly_r, context->temps, context->z_ch_shifted_pow_table, POLY_SIZE); 
  polynomial_evaluate(&context->constants[W_4_NEXT_EVAL], context->w.w_poly_4, context->temps, context->z_ch_shifted_pow_table, POLY_SIZE); 

  polynomial_evaluate(&context->constants[P_L_EVAL], context->pkp.p_poly_l, context->temps, context->z_ch_pow_table, POLY_SIZE);               // montgomery-ized
  polynomial_evaluate(&context->constants[P_R_EVAL], context->pkp.p_poly_r, context->temps, context->z_ch_pow_table, POLY_SIZE);               // montgomery-ized
  polynomial_evaluate(&context->constants[P_O_EVAL], context->pkp.p_poly_o, context->temps, context->z_ch_pow_table, POLY_SIZE);               // montgomery-ized
  polynomial_evaluate(&context->constants[Z_NEXT_EVAL], context->z.z_poly, context->temps, context->z_ch_shifted_pow_table, POLY_SIZE); 

  polynomial_evaluate(&context->constants[Q_ARITH_EVAL], context->pka.q_poly_arith, context->temps, context->z_ch_pow_table, POLY_SIZE);       // montgomery-ized
  polynomial_evaluate(&context->constants[Q_C_EVAL], context->pka.q_poly_c, context->temps, context->z_ch_pow_table, POLY_SIZE);               // montgomery-ized
  polynomial_evaluate(&context->constants[Q_L_EVAL], context->pka.q_poly_l, context->temps, context->z_ch_pow_table, POLY_SIZE);               // montgomery-ized
  polynomial_evaluate(&context->constants[Q_R_EVAL], context->pka.q_poly_r, context->temps, context->z_ch_pow_table, POLY_SIZE);               // montgomery-ized
  polynomial_evaluate(&context->constants[Q_HL_EVAL], context->pka.q_poly_hl, context->temps, context->z_ch_pow_table, POLY_SIZE);             // montgomery-ized
  polynomial_evaluate(&context->constants[Q_HR_EVAL], context->pka.q_poly_hr, context->temps, context->z_ch_pow_table, POLY_SIZE);             // montgomery-ized
  polynomial_evaluate(&context->constants[Q_H4_EVAL], context->pka.q_poly_h4, context->temps, context->z_ch_pow_table, POLY_SIZE);             // montgomery-ized

  context->set_constant(Z_CHALLENGE_POW, z_ch_power);
  context->set_constant(Z_VANISHING, z_vanishing);

  round4b_convert<<<1, 256>>>(context->temps, context->constants);

  $CUDA(cudaMemcpy(local_constants, context->temps, 32ul*CONST_COUNT, cudaMemcpyDeviceToHost));

  alpha=(impl_mont_t)local_constants[ALPHA];
  beta=(impl_mont_t)local_constants[BETA];
  gamma=(impl_mont_t)local_constants[GAMMA];
  w_l_eval=(impl_mont_t)local_constants[W_L_EVAL];
  w_r_eval=(impl_mont_t)local_constants[W_R_EVAL];
  w_o_eval=(impl_mont_t)local_constants[W_O_EVAL];
  w_4_eval=(impl_mont_t)local_constants[W_4_EVAL];
  p_l_eval=(impl_mont_t)local_constants[P_L_EVAL];
  p_r_eval=(impl_mont_t)local_constants[P_R_EVAL];
  p_o_eval=(impl_mont_t)local_constants[P_O_EVAL];
  z_next_eval=(impl_mont_t)local_constants[Z_NEXT_EVAL];

  w_l_eval_pow5=w_l_eval.sqr().sqr() * w_l_eval;
  w_r_eval_pow5=w_r_eval.sqr().sqr() * w_r_eval;
  w_4_eval_pow5=w_4_eval.sqr().sqr() * w_4_eval;

  beta_z_ch=beta * z_ch;
  three_beta_z_ch=beta_z_ch + beta_z_ch + beta_z_ch;
  seven_beta_z_ch=three_beta_z_ch + three_beta_z_ch + beta_z_ch;
  thirteen_beta_z_ch=seven_beta_z_ch + seven_beta_z_ch - beta_z_ch;
  seventeen_beta_z_ch=thirteen_beta_z_ch + three_beta_z_ch + beta_z_ch;

  a=(w_l_eval + beta_z_ch + gamma) * (w_r_eval + seven_beta_z_ch + gamma) * (w_o_eval + thirteen_beta_z_ch + gamma) * (w_4_eval + seventeen_beta_z_ch + gamma);
  a=a*alpha + l1_eval*alpha*alpha;

  b=(w_l_eval + beta*p_l_eval + gamma) * (w_r_eval + beta*p_r_eval + gamma) * (w_o_eval + beta*p_o_eval + gamma);
  b=b * alpha * beta * z_next_eval;

  context->set_constant(W_L_EVAL_POW_5, w_l_eval_pow5);
  context->set_constant(W_R_EVAL_POW_5, w_r_eval_pow5);
  context->set_constant(W_4_EVAL_POW_5, w_4_eval_pow5);
  context->set_constant(A_CONST, a);
  context->set_constant(B_CONST, b);
  
  compute_linearization_kernel<<<sm_count, 512>>>(context->round_output, context->round_output, context->constants, 
                                                  context->pka, context->pkp, context->z, POLY_SIZE);

  for(int i=0;i<18;i++) 
    ((st_t*)&res)[i]=local_constants[i + W_L_EVAL];

  return res;
}
