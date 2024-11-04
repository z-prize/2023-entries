/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

// constant indexes


#define ALPHA               0
#define BETA                1
#define GAMMA               2
#define Z_CHALLENGE         3
#define Z_CHALLENGE_SHIFTED 4
#define AW_CHALLENGE        5
#define SAW_CHALLENGE       6

#define A_EVAL              7
#define B_EVAL              8
#define C_EVAL              9
#define D_EVAL              10
#define A_NEXT_EVAL         11
#define B_NEXT_EVAL         12
#define D_NEXT_EVAL         13

#define W_L_EVAL            7
#define W_R_EVAL            8
#define W_O_EVAL            9
#define W_4_EVAL            10
#define W_L_NEXT_EVAL       11
#define W_R_NEXT_EVAL       12
#define W_4_NEXT_EVAL       13

#define P_L_EVAL            14
#define P_R_EVAL            15
#define P_O_EVAL            16
#define Z_NEXT_EVAL         17

#define Q_ARITH_EVAL        18
#define Q_C_EVAL            19
#define Q_L_EVAL            20
#define Q_R_EVAL            21
#define Q_HL_EVAL           22
#define Q_HR_EVAL           23
#define Q_H4_EVAL           24

#define Z_CHALLENGE_POW     25
#define Z_VANISHING         26
#define W_L_EVAL_POW_5      27
#define W_R_EVAL_POW_5      28
#define W_4_EVAL_POW_5      29
#define A_CONST             30
#define B_CONST             31

#define CONST_COUNT         32

class Context {
  public:
  uint32_t                  sm_count;
  MSM                       msm;
  bls12381_fr*              ntt_roots;
  bls12381_fr*              ntt_inv_roots;
  bls12381_fr*              root_pow_table;
  bls12381_fr*              z_ch_pow_table;
  bls12381_fr*              z_ch_inv_pow_table;
  bls12381_fr*              z_ch_shifted_pow_table;
  bls12381_fr*              z_ch_shifted_inv_pow_table;
  void*                     ck;
  prover_key_arithmetic_t   pka;
  prover_key_permutation_t  pkp;
  bls12381_fr*              bf;
  bls12381_fr*              pp;
  bls12381_fr*              tree;
  bls12381_fr*              constants;                       // proof_variables
  witness_t                 w;
  z_t                       z;
  bls12381_fr*              zero_eval;
  bls12381_fr*              round_output;
  bls12381_fr*              temps;

  Context(const uint64_t* commit_key, prover_key_t prover_key);

  void set_constant(uint32_t challenge_index, const uint64_t* value);
  void set_constant(uint32_t challenge_index, const impl_mont_t& value);
};

Context::Context(const uint64_t* commit_key, prover_key_t prover_key) {
  bls12381_fr* gpu_space;
  int32_t      attr;

  $CUDA(cudaDeviceGetAttribute(&attr, cudaDevAttrMultiProcessorCount, 0));
  sm_count=attr;

  $CUDA(cudaMalloc(&ck, 96ul*POLY_SIZE));
  $CUDA(cudaMemcpy(ck, commit_key, 96ul*POLY_SIZE, cudaMemcpyHostToDevice));
  fp_array_to_montgomery(ck, ck, 2*POLY_SIZE);
  msm.initialize(ck, POLY_SIZE, 8);
  $CUDA(cudaFree(ck));

  uint32_t ROOT_TABLE_COUNT=256*49;

  $CUDA(cudaMalloc((void**)&gpu_space, 32ul*ROOT_COUNT*2 + 32ul*ROOT_TABLE_COUNT*5));
  ntt_roots=gpu_space + 0*ROOT_COUNT;
  ntt_inv_roots=gpu_space + 1*ROOT_COUNT;
  gpu_space=gpu_space + 2*ROOT_COUNT;

  initialize_ntt_roots(ntt_roots, ntt_inv_roots);

  root_pow_table=gpu_space + 0*ROOT_TABLE_COUNT;
  z_ch_pow_table=gpu_space + 1*ROOT_TABLE_COUNT;
  z_ch_inv_pow_table=gpu_space + 2*ROOT_TABLE_COUNT;
  z_ch_shifted_pow_table=gpu_space + 3*ROOT_TABLE_COUNT;
  z_ch_shifted_inv_pow_table=gpu_space + 4*ROOT_TABLE_COUNT;

  generate_power_table(root_pow_table, ntt_roots + ROOTS_25_LOW_13 + 1);

  $CUDA(cudaMalloc((void**)&temps, 33ul*EVAL_SIZE));   // YES, 33ul
  $CUDA(cudaMalloc((void**)&zero_eval, 32ul*EVAL_SIZE));
  $CUDA(cudaMemset(zero_eval, 0, 32ul*EVAL_SIZE));

  $CUDA(cudaMalloc((void**)&gpu_space, 32ul*9ul*POLY_SIZE + 32u*9ul*EVAL_SIZE));
  pka.q_poly_l=gpu_space + 0*POLY_SIZE;
  pka.q_poly_r=gpu_space + 1*POLY_SIZE;
  pka.q_poly_o=gpu_space + 2*POLY_SIZE;
  pka.q_poly_4=gpu_space + 3*POLY_SIZE;
  pka.q_poly_hl=gpu_space + 4*POLY_SIZE;
  pka.q_poly_hr=gpu_space + 5*POLY_SIZE;
  pka.q_poly_h4=gpu_space + 6*POLY_SIZE;
  pka.q_poly_c=gpu_space + 7*POLY_SIZE;
  pka.q_poly_arith=gpu_space + 8*POLY_SIZE;
  gpu_space=gpu_space + 9*POLY_SIZE;

  pka.q_eval_l=gpu_space + 0*EVAL_SIZE;
  pka.q_eval_r=gpu_space + 1*EVAL_SIZE;
  pka.q_eval_o=gpu_space + 2*EVAL_SIZE;
  pka.q_eval_4=gpu_space + 3*EVAL_SIZE;
  pka.q_eval_hl=gpu_space + 4*EVAL_SIZE;
  pka.q_eval_hr=gpu_space + 5*EVAL_SIZE;
  pka.q_eval_h4=gpu_space + 6*EVAL_SIZE;
  pka.q_eval_c=gpu_space + 7*EVAL_SIZE;
  pka.q_eval_arith=gpu_space + 8*EVAL_SIZE;

  $CUDA(cudaMemcpy(pka.q_poly_l, prover_key.q_l_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_r, prover_key.q_r_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_o, prover_key.q_o_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_4, prover_key.q_4_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_hl, prover_key.q_hl_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_hr, prover_key.q_hr_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_h4, prover_key.q_h4_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_c, prover_key.q_c_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_poly_arith, prover_key.q_arith_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));

  $CUDA(cudaMemcpy(pka.q_eval_l, prover_key.q_l_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_r, prover_key.q_r_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_o, prover_key.q_o_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_4, prover_key.q_4_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_hl, prover_key.q_hl_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_hr, prover_key.q_hr_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_h4, prover_key.q_h4_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_c, prover_key.q_c_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pka.q_eval_arith, prover_key.q_arith_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));

  array_to_montgomery(pka.q_poly_l, pka.q_poly_l, POLY_SIZE);
  array_to_montgomery(pka.q_poly_r, pka.q_poly_r, POLY_SIZE);
  array_to_montgomery(pka.q_poly_o, pka.q_poly_o, POLY_SIZE);
  array_to_montgomery(pka.q_poly_4, pka.q_poly_4, POLY_SIZE);
  array_to_montgomery(pka.q_poly_hl, pka.q_poly_hl, POLY_SIZE);
  array_to_montgomery(pka.q_poly_hr, pka.q_poly_hr, POLY_SIZE);
  array_to_montgomery(pka.q_poly_h4, pka.q_poly_h4, POLY_SIZE);
  array_to_montgomery(pka.q_poly_c, pka.q_poly_c, POLY_SIZE);
  array_to_montgomery(pka.q_poly_arith, pka.q_poly_arith, POLY_SIZE);

  array_to_montgomery(pka.q_eval_l, pka.q_eval_l, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_r, pka.q_eval_r, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_o, pka.q_eval_o, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_4, pka.q_eval_4, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_hl, pka.q_eval_hl, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_hr, pka.q_eval_hr, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_h4, pka.q_eval_h4, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_c, pka.q_eval_c, EVAL_SIZE);
  array_to_montgomery(pka.q_eval_arith, pka.q_eval_arith, EVAL_SIZE);

  $CUDA(cudaMalloc((void**)&gpu_space, 32ul*8ul*POLY_SIZE + 32u*7ul*EVAL_SIZE));
  pkp.p_scalar_l=gpu_space + 0*POLY_SIZE;
  pkp.p_scalar_r=gpu_space + 1*POLY_SIZE;
  pkp.p_scalar_4=gpu_space + 2*POLY_SIZE;
  pkp.p_scalar_o=gpu_space + 3*POLY_SIZE;

  pkp.p_poly_l=gpu_space + 4*POLY_SIZE;
  pkp.p_poly_r=gpu_space + 5*POLY_SIZE;
  pkp.p_poly_4=gpu_space + 6*POLY_SIZE;
  pkp.p_poly_o=gpu_space + 7*POLY_SIZE;
  gpu_space=gpu_space + 8*POLY_SIZE;

  pkp.p_eval_l=gpu_space + 0*EVAL_SIZE;
  pkp.p_eval_r=gpu_space + 1*EVAL_SIZE;
  pkp.p_eval_4=gpu_space + 2*EVAL_SIZE;
  pkp.p_eval_o=gpu_space + 3*EVAL_SIZE;

  pkp.l1_eval=gpu_space + 4*EVAL_SIZE;
  pkp.lin_eval=gpu_space + 5*EVAL_SIZE;
  pkp.v_h_eval_inv=gpu_space + 6*EVAL_SIZE;

  $CUDA(cudaMemcpy(pkp.p_poly_l, prover_key.left_sigma_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pkp.p_poly_r, prover_key.right_sigma_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pkp.p_poly_o, prover_key.out_sigma_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pkp.p_poly_4, prover_key.fourth_sigma_poly, 32ul*POLY_SIZE, cudaMemcpyHostToDevice));

  $CUDA(cudaMemcpy(pkp.p_eval_l, prover_key.left_sigma_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pkp.p_eval_r, prover_key.right_sigma_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pkp.p_eval_o, prover_key.out_sigma_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pkp.p_eval_4, prover_key.fourth_sigma_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));

  $CUDA(cudaMemcpy(pkp.lin_eval, prover_key.linear_evals, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pkp.v_h_eval_inv, prover_key.v_h_evals_inv, 32ul*EVAL_SIZE, cudaMemcpyHostToDevice));

  array_to_montgomery(pkp.p_poly_l, pkp.p_poly_l, POLY_SIZE);
  array_to_montgomery(pkp.p_poly_r, pkp.p_poly_r, POLY_SIZE);
  array_to_montgomery(pkp.p_poly_o, pkp.p_poly_o, POLY_SIZE);
  array_to_montgomery(pkp.p_poly_4, pkp.p_poly_4, POLY_SIZE);

  array_to_montgomery(pkp.p_eval_l, pkp.p_eval_l, EVAL_SIZE);
  array_to_montgomery(pkp.p_eval_r, pkp.p_eval_r, EVAL_SIZE);
  array_to_montgomery(pkp.p_eval_o, pkp.p_eval_o, EVAL_SIZE);
  array_to_montgomery(pkp.p_eval_4, pkp.p_eval_4, EVAL_SIZE);

  array_to_montgomery(pkp.lin_eval, pkp.lin_eval, EVAL_SIZE);
  array_to_montgomery(pkp.v_h_eval_inv, pkp.v_h_eval_inv, EVAL_SIZE);

  array_set_l1(zero_eval, POLY_SIZE);
  coset_25(pkp.l1_eval, pkp.l1_eval, temps, zero_eval, ntt_roots);
  ntt_22(pkp.p_scalar_l, pkp.p_scalar_l, temps, pkp.p_poly_l, ntt_roots);
  ntt_22(pkp.p_scalar_r, pkp.p_scalar_r, temps, pkp.p_poly_r, ntt_roots);
  ntt_22(pkp.p_scalar_o, pkp.p_scalar_o, temps, pkp.p_poly_o, ntt_roots);
  ntt_22(pkp.p_scalar_4, pkp.p_scalar_4, temps, pkp.p_poly_4, ntt_roots);

  // copy the prover key
  $CUDA(cudaMalloc((void**)&bf, 8*32));
  $CUDA(cudaMalloc((void**)&pp, 198*32));
  $CUDA(cudaMemcpy(pp, PoseidonMDS::POSEIDON_MDS, 9*32, cudaMemcpyHostToDevice));
  $CUDA(cudaMemcpy(pp+9, PoseidonRC::POSEIDON_RC, 189*32, cudaMemcpyHostToDevice));
  $CUDA(cudaMalloc(&constants, 32ul*CONST_COUNT));
  $CUDA(cudaMalloc(&tree, sizeof(merkle_node_t)*N_NON_LEAF_NODES));

  // Allocate space for the witness
printf("bytes: %ld kb\n", (32ul*10ul*POLY_SIZE + 32ul*5ul*EVAL_SIZE)/1024);
  $CUDA(cudaMalloc((void**)&gpu_space, 32ul*5ul*POLY_SIZE + 32ul*10ul*EVAL_SIZE));
  $CUDA(cudaMemset(gpu_space, 0, 32ul*5ul*POLY_SIZE + 32ul*5ul*EVAL_SIZE));     // zero out the scalars & polys

  w.pi_scalar=gpu_space + 0*POLY_SIZE;
  w.w_scalar_l=gpu_space + 1*POLY_SIZE;
  w.w_scalar_r=gpu_space + 2*POLY_SIZE;
  w.w_scalar_4=gpu_space + 3*POLY_SIZE;
  w.w_scalar_o=gpu_space + 4*POLY_SIZE;
  gpu_space=gpu_space + 5*POLY_SIZE;

  w.pi_poly=gpu_space + 0*EVAL_SIZE;
  w.w_poly_l=gpu_space + 1*EVAL_SIZE;
  w.w_poly_r=gpu_space + 2*EVAL_SIZE;
  w.w_poly_4=gpu_space + 3*EVAL_SIZE;
  w.w_poly_o=gpu_space + 4*EVAL_SIZE;
  gpu_space=gpu_space + 5*EVAL_SIZE;

  w.pi_eval=gpu_space + 0*EVAL_SIZE;
  w.w_eval_l=gpu_space + 1*EVAL_SIZE;
  w.w_eval_r=gpu_space + 2*EVAL_SIZE;
  w.w_eval_4=gpu_space + 3*EVAL_SIZE;
  w.w_eval_o=gpu_space + 4*EVAL_SIZE;

  $CUDA(cudaMalloc((void**)&gpu_space, 32ul*4ul*POLY_SIZE + 32ul*EVAL_SIZE));
  z.z_num=gpu_space + 0*POLY_SIZE;
  z.z_denom=gpu_space + 1*POLY_SIZE;
  z.z_scalar=gpu_space + 2*POLY_SIZE;
  z.z_poly=gpu_space + 3*POLY_SIZE;
  z.z_alpha_eval=gpu_space + 4*POLY_SIZE;

  $CUDA(cudaMalloc((void**)&round_output, 32ul*EVAL_SIZE));

//  $CUDA(cudaMalloc((void**)&temp_eval, 32ul*EVAL_SIZE));
}

void Context::set_constant(uint32_t constant_index, const uint64_t* value) {
  st_t        value_st={value[0], value[1], value[2], value[3]};
  st_t        r_st={0x00000001FFFFFFFEull, 0x5884B7FA00034802ull, 0x998C4FEFECBC4FF5ull, 0x1824B159ACC5056Full};
  impl_mont_t r_impl=(impl_mont_t)r_st;

  value_st=value_st.mod_mul(r_impl);
  $CUDA(cudaMemcpy(constants + constant_index, &value_st, 32ul, cudaMemcpyHostToDevice));
}

void Context::set_constant(uint32_t constant_index, const impl_mont_t& value) {
  st_t        st_value=(st_t)value;
  st_t        r_st={0x00000001FFFFFFFEull, 0x5884B7FA00034802ull, 0x998C4FEFECBC4FF5ull, 0x1824B159ACC5056Full};
  impl_mont_t r_impl=(impl_mont_t)r_st;

  // man, this is horrible.  Need to build the converters.
  st_value=st_value.mod_mul(r_impl);
  $CUDA(cudaMemcpy(constants + constant_index, &st_value, 32ul, cudaMemcpyHostToDevice));
}