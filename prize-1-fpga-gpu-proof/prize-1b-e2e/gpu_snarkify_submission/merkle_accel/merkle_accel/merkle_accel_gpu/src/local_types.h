/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

typedef storage<ec::xy<ec::bls12381>,false> affine_t;

typedef storage<ff::bls12381_fr,false>      st_t;
typedef impl<ff::bls12381_fr,false>         impl_t;
typedef storage<ff::bls12381_fr,true>       st_mont_t;
typedef impl<ff::bls12381_fr,true>          impl_mont_t;

typedef host::ntt_context<ff::bls12381_fr>  ntt_context_t;
typedef host::msm_context<ec::bls12381>     msm_context_t;
typedef host::poly_context<ff::bls12381_fr> poly_context_t;

typedef struct {
  impl_mont_t alpha;
  impl_mont_t beta;
  impl_mont_t gamma;
  impl_mont_t delta;
  impl_mont_t epsilon;
  impl_mont_t z;
  impl_mont_t aw;
  impl_mont_t saw;
} challenges_t;

typedef struct {
  st_t left;
  st_t right;
  st_t hash;
} merkle_node_t;

typedef impl_mont_t poseidon_state_t[3];

typedef struct {
  bls12381_fr mds[3*3];
  bls12381_fr rc[63*3];
} poseidon_params_t;

// GPU proof structures

typedef struct {
  bls12381_fr* q_poly_l;
  bls12381_fr* q_poly_r;
  bls12381_fr* q_poly_4;
  bls12381_fr* q_poly_o;

  bls12381_fr* q_poly_hl;
  bls12381_fr* q_poly_hr;
  bls12381_fr* q_poly_h4;

  bls12381_fr* q_poly_c;
  bls12381_fr* q_poly_arith;

  bls12381_fr* q_eval_l;
  bls12381_fr* q_eval_r;
  bls12381_fr* q_eval_4;
  bls12381_fr* q_eval_o;

  bls12381_fr* q_eval_hl;
  bls12381_fr* q_eval_hr;
  bls12381_fr* q_eval_h4;

  bls12381_fr* q_eval_c;
  bls12381_fr* q_eval_arith;
} prover_key_arithmetic_t;

typedef struct {
  bls12381_fr* p_scalar_l;
  bls12381_fr* p_scalar_r;
  bls12381_fr* p_scalar_4;
  bls12381_fr* p_scalar_o;

  bls12381_fr* p_poly_l;
  bls12381_fr* p_poly_r;
  bls12381_fr* p_poly_4;
  bls12381_fr* p_poly_o;

  bls12381_fr* p_eval_l;
  bls12381_fr* p_eval_r;
  bls12381_fr* p_eval_4;
  bls12381_fr* p_eval_o;

  // misc key
  bls12381_fr* l1_eval;
  bls12381_fr* lin_eval;
  bls12381_fr* v_h_eval_inv;
} prover_key_permutation_t;

typedef struct {
  // witness structure
  bls12381_fr* pi_scalar;
  bls12381_fr* w_scalar_l;
  bls12381_fr* w_scalar_r;
  bls12381_fr* w_scalar_4;
  bls12381_fr* w_scalar_o;

  bls12381_fr* pi_poly;
  bls12381_fr* w_poly_l;
  bls12381_fr* w_poly_r;
  bls12381_fr* w_poly_4;
  bls12381_fr* w_poly_o;

  bls12381_fr* pi_eval;
  bls12381_fr* w_eval_l;
  bls12381_fr* w_eval_r;
  bls12381_fr* w_eval_4;
  bls12381_fr* w_eval_o;
} witness_t;

typedef struct {
  bls12381_fr* z_num;
  bls12381_fr* z_denom;
  bls12381_fr* z_scalar;
  bls12381_fr* z_poly;
  bls12381_fr* z_alpha_eval;
} z_t;
