/***

Copyright (c) 2024, Snarkify, Inc. and Ingonyama, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Tony Wu

***/

#include <iostream>

#define $CUDA(call) { int localEC=call; if(localEC!=cudaSuccess) { printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, localEC); exit(1); } }

#include "merkle_accel.h"
#include "asm.cu"
#include "Chain.cu"
#include "MP.cu"
#include "BLS12381_fr.cu"
#include "local_types.h"

#include "ScanReduce.cu"
#include "PowerTable.cu"
#include "Polynomial.cu"
#include "NTT/NTT.h"
#include "MSM/MSM.h"
#include "Montgomery.cu"
#include "MathAPI.cu"

storage<ff::bls12381_fp,false> compress_msm_result(storage<ec::xy<ec::bls12381>,false> msm_result) {
  storage<ff::bls12381_fp,false> res;
  uint64_t                       doubled[6];

  res=msm_result.x;
  if(msm_result.x.is_zero() && msm_result.y.is_zero())
    res.l[5] |= 1ull << 62;

  host::mp_packed<ff::bls12381_fp>::add(doubled, msm_result.y.l, msm_result.y.l);
  if (host::mp_packed<ff::bls12381_fp>::compare(doubled, ff::bls12381_fp::const_storage_n.l)==1)
    res.l[5] |= 1ull << 63;
  return res;
}

void fr_store(const char* path, st_t* data, uint32_t count) {
  FILE* file=fopen(path, "w");

  fprintf(stderr, "Writing %s\n", path);
  for(uint32_t i=0;i<count;i++)
    fprintf(file, "%016lX%016lX%016lX%016lX\n", data[i].l[3], data[i].l[2], data[i].l[1], data[i].l[0]);
  fclose(file);
}

#include "Context.h"
#include "Round0.cu"
#include "Round1.cu"
#include "Round3.cu"
#include "Round4a.cu"
#include "Round4b.cu"
#include "Round5.cu"

void *ffi_init(const uint64_t *ffi_commit_key, prover_key_t ffi_prover_key) {
  return new Context(ffi_commit_key, ffi_prover_key);
}

void ffi_round_0(void *ctx, const uint64_t *ffi_blinding_factors, const uint64_t *ffi_non_leaf_nodes,
                 const uint64_t *ffi_leaf_nodes) {
  round0((Context*)ctx, ffi_blinding_factors, ffi_non_leaf_nodes, ffi_leaf_nodes);  
  $CUDA(cudaDeviceSynchronize());
}

round_1_res_t ffi_round_1(void *ctx) {
  return round1((Context*)ctx);
}

round_3_res_t ffi_round_3(void *ctx, const uint64_t *ffi_beta, const uint64_t *ffi_gamma) {
  Context* context=(Context*)ctx;

  context->set_constant(BETA, ffi_beta);
  context->set_constant(GAMMA, ffi_gamma);
  return round3(context);
}

round_4a_res_t ffi_round_4a(void *ctx, const uint64_t *ffi_alpha, const uint64_t *ffi_beta, const uint64_t *ffi_gamma) {
  Context* context=(Context*)ctx;

  context->set_constant(ALPHA, ffi_alpha);
  return round4a(context);
}

round_4b_res_t ffi_round_4b(void *ctx, const uint64_t *ffi_alpha, const uint64_t *ffi_beta, const uint64_t *ffi_gamma,
                            const uint64_t *ffi_z_challenge) {
  Context*    context=(Context*)ctx;
  st_t        z_ch_shifted={ffi_z_challenge[0], ffi_z_challenge[1], ffi_z_challenge[2], ffi_z_challenge[3]};
  st_t        omega_st={0xC6E7A6FFB08E3363ull, 0xE08FEC7C86389BEEull, 0xF2D38F10FBB8D1BBull, 0x0ABE6A5E5ABCAA32ull};                        
  impl_mont_t omega_impl=(impl_mont_t)omega_st;

  z_ch_shifted=z_ch_shifted.mod_mul(omega_impl);

  context->set_constant(Z_CHALLENGE, ffi_z_challenge);
  context->set_constant(Z_CHALLENGE_SHIFTED, z_ch_shifted.l);
  return round4b(context, ffi_z_challenge);
}

round_5_res_t ffi_round_5(void *ctx, const uint64_t *ffi_z_challenge, const uint64_t *ffi_aw_challenge,
                          const uint64_t *ffi_saw_challenge) {
  Context* context=(Context*)ctx;

  context->set_constant(AW_CHALLENGE, ffi_aw_challenge);
  context->set_constant(SAW_CHALLENGE, ffi_saw_challenge);
  return round5(context);
}
