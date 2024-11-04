/***

Copyright (c) 2024, Snarkify, Inc. and Ingonyama, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Tony Wu

***/

#pragma once
#include <cstdlib>
#include "core.h"
#include "host.h"
#include "types.h"
#include "poseidon.h"
#include "params.h"

extern "C" {
struct round_1_res_t {
    affine_field_t w_l_commit;
    affine_field_t w_r_commit;
    affine_field_t w_o_commit;
    affine_field_t w_4_commit;
};

struct round_3_res_t {
    affine_field_t z_commit;
};

struct round_4a_res_t {
    affine_field_t t_0_commit;
    affine_field_t t_1_commit;
    affine_field_t t_2_commit;
    affine_field_t t_3_commit;
    affine_field_t t_4_commit;
    affine_field_t t_5_commit;
    affine_field_t t_6_commit;
    affine_field_t t_7_commit;
};

struct round_4b_res_t {
    st_t a_eval;
    st_t b_eval;
    st_t c_eval;
    st_t d_eval;
    st_t a_next_eval;
    st_t b_next_eval;
    st_t d_next_eval;
    st_t left_sigma_eval;
    st_t right_sigma_eval;
    st_t out_sigma_eval;
    st_t permutation_eval;
    st_t q_arith_eval;
    st_t q_c_eval;
    st_t q_l_eval;
    st_t q_r_eval;
    st_t q_hl_eval;
    st_t q_hr_eval;
    st_t q_h4_eval;
};

struct round_5_res_t {
    affine_field_t aw_opening;
    affine_field_t saw_opening;
};

struct prover_key_t {
    st_t *q_l_poly;
    st_t *q_r_poly;
    st_t *q_o_poly;
    st_t *q_4_poly;
    st_t *q_hl_poly;
    st_t *q_hr_poly;
    st_t *q_h4_poly;
    st_t *q_c_poly;
    st_t *q_arith_poly;

    st_t *q_l_evals;
    st_t *q_r_evals;
    st_t *q_o_evals;
    st_t *q_4_evals;
    st_t *q_hl_evals;
    st_t *q_hr_evals;
    st_t *q_h4_evals;
    st_t *q_c_evals;
    st_t *q_arith_evals;

    st_t *left_sigma_poly;
    st_t *right_sigma_poly;
    st_t *out_sigma_poly;
    st_t *fourth_sigma_poly;

    st_t *left_sigma_evals;
    st_t *right_sigma_evals;
    st_t *out_sigma_evals;
    st_t *fourth_sigma_evals;

    st_t *linear_evals;
    st_t *v_h_evals_inv;
};

void *ffi_init(
    const uint64_t *ffi_commit_key,
    prover_key_t ffi_prover_key);

void ffi_round_0(
    void *ctx,
    const uint64_t *ffi_blinding_factors,
    const uint64_t *ffi_non_leaf_nodes,
    const uint64_t *ffi_leaf_nodes
);

round_1_res_t ffi_round_1(void *ctx);

round_3_res_t ffi_round_3(void *ctx, const uint64_t *ffi_beta, const uint64_t *ffi_gamma);

round_4a_res_t ffi_round_4a(
    void *ctx,
    const uint64_t *ffi_alpha,
    const uint64_t *ffi_beta,
    const uint64_t *ffi_gamma
);

round_4b_res_t ffi_round_4b(
    void *ctx,
    const uint64_t *ffi_alpha,
    const uint64_t *ffi_beta,
    const uint64_t *ffi_gamma,
    const uint64_t *ffi_z_challenge
);

round_5_res_t ffi_round_5(
    void *ctx,
    const uint64_t *ffi_z_challenge,
    const uint64_t *ffi_aw_challenge,
    const uint64_t *ffi_saw_challenge
);
}
