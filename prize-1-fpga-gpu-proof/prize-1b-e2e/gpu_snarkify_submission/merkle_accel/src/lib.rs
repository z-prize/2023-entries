/***

Copyright (c) 2024, Ingonyama, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

use std::ffi::c_void;

#[repr(C)]
pub struct ProverKey {
    pub q_l_poly: *const u64,
    pub q_r_poly: *const u64,
    pub q_o_poly: *const u64,
    pub q_4_poly: *const u64,
    pub q_hl_poly: *const u64,
    pub q_hr_poly: *const u64,
    pub q_h4_poly: *const u64,
    pub q_c_poly: *const u64,
    pub q_arith_poly: *const u64,

    pub q_l_evals: *const u64,
    pub q_r_evals: *const u64,
    pub q_o_evals: *const u64,
    pub q_4_evals: *const u64,
    pub q_hl_evals: *const u64,
    pub q_hr_evals: *const u64,
    pub q_h4_evals: *const u64,
    pub q_c_evals: *const u64,
    pub q_arith_evals: *const u64,

    pub left_sigma_poly: *const u64,
    pub right_sigma_poly: *const u64,
    pub out_sigma_poly: *const u64,
    pub fourth_sigma_poly: *const u64,

    pub left_sigma_evals: *const u64,
    pub right_sigma_evals: *const u64,
    pub out_sigma_evals: *const u64,
    pub fourth_sigma_evals: *const u64,

    pub linear_evals: *const u64,
    pub v_h_evals_inv: *const u64,
}

#[repr(C)]
pub struct Round1Res {
    pub w_l_commit: [u64; 6],
    pub w_r_commit: [u64; 6],
    pub w_o_commit: [u64; 6],
    pub w_4_commit: [u64; 6],
}

#[repr(C)]
pub struct Round3Res {
    pub z_commit: [u64; 6],
}

#[repr(C)]
pub struct Round4ARes {
    pub t_0_commit: [u64; 6],
    pub t_1_commit: [u64; 6],
    pub t_2_commit: [u64; 6],
    pub t_3_commit: [u64; 6],
    pub t_4_commit: [u64; 6],
    pub t_5_commit: [u64; 6],
    pub t_6_commit: [u64; 6],
    pub t_7_commit: [u64; 6],
}

#[repr(C)]
pub struct Round4BRes {
    pub a_eval: [u64; 4],
    pub b_eval: [u64; 4],
    pub c_eval: [u64; 4],
    pub d_eval: [u64; 4],
    pub a_next_eval: [u64; 4],
    pub b_next_eval: [u64; 4],
    pub d_next_eval: [u64; 4],
    pub left_sigma_eval: [u64; 4],
    pub right_sigma_eval: [u64; 4],
    pub out_sigma_eval: [u64; 4],
    pub permutation_eval: [u64; 4],
    pub q_arith_eval: [u64; 4],
    pub q_c_eval: [u64; 4],
    pub q_l_eval: [u64; 4],
    pub q_r_eval: [u64; 4],
    pub q_hl_eval: [u64; 4],
    pub q_hr_eval: [u64; 4],
    pub q_h4_eval: [u64; 4],
}

#[repr(C)]
pub struct Round5Res {
    pub aw_opening: [u64; 6],
    pub saw_opening: [u64; 6],
}

extern "C" {
    fn ffi_init(commit_key: *const u64, prover_key: ProverKey) -> *mut c_void;
    fn ffi_round_0(ctx: *mut c_void, blinding_factors: *const u64, non_leaf_nodes: *const u64, leaf_nodes: *const u64);
    fn ffi_round_1(ctx: *mut c_void) -> Round1Res;
    fn ffi_round_3(ctx: *mut c_void, beta: *const u64, gamma: *const u64) -> Round3Res;
    fn ffi_round_4a(ctx: *mut c_void, alpha: *const u64, beta: *const u64, gamma: *const u64) -> Round4ARes;
    fn ffi_round_4b(ctx: *mut c_void, alpha: *const u64, beta: *const u64, gamma: *const u64, z_challenge: *const u64) -> Round4BRes;
    fn ffi_round_5(ctx: *mut c_void, z_challenge: *const u64, aw_challenge: *const u64, saw_challenge: *const u64) -> Round5Res;
}

pub fn init_ctx(commit_key: *const u64, prover_key: ProverKey) -> *mut c_void {
    unsafe { ffi_init(commit_key, prover_key) }
}

pub fn round_0(ctx: *mut c_void, blinding_factors: *const u64, non_leaf_nodes: *const u64, leaf_nodes: *const u64) {
    unsafe { ffi_round_0(ctx, blinding_factors, non_leaf_nodes, leaf_nodes) }
}

pub fn round_1(ctx: *mut c_void) -> Round1Res {
    unsafe { ffi_round_1(ctx) }
}

pub fn round_3(ctx: *mut c_void, beta: *const u64, gamma: *const u64) -> Round3Res {
    unsafe { ffi_round_3(ctx, beta, gamma) }
}

pub fn round_4a(ctx: *mut c_void, alpha: *const u64, beta: *const u64, gamma: *const u64) -> Round4ARes {
    unsafe { ffi_round_4a(ctx, alpha, beta, gamma) }
}

pub fn round_4b(ctx: *mut c_void, alpha: *const u64, beta: *const u64, gamma: *const u64, z_challenge: *const u64) -> Round4BRes {
    unsafe { ffi_round_4b(ctx, alpha, beta, gamma, z_challenge) }
}

pub fn round_5(ctx: *mut c_void, z_challenge: *const u64, aw_challenge: *const u64, saw_challenge: *const u64) -> Round5Res {
    unsafe { ffi_round_5(ctx, z_challenge, aw_challenge, saw_challenge) }
}
