/***

Copyright (c) 2024, Snarkify, Inc. and Ingonyama, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Tony Wu

***/

#include <iostream>
#include <cstdlib>
#include <cstring>

#include "merkle_accel.h"
#include "local_types.h"
#include "util.h"

// -- [GLOBALS] --------------------------------------------------------------------------------------------------------
static ntt_context_t ntt_context;
static poseidon_params_t params;
static prover_key_t prover_key;
static affine_t *commit_key;
static st_t *blinding;

static st_t *m_l_poly;
static st_t *m_r_poly;
static st_t *m_o_poly;
static st_t *m_4_poly;

static st_t *w_l_scalars;
static st_t *w_r_scalars;
static st_t *w_o_scalars;
static st_t *w_4_scalars;

static st_t *w_l_poly;
static st_t *w_r_poly;
static st_t *w_o_poly;
static st_t *w_4_poly;

static st_t *w_l_evals;
static st_t *w_r_evals;
static st_t *w_o_evals;
static st_t *w_4_evals;

static st_t *public_inputs;
static st_t *public_inputs_evals;

static st_t *z_poly;
static st_t *z_evals;

static st_t *l1_alpha_sq;

static st_t *gate_constraints;
static st_t *gate_constraints_evals;

static st_t *permutation;
static st_t *permutation_evals;

static st_t *quotient_poly;
static st_t *quotient_evals;

static st_t *lin_poly;

// ---------------------------------------------------------------------------------------------------------------------

// -- [HELPER] ---------------------------------------------------------------------------------------------------------
static void coset(st_t *eval_form, st_t *dense_form) {
    const impl_mont_t gen = impl_mont_t::generator();
    impl_mont_t current = impl_mont_t::one();

    for (uint32_t i = 0; i < POLY_SIZE; i++) {
        eval_form[i] = dense_form[i].mod_mul(current);
        current = current * gen;
    }
    for (uint32_t i = POLY_SIZE; i < EVAL_SIZE; i++) {
        eval_form[i] = st_t::zero();
    }

    ntt_context.ntt(eval_form, eval_form, 25);
}

static void icoset(st_t *eval_form, const st_t *dense_form) {
    const impl_mont_t igen = impl_mont_t::generator().inv();
    impl_mont_t current = impl_mont_t::one();

    ntt_context.intt(eval_form, dense_form, 25);

    for (uint32_t i = 0; i < EVAL_SIZE; i++) {
        eval_form[i] = eval_form[i].mod_mul(current);
        current = current * igen;
    }
}

// ---------------------------------------------------------------------------------------------------------------------

// -- [INIT] -----------------------------------------------------------------------------------------------------------
static void load_poseidon_params() {
    for (int i = 0; i < 9; i++) {
        params.mds[i] = st_t{PoseidonMDS::POSEIDON_MDS[i][0],
		             PoseidonMDS::POSEIDON_MDS[i][1],
		             PoseidonMDS::POSEIDON_MDS[i][2],
			     PoseidonMDS::POSEIDON_MDS[i][3]};
    }
    for (int i = 0; i < 3 * 63; i++) {
        params.rc[i] = st_t{PoseidonRC::POSEIDON_RC[i][0],
		            PoseidonRC::POSEIDON_RC[i][1],
			    PoseidonRC::POSEIDON_RC[i][2],
			    PoseidonRC::POSEIDON_RC[i][3]};
    }
}

static void load_prover_key(const prover_key_t &ffi_prover_key) {
    prover_key.q_l_poly = alloc(POLY_SIZE);
    prover_key.q_r_poly = alloc(POLY_SIZE);
    prover_key.q_o_poly = alloc(POLY_SIZE);
    prover_key.q_4_poly = alloc(POLY_SIZE);
    prover_key.q_hl_poly = alloc(POLY_SIZE);
    prover_key.q_hr_poly = alloc(POLY_SIZE);
    prover_key.q_h4_poly = alloc(POLY_SIZE);
    prover_key.q_c_poly = alloc(POLY_SIZE);
    prover_key.q_arith_poly = alloc(POLY_SIZE);

    prover_key.q_l_evals = alloc(EVAL_SIZE);
    prover_key.q_r_evals = alloc(EVAL_SIZE);
    prover_key.q_o_evals = alloc(EVAL_SIZE);
    prover_key.q_4_evals = alloc(EVAL_SIZE);
    prover_key.q_hl_evals = alloc(EVAL_SIZE);
    prover_key.q_hr_evals = alloc(EVAL_SIZE);
    prover_key.q_h4_evals = alloc(EVAL_SIZE);
    prover_key.q_c_evals = alloc(EVAL_SIZE);
    prover_key.q_arith_evals = alloc(EVAL_SIZE);

    prover_key.left_sigma_poly = alloc(POLY_SIZE);
    prover_key.right_sigma_poly = alloc(POLY_SIZE);
    prover_key.out_sigma_poly = alloc(POLY_SIZE);
    prover_key.fourth_sigma_poly = alloc(POLY_SIZE);

    prover_key.left_sigma_evals = alloc(EVAL_SIZE);
    prover_key.right_sigma_evals = alloc(EVAL_SIZE);
    prover_key.out_sigma_evals = alloc(EVAL_SIZE);
    prover_key.fourth_sigma_evals = alloc(EVAL_SIZE);

    prover_key.linear_evals = alloc(EVAL_SIZE);
    prover_key.v_h_evals_inv = alloc(EVAL_SIZE);

    std::memcpy(prover_key.q_l_poly, ffi_prover_key.q_l_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_r_poly, ffi_prover_key.q_r_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_o_poly, ffi_prover_key.q_o_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_4_poly, ffi_prover_key.q_4_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_hl_poly, ffi_prover_key.q_hl_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_hr_poly, ffi_prover_key.q_hr_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_h4_poly, ffi_prover_key.q_h4_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_c_poly, ffi_prover_key.q_c_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_arith_poly, ffi_prover_key.q_arith_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_l_evals, ffi_prover_key.q_l_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_r_evals, ffi_prover_key.q_r_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_o_evals, ffi_prover_key.q_o_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_4_evals, ffi_prover_key.q_4_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_hl_evals, ffi_prover_key.q_hl_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_hr_evals, ffi_prover_key.q_hr_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_h4_evals, ffi_prover_key.q_h4_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_c_evals, ffi_prover_key.q_c_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.q_arith_evals, ffi_prover_key.q_arith_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.left_sigma_poly, ffi_prover_key.left_sigma_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.right_sigma_poly, ffi_prover_key.right_sigma_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.out_sigma_poly, ffi_prover_key.out_sigma_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.fourth_sigma_poly, ffi_prover_key.fourth_sigma_poly, POLY_SIZE * sizeof(st_t));
    std::memcpy(prover_key.left_sigma_evals, ffi_prover_key.left_sigma_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.right_sigma_evals, ffi_prover_key.right_sigma_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.out_sigma_evals, ffi_prover_key.out_sigma_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.fourth_sigma_evals, ffi_prover_key.fourth_sigma_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.linear_evals, ffi_prover_key.linear_evals, EVAL_SIZE * sizeof(st_t));
    std::memcpy(prover_key.v_h_evals_inv, ffi_prover_key.v_h_evals_inv, EVAL_SIZE * sizeof(st_t));
}

void *ffi_init(const uint64_t *ffi_commit_key, prover_key_t ffi_prover_key) {
    std::cout << "ffi_init" << std::endl;
    load_prover_key(ffi_prover_key);
    ntt_context.initialize_root_table(25);

    m_l_poly = alloc(POLY_SIZE);
    m_r_poly = alloc(POLY_SIZE);
    m_o_poly = alloc(POLY_SIZE);
    m_4_poly = alloc(POLY_SIZE);

    ntt_context.ntt(m_l_poly, prover_key.left_sigma_poly, 22);
    ntt_context.ntt(m_r_poly, prover_key.right_sigma_poly, 22);
    ntt_context.ntt(m_o_poly, prover_key.out_sigma_poly, 22);
    ntt_context.ntt(m_4_poly, prover_key.fourth_sigma_poly, 22);

    commit_key = alloc_points(POLY_SIZE);
    for (int i = 0; i < POLY_SIZE; i++) {
        const int idx = i * 12;
        commit_key[i].x = {
            ffi_commit_key[idx + 0], ffi_commit_key[idx + 1],
            ffi_commit_key[idx + 2], ffi_commit_key[idx + 3],
            ffi_commit_key[idx + 4], ffi_commit_key[idx + 5],
        };
        commit_key[i].y = {
            ffi_commit_key[idx + 6], ffi_commit_key[idx + 7],
            ffi_commit_key[idx + 8], ffi_commit_key[idx + 9],
            ffi_commit_key[idx + 10], ffi_commit_key[idx + 11],
        };
    }
    blinding = alloc(8);
    w_l_scalars = alloc(POLY_SIZE);
    w_r_scalars = alloc(POLY_SIZE);
    w_o_scalars = alloc(POLY_SIZE);
    w_4_scalars = alloc(POLY_SIZE);
    w_l_poly = alloc(POLY_SIZE);
    w_r_poly = alloc(POLY_SIZE);
    w_o_poly = alloc(POLY_SIZE);
    w_4_poly = alloc(POLY_SIZE);
    w_l_evals = alloc(EVAL_SIZE);
    w_r_evals = alloc(EVAL_SIZE);
    w_o_evals = alloc(EVAL_SIZE);
    w_4_evals = alloc(EVAL_SIZE);

    public_inputs = alloc(POLY_SIZE);
    public_inputs_evals = alloc(EVAL_SIZE);

    z_poly = alloc(POLY_SIZE);
    z_evals = alloc(EVAL_PLUS_SIZE);
    l1_alpha_sq = alloc(EVAL_SIZE);

    gate_constraints = alloc(POLY_SIZE);
    gate_constraints_evals = alloc(EVAL_SIZE);

    permutation = alloc(POLY_SIZE);
    permutation_evals = alloc(EVAL_SIZE);

    quotient_poly = alloc(EVAL_SIZE);
    quotient_evals = alloc(EVAL_SIZE);

    lin_poly = alloc(POLY_SIZE);

    for (int i = 0; i < POLY_SIZE; i++) {
        public_inputs[i] = st_t::zero();
    }
    load_poseidon_params();

    return nullptr;
}

// ---------------------------------------------------------------------------------------------------------------------

// -- [ROUND_0] --------------------------------------------------------------------------------------------------------
static void load_blinding(const uint64_t *blinding_poly) {
    const auto tmp = reinterpret_cast<const st_t *>(blinding_poly);
    for (int i = 0; i < 8; i++) {
        blinding[i] = tmp[i];
    }
}

static merkle_node_t *load_tree(const uint64_t *non_leaf_nodes, const uint64_t *leaf_nodes) {
    constexpr uint32_t interior_count = MERKLE_NODE_COUNT / 2;
    constexpr uint32_t leaf_count = MERKLE_NODE_COUNT - interior_count;

    st_t *hashes = alloc(MERKLE_NODE_COUNT);
    const auto non_leaf_nodes_ptr = reinterpret_cast<const st_t *>(non_leaf_nodes);
    for (int i = 0; i < interior_count; i++) {
        hashes[i] = non_leaf_nodes_ptr[i];
    }
    const auto leaf_nodes_ptr = reinterpret_cast<const st_t *>(leaf_nodes);
    for (int i = 0; i < leaf_count; i++) {
        hashes[i + interior_count] = leaf_nodes_ptr[i];
    }

    auto *nodes = static_cast<merkle_node_t *>(malloc(sizeof(merkle_node_t) * interior_count));
    // reverse the tree, so the hashes are in the same order as the witness table
    uint32_t from = 0;
    uint32_t current = interior_count;
    for (int level = 0; level < MERKLE_HEIGHT - 1; level++) {
        const uint32_t level_size = 1 << level;
        current = current - level_size;
        uint32_t to = current;
        for (int i = 0; i < level_size; i++) {
            nodes[to].left = hashes[from * 2 + 1];
            nodes[to].right = hashes[from * 2 + 2];
            nodes[to++].hash = hashes[from++];
        }
    }
    free(hashes);
    return nodes;
}

static impl_mont_t pow5(const impl_mont_t &x) {
    return x.sqr().sqr() * x;
}

static void poseidon_rc(poseidon_state_t out, poseidon_state_t in, const poseidon_params_t &params,
                        const uint32_t round) {
    for (int i = 0; i < 3; i++)
        out[i] = in[i] + params.rc[round * 3 + i];
}

static void poseidon_mds(poseidon_state_t out, poseidon_state_t in, const poseidon_params_t &params) {
    impl_mont_t sum[3];

    for (int i = 0; i < 3; i++) {
        sum[i] = impl_mont_t::zero();
        for (int j = 0; j < 3; j++)
            sum[i] = sum[i] + params.mds[i * 3 + j] * in[j];
    }
    for (int i = 0; i < 3; i++)
        out[i] = sum[i];
}

static void append_gate(uint32_t &pos, const st_t &w_l, const st_t &w_r, const st_t &w_4, const st_t &w_o) {
    w_l_scalars[pos] = w_l;
    w_r_scalars[pos] = w_r;
    w_4_scalars[pos] = w_4;
    w_o_scalars[pos++] = w_o;
}

static void append_gate_zeros(uint32_t &pos) {
    w_l_scalars[pos] = st_t::zero();
    w_r_scalars[pos] = st_t::zero();
    w_4_scalars[pos] = st_t::zero();
    w_o_scalars[pos++] = st_t::zero();
}

static void append_gates_hash(uint32_t &pos, const merkle_node_t &node) {
    const st_t zero = st_t::zero();
    constexpr st_t three = {3ull, 0ull, 0ull, 0ull};
    poseidon_state_t current = {three, node.left, node.right}, next;

    poseidon_rc(next, current, params, 0);

    append_gate(pos, current[0], zero, zero, next[0]);
    append_gate(pos, current[1], zero, zero, next[1]);
    append_gate(pos, current[2], zero, zero, next[2]);

    for (int i = 0; i < 63; i++) {
        for (int j = 0; j < 3; j++)
            current[j] = next[j];

        if (i < 4 || i >= 59) {
            next[0] = pow5(next[0]);
            next[1] = pow5(next[1]);
            next[2] = pow5(next[2]);
        } else
            next[0] = pow5(next[0]);

        poseidon_mds(next, next, params);
        if (i < 62)
            poseidon_rc(next, next, params, i + 1);

        append_gate(pos, current[0], current[1], current[2], next[0]);
        append_gate(pos, current[0], current[1], current[2], next[1]);
        append_gate(pos, current[0], current[1], current[2], next[2]);
    }
    append_gate(pos, node.hash, node.hash, zero, zero);
}

void ffi_round_0(void *ctx, const uint64_t *ffi_blinding_factors, const uint64_t *ffi_non_leaf_nodes,
                 const uint64_t *ffi_leaf_nodes) {
    constexpr uint32_t node_count = (1 << MERKLE_HEIGHT - 1) - 1;
    load_blinding(ffi_blinding_factors);
    merkle_node_t *tree = load_tree(ffi_non_leaf_nodes, ffi_leaf_nodes);

    for (int i = 0; i < POLY_SIZE; i++) {
        public_inputs[i] = st_t::zero();
    }

    uint32_t pos = 0;
    append_gate_zeros(pos);
    append_gate(pos, blinding[0], blinding[1], blinding[2], blinding[3]);
    append_gate(pos, blinding[4], blinding[5], blinding[6], blinding[7]);
    append_gate(pos, blinding[4], blinding[5], st_t::zero(), st_t::zero());

    for (int i = 0; i < node_count; i++) {
        append_gates_hash(pos, tree[i]);
    }

    public_inputs[pos] = tree[node_count - 1].hash.mod_neg();

    append_gate(pos, tree[node_count - 1].hash, st_t::zero(), st_t::zero(), st_t::zero());
    free(tree);

    while (pos < POLY_SIZE) {
        append_gate_zeros(pos);
    }

    ntt_context.intt(public_inputs, public_inputs, 22);
}

// ---------------------------------------------------------------------------------------------------------------------
// -- [ROUND_1] --------------------------------------------------------------------------------------------------------
round_1_res_t ffi_round_1(void *ctx) {
    ntt_context.intt(w_l_poly, w_l_scalars, 22);
    ntt_context.intt(w_r_poly, w_r_scalars, 22);
    ntt_context.intt(w_o_poly, w_o_scalars, 22);
    ntt_context.intt(w_4_poly, w_4_scalars, 22);

    st_t *polys[4] = {w_l_poly, w_r_poly, w_o_poly, w_4_poly};
    affine_field_t commits[4];
#pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        msm_context_t msm_context;
        msm_context.zero_buckets();
        msm_context.accumulate_buckets(polys[i], commit_key, POLY_SIZE);
        commits[i] = result_for_point(msm_context.reduce_buckets());
    }

    return {
        commits[0],
        commits[1],
        commits[2],
        commits[3]
    };
}

// ---------------------------------------------------------------------------------------------------------------------

// -- [ROUND_3] --------------------------------------------------------------------------------------------------------
static void compute_z_poly(const impl_mont_t &beta, const impl_mont_t &gamma) {
    const impl_mont_t three_beta = beta + beta + beta;
    const impl_mont_t seven_beta = three_beta + three_beta + beta;
    const impl_mont_t thirteen_beta = seven_beta + seven_beta - beta;
    const impl_mont_t seventeen_beta = thirteen_beta + three_beta + beta;

    impl_mont_t current = impl_mont_t::one();
    for (int i = 0; i < POLY_SIZE; i++) {
        impl_mont_t root = ntt_context.root(i, 22);
        auto wl_i = static_cast<impl_mont_t>(w_l_scalars[i]);
        auto wr_i = static_cast<impl_mont_t>(w_r_scalars[i]);
        auto wo_i = static_cast<impl_mont_t>(w_o_scalars[i]);
        auto w4_i = static_cast<impl_mont_t>(w_4_scalars[i]);
        auto ml_i = static_cast<impl_mont_t>(m_l_poly[i]);
        auto mr_i = static_cast<impl_mont_t>(m_r_poly[i]);
        auto mo_i = static_cast<impl_mont_t>(m_o_poly[i]);
        auto m4_i = static_cast<impl_mont_t>(m_4_poly[i]);

        impl_mont_t num = (wl_i + beta * root + gamma) *
                          (wr_i + seven_beta * root + gamma) *
                          (wo_i + thirteen_beta * root + gamma) *
                          (w4_i + seventeen_beta * root + gamma);
        impl_mont_t denom = (wl_i + beta * ml_i + gamma) *
                            (wr_i + beta * mr_i + gamma) *
                            (wo_i + beta * mo_i + gamma) *
                            (w4_i + beta * m4_i + gamma);
        z_poly[i] = current;
        current = current * num / denom;
    }

    ntt_context.intt(z_poly, z_poly, 22);
}

round_3_res_t ffi_round_3(void *ctx, const uint64_t *ffi_beta, const uint64_t *ffi_gamma) {
    const auto beta = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_beta));
    const auto gamma = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_gamma));

    std::cout << "computing z_poly" << std::endl;
    compute_z_poly(beta, gamma);

    msm_context_t msm_context;
    msm_context.zero_buckets();
    msm_context.accumulate_buckets(z_poly, commit_key, POLY_SIZE);
    const auto z_res = result_for_point(msm_context.reduce_buckets());

    return {z_res};
}

// ---------------------------------------------------------------------------------------------------------------------

// -- [ROUND_4a] -------------------------------------------------------------------------------------------------------
static void compute_l1_alpha_sq_poly(const impl_mont_t &alpha) {
    impl_mont_t term_impl = alpha;
    term_impl = term_impl.sqr() * ntt_context.inverse_size(22);
    const st_t term_st = term_impl;

    for (uint32_t i = 0; i < POLY_SIZE; i++) {
        l1_alpha_sq[i] = term_st;
    }
}

static void compute_gate_constraints() {
    const st_t *a_val = w_l_evals;
    const st_t *b_val = w_r_evals;
    const st_t *c_val = w_o_evals;
    const st_t *d_val = w_4_evals;

    for (uint32_t i = 0; i < EVAL_SIZE; i++) {
        impl_mont_t ai, bi, ci, di;
        impl_mont_t ql_ei, qr_ei, qo_ei, q4_ei, qhl_ei, qhr_ei, qh4_ei, qc_ei, qarith_ei;
        impl_mont_t pi_ei;

        ai = a_val[i];
        bi = b_val[i];
        ci = c_val[i];
        di = d_val[i];

        ql_ei = prover_key.q_l_evals[i];
        qr_ei = prover_key.q_r_evals[i];
        qo_ei = prover_key.q_o_evals[i];
        q4_ei = prover_key.q_4_evals[i];
        qhl_ei = prover_key.q_hl_evals[i];
        qhr_ei = prover_key.q_hr_evals[i];
        qh4_ei = prover_key.q_h4_evals[i];
        qc_ei = prover_key.q_c_evals[i];
        qarith_ei = prover_key.q_arith_evals[i];

        pi_ei = public_inputs_evals[i];

        const impl_mont_t gc_ei = (ai * ql_ei +
                                   bi * qr_ei +
                                   ci * qo_ei +
                                   di * q4_ei +
                                   pow5(ai) * qhl_ei +
                                   pow5(bi) * qhr_ei +
                                   pow5(di) * qh4_ei +
                                   qc_ei) * qarith_ei + pi_ei;

        gate_constraints_evals[i] = gc_ei;
    }
}

static void compute_permutation(const impl_mont_t &alpha, const impl_mont_t &beta, const impl_mont_t &gamma) {
    for (uint32_t i = 0; i < EVAL_SIZE; i++) {
        impl_mont_t identity_range_ei, copy_range_ei, term_ei;
        impl_mont_t wl_ei, wr_ei, wo_ei, w4_ei, z_ei, z_next_ei;
        impl_mont_t left_sigma_ei, right_sigma_ei, out_sigma_ei, fourth_sigma_ei;
        impl_mont_t l1_alpha_sq_ei;
        impl_mont_t linear_evaluations_ei;
        impl_mont_t three, seven, thirteen, seventeen;

        wl_ei = w_l_evals[i];
        wr_ei = w_r_evals[i];
        wo_ei = w_o_evals[i];
        w4_ei = w_4_evals[i];
        z_ei = z_evals[i];
        z_next_ei = z_evals[i + 8];
        l1_alpha_sq_ei = l1_alpha_sq[i];
        linear_evaluations_ei = prover_key.linear_evals[i];
        linear_evaluations_ei = beta * linear_evaluations_ei;

        three = linear_evaluations_ei + linear_evaluations_ei + linear_evaluations_ei;
        seven = three + three + linear_evaluations_ei;
        thirteen = seven + seven - linear_evaluations_ei;
        seventeen = thirteen + three + linear_evaluations_ei;

        identity_range_ei = (wl_ei + linear_evaluations_ei + gamma) *
                            (wr_ei + seven + gamma) *
                            (wo_ei + thirteen + gamma) *
                            (w4_ei + seventeen + gamma) * z_ei * alpha;

        left_sigma_ei = prover_key.left_sigma_evals[i];
        right_sigma_ei = prover_key.right_sigma_evals[i];
        out_sigma_ei = prover_key.out_sigma_evals[i];
        fourth_sigma_ei = prover_key.fourth_sigma_evals[i];

        copy_range_ei = (wl_ei + beta * left_sigma_ei + gamma) *
                        (wr_ei + beta * right_sigma_ei + gamma) *
                        (wo_ei + beta * out_sigma_ei + gamma) *
                        (w4_ei + beta * fourth_sigma_ei + gamma) * z_next_ei * alpha;

        term_ei = z_ei * l1_alpha_sq_ei - l1_alpha_sq_ei;

        permutation_evals[i] = identity_range_ei - copy_range_ei + term_ei;
    }
}

static void compute_quotient() {
    for (uint32_t i = 0; i < EVAL_SIZE; i++) {
        const impl_mont_t g_ei = gate_constraints_evals[i];
        const impl_mont_t p_ei = permutation_evals[i];
        const impl_mont_t d_ei = prover_key.v_h_evals_inv[i]; // was denominator
        const impl_mont_t q_ei = (g_ei + p_ei) * d_ei;
        quotient_evals[i] = q_ei;
    }
}

round_4a_res_t ffi_round_4a(
    void *ctx,
    const uint64_t *ffi_alpha,
    const uint64_t *ffi_beta,
    const uint64_t *ffi_gamma
) {
    const auto alpha = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_alpha));
    const auto beta = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_beta));
    const auto gamma = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_gamma));

    compute_l1_alpha_sq_poly(alpha);

    std::cout << "computing coset NTTs" << std::endl;

    st_t *ntt_src[7] = {public_inputs, l1_alpha_sq, z_poly, w_l_poly, w_r_poly, w_o_poly, w_4_poly};
    st_t *ntt_dst[7] = {public_inputs_evals, l1_alpha_sq, z_evals, w_l_evals, w_r_evals, w_o_evals, w_4_evals};

#pragma omp parallel for
    for (int i = 0; i < 7; i++) {
        coset(ntt_dst[i], ntt_src[i]);
    }

    copy(z_evals + EVAL_SIZE, z_evals, 8);

    compute_gate_constraints();
    compute_permutation(alpha, beta, gamma);
    compute_quotient();
    icoset(quotient_poly, quotient_evals);

    // TODO: Can eliminate last two MSMs
    affine_field_t commits[8];
#pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        msm_context_t msm_context;
        msm_context.zero_buckets();
        msm_context.accumulate_buckets(quotient_poly + i * POLY_SIZE, commit_key, POLY_SIZE);
        commits[i] = result_for_point(msm_context.reduce_buckets());
    }

    return {
        commits[0],
        commits[1],
        commits[2],
        commits[3],
        commits[4],
        commits[5],
        commits[6],
        commits[7]
    };
}

// ---------------------------------------------------------------------------------------------------------------------

// -- [ROUND_4b] -------------------------------------------------------------------------------------------------------
static impl_mont_t evaluate_vanishing_poly(const impl_mont_t &z_ch) {
    constexpr uint32_t dense_size = 1 << 22;

    return z_ch.pow(dense_size) - impl_mont_t::one();
}

static impl_mont_t evaluate_first_lagrange_coefficient(const impl_mont_t &z_ch) {
    const impl_mont_t evp = evaluate_vanishing_poly(z_ch);
    const impl_mont_t one = impl_mont_t::one();

    if (evp.is_zero()) {
        return (z_ch == one) ? one : impl_mont_t::zero();
    } else {
        impl_mont_t res = z_ch - one;
        res = res.inv() * ntt_context.inverse_size(22) * evp;
        return res;
    }
}

static round_4b_res_t compute_linearization(impl_mont_t alpha, impl_mont_t beta, impl_mont_t gamma,
                                            impl_mont_t z_challenge) {
    impl_mont_t omega;
    impl_mont_t z_challenge_pow_n, z_challenge_vanishing, z_challenge_shifted;
    impl_mont_t a_eval, b_eval, c_eval, d_eval;
    impl_mont_t a_next_eval, b_next_eval, d_next_eval;
    impl_mont_t left_sigma_eval, right_sigma_eval, out_sigma_eval;
    impl_mont_t permutation_eval, q_arith_eval, l1_eval, alpha_sq = alpha.sqr();
    impl_mont_t q_c_eval, q_l_eval, q_r_eval, q_hl_eval, q_hr_eval, q_h4_eval;
    impl_mont_t term;

    omega = ntt_context.root(1, 22);

    z_challenge_pow_n = z_challenge.pow(POLY_SIZE);
    z_challenge_vanishing = z_challenge_pow_n - impl_mont_t::one();
    z_challenge_shifted = z_challenge * omega;

    a_eval = poly_context_t::evaluate(w_l_poly, z_challenge, POLY_SIZE);
    b_eval = poly_context_t::evaluate(w_r_poly, z_challenge, POLY_SIZE);
    c_eval = poly_context_t::evaluate(w_o_poly, z_challenge, POLY_SIZE);
    d_eval = poly_context_t::evaluate(w_4_poly, z_challenge, POLY_SIZE);
    a_next_eval = poly_context_t::evaluate(w_l_poly, z_challenge_shifted, POLY_SIZE);
    b_next_eval = poly_context_t::evaluate(w_r_poly, z_challenge_shifted, POLY_SIZE);
    d_next_eval = poly_context_t::evaluate(w_4_poly, z_challenge_shifted, POLY_SIZE);

    q_arith_eval = poly_context_t::evaluate(prover_key.q_arith_poly, z_challenge, POLY_SIZE);

    q_c_eval = poly_context_t::evaluate(prover_key.q_c_poly, z_challenge, POLY_SIZE);
    q_l_eval = poly_context_t::evaluate(prover_key.q_l_poly, z_challenge, POLY_SIZE);
    q_r_eval = poly_context_t::evaluate(prover_key.q_r_poly, z_challenge, POLY_SIZE);
    q_hl_eval = poly_context_t::evaluate(prover_key.q_hl_poly, z_challenge, POLY_SIZE);
    q_hr_eval = poly_context_t::evaluate(prover_key.q_hr_poly, z_challenge, POLY_SIZE);
    q_h4_eval = poly_context_t::evaluate(prover_key.q_h4_poly, z_challenge, POLY_SIZE);

    left_sigma_eval = poly_context_t::evaluate(prover_key.left_sigma_poly, z_challenge, POLY_SIZE);
    right_sigma_eval = poly_context_t::evaluate(prover_key.right_sigma_poly, z_challenge, POLY_SIZE);
    out_sigma_eval = poly_context_t::evaluate(prover_key.out_sigma_poly, z_challenge, POLY_SIZE);

    permutation_eval = poly_context_t::evaluate(z_poly, z_challenge_shifted, POLY_SIZE);

    l1_eval = evaluate_first_lagrange_coefficient(z_challenge);

    for (uint32_t i = 0; i < POLY_SIZE; i++) {
        term = (a_eval * prover_key.q_l_poly[i] +
                b_eval * prover_key.q_r_poly[i] +
                c_eval * prover_key.q_o_poly[i] +
                d_eval * prover_key.q_4_poly[i] +
                pow5(a_eval) * prover_key.q_hl_poly[i] +
                pow5(b_eval) * prover_key.q_hr_poly[i] +
                pow5(d_eval) * prover_key.q_h4_poly[i] +
                prover_key.q_c_poly[i]) * q_arith_eval;
        gate_constraints[i] = term;
    }

    // K1=7, K2=13, K3=17
    impl_mont_t beta_z, three, seven, thirteen, seventeen, a, a0, a1, a2, a3;

    beta_z = beta * z_challenge;
    three = beta_z + beta_z + beta_z;
    seven = three + three + beta_z;
    thirteen = seven + seven - beta_z;
    seventeen = thirteen + three + beta_z;
    a0 = a_eval + beta_z + gamma;
    a1 = b_eval + seven + gamma;
    a2 = c_eval + thirteen + gamma;
    a3 = d_eval + seventeen + gamma;
    a = a0 * a1 * a2 * a3 * alpha;

    impl_mont_t b, b0, b1, b2;

    b0 = a_eval + beta * left_sigma_eval + gamma;
    b1 = b_eval + beta * right_sigma_eval + gamma;
    b2 = c_eval + beta * out_sigma_eval + gamma;
    b = b0 * b1 * b2 * beta * permutation_eval * alpha;

    for (uint32_t i = 0; i < POLY_SIZE; i++) {
        term = a * z_poly[i] - b * prover_key.fourth_sigma_poly[i] + l1_eval * alpha_sq * z_poly[i];
        permutation[i] = term;
    }

    st_t *t0 = quotient_poly + POLY_SIZE * 0;
    st_t *t1 = quotient_poly + POLY_SIZE * 1;
    st_t *t2 = quotient_poly + POLY_SIZE * 2;
    st_t *t3 = quotient_poly + POLY_SIZE * 3;
    st_t *t4 = quotient_poly + POLY_SIZE * 4;
    st_t *t5 = quotient_poly + POLY_SIZE * 5;
    st_t *t6 = quotient_poly + POLY_SIZE * 6;
    st_t *t7 = quotient_poly + POLY_SIZE * 7;
    for (uint32_t i = 0; i < POLY_SIZE; i++) {
        term = t7[i];
        term = z_challenge_pow_n * term + t6[i];
        term = z_challenge_pow_n * term + t5[i];
        term = z_challenge_pow_n * term + t4[i];
        term = z_challenge_pow_n * term + t3[i];
        term = z_challenge_pow_n * term + t2[i];
        term = z_challenge_pow_n * term + t1[i];
        term = z_challenge_pow_n * term + t0[i];
        lin_poly[i] = ((impl_mont_t) gate_constraints[i]) + ((impl_mont_t) permutation[i]) - term *
                      z_challenge_vanishing;
    }

    return {
        a_eval, b_eval, c_eval, d_eval, a_next_eval, b_next_eval, d_next_eval,
        left_sigma_eval, right_sigma_eval, out_sigma_eval,
        permutation_eval, q_arith_eval,
        q_c_eval, q_l_eval, q_r_eval, q_hl_eval, q_hr_eval, q_h4_eval,
    };
}

round_4b_res_t ffi_round_4b(
    void *ctx,
    const uint64_t *ffi_alpha,
    const uint64_t *ffi_beta,
    const uint64_t *ffi_gamma,
    const uint64_t *ffi_z_challenge
) {
    const auto alpha = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_alpha));
    const auto beta = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_beta));
    const auto gamma = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_gamma));
    const auto z_challenge = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_z_challenge));
    const auto res = compute_linearization(alpha, beta, gamma, z_challenge);

    return res;
}

// ---------------------------------------------------------------------------------------------------------------------

// -- [ROUND_5] --------------------------------------------------------------------------------------------------------
round_5_res_t ffi_round_5(
    void *ctx,
    const uint64_t *ffi_z_challenge,
    const uint64_t *ffi_aw_challenge,
    const uint64_t *ffi_saw_challenge
) {
    const auto z_challenge = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_z_challenge));
    const auto aw_challenge = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_aw_challenge));
    const auto saw_challenge = static_cast<impl_mont_t>(*reinterpret_cast<const st_t *>(ffi_saw_challenge));
    impl_mont_t cmt_challenges[11];
    st_t *aw_commitment_poly = alloc(POLY_SIZE);
    st_t *saw_commitment_poly = alloc(POLY_SIZE);
    st_t *aw_witness_poly = alloc(POLY_SIZE);
    st_t *saw_witness_poly = alloc(POLY_SIZE);

    st_t *z_2_poly = alloc(POLY_SIZE);

    z_2_poly[0] = impl_mont_t::one();
    for (int i = 1; i < POLY_SIZE; i++) {
        z_2_poly[i] = st_t::zero();
    }

    commitment_challenges(cmt_challenges, aw_challenge, 11);
    commitment_initialize(aw_commitment_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[0], lin_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[1], prover_key.left_sigma_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[2], prover_key.right_sigma_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[3], prover_key.out_sigma_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[7], w_l_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[8], w_r_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[9], w_o_poly, POLY_SIZE);
    commitment_include(aw_commitment_poly, cmt_challenges[10], w_4_poly, POLY_SIZE);

    commitment_challenges(cmt_challenges, saw_challenge, 6);
    commitment_initialize(saw_commitment_poly, POLY_SIZE);
    commitment_include(saw_commitment_poly, cmt_challenges[0], z_poly, POLY_SIZE);
    commitment_include(saw_commitment_poly, cmt_challenges[1], w_l_poly, POLY_SIZE);
    commitment_include(saw_commitment_poly, cmt_challenges[2], w_r_poly, POLY_SIZE);
    commitment_include(saw_commitment_poly, cmt_challenges[3], w_4_poly, POLY_SIZE);
    commitment_include(saw_commitment_poly, cmt_challenges[5], z_2_poly, POLY_SIZE);

    poly_context_t::divide(aw_witness_poly, aw_commitment_poly, POLY_SIZE, z_challenge);
    poly_context_t::divide(saw_witness_poly, saw_commitment_poly, POLY_SIZE, ntt_context.root(1, 22) * z_challenge);

    msm_context_t msm_context;

    msm_context.zero_buckets();
    msm_context.accumulate_buckets(aw_witness_poly, commit_key, POLY_SIZE);
    const affine_field_t aw_opening = result_for_point(msm_context.reduce_buckets());

    msm_context.zero_buckets();
    msm_context.accumulate_buckets(saw_witness_poly, commit_key, POLY_SIZE);
    const affine_field_t saw_opening = result_for_point(msm_context.reduce_buckets());

    return {aw_opening, saw_opening};
}

// ---------------------------------------------------------------------------------------------------------------------
