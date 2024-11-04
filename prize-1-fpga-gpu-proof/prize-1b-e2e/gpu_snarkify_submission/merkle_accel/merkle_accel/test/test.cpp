/***

Copyright (c) 2024, Ingonyama, Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>

#include "merkle_accel.h"
#include "host_ff_reader.cpp"
#include "local_types.h"
#include "util.h"

round_1_res_t r1_correct = {
    {
        0xb7954866ad64c162, 0x29cd1d083f46a6e4, 0x1cb428418da1640a,
        0x0afad7a4b44c8559, 0x8e7cfcef925c6f7b, 0x8de523e2f09d5603
    },
    {
        0x28f8f4cab1eabfe0, 0x7e67ab262eb76716, 0x3cc1448924bbc9d0,
        0xd2860bc92d70aa72, 0x3e05a6369563c462, 0x98a61e4239edbbcd
    },
    {
        0x9f8c94e35ebb90d4, 0x734cc5836c80595d, 0xab560704406ad15b,
        0x7baf0c3a954b8796, 0x0bc1451d13db384d, 0x0d9a664f5afdc484
    },
    {
        0xaeb409bc3871c16f, 0x2c7e35ba837aa0e9, 0xdde379dd6b883cfe,
        0x6ac39808aefbab21, 0x944502ee6e78fb1b, 0x8b7693d8d19c33a0
    }
};

round_3_res_t r3_correct = {
    {
        0xbecbc5e4ac68efad, 0x27572ddbf916faf3, 0x1bb1f396bdf3c49c,
        0x2f39a50895af81a1, 0x32f40c746b773474, 0x131cf98c7f790db0
    }
};

round_4a_res_t r4a_correct = {
    {
        0x835fcd82f72b46ae, 0xf23469d75d88c538, 0x7c40c4279a163579,
        0xcdea8e9298d60841, 0xbc6165af55dc3e93, 0x946d84c02a8403ae
    },
    {
        0x5acace0a0bc139f4, 0xac10636107198d74, 0x3222e4f5978f2e09,
        0xa82583eb695839d3, 0x7ff9773378864ea8, 0x041ac34a4d34638d
    },
    {
        0x282e1f3f6f749e10, 0x8621289eef95edcc, 0x2edad49d61b70745,
        0xeb1d96b5e2c26a12, 0xadfbefbf2b6c371a, 0x150b5a9ac12cb237
    },
    {
        0x785084b42eee3505, 0xf79cedaa870eae15, 0x63ddd6faec4d810f,
        0x0845f85d714f64cd, 0x152fea2491c58656, 0x06a2191d0069f6a5
    },
    {
        0x11baf48f2d82a870, 0xfd5c995cf5c4cf17, 0x5bb5126411e83fe1,
        0xb7cd7eeb581b2d7f, 0xb5e918b0ec5dd080, 0x0d18951734f5401f
    },
    {
        0x3bc1acd13442affe, 0x60e8df9a22150022, 0x8a41679a3230c7b7,
        0x33dc27e74443fc59, 0xb20e5be92e95862e, 0x0d0d761f6a99feaa
    },
    {
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000, 0x4000000000000000
    },
    {
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000, 0x4000000000000000
    }
};

round_4b_res_t r4b_correct = {
    {0xb1a71724bcac64ba, 0x1c2a68bfe4b7a278, 0xacd858adfbd2b6fc, 0x1b72b3c1b1fe2382},
    {0x91c042ce7e48d2d5, 0x168f6dabf2fdfe89, 0x5925ed8ee6076dc6, 0x53ccf83e3a7b634a},
    {0xdf7566d6bc09c7c4, 0x5298119720dbde92, 0x94a759045c5cf32c, 0x0dba4a0e3ce26548},
    {0x7272b6c4c9fca53b, 0xd05057877dcd75cd, 0xad5927e63301cbe4, 0x0a71f8f91bd92844},
    {0x66d5e6e6112eaef2, 0xf5f677d56b3cad3c, 0xe155681b20793001, 0x18ba49111abcf66a},
    {0x1ed69e9ad2c66aa2, 0x4d6bec18490a7aa5, 0x74e7e5c6eba42bb5, 0x3df7f708cc2fa49a},
    {0x548d0fca194464fe, 0xd26e93446a550385, 0xad1c379c1af9c3bc, 0x0297a21aec934a16},
    {0xa57beca353c45434, 0xcce588749a83d39f, 0xd688db259ce16cea, 0x2edec6d9423e3efb},
    {0xd5b7d9ffe2b84167, 0x736e94e332d0f2a6, 0x9b7dab38734820e3, 0x1af549e206f96a63},
    {0x64c5a5a52234542f, 0x407365adf614b328, 0xc5895b40d54ec767, 0x40ac43da5bf32967},
    {0x700f7cbb049d32dd, 0x0f2421d7bd283990, 0xbd794412b7f87456, 0x636c4e1fb0e3840f},
    {0xcbea99590befa452, 0x7db7ad89c8faedc3, 0x428c3bd738f59f45, 0x454f0c7b8e3c877c},
    {0xb68c45d535e28230, 0x1fcb8568f5f5c90c, 0xa5737e801094587a, 0x6a802ad6eac7d3b5},
    {0xe3afc11e119d81cb, 0x4ce007d6d8b2fb43, 0x60170273cbec5ff8, 0x109a13797a0808b3},
    {0xc71ffbc2219fcb82, 0x5b0934c37bf54758, 0x5991af27c02c5cd8, 0x4a2dfab2ba044231},
    {0x2bbe062e04d40894, 0x110c7fb4acd6a11e, 0x90bf42bc9aa091e3, 0x2394101b6cb9c994},
    {0x0fef83dc2dd7fb0d, 0xc6a0e0f3600ec4ae, 0xf2100f35de507465, 0x03539a80c7f3d0d6},
    {0x7535f683aae52c39, 0xd3ef613c084f70b2, 0xece0332adb854735, 0x6519e9bfa021c20c}
};

round_5_res_t r5_correct = {
    {
        0x4c8295c2fe477070, 0x2f128dce94483676, 0xb1b952824b21e41d,
        0x5c25d8650e652b36, 0x769f70e56c1527ae, 0x869e44a7a681585d
    },
    {
        0xe26c89e674a9e82f, 0xdf4c2b75e4400f7f, 0x95ace99fbb2e20d3,
        0x776f3d782b3dda5a, 0x4455885f07e2aebe, 0x98443bead7f528a0
    }
};

static void load(st_t *data, const char *path, uint32_t count) {
    host::ff::reader reader(path);

    fprintf(stderr, "Reading %s\n", path);
    for (uint32_t i = 0; i < count; i++) {
        if (!reader.readHex(data[i])) {
            fprintf(stderr, "Insufficient data in file, aborting\n");
            exit(1);
        }
    }
}

static void load_challenges(challenges_t &challenges, const char *path) {
    st_t hex_data[8];

    load(hex_data, path, 8);
    challenges.alpha = (impl_mont_t) hex_data[0];
    challenges.beta = (impl_mont_t) hex_data[1];
    challenges.gamma = (impl_mont_t) hex_data[2];
    challenges.delta = (impl_mont_t) hex_data[3];
    challenges.epsilon = (impl_mont_t) hex_data[4];
    challenges.z = (impl_mont_t) hex_data[5];
    challenges.aw = (impl_mont_t) hex_data[6];
    challenges.saw = (impl_mont_t) hex_data[7];
}

static void coset(st_t *eval_form, st_t *dense_form) {
    ntt_context_t ntt_context;
    ntt_context.initialize_root_table(25);

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

template<typename T>
static void load_bin(T *data, const std::filesystem::path &path, const size_t count) {
    std::cout << "loading: " << path << std::endl;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to read " << path << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size == count * sizeof(T)) {
        file.read(reinterpret_cast<char *>(data), count * sizeof(T));
    } else {
        std::cerr << "failed to read " << path << std::endl;
        exit(1);
    }
}

template<typename T>
static void store_bin(const std::filesystem::path &path, const T *data, const size_t count) {
    std::cout << "storing: " << path << std::endl;
    create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data), count * sizeof(T));
}

static void load_or_precompute(st_t *dst, st_t *src, const std::string &cache_path) {
    if (const std::ifstream cache(cache_path); cache.good()) {
        load_bin(dst, cache_path.c_str(), EVAL_SIZE);
    } else {
        coset(dst, src);
        store_bin(cache_path.c_str(), dst, EVAL_SIZE);
    }
}

static void free_prover_key(const prover_key_t &pk) {
    free(pk.q_l_poly);
    free(pk.q_r_poly);
    free(pk.q_o_poly);
    free(pk.q_4_poly);
    free(pk.q_hl_poly);
    free(pk.q_hr_poly);
    free(pk.q_h4_poly);
    free(pk.q_c_poly);
    free(pk.q_arith_poly);

    free(pk.q_l_evals);
    free(pk.q_r_evals);
    free(pk.q_o_evals);
    free(pk.q_4_evals);
    free(pk.q_hl_evals);
    free(pk.q_hr_evals);
    free(pk.q_h4_evals);
    free(pk.q_c_evals);
    free(pk.q_arith_evals);

    free(pk.left_sigma_poly);
    free(pk.right_sigma_poly);
    free(pk.out_sigma_poly);
    free(pk.fourth_sigma_poly);

    free(pk.left_sigma_evals);
    free(pk.right_sigma_evals);
    free(pk.out_sigma_evals);
    free(pk.fourth_sigma_evals);

    free(pk.linear_evals);
    free(pk.v_h_evals_inv);
}

template<typename T>
static bool check_correct(T result, T correct, const std::string &name) {
    if (result != correct) {
        std::cerr << "---- [" << name << "] ----" << std::endl;
        std::cerr << "result : " << to_string(result) << std::endl;
        std::cerr << "correct: " << to_string(correct) << std::endl;
        std::cerr << "--------------------------" << std::endl;
        return false;
    }
    return true;
}

int main() {
    // -- [ALLOC] ------------------------------------------------------------------------------------------------------
    prover_key_t pk{};
    pk.q_l_poly = alloc(POLY_SIZE);
    pk.q_r_poly = alloc(POLY_SIZE);
    pk.q_o_poly = alloc(POLY_SIZE);
    pk.q_4_poly = alloc(POLY_SIZE);
    pk.q_hl_poly = alloc(POLY_SIZE);
    pk.q_hr_poly = alloc(POLY_SIZE);
    pk.q_h4_poly = alloc(POLY_SIZE);
    pk.q_c_poly = alloc(POLY_SIZE);
    pk.q_arith_poly = alloc(POLY_SIZE);

    pk.q_l_evals = alloc(EVAL_SIZE);
    pk.q_r_evals = alloc(EVAL_SIZE);
    pk.q_o_evals = alloc(EVAL_SIZE);
    pk.q_4_evals = alloc(EVAL_SIZE);
    pk.q_hl_evals = alloc(EVAL_SIZE);
    pk.q_hr_evals = alloc(EVAL_SIZE);
    pk.q_h4_evals = alloc(EVAL_SIZE);
    pk.q_c_evals = alloc(EVAL_SIZE);
    pk.q_arith_evals = alloc(EVAL_SIZE);

    pk.left_sigma_poly = alloc(POLY_SIZE);
    pk.right_sigma_poly = alloc(POLY_SIZE);
    pk.out_sigma_poly = alloc(POLY_SIZE);
    pk.fourth_sigma_poly = alloc(POLY_SIZE);

    pk.left_sigma_evals = alloc(EVAL_SIZE);
    pk.right_sigma_evals = alloc(EVAL_SIZE);
    pk.out_sigma_evals = alloc(EVAL_SIZE);
    pk.fourth_sigma_evals = alloc(EVAL_SIZE);

    pk.linear_evals = alloc(EVAL_SIZE);
    pk.v_h_evals_inv = alloc(EVAL_SIZE);

    affine_t *commit_key = alloc_points(POLY_SIZE);

    st_t *blinding = alloc(8);
    st_t *leaf_nodes = alloc(N_LEAF_NODES);
    st_t *non_leaf_nodes = alloc(N_NON_LEAF_NODES);
    challenges_t challenges;

    // -----------------------------------------------------------------------------------------------------------------

    // -- [LOAD] -------------------------------------------------------------------------------------------------------
    load_bin(commit_key, "../data/commit_key.bin", POLY_SIZE);

    load_bin(pk.q_l_poly, "../data/prover_key/q_l_poly.bin", POLY_SIZE);
    load_bin(pk.q_r_poly, "../data/prover_key/q_r_poly.bin", POLY_SIZE);
    load_bin(pk.q_o_poly, "../data/prover_key/q_o_poly.bin", POLY_SIZE);
    load_bin(pk.q_4_poly, "../data/prover_key/q_4_poly.bin", POLY_SIZE);
    load_bin(pk.q_hl_poly, "../data/prover_key/q_hl_poly.bin", POLY_SIZE);
    load_bin(pk.q_hr_poly, "../data/prover_key/q_hr_poly.bin", POLY_SIZE);
    load_bin(pk.q_h4_poly, "../data/prover_key/q_h4_poly.bin", POLY_SIZE);
    load_bin(pk.q_c_poly, "../data/prover_key/q_c_poly.bin", POLY_SIZE);
    load_bin(pk.q_arith_poly, "../data/prover_key/q_arith_poly.bin", POLY_SIZE);
    load_bin(pk.left_sigma_poly, "../data/prover_key/left_sigma_poly.bin", POLY_SIZE);
    load_bin(pk.right_sigma_poly, "../data/prover_key/right_sigma_poly.bin", POLY_SIZE);
    load_bin(pk.out_sigma_poly, "../data/prover_key/out_sigma_poly.bin", POLY_SIZE);
    load_bin(pk.fourth_sigma_poly, "../data/prover_key/fourth_sigma_poly.bin", POLY_SIZE);
    load_bin(pk.linear_evals, "../data/prover_key/linear_evals.bin", EVAL_SIZE);
    load_bin(pk.v_h_evals_inv, "../data/prover_key/v_h_evals_inv.bin", EVAL_SIZE);

    load_challenges(challenges, "../data/inputs/challenges.hex");
    load(blinding, "../data/inputs/blinding.hex", 8);
    load_bin(leaf_nodes, "../data/inputs/leaf_nodes.bin", N_LEAF_NODES);
    load_bin(non_leaf_nodes, "../data/inputs/non_leaf_nodes.bin", N_NON_LEAF_NODES);
    // -----------------------------------------------------------------------------------------------------------------

    // -- [PRECOMPUTE] -------------------------------------------------------------------------------------------------
    load_or_precompute(pk.q_l_evals, pk.q_l_poly, "../cache/prover_key/q_l_evals.bin");
    load_or_precompute(pk.q_r_evals, pk.q_r_poly, "../cache/prover_key/q_r_evals.bin");
    load_or_precompute(pk.q_o_evals, pk.q_o_poly, "../cache/prover_key/q_o_evals.bin");
    load_or_precompute(pk.q_4_evals, pk.q_4_poly, "../cache/prover_key/q_4_evals.bin");
    load_or_precompute(pk.q_hl_evals, pk.q_hl_poly, "../cache/prover_key/q_hl_evals.bin");
    load_or_precompute(pk.q_hr_evals, pk.q_hr_poly, "../cache/prover_key/q_hr_evals.bin");
    load_or_precompute(pk.q_h4_evals, pk.q_h4_poly, "../cache/prover_key/q_h4_evals.bin");
    load_or_precompute(pk.q_c_evals, pk.q_c_poly, "../cache/prover_key/q_c_evals.bin");
    load_or_precompute(pk.q_arith_evals, pk.q_arith_poly, "../cache/prover_key/q_arith_evals.bin");
    load_or_precompute(pk.left_sigma_evals, pk.left_sigma_poly, "../cache/prover_key/left_sigma_evals.bin");
    load_or_precompute(pk.right_sigma_evals, pk.right_sigma_poly, "../cache/prover_key/right_sigma_evals.bin");
    load_or_precompute(pk.out_sigma_evals, pk.out_sigma_poly, "../cache/prover_key/out_sigma_evals.bin");
    load_or_precompute(pk.fourth_sigma_evals, pk.fourth_sigma_poly, "../cache/prover_key/fourth_sigma_evals.bin");
    // -----------------------------------------------------------------------------------------------------------------

    const auto alpha = static_cast<st_t>(challenges.alpha);
    const auto beta = static_cast<st_t>(challenges.beta);
    const auto gamma = static_cast<st_t>(challenges.gamma);
    const auto z_challenge = static_cast<st_t>(challenges.z);
    const auto aw_opening = static_cast<st_t>(challenges.aw);
    const auto saw_opening = static_cast<st_t>(challenges.saw);

    std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock> end;
    std::chrono::duration<signed long int, std::ratio<1, 1000> > elapsed{};

    bool passed = true;

    std::cout << "--[INIT]------------------------------------------------------------------------" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    void *ctx = ffi_init(reinterpret_cast<uint64_t *>(commit_key), pk);
    free(commit_key);
    free_prover_key(pk);

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "--[ROUND_0]---------------------------------------------------------------------" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    ffi_round_0(
        ctx,
        reinterpret_cast<uint64_t *>(blinding),
        reinterpret_cast<uint64_t *>(non_leaf_nodes),
        reinterpret_cast<uint64_t *>(leaf_nodes)
    );
    free(non_leaf_nodes);
    free(leaf_nodes);

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "--[ROUND_1]---------------------------------------------------------------------" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    const round_1_res_t r1_res = ffi_round_1(ctx);
    passed &= check_correct(r1_res.w_l_commit, r1_correct.w_l_commit, "w_l_commit");
    passed &= check_correct(r1_res.w_r_commit, r1_correct.w_r_commit, "w_r_commit");
    passed &= check_correct(r1_res.w_o_commit, r1_correct.w_o_commit, "w_o_commit");
    passed &= check_correct(r1_res.w_4_commit, r1_correct.w_4_commit, "w_4_commit");

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "--[ROUND_3]---------------------------------------------------------------------" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    round_3_res_t r3_res = ffi_round_3(
        ctx,
        reinterpret_cast<const uint64_t *>(&beta),
        reinterpret_cast<const uint64_t *>(&gamma)
    );
    check_correct(r3_res.z_commit, r3_correct.z_commit, "z_commit");

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "--[ROUND_4A]--------------------------------------------------------------------" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    round_4a_res_t r4a_res = ffi_round_4a(
        ctx,
        reinterpret_cast<const uint64_t *>(&alpha),
        reinterpret_cast<const uint64_t *>(&beta),
        reinterpret_cast<const uint64_t *>(&gamma)
    );
    passed &= check_correct(r4a_res.t_0_commit, r4a_correct.t_0_commit, "t_0_commit");
    passed &= check_correct(r4a_res.t_1_commit, r4a_correct.t_1_commit, "t_1_commit");
    passed &= check_correct(r4a_res.t_2_commit, r4a_correct.t_2_commit, "t_2_commit");
    passed &= check_correct(r4a_res.t_3_commit, r4a_correct.t_3_commit, "t_3_commit");
    passed &= check_correct(r4a_res.t_4_commit, r4a_correct.t_4_commit, "t_4_commit");
    passed &= check_correct(r4a_res.t_5_commit, r4a_correct.t_5_commit, "t_5_commit");
    passed &= check_correct(r4a_res.t_6_commit, r4a_correct.t_6_commit, "t_6_commit");
    passed &= check_correct(r4a_res.t_7_commit, r4a_correct.t_7_commit, "t_7_commit");

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "--[ROUND_4B]--------------------------------------------------------------------" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    round_4b_res_t r4b_res = ffi_round_4b(
        ctx,
        reinterpret_cast<const uint64_t *>(&alpha),
        reinterpret_cast<const uint64_t *>(&beta),
        reinterpret_cast<const uint64_t *>(&gamma),
        reinterpret_cast<const uint64_t *>(&z_challenge)
    );
    passed &= check_correct(r4b_res.a_eval, r4b_correct.a_eval, "a_eval");
    passed &= check_correct(r4b_res.b_eval, r4b_correct.b_eval, "b_eval");
    passed &= check_correct(r4b_res.c_eval, r4b_correct.c_eval, "c_eval");
    passed &= check_correct(r4b_res.d_eval, r4b_correct.d_eval, "d_eval");
    passed &= check_correct(r4b_res.a_next_eval, r4b_correct.a_next_eval, "a_next_eval");
    passed &= check_correct(r4b_res.b_next_eval, r4b_correct.b_next_eval, "b_next_eval");
    passed &= check_correct(r4b_res.d_next_eval, r4b_correct.d_next_eval, "d_next_eval");
    passed &= check_correct(r4b_res.left_sigma_eval, r4b_correct.left_sigma_eval, "left_sigma_eval");
    passed &= check_correct(r4b_res.right_sigma_eval, r4b_correct.right_sigma_eval, "right_sigma_eval");
    passed &= check_correct(r4b_res.out_sigma_eval, r4b_correct.out_sigma_eval, "out_sigma_eval");
    passed &= check_correct(r4b_res.permutation_eval, r4b_correct.permutation_eval, "permutation_eval");
    passed &= check_correct(r4b_res.q_arith_eval, r4b_correct.q_arith_eval, "q_arith_eval");
    passed &= check_correct(r4b_res.q_c_eval, r4b_correct.q_c_eval, "q_c_eval");
    passed &= check_correct(r4b_res.q_l_eval, r4b_correct.q_l_eval, "q_l_eval");
    passed &= check_correct(r4b_res.q_r_eval, r4b_correct.q_r_eval, "q_r_eval");
    passed &= check_correct(r4b_res.q_hl_eval, r4b_correct.q_hl_eval, "q_hl_eval");
    passed &= check_correct(r4b_res.q_hr_eval, r4b_correct.q_hr_eval, "q_hr_eval");
    passed &= check_correct(r4b_res.q_h4_eval, r4b_correct.q_h4_eval, "q_h4_eval");

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "--[ROUND_5]---------------------------------------------------------------------" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    round_5_res_t r5_res = ffi_round_5(
        ctx,
        reinterpret_cast<const uint64_t *>(&z_challenge),
        reinterpret_cast<const uint64_t *>(&aw_opening),
        reinterpret_cast<const uint64_t *>(&saw_opening)
    );
    passed &= check_correct(r5_res.aw_opening, r5_correct.aw_opening, "aw_opening");
    passed &= check_correct(r5_res.saw_opening, r5_correct.saw_opening, "saw_opening");

    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "duration: " << elapsed.count() << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    if (passed) {
        std::cout << "PASSED!" << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
    }

    return 0;
}
