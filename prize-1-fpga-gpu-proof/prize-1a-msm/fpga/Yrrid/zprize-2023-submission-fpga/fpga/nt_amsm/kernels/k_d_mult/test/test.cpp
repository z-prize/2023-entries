/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "k_d_mult.h"
#include "test_vectors.h"

int main() {
    hls::stream<proc_to_d_mult_flit_t> from_proc_in;
    hls::stream<field_t> to_ec_adder_out;
    hls::stream<field_t> to_inverter_out;

    k_bls12_381_d_mult(from_proc_in, to_ec_adder_out, to_inverter_out);

    std::cout << "-- [k_bls12_381_d_mult basic test] ---------------------------------------------" << std::endl;
    for (int n = 0; n < 130; n++) {
        for (int i = 0; i < POST_SLICE_BATCH_SIZE; i++) {
            proc_to_d_mult_flit_t tmp;
            tmp(383, 0) = test[n][i].px_in;
            tmp[383] = test[n][i].px_inf;
            tmp(767, 384) = test[n][i].bx_in;
            tmp[767] = test[n][i].bx_inf;
            from_proc_in << tmp;
        }
    }

    bool good = true;
    for (int n = 0; n < 130; n++) {
        for (int i = 0; i < POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE; i++) {
            field_t ec_inverse = to_ec_adder_out.read() % base_field_t<bls12_381>::P;
            bool good_to_ec_adder = ec_inverse == test[n][i].ec_inverse_out;
            if (!good_to_ec_adder) {
                spdlog::error("BAD to_ec_adder [{}][{}]: {} != {}", n, i, ec_inverse.to_string(16), test[n][i].ec_inverse_out.to_string(16));
            }
            bool good_to_inverter = true;
            if (i < INVERTER_BATCH_SIZE) {
                field_t tree = to_inverter_out.read() % base_field_t<bls12_381>::P;
                good_to_inverter = tree == test[n][i].d_tree_out;
                if (!good_to_inverter) {
                    spdlog::error("BAD to_inverter [{}][{}]: {} != {}", n, i, tree.to_string(16), test[n][i].d_tree_out.to_string(16));
                }
            }
            good &= good_to_ec_adder && good_to_inverter;
        }
        assert(good);
        std::cout << "TEST " << n << " PASSED" << std::endl;
    }
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    return 0;
}
