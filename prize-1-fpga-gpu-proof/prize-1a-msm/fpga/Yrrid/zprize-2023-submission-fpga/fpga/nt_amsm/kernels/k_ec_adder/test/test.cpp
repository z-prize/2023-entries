/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "k_ec_adder.h"
#include "test_vectors.h"

int main() {
    hls::stream<proc_to_ec_adder_flit_t> from_proc_in;
    hls::stream<field_t> from_d_mult_in;
    hls::stream<field_t> from_inverter_in;
    hls::stream<ec_adder_to_proc_flit_t> to_proc_out;

    k_bls12_381_ec_adder(from_proc_in, from_d_mult_in, from_inverter_in, to_proc_out);

    for (int n = 0; n < 130; n++) {
        for (int i = 0; i < POST_SLICE_BATCH_SIZE; i++) {
            operation_t p;
            p.meta.point_idx = i;
            p.meta.window_idx = 0;
            p.meta.bucket_idx = 0;
            p.meta.valid = test[n][i].valid;
            p.meta.sub = test[n][i].sub;
            p.point.x = test[n][i].ec_point_x;
            p.point.y = test[n][i].ec_point_y;
            p.point.set_inf(test[n][i].ec_point_inf == 1);
            point_t b;
            b.x = test[n][i].ec_bucket_x;
            b.y = test[n][i].ec_bucket_y;
            b.set_inf(test[n][i].ec_bucket_inf == 1);
            from_proc_in << (proc_to_ec_adder_t{p, b}).to_flit();
            if (i < POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE) {
                from_d_mult_in << test[n][i].ec_inverse;
            }
            if (i < INVERTER_BATCH_SIZE) {
                from_inverter_in << test[n][i].ec_untree;
            }
        }

        for (int i = 0; i < POST_SLICE_BATCH_SIZE; i++) {
            operation_t out(to_proc_out.read());
            bool good_valid = out.point.is_inf() == test[n][i].out_inf;
            if (!good_valid) {
                spdlog::error("[{}][{}]: out.is_inf == {}, golden.is_inf == {}", n, i, out.point.is_inf(),
                              test[n][i].out_inf == 1);
            }
            out.point.set_inf(false);
            bool good_x = out.point.x == test[n][i].out_x;
            if (!good_x) {
                spdlog::error("[{}][{}]: out.x == {}, golden.x == {}", n, i, out.point.x.to_string(16),
                              test[n][i].out_x.to_string(16));
            }
            bool good_y = out.point.y == test[n][i].out_y;
            if (!good_y) {
                spdlog::error("[{}][{}]: out.y == {}, golden.y == {}", n, i, out.point.y.to_string(16),
                              test[n][i].out_y.to_string(16));
            }
            assert(good_valid && good_x && good_y);
        }
        spdlog::info("TEST {} PASSED!", n);
    }
}
