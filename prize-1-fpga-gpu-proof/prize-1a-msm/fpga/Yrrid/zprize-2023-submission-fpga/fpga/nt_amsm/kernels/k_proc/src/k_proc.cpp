/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "k_proc.h"

void k_bls12_381_proc_0(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<ec_adder_to_proc_flit_t> &from_ec_adder,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<proc_to_d_mult_flit_t> &to_d_mult,
    hls::stream<proc_to_ec_adder_flit_t> &to_ec_adder
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_main
#pragma HLS INTERFACE mode=axis port=from_ec_adder
#pragma HLS INTERFACE mode=axis port=to_main_status
#pragma HLS INTERFACE mode=axis port=to_main_data
#pragma HLS INTERFACE mode=axis port=to_d_mult
#pragma HLS INTERFACE mode=axis port=to_ec_adder
#pragma HLS AGGREGATE variable=from_main compact=byte
#pragma HLS AGGREGATE variable=from_ec_adder compact=byte
#pragma HLS AGGREGATE variable=to_main_status compact=byte
#pragma HLS AGGREGATE variable=to_main_data compact=byte
#pragma HLS AGGREGATE variable=to_d_mult compact=byte
#pragma HLS AGGREGATE variable=to_ec_adder compact=byte
    hls_thread_local hls::task t_proc(k_proc<bls12_381, 0>::k_proc_inst, from_main, from_ec_adder, to_main_status,
                                      to_main_data, to_d_mult, to_ec_adder);
}
