/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "k_d_mult.h"

void k_bls12_381_d_mult(
        hls::stream<proc_to_d_mult_flit_t> &from_proc,
        hls::stream<field_t> &to_ec_adder,
        hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_proc
#pragma HLS INTERFACE mode=axis port=to_ec_adder
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_proc compact=byte
#pragma HLS AGGREGATE variable=to_ec_adder compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::task t_k_d_mult(k_d_mult<bls12_381, 0>::k_d_mult_inst, from_proc, to_ec_adder, to_inverter);
}

void k_bls12_377_d_mult(
        hls::stream<proc_to_d_mult_flit_t> &from_proc,
        hls::stream<field_t> &to_ec_adder,
        hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_proc
#pragma HLS INTERFACE mode=axis port=to_ec_adder
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_proc compact=byte
#pragma HLS AGGREGATE variable=to_ec_adder compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::task t_k_d_mult(k_d_mult<bls12_377, 0>::k_d_mult_inst, from_proc, to_ec_adder, to_inverter);
}
