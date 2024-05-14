/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "k_ec_adder.h"

void k_bls12_381_ec_adder(
        hls::stream<proc_to_ec_adder_flit_t> &from_proc,
        hls::stream<field_t> &from_d_mult,
        hls::stream<field_t> &from_inverter,
        hls::stream<ec_adder_to_proc_flit_t> &to_proc
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_proc
#pragma HLS INTERFACE mode=axis port=from_d_mult
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_proc
#pragma HLS AGGREGATE variable=from_proc compact=byte
#pragma HLS AGGREGATE variable=from_d_mult compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_proc compact=byte
    hls_thread_local hls::task t_ec_adder(
            k_ec_adder<bls12_381, 0>::k_ec_adder_inst,
            from_proc,
            from_d_mult,
            from_inverter,
            to_proc
    );
}

void k_bls12_377_ec_adder(
        hls::stream<proc_to_ec_adder_flit_t> &from_proc,
        hls::stream<field_t> &from_d_mult,
        hls::stream<field_t> &from_inverter,
        hls::stream<ec_adder_to_proc_flit_t> &to_proc
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_proc
#pragma HLS INTERFACE mode=axis port=from_d_mult
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_proc
#pragma HLS AGGREGATE variable=from_proc compact=byte
#pragma HLS AGGREGATE variable=from_d_mult compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_proc compact=byte
    hls_thread_local hls::task t_ec_adder(
            k_ec_adder<bls12_377, 0>::k_ec_adder_inst,
            from_proc,
            from_d_mult,
            from_inverter,
            to_proc
    );
}
