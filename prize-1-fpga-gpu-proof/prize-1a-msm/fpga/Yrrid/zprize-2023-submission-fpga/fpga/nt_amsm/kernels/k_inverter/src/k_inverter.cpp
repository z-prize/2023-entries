/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "k_inverter.h"

void k_bls12_381_inverter(
    hls::stream<field_t> &from_proc_0,
    hls::stream<field_t> &from_proc_1,
    hls::stream<field_t> &from_proc_2,
    hls::stream<field_t> &to_proc_0,
    hls::stream<field_t> &to_proc_1,
    hls::stream<field_t> &to_proc_2
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_proc_0
#pragma HLS INTERFACE mode=axis port=from_proc_1
#pragma HLS INTERFACE mode=axis port=from_proc_2
#pragma HLS INTERFACE mode=axis port=to_proc_0
#pragma HLS INTERFACE mode=axis port=to_proc_1
#pragma HLS INTERFACE mode=axis port=to_proc_2
#pragma HLS AGGREGATE variable=from_proc_0 compact=byte
#pragma HLS AGGREGATE variable=from_proc_1 compact=byte
#pragma HLS AGGREGATE variable=from_proc_2 compact=byte
#pragma HLS AGGREGATE variable=to_proc_0 compact=byte
#pragma HLS AGGREGATE variable=to_proc_1 compact=byte
#pragma HLS AGGREGATE variable=to_proc_2 compact=byte
    hls_thread_local hls::task t_k_inverter(k_inverter<bls12_381>::k_inverter_inst, from_proc_0, from_proc_1,
                                            from_proc_2, to_proc_0, to_proc_1,
                                            to_proc_2);
}

void k_bls12_377_inverter(
    hls::stream<field_t> &from_proc_0,
    hls::stream<field_t> &from_proc_1,
    hls::stream<field_t> &from_proc_2,
    hls::stream<field_t> &to_proc_0,
    hls::stream<field_t> &to_proc_1,
    hls::stream<field_t> &to_proc_2
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_proc_0
#pragma HLS INTERFACE mode=axis port=from_proc_1
#pragma HLS INTERFACE mode=axis port=from_proc_2
#pragma HLS INTERFACE mode=axis port=to_proc_0
#pragma HLS INTERFACE mode=axis port=to_proc_1
#pragma HLS INTERFACE mode=axis port=to_proc_2
#pragma HLS AGGREGATE variable=from_proc_0 compact=byte
#pragma HLS AGGREGATE variable=from_proc_1 compact=byte
#pragma HLS AGGREGATE variable=from_proc_2 compact=byte
#pragma HLS AGGREGATE variable=to_proc_0 compact=byte
#pragma HLS AGGREGATE variable=to_proc_1 compact=byte
#pragma HLS AGGREGATE variable=to_proc_2 compact=byte
    hls_thread_local hls::task t_k_inverter(k_inverter<bls12_377>::k_inverter_inst, from_proc_0, from_proc_1,
                                            from_proc_2, to_proc_0, to_proc_1,
                                            to_proc_2);
}
