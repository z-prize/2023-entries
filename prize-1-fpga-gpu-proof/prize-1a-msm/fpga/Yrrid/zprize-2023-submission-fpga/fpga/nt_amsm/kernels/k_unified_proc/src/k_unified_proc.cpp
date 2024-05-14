/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "hw_msm.h"
#include "k_proc.h"
#include "k_d_mult.h"
#include "k_ec_adder.h"

void k_bls12_381_unified_proc_0(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_main
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_main_status
#pragma HLS INTERFACE mode=axis port=to_main_data
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_main compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_main_status compact=byte
#pragma HLS AGGREGATE variable=to_main_data compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::stream<ec_adder_to_proc_flit_t, 2> s_ec_adder_to_proc;
    hls_thread_local hls::stream<proc_to_d_mult_flit_t, 2> s_proc_to_d_mult;
    hls_thread_local hls::stream<proc_to_ec_adder_flit_t, 2> s_proc_to_ec_adder;
    hls_thread_local hls::stream<field_t, 2> s_d_mult_to_ec_adder;

    hls_thread_local hls::task t_proc(
        k_proc<bls12_381, 0>::k_proc_inst,
        from_main,
        s_ec_adder_to_proc,
        to_main_status,
        to_main_data,
        s_proc_to_d_mult,
        s_proc_to_ec_adder
    );

    hls_thread_local hls::task t_d_mult(
        k_d_mult<bls12_381, 0>::k_d_mult_inst,
        s_proc_to_d_mult,
        s_d_mult_to_ec_adder,
        to_inverter
    );

    hls_thread_local hls::task t_ec_adder(
        k_ec_adder<bls12_381, 0>::k_ec_adder_inst,
        s_proc_to_ec_adder,
        s_d_mult_to_ec_adder,
        from_inverter,
        s_ec_adder_to_proc
    );
}

void k_bls12_381_unified_proc_1(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_main
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_main_status
#pragma HLS INTERFACE mode=axis port=to_main_data
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_main compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_main_status compact=byte
#pragma HLS AGGREGATE variable=to_main_data compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::stream<ec_adder_to_proc_flit_t, 2> s_ec_adder_to_proc;
    hls_thread_local hls::stream<proc_to_d_mult_flit_t, 2> s_proc_to_d_mult;
    hls_thread_local hls::stream<proc_to_ec_adder_flit_t, 2> s_proc_to_ec_adder;
    hls_thread_local hls::stream<field_t, 2> s_d_mult_to_ec_adder;

    hls_thread_local hls::task t_proc(
        k_proc<bls12_381, 1>::k_proc_inst,
        from_main,
        s_ec_adder_to_proc,
        to_main_status,
        to_main_data,
        s_proc_to_d_mult,
        s_proc_to_ec_adder
    );

    hls_thread_local hls::task t_d_mult(
        k_d_mult<bls12_381, 1>::k_d_mult_inst,
        s_proc_to_d_mult,
        s_d_mult_to_ec_adder,
        to_inverter
    );

    hls_thread_local hls::task t_ec_adder(
        k_ec_adder<bls12_381, 1>::k_ec_adder_inst,
        s_proc_to_ec_adder,
        s_d_mult_to_ec_adder,
        from_inverter,
        s_ec_adder_to_proc
    );
}

void k_bls12_381_unified_proc_2(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_main
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_main_status
#pragma HLS INTERFACE mode=axis port=to_main_data
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_main compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_main_status compact=byte
#pragma HLS AGGREGATE variable=to_main_data compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::stream<ec_adder_to_proc_flit_t, 2> s_ec_adder_to_proc;
    hls_thread_local hls::stream<proc_to_d_mult_flit_t, 2> s_proc_to_d_mult;
    hls_thread_local hls::stream<proc_to_ec_adder_flit_t, 2> s_proc_to_ec_adder;
    hls_thread_local hls::stream<field_t, 2> s_d_mult_to_ec_adder;

    hls_thread_local hls::task t_proc(
        k_proc<bls12_381, 2>::k_proc_inst,
        from_main,
        s_ec_adder_to_proc,
        to_main_status,
        to_main_data,
        s_proc_to_d_mult,
        s_proc_to_ec_adder
    );

    hls_thread_local hls::task t_d_mult(
        k_d_mult<bls12_381, 2>::k_d_mult_inst,
        s_proc_to_d_mult,
        s_d_mult_to_ec_adder,
        to_inverter
    );

    hls_thread_local hls::task t_ec_adder(
        k_ec_adder<bls12_381, 2>::k_ec_adder_inst,
        s_proc_to_ec_adder,
        s_d_mult_to_ec_adder,
        from_inverter,
        s_ec_adder_to_proc
    );
}


void k_bls12_377_unified_proc_0(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_main
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_main_status
#pragma HLS INTERFACE mode=axis port=to_main_data
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_main compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_main_status compact=byte
#pragma HLS AGGREGATE variable=to_main_data compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::stream<ec_adder_to_proc_flit_t, 2> s_ec_adder_to_proc;
    hls_thread_local hls::stream<proc_to_d_mult_flit_t, 2> s_proc_to_d_mult;
    hls_thread_local hls::stream<proc_to_ec_adder_flit_t, 2> s_proc_to_ec_adder;
    hls_thread_local hls::stream<field_t, 2> s_d_mult_to_ec_adder;

    hls_thread_local hls::task t_proc(
        k_proc<bls12_377, 0>::k_proc_inst,
        from_main,
        s_ec_adder_to_proc,
        to_main_status,
        to_main_data,
        s_proc_to_d_mult,
        s_proc_to_ec_adder
    );

    hls_thread_local hls::task t_d_mult(
        k_d_mult<bls12_377, 0>::k_d_mult_inst,
        s_proc_to_d_mult,
        s_d_mult_to_ec_adder,
        to_inverter
    );

    hls_thread_local hls::task t_ec_adder(
        k_ec_adder<bls12_377, 0>::k_ec_adder_inst,
        s_proc_to_ec_adder,
        s_d_mult_to_ec_adder,
        from_inverter,
        s_ec_adder_to_proc
    );
}

void k_bls12_377_unified_proc_1(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_main
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_main_status
#pragma HLS INTERFACE mode=axis port=to_main_data
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_main compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_main_status compact=byte
#pragma HLS AGGREGATE variable=to_main_data compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::stream<ec_adder_to_proc_flit_t, 2> s_ec_adder_to_proc;
    hls_thread_local hls::stream<proc_to_d_mult_flit_t, 2> s_proc_to_d_mult;
    hls_thread_local hls::stream<proc_to_ec_adder_flit_t, 2> s_proc_to_ec_adder;
    hls_thread_local hls::stream<field_t, 2> s_d_mult_to_ec_adder;

    hls_thread_local hls::task t_proc(
        k_proc<bls12_377, 1>::k_proc_inst,
        from_main,
        s_ec_adder_to_proc,
        to_main_status,
        to_main_data,
        s_proc_to_d_mult,
        s_proc_to_ec_adder
    );

    hls_thread_local hls::task t_d_mult(
        k_d_mult<bls12_377, 1>::k_d_mult_inst,
        s_proc_to_d_mult,
        s_d_mult_to_ec_adder,
        to_inverter
    );

    hls_thread_local hls::task t_ec_adder(
        k_ec_adder<bls12_377, 1>::k_ec_adder_inst,
        s_proc_to_ec_adder,
        s_d_mult_to_ec_adder,
        from_inverter,
        s_ec_adder_to_proc
    );
}

void k_bls12_377_unified_proc_2(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<field_t> &from_inverter,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<field_t> &to_inverter
) {
#pragma HLS INTERFACE mode=ap_ctrl_none port=return
#pragma HLS INTERFACE mode=axis port=from_main
#pragma HLS INTERFACE mode=axis port=from_inverter
#pragma HLS INTERFACE mode=axis port=to_main_status
#pragma HLS INTERFACE mode=axis port=to_main_data
#pragma HLS INTERFACE mode=axis port=to_inverter
#pragma HLS AGGREGATE variable=from_main compact=byte
#pragma HLS AGGREGATE variable=from_inverter compact=byte
#pragma HLS AGGREGATE variable=to_main_status compact=byte
#pragma HLS AGGREGATE variable=to_main_data compact=byte
#pragma HLS AGGREGATE variable=to_inverter compact=byte
    hls_thread_local hls::stream<ec_adder_to_proc_flit_t, 2> s_ec_adder_to_proc;
    hls_thread_local hls::stream<proc_to_d_mult_flit_t, 2> s_proc_to_d_mult;
    hls_thread_local hls::stream<proc_to_ec_adder_flit_t, 2> s_proc_to_ec_adder;
    hls_thread_local hls::stream<field_t, 2> s_d_mult_to_ec_adder;

    hls_thread_local hls::task t_proc(
        k_proc<bls12_377, 2>::k_proc_inst,
        from_main,
        s_ec_adder_to_proc,
        to_main_status,
        to_main_data,
        s_proc_to_d_mult,
        s_proc_to_ec_adder
    );

    hls_thread_local hls::task t_d_mult(
        k_d_mult<bls12_377, 2>::k_d_mult_inst,
        s_proc_to_d_mult,
        s_d_mult_to_ec_adder,
        to_inverter
    );

    hls_thread_local hls::task t_ec_adder(
        k_ec_adder<bls12_377, 2>::k_ec_adder_inst,
        s_proc_to_ec_adder,
        s_d_mult_to_ec_adder,
        from_inverter,
        s_ec_adder_to_proc
    );
}
