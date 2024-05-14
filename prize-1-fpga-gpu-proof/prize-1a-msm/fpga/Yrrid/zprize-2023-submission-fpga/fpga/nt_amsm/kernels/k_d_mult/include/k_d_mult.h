/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#include "hw_msm.h"
#include "mod_mult.h"

void k_bls12_381_d_mult(
        hls::stream<proc_to_d_mult_flit_t> &from_proc_in,
        hls::stream<field_t> &to_ec_adder_out,
        hls::stream<field_t> &to_inverter_out
);

void k_bls12_377_d_mult(
        hls::stream<proc_to_d_mult_flit_t> &from_proc_in,
        hls::stream<field_t> &to_ec_adder_out,
        hls::stream<field_t> &to_inverter_out
);

template<bls12_t CURVE, int KERNEL_ID>
class k_d_mult {
    static constexpr char KERNEL_NAME[] = "k_d_mult";
    static constexpr char COMPONENT_NAME[] = "top";
    static constexpr int COMPONENT_ID = 0;
public:
    static void k_d_mult_inst(
            hls::stream<proc_to_d_mult_flit_t> &from_proc_in,
            hls::stream<field_t> &to_ec_adder_out,
            hls::stream<field_t> &to_inverter_out
    ) {
        hls_thread_local hls::stream<field_t, 2> s_diff_to_mult("s_diff_to_mult");
        hls_thread_local hls::stream<field_t, 512> s_recycle("s_recycle");
        hls_thread_local hls::stream<field_t, 0> s_mult_to_recycle("s_mult_to_recycle");
        hls_thread_local hls::stream<proc_to_d_mult_flit_t, 2> s_ingress_to_diff("s_ingress_to_diff");
        hls_thread_local hls::stream<field_t, 2> s_mult_to_ec_adder_egress("s_mult_to_ec_adder_egress");
        hls_thread_local hls::stream<field_t, 2> s_mult_to_inverter_egress("s_mult_to_inverter_egress");

        hls_thread_local hls::task t_recycle_connector(connector<field_t>, s_mult_to_recycle, s_recycle);
        hls_thread_local hls::task t_ingress(ingress, from_proc_in, s_ingress_to_diff);
        hls_thread_local hls::task t_diff(diff, s_ingress_to_diff, s_diff_to_mult);
        hls_thread_local hls::task t_mult(mult, s_diff_to_mult, s_recycle, s_mult_to_recycle, s_mult_to_ec_adder_egress, s_mult_to_inverter_egress);
        hls_thread_local hls::task t_ec_adder_egress(ec_adder_egress, s_mult_to_ec_adder_egress, to_ec_adder_out);
        hls_thread_local hls::task t_inverter_egress(inverter_egress, s_mult_to_inverter_egress, to_inverter_out);
    }

private:
    using ingress_buffer_t = proc_to_d_mult_flit_t[POST_SLICE_BATCH_SIZE];
    using ec_adder_egress_buffer_t = field_t[POST_SLICE_BATCH_SIZE];

    // -- [ingress] ----------------------------------------------------------------------------------------------------
    static void ingress_writer(hls::stream<proc_to_d_mult_flit_t> &in, hls::stream_of_blocks<ingress_buffer_t> &out) {
        hls::write_lock<ingress_buffer_t> buf(out);
        for (int i = 0; i < POST_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=1
            buf[i] = in.read();
        }
    }

    static void ingress_reader(hls::stream_of_blocks<ingress_buffer_t> &in, hls::stream<proc_to_d_mult_flit_t> &out) {
        hls::read_lock<ingress_buffer_t> buffer(in);
        for (int i = 0; i < POST_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=1
            out << buffer[POST_SLICE_BATCH_SIZE - 1 - i];
        }
    }

    static void ingress(hls::stream<proc_to_d_mult_flit_t> &in, hls::stream<proc_to_d_mult_flit_t> &out) {
#pragma HLS DATAFLOW
        hls::stream_of_blocks<ingress_buffer_t, 2> buf;
        ingress_writer(in, buf);
        ingress_reader(buf, out);
    }
    // -----------------------------------------------------------------------------------------------------------------

    // -- [ec_adder_egress] --------------------------------------------------------------------------------------------
    static void ec_adder_egress_writer(hls::stream<field_t> &in, hls::stream_of_blocks<ec_adder_egress_buffer_t> &out) {
        hls::write_lock<ec_adder_egress_buffer_t> buf(out);
        for (int i = 0; i < POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=1
            buf[i] = in.read();
        }
    }

    static void ec_adder_egress_reader(hls::stream_of_blocks<ec_adder_egress_buffer_t> &in, hls::stream<field_t> &out) {
        hls::read_lock<ec_adder_egress_buffer_t> buffer(in);
        for (int i = 0; i < POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=1
            out << buffer[POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE - 1 - i];
        }
    }

    static void ec_adder_egress(hls::stream<field_t> &in, hls::stream<field_t> &out) {
#pragma HLS DATAFLOW
        hls::stream_of_blocks<ec_adder_egress_buffer_t, 2> buf;
        ec_adder_egress_writer(in, buf);
        ec_adder_egress_reader(buf, out);
    }
    // -----------------------------------------------------------------------------------------------------------------

    // -- [inverter_egress] --------------------------------------------------------------------------------------------
    static void inverter_egress(hls::stream<field_t> &in, hls::stream<field_t> &out) {
        field_t buffer[INVERTER_BATCH_SIZE];
        for (int i = 0; i < INVERTER_BATCH_SIZE; i++) {
            buffer[i] = in.read();
        }
        for (int i = 0; i < INVERTER_BATCH_SIZE; i++) {
            out << buffer[INVERTER_BATCH_SIZE - 1 - i];
        }
    }
    // -----------------------------------------------------------------------------------------------------------------

    static void diff(hls::stream<proc_to_d_mult_flit_t> &in, hls::stream<field_t> &out) {
#pragma HLS PIPELINE II=1 style=frp
        proc_to_d_mult_flit_t pkt = in.read();
        field_t px = pkt(383, 0);
        field_t bx = pkt(767, 384);

        bool px_inf = px[383];
        bool bx_inf = bx[383];
        px[383] = 0;
        bx[383] = 0;

        field_t dx = sub_red_inline<CURVE>(bx, px);

        if (px_inf || bx_inf || dx == 0) {
            dx = base_field_t<CURVE>::MONT_ONE;
        }

        out << dx;
    }

    static void
    mult(hls::stream<field_t> &dx_in, hls::stream<field_t> &recycle_in, hls::stream<field_t> &recycle_out,
             hls::stream<field_t> &to_ec_adder,
             hls::stream<field_t> &to_inverter) {
#pragma HLS PIPELINE II=1 style=frp
#pragma HLS LATENCY max=38
#pragma HLS EXPRESSION_BALANCE off
        static hls_thread_local int cnt = 0;

        field_t a = dx_in.read();
        field_t b = base_field_t<CURVE>::MONT_ONE;
        if (cnt >= INVERTER_BATCH_SIZE) {
            b = recycle_in.read();
            to_ec_adder << b;
        }
#ifndef __XOR_STUB__
        field_t res = mod_mult<CURVE>::ll_mod_mult(a, b);
#else
        field_t res = a ^ b;
#endif
        if (cnt > POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE - 1) {
            to_inverter << res;
        } else {
            recycle_out << res;
        }

        if (cnt == POST_SLICE_BATCH_SIZE - 1) {
            cnt = 0;
        } else {
            cnt++;
        }
    }
};
