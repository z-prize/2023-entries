/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#include "hw_msm.h"
#include "mod_mult.h"

void k_bls12_381_ec_adder(
        hls::stream<proc_to_ec_adder_flit_t> &from_proc,
        hls::stream<field_t> &from_d_mult,
        hls::stream<field_t> &from_inverter,
        hls::stream<ec_adder_to_proc_flit_t> &to_proc
);

void k_bls12_377_ec_adder(
        hls::stream<proc_to_ec_adder_flit_t> &from_proc,
        hls::stream<field_t> &from_d_mult,
        hls::stream<field_t> &from_inverter,
        hls::stream<ec_adder_to_proc_flit_t> &to_proc
);

template<bls12_t CURVE, int KERNEL_ID>
class k_ec_adder {
    static constexpr char KERNEL_NAME[] = "k_ec_adder";
    static constexpr char COMPONENT_NAME[] = "top";
    static constexpr int COMPONENT_ID = 0;
public:
    static void k_ec_adder_inst(
            hls::stream<proc_to_ec_adder_flit_t> &from_proc,
            hls::stream<field_t> &from_d_mult,
            hls::stream<field_t> &from_inverter,
            hls::stream<ec_adder_to_proc_flit_t> &to_proc
    ) {
        hls_thread_local hls::stream<proc_to_ec_adder_flit_t, 4096> s_from_proc_buffer("s_from_proc_buffer");
        hls_thread_local hls::stream<field_t, 4096> s_from_d_mult_buffer("s_from_d_mult_buffer");
        hls_thread_local hls::stream<field_t, 512> s_from_inverter_buffer("s_from_inverter_buffer");
        hls_thread_local hls::stream<field_t, 32> s_dx("s_dx");
        hls_thread_local hls::stream<preprocess_to_stage_1_t, 32> s_pre_process_to_stage_1("s_pre_process_to_stage_1");
        hls_thread_local hls::stream<field_t, 0> s_mult_to_recycle("s_mult_to_recycle");
        hls_thread_local hls::stream<field_t, 64> s_recycle("s_recycle");
        hls_thread_local hls::stream<field_t, 2> s_stage_0_to_stage_1("s_stage_0_to_stage_1");
        hls_thread_local hls::stream<operation_t, 2> s_stage_1_to_post_process("s_stage_1_to_post_process");

        hls_thread_local hls::task t_from_proc_buffer(connector<proc_to_ec_adder_flit_t>, from_proc,
                                                      s_from_proc_buffer);
        hls_thread_local hls::task t_from_d_mult_buffer(connector<field_t>, from_d_mult, s_from_d_mult_buffer);
        hls_thread_local hls::task t_from_inverter_buffer(connector<field_t>, from_inverter, s_from_inverter_buffer);
        hls_thread_local hls::task t_pre_process(pre_process, s_from_proc_buffer, s_dx,
                                                 s_pre_process_to_stage_1);
        hls_thread_local hls::task t_recycle(connector<field_t>, s_mult_to_recycle, s_recycle);
        hls_thread_local hls::task t_stage_0_mult(stage_0_mult, s_dx, s_from_inverter_buffer, s_recycle,
                                                  s_mult_to_recycle, s_stage_0_to_stage_1);
        hls_thread_local hls::task t_stage_1(stage_1, s_from_d_mult_buffer, s_pre_process_to_stage_1,
                                             s_stage_0_to_stage_1,
                                             s_stage_1_to_post_process);
        hls_thread_local hls::task t_post_process(post_process, s_stage_1_to_post_process, to_proc);
    }

private:
    enum exec_t {
        EXECUTE,
        FORWARD_INFTY,
        FORWARD_PT_1,
        FORWARD_PT_2
    };

    struct preprocess_to_stage_1_t {
        field_t dx;
        field_t dy;
        point_t pt_1;
        point_t pt_2;
        exec_t exec;
        meta_t meta;
    };

    static bool is_zero_no_inline(const field_t &a) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=2
        return a == 0;
    }

    static void pre_process(
            hls::stream<proc_to_ec_adder_flit_t> &from_proc,
            hls::stream<field_t> &to_stage_0,
            hls::stream<preprocess_to_stage_1_t> &to_stage_1
    ) {
#pragma HLS PIPELINE II=1 style=frp
        proc_to_ec_adder_t pkt(from_proc.read());
        point_t pt_1 = pkt.ec_bucket;
        point_t pt_2 = pkt.ec_point.point;

        bool pt_1_is_infty = pt_1.is_inf();
        bool pt_2_is_infty = pt_2.is_inf();
        pt_1.set_inf(false);
        pt_2.set_inf(false);

        bool pt_2_y_is_zero = is_zero_no_inline(pt_2.y);

        if (pkt.ec_point.meta.sub && !pt_2_y_is_zero) {
            pt_2.y = negate_inline<CURVE>(pt_2.y);
        }

        field_t dx = sub_red_inline<CURVE>(pt_1.x, pt_2.x);
        field_t dy = sub_red_inline<CURVE>(pt_1.y, pt_2.y);
        bool dx_is_zero = is_zero_no_inline(dx);
        bool dy_is_zero = is_zero_no_inline(dy);

        preprocess_to_stage_1_t out_pkt;
        if (pt_1_is_infty) {
            out_pkt.exec = FORWARD_PT_2;
            dx = base_field_t<CURVE>::MONT_ONE;
            pt_2.set_inf(pt_2_is_infty);
        } else if (pt_2_is_infty) {
            out_pkt.exec = FORWARD_PT_1;
            dx = base_field_t<CURVE>::MONT_ONE;
            pt_1.set_inf(pt_1_is_infty);
        } else if (dx_is_zero) {
            if (dy_is_zero) {
                out_pkt.exec = FORWARD_INFTY;
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
//                TODO: raise an exception in hardware.
//                TODO: i.e. send window_idx and bucket_idx back to host
//                TODO:      we can use the processor status flags to communicate this
//                TODO:      this ensures the host knows there has been a doubling event and can fix the results.
//                TODO: NOTE! This is extremely unlikely given random points.
//                CSIM_ERROR_ARGS("DOUBLING EVENT: point_idx = {}, window_idx = {}, bucket_idx = {}",
//                                pkt.ec_point.point_idx, pkt.ec_point.window_idx, pkt.ec_point.bucket_idx);
//                assert("ERROR: Doubling Event!");
#endif
            } else {
                out_pkt.exec = FORWARD_INFTY;
            }
            dx = base_field_t<CURVE>::MONT_ONE;
        } else {
            out_pkt.exec = EXECUTE;
        }

        to_stage_0 << dx;
        out_pkt.dx = dx;
        out_pkt.dy = dy;
        out_pkt.pt_1 = pt_1;
        out_pkt.pt_2 = pt_2;
        out_pkt.meta = pkt.ec_point.meta;
        to_stage_1 << out_pkt;
    }

    //       from inverter (42)                 recycle (252)               blank (42)
    // |----------------------------|--------------------------------|-------------------------|

    static void stage_0_mult(
            hls::stream<field_t> &dx_in,
            hls::stream<field_t> &from_inverter,
            hls::stream<field_t> &recycle_in,
            hls::stream<field_t> &recycle_out,
            hls::stream<field_t> &to_stage_1
    ) {
#pragma HLS PIPELINE II=1 style=frp
#pragma HLS LATENCY max=38
#pragma HLS EXPRESSION_BALANCE off
        static hls_thread_local int cnt = 0;
        field_t a = dx_in.read();
        field_t b;
        if (cnt < INVERTER_BATCH_SIZE) {
            b = from_inverter.read();
        } else {
            b = recycle_in.read();
        }

        to_stage_1 << b;

        if (cnt < POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE) {
#ifndef __XOR_STUB__
            recycle_out << mod_mult<CURVE>::ll_mod_mult(a, b);
#else
            recycle_out << (a ^ b);
#endif
        }
        cnt = cnt < POST_SLICE_BATCH_SIZE - 1 ? cnt + 1 : 0;
    }

    static void mod_mult_no_inline(field_t a, field_t b) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off
#pragma HLS INLINE recursive
        return mod_mult<CURVE>::ma_mod_mult(a, b);
    }

    static void stage_1(
            hls::stream<field_t> &from_d_mult,
            hls::stream<preprocess_to_stage_1_t> &from_preprocess,
            hls::stream<field_t> &from_stage_0,
            hls::stream<operation_t> &to_proc
    ) {
#pragma HLS LATENCY min=180
#pragma HLS PIPELINE II=1 style=frp
#pragma HLS EXPRESSION_BALANCE off
        static hls_thread_local int cnt = 0;
        field_t ec_inverse = base_field_t<CURVE>::MONT_ONE;
        if (cnt < POST_SLICE_BATCH_SIZE - INVERTER_BATCH_SIZE) {
            ec_inverse = from_d_mult.read();
        }
        cnt = cnt < POST_SLICE_BATCH_SIZE - 1 ? cnt + 1 : 0;
        field_t recycle = from_stage_0.read();
        preprocess_to_stage_1_t pkt = from_preprocess.read();

        operation_t res;
        res.meta = pkt.meta;
        if (pkt.exec == EXECUTE) {
#ifndef __XOR_STUB__
            field_t dx_inv = mod_mult<CURVE>::ma_mod_mult(recycle, ec_inverse);
            field_t lambda = mod_mult<CURVE>::ma_mod_mult(pkt.dy, dx_inv);
            field_t lambda_squared = mod_mult<CURVE>::ma_mod_mult(lambda, lambda);
            field_t xx = add_red_mont_inline<CURVE>(pkt.pt_1.x, pkt.pt_2.x);
            field_t x_out = sub_red_mont_inline<CURVE>(lambda_squared, xx);
            field_t pt_2_x_sub_x_out = sub_red_mont_inline<CURVE>(pkt.pt_2.x, x_out);
            field_t lambda_times_pt_2_x_sub_x_out = mod_mult<CURVE>::ma_mod_mult(lambda, pt_2_x_sub_x_out);
            field_t y_out = sub_red_mont_inline<CURVE>(lambda_times_pt_2_x_sub_x_out, pkt.pt_2.y);
#else
            field_t dx_inv = recycle ^ ec_inverse;
            field_t lambda = pkt.dy ^ dx_inv;
            field_t lambda_squared = lambda ^ lambda;
            field_t xx = add_red_mont_inline<CURVE>(pkt.pt_1.x, pkt.pt_2.x);
            field_t x_out = sub_red_mont_inline<CURVE>(lambda_squared, xx);
            field_t pt_2_x_sub_x_out = sub_red_mont_inline<CURVE>(pkt.pt_2.x, x_out);
            field_t lambda_times_pt_2_x_sub_x_out = lambda ^ pt_2_x_sub_x_out;
            field_t y_out = sub_red_mont_inline<CURVE>(lambda_times_pt_2_x_sub_x_out, pkt.pt_2.y);
#endif
            res.point.x = x_out;
            res.point.y = y_out;
        } else if (pkt.exec == FORWARD_INFTY) {
            res.point = point_t(true);
        } else if (pkt.exec == FORWARD_PT_1) {
            res.point = pkt.pt_1;
        } else if (pkt.exec == FORWARD_PT_2) {
            res.point = pkt.pt_2;
        }
        to_proc << res;
    }

    static void post_process(hls::stream<operation_t> &in, hls::stream<ec_adder_to_proc_flit_t> &out) {
#pragma HLS PIPELINE II=1 style=frp
        operation_t op = in.read();
        bool infty = op.point.is_inf();
        op.point.set_inf(false);
        op.point.x = red_mont_inline<CURVE>(op.point.x);
        op.point.y = red_mont_inline<CURVE>(op.point.y);
        op.point.set_inf(infty);
        out << op.to_flit();
    }
};
