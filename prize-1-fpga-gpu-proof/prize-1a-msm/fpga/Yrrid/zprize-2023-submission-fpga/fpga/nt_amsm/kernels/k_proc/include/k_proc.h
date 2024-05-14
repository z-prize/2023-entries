/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#include "hw_msm.h"
#include "collision_table.h"

void k_proc_0(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<ec_adder_to_proc_flit_t> &from_ec_adder,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<proc_to_d_mult_flit_t> &to_d_mult,
    hls::stream<proc_to_ec_adder_flit_t> &to_ec_adder
);

void k_proc_1(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<ec_adder_to_proc_flit_t> &from_ec_adder,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<proc_to_d_mult_flit_t> &to_d_mult,
    hls::stream<proc_to_ec_adder_flit_t> &to_ec_adder
);

void k_proc_2(
    hls::stream<main_to_proc_flit_t> &from_main,
    hls::stream<ec_adder_to_proc_flit_t> &from_ec_adder,
    hls::stream<proc_status_t> &to_main_status,
    hls::stream<proc_to_main_flit_t> &to_main_data,
    hls::stream<proc_to_d_mult_flit_t> &to_d_mult,
    hls::stream<proc_to_ec_adder_flit_t> &to_ec_adder
);

template<bls12_t CURVE, int KERNEL_ID>
class k_proc {
public:
    static constexpr char KERNEL_NAME[] = "k_proc";
    static constexpr char COMPONENT_NAME[] = "top";
    static constexpr int COMPONENT_ID = 0;

    static void k_proc_inst(
        hls::stream<main_to_proc_flit_t> &from_main,
        hls::stream<ec_adder_to_proc_flit_t> &from_ec_adder,
        hls::stream<proc_status_t> &to_main_status,
        hls::stream<proc_to_main_flit_t> &to_main_data,
        hls::stream<proc_to_d_mult_flit_t> &to_d_mult,
        hls::stream<proc_to_ec_adder_flit_t> &to_ec_adder
    ) {
        hls_thread_local hls::stream<i_to_t, 2> s_i_to_t;
        hls_thread_local hls::stream<tagger_to_point_cache_t, 2> s_tagger_to_point_cache;
        hls_thread_local hls::stream<tagger_to_slicer_t, 2> s_tagger_to_slicer;
        hls_thread_local hls::stream<point_tag_t, 1024> s_point_cache_to_tagger;

        hls_thread_local hls::stream<meta_t, 2> s_slice_0;
        hls_thread_local hls::stream<meta_t, 2> s_slice_1;
        hls_thread_local hls::stream<meta_t, 2> s_slice_2;
        hls_thread_local hls::stream<meta_t, 2> s_slice_3;
        hls_thread_local hls::stream<meta_t, 2> s_slice_4;
        hls_thread_local hls::stream<meta_t, 2> s_slice_5;
        hls_thread_local hls::stream<meta_t, 2> s_slice_6;
        hls_thread_local hls::stream<bucket_idx_t, 2> s_unlock_0;
        hls_thread_local hls::stream<bucket_idx_t, 2> s_unlock_1;
        hls_thread_local hls::stream<bucket_idx_t, 2> s_unlock_2;
        hls_thread_local hls::stream<bucket_idx_t, 2> s_unlock_3;
        hls_thread_local hls::stream<bucket_idx_t, 2> s_unlock_4;
        hls_thread_local hls::stream<bucket_idx_t, 2> s_unlock_5;
        hls_thread_local hls::stream<bucket_idx_t, 2> s_unlock_6;
        hls_thread_local hls::stream<meta_t, 0> s_enqueue_0;
        hls_thread_local hls::stream<meta_t, 0> s_enqueue_1;
        hls_thread_local hls::stream<meta_t, 0> s_enqueue_2;
        hls_thread_local hls::stream<meta_t, 0> s_enqueue_3;
        hls_thread_local hls::stream<meta_t, 0> s_enqueue_4;
        hls_thread_local hls::stream<meta_t, 0> s_enqueue_5;
        hls_thread_local hls::stream<meta_t, 0> s_enqueue_6;
        hls_thread_local hls::stream<meta_t, 512> s_queue_0;
        hls_thread_local hls::stream<meta_t, 512> s_queue_1;
        hls_thread_local hls::stream<meta_t, 512> s_queue_2;
        hls_thread_local hls::stream<meta_t, 512> s_queue_3;
        hls_thread_local hls::stream<meta_t, 512> s_queue_4;
        hls_thread_local hls::stream<meta_t, 512> s_queue_5;
        hls_thread_local hls::stream<meta_t, 512> s_queue_6;
        hls_thread_local hls::stream<meta_t, 2> s_queue_to_dispatch_0;
        hls_thread_local hls::stream<meta_t, 2> s_queue_to_dispatch_1;
        hls_thread_local hls::stream<meta_t, 2> s_queue_to_dispatch_2;
        hls_thread_local hls::stream<meta_t, 2> s_queue_to_dispatch_3;
        hls_thread_local hls::stream<meta_t, 2> s_queue_to_dispatch_4;
        hls_thread_local hls::stream<meta_t, 2> s_queue_to_dispatch_5;
        hls_thread_local hls::stream<meta_t, 2> s_queue_to_dispatch_6;
        hls_thread_local hls::stream<queue_status_t, 2> s_status_0;
        hls_thread_local hls::stream<queue_status_t, 2> s_status_1;
        hls_thread_local hls::stream<queue_status_t, 2> s_status_2;
        hls_thread_local hls::stream<queue_status_t, 2> s_status_3;
        hls_thread_local hls::stream<queue_status_t, 2> s_status_4;
        hls_thread_local hls::stream<queue_status_t, 2> s_status_5;
        hls_thread_local hls::stream<queue_status_t, 2> s_status_6;

        hls_thread_local hls::stream<operation_t, 2> s_retire_to_bw_mem_0;
        hls_thread_local hls::stream<operation_t, 2> s_retire_to_bw_mem_1;

        hls_thread_local hls::stream<operation_t, 2> s_bw_mem_0_to_switch;
        hls_thread_local hls::stream<operation_t, 2> s_bw_mem_1_to_switch;

        hls_thread_local hls::stream<point_t, 2> s_bw_mem_0_to_main_egress;
        hls_thread_local hls::stream<point_t, 2> s_bw_mem_1_to_main_egress;

        hls_thread_local hls::stream<bool, 64> s_dispatch_tokens;
        hls_thread_local hls::stream<meta_t, 2> s_dispatch_to_point_cache;
        hls_thread_local hls::stream<field_t, 32> s_ebs_to_d_mult_egress;
        hls_thread_local hls::stream<point_t, 32> s_ebs_to_ec_adder_egress;
        hls_thread_local hls::stream<operation_t, 32> s_pc_to_ec_adder_egress;
        hls_thread_local hls::stream<field_t, 32> s_pc_to_d_mult_egress;

        hls_thread_local hls::stream<bucket_unlock_t, 32> s_bw_mem_0_to_us_0;
        hls_thread_local hls::stream<bucket_unlock_t, 32> s_bw_mem_1_to_us_1;
        hls_thread_local hls::stream<point_t, 32> s_bw_mem_0_to_ec_bs_0;
        hls_thread_local hls::stream<point_t, 32> s_bw_mem_1_to_ec_bs_1;
        hls_thread_local hls::stream<meta_t, 2> s_dispatch_to_bw_mem_0;
        hls_thread_local hls::stream<meta_t, 2> s_dispatch_to_bw_mem_1;

        // -- [ingress] ------------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_ingress(ingress, from_main, s_i_to_t);
        hls_thread_local hls::task t_retirer(
            retirer,
            from_ec_adder,
            s_retire_to_bw_mem_0,
            s_retire_to_bw_mem_1
        );
        // -------------------------------------------------------------------------------------------------------------
        // -- [egress] -------------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_main_egress(
            main_egress,
            s_bw_mem_0_to_main_egress,
            s_bw_mem_1_to_main_egress,
            to_main_data
        );

        hls_thread_local hls::task t_d_mult_egress(
            d_mult_egress,
            s_pc_to_d_mult_egress,
            s_ebs_to_d_mult_egress,
            to_d_mult
        );

        hls_thread_local hls::task t_ec_adder_egress(
            ec_adder_egress,
            s_pc_to_ec_adder_egress,
            s_ebs_to_ec_adder_egress,
            to_ec_adder
        );

        hls_thread_local hls::task t_status_monitor(
            status_monitor,
            s_status_0,
            s_status_1,
            s_status_2,
            s_status_3,
            s_status_4,
            s_status_5,
            s_status_6,
            to_main_status
        );
        // -------------------------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_tagger(tagger, s_point_cache_to_tagger, s_i_to_t,
                                            s_tagger_to_point_cache, s_tagger_to_slicer);
        hls_thread_local hls::task t_slicer(
            slicer,
            s_tagger_to_slicer,
            s_slice_0,
            s_slice_1,
            s_slice_2,
            s_slice_3,
            s_slice_4,
            s_slice_5,
            s_slice_6
        );
        // -- [queues] -------------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_queue_controller_0(
            queue_controller<0>,
            s_slice_0,
            s_unlock_0,
            s_queue_0,
            s_enqueue_0,
            s_queue_to_dispatch_0,
            s_status_0
        );
        hls_thread_local hls::task t_queue_0(connector<meta_t>, s_enqueue_0, s_queue_0);
        hls_thread_local hls::task t_queue_controller_1(
            queue_controller<1>,
            s_slice_1,
            s_unlock_1,
            s_queue_1,
            s_enqueue_1,
            s_queue_to_dispatch_1,
            s_status_1
        );
        hls_thread_local hls::task t_queue_1(connector<meta_t>, s_enqueue_1, s_queue_1);
        hls_thread_local hls::task t_queue_controller_2(
            queue_controller<2>,
            s_slice_2,
            s_unlock_2,
            s_queue_2,
            s_enqueue_2,
            s_queue_to_dispatch_2,
            s_status_2
        );
        hls_thread_local hls::task t_queue_2(connector<meta_t>, s_enqueue_2, s_queue_2);
        hls_thread_local hls::task t_queue_controller_3(
            queue_controller<3>,
            s_slice_3,
            s_unlock_3,
            s_queue_3,
            s_enqueue_3,
            s_queue_to_dispatch_3,
            s_status_3
        );
        hls_thread_local hls::task t_queue_3(connector<meta_t>, s_enqueue_3, s_queue_3);
        hls_thread_local hls::task t_queue_controller_4(
            queue_controller<4>,
            s_slice_4,
            s_unlock_4,
            s_queue_4,
            s_enqueue_4,
            s_queue_to_dispatch_4,
            s_status_4
        );
        hls_thread_local hls::task t_queue_4(connector<meta_t>, s_enqueue_4, s_queue_4);
        hls_thread_local hls::task t_queue_controller_5(
            queue_controller<5>,
            s_slice_5,
            s_unlock_5,
            s_queue_5,
            s_enqueue_5,
            s_queue_to_dispatch_5,
            s_status_5
        );
        hls_thread_local hls::task t_queue_5(connector<meta_t>, s_enqueue_5, s_queue_5);
        hls_thread_local hls::task t_queue_controller_6(
            queue_controller<6>,
            s_slice_6,
            s_unlock_6,
            s_queue_6,
            s_enqueue_6,
            s_queue_to_dispatch_6,
            s_status_6
        );
        hls_thread_local hls::task t_queue_6(connector<meta_t>, s_enqueue_6, s_queue_6);
        // -------------------------------------------------------------------------------------------------------------
        // -- [window mems] --------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_bw_mem_0(
            bw_mem<4, 0, 0>,
            s_dispatch_to_bw_mem_0,
            s_retire_to_bw_mem_0,
            s_bw_mem_0_to_us_0,
            s_bw_mem_0_to_ec_bs_0,
            s_bw_mem_0_to_main_egress
        );
        hls_thread_local hls::task t_bw_mem_1(
            bw_mem<3, 4, 1>,
            s_dispatch_to_bw_mem_1,
            s_retire_to_bw_mem_1,
            s_bw_mem_1_to_us_1,
            s_bw_mem_1_to_ec_bs_1,
            s_bw_mem_1_to_main_egress
        );
        // -------------------------------------------------------------------------------------------------------------
        // -- [dispatch] -----------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_dispatch(
            dispatch,
            s_dispatch_tokens,
            s_queue_to_dispatch_0,
            s_queue_to_dispatch_1,
            s_queue_to_dispatch_2,
            s_queue_to_dispatch_3,
            s_queue_to_dispatch_4,
            s_queue_to_dispatch_5,
            s_queue_to_dispatch_6,
            s_dispatch_to_bw_mem_0,
            s_dispatch_to_bw_mem_1,
            s_dispatch_to_point_cache
        );
        // -------------------------------------------------------------------------------------------------------------
        // -- [point_cache] --------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_point_cache(
            point_cache,
            s_tagger_to_point_cache,
            s_dispatch_to_point_cache,
            s_point_cache_to_tagger,
            s_dispatch_tokens,
            s_pc_to_d_mult_egress,
            s_pc_to_ec_adder_egress
        );
        // -------------------------------------------------------------------------------------------------------------
        // -- [unlock switches] ----------------------------------------------------------------------------------------
        hls_thread_local hls::task t_unlock_switch_0(
            unlock_switch_0,
            s_bw_mem_0_to_us_0,
            s_unlock_0,
            s_unlock_1,
            s_unlock_2,
            s_unlock_3
        );
        hls_thread_local hls::task t_unlock_switch_1(
            unlock_switch_1,
            s_bw_mem_1_to_us_1,
            s_unlock_4,
            s_unlock_5,
            s_unlock_6
        );
        // -------------------------------------------------------------------------------------------------------------
        // -- [ec_bucket switch] ---------------------------------------------------------------------------------------
        hls_thread_local hls::task t_ec_bucket_switch(
            ec_bucket_switch,
            s_bw_mem_0_to_ec_bs_0,
            s_bw_mem_1_to_ec_bs_1,
            s_ebs_to_d_mult_egress,
            s_ebs_to_ec_adder_egress
        );
        // -------------------------------------------------------------------------------------------------------------
    }

private:
    // -- [internal defines] -------------------------------------------------------------------------------------------
    static constexpr int INVALID_TAG = 1023;

    using queue_status_t = ap_uint<2>;

    static constexpr int QUEUE_STATUS_QUEUE_EMPTY = 0;
    static constexpr int QUEUE_STATUS_QUEUE_ALMOST_EMPTY = 1;
    static constexpr int QUEUE_STATUS_QUEUE_ALMOST_FULL = 2;
    static constexpr int QUEUE_STATUS_NORMAL = 3;

    struct i_to_t {
        scalar_t scalar;
        point_t point;
        point_idx_t idx;
        bool_t report;
        bool_t valid;
        bool_t last;
    };

    struct tagger_to_point_cache_t {
        point_t point;
        point_idx_t idx;
        point_tag_t tag;
        bool_t valid;
    };

    struct tagger_to_slicer_t {
        scalar_t scalar;
        point_idx_t idx;
        point_tag_t tag;
        bool_t report;
        bool_t valid;
        bool_t last;
    };

    struct bucket_unlock_t {
        window_idx_t window_idx;
        bucket_idx_t bucket_idx;
    };

    // -----------------------------------------------------------------------------------------------------------------

    // -- [ingress tasks] ----------------------------------------------------------------------------------------------
    static void ingress(
        hls::stream<field_t> &from_main,
        hls::stream<i_to_t> &to_tagger
    ) {
#pragma HLS PIPELINE II=3
        static hls_thread_local point_idx_t point_idx = 0;
        main_to_proc_t pkt_0(from_main.read());
        main_to_proc_t pkt_1(from_main.read());
        main_to_proc_t pkt_2(from_main.read());
        i_to_t out_pkt;
        out_pkt.scalar = pkt_0.data;
        out_pkt.point.x = pkt_1.data;
        out_pkt.point.y = pkt_2.data;
        out_pkt.valid = pkt_0.valid();
        out_pkt.report = pkt_0.report();
        out_pkt.last = pkt_0.last();
        out_pkt.idx = point_idx;
        to_tagger << out_pkt;
        if (out_pkt.last) {
            point_idx = 0;
        } else {
            point_idx++;
        }
    }

    static void retirer(
        hls::stream<ec_adder_to_proc_flit_t> &from_ec_adder,
        hls::stream<operation_t> &to_bw_mem_0,
        hls::stream<operation_t> &to_bw_mem_1
    ) {
#pragma HLS PIPELINE II=1
        const operation_t op(from_ec_adder.read());
        if (op.meta.window_idx < 4) {
            to_bw_mem_0 << op;
        } else {
            to_bw_mem_1 << op;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------

    // -- [egress tasks] -----------------------------------------------------------------------------------------------
    static void main_egress(
        hls::stream<point_t> &from_bw_mem_0,
        hls::stream<point_t> &from_bw_mem_1,
        hls::stream<proc_to_main_flit_t> &to_main
    ) {
        for (int i = 0; i < 7 * 4096; i++) {
#pragma HLS PIPELINE II=3
            point_t tmp;
            if (i < 4 * 4096) {
                tmp = from_bw_mem_0.read();
            } else {
                tmp = from_bw_mem_1.read();
            }
            ap_uint<256> lo = tmp.x(255, 0);
            ap_uint<256> mi = (ap_uint<128>(tmp.y(127, 0)), ap_uint<128>(tmp.x(383, 256)));
            ap_uint<256> hi = tmp.y(383, 128);
            to_main << lo;
            to_main << mi;
            to_main << hi;
        }
    }

    static void d_mult_egress(
        hls::stream<field_t> &from_point_cache,
        hls::stream<field_t> &from_window,
        hls::stream<proc_to_d_mult_flit_t> &to_d_mult
    ) {
#pragma HLS PIPELINE II=1
        const field_t px = from_point_cache.read();
        const field_t bx = from_window.read();
        to_d_mult << (bx, px);
    }

    static void ec_adder_egress(
        hls::stream<operation_t> &from_point_cache,
        hls::stream<point_t> &from_window,
        hls::stream<proc_to_ec_adder_flit_t> &to_ec_adder
    ) {
#pragma HLS PIPELINE II=1
        const operation_t ec_point = from_point_cache.read();
        const point_t ec_bucket = from_window.read();
        to_ec_adder << proc_to_ec_adder_t(ec_point, ec_bucket).to_flit();
    }

    static void status_monitor(
        hls::stream<queue_status_t> &from_queue_0,
        hls::stream<queue_status_t> &from_queue_1,
        hls::stream<queue_status_t> &from_queue_2,
        hls::stream<queue_status_t> &from_queue_3,
        hls::stream<queue_status_t> &from_queue_4,
        hls::stream<queue_status_t> &from_queue_5,
        hls::stream<queue_status_t> &from_queue_6,
        hls::stream<proc_status_t> &to_main
    ) {
#pragma HLS PIPELINE II=1
        static hls_thread_local bool throttle = false;

        const queue_status_t status_0 = from_queue_0.read();
        const queue_status_t status_1 = from_queue_1.read();
        const queue_status_t status_2 = from_queue_2.read();
        const queue_status_t status_3 = from_queue_3.read();
        const queue_status_t status_4 = from_queue_4.read();
        const queue_status_t status_5 = from_queue_5.read();
        const queue_status_t status_6 = from_queue_6.read();

        const bool all_empty =
                (status_0 == QUEUE_STATUS_QUEUE_EMPTY) && (status_1 == QUEUE_STATUS_QUEUE_EMPTY) &&
                (status_2 == QUEUE_STATUS_QUEUE_EMPTY) && (status_3 == QUEUE_STATUS_QUEUE_EMPTY) &&
                (status_4 == QUEUE_STATUS_QUEUE_EMPTY) && (status_5 == QUEUE_STATUS_QUEUE_EMPTY) &&
                (status_6 == QUEUE_STATUS_QUEUE_EMPTY);
        const bool all_almost_empty =
                (status_0 == QUEUE_STATUS_QUEUE_ALMOST_EMPTY) && (status_1 == QUEUE_STATUS_QUEUE_ALMOST_EMPTY) &&
                (status_2 == QUEUE_STATUS_QUEUE_ALMOST_EMPTY) && (status_3 == QUEUE_STATUS_QUEUE_ALMOST_EMPTY) &&
                (status_4 == QUEUE_STATUS_QUEUE_ALMOST_EMPTY) && (status_5 == QUEUE_STATUS_QUEUE_ALMOST_EMPTY) &&
                (status_6 == QUEUE_STATUS_QUEUE_ALMOST_EMPTY);
        const bool any_almost_full =
                (status_0 == QUEUE_STATUS_QUEUE_ALMOST_FULL) || (status_1 == QUEUE_STATUS_QUEUE_ALMOST_FULL) ||
                (status_2 == QUEUE_STATUS_QUEUE_ALMOST_FULL) || (status_3 == QUEUE_STATUS_QUEUE_ALMOST_FULL) ||
                (status_4 == QUEUE_STATUS_QUEUE_ALMOST_FULL) || (status_5 == QUEUE_STATUS_QUEUE_ALMOST_FULL) ||
                (status_6 == QUEUE_STATUS_QUEUE_ALMOST_FULL);

        if (all_almost_empty || all_empty) {
            throttle = false;
        } else if (any_almost_full) {
            throttle = true;
        }

        if (throttle) {
            to_main << PROC_STATUS_THROTTLE;
        } else if (all_empty) {
            to_main << PROC_STATUS_FLUSHED;
        } else {
            to_main << PROC_STATUS_NORMAL;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------

    static void tagger(
        hls::stream<point_tag_t> &from_point_cache,
        hls::stream<i_to_t> &from_ingress,
        hls::stream<tagger_to_point_cache_t> &to_point_cache,
        hls::stream<tagger_to_slicer_t> &to_slicer
    ) {
#pragma HLS PIPELINE II=1
        static hls_thread_local short init_cnt = 0;
        point_tag_t tag = INVALID_TAG;
        i_to_t pkt = from_ingress.read();
        if (pkt.valid) {
            if (init_cnt < 1023) {
                tag = init_cnt;
                init_cnt++;
            } else {
                tag = from_point_cache.read();
            }
        }
        to_point_cache << tagger_to_point_cache_t{pkt.point, pkt.idx, tag, pkt.valid};
        to_slicer << tagger_to_slicer_t{pkt.scalar, pkt.idx, tag, pkt.report, pkt.valid, pkt.last};
    }

    static void slicer(
        hls::stream<tagger_to_slicer_t> &from_tagger,
        hls::stream<meta_t> &to_queue_0,
        hls::stream<meta_t> &to_queue_1,
        hls::stream<meta_t> &to_queue_2,
        hls::stream<meta_t> &to_queue_3,
        hls::stream<meta_t> &to_queue_4,
        hls::stream<meta_t> &to_queue_5,
        hls::stream<meta_t> &to_queue_6
    ) {
#pragma HLS PIPELINE II=1
        tagger_to_slicer_t pkt = from_tagger.read();
        ap_uint<259> scalar = pkt.scalar;
#ifndef __NO_SCALAR_SCRAMBLE__
        if constexpr (CURVE == bls12_381) {
            const ap_uint<259> scramble_table[17] = {
                0 * base_field_t<bls12_381>::CURVE_ORDER,
                1 * base_field_t<bls12_381>::CURVE_ORDER,
                2 * base_field_t<bls12_381>::CURVE_ORDER,
                3 * base_field_t<bls12_381>::CURVE_ORDER,
                4 * base_field_t<bls12_381>::CURVE_ORDER,
                5 * base_field_t<bls12_381>::CURVE_ORDER,
                6 * base_field_t<bls12_381>::CURVE_ORDER,
                7 * base_field_t<bls12_381>::CURVE_ORDER,
                8 * base_field_t<bls12_381>::CURVE_ORDER,
                9 * base_field_t<bls12_381>::CURVE_ORDER,
                10 * base_field_t<bls12_381>::CURVE_ORDER,
                11 * base_field_t<bls12_381>::CURVE_ORDER,
                12 * base_field_t<bls12_381>::CURVE_ORDER,
                13 * base_field_t<bls12_381>::CURVE_ORDER,
                14 * base_field_t<bls12_381>::CURVE_ORDER,
                15 * base_field_t<bls12_381>::CURVE_ORDER,
                16 * base_field_t<bls12_381>::CURVE_ORDER
            };
            static hls_thread_local ap_uint<5> cnt = 0;
            if (pkt.valid) {
                scalar = scalar + scramble_table[cnt];
                cnt = cnt < 16 ? ap_uint<5>(cnt + 1) : ap_uint<5>(0);
            }
        } else if (CURVE == bls12_377) {
            const ap_uint<259> scramble_table[109] = {
                0 * base_field_t<bls12_377>::CURVE_ORDER,
                1 * base_field_t<bls12_377>::CURVE_ORDER,
                2 * base_field_t<bls12_377>::CURVE_ORDER,
                3 * base_field_t<bls12_377>::CURVE_ORDER,
                4 * base_field_t<bls12_377>::CURVE_ORDER,
                5 * base_field_t<bls12_377>::CURVE_ORDER,
                6 * base_field_t<bls12_377>::CURVE_ORDER,
                7 * base_field_t<bls12_377>::CURVE_ORDER,
                8 * base_field_t<bls12_377>::CURVE_ORDER,
                9 * base_field_t<bls12_377>::CURVE_ORDER,
                10 * base_field_t<bls12_377>::CURVE_ORDER,
                11 * base_field_t<bls12_377>::CURVE_ORDER,
                12 * base_field_t<bls12_377>::CURVE_ORDER,
                13 * base_field_t<bls12_377>::CURVE_ORDER,
                14 * base_field_t<bls12_377>::CURVE_ORDER,
                15 * base_field_t<bls12_377>::CURVE_ORDER,
                16 * base_field_t<bls12_377>::CURVE_ORDER,
                17 * base_field_t<bls12_377>::CURVE_ORDER,
                18 * base_field_t<bls12_377>::CURVE_ORDER,
                19 * base_field_t<bls12_377>::CURVE_ORDER,
                20 * base_field_t<bls12_377>::CURVE_ORDER,
                21 * base_field_t<bls12_377>::CURVE_ORDER,
                22 * base_field_t<bls12_377>::CURVE_ORDER,
                23 * base_field_t<bls12_377>::CURVE_ORDER,
                24 * base_field_t<bls12_377>::CURVE_ORDER,
                25 * base_field_t<bls12_377>::CURVE_ORDER,
                26 * base_field_t<bls12_377>::CURVE_ORDER,
                27 * base_field_t<bls12_377>::CURVE_ORDER,
                28 * base_field_t<bls12_377>::CURVE_ORDER,
                29 * base_field_t<bls12_377>::CURVE_ORDER,
                30 * base_field_t<bls12_377>::CURVE_ORDER,
                31 * base_field_t<bls12_377>::CURVE_ORDER,
                32 * base_field_t<bls12_377>::CURVE_ORDER,
                33 * base_field_t<bls12_377>::CURVE_ORDER,
                34 * base_field_t<bls12_377>::CURVE_ORDER,
                35 * base_field_t<bls12_377>::CURVE_ORDER,
                36 * base_field_t<bls12_377>::CURVE_ORDER,
                37 * base_field_t<bls12_377>::CURVE_ORDER,
                38 * base_field_t<bls12_377>::CURVE_ORDER,
                39 * base_field_t<bls12_377>::CURVE_ORDER,
                40 * base_field_t<bls12_377>::CURVE_ORDER,
                41 * base_field_t<bls12_377>::CURVE_ORDER,
                42 * base_field_t<bls12_377>::CURVE_ORDER,
                43 * base_field_t<bls12_377>::CURVE_ORDER,
                44 * base_field_t<bls12_377>::CURVE_ORDER,
                45 * base_field_t<bls12_377>::CURVE_ORDER,
                46 * base_field_t<bls12_377>::CURVE_ORDER,
                47 * base_field_t<bls12_377>::CURVE_ORDER,
                48 * base_field_t<bls12_377>::CURVE_ORDER,
                49 * base_field_t<bls12_377>::CURVE_ORDER,
                50 * base_field_t<bls12_377>::CURVE_ORDER,
                51 * base_field_t<bls12_377>::CURVE_ORDER,
                52 * base_field_t<bls12_377>::CURVE_ORDER,
                53 * base_field_t<bls12_377>::CURVE_ORDER,
                54 * base_field_t<bls12_377>::CURVE_ORDER,
                55 * base_field_t<bls12_377>::CURVE_ORDER,
                56 * base_field_t<bls12_377>::CURVE_ORDER,
                57 * base_field_t<bls12_377>::CURVE_ORDER,
                58 * base_field_t<bls12_377>::CURVE_ORDER,
                59 * base_field_t<bls12_377>::CURVE_ORDER,
                60 * base_field_t<bls12_377>::CURVE_ORDER,
                61 * base_field_t<bls12_377>::CURVE_ORDER,
                62 * base_field_t<bls12_377>::CURVE_ORDER,
                63 * base_field_t<bls12_377>::CURVE_ORDER,
                64 * base_field_t<bls12_377>::CURVE_ORDER,
                65 * base_field_t<bls12_377>::CURVE_ORDER,
                66 * base_field_t<bls12_377>::CURVE_ORDER,
                67 * base_field_t<bls12_377>::CURVE_ORDER,
                68 * base_field_t<bls12_377>::CURVE_ORDER,
                69 * base_field_t<bls12_377>::CURVE_ORDER,
                70 * base_field_t<bls12_377>::CURVE_ORDER,
                71 * base_field_t<bls12_377>::CURVE_ORDER,
                72 * base_field_t<bls12_377>::CURVE_ORDER,
                73 * base_field_t<bls12_377>::CURVE_ORDER,
                74 * base_field_t<bls12_377>::CURVE_ORDER,
                75 * base_field_t<bls12_377>::CURVE_ORDER,
                76 * base_field_t<bls12_377>::CURVE_ORDER,
                77 * base_field_t<bls12_377>::CURVE_ORDER,
                78 * base_field_t<bls12_377>::CURVE_ORDER,
                79 * base_field_t<bls12_377>::CURVE_ORDER,
                80 * base_field_t<bls12_377>::CURVE_ORDER,
                81 * base_field_t<bls12_377>::CURVE_ORDER,
                82 * base_field_t<bls12_377>::CURVE_ORDER,
                83 * base_field_t<bls12_377>::CURVE_ORDER,
                84 * base_field_t<bls12_377>::CURVE_ORDER,
                85 * base_field_t<bls12_377>::CURVE_ORDER,
                86 * base_field_t<bls12_377>::CURVE_ORDER,
                87 * base_field_t<bls12_377>::CURVE_ORDER,
                88 * base_field_t<bls12_377>::CURVE_ORDER,
                89 * base_field_t<bls12_377>::CURVE_ORDER,
                90 * base_field_t<bls12_377>::CURVE_ORDER,
                91 * base_field_t<bls12_377>::CURVE_ORDER,
                92 * base_field_t<bls12_377>::CURVE_ORDER,
                93 * base_field_t<bls12_377>::CURVE_ORDER,
                94 * base_field_t<bls12_377>::CURVE_ORDER,
                95 * base_field_t<bls12_377>::CURVE_ORDER,
                96 * base_field_t<bls12_377>::CURVE_ORDER,
                97 * base_field_t<bls12_377>::CURVE_ORDER,
                98 * base_field_t<bls12_377>::CURVE_ORDER,
                99 * base_field_t<bls12_377>::CURVE_ORDER,
                100 * base_field_t<bls12_377>::CURVE_ORDER,
                101 * base_field_t<bls12_377>::CURVE_ORDER,
                102 * base_field_t<bls12_377>::CURVE_ORDER,
                103 * base_field_t<bls12_377>::CURVE_ORDER,
                104 * base_field_t<bls12_377>::CURVE_ORDER,
                105 * base_field_t<bls12_377>::CURVE_ORDER,
                106 * base_field_t<bls12_377>::CURVE_ORDER,
                107 * base_field_t<bls12_377>::CURVE_ORDER,
                108 * base_field_t<bls12_377>::CURVE_ORDER
            };
            static hls_thread_local ap_uint<7> cnt = 0;
            if (pkt.valid) {
                scalar = scalar + scramble_table[cnt];
                cnt = cnt < 108 ? ap_uint<7>(cnt + 1) : ap_uint<7>(0);
            }
        }
#endif

        bucket_idx_t slices[21];
        for (auto &slice: slices) {
#pragma HLS UNROLL
            slice = scalar(12, 0);
            scalar >>= 13;
        }

        unsigned g = 0;
        unsigned p = 0;
        for (int i = 0; i < 21; i++) {
#pragma HLS UNROLL
            if (slices[i] >= 4095) {
                p = p + (1 << i);
            }
            if (slices[i] > 4095) {
                g = g + (1 << i);
            }
        }

        const unsigned selector = (g + p) ^ (g ^ p);

        for (int i = 0; i < 21; i++) {
#pragma HLS UNROLL
            if ((selector & (1 << i)) != 0) {
                slices[i]++;
            }
        }

        hls_thread_local hls::vector<meta_t, 21> to_queues;
        for (int i = 0; i < 21; i++) {
#pragma HLS UNROLL
            bool msb = (slices[i] >> 12) != 0;
            bool valid = true;

            if (msb) {
                slices[i] = bucket_idx_t(0) - slices[i];
            }

            if (slices[i] == 0) {
                slices[i] = INVALID_BUCKET;
                valid = false;
            } else {
                slices[i] = slices[i] - 1;
            }

            to_queues[i].point_idx = pkt.idx;
            to_queues[i].bucket_idx = slices[i];
            to_queues[i].sub = msb;
            to_queues[i].valid = valid && pkt.valid;
            to_queues[i].tag = to_queues[i].valid ? pkt.tag : point_tag_t(INVALID_TAG);
            to_queues[i].report = pkt.report;
            to_queues[i].last = pkt.last;
            // if (KERNEL_ID == 0 && to_queues[i].valid) {
            //     CSIM_INFO_ARGS("slicer: window_idx: {}, bucket_idx: {}, add_sub: {}", i, to_queues[i].bucket_idx, to_queues[i].sub);
            // }
        }

        if constexpr (KERNEL_ID == 0) {
            int tag_cnt = 0;
            for (int i = 0; i < 7; i++) {
#pragma HLS UNROLL
                tag_cnt += to_queues[i].valid ? 1 : 0;
            }
            for (int i = 0; i < 7; i++) {
#pragma HLS UNROLL
                to_queues[i].tag_cnt = tag_cnt;
            }
            to_queue_0 << to_queues[0];
            to_queue_1 << to_queues[1];
            to_queue_2 << to_queues[2];
            to_queue_3 << to_queues[3];
            to_queue_4 << to_queues[4];
            to_queue_5 << to_queues[5];
            to_queue_6 << to_queues[6];
        } else if constexpr (KERNEL_ID == 1) {
            int tag_cnt = 0;
            for (int i = 7; i < 14; i++) {
#pragma HLS UNROLL
                tag_cnt += to_queues[i].valid ? 1 : 0;
            }
            for (int i = 7; i < 14; i++) {
#pragma HLS UNROLL
                to_queues[i].tag_cnt = tag_cnt;
            }
            to_queue_0 << to_queues[7];
            to_queue_1 << to_queues[8];
            to_queue_2 << to_queues[9];
            to_queue_3 << to_queues[10];
            to_queue_4 << to_queues[11];
            to_queue_5 << to_queues[12];
            to_queue_6 << to_queues[13];
        } else if constexpr (KERNEL_ID == 2) {
            int tag_cnt = 0;
            for (int i = 14; i < 21; i++) {
#pragma HLS UNROLL
                tag_cnt += to_queues[i].valid ? 1 : 0;
            }
            for (int i = 14; i < 21; i++) {
#pragma HLS UNROLL
                to_queues[i].tag_cnt = tag_cnt;
            }
            to_queue_0 << to_queues[14];
            to_queue_1 << to_queues[15];
            to_queue_2 << to_queues[16];
            to_queue_3 << to_queues[17];
            to_queue_4 << to_queues[18];
            to_queue_5 << to_queues[19];
            to_queue_6 << to_queues[20];
        }
    }

    template<int COMPONENT_ID>
    static void queue_controller(
        hls::stream<meta_t> &from_slicer,
        hls::stream<bucket_idx_t> &unlock_in,
        hls::stream<meta_t> &dequeue_in,
        hls::stream<meta_t> &enqueue_out,
        hls::stream<meta_t> &to_window,
        hls::stream<queue_status_t> &to_status_monitor
    ) {
#pragma HLS PIPELINE II=7
        static hls_thread_local bool_t locks[N_BUCKETS] = {0};
        static hls_thread_local meta_t ce_0;
        static hls_thread_local meta_t ce_1;
        static hls_thread_local meta_t last_dequeue;
        static hls_thread_local short queue_cnt = 0;
        static hls_thread_local short lock_cnt = 0;
        using namespace collision_table;
        static constexpr char COMPONENT_NAME[] = "queue";

        if (!unlock_in.empty()) {
            bucket_idx_t idx = unlock_in.read();
            locks[idx] = 0;
            lock_cnt--;
        }

        if (!from_slicer.empty() && !to_window.full()) {
            meta_t input = from_slicer.read();
            ap_uint<1> input_valid = input.valid;
            ap_uint<1> ce_0_valid = ce_0.valid;
            ap_uint<1> ce_1_valid = ce_1.valid;
            ap_uint<1> last_dequeue_valid = last_dequeue.valid;
            if (!input_valid) {
                input.bucket_idx = INVALID_BUCKET;
            }
            if (!ce_0_valid) {
                ce_0.bucket_idx = INVALID_BUCKET;
            }
            if (!ce_1_valid) {
                ce_1.bucket_idx = INVALID_BUCKET;
            }
            if (!last_dequeue.valid) {
                last_dequeue.bucket_idx = INVALID_BUCKET;
            }
            ap_uint<1> input_matches_ce_0 = input_valid && (input.bucket_idx == ce_0.bucket_idx);
            ap_uint<1> input_matches_ce_1 = input_valid && (input.bucket_idx == ce_1.bucket_idx);
            ap_uint<1> input_locked = input_valid && locks[input.bucket_idx];
            ap_uint<1> last_locked = last_dequeue_valid && locks[last_dequeue.bucket_idx];
            ap_uint<1> last_bucket_match = last_dequeue_valid && ((last_dequeue.bucket_idx == input.bucket_idx) ||
                                                                  (last_dequeue.bucket_idx == ce_0.bucket_idx) ||
                                                                  (last_dequeue.bucket_idx == ce_1.bucket_idx));

            ap_uint<9> action_index = (last_bucket_match, last_locked, input_locked, input_matches_ce_1,
                                       input_matches_ce_0, last_dequeue_valid, ce_1_valid, ce_0_valid, input_valid);
            action_t action = action_table[action_index];
#pragma HLS AGGREGATE variable=action_table compact=bit
#pragma HLS BIND_STORAGE variable=action_table type=ROM_1P impl=BRAM

            meta_t res;
            if (action.output == OUTPUT_INPUT) {
                res = input;
            } else if ((action.output == OUTPUT_CE) && (action.ce == CE_SELECT_0)) {
                res = ce_0;
            } else if ((action.output == OUTPUT_CE) && (action.ce == CE_SELECT_1)) {
                res = ce_1;
            } else {
                // copy all fields
                res = input;
                res.bucket_idx = INVALID_BUCKET;
                res.point_idx = -1;
                res.tag = INVALID_TAG;
                res.valid = false;
            }

            meta_t next_enqueue;
            if (action.enqueue == ENQUEUE_INPUT) {
                next_enqueue = input;
            } else if ((action.enqueue == ENQUEUE_CE) && (action.ce == CE_SELECT_0)) {
                next_enqueue = ce_0;
            } else if ((action.enqueue == ENQUEUE_CE) && (action.ce == CE_SELECT_1)) {
                next_enqueue = ce_1;
            } else if (action.enqueue == ENQUEUE_LAST) {
                next_enqueue = last_dequeue;
            }

            if (action.lock == LOCK_INPUT) {
                locks[input.bucket_idx] = 1;
                lock_cnt++;
            } else if (action.lock == LOCK_CE && action.ce == CE_SELECT_0) {
                locks[ce_0.bucket_idx] = 1;
                lock_cnt++;
            } else if (action.lock == LOCK_CE && action.ce == CE_SELECT_1) {
                locks[ce_1.bucket_idx] = 1;
                lock_cnt++;
            }

            if ((action.ce_update == CE_UPDATE_LAST) && (action.ce == CE_SELECT_0)) {
                ce_0 = last_dequeue;
            } else if (action.ce_update == CE_UPDATE_LAST && action.ce == CE_SELECT_1) {
                ce_1 = last_dequeue;
            } else if (action.ce_update == CE_UPDATE_INVALIDATE && action.ce == CE_SELECT_0) {
                ce_0.valid = false;
            } else if (action.ce_update == CE_UPDATE_INVALIDATE && action.ce == CE_SELECT_1) {
                ce_1.valid = false;
            }

            if (action.last_update == LAST_UPDATE_DEQUEUE) {
                if (queue_cnt != 0) {
                    last_dequeue = dequeue_in.read();
                    queue_cnt--;
                } else {
                    last_dequeue.valid = false;
                }
            }

            if (action.enqueue != ENQUEUE_NOTHING) {
                enqueue_out << next_enqueue;
                queue_cnt++;
            }

            res.window_idx = COMPONENT_ID;
            to_window << res;

            if (input.report) {
                bool empty =
                        !ce_0_valid && !ce_1_valid && !last_dequeue.valid && (queue_cnt == 0) && (lock_cnt == 0);
                proc_status_t status = PROC_STATUS_NORMAL;
                if (empty) {
                    status = QUEUE_STATUS_QUEUE_EMPTY;
                } else if (queue_cnt > 128) {
                    status = QUEUE_STATUS_QUEUE_ALMOST_FULL;
                } else if (queue_cnt < 64) {
                    status = QUEUE_STATUS_QUEUE_ALMOST_EMPTY;
                }
                // CSIM_INFO_ARGS("queue_cnt: {}, lock_cnt: {}", queue_cnt, lock_cnt);
                to_status_monitor << status;
            }
        }
    }

    static void dispatch(
        hls::stream<bool> &dispatch_token_in,
        hls::stream<meta_t> &from_queue_0,
        hls::stream<meta_t> &from_queue_1,
        hls::stream<meta_t> &from_queue_2,
        hls::stream<meta_t> &from_queue_3,
        hls::stream<meta_t> &from_queue_4,
        hls::stream<meta_t> &from_queue_5,
        hls::stream<meta_t> &from_queue_6,
        hls::stream<meta_t> &to_bw_mem_0,
        hls::stream<meta_t> &to_bw_mem_1,
        hls::stream<meta_t> &to_point_cache
    ) {
#pragma HLS PIPELINE II=7
        dispatch_token_in.read();
        meta_t meta = from_queue_0.read();
        to_point_cache << meta;
        to_bw_mem_0 << meta;

        meta = from_queue_1.read();
        to_point_cache << meta;
        to_bw_mem_0 << meta;

        meta = from_queue_2.read();
        to_point_cache << meta;
        to_bw_mem_0 << meta;

        meta = from_queue_3.read();
        to_point_cache << meta;
        to_bw_mem_0 << meta;

        meta = from_queue_4.read();
        to_point_cache << meta;
        to_bw_mem_1 << meta;

        meta = from_queue_5.read();
        to_point_cache << meta;
        to_bw_mem_1 << meta;

        meta = from_queue_6.read();
        to_point_cache << meta;
        to_bw_mem_1 << meta;
    }

    static void unlock_switch_0(
        hls::stream<bucket_unlock_t> &from_bw_mem,
        hls::stream<bucket_idx_t> &to_queue_0,
        hls::stream<bucket_idx_t> &to_queue_1,
        hls::stream<bucket_idx_t> &to_queue_2,
        hls::stream<bucket_idx_t> &to_queue_3
    ) {
#pragma HLS PIPELINE II=1
        bucket_unlock_t pkt = from_bw_mem.read();
        if (pkt.window_idx == 0) {
            to_queue_0 << pkt.bucket_idx;
        } else if (pkt.window_idx == 1) {
            to_queue_1 << pkt.bucket_idx;
        } else if (pkt.window_idx == 2) {
            to_queue_2 << pkt.bucket_idx;
        } else {
            to_queue_3 << pkt.bucket_idx;
        }
    }

    static void unlock_switch_1(
        hls::stream<bucket_unlock_t> &from_bw_mem,
        hls::stream<bucket_idx_t> &to_queue_4,
        hls::stream<bucket_idx_t> &to_queue_5,
        hls::stream<bucket_idx_t> &to_queue_6
    ) {
#pragma HLS PIPELINE II=1
        bucket_unlock_t pkt = from_bw_mem.read();
        if (pkt.window_idx == 4) {
            to_queue_4 << pkt.bucket_idx;
        } else if (pkt.window_idx == 5) {
            to_queue_5 << pkt.bucket_idx;
        } else {
            to_queue_6 << pkt.bucket_idx;
        }
    }

    template<int BANKS, int START_WINDOW_IDX, int COMPONENT_ID>
    static void bw_mem(
        hls::stream<meta_t> &from_dispatch,
        hls::stream<operation_t> &from_retire,
        hls::stream<bucket_unlock_t> &unlock_out,
        hls::stream<point_t> &to_ec_bucket_switch,
        hls::stream<point_t> &to_main_egress
    ) {
#ifndef __SYNTHESIS__
        static hls_thread_local point_t *mem;
        static hls_thread_local bool sim_init = true;
        if (sim_init) {
            mem = new point_t[BANKS * 4096];
            sim_init = false;
        }
#else
        static hls_thread_local point_t mem[BANKS * 4096];
#endif
#pragma HLS BIND_STORAGE variable=mem type=RAM_T2P impl=URAM
        static hls_thread_local bool dump = false;
        static constexpr char COMPONENT_NAME[] = "window";

        for (int i = 0; i < BANKS * 4096; i++) {
            if (dump) {
                to_main_egress << mem[i];
            }
            mem[i] = point_t(true);
        }

        int last_cnt = 0;
        while (last_cnt < BANKS) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=mem inter false
            if (!from_retire.empty()) {
                const operation_t retire = from_retire.read();
                if (retire.meta.valid) {
                    const int idx = ((retire.meta.window_idx - START_WINDOW_IDX) << 12) | retire.meta.bucket_idx;
                    mem[idx] = retire.point;
                    unlock_out << bucket_unlock_t{retire.meta.window_idx, retire.meta.bucket_idx};
                } else if (retire.meta.last) {
                    last_cnt++;
                }
            }
            if (!from_dispatch.empty()) {
                const meta_t meta = from_dispatch.read();
                if (meta.valid) {
                    const int idx = ((meta.window_idx - START_WINDOW_IDX) << 12) | meta.bucket_idx;
                    to_ec_bucket_switch << mem[idx];
                } else {
                    to_ec_bucket_switch << point_t(true);
                }
            }
        }
        dump = true;
    }

    static void ec_bucket_switch(
        hls::stream<point_t> &from_bw_0,
        hls::stream<point_t> &from_bw_1,
        hls::stream<field_t> &to_d_mult_egress,
        hls::stream<point_t> &to_ec_adder_egress
    ) {
#pragma HLS PIPELINE II=7
        point_t ec_bucket = from_bw_0.read();
        to_d_mult_egress << ec_bucket.x;
        to_ec_adder_egress << ec_bucket;
        ec_bucket = from_bw_0.read();
        to_d_mult_egress << ec_bucket.x;
        to_ec_adder_egress << ec_bucket;
        ec_bucket = from_bw_0.read();
        to_d_mult_egress << ec_bucket.x;
        to_ec_adder_egress << ec_bucket;
        ec_bucket = from_bw_0.read();
        to_d_mult_egress << ec_bucket.x;
        to_ec_adder_egress << ec_bucket;

        ec_bucket = from_bw_1.read();
        to_d_mult_egress << ec_bucket.x;
        to_ec_adder_egress << ec_bucket;
        ec_bucket = from_bw_1.read();
        to_d_mult_egress << ec_bucket.x;
        to_ec_adder_egress << ec_bucket;
        ec_bucket = from_bw_1.read();
        to_d_mult_egress << ec_bucket.x;
        to_ec_adder_egress << ec_bucket;
    }

    static void point_cache(
        hls::stream<tagger_to_point_cache_t> &from_tagger,
        hls::stream<meta_t> &from_dispatch,
        hls::stream<point_tag_t> &to_tagger,
        hls::stream<bool> &dispatch_token_out,
        hls::stream<field_t> &to_d_mult_egress,
        hls::stream<operation_t> &to_ec_adder_egress
    ) {
#pragma HLS PIPELINE II=1
        static hls_thread_local point_t points[1024];
        static hls_thread_local ap_uint<3> access_cnts[1024] = {0};
        static hls_thread_local point_tag_t write_tag = INVALID_TAG;
        static hls_thread_local point_tag_cnt_t write_cnt = 0;

        // write port
        if (!from_tagger.empty()) {
            tagger_to_point_cache_t tmp = from_tagger.read();
            if (tmp.valid) {
                points[tmp.tag] = tmp.point;
            }
            dispatch_token_out << true;
        }

        // read port
        if (!from_dispatch.empty()) {
            const meta_t meta = from_dispatch.read();
            if (meta.tag != INVALID_TAG) {
                const point_t p = points[meta.tag];
                to_d_mult_egress << p.x;
                to_ec_adder_egress << operation_t{p, meta};
                point_tag_cnt_t cnt;
                if (write_tag == INVALID_TAG) {
                    cnt = access_cnts[meta.tag];
                } else if (write_tag != meta.tag) {
                    cnt = access_cnts[meta.tag];
                    access_cnts[write_tag] = write_cnt;
                } else {
                    cnt = write_cnt;
                }
                cnt = (cnt == meta.tag_cnt - 1) ? point_tag_cnt_t(0) : point_tag_cnt_t(cnt + 1);
                if (cnt == 0) {
                    to_tagger << meta.tag;
                }
                write_cnt = cnt;
                write_tag = meta.tag;
            } else {
                const point_t p(true);
                to_d_mult_egress << p.x;
                to_ec_adder_egress << operation_t{p, meta};
            }
        }
    }
};
