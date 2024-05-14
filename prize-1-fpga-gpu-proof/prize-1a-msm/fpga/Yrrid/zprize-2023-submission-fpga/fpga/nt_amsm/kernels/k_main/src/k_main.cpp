/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#include "k_main.h"

static constexpr char KERNEL_NAME[] = "k_main";
static constexpr char COMPONENT_NAME[] = "top";
static constexpr int KERNEL_ID = 0;
static constexpr int COMPONENT_ID = 0;

static void scalar_reader(
    const ap_uint<256> *scalars_mem,
    int msm_size,
    int full_batches,
    hls::stream<scalar_t> &out
) {
    int trailing = msm_size - PRE_SLICE_BATCH_SIZE * full_batches;
    for (int i = 0; i < full_batches * PRE_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=1
        out << scalars_mem[i];
    }

    for (int i = 0; i < trailing; i++) {
#pragma HLS PIPELINE II=1
        out << scalars_mem[full_batches * PRE_SLICE_BATCH_SIZE + i];
    }
}

static void point_reader(
    const ap_uint<256> *points_mem,
    int msm_size,
    int full_batches,
    hls::stream<ap_uint<256> > &out
) {
    int trailing = msm_size - PRE_SLICE_BATCH_SIZE * full_batches;
    for (int i = 0; i < 3 * full_batches * PRE_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=1
        out << points_mem[i];
    }

    for (int i = 0; i < 3 * trailing; i++) {
#pragma HLS PIPELINE II=1
        out << points_mem[3 * full_batches * PRE_SLICE_BATCH_SIZE + i];
    }
}

static void point_aggregator(
    int msm_size,
    hls::stream<ap_uint<256> > &in,
    hls::stream<point_t> &out
) {
    for (int i = 0; i < msm_size; i++) {
#pragma HLS PIPELINE II=3
        ap_uint<256> p_lo = in.read();
        ap_uint<256> p_mi = in.read();
        ap_uint<256> p_hi = in.read();
        out << point_t(p_hi, p_mi, p_lo);
    }
}

static void writer(
    ap_uint<256> *res,
    hls::stream<proc_to_main_flit_t> &from_proc_0,
    hls::stream<proc_to_main_flit_t> &from_proc_1,
    hls::stream<proc_to_main_flit_t> &from_proc_2
) {
    for (int i = 0; i < 3 * 7 * 4096; i++) {
#pragma HLS PIPELINE II=1
        res[i] = from_proc_0.read();
    }

    for (int i = 0; i < 3 * 7 * 4096; i++) {
#pragma HLS PIPELINE II=1
        res[i + 3 * N_PROC0_RESULTS] = from_proc_1.read();
    }

    for (int i = 0; i < 3 * 7 * 4096; i++) {
#pragma HLS PIPELINE II=1
        res[i + 3 * (N_PROC0_RESULTS + N_PROC1_RESULTS)] = from_proc_2.read();
    }
}

static void flow_controller(
    int msm_size,
    int full_batches,
    hls::stream<scalar_t> &scalars_in,
    hls::stream<point_t> &points_in,
    hls::stream<proc_status_t> &from_proc_0_status,
    hls::stream<proc_status_t> &from_proc_1_status,
    hls::stream<proc_status_t> &from_proc_2_status,
    hls::stream<main_to_proc_flit_t> &to_proc_0,
    hls::stream<main_to_proc_flit_t> &to_proc_1,
    hls::stream<main_to_proc_flit_t> &to_proc_2
) {
    const unsigned remainder = msm_size - PRE_SLICE_BATCH_SIZE * full_batches;
    const unsigned padding = remainder == 0 ? 0 : (PRE_SLICE_BATCH_SIZE - remainder);
    unsigned n = 0;
    bool throttle = false;
    while (n < msm_size + padding) {
        if (!throttle) {
            for (int i = 0; i < PRE_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=3
                const bool report = i == 12;
                scalar_t scalar = 0;
                point_t p(true);
                if (n < msm_size) {
                    scalar = scalars_in.read();
                    p = points_in.read();
                }
                const bool valid = scalar != 0;
                to_proc_0 << main_to_proc_t(scalar, valid, report, false).to_flit();
                to_proc_1 << main_to_proc_t(scalar, valid, report, false).to_flit();
                to_proc_2 << main_to_proc_t(scalar, valid, report, false).to_flit();
                to_proc_0 << main_to_proc_t(p.x).to_flit();
                to_proc_1 << main_to_proc_t(p.x).to_flit();
                to_proc_2 << main_to_proc_t(p.x).to_flit();
                to_proc_0 << main_to_proc_t(p.y).to_flit();
                to_proc_1 << main_to_proc_t(p.y).to_flit();
                to_proc_2 << main_to_proc_t(p.y).to_flit();
                n++;
            }
            CSIM_INFO_ARGS("POINTS CONSUMED: {}", n);
        } else {
            for (int i = 0; i < PRE_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=3
                bool report = i == 12;
                point_t inf_point(true);
                to_proc_0 << main_to_proc_t(0, false, report, false).to_flit();
                to_proc_1 << main_to_proc_t(0, false, report, false).to_flit();
                to_proc_2 << main_to_proc_t(0, false, report, false).to_flit();
                to_proc_0 << main_to_proc_t(inf_point.x).to_flit();
                to_proc_1 << main_to_proc_t(inf_point.x).to_flit();
                to_proc_2 << main_to_proc_t(inf_point.x).to_flit();
                to_proc_0 << main_to_proc_t(inf_point.y).to_flit();
                to_proc_1 << main_to_proc_t(inf_point.y).to_flit();
                to_proc_2 << main_to_proc_t(inf_point.y).to_flit();
            }
            CSIM_INFO("FLUSHING");
        }
        proc_status_t status_0 = from_proc_0_status.read();
        proc_status_t status_1 = from_proc_1_status.read();
        proc_status_t status_2 = from_proc_2_status.read();
        throttle = (status_0 == PROC_STATUS_THROTTLE) | (status_1 == PROC_STATUS_THROTTLE) | (
                       status_2 == PROC_STATUS_THROTTLE);
        CSIM_USLEEP(100000);
    }

    bool flushed = false;
    while (!flushed) {
        for (int i = 0; i < PRE_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=3
            bool end_of_batch = i == 12;
            point_t inf_point(true);
            to_proc_0 << main_to_proc_t(0, false, end_of_batch, false).to_flit();
            to_proc_1 << main_to_proc_t(0, false, end_of_batch, false).to_flit();
            to_proc_2 << main_to_proc_t(0, false, end_of_batch, false).to_flit();
            to_proc_0 << main_to_proc_t(inf_point.x).to_flit();
            to_proc_1 << main_to_proc_t(inf_point.x).to_flit();
            to_proc_2 << main_to_proc_t(inf_point.x).to_flit();
            to_proc_0 << main_to_proc_t(inf_point.y).to_flit();
            to_proc_1 << main_to_proc_t(inf_point.y).to_flit();
            to_proc_2 << main_to_proc_t(inf_point.y).to_flit();
        }
        CSIM_INFO("FLUSHING");
        proc_status_t status_0 = from_proc_0_status.read();
        proc_status_t status_1 = from_proc_1_status.read();
        proc_status_t status_2 = from_proc_2_status.read();
        flushed = (status_0 == PROC_STATUS_FLUSHED) && (status_1 == PROC_STATUS_FLUSHED) && (
                      status_2 == PROC_STATUS_FLUSHED);
        CSIM_USLEEP(100000);
    }

    for (int i = 0; i < PRE_SLICE_BATCH_SIZE; i++) {
#pragma HLS PIPELINE II=3
        point_t inf_point(true);
        to_proc_0 << main_to_proc_t(0, false, false, i == PRE_SLICE_BATCH_SIZE - 1).to_flit();
        to_proc_1 << main_to_proc_t(0, false, false, i == PRE_SLICE_BATCH_SIZE - 1).to_flit();
        to_proc_2 << main_to_proc_t(0, false, false, i == PRE_SLICE_BATCH_SIZE - 1).to_flit();
        to_proc_0 << main_to_proc_t(inf_point.x).to_flit();
        to_proc_1 << main_to_proc_t(inf_point.x).to_flit();
        to_proc_2 << main_to_proc_t(inf_point.x).to_flit();
        to_proc_0 << main_to_proc_t(inf_point.y).to_flit();
        to_proc_1 << main_to_proc_t(inf_point.y).to_flit();
        to_proc_2 << main_to_proc_t(inf_point.y).to_flit();
    }
}

void k_main(
    const ap_uint<256> *scalars_mem,
    const ap_uint<256> *points_mem,
    ap_uint<256> *res_mem,
    int msm_size,
    int full_batches,
    hls::stream<proc_status_t> &from_proc_0_status,
    hls::stream<proc_status_t> &from_proc_1_status,
    hls::stream<proc_status_t> &from_proc_2_status,
    hls::stream<proc_to_main_flit_t> &from_proc_0_data,
    hls::stream<proc_to_main_flit_t> &from_proc_1_data,
    hls::stream<proc_to_main_flit_t> &from_proc_2_data,
    hls::stream<main_to_proc_flit_t> &to_proc_0,
    hls::stream<main_to_proc_flit_t> &to_proc_1,
    hls::stream<main_to_proc_flit_t> &to_proc_2
) {
#pragma HLS INTERFACE mode=s_axilite port=return
#pragma HLS INTERFACE mode=s_axilite port=msm_size
#pragma HLS INTERFACE mode=s_axilite port=full_batches
#pragma HLS INTERFACE mode=m_axi port=scalars_mem bundle=gmem channel=0
#pragma HLS INTERFACE mode=m_axi port=points_mem bundle=gmem channel=1
#pragma HLS INTERFACE mode=m_axi port=res_mem bundle=gmem channel=0 max_write_burst_length=256
#pragma HLS INTERFACE mode=axis port=from_proc_0_status
#pragma HLS INTERFACE mode=axis port=from_proc_1_status
#pragma HLS INTERFACE mode=axis port=from_proc_2_status
#pragma HLS INTERFACE mode=axis port=from_proc_0_data
#pragma HLS INTERFACE mode=axis port=from_proc_1_data
#pragma HLS INTERFACE mode=axis port=from_proc_2_data
#pragma HLS INTERFACE mode=axis port=to_proc_0
#pragma HLS INTERFACE mode=axis port=to_proc_1
#pragma HLS INTERFACE mode=axis port=to_proc_2
#pragma HLS AGGREGATE variable=from_proc_0_status compact=byte
#pragma HLS AGGREGATE variable=from_proc_1_status compact=byte
#pragma HLS AGGREGATE variable=from_proc_2_status compact=byte
#pragma HLS AGGREGATE variable=from_proc_0_data compact=byte
#pragma HLS AGGREGATE variable=from_proc_1_data compact=byte
#pragma HLS AGGREGATE variable=from_proc_2_data compact=byte
#pragma HLS AGGREGATE variable=to_proc_0 compact=byte
#pragma HLS AGGREGATE variable=to_proc_1 compact=byte
#pragma HLS AGGREGATE variable=to_proc_2 compact=byte
#pragma HLS DATAFLOW
    hls::stream<scalar_t, 512> s_scalar_reader_to_flow_controller;
    hls::stream<ap_uint<256>, 512> s_point_reader_to_point_aggregator;
    hls::stream<field_t, 2> s_egress_to_writer;
    hls::stream<point_t, 2> s_point_aggregator_to_flow_controller;

    CSIM_USLEEP(10000000);
    scalar_reader(scalars_mem, msm_size, full_batches, s_scalar_reader_to_flow_controller);
    point_reader(points_mem, msm_size, full_batches, s_point_reader_to_point_aggregator);
    point_aggregator(msm_size, s_point_reader_to_point_aggregator, s_point_aggregator_to_flow_controller);
    flow_controller(msm_size, full_batches, s_scalar_reader_to_flow_controller, s_point_aggregator_to_flow_controller,
                    from_proc_0_status, from_proc_1_status,
                    from_proc_2_status, to_proc_0, to_proc_1, to_proc_2);
    writer(res_mem, from_proc_0_data, from_proc_1_data, from_proc_2_data);
}
