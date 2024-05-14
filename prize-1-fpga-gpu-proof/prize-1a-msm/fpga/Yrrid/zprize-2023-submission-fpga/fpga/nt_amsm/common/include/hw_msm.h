/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#define AP_INT_MAX_W 4096

#include <hls_task.h>
#include "mp_int.h"
#include "bls12.h"
#include "csim_utils.h"

static constexpr int PRE_SLICE_BATCH_SIZE = 48;
static constexpr int POST_SLICE_BATCH_SIZE = 336;
static constexpr int INVERTER_BATCH_SIZE = 42;

static constexpr int BUCKET_IDX_SIZE = 12;
static constexpr int N_WINDOWS = 21;
static constexpr int N_BUCKETS = (1 << BUCKET_IDX_SIZE);
static constexpr int N_PROC0_RESULTS = N_BUCKETS * 7;
static constexpr int N_PROC1_RESULTS = N_BUCKETS * 7;
static constexpr int N_PROC2_RESULTS = N_BUCKETS * 7;

using bool_t = ap_uint<1>;
using scalar_t = ap_uint<256>;
using field_t = ap_uint<384>;
using window_idx_t = ap_uint<5>;
using bucket_idx_t = ap_uint<13>;
using point_idx_t = ap_uint<32>;

using main_to_proc_flit_t = ap_uint<384>;
using proc_to_main_flit_t = ap_uint<256>;
using proc_to_d_mult_flit_t = ap_uint<768>;
using ec_adder_to_proc_flit_t = ap_uint<840>;
using proc_to_ec_adder_flit_t = ap_uint<1608>;

using point_tag_t = ap_uint<11>;

using proc_status_t = ap_uint<8>;

static constexpr int PROC_STATUS_THROTTLE = 0;
static constexpr int PROC_STATUS_NORMAL = 1;
static constexpr int PROC_STATUS_FLUSHED = 2;

static const bucket_idx_t INVALID_BUCKET = -1;

struct point_t {
    field_t x;
    field_t y;

    point_t() = default;

    point_t(ap_uint<768> raw) {
        x = raw(383, 0);
        y = raw(767, 384);
    }

    point_t(ap_uint<256> hi, ap_uint<256> mi, ap_uint<256> lo) {
        x(255, 0) = lo;
        x(383, 256) = mi(127, 0);
        y(127, 0) = mi(255, 128);
        y(383, 128) = hi;
    }

    point_t(const field_t &p_x, const field_t &p_y, bool infty = false) {
        x = p_x;
        y = p_y;
        set_inf(infty);
    }

    explicit point_t(bool inf) {
        set_inf(inf);
    }

    void set_inf(bool is_inf) {
        x[383] = is_inf;
    }

    bool is_inf() {
        return x[383];
    }
};

using point_tag_cnt_t = ap_uint<3>;

struct meta_t {
    point_idx_t point_idx; // 32b
    window_idx_t window_idx; // 5b
    bucket_idx_t bucket_idx; // 13b
    point_tag_t tag; // 11b
    point_tag_cnt_t tag_cnt; // 3b
    bool_t sub; // 1b
    bool_t report; // 1b
    bool_t valid; // 1b
    bool_t last; // 1b
};

struct operation_t {
    point_t point;
    meta_t meta;

    operation_t() = default;

    operation_t(point_t point, meta_t meta) : point(point), meta(meta) {}

    explicit operation_t(ec_adder_to_proc_flit_t flit) {
        (
            meta.last,
            meta.valid,
            meta.report,
            meta.sub,
            meta.tag_cnt,
            meta.tag,
            meta.bucket_idx,
            meta.window_idx,
            meta.point_idx,
            point.y,
            point.x
        ) = flit;
    }

    ec_adder_to_proc_flit_t to_flit() {
        ec_adder_to_proc_flit_t flit = (
            meta.last,
            meta.valid,
            meta.report,
            meta.sub,
            meta.tag_cnt,
            meta.tag,
            meta.bucket_idx,
            meta.window_idx,
            meta.point_idx,
            point.y,
            point.x
        );
        return flit;
    }
};

struct main_to_proc_t {
    using cmd_t = ap_uint<8>;
    static constexpr int CMD_VALID_BIT = 0;
    static constexpr int CMD_REPORT_BIT = 1;
    static constexpr int CMD_LAST_BIT = 2;

    field_t data;

    main_to_proc_t() = default;

    explicit main_to_proc_t(field_t field) : data(field) {
    }

    explicit main_to_proc_t(scalar_t scalar, bool valid, bool report, bool last) {
        ap_uint<128> cmd = 0;
        cmd[CMD_VALID_BIT] = valid;
        cmd[CMD_REPORT_BIT] = report;
        cmd[CMD_LAST_BIT] = last;
        data = (cmd, scalar);
    }

    bool valid() {
        return data[256 + CMD_VALID_BIT];
    }

    bool report() {
        return data[256 + CMD_REPORT_BIT];
    }

    bool last() {
        return data[256 + CMD_LAST_BIT];
    }

    main_to_proc_flit_t to_flit() const {
        return data;
    }
};

// struct meta_t {
//     point_idx_t point_idx; // 32b
//     window_idx_t window_idx; // 5b
//     bucket_idx_t bucket_idx; // 13b
//     point_tag_t tag; // 11b
//     bool_t sub; // 1b
//     bool_t report; // 1b
//     bool_t valid; // 1b
//     bool_t last; // 1b
// };

struct proc_to_ec_adder_t {
    operation_t ec_point;
    point_t ec_bucket;

    proc_to_ec_adder_t() = default;

    proc_to_ec_adder_t(operation_t ec_point, point_t ec_bucket) : ec_point(ec_point), ec_bucket(ec_bucket) {
    }

    explicit proc_to_ec_adder_t(proc_to_ec_adder_flit_t flit) {
        (
            ec_point.meta.last,
            ec_point.meta.valid,
            ec_point.meta.report,
            ec_point.meta.sub,
            ec_point.meta.tag_cnt,
            ec_point.meta.tag,
            ec_point.meta.bucket_idx,
            ec_point.meta.window_idx,
            ec_point.meta.point_idx,
            ec_bucket.y,
            ec_bucket.x,
            ec_point.point.y,
            ec_point.point.x
        ) = flit;
    }

    proc_to_ec_adder_flit_t to_flit() {
        proc_to_ec_adder_flit_t flit = (
            ec_point.meta.last,
            ec_point.meta.valid,
            ec_point.meta.report,
            ec_point.meta.sub,
            ec_point.meta.tag_cnt,
            ec_point.meta.tag,
            ec_point.meta.bucket_idx,
            ec_point.meta.window_idx,
            ec_point.meta.point_idx,
            ec_bucket.y,
            ec_bucket.x,
            ec_point.point.y,
            ec_point.point.x
        );
        return flit;
    }
};

template<bls12_t CURVE>
field_t add_red_inline(const field_t &a, const field_t &b) {
#pragma HLS INLINE
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    assert(a < base_field_t<CURVE>::P);
    assert(b < base_field_t<CURVE>::P);
#endif
    field_t sum = a + b;
#pragma HLS BIND_OP variable=sum op=add impl=fabric latency=3
    field_t fix = sum - base_field_t<CURVE>::P;
#pragma HLS BIND_OP variable=fix op=sub impl=fabric latency=3
    if (fix[383]) {
        return sum;
    } else {
        return fix;
    }
}

template<bls12_t CURVE>
field_t add_red_mont_inline(const field_t &a, const field_t &b) {
#pragma HLS INLINE
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P2);
    //    assert(b < base_field_t<CURVE>::P2);
#endif
    field_t sum = a + b;
#pragma HLS BIND_OP variable=sum op=add impl=fabric latency=3
    field_t fix = sum - base_field_t<CURVE>::P2;
#pragma HLS BIND_OP variable=fix op=sub impl=fabric latency=3
    if (fix[383]) {
        return sum;
    } else {
        return fix;
    }
}

template<bls12_t CURVE>
field_t add_red_mont_no_inline(const field_t &a, const field_t &b) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P2);
    //    assert(b < base_field_t<CURVE>::P2);
#endif
    field_t sum = a + b;
#pragma HLS BIND_OP variable=sum op=add impl=fabric latency=2
    field_t fix = sum - base_field_t<CURVE>::P2;
#pragma HLS BIND_OP variable=fix op=sub impl=fabric latency=2
    if (fix[383]) {
        return sum;
    } else {
        return fix;
    }
}

template<bls12_t CURVE>
field_t sub_red_inline(const field_t &a, const field_t &b) {
#pragma HLS INLINE
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P);
    //    assert(b < base_field_t<CURVE>::P);
#endif
    field_t diff = a - b;
#pragma HLS BIND_OP variable=diff op=sub impl=fabric latency=3
    field_t fix = diff + base_field_t<CURVE>::P;
#pragma HLS BIND_OP variable=fix op=add impl=fabric latency=3
    if (diff[383]) {
        return fix;
    } else {
        return diff;
    }
}

template<bls12_t CURVE>
field_t sub_red_mont_inline(const field_t &a, const field_t &b) {
#pragma HLS INLINE
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P2);
    //    assert(b < base_field_t<CURVE>::P2);
#endif
    field_t diff = a - b;
#pragma HLS BIND_OP variable=diff op=sub impl=fabric latency=3
    field_t fix = diff + base_field_t<CURVE>::P2;
#pragma HLS BIND_OP variable=fix op=add impl=fabric latency=3
    if (diff[383]) {
        return fix;
    } else {
        return diff;
    }
}

template<bls12_t CURVE>
field_t sub_red_mont_no_inline(const field_t &a, const field_t &b) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P2);
    //    assert(b < base_field_t<CURVE>::P2);
#endif
    field_t diff = a - b;
#pragma HLS BIND_OP variable=diff op=sub impl=fabric latency=2
    field_t fix = diff + base_field_t<CURVE>::P2;
#pragma HLS BIND_OP variable=fix op=add impl=fabric latency=2
    if (diff[383]) {
        return fix;
    } else {
        return diff;
    }
}

template<bls12_t CURVE>
field_t negate_inline(const field_t &a) {
#pragma HLS INLINE
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P);
#endif
    return sub_red_inline<CURVE>(0, a);
}

template<bls12_t CURVE>
field_t red_mont_inline(const field_t &a) {
#pragma HLS INLINE
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P2);
#endif
    field_t red = a - base_field_t<CURVE>::P;
#pragma HLS BIND_OP variable=red op=sub impl=fabric latency=3
    if (red[383]) {
        return a;
    } else {
        return red;
    }
}

template<bls12_t CURVE>
field_t red_mont_no_inline(const field_t &a) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#if !defined(__SYNTHESIS__) && !defined(__XOR_STUB__)
    //    assert(a < base_field_t<CURVE>::P2);
#endif
    field_t red = a - base_field_t<CURVE>::P;
#pragma HLS BIND_OP variable=red op=sub impl=fabric latency=2
    if (red[383]) {
        return a;
    } else {
        return red;
    }
}

template<typename T>
void round_robin_1_to_2(
    hls::stream<T> &in,
    hls::stream<T> &out_0,
    hls::stream<T> &out_1
) {
#pragma HLS PIPELINE II=2
    T tmp = in.read();
    out_0 << tmp;
    tmp = in.read();
    out_1 << tmp;
}

template<typename T>
void round_robin_2_to_1(
    hls::stream<T> &in_0,
    hls::stream<T> &in_1,
    hls::stream<T> &out
) {
#pragma HLS PIPELINE II=2
    T tmp = in_0.read();
    out << tmp;
    tmp = in_1.read();
    out << tmp;
}

template<typename T>
void round_robin_4_to_1(
    hls::stream<T> &in_0,
    hls::stream<T> &in_1,
    hls::stream<T> &in_2,
    hls::stream<T> &in_3,
    hls::stream<T> &out
) {
#pragma HLS PIPELINE II=4
    T tmp = in_0.read();
    out << tmp;
    tmp = in_1.read();
    out << tmp;
    tmp = in_2.read();
    out << tmp;
    tmp = in_3.read();
    out << tmp;
}

template<typename T>
void round_robin_1_to_7(
    hls::stream<T> &in,
    hls::stream<T> &out_0,
    hls::stream<T> &out_1,
    hls::stream<T> &out_2,
    hls::stream<T> &out_3,
    hls::stream<T> &out_4,
    hls::stream<T> &out_5,
    hls::stream<T> &out_6
) {
#pragma HLS PIPELINE II=7
    T tmp = in.read();
    out_0 << tmp;
    tmp = in.read();
    out_1 << tmp;
    tmp = in.read();
    out_2 << tmp;
    tmp = in.read();
    out_3 << tmp;
    tmp = in.read();
    out_4 << tmp;
    tmp = in.read();
    out_5 << tmp;
    tmp = in.read();
    out_6 << tmp;
}


template<typename T>
void round_robin_7_to_1(
    hls::stream<T> &in_0,
    hls::stream<T> &in_1,
    hls::stream<T> &in_2,
    hls::stream<T> &in_3,
    hls::stream<T> &in_4,
    hls::stream<T> &in_5,
    hls::stream<T> &in_6,
    hls::stream<T> &out
) {
#pragma HLS PIPELINE II=7
    T tmp = in_0.read();
    out << tmp;
    tmp = in_1.read();
    out << tmp;
    tmp = in_2.read();
    out << tmp;
    tmp = in_3.read();
    out << tmp;
    tmp = in_4.read();
    out << tmp;
    tmp = in_5.read();
    out << tmp;
    tmp = in_6.read();
    out << tmp;
}

template<typename T>
void broadcast_1_to_7(
    hls::stream<T> &in,
    hls::stream<T> &out_0,
    hls::stream<T> &out_1,
    hls::stream<T> &out_2,
    hls::stream<T> &out_3,
    hls::stream<T> &out_4,
    hls::stream<T> &out_5,
    hls::stream<T> &out_6
) {
#pragma HLS PIPELINE II=1
    T tmp = in.read();
    out_0 << tmp;
    out_1 << tmp;
    out_2 << tmp;
    out_3 << tmp;
    out_4 << tmp;
    out_5 << tmp;
    out_6 << tmp;
}

template<typename T, int DELAY = 1>
T pipe_this(T x) {
#pragma HLS INLINE off
#pragma HLS LATENCY min=DELAY max=DELAY
#pragma HLS PIPELINE II=1
    return x;
}

template<typename T>
void connector(hls::stream<T> &in, hls::stream<T> &out) {
#pragma HLS PIPELINE II=1 style=frp
    out << in.read();
}
