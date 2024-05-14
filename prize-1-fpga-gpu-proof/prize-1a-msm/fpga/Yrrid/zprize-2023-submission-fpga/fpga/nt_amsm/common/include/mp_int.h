/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once
#include <ap_int.h>
#include <hls_vector.h>

template <int LIMB_SIZE, int N_TOTAL_BITS>
struct mp_int_t {
    static_assert(N_TOTAL_BITS % LIMB_SIZE == 0);
    static constexpr int N_LIMBS = N_TOTAL_BITS / LIMB_SIZE;

    hls::vector<ap_uint<LIMB_SIZE>, N_LIMBS> limbs = ap_uint<LIMB_SIZE>(0);

    mp_int_t() = default;

    explicit mp_int_t(ap_uint<N_TOTAL_BITS> x) {
        for (int i = 0; i < N_LIMBS; i++) {
#pragma HLS UNROLL
            limbs[i] = x(LIMB_SIZE * (i + 1) - 1, LIMB_SIZE * i);
        }
    }

    void from_ap_uint(ap_uint<N_TOTAL_BITS> x) {
        for (int i = 0; i < N_LIMBS; i++) {
#pragma HLS UNROLL
            limbs[i] = x(LIMB_SIZE * (i + 1) - 1, LIMB_SIZE * i);
        }
    }

    ap_uint<N_TOTAL_BITS> to_ap_uint() {
        ap_uint<N_TOTAL_BITS> tmp;
        for (int i = 0; i < N_LIMBS; i++) {
#pragma HLS UNROLL
            tmp(LIMB_SIZE * (i + 1) - 1, LIMB_SIZE * i) = limbs[i];
        }
        return tmp;
    }

    ap_uint<LIMB_SIZE>& operator[](int i) {
        return limbs[i];
    }

    const ap_uint<LIMB_SIZE>& operator[](int i) const {
        return limbs[i];
    }

    [[nodiscard]] bool is_even() const {
        return limbs[0][0] == 0;
    }

    void shift_left(int x) {
        ap_uint<N_TOTAL_BITS> tmp = to_ap_uint();
        tmp <<= x;
        from_ap_uint(tmp);
    }

    void shift_right(int x) {
        ap_uint<N_TOTAL_BITS> tmp = to_ap_uint();
        tmp >>= x;
        from_ap_uint(tmp);
    }

    static bool fast_add_no_red_inline(mp_int_t &r, const mp_int_t &a, const mp_int_t &b) {
#pragma HLS INLINE
        mp_int_t r_plus;
        ap_uint<N_LIMBS> g, p;

        for (int i = 0; i < N_LIMBS; i++) {
#pragma HLS UNROLL
            ap_uint<LIMB_SIZE + 1> tmp;
            tmp = a[i] + b[i];
            g[i] = tmp[LIMB_SIZE];
            r[i] = tmp;
            tmp = a[i] + b[i] + 1;
            p[i] = tmp[LIMB_SIZE];
            r_plus[i] = tmp;
        }

        ap_uint<N_LIMBS + 1> sel = (g + p) ^ (g ^ p);

        for (int i = 0; i < N_LIMBS; i++) {
#pragma HLS UNROLL
            r[i] = (sel[i] == 0) ? r[i] : r_plus[i];
        }

        return sel[N_LIMBS];
    }

    static bool fast_sub_no_red_inline(mp_int_t &r, const mp_int_t &a, const mp_int_t &b) {
#pragma HLS INLINE
        mp_int_t r_minus;
        ap_uint<N_LIMBS> g, p;

        for (int i = 0; i < N_LIMBS; i++) {
#pragma HLS UNROLL
            ap_uint<LIMB_SIZE + 1> tmp;
            tmp = a[i] + ~b[i];
            g[i] = tmp[LIMB_SIZE];
            r[i] = tmp;
            tmp = a[i] + ~b[i] + 1;
            p[i] = tmp[LIMB_SIZE];
            r_minus[i] = tmp;
        }

        ap_uint<N_LIMBS + 1> sel = (g + p + 1) ^ (g ^ p);

        for (int i = 0; i < N_LIMBS; i++) {
#pragma HLS UNROLL
            r[i] = (sel[i] == 0) ? r[i] : r_minus[i];
        }

        return sel[N_LIMBS];
    }
};

using mp_16_384_t = mp_int_t<16, 384>;
using mp_24_384_t = mp_int_t<24, 384>;
using mp_32_384_t = mp_int_t<32, 384>;
using mp_48_384_t = mp_int_t<48, 384>;
using mp_64_384_t = mp_int_t<64, 384>;