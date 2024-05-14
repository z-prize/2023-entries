/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#include "hw_msm.h"
#include "toom_cook_mult.h"

template<bls12_t CURVE>
class mod_mult {
public:
    static field_t ll_mod_mult(field_t a_in, field_t b_in) {
#ifndef __SYNTHESIS__a
        if (a_in >= base_field_t<CURVE>::P2) {
            CSIM_ERROR_NO_LABEL("ll_mod_mult: a_in >= P2");
        }
        if (b_in >= base_field_t<CURVE>::P2) {
            CSIM_ERROR_NO_LABEL("ll_mod_mult: b_in >= P2");
        }
#endif
        hls::vector<ap_uint<48>, 9> a, b;
        a[0] = a_in(47, 0);
        a[1] = a_in(95, 48);
        a[2] = a_in(143, 96);
        a[3] = a_in(191, 144);
        a[4] = a_in(239, 192);
        a[5] = a_in(287, 240);
        a[6] = a_in(335, 288);
        a[7] = a_in(383, 336);
        a[8] = 0;
        b[0] = b_in(47, 0);
        b[1] = b_in(95, 48);
        b[2] = b_in(143, 96);
        b[3] = b_in(191, 144);
        b[4] = b_in(239, 192);
        b[5] = b_in(287, 240);
        b[6] = b_in(335, 288);
        b[7] = b_in(383, 336);
        b[8] = 0;

        hi_lo_t ab = ab_mult(a, b);
        hls::vector<ap_uint<49>, 9> lo_pr = partial_resolve(ab.lo);
        hls::vector<ap_uint<48>, 9> lo_fr = full_resolve(lo_pr);
        hls::vector<long, 9> q = low_reduce(lo_fr);
        hls::vector<ap_uint<49>, 9> q_pr = partial_resolve(q);
        hls::vector<long, 9> final = high_reduce(ab.hi, lo_fr, q_pr);
        hls::vector<ap_uint<49>, 9> final_pr = partial_resolve(final);
        hls::vector<ap_uint<48>, 9> final_fr = full_resolve(final_pr);
        ap_uint<384> res = (final_fr[7], final_fr[6], final_fr[5], final_fr[4], final_fr[3], final_fr[2], final_fr[1], final_fr[0]);
        return res;
    }

    static field_t ma_mod_mult(field_t a_in, field_t b_in) {
#ifndef __SYNTHESIS__
        if (a_in >= base_field_t<CURVE>::P2) {
            CSIM_ERROR_NO_LABEL("la_mod_mult: a_in >= P2");
        }
        if (b_in >= base_field_t<CURVE>::P2) {
            CSIM_ERROR_NO_LABEL("la_mod_mult: b_in >= P2");
        }
#endif
        hls::vector<ap_uint<48>, 9> a, b;
        a[0] = a_in(47, 0);
        a[1] = a_in(95, 48);
        a[2] = a_in(143, 96);
        a[3] = a_in(191, 144);
        a[4] = a_in(239, 192);
        a[5] = a_in(287, 240);
        a[6] = a_in(335, 288);
        a[7] = a_in(383, 336);
        a[8] = 0;
        b[0] = b_in(47, 0);
        b[1] = b_in(95, 48);
        b[2] = b_in(143, 96);
        b[3] = b_in(191, 144);
        b[4] = b_in(239, 192);
        b[5] = b_in(287, 240);
        b[6] = b_in(335, 288);
        b[7] = b_in(383, 336);
        b[8] = 0;

        hi_lo_t ab = ab_mult(a, b);
        hls::vector<long, 16> ab_vec;
        for (int i = 0; i < 8; i++) {
            ab_vec[i] = ab.lo[i];
            ab_vec[i + 8] = ab.hi[i];
        }
        hls::vector<long, 9> reduced = slow_reduce(ab_vec);
        hls::vector<ap_uint<49>, 9> pr = partial_resolve(reduced);
        hls::vector<ap_uint<48>, 9> fr = full_resolve(pr);
        ap_uint<384> res = (fr[7], fr[6], fr[5], fr[4], fr[3], fr[2], fr[1], fr[0]);
        return res;
    }

    static field_t la_mod_mult(field_t a_in, field_t b_in) {
#ifndef __SYNTHESIS__
        if (a_in >= base_field_t<CURVE>::P2) {
            CSIM_ERROR_NO_LABEL("la_mod_mult: a_in >= P2");
        }
        if (b_in >= base_field_t<CURVE>::P2) {
            CSIM_ERROR_NO_LABEL("la_mod_mult: b_in >= P2");
        }
#endif
#ifndef __CSIM_SAVE_STACK__
        field_t a = a_in;
        field_t b = b_in;
        if (a_in(383, 372) > 0x1a3) {
            a = a_in - base_field_t<CURVE>::P;
#pragma HLS BIND_OP variable=a_in op=sub impl=fabric latency=2
        }
        if (b_in(383, 372) > 0x1a3) {
            b = b_in - base_field_t<CURVE>::P;
#pragma HLS BIND_OP variable=b_in op=sub impl=fabric latency=2
        }

        hls::vector<long, 16> ab = toom_cook_mult::mult(a, b);
        hls::vector<long, 9> reduced = slow_reduce(ab);
        hls::vector<ap_uint<49>, 9> pr = partial_resolve(reduced);
        hls::vector<ap_uint<48>, 9> fr = full_resolve(pr);
        ap_uint<384> res = (fr[7], fr[6], fr[5], fr[4], fr[3], fr[2], fr[1], fr[0]);
#else
        ap_uint<384> res = ll_mod_mult(a_in, b_in);
#endif
        return res;
    }

    inline static const mp_48_384_t NP = mp_48_384_t(base_field_t<CURVE>::NP);
    inline static const mp_48_384_t P = mp_48_384_t(base_field_t<CURVE>::P);

    struct hi_lo_t {
        hls::vector<long, 9> hi;
        hls::vector<long, 9> lo;
    };

    static ap_uint<56> hi(ap_uint<102> x) {
#pragma HLS INLINE
        return x(101, 48);
    }

    static ap_uint<56> lo(ap_uint<102> x) {
#pragma HLS INLINE
        return x(47, 0);
    }

    static long mult_very_high(ap_uint<49> a, ap_uint<48> b) {
#pragma HLS INLINE
        ap_uint<14> a0 = a >> 36;
        ap_uint<12> b0 = b >> 36;
        long tmp = (long) (a0 * b0);
        return tmp << 24;
    }

    static ap_uint<96> p_0_term_48(ap_uint<48> a) {
#pragma HLS INLINE
        if constexpr (CURVE == bls12_381) {
            ap_uint<96> bi = a;
            ap_uint<96> v55 = bi + (bi << 2);
            v55 = v55 + (v55 << 4);
            ap_uint<96> v5555 = v55 + (v55 << 8);
            return (ap_uint<128>(bi) << 48) - v5555;
        } else {
            ap_uint<128> v = a;
            ap_uint<128> v3 = (v << 2) - v;
            return (v3 << 46) + v;
        }
    }

    static ap_uint<96> np_0_term_48(ap_uint<48> a) {
#pragma HLS INLINE
        if constexpr (CURVE == bls12_381) {
            ap_uint<96> bi = a;
            ap_uint<96> v333 = bi + (bi << 1) + (bi << 16) + (bi << 17) + (bi << 32) + (bi << 33);
            return (bi << 48) - v333;
        } else {
            ap_uint<128> v = a;
            return (v << 48) - (v << 46) - v;
        }
    }

    static ap_uint<48> q_0_term_48(ap_uint<48> a) {
#pragma HLS INLINE
        return np_0_term_48(a);
    }

    static ap_uint<102> mult_51(ap_uint<51> a, ap_uint<51> b) {
#pragma HLS INLINE
        ap_uint<26> a0 = a(25, 0);
        ap_uint<25> a1 = a(50, 26);

        ap_uint<77> a0b = a0 * b;
        ap_uint<76> a1b = a1 * b;

        ap_uint<102> res = ap_uint<102>(a0b) + (ap_uint<102>(a1b) << 26);
        return res;
    }

    static hi_lo_t ab_mult(const hls::vector<ap_uint<48>, 9> &a_in, const hls::vector<ap_uint<48>, 9> &b_in) {
        ap_uint<102> p[8];
        p[0] = mult_51(a_in[0], b_in[0]);
        p[1] = mult_51(a_in[1], b_in[1]);
        p[2] = mult_51(a_in[2], b_in[2]);
        p[3] = mult_51(a_in[3], b_in[3]);
        p[4] = mult_51(a_in[4], b_in[4]);
        p[5] = mult_51(a_in[5], b_in[5]);
        p[6] = mult_51(a_in[6], b_in[6]);
        p[7] = mult_51(a_in[7], b_in[7]);

        ap_uint<49> as[4], bs[4], als[2], ahs[2], bls[2], bhs[2];
        as[0] = a_in[0] + a_in[4];
        as[1] = a_in[1] + a_in[5];
        as[2] = a_in[2] + a_in[6];
        as[3] = a_in[3] + a_in[7];
        bs[0] = b_in[0] + b_in[4];
        bs[1] = b_in[1] + b_in[5];
        bs[2] = b_in[2] + b_in[6];
        bs[3] = b_in[3] + b_in[7];
        als[0] = a_in[0] + a_in[2];
        als[1] = a_in[1] + a_in[3];
        ahs[0] = a_in[4] + a_in[6];
        ahs[1] = a_in[5] + a_in[7];
        bls[0] = b_in[0] + b_in[2];
        bls[1] = b_in[1] + b_in[3];
        bhs[0] = b_in[4] + b_in[6];
        bhs[1] = b_in[5] + b_in[7];
        ap_uint<49> alls = a_in[0] + a_in[1];
        ap_uint<49> alhs = a_in[2] + a_in[3];
        ap_uint<49> ahls = a_in[4] + a_in[5];
        ap_uint<49> ahhs = a_in[6] + a_in[7];
        ap_uint<49> blls = b_in[0] + b_in[1];
        ap_uint<49> blhs = b_in[2] + b_in[3];
        ap_uint<49> bhls = b_in[4] + b_in[5];
        ap_uint<49> bhhs = b_in[6] + b_in[7];

        ap_uint<102> ps[4], pls[2], phs[2];
        ps[0] = mult_51(as[0], bs[0]);
        ps[1] = mult_51(as[1], bs[1]);
        ps[2] = mult_51(as[2], bs[2]);
        ps[3] = mult_51(as[3], bs[3]);
        pls[0] = mult_51(als[0], bls[0]);
        pls[1] = mult_51(als[1], bls[1]);
        phs[0] = mult_51(ahs[0], bhs[0]);
        phs[1] = mult_51(ahs[1], bhs[1]);
        ap_uint<102> plls = mult_51(alls, blls);
        ap_uint<102> plhs = mult_51(alhs, blhs);
        ap_uint<102> phls = mult_51(ahls, bhls);
        ap_uint<102> phhs = mult_51(ahhs, bhhs);

        ap_uint<50> ass[2], bss[2];
        ass[0] = as[0] + as[2];
        ass[1] = as[1] + as[3];
        bss[0] = bs[0] + bs[2];
        bss[1] = bs[1] + bs[3];
        ap_uint<50> alss = als[0] + als[1];
        ap_uint<50> ahss = ahs[0] + ahs[1];
        ap_uint<50> asls = as[0] + as[1];
        ap_uint<50> ashs = as[2] + as[3];
        ap_uint<50> blss = bls[0] + bls[1];
        ap_uint<50> bhss = bhs[0] + bhs[1];
        ap_uint<50> bsls = bs[0] + bs[1];
        ap_uint<50> bshs = bs[2] + bs[3];

        ap_uint<102> pss[2];
        pss[0] = mult_51(ass[0], bss[0]);
        pss[1] = mult_51(ass[1], bss[1]);
        ap_uint<102> plss = mult_51(alss, blss);
        ap_uint<102> phss = mult_51(ahss, bhss);
        ap_uint<102> psls = mult_51(asls, bsls);
        ap_uint<102> pshs = mult_51(ashs, bshs);

        ap_uint<51> asss = ass[0] + ass[1];
        ap_uint<51> bsss = bss[0] + bss[1];

        ap_uint<102> psss = mult_51(asss, bsss);

        long oo_s5[8], ooo_s5[8];
        oo_s5[0] = -lo(p[0]) - lo(p[2]);
        oo_s5[1] = hi(p[1]);
        oo_s5[2] = lo(p[2]);
        oo_s5[3] = -hi(p[1]) - hi(p[3]);
        oo_s5[4] = -lo(p[4]) - lo(p[6]);
        oo_s5[5] = hi(p[5]);
        oo_s5[6] = lo(p[6]);
        oo_s5[7] = -hi(p[5]) - hi(p[7]);

        ooo_s5[0] = hi(p[0]) - lo(p[0]) - lo(p[1]);
        ooo_s5[1] = lo(p[1]) - hi(p[0]) - hi(p[1]);
        ooo_s5[2] = hi(p[2]) - lo(p[2]) - lo(p[3]);
        ooo_s5[3] = lo(p[3]) - hi(p[2]) - hi(p[3]);
        ooo_s5[4] = hi(p[4]) - lo(p[4]) - lo(p[5]);
        ooo_s5[5] = lo(p[5]) - hi(p[4]) - hi(p[5]);
        ooo_s5[6] = hi(p[6]) - lo(p[6]) - lo(p[7]);
        ooo_s5[7] = lo(p[7]) - hi(p[6]) - hi(p[7]);

        long oo_s6[8], ooo[8], oooss_s6[4];
        oo_s6[0] = oo_s5[0] + lo(pls[0]);
        oo_s6[1] = oo_s5[1] + hi(pls[0]) - lo(pls[0]) - lo(pls[1]);
        oo_s6[2] = oo_s5[2] + lo(pls[1]) - hi(pls[0]) - hi(pls[1]);
        oo_s6[3] = oo_s5[3] + hi(pls[1]);
        oo_s6[4] = oo_s5[4] + lo(phs[0]);
        oo_s6[5] = oo_s5[5] + hi(phs[0]) - lo(phs[0]) - lo(phs[1]);
        oo_s6[6] = oo_s5[6] + lo(phs[1]) - hi(phs[0]) - hi(phs[1]);
        oo_s6[7] = oo_s5[7] + hi(phs[1]);

        ooo[0] = ooo_s5[0] + lo(plls);
        ooo[1] = ooo_s5[1] + hi(plls);
        ooo[2] = ooo_s5[2] + lo(plhs);
        ooo[3] = ooo_s5[3] + hi(plhs);
        ooo[4] = ooo_s5[4] + lo(phls);
        ooo[5] = ooo_s5[5] + hi(phls);
        ooo[6] = ooo_s5[6] + lo(phhs);
        ooo[7] = ooo_s5[7] + hi(phhs);

        oooss_s6[0] = hi(ps[0]) - lo(ps[0]) - lo(ps[1]);
        oooss_s6[1] = lo(ps[1]) - hi(ps[0]) - hi(ps[1]);
        oooss_s6[2] = hi(ps[2]) - lo(ps[2]) - lo(ps[3]);
        oooss_s6[3] = lo(ps[3]) - hi(ps[2]) - hi(ps[3]);

        long oo[8], oooss[4];
        oo[0] = oo_s6[0] + ooo[1];
        oo[1] = oo_s6[1] - ooo[0] - ooo[2] + lo(plss);
        oo[2] = oo_s6[2] - ooo[1] - ooo[3] + hi(plss);
        oo[3] = oo_s6[3] + ooo[2];
        oo[4] = oo_s6[4] + ooo[5];
        oo[5] = oo_s6[5] - ooo[4] - ooo[6] + lo(phss);
        oo[6] = oo_s6[6] - ooo[5] - ooo[7] + hi(phss);
        oo[7] = oo_s6[7] + ooo[6];

        oooss[0] = oooss_s6[0] + lo(psls);
        oooss[1] = oooss_s6[1] + hi(psls);
        oooss[2] = oooss_s6[2] + lo(pshs);
        oooss[3] = oooss_s6[3] + hi(pshs);

        hi_lo_t res;
        res.lo[0] = lo(p[0]);
        res.lo[1] = ooo[0];
        res.lo[2] = oo[0];
        res.lo[3] = oo[1];
        res.lo[4] = oo[2] - lo(p[0]) - lo(p[4]) + lo(ps[0]);
        res.lo[5] = oo[3] - ooo[0] - ooo[4] + oooss[0];
        res.lo[6] = ooo[3] - oo[0] - oo[4] + oooss[1] - lo(ps[0]) - lo(ps[2]) + lo(pss[0]);
        res.lo[7] = hi(p[3]) - oo[1] - oo[5] + hi(ps[1]) - oooss[0] - oooss[2] + hi(pss[0]) - lo(pss[0]) - lo(pss[1]) +
                    lo(psss);
        res.lo[8] = 0;

        bool neg = b_in[8] == -1;
        hls::vector<ap_uint<48>, 9> tmp;
        tmp[0] = neg ? ap_uint<48>(-a_in[0]) : ap_uint<48>(0);
        tmp[1] = neg ? ap_uint<48>(-a_in[1]) : ap_uint<48>(0);
        tmp[2] = neg ? ap_uint<48>(-a_in[2]) : ap_uint<48>(0);
        tmp[3] = neg ? ap_uint<48>(-a_in[3]) : ap_uint<48>(0);
        tmp[4] = neg ? ap_uint<48>(-a_in[4]) : ap_uint<48>(0);
        tmp[5] = neg ? ap_uint<48>(-a_in[5]) : ap_uint<48>(0);
        tmp[6] = neg ? ap_uint<48>(-a_in[6]) : ap_uint<48>(0);
        tmp[7] = neg ? ap_uint<48>(-a_in[7]) : ap_uint<48>(0);

        res.hi[0] = tmp[0] + lo(p[4]) - oo[2] - oo[6] + lo(ps[2]) - oooss[1] - oooss[3] + lo(pss[1]) - hi(pss[0]) -
                    hi(pss[1]) + hi(psss);
        res.hi[1] = tmp[1] + ooo[4] - oo[3] - oo[7] + oooss[2] - hi(ps[1]) - hi(ps[3]) + hi(pss[1]);
        res.hi[2] = tmp[2] + oo[4] - ooo[3] - ooo[7] + oooss[3];
        res.hi[3] = tmp[3] + oo[5] - hi(p[3]) - hi(p[7]) + hi(ps[3]);
        res.hi[4] = tmp[4] + oo[6];
        res.hi[5] = tmp[5] + oo[7];
        res.hi[6] = tmp[6] + ooo[7];
        res.hi[7] = tmp[7] + hi(p[7]);
        res.hi[8] = 0;

        return res;
    }

    static hls::vector<long, 9> low_reduce(const hls::vector<ap_uint<48>, 9> &lo_in) {
        const ap_uint<64> NPS0 = NP[1] + NP[3];
        const ap_uint<64> NPS1 = NP[2] + NP[4];

        ap_uint<49> los0 = lo_in[0] + lo_in[2];
        ap_uint<49> los1 = lo_in[1] + lo_in[3];

        ap_uint<102> lo0np0 = np_0_term_48(lo_in[0]);
        ap_uint<102> lo1np0 = np_0_term_48(lo_in[1]);
        ap_uint<102> lo2np0 = np_0_term_48(lo_in[2]);
        ap_uint<102> lo3np0 = np_0_term_48(lo_in[3]);
        ap_uint<102> lo4np0 = np_0_term_48(lo_in[4]);
        ap_uint<102> lo5np0 = np_0_term_48(lo_in[5]);
        ap_uint<102> lo6np0 = np_0_term_48(lo_in[6]);
        ap_uint<102> lo7np0 = np_0_term_48(lo_in[7]);

        ap_uint<48> lo6np1l = mult_51(lo_in[6], NP[1]);
        ap_uint<48> lo4np3l = mult_51(lo_in[4], NP[3]);
        ap_uint<48> lo2np5l = mult_51(lo_in[2], NP[5]);
        ap_uint<48> lo0np7l = mult_51(lo_in[0], NP[7]);

        ap_uint<102> lo0np1 = mult_51(lo_in[0], NP[1]);
        ap_uint<102> lo1np2 = mult_51(lo_in[1], NP[2]);
        ap_uint<102> lo2np3 = mult_51(lo_in[2], NP[3]);
        ap_uint<102> lo3np4 = mult_51(lo_in[3], NP[4]);

        ap_uint<102> lo0np5 = mult_51(lo_in[0], NP[5]);
        ap_uint<102> lo1np6 = mult_51(lo_in[1], NP[6]);
        ap_uint<102> lo4np1 = mult_51(lo_in[4], NP[1]);
        ap_uint<102> lo5np2 = mult_51(lo_in[5], NP[2]);

        ap_uint<102> p0 = mult_51(lo_in[0] + lo_in[1], NP[1] + NP[2]);
        ap_uint<102> p1 = mult_51(lo_in[2] + lo_in[3], NP[3] + NP[4]);
        ap_uint<102> p2 = mult_51(los0, NPS0);
        ap_uint<102> p3 = mult_51(los1, NPS1);
        ap_uint<100> p4 = mult_51(los0 + los1, NPS0 + NPS1);
        ap_uint<102> p5 = mult_51(lo_in[0] + lo_in[1], NP[5] + NP[6]);
        ap_uint<102> p6 = mult_51(lo_in[4] + lo_in[5], NP[1] + NP[2]);

        long l0 = hi(lo0np1) + lo(p0) - lo(lo0np1) - lo(lo1np2);
        long l1 = lo(lo1np2) + hi(p0) - hi(lo0np1) - hi(lo1np2);
        long h0 = hi(lo2np3) + lo(p1) - lo(lo2np3) - lo(lo3np4);
        long h1 = lo(lo3np4) + hi(p1) - hi(lo2np3) - hi(lo3np4);
        long s0 = hi(p2) + lo(p4) - lo(p2) - lo(p3);
        long s1 = lo(p3) + hi(p4) - hi(p2) - hi(p3);

        hls::vector<long, 9> res = 0;
        res[0] = lo(lo0np0);
        res[1] = hi(lo0np0) + lo(lo1np0) + lo(lo0np1);
        res[2] = hi(lo1np0) + lo(lo2np0) + l0;
        res[3] = hi(lo2np0) + lo(lo3np0) + l1 + lo(p2) - lo(lo0np1) - lo(lo2np3);
        res[4] = hi(lo3np0) + lo(lo4np0) + hi(lo1np2) + s0 - l0 - h0;
        res[5] = hi(lo4np0) + lo(lo5np0) + lo(lo2np3) + s1 - l1 - h1;
        res[6] = hi(lo5np0) + lo(lo6np0) + h0 + hi(p3) - hi(lo1np2) - hi(lo3np4);
        res[7] = hi(lo6np0) + lo(lo7np0) + h1;

        res[5] += lo(lo0np5) + lo(lo4np1);
        res[6] += hi(lo0np5) + hi(lo4np1) + lo(p5) + lo(p6) - lo(lo0np5) - lo(lo1np6) - lo(lo4np1) - lo(lo5np2);
        res[7] += lo(lo1np6) + lo(lo5np2) + hi(p5) + hi(p6) - hi(lo0np5) - hi(lo1np6) - hi(lo4np1) - hi(lo5np2);

        // TODO: Looks fishy? Seems to work *shrugs*.
        res[7] += lo6np1l + lo4np3l + lo2np5l + lo0np7l;
        res[8] = 0;

        return res;
    }

    static hls::vector<long, 9> high_reduce(
            const hls::vector<long, 9> &hi_in,
            const hls::vector<ap_uint<48>, 9> &lo_in,
            const hls::vector<ap_uint<49>, 9> &q_in
    ) {
        const ap_uint<64> PS0 = P[4] + P[6];
        const ap_uint<64> PS1 = P[5] + P[7];

        ap_uint<50> qs0 = q_in[4] + q_in[6];
        ap_uint<50> qs1 = q_in[5] + q_in[7];

        ap_uint<102> q0p0 = p_0_term_48(q_in[0]);
        ap_uint<102> q1p1 = mult_51(q_in[1], P[1]);
        ap_uint<102> q2p2 = mult_51(q_in[2], P[2]);
        ap_uint<102> q3p3 = mult_51(q_in[3], P[3]);
        ap_uint<102> q4p4 = mult_51(q_in[4], P[4]);
        ap_uint<102> q5p5 = mult_51(q_in[5], P[5]);
        ap_uint<102> q6p6 = mult_51(q_in[6], P[6]);
        ap_uint<102> q7p7 = mult_51(q_in[7], P[7]);

        ap_uint<102> pair07 = mult_51(q_in[0] + q_in[7], P[0] + P[7]);
        ap_uint<102> pair16 = mult_51(q_in[1] + q_in[6], P[1] + P[6]);
        ap_uint<102> pair25 = mult_51(q_in[2] + q_in[5], P[2] + P[5]);
        ap_uint<102> pair34 = mult_51(q_in[3] + q_in[4], P[3] + P[4]);
        ap_uint<102> pair17 = mult_51(q_in[1] + q_in[7], P[1] + P[7]);
        ap_uint<102> pair26 = mult_51(q_in[2] + q_in[6], P[2] + P[6]);
        ap_uint<102> pair35 = mult_51(q_in[3] + q_in[5], P[3] + P[5]);
        ap_uint<102> pair27 = mult_51(q_in[2] + q_in[7], P[2] + P[7]);
        ap_uint<102> pair36 = mult_51(q_in[3] + q_in[6], P[3] + P[6]);
        ap_uint<102> pair37 = mult_51(q_in[3] + q_in[7], P[3] + P[7]);

        ap_uint<102> p0 = mult_51(q_in[4] + q_in[5], P[4] + P[5]);
        ap_uint<102> p1 = mult_51(q_in[6] + q_in[7], P[6] + P[7]);
        ap_uint<102> p2 = mult_51(qs0, PS0);
        ap_uint<102> p3 = mult_51(qs1, PS1);
        ap_uint<102> p4 = mult_51(qs0 + qs1, PS0 + PS1);

        long vh = hi(q3p3);
        vh += mult_very_high(q_in[0], P[6]);
        vh += mult_very_high(q_in[1], P[5]);
        vh += mult_very_high(q_in[2], P[4]);
        vh += mult_very_high(q_in[4], P[2]);
        vh += mult_very_high(q_in[5], P[1]);
        vh += mult_very_high(q_in[6], P[0]);

        long sl37 = lo(q3p3) + lo(q7p7);
        long sh37 = hi(q3p3) + hi(q7p7);
        long sl2367 = sl37 + lo(q2p2) + lo(q6p6);
        long sh2367 = sh37 + hi(q2p2) + hi(q6p6);
        long sl123567 = sl2367 + lo(q1p1) + lo(q5p5);
        long sh123567 = sh2367 + hi(q1p1) + hi(q5p5);
        long sl01234567 = sl123567 + lo(q0p0) + lo(q4p4);
        long sh01234567 = sh123567 + hi(q0p0) + hi(q4p4);

        long l0 = hi(q4p4) + lo(p0) - lo(q4p4) - lo(q5p5);
        long l1 = lo(q5p5) + hi(p0) - hi(q4p4) - hi(q5p5);
        long h0 = hi(q6p6) + lo(p1) - lo(q6p6) - lo(q7p7);
        long h1 = lo(q7p7) + hi(p1) - hi(q6p6) - hi(q7p7);
        long s0 = hi(p2) + lo(p4) - lo(p2) - lo(p3);
        long s1 = lo(p3) + hi(p4) - hi(p2) - hi(p3);

        long round = lo_in[7] + 0x800000000000L + vh + lo(pair07) + lo(pair16) + lo(pair25) + lo(pair34) - sl01234567;

        hls::vector<long, 9> res = 0;
        res[0] = hi_in[0] + ap_int<48>(lo_in[8]) + lo(q4p4) + (round >> 48);
        res[1] = hi_in[1] + l0;
        res[2] = hi_in[2] + l1 + lo(p2) - lo(q4p4) - lo(q6p6);
        res[3] = hi_in[3] + hi(q5p5) + s0 - l0 - h0;
        res[4] = hi_in[4] + lo(q6p6) + s1 - l1 - h1;
        res[5] = hi_in[5] + h0 + hi(p3) - hi(q5p5) - hi(q7p7);
        res[6] = hi_in[6] + h1;
        res[7] = hi_in[7] + hi(q7p7);
        res[8] = hi_in[8];

        res[0] +=
                hi(pair07) + hi(pair16) + hi(pair25) + hi(pair34) - sh01234567 + lo(pair17) + lo(pair26) + lo(pair35) -
                sl123567;
        res[1] += hi(pair17) + hi(pair26) + hi(pair35) - sh123567 + lo(pair27) + lo(pair36) - sl2367;
        res[2] += hi(pair27) + hi(pair36) - sh2367 + lo(pair37) - sl37;
        res[3] += hi(pair37) - sh37;

        return res;
    }

    static hls::vector<ap_uint<49>, 9> partial_resolve(const hls::vector<long, 9> &data) {
        constexpr long FORTY_NINE = 0x1FFFFFFFFFFFFL;
        constexpr long FORTY_EIGHT = 0xFFFFFFFFFFFFL;

        hls::vector<ap_uint<48>, 9> rem;
        hls::vector<ap_int<16>, 8> carry;
        hls::vector<ap_uint<1>, 8> c_out;

        rem[0] = ap_uint<48>(data[0]);
        carry[0] = data[0] >> 48;
        c_out[0] = 1;
        for (int i = 1; i < 8; i++) {
#pragma HLS UNROLL
            rem[i] = data[i];
            carry[i] = data[i] >> 48;
            c_out[i] = rem[i][47];
        }

        hls::vector<ap_uint<49>, 9> res;
        res[0] = rem[0];
        for (int i = 1; i < 8; i++) {
#pragma HLS UNROLL
            res[i] = rem[i] + carry[i - 1] + c_out[i - 1] + (c_out[i] ? FORTY_NINE : FORTY_EIGHT);
        }

        res[8] = data[8] + carry[7] + c_out[7] - 1;
        return res;
    }

    static hls::vector<ap_uint<48>, 9> full_resolve(const hls::vector<ap_uint<49>, 9> &data) {
        long with_carry[9];
        ap_uint<8> g = 0;
        ap_uint<8> p = 0;

        for (int i = 7; i >= 0; i--) {
#pragma HLS UNROLL
            with_carry[i + 1] = data[i + 1] + 1;
            g = g * 2 + (data[i + 1] >> 48);
            p = p * 2 + (with_carry[i + 1] >> 48);
        }

        ap_uint<8> mask = (g + p) ^ (g ^ p);

        hls::vector<ap_uint<48>, 9> res;

        res[0] = data[0];
        for (int i = 0; i < 7; i++) {
#pragma HLS UNROLL
            long tmp = (mask[i] == 0) ? ((long) data[i + 1]) : with_carry[i + 1];
            res[i + 1] = tmp;
        }

        long tmp = (mask[7] == 0) ? ((long) data[8]) : with_carry[8];
        res[8] = tmp;

        return res;
    }

    static hls::vector<long, 9> slow_reduce(hls::vector<long, 16> &in) {
#pragma HLS INLINE
        hls::vector<long, 16> out;
        hls::vector<ap_uint<48>, 8> q;

        for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
            out[i] = in[i];
        }

        // -- [step 0] -------------------------------------------------------------------------------------------------
        q[0] = q_0_term_48(out[0]);
        ap_uint<96> q0p0 = p_0_term_48(q[0]);
        ap_uint<96> q0p1 = mult_51(q[0], P[1]);
        ap_uint<96> q0p2 = mult_51(q[0], P[2]);
        ap_uint<96> q0p3 = mult_51(q[0], P[3]);
        ap_uint<96> q0p4 = mult_51(q[0], P[4]);
        ap_uint<96> q0p5 = mult_51(q[0], P[5]);
        ap_uint<96> q0p6 = mult_51(q[0], P[6]);
        ap_uint<96> q0p7 = mult_51(q[0], P[7]);

        out[0] += lo(q0p0);
        out[1] += lo(q0p1) + hi(q0p0);
        out[2] += lo(q0p2) + hi(q0p1);
        out[3] += lo(q0p3) + hi(q0p2);
        out[4] += lo(q0p4) + hi(q0p3);
        out[5] += lo(q0p5) + hi(q0p4);
        out[6] += lo(q0p6) + hi(q0p5);
        out[7] += lo(q0p7) + hi(q0p6);
        out[8] += hi(q0p7);
        // -------------------------------------------------------------------------------------------------------------

        // -- [step 1] -------------------------------------------------------------------------------------------------
        out[1] += out[0] >> 48;
        q[1] = q_0_term_48(out[1]);

        ap_uint<96> q1p0 = p_0_term_48(q[1]);
        ap_uint<96> q1p1 = mult_51(q[1], P[1]);
        out[1] += lo(q1p0);
        out[2] += hi(q1p0);
        out[2] += lo(q1p1);
        out[3] += hi(q1p1);
        long sum_lo = lo(q1p1);
        long sum_hi = hi(q1p1);
        // -------------------------------------------------------------------------------------------------------------

        // -- [step 2] -------------------------------------------------------------------------------------------------
        out[9] += sum_lo;
        out[10] += sum_hi;

        out[2] += out[1] >> 48;
        q[2] = q_0_term_48(out[2]);

        ap_uint<96> q2p0 = p_0_term_48(q[2]);
        ap_uint<98> pair12 = mult_51(q[1] + q[2], P[1] + P[2]);
        ap_uint<96> q2p2 = mult_51(q[2], P[2]);

        out[2] += lo(q2p0);
        out[3] += hi(q2p0) + lo(pair12);
        out[4] += 2 * lo(q2p2) + hi(pair12);
        out[5] += 2 * hi(q2p2);

        sum_lo += lo(q2p2);
        sum_hi += hi(q2p2);
        // -------------------------------------------------------------------------------------------------------------

        // -- [step 3] -------------------------------------------------------------------------------------------------
        out[3] -= sum_lo;
        out[4] -= sum_hi;
        out[10] += sum_lo;
        out[11] += sum_hi;

        out[3] += out[2] >> 48;
        q[3] = q_0_term_48(out[3]);

        ap_uint<96> q3p0 = p_0_term_48(q[3]);
        ap_uint<98> pair13 = mult_51(q[1] + q[3], P[1] + P[3]);
        ap_uint<98> pair23 = mult_51(q[2] + q[3], P[2] + P[3]);
        ap_uint<96> q3p3 = mult_51(q[3], P[3]);

        out[3] += lo(q3p0);
        out[4] += hi(q3p0) + lo(pair13);
        out[5] += hi(pair13) + lo(pair23);
        out[6] += hi(pair23) + 2 * lo(q3p3);
        out[7] += 2 * hi(q3p3);

        sum_lo += lo(q3p3);
        sum_hi += hi(q3p3);
        // -------------------------------------------------------------------------------------------------------------

        // -- [step 4] -------------------------------------------------------------------------------------------------
        out[4] -= sum_lo;
        out[5] -= sum_hi;
        out[11] += sum_lo;
        out[12] += sum_hi;

        out[4] += out[3] >> 48;
        q[4] = q_0_term_48(out[4]);

        ap_uint<96> q4p0 = p_0_term_48(q[4]);
        ap_uint<98> pair14 = mult_51(q[1] + q[4], P[1] + P[4]);
        ap_uint<98> pair24 = mult_51(q[2] + q[4], P[2] + P[4]);
        ap_uint<98> pair34 = mult_51(q[3] + q[4], P[3] + P[4]);
        ap_uint<96> q4p4 = mult_51(q[4], P[4]);

        out[4] += lo(q4p0);
        out[5] += hi(q4p0) + lo(pair14);
        out[6] += hi(pair14) + lo(pair24);
        out[7] += hi(pair24) + lo(pair34);
        out[8] += hi(pair34) + 2 * lo(q4p4);
        out[9] += 2 * hi(q4p4);

        sum_lo += lo(q4p4);
        sum_hi += hi(q4p4);
        // -------------------------------------------------------------------------------------------------------------

        // -- [step 5] -------------------------------------------------------------------------------------------------
        out[5] -= sum_lo;
        out[6] -= sum_hi;
        out[12] += sum_lo;
        out[13] += sum_hi;

        out[5] += out[4] >> 48;
        q[5] = q_0_term_48(out[5]);

        ap_uint<96> q5p0 = p_0_term_48(q[5]);
        ap_uint<98> pair15 = mult_51(q[1] + q[5], P[1] + P[5]);
        ap_uint<98> pair25 = mult_51(q[2] + q[5], P[2] + P[5]);
        ap_uint<98> pair35 = mult_51(q[3] + q[5], P[3] + P[5]);
        ap_uint<98> pair45 = mult_51(q[4] + q[5], P[4] + P[5]);
        ap_uint<96> q5p5 = mult_51(q[5], P[5]);

        out[5] += lo(q5p0);
        out[6] += hi(q5p0) + lo(pair15);
        out[7] += hi(pair15) + lo(pair25);
        out[8] += hi(pair25) + lo(pair35);
        out[9] += hi(pair35) + lo(pair45);
        out[10] += hi(pair45) + 2 * lo(q5p5);
        out[11] += 2 * hi(q5p5);

        sum_lo += lo(q5p5);
        sum_hi += hi(q5p5);
        // -------------------------------------------------------------------------------------------------------------

        // -- [step 6] -------------------------------------------------------------------------------------------------
        out[6] -= sum_lo;
        out[7] -= sum_hi;
        out[13] += sum_lo;
        out[14] += sum_hi;

        out[6] += out[5] >> 48;
        q[6] = q_0_term_48(out[6]);

        ap_uint<96> q6p0 = p_0_term_48(q[6]);
        ap_uint<98> pair16 = mult_51(q[1] + q[6], P[1] + P[6]);
        ap_uint<98> pair26 = mult_51(q[2] + q[6], P[2] + P[6]);
        ap_uint<98> pair36 = mult_51(q[3] + q[6], P[3] + P[6]);
        ap_uint<98> pair46 = mult_51(q[4] + q[6], P[4] + P[6]);
        ap_uint<98> pair56 = mult_51(q[5] + q[6], P[5] + P[6]);
        ap_uint<96> q6p6 = mult_51(q[6], P[6]);

        out[6] += lo(q6p0);
        out[7] += hi(q6p0) + lo(pair16);
        out[8] += hi(pair16) + lo(pair26);
        out[9] += hi(pair26) + lo(pair36);
        out[10] += hi(pair36) + lo(pair46);
        out[11] += hi(pair46) + lo(pair56);
        out[12] += hi(pair56) + 2 * lo(q6p6);
        out[13] += 2 * hi(q6p6);

        sum_lo += lo(q6p6);
        sum_hi += hi(q6p6);
        // -------------------------------------------------------------------------------------------------------------

        // -- [step 7] -------------------------------------------------------------------------------------------------
        out[7] -= sum_lo;
        out[8] -= sum_hi;

        out[7] += out[6] >> 48;
        q[7] = q_0_term_48(out[7]);

        ap_uint<96> q7p0 = p_0_term_48(q[7]);
        ap_uint<98> pair17 = mult_51(q[1] + q[7], P[1] + P[7]);
        ap_uint<98> pair27 = mult_51(q[2] + q[7], P[2] + P[7]);
        ap_uint<98> pair37 = mult_51(q[3] + q[7], P[3] + P[7]);
        ap_uint<98> pair47 = mult_51(q[4] + q[7], P[4] + P[7]);
        ap_uint<98> pair57 = mult_51(q[5] + q[7], P[5] + P[7]);
        ap_uint<98> pair67 = mult_51(q[6] + q[7], P[6] + P[7]);
        ap_uint<96> q7p7 = mult_51(q[7], P[7]);

        out[7] += lo(q7p0);
        out[8] += hi(q7p0) + lo(pair17);
        out[9] += hi(pair17) + lo(pair27);
        out[10] += hi(pair27) + lo(pair37);
        out[11] += hi(pair37) + lo(pair47);
        out[12] += hi(pair47) + lo(pair57);
        out[13] += hi(pair57) + lo(pair67);
        out[14] += hi(pair67) + lo(q7p7);
        out[15] += hi(q7p7);

        sum_lo += lo(q7p7);
        sum_hi += hi(q7p7);

        out[8] += out[7] >> 48;

        out[8] -= sum_lo;
        out[9] -= sum_lo + sum_hi;
        out[10] -= sum_lo + sum_hi;
        out[11] -= sum_lo + sum_hi;
        out[12] -= sum_lo + sum_hi;
        out[13] -= sum_lo + sum_hi;
        out[14] -= sum_hi;
        // -------------------------------------------------------------------------------------------------------------

        hls::vector<long, 9> res;
        for (int i = 0; i < 8; i++) {
            res[i] = out[i + 8];
        }
        res[8] = 0;

        return res;
    }
};
