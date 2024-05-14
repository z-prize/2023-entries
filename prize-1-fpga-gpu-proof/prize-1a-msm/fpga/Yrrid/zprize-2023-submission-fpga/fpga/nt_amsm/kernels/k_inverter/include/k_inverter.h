/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

#pragma once

#include "hw_msm.h"
#include "mod_mult.h"

void k_bls12_381_inverter(
    hls::stream<field_t> &from_proc_0,
    hls::stream<field_t> &from_proc_1,
    hls::stream<field_t> &from_proc_2,
    hls::stream<field_t> &to_proc_0,
    hls::stream<field_t> &to_proc_1,
    hls::stream<field_t> &to_proc_2
);

void k_bls12_377_inverter(
    hls::stream<field_t> &from_proc_0,
    hls::stream<field_t> &from_proc_1,
    hls::stream<field_t> &from_proc_2,
    hls::stream<field_t> &to_proc_0,
    hls::stream<field_t> &to_proc_1,
    hls::stream<field_t> &to_proc_2
);

template<bls12_t CURVE>
class k_inverter {
    static constexpr char KERNEL_NAME[] = "k_inverter";
    static constexpr char COMPONENT_NAME[] = "top";

public:
    static void k_inverter_inst(
        hls::stream<field_t> &from_proc_0,
        hls::stream<field_t> &from_proc_1,
        hls::stream<field_t> &from_proc_2,
        hls::stream<field_t> &to_proc_0,
        hls::stream<field_t> &to_proc_1,
        hls::stream<field_t> &to_proc_2
    ) {
        hls_thread_local hls::stream<field_t, 512> s_a_ingress("s_a_ingress");
        hls_thread_local hls::stream<field_t, 512> s_b_ingress("s_b_ingress");
        hls_thread_local hls::stream<field_t, 512> s_c_ingress("s_c_ingress");
        hls_thread_local hls::stream<wide_t, 2> s_ingress_to_d_tree("s_ingress_to_d_tree");
        hls_thread_local hls::stream<field_t, 2> s_d_tree_to_reorderer("s_d_tree_to_reorderer");
        hls_thread_local hls::stream<field_t, 1024> s_reorderer_to_d_untree("s_reorderer_to_d_untree");
        hls_thread_local hls::stream<field_t, 2> s_d_tree_to_inverter("s_d_tree_to_inverter");
        hls_thread_local hls::stream<field_t, 512> s_d_untree_to_egress("s_d_untree_to_egress");
        hls_thread_local hls::stream<field_t, 2> s_inverter_to_d_untree("s_inverter_to_d_untree");
        // -- [ingress] ------------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_ingress_buffer_0(ingress_buffer, from_proc_0, s_a_ingress);
        hls_thread_local hls::task t_ingress_buffer_1(ingress_buffer, from_proc_1, s_b_ingress);
        hls_thread_local hls::task t_ingress_buffer_2(ingress_buffer, from_proc_2, s_c_ingress);
        hls_thread_local hls::task t_ingress(ingress, s_a_ingress, s_b_ingress, s_c_ingress, s_ingress_to_d_tree);
        // -------------------------------------------------------------------------------------------------------------
        // -- [d_tree] -------------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_d_tree(d_tree::d_tree_inst, s_ingress_to_d_tree, s_d_tree_to_reorderer,
                                            s_d_tree_to_inverter);
        // -------------------------------------------------------------------------------------------------------------
        // -- [reorderer] ----------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_reorderer(reorderer, s_d_tree_to_reorderer, s_reorderer_to_d_untree);
        // -------------------------------------------------------------------------------------------------------------
        // -- [inverter] -----------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_inverter(inverter<0>::inverter_inst, s_d_tree_to_inverter, s_inverter_to_d_untree);
        // -------------------------------------------------------------------------------------------------------------
        // -- [d_untree] -----------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_d_untree(d_untree::d_untree_inst, s_reorderer_to_d_untree, s_inverter_to_d_untree,
                                              s_d_untree_to_egress);
        // -------------------------------------------------------------------------------------------------------------
        // -- [egress] -------------------------------------------------------------------------------------------------
        hls_thread_local hls::task t_egress(egress, s_d_untree_to_egress, to_proc_0, to_proc_1, to_proc_2);
        // -------------------------------------------------------------------------------------------------------------
    }

private:
    struct wide_t {
        field_t a;
        field_t b;
    };

    struct inverter_to_reducer_t {
        field_t v;
        int shift_count = 0;
    };

    static void ingress_buffer(hls::stream<field_t> &in, hls::stream<field_t> &out) {
#pragma HLS PIPELINE II=1
        out << in.read();
    }

    static void ingress(
        hls::stream<field_t> &a_in,
        hls::stream<field_t> &b_in,
        hls::stream<field_t> &c_in,
        hls::stream<wide_t> &out
    ) {
        for (int i = 0; i < 21; i++) {
#pragma HLS PIPELINE II=3
            out << wide_t{a_in.read(), b_in.read()};
            out << wide_t{c_in.read(), a_in.read()};
            out << wide_t{b_in.read(), c_in.read()};
        }
        out << wide_t{base_field_t<CURVE>::MONT_ONE, base_field_t<CURVE>::MONT_ONE};
    }

    static void egress(hls::stream<field_t> &in, hls::stream<field_t> &a_out, hls::stream<field_t> &b_out,
                       hls::stream<field_t> &c_out) {
        for (int i = 0; i < 42; i++) {
#pragma HLS PIPELINE II=3
            a_out << in.read();
            b_out << in.read();
            c_out << in.read();
        }
        in.read();
        in.read();
    }

    class d_tree {
        static constexpr char COMPONENT_NAME[] = "d_tree";

    public:
        static void d_tree_inst(
            hls::stream<wide_t> &in,
            hls::stream<field_t> &to_reorderer,
            hls::stream<field_t> &to_inverter
        ) {
            hls_thread_local hls::stream<exec_pkt_t, 32> s_dispatch_to_mult("s_d_tree_dispatch_to_mult");
            hls_thread_local hls::stream<field_t, 2> s_mult_to_recycle("s_d_tree_mult_to_recycle");
            hls_thread_local hls::stream<wide_t, 64> s_recycle_to_dispatch("s_d_tree_recycle_to_dispatch");
            hls_thread_local hls::stream<wide_t, 64> s_mult_to_reorderer("s_d_tree_mult_to_reorderer");

            hls_thread_local hls::task t_dispatch(dispatch, in, s_recycle_to_dispatch, s_dispatch_to_mult);
            hls_thread_local hls::task t_recycle(recycle, s_mult_to_recycle, s_recycle_to_dispatch);
            hls_thread_local hls::task t_mult(mult, s_dispatch_to_mult, s_mult_to_reorderer, s_mult_to_recycle,
                                              to_inverter);
            hls_thread_local hls::task t_forward(forward, s_mult_to_reorderer, to_reorderer);
        }

    private:
        struct exec_pkt_t {
            field_t a;
            field_t b;
            bool inv_mode = false;
        };

        static void dispatch(hls::stream<wide_t> &in, hls::stream<wide_t> &recycle_in, hls::stream<exec_pkt_t> &out) {
            for (int i = 0; i < 128; i++) {
#pragma HLS PIPELINE II=1
                if (i < 64) {
                    wide_t tmp = in.read();
                    out << exec_pkt_t{tmp.a, tmp.b, false};
                } else if (i < 124) {
                    wide_t tmp = recycle_in.read();
                    out << exec_pkt_t{tmp.a, tmp.b, false};
                }
            }

            wide_t tmp = recycle_in.read();
            out << exec_pkt_t{tmp.a, base_field_t<CURVE>::R_INV_MOD_P, true};
            out << exec_pkt_t{tmp.b, base_field_t<CURVE>::R_INV_MOD_P, true};
            tmp = recycle_in.read();
            out << exec_pkt_t{tmp.a, base_field_t<CURVE>::R_INV_MOD_P, true};
            out << exec_pkt_t{tmp.b, base_field_t<CURVE>::R_INV_MOD_P, true};
        }

        static void recycle(hls::stream<field_t> &in, hls::stream<wide_t> &out) {
#pragma HLS PIPELINE II=2
            field_t a = in.read();
            field_t b = in.read();
            out << wide_t{a, b};
        }

        static void forward(hls::stream<wide_t> &in, hls::stream<field_t> &out) {
#pragma HLS PIPELINE II=2
            wide_t tmp = in.read();
            out << tmp.a;
            out << tmp.b;
        }

        static void mult(
            hls::stream<exec_pkt_t> &in,
            hls::stream<wide_t> &to_reorderer,
            hls::stream<field_t> &recycle_out,
            hls::stream<field_t> &out
        ) {
#pragma HLS PIPELINE II=1 style=frp
#pragma HLS LATENCY max=38
#pragma HLS EXPRESSION_BALANCE off
            exec_pkt_t pkt = in.read();
#ifndef __XOR_STUB__
            field_t res = mod_mult<CURVE>::ll_mod_mult(pkt.a, pkt.b);
#else
            field_t res = pkt.a ^ pkt.b;
#endif
            if (pkt.inv_mode) {
                out << res;
            } else {
                recycle_out << res;
                to_reorderer << wide_t{pkt.a, pkt.b};
            }
        }
    };

    static void reorderer_writer(hls::stream<field_t> &in, hls::stream_of_blocks<field_t[256]> &out) {
        hls::write_lock<field_t[256]> buffer(out);
        for (int i = 0; i < 248; i++) {
#pragma HLS PIPELINE II=1
            buffer[i] = in.read();
        }
    }

    static void reorderer_reader(hls::stream_of_blocks<field_t[256]> &in, hls::stream<field_t> &out) {
        const int reorder_table[248] = {
            241, 240, 243, 242, 245, 244, 247, 246,
            225, 224, 227, 226, 229, 228, 231, 230, 233, 232, 235, 234, 237, 236, 239, 238,
            193, 192, 195, 194, 197, 196, 199, 198, 201, 200, 203, 202, 205, 204, 207, 206, 209, 208, 211, 210,
            213, 212, 215, 214, 217, 216, 219, 218, 221, 220, 223, 222,
            129, 128, 131, 130, 133, 132, 135, 134, 137, 136, 139, 138, 141, 140, 143, 142, 145, 144, 147, 146,
            149, 148, 151, 150, 153, 152, 155, 154, 157, 156, 159, 158, 161, 160, 163, 162, 165, 164, 167, 166,
            169, 168, 171, 170, 173, 172, 175, 174, 177, 176, 179, 178, 181, 180, 183, 182, 185, 184, 187, 186,
            189, 188, 191, 190,
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27,
            26, 29, 28, 31, 30, 33, 32, 35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 49, 48, 51, 50,
            53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62, 65, 64, 67, 66, 69, 68, 71, 70, 73, 72, 75, 74, 77,
            76, 79, 78, 81, 80, 83, 82, 85, 84, 87, 86, 89, 88, 91, 90, 93, 92, 95, 94, 97, 96, 99, 98, 101,
            100, 103, 102, 105, 104, 107, 106, 109, 108, 111, 110, 113, 112, 115, 114, 117, 116, 119, 118, 121,
            120, 123, 122, 125, 124, 127, 126
        };

        hls::read_lock<field_t[256]> buffer(in);
        for (int i: reorder_table) {
#pragma HLS PIPELINE II=1
            out << buffer[i];
        }
    }

    static void reorderer(hls::stream<field_t> &in, hls::stream<field_t> &out) {
#pragma HLS DATAFLOW
        hls::stream_of_blocks<field_t[256], 3> buffer;
        reorderer_writer(in, buffer);
        reorderer_reader(buffer, out);
    }

    class d_untree {
        static constexpr char COMPONENT_NAME[] = "d_untree";

    public:
        static void d_untree_inst(hls::stream<field_t> &from_d_tree, hls::stream<field_t> &from_inverter,
                                  hls::stream<field_t> &out) {
            hls_thread_local hls::stream<field_t, 64> s_recycle("s_d_untree_recycle");
            hls_thread_local hls::stream<exec_pkt_t, 2> s_dispatch_to_mult("s_d_untree_dispatch_to_mult");

            hls_thread_local hls::task t_dispatch(dispatch, from_d_tree, from_inverter, s_recycle, s_dispatch_to_mult);
            hls_thread_local hls::task t_mult(mult, s_dispatch_to_mult, s_recycle, out);
        }

    private:
        struct exec_pkt_t {
            field_t a;
            field_t b;
            bool final = false;
        };

        static void dispatch(
            hls::stream<field_t> &from_d_tree,
            hls::stream<field_t> &from_inverter,
            hls::stream<field_t> &recycle_in,
            hls::stream<exec_pkt_t> &out
        ) {
            for (int i = 0; i < 124; i++) {
#pragma HLS PIPELINE II=2
                if (i < 4) {
                    field_t op_b = from_inverter.read();
                    out << exec_pkt_t{from_d_tree.read(), op_b, false};
                    out << exec_pkt_t{from_d_tree.read(), op_b, false};
                } else {
                    field_t op_b = recycle_in.read();
                    out << exec_pkt_t{from_d_tree.read(), op_b, i >= 60};
                    out << exec_pkt_t{from_d_tree.read(), op_b, i >= 60};
                }
            }
        }

        static void mult(
            hls::stream<exec_pkt_t> &in,
            hls::stream<field_t> &recycle_out,
            hls::stream<field_t> &out
        ) {
#pragma HLS PIPELINE II=1 style=frp
#pragma HLS LATENCY max=38
#pragma HLS EXPRESSION_BALANCE off
            exec_pkt_t pkt = in.read();
#ifndef __XOR_STUB__
            field_t res = mod_mult<CURVE>::ll_mod_mult(pkt.a, pkt.b);
#else
            field_t res = pkt.a ^ pkt.b;
#endif
            if (pkt.final) {
                out << res;
            } else {
                recycle_out << res;
            }
        }
    };

    template<int COMPONENT_ID>
    class inverter {
        static constexpr char COMPONENT_NAME[] = "inverter";

    public:
        static void inverter_inst(hls::stream<field_t> &in, hls::stream<field_t> &out) {
            hls_thread_local hls::stream<state_t, 4> s_inverter_muxer_to_pipe("s_inverter_muxer_to_pipe");
            hls_thread_local hls::stream<state_t, 4> s_pipe_to_inverter_muxer("s_pipe_to_inverter_muxer");
            hls_thread_local hls::stream<inverter_to_reducer_t, 4> s_muxer_to_reducer("s_muxer_to_reducer");
            hls_thread_local hls::stream<reducer_t, 4> s_reducer_stage_0_to_muxer("s_reducer_stage_0_to_muxer");
            hls_thread_local hls::stream<reducer_t, 4> s_reducer_muxer_to_stage_1("s_reducer_muxer_to_stage_1");
            hls_thread_local hls::stream<reducer_t, 4> s_stage_1_to_reducer_muxer("s_stage_1_to_reducer_muxer");
            hls_thread_local hls::stream<reducer_t, 4> s_reducer_muxer_to_stage_2("s_reducer_muxer_to_stage_2");

            hls_thread_local hls::task t_pipe(pipe, s_inverter_muxer_to_pipe, s_pipe_to_inverter_muxer);
            hls_thread_local hls::task t_inverter_muxer(inverter_muxer, in, s_pipe_to_inverter_muxer,
                                                        s_inverter_muxer_to_pipe, s_muxer_to_reducer);
            hls_thread_local hls::task t_reducer_stage_0(reducer_stage_0, s_muxer_to_reducer,
                                                         s_reducer_stage_0_to_muxer);
            hls_thread_local hls::task t_reducer_stage_1(reducer_stage_1, s_reducer_muxer_to_stage_1,
                                                         s_stage_1_to_reducer_muxer);
            hls_thread_local hls::task t_reducer_muxer(reducer_muxer, s_reducer_stage_0_to_muxer,
                                                       s_stage_1_to_reducer_muxer, s_reducer_muxer_to_stage_1,
                                                       s_reducer_muxer_to_stage_2);
            hls_thread_local hls::task t_reducer_stage_2(reducer_stage_2, s_reducer_muxer_to_stage_2, out);
        }

    private:
        struct state_t {
            mp_16_384_t a;
            mp_16_384_t b;
            mp_16_384_t u;
            mp_16_384_t v;
            int shift_count = 0;
            int step_count = 0;
        };

        static void pipe(hls::stream<state_t> &in, hls::stream<state_t> &out) {
#pragma HLS PIPELINE II=1 style=frp
#pragma HLS LATENCY max=12
            state_t state = in.read();
            state.step_count++;
            state = step(state);
            state = step(state);
            state = step(state);
            state = step(state);
            out << state;
        }

        static void inverter_muxer(
            hls::stream<field_t> &from_d_tree,
            hls::stream<state_t> &from_pipe,
            hls::stream<state_t> &to_pipe,
            hls::stream<inverter_to_reducer_t> &to_reducer
        ) {
#pragma HLS PIPELINE II=1
            if (!from_pipe.empty()) {
                state_t state = from_pipe.read();
                if (state.step_count == 81) {
                    to_reducer << inverter_to_reducer_t{state.v.to_ap_uint(), state.shift_count};
                } else {
                    to_pipe << state;
                }
            } else if (!from_d_tree.empty()) {
                auto a = mp_16_384_t(from_d_tree.read());
                auto b = mp_16_384_t(base_field_t<CURVE>::P);
                auto u = mp_16_384_t(field_t(1));
                auto v = mp_16_384_t(field_t(0));
                to_pipe << state_t{a, b, u, v, 0, 0};
            }
        }

        static state_t step(state_t state) {
#pragma HLS INLINE
            const ap_uint<3> shift_lookup[128] = {
                4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 1, 3, 1, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 1, 3, 1, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 1, 2, 1, 3, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 2, 1, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 1, 4, 1, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 1, 4, 1, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 2, 1, 4, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 2, 1, 4
            };

            mp_16_384_t ignore, a_minus_b, b_minus_a, u_minus_v, v_minus_u;
            const bool even = state.a.is_even();
            const bool zero = mp_16_384_t::fast_sub_no_red_inline(ignore, mp_16_384_t(0), state.a);
            const bool ge = mp_16_384_t::fast_sub_no_red_inline(a_minus_b, state.a, state.b);
            const bool le = mp_16_384_t::fast_sub_no_red_inline(b_minus_a, state.b, state.a);
            const bool done = zero || (ge && le);
            mp_16_384_t::fast_sub_no_red_inline(u_minus_v, state.u, state.v);
            mp_16_384_t::fast_sub_no_red_inline(v_minus_u, state.v, state.u);

            const ap_uint<4> a_addr_bits = state.a[0](3, 0);
            const ap_uint<3> b_addr_bits = state.b[0](3, 1);
            const ap_uint<7> addr = (a_addr_bits, b_addr_bits);
            const ap_uint<3> bits = done ? ap_uint<3>(0) : shift_lookup[addr];

            state.b = (even || ge) ? state.b : state.a;
            state.v = (even || ge) ? state.v : state.u;
            state.a = even ? state.a : (ge ? a_minus_b : b_minus_a);
            state.u = even ? state.u : (ge ? u_minus_v : v_minus_u);

            state.a.shift_right(bits.to_int());
            state.v.shift_left(bits.to_int());

            state.shift_count += bits.to_int();
            return state;
        }

        struct reducer_t {
            ap_uint<25> reduce_v[17];
            int shift_count = 0;
            ap_uint<6> cnt;
        };

        static void reducer_stage_0(hls::stream<inverter_to_reducer_t> &in, hls::stream<reducer_t> &out) {
#pragma HLS PIPELINE II=1
            inverter_to_reducer_t tmp = in.read();

            tmp.v = tmp.v[383] == 1 ? field_t(base_field_t<CURVE>::P + tmp.v) : tmp.v;
            int shift = 16 - (tmp.shift_count & 0x0F);
            ap_uint<400> v = ap_uint<400>(tmp.v) << shift;
            tmp.shift_count += shift;

            reducer_t out_pkt;
            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                out_pkt.reduce_v[i] = v((i + 1) * 24 - 1, i * 24);
            }
            out_pkt.reduce_v[16] = v(399, 384);
            out_pkt.shift_count = tmp.shift_count;
            out_pkt.cnt = 22;
            out << out_pkt;
        }

        static void reducer_stage_1(hls::stream<reducer_t> &in, hls::stream<reducer_t> &out) {
#pragma HLS PIPELINE II=1 style=frp
            const mp_24_384_t reduce_p = mp_24_384_t(base_field_t<CURVE>::P);

            reducer_t tmp = in.read();

            if (tmp.shift_count > 0) {
                hls::vector<ap_uint<16>, 17> l16;
                hls::vector<ap_uint<24>, 17> m24;
                hls::vector<ap_uint<1>, 17> h1;
                ap_uint<16> q;
                ap_uint<41> prod;

                q = tmp.reduce_v[0] * base_field_t<CURVE>::NP(15, 0);
                for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                    prod = tmp.reduce_v[i] + reduce_p[i] * q;
                    l16[i] = prod;
                    m24[i] = prod >> 16;
                    h1[i] = prod >> 40;
                }
                prod = tmp.reduce_v[16];
                l16[16] = prod;
                m24[16] = prod >> 16;
                h1[16] = prod >> 40;

                tmp.reduce_v[0] = m24[0] + (ap_uint<24>(l16[1]) << 8);
                for (int i = 1; i < 16; i++) {
#pragma HLS UNROLL
                    tmp.reduce_v[i] = h1[i - 1] + m24[i] + (ap_uint<24>(l16[i + 1]) << 8);
                }
                tmp.reduce_v[16] = h1[15] + m24[16] + (ap_uint<25>(h1[16]) << 24);

                tmp.shift_count -= 16;
            }

            if (tmp.shift_count > 0) {
                hls::vector<ap_uint<16>, 17> l16;
                hls::vector<ap_uint<24>, 17> m24;
                hls::vector<ap_uint<1>, 17> h1;
                ap_uint<16> q;
                ap_uint<41> prod;

                q = tmp.reduce_v[0] * base_field_t<CURVE>::NP(15, 0);
                for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                    prod = tmp.reduce_v[i] + reduce_p[i] * q;
                    l16[i] = prod;
                    m24[i] = prod >> 16;
                    h1[i] = prod >> 40;
                }
                prod = tmp.reduce_v[16];
                l16[16] = prod;
                m24[16] = prod >> 16;
                h1[16] = prod >> 40;

                tmp.reduce_v[0] = m24[0] + (ap_uint<24>(l16[1]) << 8);
                for (int i = 1; i < 16; i++) {
#pragma HLS UNROLL
                    tmp.reduce_v[i] = h1[i - 1] + m24[i] + (ap_uint<24>(l16[i + 1]) << 8);
                }
                tmp.reduce_v[16] = h1[15] + m24[16] + (ap_uint<25>(h1[16]) << 24);

                tmp.shift_count -= 16;
            }

            out << tmp;
        }

        static void reducer_stage_2(hls::stream<reducer_t> &in, hls::stream<field_t> &out) {
#pragma HLS PIPELINE II=1
            reducer_t tmp = in.read();

            field_t res;
            for (int i = 0; i < 15; i++) {
#pragma HLS UNROLL
                tmp.reduce_v[i + 1] += tmp.reduce_v[i][24];
            }

            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                res((i + 1) * 24 - 1, i * 24) = tmp.reduce_v[i];
            }
            out << res;
        }

        static void reducer_muxer(
            hls::stream<reducer_t> &from_stage_0,
            hls::stream<reducer_t> &from_stage_1,
            hls::stream<reducer_t> &to_stage_1,
            hls::stream<reducer_t> &to_stage_2
        ) {
#pragma HLS PIPELINE II=1
            if (!from_stage_1.empty()) {
                reducer_t tmp = from_stage_1.read();
                tmp.cnt--;
                if (tmp.cnt == 0) {
                    to_stage_2 << tmp;
                } else {
                    to_stage_1 << tmp;
                }
            } else if (!from_stage_0.empty()) {
                to_stage_1 << from_stage_0.read();
            }
        }
    };
};
