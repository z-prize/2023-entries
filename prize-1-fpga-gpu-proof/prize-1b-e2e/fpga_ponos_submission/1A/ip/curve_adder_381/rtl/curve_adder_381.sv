
`ifndef CURVE_ADDER_381_SV
`define CURVE_ADDER_381_SV

module curve_adder_381
  import krnl_msm_381_pkg::*;
#(
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // port from bucket memory
  // bucket_mem_i in XYZZ coordinates (X Y ZZ ZZZ)
  input [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] bucket_mem_i,

  // port from acc controller
  // data_point_i in Affine coordinates (x, y)
  input [P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] data_point_i,
  input                                    add_sub_i,          // 1: add, 0: sub
  input [                P_RED_SCLR_W-2:0] bucket_addr_i,
  input [           $clog2(P_NUM_WIN)-1:0] bucket_set_addr_i,
  input                                    valid_i,

  // port to bucket memory
  // sum_o in XYZZ coordinates (X Y ZZ ZZZ)
  output logic [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] sum_o,
  output logic [                 P_RED_SCLR_W-2:0] bucket_addr_o,
  output logic [            $clog2(P_NUM_WIN)-1:0] bucket_set_addr_o,
  output logic                                     valid_o
);

  localparam int W = P_DATA_PNT_W + 1;  // ops in the range [0...2Q) for montgomery
  localparam int LP_SUB_DLY = 4;
  localparam int LP_ADD_DLY = 4;
  localparam int LP_MUL_DLY = 35;  // Delay of multipliers with NAF
  localparam int LP_CURVE_DLY = 4*LP_MUL_DLY + LP_ADD_DLY + 4*LP_SUB_DLY + 1; // (+1: add_sub logic registered)

  logic [W-1:0] bucket_x;
  logic [W-1:0] bucket_y;
  logic [W-1:0] bucket_zz;
  logic [W-1:0] bucket_zzz;
  logic [W-1:0] point_x;
  logic [W-1:0] point_y;
  logic [W-1:0] u2;
  logic [W-1:0] s2;
  logic [W-1:0] p;
  logic [W-1:0] v;
  logic [W-1:0] pp;
  logic [W-1:0] ppp;
  logic [W-1:0] t;
  logic [W-1:0] t0;
  logic [W-1:0] t1;
  logic [W-1:0] t2;
  logic [W-1:0] x3;
  logic [W-1:0] t3;
  logic [W-1:0] t4;
  logic [W-1:0] t5;
  logic [W-1:0] zz3;
  logic [W-1:0] zzz3;
  logic [W-1:0] x1_dly_1;
  logic [W-1:0] y1_dly_1;
  logic [W-1:0] p_dly_1;
  logic [W-1:0] x1_dly_2;
  logic [W-1:0] t0_dly_1;
  logic [W-1:0] t_dly_1;
  logic [W-1:0] y1_dly_2;
  logic [W-1:0] v_dly_1;
  logic [W-1:0] t4_dly_1;
  logic [W-1:0] zz1_dly_1;
  logic [W-1:0] zzz1_dly_1;
  logic [W-1:0] t0_sqr;
  logic [W-1:0] pp_sqr;

  assign rst = ~rst_n;

  // add_sub logic registered
  always_ff @(posedge clk) begin
    // 1: add, 0: sub.
    point_x <= data_point_i[0+:P_DATA_PNT_W];
    point_y <= (~add_sub_i & |data_point_i[P_DATA_PNT_W+:P_DATA_PNT_W]) ? Q - data_point_i[P_DATA_PNT_W+:P_DATA_PNT_W] : data_point_i[P_DATA_PNT_W+:P_DATA_PNT_W];
    // register the remaining inputs 
    bucket_x <= bucket_mem_i[0+:P_DATA_PNT_W];
    bucket_y <= bucket_mem_i[P_DATA_PNT_W+:P_DATA_PNT_W];
    bucket_zz <= bucket_mem_i[2*P_DATA_PNT_W+:P_DATA_PNT_W];
    bucket_zzz <= bucket_mem_i[3*P_DATA_PNT_W+:P_DATA_PNT_W];
  end

  // x1 = bucket_x;
  // y1 = bucket_y;
  // zz1 = bucket_zz;
  // zzz1 = bucket_zzz;
  // x2 = point_x;
  // y2 = point_y;

  // U2 = X2*ZZ1 
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_u2 (
    .clk,
    .rst,
    .in0 (point_x),
    .in1 (bucket_zz),
    .out0(u2)
  );

  // S2 = Y2*ZZZ1
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_s2 (
    .clk,
    .rst,
    .in0 (point_y),
    .in1 (bucket_zzz),
    .out0(s2)
  );

  // delaying x1
  shift_reg_ca #(
    .D(LP_MUL_DLY),
    .W(W)
  ) i_shiftreg_x1_dly_1 (
    .clk,
    .rst,
    .in0 (bucket_x),
    .out0(x1_dly_1)
  );

  // delaying y1
  shift_reg_ca #(
    .D(LP_MUL_DLY),
    .W(W)
  ) i_shiftreg_y1_dly_1 (
    .clk,
    .rst,
    .in0 (bucket_y),
    .out0(y1_dly_1)
  );

  // P = U2-X1
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(1)
  ) i_subred_p (
    .clk,
    .rst,
    .in0 (u2),
    .in1 (x1_dly_1),
    .out0(p)
  );

  // V = S2-Y1 
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(1)
  ) i_subred_v (
    .clk,
    .rst,
    .in0 (s2),
    .in1 (y1_dly_1),
    .out0(v)
  );

  // PP = P^2 
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_pp (
    .clk,
    .rst,
    .in0 (p),
    .in1 (p),
    .out0(pp)
  );

  // delaying p
  shift_reg_ca #(
    .D(LP_MUL_DLY),
    .W(W)
  ) i_shiftreg_p_dly_1 (
    .clk,
    .rst,
    .in0 (p),
    .out0(p_dly_1)
  );

  // PPP = P*PP
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_ppp (
    .clk,
    .rst,
    .in0 (pp),
    .in1 (p_dly_1),
    .out0(ppp)
  );

  // delaying x1_dly_1
  shift_reg_ca #(
    .D(LP_MUL_DLY + LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_x1_dly_2 (
    .clk,
    .rst,
    .in0 (x1_dly_1),
    .out0(x1_dly_2)
  );

  // T = X1*PP
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_t (
    .clk,
    .rst,
    .in0 (x1_dly_2),
    .in1 (pp),
    .out0(t)
  );

  // t0 = V^2
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_t0 (
    .clk,
    .rst,
    .in0 (v),
    .in1 (v),
    .out0(t0)
  );

  // t1 = 2*T
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(1),  // ADD
    .RED(1)
  ) i_addred_t1 (
    .clk,
    .rst,
    .in0 (t),
    .in1 (t),
    .out0(t1)
  );

  // delaying t0
  shift_reg_ca #(
    .D(LP_MUL_DLY),
    .W(W)
  ) i_shiftreg_t0_dly_1 (
    .clk,
    .rst,
    .in0 (t0),
    .out0(t0_dly_1)
  );

  // t2 = t0-PPP
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(1)
  ) i_subred_t2 (
    .clk,
    .rst,
    .in0 (t0_dly_1),
    .in1 (ppp),
    .out0(t2)
  );

  //equal delay on computing t1 and t2 (addred delay = subred delay)
  // X3 = t2-t1
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(1)
  ) i_subred_x3 (
    .clk,
    .rst,
    .in0 (t2),
    .in1 (t1),
    .out0(x3)
  );

  // delaying x3
  shift_reg_ca #(
    .D(LP_MUL_DLY + 2 * LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_x3 (
    .clk,
    .rst,
    .in0 (x3),
    .out0(sum_o[0+:P_DATA_PNT_W])
  );

  // delaying t
  shift_reg_ca #(
    .D(LP_ADD_DLY + LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_t_dly_1 (
    .clk,
    .rst,
    .in0 (t),
    .out0(t_dly_1)
  );

  // t3 = T-X3
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(1)
  ) i_subred_t3 (
    .clk,
    .rst,
    .in0 (t_dly_1),
    .in1 (x3),
    .out0(t3)
  );

  // delaying y1_dly_1
  shift_reg_ca #(
    .D(2 * LP_MUL_DLY + LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_y1_dly_2 (
    .clk,
    .rst,
    .in0 (y1_dly_1),
    .out0(y1_dly_2)
  );

  // t4 = Y1*PPP 
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_t4 (
    .clk,
    .rst,
    .in0 (y1_dly_2),
    .in1 (ppp),
    .out0(t4)
  );

  // delaying v
  shift_reg_ca #(
    .D(2 * LP_MUL_DLY + LP_ADD_DLY + 2 * LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_v_dly_1 (
    .clk,
    .rst,
    .in0 (v),
    .out0(v_dly_1)
  );

  // t5 = V*t3
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_t5 (
    .clk,
    .rst,
    .in0 (v_dly_1),
    .in1 (t3),
    .out0(t5)
  );

  // delaying t4
  shift_reg_ca #(
    .D(LP_ADD_DLY + 2 * LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_t4_dly_1 (
    .clk,
    .rst,
    .in0 (t4),
    .out0(t4_dly_1)
  );

  // Y3 = t5-t4 
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(1)
  ) i_subred_y3 (
    .clk,
    .rst,
    .in0 (t5),
    .in1 (t4_dly_1),
    .out0(sum_o[P_DATA_PNT_W+:P_DATA_PNT_W])
  );

  // delaying zz1
  shift_reg_ca #(
    .D(2 * LP_MUL_DLY + LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_zz1_dly_1 (
    .clk,
    .rst,
    .in0 (bucket_zz),
    .out0(zz1_dly_1)
  );

  // ZZ3 = ZZ1*PP
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_zz3 (
    .clk,
    .rst,
    .in0 (zz1_dly_1),
    .in1 (pp),
    .out0(zz3)
  );

  // delaying zz3
  shift_reg_ca #(
    .D(LP_MUL_DLY + LP_ADD_DLY + 3 * LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_sum_zz (
    .clk,
    .rst,
    .in0 (zz3),
    .out0(sum_o[2*P_DATA_PNT_W+:P_DATA_PNT_W])
  );

  // delaying zzz1
  shift_reg_ca #(
    .D(3 * LP_MUL_DLY + LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_zzz1_dly_1 (
    .clk,
    .rst,
    .in0 (bucket_zzz),
    .out0(zzz1_dly_1)
  );

  // ZZZ3 = ZZZ1*PPP
  mul_montgomery #(
    .WI(W),
    .WQ(P_DATA_PNT_W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),  // NAF
    .T2(32'h07F7_F999),
    .Q (Q),
    .QP(QP)
  ) i_mul_montgomery_zzz3 (
    .clk,
    .rst,
    .in0 (zzz1_dly_1),
    .in1 (ppp),
    .out0(zzz3)
  );

  // delaying zzz3
  shift_reg_ca #(
    .D(LP_ADD_DLY + 3 * LP_SUB_DLY),
    .W(W)
  ) i_shiftreg_sum_zzz (
    .clk,
    .rst,
    .in0 (zzz3),
    .out0(sum_o[3*P_DATA_PNT_W+:P_DATA_PNT_W])
  );

  //*******************************************************//
  // Bypassing inputs:
  // valid
  shift_reg_ca #(
    .D(LP_CURVE_DLY),
    .W(1)
  ) i_shiftreg_valid (
    .clk,
    .rst,
    .in0 (valid_i),
    .out0(valid_o)
  );

  // bucket_addr
  shift_reg_ca #(
    .D(LP_CURVE_DLY),
    .W(P_RED_SCLR_W - 1)
  ) i_shiftreg_addr (
    .clk,
    .rst,
    .in0 (bucket_addr_i),
    .out0(bucket_addr_o)
  );

  // bucket_set_addr
  shift_reg_ca #(
    .D(LP_CURVE_DLY),
    .W($clog2(P_NUM_WIN))
  ) i_shiftreg_set_addr (
    .clk,
    .rst,
    .in0 (bucket_set_addr_i),
    .out0(bucket_set_addr_o)
  );

endmodule

`endif  // CURVE_ADDER_381_SV
