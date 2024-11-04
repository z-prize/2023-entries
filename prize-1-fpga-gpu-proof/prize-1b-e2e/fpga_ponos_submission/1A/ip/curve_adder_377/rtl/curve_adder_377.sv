`ifndef CURVE_ADDER_377_SV
`define CURVE_ADDER_377_SV

module curve_adder_377
  import krnl_msm_377_pkg::*;
#(
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // port from bucket memory
  // bucket_mem_i in Extended Projective Twisted Edwards coordinates (X:Y:T:Z)
  input [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] bucket_mem_i,

  // port from acc controller
  // data_point_i in Extended Affine Twisted Edwards coordinates (x, y, u)
  input [P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] data_point_i,
  input                                    add_sub_i,          // 1: add, 0: sub
  input [                P_RED_SCLR_W-2:0] bucket_addr_i,
  input [           $clog2(P_NUM_WIN)-1:0] bucket_set_addr_i,
  input                                    valid_i,

  // port to bucket memory
  // sum_o in Extended Projective Twisted Edwards coordinates (X:Y:T:Z)
  output logic [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] sum_o,
  output logic [                 P_RED_SCLR_W-2:0] bucket_addr_o,
  output logic [            $clog2(P_NUM_WIN)-1:0] bucket_set_addr_o,
  output logic                                     valid_o
);


  localparam int W = P_DATA_PNT_W + 1;  // ops in the range [0...2Q) for montgomery
  localparam int LP_SUB_ADD_DLY = 2;
  localparam int LP_SUB_ADD_DLY_RED = 4;
  localparam int LP_MUL_DLY = 35;
  localparam int LP_CURVE_DLY = 2*LP_MUL_DLY + 2*LP_SUB_ADD_DLY + 1; // (+1: add_sub logic registered)

  logic [W-1:0] bucket_x;
  logic [W-1:0] bucket_y;
  logic [W-1:0] bucket_t;
  logic [W-1:0] bucket_z;
  logic [W-1:0] a2;  // (y2 - x2)
  logic [W-1:0] b2;  // (y2 + x2)
  logic [W-1:0] point_u;
  logic [W-1:0] a1;
  logic [W-1:0] b1;
  logic [W-1:0] a2_dly_1;
  logic [W-1:0] b2_dly_1;
  logic [W-1:0] a;
  logic [W-1:0] b;
  logic [W-1:0] c;
  logic [W-1:0] d;
  logic [W-1:0] d_dly_1;
  logic [W-1:0] e;
  logic [W-1:0] f;
  logic [W-1:0] f_dly_1;
  logic [W-1:0] g;
  logic [W-1:0] g_dly_1;
  logic [W-1:0] h;
  logic         rst;

  assign rst = ~rst_n;

  // add_sub logic registered
  always_ff @(posedge clk) begin
    // 1: add, 0: sub.
    a2 <= (~add_sub_i) ? data_point_i[P_DATA_PNT_W+:P_DATA_PNT_W] : data_point_i[0+:P_DATA_PNT_W];
    b2 <= (~add_sub_i) ? data_point_i[0+:P_DATA_PNT_W] : data_point_i[P_DATA_PNT_W+:P_DATA_PNT_W];
    point_u   <= (~add_sub_i & |data_point_i[2*P_DATA_PNT_W+:P_DATA_PNT_W]) ? Q - data_point_i[2*P_DATA_PNT_W+:P_DATA_PNT_W] : data_point_i[2*P_DATA_PNT_W+:P_DATA_PNT_W];
    // register the remaining inputs 
    bucket_x <= bucket_mem_i[0+:P_DATA_PNT_W];
    bucket_y <= bucket_mem_i[P_DATA_PNT_W+:P_DATA_PNT_W];
    bucket_t <= bucket_mem_i[2*P_DATA_PNT_W+:P_DATA_PNT_W];
    bucket_z <= bucket_mem_i[3*P_DATA_PNT_W+:P_DATA_PNT_W];
  end

  // a1 = Y1-X1
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(0)
  ) i_sub_a1 (
    .clk,
    .rst,
    .in0 (bucket_y),
    .in1 (bucket_x),
    .out0(a1)
  );

  // a2 delay
  shift_reg_ca #(
    .D(LP_SUB_ADD_DLY),
    .W(W)
  ) i_shiftreg_a2_dly_1 (
    .clk,
    .rst,
    .in0 (a2),
    .out0(a2_dly_1)
  );

  // b1 = Y1+X1
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(1),
    .RED(0)
  ) i_add_b1 (
    .clk,
    .rst,
    .in0 (bucket_x),
    .in1 (bucket_y),
    .out0(b1)
  );

  // b2 delay
  shift_reg_ca #(
    .D(LP_SUB_ADD_DLY),
    .W(W)
  ) i_shiftreg_pyx_dly_1 (
    .clk,
    .rst,
    .in0 (b2),
    .out0(b2_dly_1)
  );

  // A = a1*a2
  mul_montgomery #(
    .WI(W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),
    .T2(32'h07F7_F999)
  ) i_mul_montgomery_a (
    .clk,
    .rst,
    .in0 (a1),
    .in1 (a2_dly_1),
    .out0(a)
  );

  // B = b1*b2
  mul_montgomery #(
    .WI(W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),
    .T2(32'h07F7_F999)
  ) i_mul_montgomery_b (
    .clk,
    .rst,
    .in0 (b1),
    .in1 (b2_dly_1),
    .out0(b)
  );

  // C = T1*u2
  mul_montgomery #(
    .WI(W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),
    .T2(32'h07F7_F999)
  ) i_mul_montgomery_c (
    .clk,
    .rst,
    .in0 (bucket_t),
    .in1 (point_u),
    .out0(c)
  );

  // D = 2*Z1
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(1),
    .RED(1)
  ) i_addred_d (
    .clk,
    .rst,
    .in0 (bucket_z),
    .in1 (bucket_z),
    .out0(d)
  );

  // d delay
  shift_reg_ca #(
    .D(LP_MUL_DLY - LP_SUB_ADD_DLY_RED),
    .W(W)
  ) i_shiftreg_d_dly_1 (
    .clk,
    .rst,
    .in0 (d),
    .out0(d_dly_1)
  );

  // E = B-A
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(0)
  ) i_sub_e (
    .clk,
    .rst,
    .in0 (b),
    .in1 (a),
    .out0(e)
  );

  // F = D-C
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(0),  // SUB
    .RED(0)
  ) i_subred_f (
    .clk,
    .rst,
    .in0 (d_dly_1),
    .in1 (c),
    .out0(f)
  );

  // f delay
  shift_reg_ca #(
    .D(LP_SUB_ADD_DLY),
    .W(W)
  ) i_shiftreg_f_dly_1 (
    .clk,
    .rst,
    .in0 (f),
    .out0(f_dly_1)
  );

  // G = D+C
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(1),
    .RED(0)
  ) i_add_g (
    .clk,
    .rst,
    .in0 (d_dly_1),
    .in1 (c),
    .out0(g)
  );

  // g delay
  shift_reg_ca #(
    .D(LP_SUB_ADD_DLY),
    .W(W)
  ) i_shiftreg_g_dly_1 (
    .clk,
    .rst,
    .in0 (g),
    .out0(g_dly_1)
  );

  // H = B + A
  f_add_sub #(
    .W  (W),
    .Q  (Q),
    .ADD(1),
    .RED(0)
  ) i_addpipe_h (
    .clk,
    .rst,
    .in0 (b),
    .in1 (a),
    .out0(h)
  );

  // X3 = E*F 
  mul_montgomery #(
    .WI(W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),
    .T2(32'h07F7_F999)
  ) i_mul_montgomery_x3 (
    .clk,
    .rst,
    .in0 (e),
    .in1 (f_dly_1),
    .out0(sum_o[0+:P_DATA_PNT_W])
  );

  // Y3 = G*H 
  mul_montgomery #(
    .WI(W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),
    .T2(32'h07F7_F999)
  ) i_mul_montgomery_y3 (
    .clk,
    .rst,
    .in0 (g_dly_1),
    .in1 (h),
    .out0(sum_o[P_DATA_PNT_W+:P_DATA_PNT_W])
  );

  // T3 = E*H 
  mul_montgomery #(
    .WI(W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),
    .T2(32'h07F7_F999)
  ) i_mul_montgomery_t3 (
    .clk,
    .rst,
    .in0 (e),
    .in1 (h),
    .out0(sum_o[2*P_DATA_PNT_W+:P_DATA_PNT_W])
  );

  // Z3 = F*G 
  mul_montgomery #(
    .WI(W),
    .L (3),
    .T0(32'h07F7_F999),
    .T1(32'h0000_005F),
    .T2(32'h07F7_F999)
  ) i_mul_montgomery_z3 (
    .clk,
    .rst,
    .in0 (f_dly_1),
    .in1 (g_dly_1),
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

`endif  // CURVE_ADDER_377_SV
