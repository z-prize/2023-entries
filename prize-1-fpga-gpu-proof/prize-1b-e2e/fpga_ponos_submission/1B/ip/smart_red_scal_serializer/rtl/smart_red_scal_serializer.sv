//////////////////////////////////////////////////////////////////////////////////
// (C) Copyright Ponos Technology 2023
// All Rights Reserved
// *** Ponos Technology Confidential ***
//////////////////////////////////////////////////////////////////////////////////

`ifndef SMART_RED_SCAL_SERIALIZER_SV
`define SMART_RED_SCAL_SERIALIZER_SV

module smart_red_scal_serializer #(
  parameter unsigned P_DATA_PNT_W = 377,
  parameter unsigned P_RED_SCAL_W = 13,
  parameter unsigned P_NUM_AFF_COORD = 3,
  parameter unsigned P_NUM_WIN = 7
) (
  input clk,
  input rst_n,

  /* verilog_format: off */
  input  [P_NUM_AFF_COORD * P_DATA_PNT_W-1:0] data_pnt_i,
  input  [   P_NUM_WIN-1:0][P_RED_SCAL_W-1:0] red_scal_array_i,
  input  [                     P_NUM_WIN-1:0] flags_i,
  input                                       valid_i,
  output                                      ready_o,

  output [P_NUM_AFF_COORD * P_DATA_PNT_W-1:0] data_pnt_o,
  output [                  P_RED_SCAL_W-1:0] red_scal_o,
  output [             $clog2(P_NUM_WIN)-1:0] window_o,
  output                                      valid_o,
  output                                      last_o,
  input                                       ready_i
);

  logic [P_NUM_AFF_COORD * P_DATA_PNT_W-1:0] data_pnt_q;
  logic [   P_NUM_WIN-1:0][P_RED_SCAL_W-1:0] red_scal_array_q;
  logic [                     P_NUM_WIN-1:0] flags;
  logic [                     P_NUM_WIN-1:0] flags_q;
  logic                                      valid_q;
  logic                                      last_flag;
  /* verilog_format: on */

  // Latch inputs
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      flags_q <= '{default: '0};
      valid_q <= 1'b0;
    end else begin
      if (valid_i && ready_o) begin
        data_pnt_q       <= data_pnt_i;
        red_scal_array_q <= red_scal_array_i;
        valid_q          <= valid_i;
      end
      if (!valid_i && ready_o) begin
        valid_q <= 1'b0;
      end
      if (ready_i) begin
        flags_q <= flags;
      end
    end
  end

  // ********** BLS12-377 (6 windows) **********
  if (P_NUM_WIN == 6) begin
    assign last_flag = (flags_q == 6'b100000) | (flags_q == 6'b010000) |
                       (flags_q == 6'b001000) | (flags_q == 6'b000100) |
                       (flags_q == 6'b000010) | (flags_q == 6'b000001) | (flags_q == 6'b0);

    assign {window_o, red_scal_o} = (flags_q[5]   == 1'b1)      ? {5, red_scal_array_q[5]} :
                                    (flags_q[5:4] == 2'b01)     ? {4, red_scal_array_q[4]} :
                                    (flags_q[5:3] == 3'b001)    ? {3, red_scal_array_q[3]} :
                                    (flags_q[5:2] == 4'b0001)   ? {2, red_scal_array_q[2]} :
                                    (flags_q[5:1] == 5'b00001)  ? {1, red_scal_array_q[1]} :
                                    (flags_q      == 6'b000001) ? {0, red_scal_array_q[0]} :
                                                                  {6, {P_RED_SCAL_W{1'b0}}};

    assign flags = (last_flag)               ? flags_i :
                   (flags_q[5])              ? {1'b0, flags_q[4:0]} :
                   (flags_q[5:4] == 2'b01)   ? {2'b0, flags_q[3:0]} :
                   (flags_q[5:3] == 3'b001)  ? {3'b0, flags_q[2:0]} :
                   (flags_q[5:2] == 4'b0001) ? {4'b0, flags_q[1:0]} :
                                               {5'b0, flags_q[0]};

    // ********** BLS12-377 (7 windows) **********
  end else if (P_NUM_WIN == 7) begin
    assign last_flag = (flags_q == 7'b1000000) | (flags_q == 7'b0100000) | (flags_q == 7'b0010000) |
                       (flags_q == 7'b0001000) | (flags_q == 7'b0000100) |
                       (flags_q == 7'b0000010) | (flags_q == 7'b0000001) | (flags_q == 7'b0);

    assign {window_o, red_scal_o} = (flags_q[6])                 ? {6, red_scal_array_q[6]} :
                                    (flags_q[6:5] == 2'b01)      ? {5, red_scal_array_q[5]} :
                                    (flags_q[6:4] == 3'b001)     ? {4, red_scal_array_q[4]} :
                                    (flags_q[6:3] == 4'b0001)    ? {3, red_scal_array_q[3]} :
                                    (flags_q[6:2] == 5'b00001)   ? {2, red_scal_array_q[2]} :
                                    (flags_q[6:1] == 6'b000001)  ? {1, red_scal_array_q[1]} :
                                    (flags_q      == 7'b0000001) ? {0, red_scal_array_q[0]} :
                                                                   {7, {P_RED_SCAL_W{1'b0}}};

    assign flags = (last_flag)                ? flags_i :
                   (flags_q[6])               ? {1'b0, flags_q[5:0]} :
                   (flags_q[6:5] == 2'b01)    ? {2'b0, flags_q[4:0]} :
                   (flags_q[6:4] == 3'b001)   ? {3'b0, flags_q[3:0]} :
                   (flags_q[6:3] == 4'b0001)  ? {4'b0, flags_q[2:0]} :
                   (flags_q[6:2] == 5'b00001) ? {5'b0, flags_q[1:0]} :
                                                {6'b0, flags_q[0]};

    // ********** BLS12-381 (10 windows) **********
  end else if (P_NUM_WIN == 10) begin
    assign last_flag = (flags_q == 10'b1000000000) | (flags_q == 10'b0100000000) | 
                       (flags_q == 10'b0010000000) | (flags_q == 10'b0001000000) | 
                       (flags_q == 10'b0000100000) | (flags_q == 10'b0000010000) |
                       (flags_q == 10'b0000001000) | (flags_q == 10'b0000000100) |
                       (flags_q == 10'b0000000010) | (flags_q == 10'b0000000001) | 
                       (flags_q == 10'b0);

    assign {window_o, red_scal_o} = (flags_q[9])                     ? {9, red_scal_array_q[9]} :
                                    (flags_q[9:8] ==  2'b01)         ? {8, red_scal_array_q[8]} :
                                    (flags_q[9:7] ==  3'b001)        ? {7, red_scal_array_q[7]} :
                                    (flags_q[9:6] ==  4'b0001)       ? {6, red_scal_array_q[6]} :
                                    (flags_q[9:5] ==  5'b00001)      ? {5, red_scal_array_q[5]} :
                                    (flags_q[9:4] ==  6'b000001)     ? {4, red_scal_array_q[4]} :
                                    (flags_q[9:3] ==  7'b0000001)    ? {3, red_scal_array_q[3]} :
                                    (flags_q[9:2] ==  8'b00000001)   ? {2, red_scal_array_q[2]} :
                                    (flags_q[9:1] ==  9'b000000001)  ? {1, red_scal_array_q[1]} :
                                    (flags_q      == 10'b0000000001) ? {0, red_scal_array_q[0]} :
                                                                       {10, {P_RED_SCAL_W{1'b0}}};

    assign flags = (last_flag) ? flags_i :
                   (flags_q[9])                  ? {1'b0, flags_q[8:0]} :
                   (flags_q[9:8] == 2'b01)       ? {2'b0, flags_q[7:0]} :
                   (flags_q[9:7] == 3'b001)      ? {3'b0, flags_q[6:0]} :
                   (flags_q[9:6] == 4'b0001)     ? {4'b0, flags_q[5:0]} :
                   (flags_q[9:5] == 5'b00001)    ? {5'b0, flags_q[4:0]} :
                   (flags_q[9:4] == 6'b000001)   ? {6'b0, flags_q[3:0]} :
                   (flags_q[9:3] == 7'b0000001)  ? {7'b0, flags_q[2:0]} :
                   (flags_q[9:2] == 8'b00000001) ? {8'b0, flags_q[1:0]} :
                                                   {9'b0, flags_q[0]};
  end

  assign ready_o = last_flag & ready_i;
  assign data_pnt_o = data_pnt_q;
  assign valid_o = valid_q;
  assign last_o = last_flag;

  // pragma translate_off
`ifndef VERILATOR
  initial begin : p_assertions
    assert ((P_NUM_WIN == 6) || (P_NUM_WIN == 7) || (P_NUM_WIN == 10))
    else $fatal(1, "SMART_SERIALISER number of windows has to 6, 7 or 10!");
  end
`endif
  // pragma translate_on
endmodule

`endif  // SMART_RED_SCAL_SERIALIZER_SV
