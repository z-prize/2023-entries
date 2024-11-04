

`ifndef SCHEDULER
`define SCHEDULER

module scheduler #(
  parameter int P_DATA_PNT_W = 377,
  parameter int P_NUM_AFF_COORD = 3,
  parameter int P_RED_SCLR_W = 13,
  parameter int P_NUM_WIN    = 7
) (
  input clk,
  input rst_n,

  // AXIS input stream of (point, red. scalar, index i.e. which bucket set)
  input         [P_NUM_AFF_COORD * P_DATA_PNT_W - 1:0] prsi_point_i,
  input  signed [                  P_RED_SCLR_W - 1:0] prsi_red_scalar_i,
  input         [             $clog2(P_NUM_WIN) - 1:0] prsi_index_i,
  input                                                prsi_valid_i,
  input                                                prsi_last_i,
  output logic                                         prsi_ready_o,

  // Acceptence interface: notifies merger which reduced scalars have been accepted and sent to curve adder
  output logic [                      P_NUM_WIN - 1:0] accept_flags_o,
  output logic                                         accept_valid_o,

  // Interface with accumulator controller
  output       [P_NUM_AFF_COORD * P_DATA_PNT_W - 1:0] acc_point_o,
  output logic [                  P_RED_SCLR_W - 2:0] acc_bucket_addr_o,
  output logic [             $clog2(P_NUM_WIN) - 1:0] acc_bucket_set_addr_o,
  output logic                                        acc_add_sub_o,
  output logic                                        acc_valid_o,

  // Interface with feedback from bucket memory: indicates curve addition is completed for specific bucket (to clear flag)
  input [                         P_RED_SCLR_W - 2:0] bm_bucket_addr_i,
  input [                    $clog2(P_NUM_WIN) - 1:0] bm_bucket_set_addr_i,
  input                                               bm_valid_i,

  // Signal from Acc. controller indicating bucket memory has been initialized with special initial EC points
  input init_done_i
);

  // Address widths for bucket(set) and flag memory
  localparam int LP_BUCKET_ADDR_WIDTH = P_RED_SCLR_W - 1;
  localparam int LP_BUCKET_SET_ADDR_WIDTH = $clog2(P_NUM_WIN);
  localparam int LP_FLAG_ADDR_WIDTH = LP_BUCKET_SET_ADDR_WIDTH + LP_BUCKET_ADDR_WIDTH;
  
  // Latencies and delays for pipelining
  localparam int LP_FLAG_MEM_READ_LATENCY = 3;

  logic                                           calc_is_dummy;
  logic                                           calc_is_zero;
  logic                                           calc_add_sub_i;
  logic unsigned [    LP_BUCKET_ADDR_WIDTH - 1:0] calc_bucket_addr_i;
  logic unsigned [LP_BUCKET_SET_ADDR_WIDTH - 1:0] calc_bucket_set_addr_i;
  logic                                           calc_is_dummy_dld;
  logic                                           calc_is_zero_dld;
  logic                                           calc_add_sub_i_dld;
  logic unsigned [    LP_BUCKET_ADDR_WIDTH - 1:0] calc_bucket_addr_i_dld;
  logic unsigned [LP_BUCKET_SET_ADDR_WIDTH - 1:0] calc_bucket_set_addr_i_dld;

  logic unsigned [      LP_FLAG_ADDR_WIDTH - 1:0] mark_check_bucket_flag_addr;
  logic                                           mark_check_bucket_flag_we;
  logic                                           mark_check_bucket_flag_in;
  logic                                           mark_check_bucket_flag_out;
  logic unsigned [      LP_FLAG_ADDR_WIDTH - 1:0] clear_bucket_flag_addr;
  logic                                           clear_bucket_flag_we;

  // State machine handling collision detection
  if (P_DATA_PNT_W == 377) begin
    `include "scheduler_ctrl_377.svh"
  end else begin
    `include "scheduler_ctrl_381.svh"
  end

  shiftreg #(
    .SHIFT (1 + LP_FLAG_MEM_READ_LATENCY),
    .DATA_W(1)
  ) i_shiftreg_calc_add_sub (
    .clk,
    .data_i(calc_add_sub_i),
    .data_o(calc_add_sub_i_dld)
  );

  shiftreg #(
    .SHIFT (1 + LP_FLAG_MEM_READ_LATENCY),
    .DATA_W(LP_BUCKET_ADDR_WIDTH)
  ) i_shiftreg_calc_bucket_addr (
    .clk,
    .data_i(calc_bucket_addr_i),
    .data_o(calc_bucket_addr_i_dld)
  );

  shiftreg #(
    .SHIFT (1 + LP_FLAG_MEM_READ_LATENCY),
    .DATA_W(1)
  ) i_shiftreg_calc_is_dummy (
    .clk,
    .data_i(calc_is_dummy),
    .data_o(calc_is_dummy_dld)
  );

  shiftreg #(
    .SHIFT (1 + LP_FLAG_MEM_READ_LATENCY),
    .DATA_W(1)
  ) i_shiftreg_calc_is_zero (
    .clk,
    .data_i(calc_is_zero),
    .data_o(calc_is_zero_dld)
  );

  shiftreg #(
    .SHIFT (1 + LP_FLAG_MEM_READ_LATENCY),
    .DATA_W(LP_BUCKET_SET_ADDR_WIDTH)
  ) i_shiftreg_calc_bucket_set_addr (
    .clk,
    .data_i(calc_bucket_set_addr_i),
    .data_o(calc_bucket_set_addr_i_dld)
  );

  // Delay point to align with output 
  shiftreg #(
    .SHIFT (1 + LP_FLAG_MEM_READ_LATENCY + 1),
    .DATA_W(P_NUM_AFF_COORD * P_DATA_PNT_W)
  ) i_shiftreg_point (
    .clk,
    .data_i(prsi_point_i),
    .data_o(acc_point_o)
  );

  assign calc_bucket_set_addr_i = prsi_index_i;

  // True Dual-Port BRAM to keep track of reduced scalars in the pipeline:
  // It is configured such that in a single cycle a red. scalar is checked
  // for collision, flagged as scheduled and cleared when finished by the
  // curveadder pipeline.
  xpm_memory_tdpram #(
    .ADDR_WIDTH_A      (LP_BUCKET_SET_ADDR_WIDTH + LP_BUCKET_ADDR_WIDTH),
    .ADDR_WIDTH_B      (LP_BUCKET_SET_ADDR_WIDTH + LP_BUCKET_ADDR_WIDTH),
    .BYTE_WRITE_WIDTH_A(1),
    .BYTE_WRITE_WIDTH_B(1),
    .CLOCKING_MODE     ("common_clock"),
    .MEMORY_PRIMITIVE  ("block"),
    .MEMORY_SIZE       (2 ** (LP_BUCKET_SET_ADDR_WIDTH + LP_BUCKET_ADDR_WIDTH)),
    .READ_DATA_WIDTH_A (1),
    .READ_DATA_WIDTH_B (1),
    .READ_LATENCY_A    (LP_FLAG_MEM_READ_LATENCY),
    .READ_LATENCY_B    (LP_FLAG_MEM_READ_LATENCY),
    .WRITE_DATA_WIDTH_A(1),
    .WRITE_DATA_WIDTH_B(1),
    .WRITE_MODE_A      ("read_first"),
    .WRITE_MODE_B      ("read_first")
  ) i_tdpram_flag_mem (
    .douta (mark_check_bucket_flag_out),
    .addra (mark_check_bucket_flag_addr),
    .addrb (clear_bucket_flag_addr),
    .clka  (clk),
    .clkb  (clk),
    .dina  (mark_check_bucket_flag_in),
    .dinb  (1'b0),
    .ena   (1'b1),
    .enb   (1'b1),
    .regcea(1'b1),
    .regceb(1'b1),
    .rsta  (~rst_n),
    .rstb  (~rst_n),
    .sleep (1'b0),
    .wea   (mark_check_bucket_flag_we),
    .web   (clear_bucket_flag_we)
  );

endmodule

`endif
