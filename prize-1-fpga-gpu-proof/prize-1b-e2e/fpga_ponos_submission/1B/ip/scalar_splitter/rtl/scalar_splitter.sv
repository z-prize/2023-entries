///////////////////////////////////////////////////////////////////////////////////////////////////
// (C) Copyright Ponos Technology 2024
// All Rights Reserved
// *** Ponos Technology Confidential ***
//
// Description: AXI-Stream compliant "reduced scalar splitter". 
// Parameterizable to split for either the BLS12-377 or the BLS12-381
///////////////////////////////////////////////////////////////////////////////////////////////////
`ifndef SCALAR_SPLITTER_SV
`define SCALAR_SPLITTER_SV

module scalar_splitter #(
  parameter unsigned P_NUM_ACCU = 3,  // 3: BLS12-377, 2: BLS12-381
  parameter unsigned P_NUM_WIN  = 7,  // 7: BLS12-377 , 10: BLS12-381
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // Input full scalars 
  input        [          P_FUL_SCLR_W-1:0] axis_s_data_i,
  input                                     axis_s_valid_i,
  input                                     axis_s_last_i,
  output logic                              axis_s_ready_o,
  // 2 (377) or 3 (381) accu outputs reduced scalars
  output logic [P_NUM_WIN*P_RED_SCLR_W-1:0] axis_m_data_array_o[P_NUM_ACCU-1:0],
  output logic [            P_NUM_ACCU-1:0] axis_m_valid_o,
  output logic [            P_NUM_ACCU-1:0] axis_m_last_o,
  input        [            P_NUM_ACCU-1:0] axis_m_ready_i
);

  localparam unsigned LP_ZERO_PAD_W = P_TOTAL_WIN * P_RED_SCLR_W - P_FUL_SCLR_W;

  logic [P_TOTAL_WIN*P_RED_SCLR_W-1:0] in_data_zero_paded;
  logic [P_RED_SCLR_W-1:0] split_init_lane[P_TOTAL_WIN-1:0];
  logic [P_RED_SCLR_W-1:0] split_in_data[P_TOTAL_WIN-1:0];
  logic [P_TOTAL_WIN-1:0] split_in_carry;
  logic [P_RED_SCLR_W:0] split_signed_lane[P_TOTAL_WIN-1:0];
  logic [P_RED_SCLR_W-1:0] split_signed_lane_reg[P_TOTAL_WIN-1:0];

  logic [P_TOTAL_WIN*P_RED_SCLR_W-1:0] fifo_in_data;
  logic fifo_in_valid, fifo_in_last;

  logic [P_TOTAL_WIN*P_RED_SCLR_W-1:0] fifo_out_data;
  logic fifo_out_valid, fifo_out_last;
  logic fifo_out_ready;

  logic [P_NUM_ACCU-1:0] forked_valid_array;
  logic [P_NUM_ACCU-1:0] forked_ready_array;

  assign axis_s_ready_o = fifo_out_ready | ~fifo_out_valid;

  // zero-pad the input axis_s_data_i
  assign in_data_zero_paded = {{LP_ZERO_PAD_W{1'b0}}, axis_s_data_i};

  // The split lane pipeline applying the signed/carry bit over reduced scalars
  // Registering input path to the signed split_signed_lane
  for (genvar i = 0; i < P_TOTAL_WIN; i++) begin : split_lane

    // Initial split from P_FUL_SCLR_W bits to P_RED_SCLR_W bits 
    assign split_init_lane[i] = in_data_zero_paded[(i+1)*P_RED_SCLR_W-1 : i*P_RED_SCLR_W];

    // Delay input data path before adding sign for the lane index
    shiftreg #(
      .SHIFT (i),
      .DATA_W(P_RED_SCLR_W)
    ) i_shreg_in_data (
      .clk,
      .data_i(split_init_lane[i]),
      .data_o(split_in_data[i])
    );

    // Assign a full split_signed_lane (P_RED_SCLR_W bits + 1 overflowed carry bit)
    assign split_signed_lane[i] = (i == 0) ? {1'b0, split_in_data[0]} :
                                             split_in_data[i] + split_in_carry[i-1];
    // Carry is MSB split_signed_lane, unless MSB=0, then it's MSB-1   
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        split_in_carry[i] <= 1'b0;
      end else if (!split_signed_lane[i][P_RED_SCLR_W]) begin
        split_in_carry[i] = split_signed_lane[i][P_RED_SCLR_W-1];
      end else begin
        split_in_carry[i] = split_signed_lane[i][P_RED_SCLR_W];
      end
    end

    // Delay split_signed_lane based on its index
    shiftreg #(
      .SHIFT (i),
      .DATA_W(P_RED_SCLR_W)
    ) i_shreg_output (
      .clk,
      .data_i(split_signed_lane[P_TOTAL_WIN-1-i][P_RED_SCLR_W-1:0]),
      .data_o(split_signed_lane_reg[P_TOTAL_WIN-1-i])
    );

    // Combining split lanes into one 256-b register to be fed to the backpressure fifo
    assign fifo_in_data[(i+1)*P_RED_SCLR_W-1 : i*P_RED_SCLR_W] = (i<=P_TOTAL_WIN - 1) ? split_signed_lane_reg[i] : 0;

  end

  // Delay valid and last
  shiftreg #(
    .SHIFT (P_TOTAL_WIN - 1),
    .DATA_W(2)
  ) i_shreg_vld_dly (
    .clk,
    .data_i({axis_s_valid_i & axis_s_ready_o, axis_s_last_i & axis_s_ready_o}),
    .data_o({fifo_in_valid, fifo_in_last})
  );

  // Backpressure FIFO
  fifo_axis #(
    .FIFO_DEPTH    (P_TOTAL_WIN + 8),
    .FIFO_W        (P_TOTAL_WIN * P_RED_SCLR_W),
    .MEM_MACRO_TYPE("distributed"),
    .HAS_LAST      ("true")
  ) i_fifo_axis_sclr_splitter (
    .clk,
    .rst_n,

    .axis_data_i (fifo_in_data),
    .axis_valid_i(fifo_in_valid),
    .axis_last_i (fifo_in_last),
    .axis_ready_o(),

    .axis_data_o (fifo_out_data),
    .axis_valid_o(fifo_out_valid),
    .axis_last_o (fifo_out_last),
    .axis_ready_i(fifo_out_ready)
  );

  stream_fork #(
    .N_OUP(P_NUM_ACCU)
  ) i_stream_fork (
    .clk_i (clk),
    .rst_ni(rst_n),

    .valid_i(fifo_out_valid),
    .ready_o(fifo_out_ready),

    .valid_o(forked_valid_array),
    .ready_i(forked_ready_array)
  );

  // Output assignment
  for (genvar i = 0; i < P_NUM_ACCU; i++) begin : axis_m_lanes
    if (i == P_NUM_ACCU - 1 && P_NUM_ACCU > 2) begin
      assign axis_m_data_array_o[i][6*P_RED_SCLR_W-1:0] = fifo_out_data[P_TOTAL_WIN*P_RED_SCLR_W-1:i*P_NUM_WIN*P_RED_SCLR_W];
      assign axis_m_data_array_o[i][P_NUM_WIN*P_RED_SCLR_W-1 : 6*P_RED_SCLR_W] = '0;
    end else begin
      assign axis_m_data_array_o[i] = fifo_out_data[(i+1)*P_NUM_WIN*P_RED_SCLR_W-1:i*P_NUM_WIN*P_RED_SCLR_W];
    end
    assign axis_m_valid_o[i] = forked_valid_array[i];
    assign forked_ready_array[i] = axis_m_ready_i[i];
    assign axis_m_last_o[i] = fifo_out_last;
  end

endmodule

`endif  // SCALAR_SPLITTER_SV
