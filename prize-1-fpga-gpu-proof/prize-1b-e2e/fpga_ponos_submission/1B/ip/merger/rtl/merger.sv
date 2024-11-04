//////////////////////////////////////////////////////////////////////////////////
// (C) Copyright Ponos Technology 2024
// All Rights Reserved
// *** Ponos Technology Confidential ***
//
//////////////////////////////////////////////////////////////////////////////////

`ifndef MERGER_SV
`define MERGER_SV

module merger #(
  parameter int P_DATA_PNT_W    = 377,  // Width of a data point
  parameter int P_NUM_WIN       = 7,    // Number of Scalars per Data Point, 7=12377, 10=12381
  parameter int P_NUM_AFF_COORD = 3,    // Number of Coordinates
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // Port from point reader
  input        [P_NUM_WIN*P_RED_SCLR_W-1:0] red_sclr_data_i,
  input                                     red_sclr_last_i,
  input                                     red_sclr_valid_i,
  output logic                              red_sclr_ready_o,

  // Port from point reader
  input        [P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] ddr_data_point_i,
  input                                           ddr_data_valid_i,
  output logic                                    ddr_data_ready_o,

  // Port to scheduler
  /* verilog_format: off */
  output logic [P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] merged_point_o,
  output logic [ P_NUM_WIN-1:0][P_RED_SCLR_W-1:0] merged_red_sclr_2d_o,
  output logic                    [P_NUM_WIN-1:0] merged_select_o,
  output logic                                    merged_last_o,
  output logic                                    merged_valid_o,
  input                                           merged_ready_i,
  /* verilog_format: on */

  // Port from scheduler
  input                        accept_fifo_empty_i,
  input        [P_NUM_WIN-1:0] accept_fifo_flags_i,
  output logic                 accept_fifo_rd_o
);

  /* verilog_format: off */
  parameter int P_CASCADE_HEIGHT = 1;    // how many URAM blocks to cascade
  parameter int P_MEM_OUT_STAGES = 2;
  parameter int P_MEM_DEPTH      = 512;
  parameter int P_MEM_MACRO_TYPE = "ultra"; // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM

  // Memory related parameters
  localparam int LP_MEM_ADDR_W       = $clog2(P_MEM_DEPTH);
  localparam int LP_MEM_DATA_W       = (P_NUM_AFF_COORD * P_DATA_PNT_W) + (P_NUM_WIN * P_RED_SCLR_W);
  localparam int LP_BITS_PER_MEM     = 72;
  localparam int LP_NUM_MEM_BLOCKS   = (LP_MEM_DATA_W + (LP_BITS_PER_MEM-1)) / LP_BITS_PER_MEM;
  localparam int LP_MEM_FULL_W       = LP_NUM_MEM_BLOCKS*LP_BITS_PER_MEM;
  localparam int LP_NUM_PADDING_BITS = (LP_MEM_FULL_W-LP_MEM_DATA_W);

  typedef enum logic [2:0] {
    ST_PH1_WR_ONLY = 3'd0,  // fill memory
    ST_PH2_RD_WR   = 3'd1,  // send data to scheduler and refill as slots become available
    ST_PH3_RD_ONLY = 3'd2, // no more new data available, just send remaining valid data from memory
    ST_PH4_CLEAR   = 3'd3,  // last data send, clear all counters
    ST_PH5_DONE    = 3'd4  // 
  } fsm_state;
  fsm_state state;
  /* verilog_format: on */

  logic [               LP_MEM_ADDR_W-1:0] wr_addr;  // shared for select-ram and data-ram
  logic [               LP_MEM_ADDR_W-1:0] rd_addr;  // shared for select-ram and data-ram
  logic [               LP_MEM_FULL_W-1:0] sdp_ram_data_in;
  logic [               LP_MEM_FULL_W-1:0] sdp_ram_data_out;

  logic [                   P_NUM_WIN-1:0] sclr_select;
  logic [                   P_NUM_WIN-1:0] sclr_select_ram;
  logic [                   P_NUM_WIN-1:0] sclr_select_dly;

  logic [P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] uram_point;
  logic [      P_NUM_WIN*P_RED_SCLR_W-1:0] uram_red_scr;
  logic [                   P_NUM_WIN-1:0] reject_flags;
  logic                                    all_sclr_accepted;
  logic [      P_NUM_WIN*P_RED_SCLR_W-1:0] merged_red_sclr_1d;
  logic [           $clog2(P_MEM_DEPTH):0] counter_cleared_selects;  // extra bit

  logic                                    wr_en_sdpram;
  logic                                    wr_en_bram;
  logic                                    rd_mem_vld;
  logic                                    rd_mem_en;
  logic                                    rd_mem_en_dly;
  logic                                    receive_ready;
  logic                                    merged_last_q;
  logic                                    transmit_ready;
  logic                                    accept_fifo_rd;

  assign red_sclr_ready_o = receive_ready;
  assign ddr_data_ready_o = receive_ready;
  assign accept_fifo_rd_o = accept_fifo_rd;

  assign reject_flags = ~accept_fifo_flags_i;
  assign all_sclr_accepted = (reject_flags == 0);  // all select bits low

  assign sclr_select   = (state == ST_PH1_WR_ONLY || 
                         (state == ST_PH2_RD_WR && all_sclr_accepted)) ? '1 : reject_flags; // ST_PH3_RD_ONLY
  assign receive_ready = (state == ST_PH1_WR_ONLY) ? red_sclr_valid_i & ddr_data_valid_i :
                         (state == ST_PH2_RD_WR)   ? red_sclr_valid_i & ddr_data_valid_i & accept_fifo_rd & all_sclr_accepted : 1'b0;
  assign wr_en_sdpram  = (state == ST_PH1_WR_ONLY && red_sclr_valid_i && ddr_data_valid_i) ||
                         (state == ST_PH2_RD_WR   && accept_fifo_rd && all_sclr_accepted); // PH2: replace used location, PH3: clear used locations
  assign wr_en_bram    = (state == ST_PH1_WR_ONLY && red_sclr_valid_i && ddr_data_valid_i) ||
                         (state == ST_PH2_RD_WR   && red_sclr_valid_i && ddr_data_valid_i && !accept_fifo_empty_i) || // PH2: replace used location, PH3: clear used locations
                         (state == ST_PH3_RD_ONLY && accept_fifo_rd); // PH3:
  assign sdp_ram_data_in = {{LP_NUM_PADDING_BITS{1'b0}}, ddr_data_point_i, red_sclr_data_i};

  // ********** State machine **********
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state <= ST_PH1_WR_ONLY;
    end else if (state == ST_PH1_WR_ONLY && wr_addr == P_MEM_DEPTH-1 && 
                 red_sclr_valid_i && ddr_data_valid_i) begin // make sure last mem position is actually filled
      state <= ST_PH2_RD_WR;
    end else if (state == ST_PH2_RD_WR && red_sclr_last_i && red_sclr_valid_i && red_sclr_ready_o) begin
      state <= ST_PH3_RD_ONLY;
    end else if (state == ST_PH3_RD_ONLY && merged_last_q) begin
      state <= ST_PH4_CLEAR;
    end else if (state == ST_PH4_CLEAR && merged_valid_o && merged_last_o) begin
      state <= ST_PH5_DONE;
    end else if (state == ST_PH5_DONE && accept_fifo_empty_i) begin
      state <= ST_PH1_WR_ONLY;
    end
  end

  // ********** Receive Side **********
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_addr <= '0;
    end else if (state == ST_PH5_DONE) begin  // ready not needed here
      wr_addr <= '0;
    end else if (state == ST_PH1_WR_ONLY && red_sclr_valid_i && ddr_data_valid_i) begin // ready not needed here
      wr_addr <= wr_addr + 1;
    end else if ((state == ST_PH2_RD_WR || state == ST_PH3_RD_ONLY) && accept_fifo_rd) begin
      wr_addr <= wr_addr + 1;
    end else begin
      wr_addr <= wr_addr;
    end
  end

  // ********** Transceive Side **********
  assign transmit_ready = (merged_ready_i | ~merged_valid_o);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_mem_en <= 1'b0;
      rd_addr   <= '0;
    end else if (state == ST_PH5_DONE) begin
      rd_mem_en <= 1'b0;
      rd_addr   <= '0;
    end else if (state == ST_PH2_RD_WR && transmit_ready && red_sclr_valid_i && ddr_data_valid_i) begin
      rd_mem_en <= 1'b1;
      rd_addr   <= rd_addr + 1;
    end else if (state == ST_PH3_RD_ONLY && transmit_ready) begin
      rd_mem_en <= 1'b1;
      rd_addr   <= rd_addr + 1;
    end else begin
      rd_mem_en <= 1'b0;
      rd_addr   <= rd_addr;
    end
  end

  // ********** Counter for Done detection **********
  always_ff @(posedge clk) begin
    rd_mem_vld <= rd_mem_en;  // in sync with output data
    if (!rst_n) begin
      counter_cleared_selects <= P_MEM_DEPTH;
      merged_last_q <= 1'b0;
    end else if (state == ST_PH5_DONE) begin  // found valid data, start counter again
      counter_cleared_selects <= P_MEM_DEPTH;
      merged_last_q <= 1'b0;
    end else if (counter_cleared_selects == 0 && rd_mem_vld) begin // no more valid data available, stop everything
      counter_cleared_selects <= P_MEM_DEPTH;
      merged_last_q <= 1'b1;
    end else if (sclr_select_ram != 0 && rd_mem_vld) begin  // found valid data, start counter again
      counter_cleared_selects <= P_MEM_DEPTH;
      merged_last_q <= 1'b0;
    end else if (sclr_select_ram == 0 && rd_mem_vld) begin
      counter_cleared_selects <= counter_cleared_selects - 1;
      merged_last_q <= 1'b0;
    end else begin
    end
  end

  // ********** Accept Response Fifo **********
  // for case that scalar and data point valid are not ready when accept_valid arrives
  assign accept_fifo_rd = ~accept_fifo_empty_i & ((state == ST_PH2_RD_WR && red_sclr_valid_i && ddr_data_valid_i) ||
                                                   state == ST_PH3_RD_ONLY || // read as fast as possible
                                                   state == ST_PH5_DONE);     // clear accept response fifo of anything left in it

  // ********** Memory Instance - select bits **********
  bram #(
    .BRAM_DEPTH    (P_MEM_DEPTH),
    .BRAM_W        (P_NUM_WIN),
    .MEM_MACRO_TYPE("block")       // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM
  ) i_select_ram (
    .clk,
    .wen_i  (wr_en_bram),
    .waddr_i(wr_addr),
    .data_i (sclr_select),
    .raddr_i(rd_addr),
    .data_o (sclr_select_ram)
  );

  // synthesis translate_off
  // for debug purposes only, floating output
  logic [P_NUM_WIN * P_RED_SCLR_W-1:0] sim_red_sclr_bram_out;
  bram #(
    .BRAM_DEPTH(P_MEM_DEPTH),
    .BRAM_W(P_NUM_WIN * P_RED_SCLR_W),
    .MEM_MACRO_TYPE("block")  // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM
  ) i_sim_only_red_sclr_bram (
    .clk,
    .wen_i  (wr_en_bram),
    .waddr_i(wr_addr),
    .data_i (red_sclr_data_i),
    .raddr_i(rd_addr),
    .data_o (sim_red_sclr_bram_out)
  );
  // synthesis translate_on

  // ********** Memory Instance - scalars and data points **********
  for (genvar i = 0; i < LP_NUM_MEM_BLOCKS; i = i + 1) begin : sdpram_block
    xpm_memory_sdpram #(
      .ADDR_WIDTH_A(LP_MEM_ADDR_W),  // DECIMAL
      .ADDR_WIDTH_B(LP_MEM_ADDR_W),  // DECIMAL
      .BYTE_WRITE_WIDTH_A(9),  // DECIMAL - DATA WIDTH IN BYTES
      .CASCADE_HEIGHT(P_CASCADE_HEIGHT),  // DECIMAL
      .MEMORY_PRIMITIVE(P_MEM_MACRO_TYPE),  // String
      .MEMORY_SIZE(P_MEM_DEPTH * LP_BITS_PER_MEM),  // DECIMAL
      .READ_DATA_WIDTH_B(LP_BITS_PER_MEM),  // DECIMAL
      .READ_LATENCY_B(P_MEM_OUT_STAGES),  // DECIMAL - INDICATE the depth of the output stage
      .WRITE_DATA_WIDTH_A(LP_BITS_PER_MEM),  // DECIMAL
      .WRITE_MODE_B("read_first")  // String
    ) xpm_memory_sdpram (

      .clka  (clk),     // 1-bit input: Clock signal for port A. Also clocks port B when
                        // parameter CLOCKING_MODE is "common_clock".
      .clkb  (clk),     // 1-bit input: Clock signal for port B when parameter CLOCKING_MODE is
                        // "independent_clock". Unused when parameter CLOCKING_MODE is
                        // "common_clock".
      .rstb  (~rst_n),  // 1-bit input: Reset signal for the final port B output register stage.
                        // Synchronously resets output port doutb to the value specified by
                        // parameter READ_RESET_VALUE_B.
      .regceb(1'b1),    // 1-bit input: Clock Enable for the last register stage on the output
                        // data path.

      .wea({8{wr_en_sdpram}}),  // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
      // for port A input data port dina. 1 bit wide when word-wide writes are
      // used. In byte-wide write configurations, each bit controls the
      // writing one byte of dina to address addra. For example, to
      // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
      // is 32, wea would be 4'b0010.
      .addra(wr_addr),  // ADDR_WIDTH_A-bit input: Address for port A write operations.
      .dina(sdp_ram_data_in[LP_BITS_PER_MEM*i+:LP_BITS_PER_MEM]),  // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
      .ena(1'b1),  // 1-bit input: Memory enable signal for port A. Must be high on clock
                   // cycles when write operations are initiated. Pipelined internally.
      .addrb(rd_addr),  // ADDR_WIDTH_B-bit input: Address for port B read operations.
      .doutb(sdp_ram_data_out[LP_BITS_PER_MEM*i+:LP_BITS_PER_MEM]),  // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
      .enb(1'b1)  // 1-bit input: Memory enable signal for port B. Must be high on clock
                  // cycles when read operations are initiated. Pipelined internally.
    );
  end

  // as the data_point  xpm_memory_sdpram  is slower than the scalar block ram
  shiftreg #(
    .SHIFT (P_MEM_OUT_STAGES - 1),
    .DATA_W(P_NUM_WIN + 1)
  ) i_shiftreg_rd_mem_en (
    .clk,
    .data_i({sclr_select_ram, rd_mem_en}),
    .data_o({sclr_select_dly, rd_mem_en_dly})
  );

  assign uram_point   = sdp_ram_data_out[LP_MEM_DATA_W-1:(P_NUM_WIN*P_RED_SCLR_W)];
  assign uram_red_scr = sdp_ram_data_out[(P_NUM_WIN*P_RED_SCLR_W)-1:0];

  fifo_axis #(
    .FIFO_DEPTH(2 * P_MEM_OUT_STAGES),
    .FIFO_W(1 + P_NUM_WIN + LP_MEM_DATA_W),
    .MEM_MACRO_TYPE("block")  // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM
  ) i_fifo_out_backpressure (
    .clk,
    .rst_n,
    //
    .axis_data_i({merged_last_q, uram_point, uram_red_scr, sclr_select_dly}),
    .axis_valid_i(rd_mem_en_dly & (state != ST_PH4_CLEAR && state != ST_PH5_DONE)), // stop transmit
    .axis_ready_o(),
    //
    .axis_data_o({merged_last_o, merged_point_o, merged_red_sclr_1d, merged_select_o}),
    .axis_valid_o(merged_valid_o),
    .axis_ready_i(merged_ready_i)
  );

  // ********** convert 2d input ports to 1d vector for memory **********
  integer j;
  always_comb begin
    for (j = 0; j < P_NUM_WIN; j++) begin
      merged_red_sclr_2d_o[j] <= merged_red_sclr_1d[P_RED_SCLR_W*j+:P_RED_SCLR_W];
    end
  end

endmodule

`endif
