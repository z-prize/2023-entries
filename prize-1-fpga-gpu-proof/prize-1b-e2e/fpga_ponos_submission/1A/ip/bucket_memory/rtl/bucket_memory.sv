

`ifndef BUCKET_MEMORY_SV
`define BUCKET_MEMORY_SV

module bucket_memory #(
  parameter unsigned P_DATA_PNT_W = 377,  // data point width, 377: BLS12-377, 381: BLS12-381
  parameter unsigned P_NUM_WIN = 7,  // Number of bucket sets: 7 or 10
  parameter unsigned P_NUM_AFF_COORD = 3,
  parameter unsigned P_CASCADE_HEIGHT = 1,  // how many URAM blocks to cascade
  parameter unsigned P_OUTPUT_STAGE_NUM = 9,
  parameter          P_MEM_MACRO_TYPE    = "ultra", // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // From acc. controller block
  /* verilog_format: off */
  input  [                 P_RED_SCLR_W-2:0] acc_ctrl_bucket_addr_i,
  input  [            $clog2(P_NUM_WIN)-1:0] acc_ctrl_bucket_set_addr_i,  // Bucket set address (0..6) or (0..9)
  input                                      acc_ctrl_clear_bucket_i,  // initialize the bucket memory
  // From curve adder side
  input  [P_DATA_PNT_W*P_NUM_PROJ_COORD-1:0] ca_sum_i,
  input  [                 P_RED_SCLR_W-2:0] ca_bucket_addr_i,
  input  [            $clog2(P_NUM_WIN)-1:0] ca_bucket_set_addr_i,
  input                                      ca_valid_i,

  output [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] bucket_data_o,
  /* verilog_format: on */

  // To Scheduler Signals - backward path - FIXME no delay, just assign
  // is the delay need for the scheduler to be synced with the output data
  // Part of bucket-(set, addr) pair used by the scheduler to clear the flag such that the corresponding bucket can be used again for computation
  output logic [     P_RED_SCLR_W-2:0] sch_bucket_addr_o,
  output logic [$clog2(P_NUM_WIN)-1:0] sch_bucket_set_addr_o,
  output logic                         sch_valid_o
);

  localparam LP_SDPRAM_W = 72;
  localparam LP_MEM_BASE_DEPTH = 4096;

  localparam LP_MEM_DATA_W = (P_DATA_PNT_W == 377) ? 378 * 4 : 396 * 4;
  localparam [P_DATA_PNT_W-1:0] LP_INIT_MEM_0 = (P_DATA_PNT_W == 377) ? 377'd0 : 381'd0;
  localparam [P_DATA_PNT_W-1:0] LP_INIT_MEM_1 = ((P_MUL_TYPE == "montgomery") && (P_DATA_PNT_W == 377)) ? 377'h8d6661e2fdf49a4cf495bf803c84e87b4e97b76e7c63059f7db3a98a7d3ff251409f837fffffb102cdffffffff :
                                                ((P_MUL_TYPE == "montgomery") && (P_DATA_PNT_W == 377)) ? 377'd1 :
                                                ((P_MUL_TYPE == "montgomery") && (P_DATA_PNT_W == 381)) ? 381'h8d6661e2fdf49a4cf495bf803c84e87b4e97b76e7c63059f7db3a98a7d3ff251409f837fffffb102cdffffffff :
                                                                                                 381'd1;

  localparam LP_ZERO_PRE = (P_DATA_PNT_W == 377) ? 4'd0 : 60'd0;
  localparam [P_DATA_PNT_W*P_NUM_PROJ_COORD-1:0] BUCKET_MEM_INIT_ARRAY = (P_DATA_PNT_W == 377) ? {LP_INIT_MEM_1, LP_INIT_MEM_0, LP_INIT_MEM_1, LP_INIT_MEM_0} : 
                                                                                                 {LP_INIT_MEM_1, LP_INIT_MEM_1, LP_INIT_MEM_1, LP_INIT_MEM_0};
  localparam LP_MEM_NUM_BLOCKS = (P_DATA_PNT_W == 377) ? 21 : 22;

  localparam LP_XPM_ADDR_W = $clog2(P_NUM_WIN) + P_RED_SCLR_W - 1;

  logic [LP_MEM_DATA_W-1:0] mem_data_in;
  logic [LP_MEM_DATA_W-1:0] mem_data_out;

  logic [LP_XPM_ADDR_W-1:0] wr_addr;
  logic [LP_XPM_ADDR_W-1:0] rd_addr;

  assign mem_data_in = acc_ctrl_clear_bucket_i ? {LP_ZERO_PRE, BUCKET_MEM_INIT_ARRAY} : {LP_ZERO_PRE, ca_sum_i};

  assign wr_addr = acc_ctrl_clear_bucket_i ? {acc_ctrl_bucket_set_addr_i,acc_ctrl_bucket_addr_i} :
                                             {      ca_bucket_set_addr_i,      ca_bucket_addr_i};
  assign rd_addr = {acc_ctrl_bucket_set_addr_i, acc_ctrl_bucket_addr_i};

  for (genvar i = 0; i < LP_MEM_NUM_BLOCKS; i = i + 1) begin : GEN_BUILDING_UNIT
    xpm_memory_sdpram #(
      .ADDR_WIDTH_A(LP_XPM_ADDR_W),  // DECIMAL
      .ADDR_WIDTH_B(LP_XPM_ADDR_W),  // DECIMAL
      .BYTE_WRITE_WIDTH_A(9),  // DECIMAL - DATA WIDTH IN BYTES
      .CASCADE_HEIGHT(P_CASCADE_HEIGHT),  // DECIMAL
      .MEMORY_PRIMITIVE(P_MEM_MACRO_TYPE),  // String
      .MEMORY_SIZE(LP_MEM_BASE_DEPTH * P_NUM_WIN * LP_SDPRAM_W),  // DECIMAL
      .READ_DATA_WIDTH_B(LP_SDPRAM_W),  // DECIMAL
      .READ_LATENCY_B(P_OUTPUT_STAGE_NUM),  // DECIMAL - INDICATE the depth of the output stage
      .WRITE_DATA_WIDTH_A(LP_SDPRAM_W),  // DECIMAL
      .WRITE_MODE_B("read_first")  // String
    ) i_xpm_memory_sdpram_bkt (

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

      .wea({8{ca_valid_i | acc_ctrl_clear_bucket_i}}), // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
      // for port A input data port dina. 1 bit wide when word-wide writes are
      // used. In byte-wide write configurations, each bit controls the
      // writing one byte of dina to address addra. For example, to
      // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
      // is 32, wea would be 4'b0010.
      .addra(wr_addr),  // ADDR_WIDTH_A-bit input: Address for port A write operations.
      .dina(mem_data_in[LP_SDPRAM_W*i+:LP_SDPRAM_W]),  // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
      .ena(1'b1),  // 1-bit input: Memory enable signal for port A. Must be high on clock 
                   // cycles when write operations are initiated. Pipelined internally.
      .addrb(rd_addr),  // ADDR_WIDTH_B-bit input: Address for port B read operations.
      .doutb(mem_data_out[LP_SDPRAM_W*i+:LP_SDPRAM_W]),  // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
      .enb(1'b1)  // 1-bit input: Memory enable signal for port B. Must be high on clock
                  // cycles when read operations are initiated. Pipelined internally.
    );
  end

  // delay to make sure data is written before the data is passed to the scheduler
  always_ff @(posedge clk) begin
    sch_bucket_addr_o     = ca_bucket_addr_i;
    sch_bucket_set_addr_o = ca_bucket_set_addr_i;
    sch_valid_o           = ca_valid_i;
  end

  assign bucket_data_o = mem_data_out[P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0];

endmodule

`endif  // BUCKET_MEMORY_SV
