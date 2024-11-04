

`ifndef ACCUMULATOR_SV
`define ACCUMULATOR_SV

(* DONT_TOUCH = "yes" *)
module accumulator #(
  parameter unsigned P_DATA_PNT_W          = 377,
  parameter unsigned P_NUM_WIN             = 7,
  parameter unsigned P_NUM_AFF_COORD       = 3,
  parameter unsigned P_BM_OUTPUT_STAGE_NUM = 9,    // Number of bucket mem out stages
  parameter unsigned P_CASCADE_HEIGHT      = 1,
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // Control interface: ap_start/ap_done from XRT and mode from axi4-lite control reg
  input        ctrl_ap_start_i,
  input  [2:0] ctrl_mode_i,
  output       ctrl_ap_done_o,

  // Read stream - reduced scalars
  input  [P_NUM_WIN*P_RED_SCLR_W-1:0] axis_s_red_scal_data_i,
  input                               axis_s_red_scal_valid_i,
  input                               axis_s_red_scal_last_i,
  output                              axis_s_red_scal_ready_o,

  // Read stream - data points
  input  [P_DATA_PNT_W-1:0] axis_s_pnt_data_i,
  input                     axis_s_pnt_valid_i,
  output                    axis_s_pnt_ready_o,

  // Write stream - buckets
  output [P_DATA_PNT_W-1:0] axis_m_bucket_data_o,
  output                    axis_m_bucket_valid_o,
  output                    axis_m_bucket_last_o,
  input                     axis_m_bucket_ready_i
);

  logic [          P_NUM_AFF_COORD*384-1:0] data_pnt_conv_padded_data;
  logic [ P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] data_pnt_conv_data;
  logic                                     data_pnt_conv_valid;
  logic                                     data_pnt_conv_ready;

  logic [ P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] data_pnt_fifo_array;
  logic                                     data_pnt_fifo_valid;
  logic                                     data_pnt_fifo_ready;

  logic [       P_NUM_WIN*P_RED_SCLR_W-1:0] red_scal_fifo_data;
  logic                                     red_scal_fifo_valid;
  logic                                     red_scal_fifo_last;
  logic                                     red_scal_fifo_ready;

  logic [ P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] merger_data_pnt;
  logic [  P_NUM_WIN-1:0][P_RED_SCLR_W-1:0] merger_red_scal_array;
  logic [                    P_NUM_WIN-1:0] merger_select;
  logic                                     merger_last;
  logic                                     merger_valid;
  logic                                     merger_ready;

  logic                                     sched_rspn_fifo_rd;
  logic [                    P_NUM_WIN-1:0] sched_rspn_fifo_flags;
  logic                                     sched_rspn_fifo_empty;

  logic                                     sched_rspn_valid;
  logic [                    P_NUM_WIN-1:0] sched_rspn_flags;

  logic [ P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] serial_data_pnt;
  logic [                 P_RED_SCLR_W-1:0] serial_red_sclr;
  logic [            $clog2(P_NUM_WIN)-1:0] serial_window;
  logic                                     serial_valid;
  logic                                     serial_last;
  logic                                     serial_ready;

  logic [ P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] sched_point;
  logic [                 P_RED_SCLR_W-2:0] sched_bucket_addr;
  logic [          $clog2(P_NUM_WIN) - 1:0] sched_bucket_set_addr;
  logic                                     sched_add_sub;
  logic                                     sched_valid;

  logic [ P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] acc_point;
  logic [                 P_RED_SCLR_W-2:0] acc_bucket_addr;
  logic [          $clog2(P_NUM_WIN) - 1:0] acc_bucket_set_addr;
  logic                                     acc_add_sub;
  logic                                     acc_valid;
  logic                                     acc_ctrl_init_done;

  logic [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] bm_data;
  logic [                 P_RED_SCLR_W-2:0] bm_bucket_addr;
  logic [            $clog2(P_NUM_WIN)-1:0] bm_bucket_set_addr;
  logic                                     bm_valid;

  logic [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] ca_sum;
  logic [                 P_RED_SCLR_W-2:0] ca_bucket_addr;
  logic [            $clog2(P_NUM_WIN)-1:0] ca_bucket_set_addr;
  logic                                     ca_valid;

  logic [                 P_RED_SCLR_W-2:0] acc_ctrl_bm_addr;
  logic [            $clog2(P_NUM_WIN)-1:0] acc_ctrl_bm_set_addr;
  logic                                     acc_ctrl_clear_bucket;

  // ****************************************************************************
  // Data points read path from DDR
  if (P_DATA_PNT_W == 377) begin
    xlnx_axis_conv_384to1152 i_xlnx_axis_conv (
      .aclk   (clk),
      .aresetn(rst_n),

      .s_axis_tdata ({{(384 - P_DATA_PNT_W) {1'b0}}, axis_s_pnt_data_i}),
      .s_axis_tvalid(axis_s_pnt_valid_i),
      .s_axis_tready(axis_s_pnt_ready_o),

      .m_axis_tdata (data_pnt_conv_padded_data),
      .m_axis_tvalid(data_pnt_conv_valid),
      .m_axis_tready(data_pnt_conv_ready)
    );
  end else begin
    xlnx_axis_conv_384to768 i_xlnx_axis_conv (
      .aclk   (clk),
      .aresetn(rst_n),

      .s_axis_tdata ({{(384 - P_DATA_PNT_W) {1'b0}}, axis_s_pnt_data_i}),
      .s_axis_tvalid(axis_s_pnt_valid_i),
      .s_axis_tready(axis_s_pnt_ready_o),

      .m_axis_tdata (data_pnt_conv_padded_data),
      .m_axis_tvalid(data_pnt_conv_valid),
      .m_axis_tready(data_pnt_conv_ready)
    );
  end

  for (genvar i = 0; i < P_NUM_AFF_COORD; i = i + 1) begin : axis_data_conv
    assign data_pnt_conv_data[i*P_DATA_PNT_W+:P_DATA_PNT_W] = data_pnt_conv_padded_data[i*384+:P_DATA_PNT_W];
  end

  fifo_axis #(
    .FIFO_DEPTH    (512),
    .FIFO_W        (3 * P_DATA_PNT_W),
    .MEM_MACRO_TYPE("block")
  ) i_data_pnt_fifo_array (
    .clk,
    .rst_n,

    .axis_data_i (data_pnt_conv_data),
    .axis_valid_i(data_pnt_conv_valid),
    .axis_ready_o(data_pnt_conv_ready),

    .axis_data_o (data_pnt_fifo_array),
    .axis_valid_o(data_pnt_fifo_valid),
    .axis_ready_i(data_pnt_fifo_ready)
  );

  // ****************************************************************************
  // Reduced scalars read path from scalar splitter
  fifo_axis #(
    .FIFO_DEPTH    (512),
    .FIFO_W        (P_NUM_WIN * P_RED_SCLR_W),
    .MEM_MACRO_TYPE("block"),
    .HAS_LAST      ("true")
  ) i_red_scal_fifo (
    .clk,
    .rst_n,

    .axis_data_i (axis_s_red_scal_data_i),
    .axis_valid_i(axis_s_red_scal_valid_i),
    .axis_last_i (axis_s_red_scal_last_i),
    .axis_ready_o(axis_s_red_scal_ready_o),

    .axis_data_o (red_scal_fifo_data),
    .axis_valid_o(red_scal_fifo_valid),
    .axis_last_o (red_scal_fifo_last),
    .axis_ready_i(red_scal_fifo_ready)
  );

  // ****************************************************************************
  merger #(
    .P_NUM_WIN      (P_NUM_WIN),        // Number of Scalars per Data Point, 7=12377, 10=12381
    .P_NUM_AFF_COORD(P_NUM_AFF_COORD),  // Number of Coordinates / Dimensions
    .P_DATA_PNT_W   (P_DATA_PNT_W)      // Width of data point
  ) i_merger (
    .clk,
    .rst_n,
    // port from point reader
    .red_sclr_data_i     (red_scal_fifo_data),
    .red_sclr_last_i     (red_scal_fifo_last),
    .red_sclr_valid_i    (red_scal_fifo_valid),
    .red_sclr_ready_o    (red_scal_fifo_ready),
    // port from point reader
    .ddr_data_point_i    (data_pnt_fifo_array),
    .ddr_data_valid_i    (data_pnt_fifo_valid),
    .ddr_data_ready_o    (data_pnt_fifo_ready),
    // port to serialiser
    .merged_point_o      (merger_data_pnt),
    .merged_red_sclr_2d_o(merger_red_scal_array),
    .merged_select_o     (merger_select),
    .merged_last_o       (merger_last),
    .merged_valid_o      (merger_valid),
    .merged_ready_i      (merger_ready),
    // port from scheduler
    .accept_fifo_empty_i (sched_rspn_fifo_empty),
    .accept_fifo_flags_i (sched_rspn_fifo_flags),
    .accept_fifo_rd_o    (sched_rspn_fifo_rd)
  );

  fifo_sync #(
    .FIFO_DEPTH      (16),
    .FIFO_W          (P_NUM_WIN),
    .MEM_MACRO_TYPE  ("distributed"),
    .ALMOST_EMPTY_THR(0),
    .ALMOST_FULL_THR (0)
  ) i_fifo_sync_rspn_flags (
    .clk,
    .rst_n,

    .wr_req_i (sched_rspn_valid),
    .wr_data_i(sched_rspn_flags),
    .full_o   (),

    .rd_req_i (sched_rspn_fifo_rd),
    .rd_data_o(sched_rspn_fifo_flags),
    .empty_o  (sched_rspn_fifo_empty)
  );

  // ****************************************************************************
  smart_red_scal_serializer #(
    .P_DATA_PNT_W   (P_DATA_PNT_W),
    .P_RED_SCAL_W   (P_RED_SCLR_W),
    .P_NUM_AFF_COORD(P_NUM_AFF_COORD),
    .P_NUM_WIN      (P_NUM_WIN)
  ) i_smart_ser (
    .clk,
    .rst_n,

    .data_pnt_i      (merger_data_pnt),
    .red_scal_array_i(merger_red_scal_array),
    .flags_i         (merger_select),
    .valid_i         (merger_valid),
    .ready_o         (merger_ready),

    .data_pnt_o(serial_data_pnt),
    .red_scal_o(serial_red_sclr),
    .window_o  (serial_window),
    .valid_o   (serial_valid),
    .last_o    (serial_last),
    .ready_i   (serial_ready)
  );

  // ****************************************************************************
  scheduler #(
    .P_DATA_PNT_W   (P_DATA_PNT_W),
    .P_NUM_AFF_COORD(P_NUM_AFF_COORD),
    .P_RED_SCLR_W   (P_RED_SCLR_W),
    .P_NUM_WIN      (P_NUM_WIN)
  ) i_scheduler (
    .clk,
    .rst_n,

    // AXIS input stream of (point, red. scalar, index i.e. which bucket set)
    .prsi_point_i     (serial_data_pnt),
    .prsi_red_scalar_i(serial_red_sclr),
    .prsi_index_i     (serial_window),
    .prsi_valid_i     (serial_valid),
    .prsi_last_i      (serial_last),
    .prsi_ready_o     (serial_ready),

    // Acceptence interface: notifies merger which reduced scalars have been accepted and sent to curve adder
    .accept_flags_o(sched_rspn_flags),
    .accept_valid_o(sched_rspn_valid),

    // Interface with accumulator controller
    .acc_point_o          (sched_point),
    .acc_bucket_addr_o    (sched_bucket_addr),
    .acc_bucket_set_addr_o(sched_bucket_set_addr),
    .acc_add_sub_o        (sched_add_sub),
    .acc_valid_o          (sched_valid),

    // Interface with feedback from bucket memory: indicates curve addition is completed for specific bucket (to clear flag)
    .bm_bucket_addr_i    (bm_bucket_addr),
    .bm_bucket_set_addr_i(bm_bucket_set_addr),
    .bm_valid_i          (bm_valid),

    // Signal from Acc. controller indicating bucket memory has been initialized with special initial EC points
    .init_done_i(acc_ctrl_init_done)
  );

  // ****************************************************************************
  if (P_DATA_PNT_W == 377) begin : GEN_CURVE_ADDER_377
    curve_adder_377 i_curve_adder_377 (
      .clk,
      .rst_n,

      // port from bucket memory
      .bucket_mem_i(bm_data),

      .data_point_i     (acc_point),
      .add_sub_i        (acc_add_sub),
      .bucket_addr_i    (acc_bucket_addr),
      .bucket_set_addr_i(acc_bucket_set_addr),
      .valid_i          (acc_valid),

      // port to bucket memory
      .sum_o            (ca_sum),
      .bucket_addr_o    (ca_bucket_addr),
      .bucket_set_addr_o(ca_bucket_set_addr),
      .valid_o          (ca_valid)
    );
  end else if (P_DATA_PNT_W == 381) begin : GEN_CURVE_ADDER_381
    curve_adder_381 i_curve_adder_381 (
      .clk,
      .rst_n,

      // port from bucket memory
      .bucket_mem_i(bm_data),

      .data_point_i     (acc_point),
      .add_sub_i        (acc_add_sub),
      .bucket_addr_i    (acc_bucket_addr),
      .bucket_set_addr_i(acc_bucket_set_addr),
      .valid_i          (acc_valid),

      // port to bucket memory
      .sum_o            (ca_sum),
      .bucket_addr_o    (ca_bucket_addr),
      .bucket_set_addr_o(ca_bucket_set_addr),
      .valid_o          (ca_valid)
    );
  end

  // ****************************************************************************
  bucket_memory #(
    .P_DATA_PNT_W(P_DATA_PNT_W),  // data point width, 377: BLS12-377, 381: BLS12-381
    .P_NUM_WIN(P_NUM_WIN),  // Number of bucket sets: 7 or 10
    .P_NUM_AFF_COORD(P_NUM_AFF_COORD),
    .P_CASCADE_HEIGHT(P_CASCADE_HEIGHT),  // how many URAM blocks to cascade - FIXED at 1 "one"
    .P_OUTPUT_STAGE_NUM(P_BM_OUTPUT_STAGE_NUM),
    .P_MEM_MACRO_TYPE("ultra")  // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM
  ) i_bucket_memory (
    .clk,
    .rst_n,

    // from acc. controller block
    .acc_ctrl_bucket_addr_i    (acc_ctrl_bm_addr),
    .acc_ctrl_bucket_set_addr_i(acc_ctrl_bm_set_addr),  // Bucket set address (0..6) or (0..9)
    .acc_ctrl_clear_bucket_i   (acc_ctrl_clear_bucket), // initialize the bucket memory

    .ca_sum_i            (ca_sum),
    .ca_bucket_addr_i    (ca_bucket_addr),
    .ca_bucket_set_addr_i(ca_bucket_set_addr),
    .ca_valid_i          (ca_valid),

    .bucket_data_o(bm_data),

    .sch_bucket_addr_o    (bm_bucket_addr),
    .sch_bucket_set_addr_o(bm_bucket_set_addr),
    .sch_valid_o          (bm_valid)
  );

  // ****************************************************************************
  acc_ctrl #(
    .P_DATA_PNT_W(P_DATA_PNT_W),  // Data point width, 377: BLS12-377, 381: BLS12-381
    .P_NUM_WIN(P_NUM_WIN),  // Number of bucket sets: 7 or 10
    .P_BM_OUTPUT_STAGE_NUM(P_BM_OUTPUT_STAGE_NUM),  // Number of clk cycles delay in BM: addr -> bucket_val
    .P_NUM_AFF_COORD(P_NUM_AFF_COORD)
  ) i_acc_ctrl (
    .clk,
    .rst_n,

    // Control interface: ap_start/ap_done from XRT and mode from axi4-lite control reg
    .ctrl_ap_start_i(ctrl_ap_start_i),
    .ctrl_mode_i    (ctrl_mode_i),
    .ctrl_ap_done_o (ctrl_ap_done_o),

    // Signal from merger indicating all scalar/points have been processed i.e. merger table completely empty
    .mrgr_merged_last_i(merger_last & merger_valid),

    // Interface with the scheduler
    .sch_point_i          (sched_point),
    .sch_bucket_addr_i    (sched_bucket_addr),
    .sch_bucket_set_addr_i(sched_bucket_set_addr),
    .sch_add_sub_i        (sched_add_sub),
    .sch_valid_i          (sched_valid),
    .sch_init_done_o      (acc_ctrl_init_done),

    // Interface with bucket memory
    .bm_bucket_addr_o    (acc_ctrl_bm_addr),
    .bm_bucket_set_addr_o(acc_ctrl_bm_set_addr),
    .bm_clear_bucket_o   (acc_ctrl_clear_bucket),
    .bm_bucket_val_i     (bm_data),

    // Interface towards curve adder
    .ca_point_o          (acc_point),
    .ca_bucket_addr_o    (acc_bucket_addr),
    .ca_bucket_set_addr_o(acc_bucket_set_addr),
    .ca_add_sub_o        (acc_add_sub),
    .ca_valid_o          (acc_valid),

    // AXI-S interface towards bucket writer
    .bw_bucket_val_o(axis_m_bucket_data_o),
    .bw_valid_o     (axis_m_bucket_valid_o),
    .bw_last_o      (axis_m_bucket_last_o),
    .bw_ready_i     (axis_m_bucket_ready_i)
  );


endmodule

`endif  // ACCUMULATOR_SV