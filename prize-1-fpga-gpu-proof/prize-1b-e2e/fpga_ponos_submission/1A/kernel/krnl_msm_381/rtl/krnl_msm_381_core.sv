

`ifndef KRNL_MSM_381_CORE_SV
`define KRNL_MSM_381_CORE_SV

`include "makefile_temp.vh"

module krnl_msm_381_core
  import krnl_msm_381_pkg::*;
#(
  parameter int C_NUM_AXI_M_PORTS = 3,
  `include "krnl_common.vh"
) (
  // System signals
  input                         ap_clk,
  input                         ap_rst_n,
  // Input scals from HOST/Write buckets to HOST 
  output                        axi_m_0_ARVALID,
  input                         axi_m_0_ARREADY,
  output [  C_AXI_M_ADDR_W-1:0] axi_m_0_ARADDR,
  output [                 7:0] axi_m_0_ARLEN,
  output [                 2:0] axi_m_0_ARSIZE,
  input                         axi_m_0_RVALID,
  output                        axi_m_0_RREADY,
  input  [  C_AXI_M_DATA_W-1:0] axi_m_0_RDATA,
  input                         axi_m_0_RLAST,
  input  [                 1:0] axi_m_0_RRESP,
  output                        axi_m_0_AWVALID,
  input                         axi_m_0_AWREADY,
  output [  C_AXI_M_ADDR_W-1:0] axi_m_0_AWADDR,
  output [                 7:0] axi_m_0_AWLEN,
  output [                 2:0] axi_m_0_AWSIZE,
  output                        axi_m_0_WVALID,
  input                         axi_m_0_WREADY,
  output [  C_AXI_M_DATA_W-1:0] axi_m_0_WDATA,
  output [C_AXI_M_DATA_W/8-1:0] axi_m_0_WSTRB,
  output                        axi_m_0_WLAST,
  input                         axi_m_0_BVALID,
  output                        axi_m_0_BREADY,
  input  [                 1:0] axi_m_0_BRESP,

  // Accumulator0 - Read data point
  output                      axi_m_1_ARVALID,
  input                       axi_m_1_ARREADY,
  output [C_AXI_M_ADDR_W-1:0] axi_m_1_ARADDR,
  output [               7:0] axi_m_1_ARLEN,
  output [               2:0] axi_m_1_ARSIZE,
  input                       axi_m_1_RVALID,
  output                      axi_m_1_RREADY,
  input  [C_AXI_M_DATA_W-1:0] axi_m_1_RDATA,
  input                       axi_m_1_RLAST,
  input  [               1:0] axi_m_1_RRESP,

  // Accumulator1 - Read data point
  output                      axi_m_2_ARVALID,
  input                       axi_m_2_ARREADY,
  output [C_AXI_M_ADDR_W-1:0] axi_m_2_ARADDR,
  output [               7:0] axi_m_2_ARLEN,
  output [               2:0] axi_m_2_ARSIZE,
  input                       axi_m_2_RVALID,
  output                      axi_m_2_RREADY,
  input  [C_AXI_M_DATA_W-1:0] axi_m_2_RDATA,
  input                       axi_m_2_RLAST,
  input  [               1:0] axi_m_2_RRESP,

  // AXI4-Lite slave interface
  input                              axi_s_ctrl_AWVALID,
  output                             axi_s_ctrl_AWREADY,
  input  [  C_AXI_S_CTRL_ADDR_W-1:0] axi_s_ctrl_AWADDR,
  input                              axi_s_ctrl_WVALID,
  output                             axi_s_ctrl_WREADY,
  input  [  C_AXI_S_CTRL_DATA_W-1:0] axi_s_ctrl_WDATA,
  input  [C_AXI_S_CTRL_DATA_W/8-1:0] axi_s_ctrl_WSTRB,
  input                              axi_s_ctrl_ARVALID,
  output                             axi_s_ctrl_ARREADY,
  input  [  C_AXI_S_CTRL_ADDR_W-1:0] axi_s_ctrl_ARADDR,
  output                             axi_s_ctrl_RVALID,
  input                              axi_s_ctrl_RREADY,
  output [  C_AXI_S_CTRL_DATA_W-1:0] axi_s_ctrl_RDATA,
  output [                      1:0] axi_s_ctrl_RRESP,
  output                             axi_s_ctrl_BVALID,
  input                              axi_s_ctrl_BREADY,
  output [                      1:0] axi_s_ctrl_BRESP,
  output                             interrupt
);

  ///////////////////////////////////////////////////////////////////////////////
  // Variables
  ///////////////////////////////////////////////////////////////////////////////
  (* max_fanout = 500 *)
  logic                              rst_n;
  (* max_fanout = 1000 *)
  logic [            P_NUM_ACCU-1:0] acc_rst_n;
  logic                              ap_start;
  logic                              ap_done;
  logic                              ap_done_int;
  logic                              ap_ready;
  logic                              ap_idle;
  logic                              ap_start_pulse;

  logic [            P_NUM_ACCU-1:0] accu_done;
  logic                              all_accu_done;
  logic                              start_host_wr;
  logic                              host_wr_done;

  ///////////////////////////////////////////////////////////////////////////////
  // Host AXI interface + FIFOs
  ///////////////////////////////////////////////////////////////////////////////
  // AXI Lite CTRL regs
  logic [        C_AXI_M_ADDR_W-1:0] src_addr_array           [C_NUM_AXI_M_PORTS-1:0];
  logic [        C_AXI_M_ADDR_W-1:0] dst_addr_array           [C_NUM_AXI_M_PORTS-1:0];
  logic [   C_AXI_S_CTRL_DATA_W-1:0] src_size_array           [C_NUM_AXI_M_PORTS-1:0];
  logic [   C_AXI_S_CTRL_DATA_W-1:0] dst_size_array           [C_NUM_AXI_M_PORTS-1:0];
  logic [   C_AXI_S_CTRL_DATA_W-1:0] mode                     [                  0:0];

  // AXI-M RD signals
  logic [        C_AXI_M_DATA_W-1:0] scal_rd_data;
  logic                              scal_rd_valid;
  logic                              scal_rd_last;
  logic                              scal_rd_ready;

  // FIFOs and intermediate signals in the read path
  logic [        C_AXI_M_DATA_W-1:0] scal_fifo_data;
  logic                              scal_fifo_valid;
  logic                              scal_fifo_last;
  logic                              scal_fifo_ready;

  logic [      C_AXI_M_DATA_W/2-1:0] scal_conv2to1_data;
  logic                              scal_conv2to1_valid;
  logic                              scal_conv2to1_last;
  logic                              scal_conv2to1_ready;

  logic [P_NUM_WIN*P_RED_SCLR_W-1:0] red_scal_split_data_array[       P_NUM_ACCU-1:0];
  logic [            P_NUM_ACCU-1:0] red_scal_split_valid;
  logic [            P_NUM_ACCU-1:0] red_scal_split_last;
  logic [            P_NUM_ACCU-1:0] red_scal_split_ready;

  // Intermediate signals in the write path
  logic [          P_DATA_PNT_W-1:0] bucket_accu_data_array   [       P_NUM_ACCU-1:0];
  logic [            P_NUM_ACCU-1:0] bucket_accu_valid;
  logic [            P_NUM_ACCU-1:0] bucket_accu_last;
  logic [            P_NUM_ACCU-1:0] bucket_accu_ready;

  logic [          P_DATA_PNT_W-1:0] bucket_wr_data;
  logic                              bucket_wr_valid;
  logic                              bucket_wr_ready;

  ///////////////////////////////////////////////////////////////////////////////
  // DDR AXI interface
  ///////////////////////////////////////////////////////////////////////////////
  // AXI-M RD DDR signals
  logic [          P_DATA_PNT_W-1:0] data_pnt_rd_data_array   [       P_NUM_ACCU-1:0];
  logic [            P_NUM_ACCU-1:0] data_pnt_rd_valid;
  logic [            P_NUM_ACCU-1:0] data_pnt_rd_ready;

  ///////////////////////////////////////////////////////////////////////////////
  // RTL Logic 
  ///////////////////////////////////////////////////////////////////////////////
  ap_rst_ctrl #(
    .C_RST_DELAY(C_RST_DELAY)
  ) i_ap_rst_ctrl (
    .ap_clk,
    .ap_rst_n,

    .ap_start,
    .ap_done,

    .ap_srst_n (rst_n),
    .auto_rst_n(),
    .ap_ready,
    .ap_idle,
    .ap_start_pulse
  );

  // AXI4-Lite slave
  axi_s_lite_ctrl #(
    .C_NUM_AXI_M_PORTS(C_NUM_AXI_M_PORTS),
    .C_S_AXI_ADDR_W   (C_AXI_S_CTRL_ADDR_W),
    .C_S_AXI_DATA_W   (C_AXI_S_CTRL_DATA_W)
  ) i_axi_s_lite_ctrl (
    .clk   (ap_clk),
    .rst_n,
    .clk_en(1'b1),

    .ap_start (ap_start),
    .interrupt(interrupt),
    .ap_ready (ap_ready),
    .ap_done  (ap_done),
    .ap_idle  (ap_idle),

    .axi_s_awaddr_i (axi_s_ctrl_AWADDR),
    .axi_s_awvalid_i(axi_s_ctrl_AWVALID),
    .axi_s_awready_o(axi_s_ctrl_AWREADY),
    .axi_s_wdata_i  (axi_s_ctrl_WDATA),
    .axi_s_wstrb_i  (axi_s_ctrl_WSTRB),
    .axi_s_wvalid_i (axi_s_ctrl_WVALID),
    .axi_s_wready_o (axi_s_ctrl_WREADY),
    .axi_s_araddr_i (axi_s_ctrl_ARADDR),
    .axi_s_avalid_i (axi_s_ctrl_ARVALID),
    .axi_s_arready_o(axi_s_ctrl_ARREADY),
    .axi_s_rdata_o  (axi_s_ctrl_RDATA),
    .axi_s_rresp_o  (axi_s_ctrl_RRESP),
    .axi_s_rvalid_o (axi_s_ctrl_RVALID),
    .axi_s_rready_i (axi_s_ctrl_RREADY),
    .axi_s_bresp_o  (axi_s_ctrl_BRESP),
    .axi_s_bvalid_o (axi_s_ctrl_BVALID),
    .axi_s_bready_i (axi_s_ctrl_BREADY),

    // user signals: memory pointer and scal registers parameterized 
    .src_addr_array_o(src_addr_array),
    .dst_addr_array_o(dst_addr_array),
    .src_size_array_o(src_size_array),
    .dst_size_array_o(dst_size_array),
    .reg_array_o     (mode)
  );

  ///////////////////////////////////////////////////////////////////////////////
  // Scalaras path - in SLR1 
  ///////////////////////////////////////////////////////////////////////////////
  // AXI4 Master Read scals from HOST
  axi_m_rd #(
    .C_ADDR_W         (C_AXI_M_ADDR_W),
    .C_DATA_W         (C_AXI_M_DATA_W),
    .C_SIZE_W         (C_AXI_S_CTRL_DATA_W),
    .C_BURST_LEN      (C_AXI_BURST_LEN),
    .C_MAX_OUTSTANDING(C_RD_MAX_OUTSTANDING)
  ) i_axi_m_rd_scals (
    .clk(ap_clk),
    .rst_n,

    .ctrl_start_i (ap_start_pulse),
    .ctrl_done_o  (),
    .ctrl_offset_i(src_addr_array[0]),
    .ctrl_size_i  (src_size_array[0]),

    .axi_m_arvalid_o(axi_m_0_ARVALID),
    .axi_m_arready_i(axi_m_0_ARREADY),
    .axi_m_araddr_o (axi_m_0_ARADDR),
    .axi_m_arlen_o  (axi_m_0_ARLEN),
    .axi_m_arsize_o (axi_m_0_ARSIZE),
    .axi_m_rvalid_i (axi_m_0_RVALID),
    .axi_m_rready_o (axi_m_0_RREADY),
    .axi_m_rdata_i  (axi_m_0_RDATA),
    .axi_m_rlast_i  (axi_m_0_RLAST),
    .axi_m_rresp_i  (axi_m_0_RRESP),

    .axis_m_tdata_o (scal_rd_data),
    .axis_m_tvalid_o(scal_rd_valid),
    .axis_m_tlast_o (),
    .axis_m_tdone_o (scal_rd_last),
    .axis_m_tready_i(scal_rd_ready)
  );

  // Input FIFO
  fifo_axis #(
    .FIFO_DEPTH    (512),
    .FIFO_W        (C_AXI_M_DATA_W),
    .MEM_MACRO_TYPE("block"),
    .HAS_LAST      ("true")
  ) i_fifo_axis_rd (
    .clk(ap_clk),
    .rst_n,

    .axis_data_i (scal_rd_data),
    .axis_valid_i(scal_rd_valid),
    .axis_last_i (scal_rd_last),
    .axis_ready_o(scal_rd_ready),

    .axis_data_o (scal_fifo_data),
    .axis_valid_o(scal_fifo_valid),
    .axis_last_o (scal_fifo_last),
    .axis_ready_i(scal_fifo_ready)
  );

  // Convert scalars from 512b to 256b
  xlnx_axis_conv_512to256 i_xlnx_conv_512to256 (
    .aclk   (ap_clk),
    .aresetn(rst_n),

    .s_axis_tdata (scal_fifo_data),
    .s_axis_tvalid(scal_fifo_valid),
    .s_axis_tlast (scal_fifo_last),
    .s_axis_tready(scal_fifo_ready),

    .m_axis_tdata (scal_conv2to1_data),
    .m_axis_tvalid(scal_conv2to1_valid),
    .m_axis_tlast (scal_conv2to1_last),
    .m_axis_tready(scal_conv2to1_ready)
  );

  // Scalar splitter 
  scalar_splitter #(
    .P_NUM_ACCU(P_NUM_ACCU),
    .P_NUM_WIN (P_NUM_WIN)
  ) i_scalar_splitter (
    .clk(ap_clk),
    .rst_n,

    .axis_s_data_i (scal_conv2to1_data),
    .axis_s_valid_i(scal_conv2to1_valid),
    .axis_s_last_i (scal_conv2to1_last),
    .axis_s_ready_o(scal_conv2to1_ready),

    .axis_m_data_array_o(red_scal_split_data_array),
    .axis_m_valid_o     (red_scal_split_valid),
    .axis_m_last_o      (red_scal_split_last),
    .axis_m_ready_i     (red_scal_split_ready)
  );

  ///////////////////////////////////////////////////////////////////////////////
  // Data points path - in other SLRs 
  ///////////////////////////////////////////////////////////////////////////////
  // AXI4 Master Read - Data Points for Accu0
  axi_m_rd #(
    .C_ADDR_W         (C_AXI_M_ADDR_W),
    .C_DATA_W         (P_DATA_PNT_W),
    .C_SIZE_W         (C_AXI_S_CTRL_DATA_W),
    .C_BURST_LEN      (C_AXI_BURST_LEN),
    .C_MAX_OUTSTANDING(C_RD_MAX_OUTSTANDING)
  ) i_axi_m_rd_pnt_accu0 (
    .clk(ap_clk),
    .rst_n,

    .ctrl_start_i (ap_start_pulse),
    .ctrl_done_o  (),
    .ctrl_offset_i(src_addr_array[1]),
    .ctrl_size_i  (src_size_array[1]),

    .axi_m_arvalid_o(axi_m_1_ARVALID),
    .axi_m_arready_i(axi_m_1_ARREADY),
    .axi_m_araddr_o (axi_m_1_ARADDR),
    .axi_m_arlen_o  (axi_m_1_ARLEN),
    .axi_m_arsize_o (axi_m_1_ARSIZE),
    .axi_m_rvalid_i (axi_m_1_RVALID),
    .axi_m_rready_o (axi_m_1_RREADY),
    .axi_m_rdata_i  (axi_m_1_RDATA[P_DATA_PNT_W-1:0]),
    .axi_m_rlast_i  (axi_m_1_RLAST),
    .axi_m_rresp_i  (axi_m_1_RRESP),

    .axis_m_tdata_o (data_pnt_rd_data_array[0]),
    .axis_m_tvalid_o(data_pnt_rd_valid[0]),
    .axis_m_tlast_o (),
    .axis_m_tready_i(data_pnt_rd_ready[0])
  );

  // AXI4 Master Read - Data Points for Accu1
  axi_m_rd #(
    .C_ADDR_W         (C_AXI_M_ADDR_W),
    .C_DATA_W         (P_DATA_PNT_W),
    .C_SIZE_W         (C_AXI_S_CTRL_DATA_W),
    .C_BURST_LEN      (C_AXI_BURST_LEN),
    .C_MAX_OUTSTANDING(C_RD_MAX_OUTSTANDING)
  ) i_axi_m_rd_pnt_accu1 (
    .clk  (ap_clk),
    .rst_n,

    .ctrl_start_i (ap_start_pulse),
    .ctrl_done_o  (),
    .ctrl_offset_i(src_addr_array[2]),
    .ctrl_size_i  (src_size_array[2]),

    .axi_m_arvalid_o(axi_m_2_ARVALID),
    .axi_m_arready_i(axi_m_2_ARREADY),
    .axi_m_araddr_o (axi_m_2_ARADDR),
    .axi_m_arlen_o  (axi_m_2_ARLEN),
    .axi_m_arsize_o (axi_m_2_ARSIZE),
    .axi_m_rvalid_i (axi_m_2_RVALID),
    .axi_m_rready_o (axi_m_2_RREADY),
    .axi_m_rdata_i  (axi_m_2_RDATA[P_DATA_PNT_W-1:0]),
    .axi_m_rlast_i  (axi_m_2_RLAST),
    .axi_m_rresp_i  (axi_m_2_RRESP),

    .axis_m_tdata_o (data_pnt_rd_data_array[1]),
    .axis_m_tvalid_o(data_pnt_rd_valid[1]),
    .axis_m_tlast_o (),
    .axis_m_tready_i(data_pnt_rd_ready[1])
  );

  // Accumulators 0 and 1
  for (genvar i = 0; i < P_NUM_ACCU; i = i + 1) begin : GEN_ACCU

    accumulator #(
      .P_DATA_PNT_W(P_DATA_PNT_W),  // in the core level, accumulator level (so far in all AXIS ports), we keep data point W = 384,
      .P_RED_SCLR_W(P_RED_SCLR_W),
      .P_NUM_WIN(P_NUM_WIN),
      .P_TOTAL_WIN(P_TOTAL_WIN),
      .P_NUM_AFF_COORD(P_NUM_AFF_COORD),
      .P_NUM_PROJ_COORD(P_NUM_PROJ_COORD)
    ) i_accumulator (
      .clk  (ap_clk),
      .rst_n,

      // Control interface: ap_start/ap_done from XRT and mode from axi4-lite control reg
      .ctrl_ap_start_i(ap_start_pulse),
      .ctrl_mode_i(mode[0][2:0]),
      .ctrl_ap_done_o (accu_done[i]), // This ap_done is geneated when either P_INIT_ACCUM_MODE or P_JUST_ACCUM_MODE is done

      // Read stream - reduced scalars
      .axis_s_red_scal_data_i (red_scal_split_data_array[i]),
      .axis_s_red_scal_valid_i(red_scal_split_valid[i]),
      .axis_s_red_scal_last_i (red_scal_split_last[i]),
      .axis_s_red_scal_ready_o(red_scal_split_ready[i]),

      // Read stream - data points
      .axis_s_pnt_data_i (data_pnt_rd_data_array[i]),
      .axis_s_pnt_valid_i(data_pnt_rd_valid[i]),
      .axis_s_pnt_ready_o(data_pnt_rd_ready[i]),

      // Write stream - buckets
      .axis_m_bucket_data_o (bucket_accu_data_array[i]),
      .axis_m_bucket_valid_o(bucket_accu_valid[i]),
      .axis_m_bucket_last_o (bucket_accu_last[i]),
      .axis_m_bucket_ready_i(bucket_accu_ready[i])
    );
  end

  bucket_wr_scheduler #(
    .P_NUM_ACCU  (P_NUM_ACCU),
    .P_DATA_PNT_W(P_DATA_PNT_W)
  ) i_bucket_wr_scheduler (
    .clk      (ap_clk),
    .rst_n,
    // Bucket streams from the accumulator(s)
    .s_data_i (bucket_accu_data_array),
    .s_valid_i(bucket_accu_valid),
    .s_last_i (bucket_accu_last),
    .s_ready_o(bucket_accu_ready),
    // Bucket write streams to the HOST[0]
    .m_data_o (bucket_wr_data),
    .m_valid_o(bucket_wr_valid),
    .m_ready_i(bucket_wr_ready),

    .start_host_wr_o(start_host_wr)
  );

  // AXI4 Master Write Host
  axi_m_wr #(
    .C_ADDR_W      (C_AXI_M_ADDR_W),
    .C_DATA_W      (C_AXI_M_DATA_W),
    .C_MAX_LENGTH_W(C_AXI_S_CTRL_DATA_W),
    .C_BURST_LEN   (C_AXI_WR_BURST_LEN)
  ) i_axi_m_wr_bucket (
    .clk(ap_clk),
    .rst_n,

    .ctrl_start_i (start_host_wr),
    .ctrl_offset_i(dst_addr_array[0]),
    .ctrl_size_i  (dst_size_array[0]),
    .ctrl_done_o  (host_wr_done),

    .axi_m_awvalid_o(axi_m_0_AWVALID),
    .axi_m_awready_i(axi_m_0_AWREADY),
    .axi_m_awaddr_o (axi_m_0_AWADDR),
    .axi_m_awlen_o  (axi_m_0_AWLEN),
    .axi_m_awsize_o (axi_m_0_AWSIZE),

    .axi_m_wvalid_o(axi_m_0_WVALID),
    .axi_m_wready_i(axi_m_0_WREADY),
    .axi_m_wdata_o (axi_m_0_WDATA),
    .axi_m_wstrb_o (axi_m_0_WSTRB),
    .axi_m_wlast_o (axi_m_0_WLAST),

    .axi_m_bvalid_i(axi_m_0_BVALID),
    .axi_m_bready_o(axi_m_0_BREADY),
    .axi_m_bresp_i (axi_m_0_BRESP),

    .axis_s_tdata_i ({{(C_AXI_M_DATA_W - P_DATA_PNT_W) {1'b0}}, bucket_wr_data}),
    .axis_s_tvalid_i(bucket_wr_valid),
    .axis_s_tready_o(bucket_wr_ready)
  );

  pulse_combine #(
    .P_NUM_PULSES(P_NUM_ACCU)
  ) i_pulse_combine (
    .clk(ap_clk),
    .rst_n,

    .pulse_vector_i(accu_done),
    .pulse_o       (all_accu_done)
  );

  // In P_ACCUM_WRITE_MODE use done from bucket_wr_scheduler
  assign ap_done = (mode[0][P_ACCUM_WRITE_MODE_IDX]) ? host_wr_done : all_accu_done;

endmodule : krnl_msm_381_core

`endif  // KRNL_MSM_381_CORE_SV