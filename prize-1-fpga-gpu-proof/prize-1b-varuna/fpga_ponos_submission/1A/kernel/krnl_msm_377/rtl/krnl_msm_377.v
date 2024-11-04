

// default_nettype of none prevents implicit wire declaration.
`default_nettype none

`include "makefile_temp.vh"

module krnl_msm_377 #(
  parameter integer C_NUM_AXI_M_PORTS = 4,  // Number of AXI master ports connected to this kernel.
  `include "krnl_common.vh"
) (
  // System signals
  input wire ap_clk,
  input wire ap_rst_n,

  output wire                        axi_m_0_AWVALID,
  input  wire                        axi_m_0_AWREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_0_AWADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_0_AWID,
  output wire [                 7:0] axi_m_0_AWLEN,
  output wire [                 2:0] axi_m_0_AWSIZE,
  output wire [                 1:0] axi_m_0_AWBURST,
  output wire                        axi_m_0_AWLOCK,
  output wire [                 3:0] axi_m_0_AWCACHE,
  output wire [                 2:0] axi_m_0_AWPROT,
  output wire [                 3:0] axi_m_0_AWQOS,
  output wire [                 3:0] axi_m_0_AWREGION,
  output wire                        axi_m_0_WVALID,
  input  wire                        axi_m_0_WREADY,
  output wire [  C_AXI_M_DATA_W-1:0] axi_m_0_WDATA,
  output wire [C_AXI_M_DATA_W/8-1:0] axi_m_0_WSTRB,
  output wire                        axi_m_0_WLAST,
  output wire                        axi_m_0_ARVALID,
  input  wire                        axi_m_0_ARREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_0_ARADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_0_ARID,
  output wire [                 7:0] axi_m_0_ARLEN,
  output wire [                 2:0] axi_m_0_ARSIZE,
  output wire [                 1:0] axi_m_0_ARBURST,
  output wire                        axi_m_0_ARLOCK,
  output wire [                 3:0] axi_m_0_ARCACHE,
  output wire [                 2:0] axi_m_0_ARPROT,
  output wire [                 3:0] axi_m_0_ARQOS,
  output wire [                 3:0] axi_m_0_ARREGION,
  input  wire                        axi_m_0_RVALID,
  output wire                        axi_m_0_RREADY,
  input  wire [  C_AXI_M_DATA_W-1:0] axi_m_0_RDATA,
  input  wire                        axi_m_0_RLAST,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_0_RID,
  input  wire [                 1:0] axi_m_0_RRESP,
  input  wire                        axi_m_0_BVALID,
  output wire                        axi_m_0_BREADY,
  input  wire [                 1:0] axi_m_0_BRESP,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_0_BID,

  output wire                        axi_m_1_AWVALID,
  input  wire                        axi_m_1_AWREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_1_AWADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_1_AWID,
  output wire [                 7:0] axi_m_1_AWLEN,
  output wire [                 2:0] axi_m_1_AWSIZE,
  output wire [                 1:0] axi_m_1_AWBURST,
  output wire                        axi_m_1_AWLOCK,
  output wire [                 3:0] axi_m_1_AWCACHE,
  output wire [                 2:0] axi_m_1_AWPROT,
  output wire [                 3:0] axi_m_1_AWQOS,
  output wire [                 3:0] axi_m_1_AWREGION,
  output wire                        axi_m_1_WVALID,
  input  wire                        axi_m_1_WREADY,
  output wire [  C_AXI_M_DATA_W-1:0] axi_m_1_WDATA,
  output wire [C_AXI_M_DATA_W/8-1:0] axi_m_1_WSTRB,
  output wire                        axi_m_1_WLAST,
  output wire                        axi_m_1_ARVALID,
  input  wire                        axi_m_1_ARREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_1_ARADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_1_ARID,
  output wire [                 7:0] axi_m_1_ARLEN,
  output wire [                 2:0] axi_m_1_ARSIZE,
  output wire [                 1:0] axi_m_1_ARBURST,
  output wire                        axi_m_1_ARLOCK,
  output wire [                 3:0] axi_m_1_ARCACHE,
  output wire [                 2:0] axi_m_1_ARPROT,
  output wire [                 3:0] axi_m_1_ARQOS,
  output wire [                 3:0] axi_m_1_ARREGION,
  input  wire                        axi_m_1_RVALID,
  output wire                        axi_m_1_RREADY,
  input  wire [  C_AXI_M_DATA_W-1:0] axi_m_1_RDATA,
  input  wire                        axi_m_1_RLAST,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_1_RID,
  input  wire [                 1:0] axi_m_1_RRESP,
  input  wire                        axi_m_1_BVALID,
  output wire                        axi_m_1_BREADY,
  input  wire [                 1:0] axi_m_1_BRESP,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_1_BID,

  output wire                        axi_m_2_AWVALID,
  input  wire                        axi_m_2_AWREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_2_AWADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_2_AWID,
  output wire [                 7:0] axi_m_2_AWLEN,
  output wire [                 2:0] axi_m_2_AWSIZE,
  output wire [                 1:0] axi_m_2_AWBURST,
  output wire                        axi_m_2_AWLOCK,
  output wire [                 3:0] axi_m_2_AWCACHE,
  output wire [                 2:0] axi_m_2_AWPROT,
  output wire [                 3:0] axi_m_2_AWQOS,
  output wire [                 3:0] axi_m_2_AWREGION,
  output wire                        axi_m_2_WVALID,
  input  wire                        axi_m_2_WREADY,
  output wire [  C_AXI_M_DATA_W-1:0] axi_m_2_WDATA,
  output wire [C_AXI_M_DATA_W/8-1:0] axi_m_2_WSTRB,
  output wire                        axi_m_2_WLAST,
  output wire                        axi_m_2_ARVALID,
  input  wire                        axi_m_2_ARREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_2_ARADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_2_ARID,
  output wire [                 7:0] axi_m_2_ARLEN,
  output wire [                 2:0] axi_m_2_ARSIZE,
  output wire [                 1:0] axi_m_2_ARBURST,
  output wire                        axi_m_2_ARLOCK,
  output wire [                 3:0] axi_m_2_ARCACHE,
  output wire [                 2:0] axi_m_2_ARPROT,
  output wire [                 3:0] axi_m_2_ARQOS,
  output wire [                 3:0] axi_m_2_ARREGION,
  input  wire                        axi_m_2_RVALID,
  output wire                        axi_m_2_RREADY,
  input  wire [  C_AXI_M_DATA_W-1:0] axi_m_2_RDATA,
  input  wire                        axi_m_2_RLAST,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_2_RID,
  input  wire [                 1:0] axi_m_2_RRESP,
  input  wire                        axi_m_2_BVALID,
  output wire                        axi_m_2_BREADY,
  input  wire [                 1:0] axi_m_2_BRESP,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_2_BID,

  output wire                        axi_m_3_AWVALID,
  input  wire                        axi_m_3_AWREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_3_AWADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_3_AWID,
  output wire [                 7:0] axi_m_3_AWLEN,
  output wire [                 2:0] axi_m_3_AWSIZE,
  output wire [                 1:0] axi_m_3_AWBURST,
  output wire                        axi_m_3_AWLOCK,
  output wire [                 3:0] axi_m_3_AWCACHE,
  output wire [                 2:0] axi_m_3_AWPROT,
  output wire [                 3:0] axi_m_3_AWQOS,
  output wire [                 3:0] axi_m_3_AWREGION,
  output wire                        axi_m_3_WVALID,
  input  wire                        axi_m_3_WREADY,
  output wire [  C_AXI_M_DATA_W-1:0] axi_m_3_WDATA,
  output wire [C_AXI_M_DATA_W/8-1:0] axi_m_3_WSTRB,
  output wire                        axi_m_3_WLAST,
  output wire                        axi_m_3_ARVALID,
  input  wire                        axi_m_3_ARREADY,
  output wire [  C_AXI_M_ADDR_W-1:0] axi_m_3_ARADDR,
  output wire [    C_AXI_M_ID_W-1:0] axi_m_3_ARID,
  output wire [                 7:0] axi_m_3_ARLEN,
  output wire [                 2:0] axi_m_3_ARSIZE,
  output wire [                 1:0] axi_m_3_ARBURST,
  output wire                        axi_m_3_ARLOCK,
  output wire [                 3:0] axi_m_3_ARCACHE,
  output wire [                 2:0] axi_m_3_ARPROT,
  output wire [                 3:0] axi_m_3_ARQOS,
  output wire [                 3:0] axi_m_3_ARREGION,
  input  wire                        axi_m_3_RVALID,
  output wire                        axi_m_3_RREADY,
  input  wire [  C_AXI_M_DATA_W-1:0] axi_m_3_RDATA,
  input  wire                        axi_m_3_RLAST,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_3_RID,
  input  wire [                 1:0] axi_m_3_RRESP,
  input  wire                        axi_m_3_BVALID,
  output wire                        axi_m_3_BREADY,
  input  wire [                 1:0] axi_m_3_BRESP,
  input  wire [    C_AXI_M_ID_W-1:0] axi_m_3_BID,

  // AXI4-Lite slave interface
  input  wire                             s_axi_control_AWVALID,
  output wire                             s_axi_control_AWREADY,
  input  wire [  C_AXI_S_CTRL_ADDR_W-1:0] s_axi_control_AWADDR,
  input  wire                             s_axi_control_WVALID,
  output wire                             s_axi_control_WREADY,
  input  wire [  C_AXI_S_CTRL_DATA_W-1:0] s_axi_control_WDATA,
  input  wire [C_AXI_S_CTRL_DATA_W/8-1:0] s_axi_control_WSTRB,
  input  wire                             s_axi_control_ARVALID,
  output wire                             s_axi_control_ARREADY,
  input  wire [  C_AXI_S_CTRL_ADDR_W-1:0] s_axi_control_ARADDR,
  output wire                             s_axi_control_RVALID,
  input  wire                             s_axi_control_RREADY,
  output wire [  C_AXI_S_CTRL_DATA_W-1:0] s_axi_control_RDATA,
  output wire [                      1:0] s_axi_control_RRESP,
  output wire                             s_axi_control_BVALID,
  input  wire                             s_axi_control_BREADY,
  output wire [                      1:0] s_axi_control_BRESP,
  output wire                             interrupt
);

  // Tie-off unused AXI protocol features
  assign axi_m_0_AWID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_0_AWBURST  = 2'b01;
  assign axi_m_0_AWLOCK   = 1'b0;
  assign axi_m_0_AWCACHE  = 4'b0011;
  assign axi_m_0_AWPROT   = 3'b000;
  assign axi_m_0_AWQOS    = 4'b0000;
  assign axi_m_0_AWREGION = 4'b0000;
  assign axi_m_0_ARID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_0_ARBURST  = 2'b01;
  assign axi_m_0_ARLOCK   = 1'b0;
  assign axi_m_0_ARCACHE  = 4'b0011;
  assign axi_m_0_ARPROT   = 3'b000;
  assign axi_m_0_ARQOS    = 4'b0000;
  assign axi_m_0_ARREGION = 4'b0000;

  assign axi_m_1_AWID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_1_AWBURST  = 2'b01;
  assign axi_m_1_AWLOCK   = 1'b0;
  assign axi_m_1_AWCACHE  = 4'b0011;
  assign axi_m_1_AWPROT   = 3'b000;
  assign axi_m_1_AWQOS    = 4'b0000;
  assign axi_m_1_AWREGION = 4'b0000;
  assign axi_m_1_ARID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_1_ARBURST  = 2'b01;
  assign axi_m_1_ARLOCK   = 1'b0;
  assign axi_m_1_ARCACHE  = 4'b0011;
  assign axi_m_1_ARPROT   = 3'b000;
  assign axi_m_1_ARQOS    = 4'b0000;
  assign axi_m_1_ARREGION = 4'b0000;

  assign axi_m_2_AWID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_2_AWBURST  = 2'b01;
  assign axi_m_2_AWLOCK   = 1'b0;
  assign axi_m_2_AWCACHE  = 4'b0011;
  assign axi_m_2_AWPROT   = 3'b000;
  assign axi_m_2_AWQOS    = 4'b0000;
  assign axi_m_2_AWREGION = 4'b0000;
  assign axi_m_2_ARID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_2_ARBURST  = 2'b01;
  assign axi_m_2_ARLOCK   = 1'b0;
  assign axi_m_2_ARCACHE  = 4'b0011;
  assign axi_m_2_ARPROT   = 3'b000;
  assign axi_m_2_ARQOS    = 4'b0000;
  assign axi_m_2_ARREGION = 4'b0000;

  assign axi_m_3_AWID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_3_AWBURST  = 2'b01;
  assign axi_m_3_AWLOCK   = 1'b0;
  assign axi_m_3_AWCACHE  = 4'b0011;
  assign axi_m_3_AWPROT   = 3'b000;
  assign axi_m_3_AWQOS    = 4'b0000;
  assign axi_m_3_AWREGION = 4'b0000;
  assign axi_m_3_ARID     = {C_AXI_M_ID_W{1'b0}};
  assign axi_m_3_ARBURST  = 2'b01;
  assign axi_m_3_ARLOCK   = 1'b0;
  assign axi_m_3_ARCACHE  = 4'b0011;
  assign axi_m_3_ARPROT   = 3'b000;
  assign axi_m_3_ARQOS    = 4'b0000;
  assign axi_m_3_ARREGION = 4'b0000;

  // Tie-off unused part of each channel (write for all DDRs - AXI_M_1, AXI_M_2, AXI_M_3)
  assign axi_m_1_AWVALID  = 1'b0;
  assign axi_m_1_AWADDR   = {C_AXI_M_ADDR_W{1'b0}};
  assign axi_m_1_AWLEN    = 7'b0;
  assign axi_m_1_AWSIZE   = 2'b0;
  assign axi_m_1_WVALID   = 1'b0;
  assign axi_m_1_WDATA    = {C_AXI_M_DATA_W{1'b0}};
  assign axi_m_1_WSTRB    = {C_AXI_M_DATA_W / 8{1'b0}};
  assign axi_m_1_WLAST    = 1'b0;
  assign axi_m_1_BREADY   = 1'b0;

  assign axi_m_2_AWVALID  = 1'b0;
  assign axi_m_2_AWADDR   = {C_AXI_M_ADDR_W{1'b0}};
  assign axi_m_2_AWLEN    = 7'b0;
  assign axi_m_2_AWSIZE   = 2'b0;
  assign axi_m_2_WVALID   = 1'b0;
  assign axi_m_2_WDATA    = {C_AXI_M_DATA_W{1'b0}};
  assign axi_m_2_WSTRB    = {C_AXI_M_DATA_W / 8{1'b0}};
  assign axi_m_2_WLAST    = 1'b0;
  assign axi_m_2_BREADY   = 1'b0;

  assign axi_m_3_AWVALID  = 1'b0;
  assign axi_m_3_AWADDR   = {C_AXI_M_ADDR_W{1'b0}};
  assign axi_m_3_AWLEN    = 7'b0;
  assign axi_m_3_AWSIZE   = 2'b0;
  assign axi_m_3_WVALID   = 1'b0;
  assign axi_m_3_WDATA    = {C_AXI_M_DATA_W{1'b0}};
  assign axi_m_3_WSTRB    = {C_AXI_M_DATA_W / 8{1'b0}};
  assign axi_m_3_WLAST    = 1'b0;
  assign axi_m_3_BREADY   = 1'b0;

  krnl_msm_377_core #(
    .C_NUM_AXI_M_PORTS  (C_NUM_AXI_M_PORTS),
    .C_AXI_S_CTRL_DATA_W(C_AXI_S_CTRL_DATA_W),
    .C_AXI_S_CTRL_ADDR_W(C_AXI_S_CTRL_ADDR_W),
    .C_AXI_M_ADDR_W     (C_AXI_M_ADDR_W),
    .C_AXI_M_DATA_W     (C_AXI_M_DATA_W)
  ) i_krnl_msm_377_core (
    .ap_clk  (ap_clk),
    .ap_rst_n(ap_rst_n),

    .axi_m_0_ARVALID(axi_m_0_ARVALID),
    .axi_m_0_ARREADY(axi_m_0_ARREADY),
    .axi_m_0_ARADDR (axi_m_0_ARADDR),
    .axi_m_0_ARLEN  (axi_m_0_ARLEN),
    .axi_m_0_ARSIZE (axi_m_0_ARSIZE),
    .axi_m_0_RVALID (axi_m_0_RVALID),
    .axi_m_0_RREADY (axi_m_0_RREADY),
    .axi_m_0_RDATA  (axi_m_0_RDATA),
    .axi_m_0_RLAST  (axi_m_0_RLAST),
    .axi_m_0_RRESP  (axi_m_0_RRESP),
    .axi_m_0_AWVALID(axi_m_0_AWVALID),
    .axi_m_0_AWREADY(axi_m_0_AWREADY),
    .axi_m_0_AWADDR (axi_m_0_AWADDR),
    .axi_m_0_AWLEN  (axi_m_0_AWLEN),
    .axi_m_0_AWSIZE (axi_m_0_AWSIZE),
    .axi_m_0_WVALID (axi_m_0_WVALID),
    .axi_m_0_WREADY (axi_m_0_WREADY),
    .axi_m_0_WDATA  (axi_m_0_WDATA),
    .axi_m_0_WSTRB  (axi_m_0_WSTRB),
    .axi_m_0_WLAST  (axi_m_0_WLAST),
    .axi_m_0_BVALID (axi_m_0_BVALID),
    .axi_m_0_BREADY (axi_m_0_BREADY),
    .axi_m_0_BRESP  (axi_m_0_BRESP),

    .axi_m_1_ARVALID(axi_m_1_ARVALID),
    .axi_m_1_ARREADY(axi_m_1_ARREADY),
    .axi_m_1_ARADDR (axi_m_1_ARADDR),
    .axi_m_1_ARLEN  (axi_m_1_ARLEN),
    .axi_m_1_ARSIZE (axi_m_1_ARSIZE),
    .axi_m_1_RVALID (axi_m_1_RVALID),
    .axi_m_1_RREADY (axi_m_1_RREADY),
    .axi_m_1_RDATA  (axi_m_1_RDATA),
    .axi_m_1_RLAST  (axi_m_1_RLAST),
    .axi_m_1_RRESP  (axi_m_1_RRESP),

    .axi_m_2_ARVALID(axi_m_2_ARVALID),
    .axi_m_2_ARREADY(axi_m_2_ARREADY),
    .axi_m_2_ARADDR (axi_m_2_ARADDR),
    .axi_m_2_ARLEN  (axi_m_2_ARLEN),
    .axi_m_2_ARSIZE (axi_m_2_ARSIZE),
    .axi_m_2_RVALID (axi_m_2_RVALID),
    .axi_m_2_RREADY (axi_m_2_RREADY),
    .axi_m_2_RDATA  (axi_m_2_RDATA),
    .axi_m_2_RLAST  (axi_m_2_RLAST),
    .axi_m_2_RRESP  (axi_m_2_RRESP),

    .axi_m_3_ARVALID(axi_m_3_ARVALID),
    .axi_m_3_ARREADY(axi_m_3_ARREADY),
    .axi_m_3_ARADDR (axi_m_3_ARADDR),
    .axi_m_3_ARLEN  (axi_m_3_ARLEN),
    .axi_m_3_ARSIZE (axi_m_3_ARSIZE),
    .axi_m_3_RVALID (axi_m_3_RVALID),
    .axi_m_3_RREADY (axi_m_3_RREADY),
    .axi_m_3_RDATA  (axi_m_3_RDATA),
    .axi_m_3_RLAST  (axi_m_3_RLAST),
    .axi_m_3_RRESP  (axi_m_3_RRESP),

    .axi_s_ctrl_AWVALID(s_axi_control_AWVALID),
    .axi_s_ctrl_AWREADY(s_axi_control_AWREADY),
    .axi_s_ctrl_AWADDR (s_axi_control_AWADDR),
    .axi_s_ctrl_WVALID (s_axi_control_WVALID),
    .axi_s_ctrl_WREADY (s_axi_control_WREADY),
    .axi_s_ctrl_WDATA  (s_axi_control_WDATA),
    .axi_s_ctrl_WSTRB  (s_axi_control_WSTRB),
    .axi_s_ctrl_ARVALID(s_axi_control_ARVALID),
    .axi_s_ctrl_ARREADY(s_axi_control_ARREADY),
    .axi_s_ctrl_ARADDR (s_axi_control_ARADDR),
    .axi_s_ctrl_RVALID (s_axi_control_RVALID),
    .axi_s_ctrl_RREADY (s_axi_control_RREADY),
    .axi_s_ctrl_RDATA  (s_axi_control_RDATA),
    .axi_s_ctrl_RRESP  (s_axi_control_RRESP),
    .axi_s_ctrl_BVALID (s_axi_control_BVALID),
    .axi_s_ctrl_BREADY (s_axi_control_BREADY),
    .axi_s_ctrl_BRESP  (s_axi_control_BRESP),
    .interrupt         (interrupt)
  );

endmodule

`default_nettype wire