

// default_nettype of none prevents implicit wire declaration.
`default_nettype none

`include "makefile_temp.vh"

module krnl_msm_381 #(
  parameter integer C_NUM_AXI_M_PORTS = 1,  // Number of AXI master ports connected to this kernel.
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

  krnl_msm_381_core #(
    .C_NUM_AXI_M_PORTS  (C_NUM_AXI_M_PORTS),
    .C_AXI_S_CTRL_DATA_W(C_AXI_S_CTRL_DATA_W),
    .C_AXI_S_CTRL_ADDR_W(C_AXI_S_CTRL_ADDR_W),
    .C_AXI_M_ADDR_W     (C_AXI_M_ADDR_W),
    .C_AXI_M_DATA_W     (C_AXI_M_DATA_W)
  ) i_krnl_msm_381_core (
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
endmodule : krnl_msm_381

`default_nettype wire