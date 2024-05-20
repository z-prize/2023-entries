/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

///////////////////////////////////////////////////////////////////////////////
// Description: This is a example of how to create an RTL Kernel.  The function
// of this module is to add two 32-bit values and produce a result.  The values
// are read from one AXI4 memory mapped master, processed and then written out.
//
// Data flow: axi_read_master->fifo[2]->adder->fifo->axi_write_master
///////////////////////////////////////////////////////////////////////////////

// default_nettype of none prevents implicit wire declaration.
`default_nettype none
`timescale 1 ns / 1 ps 
`include "basic.vh"
`include "env.vh"

module krnl_vadd_rtl_int import zprize_param::* ; #( 
  parameter integer  C_S_AXI_CONTROL_DATA_WIDTH = 32,
  parameter integer  C_S_AXI_CONTROL_ADDR_WIDTH = 6,
  //parameter integer  C_M_AXI_GMEM_ID_WIDTH = 1,
  parameter integer  C_M_AXI_GMEM_ID_WIDTH0 = 1,
  parameter integer  C_M_AXI_GMEM_ID_WIDTH1 = 1,
  parameter integer  C_M_AXI_GMEM_ID_WIDTH2 = 1,
  parameter integer  C_M_AXI_GMEM_ID_WIDTH3 = 1,
  parameter integer  C_M_AXI_GMEM_ID_WIDTH4 = 1,
  parameter integer  C_M_AXI_GMEM_ADDR_WIDTH = 64,
  parameter integer  C_M_AXI_GMEM_DATA_WIDTH = 32
)
(
  // System signals
  input  wire  ap_clk,
  input  wire  ap_rst_n,
//-----------------------------------------------------------------
//-----------------------------------------------------------------
  // AXI4 master interface 
  output wire                                 m_axi_gmem0_AWVALID,
  input  wire                                 m_axi_gmem0_AWREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem0_AWADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH0 - 1:0]  m_axi_gmem0_AWID,
  output wire [7:0]                           m_axi_gmem0_AWLEN,
  output wire [2:0]                           m_axi_gmem0_AWSIZE,
  // Tie-off AXI4 transaction options that are not being used.
  output wire [1:0]                           m_axi_gmem0_AWBURST,
  output wire [1:0]                           m_axi_gmem0_AWLOCK,
  output wire [3:0]                           m_axi_gmem0_AWCACHE,
  output wire [2:0]                           m_axi_gmem0_AWPROT,
  output wire [3:0]                           m_axi_gmem0_AWQOS,
  output wire [3:0]                           m_axi_gmem0_AWREGION,
  output wire                                 m_axi_gmem0_WVALID,
  input  wire                                 m_axi_gmem0_WREADY,
  output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem0_WDATA,
  output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem0_WSTRB,
  output wire                                 m_axi_gmem0_WLAST,
  output wire                                 m_axi_gmem0_ARVALID,
  input  wire                                 m_axi_gmem0_ARREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem0_ARADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH0-1:0]    m_axi_gmem0_ARID,
  output wire [7:0]                           m_axi_gmem0_ARLEN,
  output wire [2:0]                           m_axi_gmem0_ARSIZE,
  output wire [1:0]                           m_axi_gmem0_ARBURST,
  output wire [1:0]                           m_axi_gmem0_ARLOCK,
  output wire [3:0]                           m_axi_gmem0_ARCACHE,
  output wire [2:0]                           m_axi_gmem0_ARPROT,
  output wire [3:0]                           m_axi_gmem0_ARQOS,
  output wire [3:0]                           m_axi_gmem0_ARREGION,
  input  wire                                 m_axi_gmem0_RVALID,
  output wire                                 m_axi_gmem0_RREADY,
  input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem0_RDATA,
  input  wire                                 m_axi_gmem0_RLAST,
  input  wire [C_M_AXI_GMEM_ID_WIDTH0 - 1:0]  m_axi_gmem0_RID,
  input  wire [1:0]                           m_axi_gmem0_RRESP,
  input  wire                                 m_axi_gmem0_BVALID,
  output wire                                 m_axi_gmem0_BREADY,
  input  wire [1:0]                           m_axi_gmem0_BRESP,
  input  wire [C_M_AXI_GMEM_ID_WIDTH0 - 1:0]  m_axi_gmem0_BID,

  // AXI4 master interface 
  output wire                                 m_axi_gmem1_AWVALID,
  input  wire                                 m_axi_gmem1_AWREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem1_AWADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH1 - 1:0]  m_axi_gmem1_AWID,
  output wire [7:0]                           m_axi_gmem1_AWLEN,
  output wire [2:0]                           m_axi_gmem1_AWSIZE,
  // Tie-off AXI4 transaction options that are not being used.
  output wire [1:0]                           m_axi_gmem1_AWBURST,
  output wire [1:0]                           m_axi_gmem1_AWLOCK,
  output wire [3:0]                           m_axi_gmem1_AWCACHE,
  output wire [2:0]                           m_axi_gmem1_AWPROT,
  output wire [3:0]                           m_axi_gmem1_AWQOS,
  output wire [3:0]                           m_axi_gmem1_AWREGION,
  output wire                                 m_axi_gmem1_WVALID,
  input  wire                                 m_axi_gmem1_WREADY,
  output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem1_WDATA,
  output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem1_WSTRB,
  output wire                                 m_axi_gmem1_WLAST,
  output wire                                 m_axi_gmem1_ARVALID,
  input  wire                                 m_axi_gmem1_ARREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem1_ARADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH1-1:0]    m_axi_gmem1_ARID,
  output wire [7:0]                           m_axi_gmem1_ARLEN,
  output wire [2:0]                           m_axi_gmem1_ARSIZE,
  output wire [1:0]                           m_axi_gmem1_ARBURST,
  output wire [1:0]                           m_axi_gmem1_ARLOCK,
  output wire [3:0]                           m_axi_gmem1_ARCACHE,
  output wire [2:0]                           m_axi_gmem1_ARPROT,
  output wire [3:0]                           m_axi_gmem1_ARQOS,
  output wire [3:0]                           m_axi_gmem1_ARREGION,
  input  wire                                 m_axi_gmem1_RVALID,
  output wire                                 m_axi_gmem1_RREADY,
  input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem1_RDATA,
  input  wire                                 m_axi_gmem1_RLAST,
  input  wire [C_M_AXI_GMEM_ID_WIDTH1 - 1:0]  m_axi_gmem1_RID,
  input  wire [1:0]                           m_axi_gmem1_RRESP,
  input  wire                                 m_axi_gmem1_BVALID,
  output wire                                 m_axi_gmem1_BREADY,
  input  wire [1:0]                           m_axi_gmem1_BRESP,
  input  wire [C_M_AXI_GMEM_ID_WIDTH1 - 1:0]  m_axi_gmem1_BID,

  // AXI4 master interface 
  output wire                                 m_axi_gmem2_AWVALID,
  input  wire                                 m_axi_gmem2_AWREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem2_AWADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH2 - 1:0]  m_axi_gmem2_AWID,
  output wire [7:0]                           m_axi_gmem2_AWLEN,
  output wire [2:0]                           m_axi_gmem2_AWSIZE,
  // Tie-off AXI4 transaction options that are not being used.
  output wire [1:0]                           m_axi_gmem2_AWBURST,
  output wire [1:0]                           m_axi_gmem2_AWLOCK,
  output wire [3:0]                           m_axi_gmem2_AWCACHE,
  output wire [2:0]                           m_axi_gmem2_AWPROT,
  output wire [3:0]                           m_axi_gmem2_AWQOS,
  output wire [3:0]                           m_axi_gmem2_AWREGION,
  output wire                                 m_axi_gmem2_WVALID,
  input  wire                                 m_axi_gmem2_WREADY,
  output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem2_WDATA,
  output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem2_WSTRB,
  output wire                                 m_axi_gmem2_WLAST,
  output wire                                 m_axi_gmem2_ARVALID,
  input  wire                                 m_axi_gmem2_ARREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem2_ARADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH2-1:0]    m_axi_gmem2_ARID,
  output wire [7:0]                           m_axi_gmem2_ARLEN,
  output wire [2:0]                           m_axi_gmem2_ARSIZE,
  output wire [1:0]                           m_axi_gmem2_ARBURST,
  output wire [1:0]                           m_axi_gmem2_ARLOCK,
  output wire [3:0]                           m_axi_gmem2_ARCACHE,
  output wire [2:0]                           m_axi_gmem2_ARPROT,
  output wire [3:0]                           m_axi_gmem2_ARQOS,
  output wire [3:0]                           m_axi_gmem2_ARREGION,
  input  wire                                 m_axi_gmem2_RVALID,
  output wire                                 m_axi_gmem2_RREADY,
  input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem2_RDATA,
  input  wire                                 m_axi_gmem2_RLAST,
  input  wire [C_M_AXI_GMEM_ID_WIDTH2 - 1:0]  m_axi_gmem2_RID,
  input  wire [1:0]                           m_axi_gmem2_RRESP,
  input  wire                                 m_axi_gmem2_BVALID,
  output wire                                 m_axi_gmem2_BREADY,
  input  wire [1:0]                           m_axi_gmem2_BRESP,
  input  wire [C_M_AXI_GMEM_ID_WIDTH2 - 1:0]  m_axi_gmem2_BID,

  // AXI4 master interface 
  output wire                                 m_axi_gmem3_AWVALID,
  input  wire                                 m_axi_gmem3_AWREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem3_AWADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH3 - 1:0]  m_axi_gmem3_AWID,
  output wire [7:0]                           m_axi_gmem3_AWLEN,
  output wire [2:0]                           m_axi_gmem3_AWSIZE,
  // Tie-off AXI4 transaction options that are not being used.
  output wire [1:0]                           m_axi_gmem3_AWBURST,
  output wire [1:0]                           m_axi_gmem3_AWLOCK,
  output wire [3:0]                           m_axi_gmem3_AWCACHE,
  output wire [2:0]                           m_axi_gmem3_AWPROT,
  output wire [3:0]                           m_axi_gmem3_AWQOS,
  output wire [3:0]                           m_axi_gmem3_AWREGION,
  output wire                                 m_axi_gmem3_WVALID,
  input  wire                                 m_axi_gmem3_WREADY,
  output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem3_WDATA,
  output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem3_WSTRB,
  output wire                                 m_axi_gmem3_WLAST,
  output wire                                 m_axi_gmem3_ARVALID,
  input  wire                                 m_axi_gmem3_ARREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem3_ARADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH3-1:0]    m_axi_gmem3_ARID,
  output wire [7:0]                           m_axi_gmem3_ARLEN,
  output wire [2:0]                           m_axi_gmem3_ARSIZE,
  output wire [1:0]                           m_axi_gmem3_ARBURST,
  output wire [1:0]                           m_axi_gmem3_ARLOCK,
  output wire [3:0]                           m_axi_gmem3_ARCACHE,
  output wire [2:0]                           m_axi_gmem3_ARPROT,
  output wire [3:0]                           m_axi_gmem3_ARQOS,
  output wire [3:0]                           m_axi_gmem3_ARREGION,
  input  wire                                 m_axi_gmem3_RVALID,
  output wire                                 m_axi_gmem3_RREADY,
  input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem3_RDATA,
  input  wire                                 m_axi_gmem3_RLAST,
  input  wire [C_M_AXI_GMEM_ID_WIDTH3 - 1:0]  m_axi_gmem3_RID,
  input  wire [1:0]                           m_axi_gmem3_RRESP,
  input  wire                                 m_axi_gmem3_BVALID,
  output wire                                 m_axi_gmem3_BREADY,
  input  wire [1:0]                           m_axi_gmem3_BRESP,
  input  wire [C_M_AXI_GMEM_ID_WIDTH3 - 1:0]  m_axi_gmem3_BID,

  // AXI4 master interface 
  output wire                                 m_axi_gmem4_AWVALID,
  input  wire                                 m_axi_gmem4_AWREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem4_AWADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH4 - 1:0]  m_axi_gmem4_AWID,
  output wire [7:0]                           m_axi_gmem4_AWLEN,
  output wire [2:0]                           m_axi_gmem4_AWSIZE,
  // Tie-off AXI4 transaction options that are not being used.
  output wire [1:0]                           m_axi_gmem4_AWBURST,
  output wire [1:0]                           m_axi_gmem4_AWLOCK,
  output wire [3:0]                           m_axi_gmem4_AWCACHE,
  output wire [2:0]                           m_axi_gmem4_AWPROT,
  output wire [3:0]                           m_axi_gmem4_AWQOS,
  output wire [3:0]                           m_axi_gmem4_AWREGION,
  output wire                                 m_axi_gmem4_WVALID,
  input  wire                                 m_axi_gmem4_WREADY,
  output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem4_WDATA,
  output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem4_WSTRB,
  output wire                                 m_axi_gmem4_WLAST,
  output wire                                 m_axi_gmem4_ARVALID,
  input  wire                                 m_axi_gmem4_ARREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem4_ARADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH4-1:0]    m_axi_gmem4_ARID,
  output wire [7:0]                           m_axi_gmem4_ARLEN,
  output wire [2:0]                           m_axi_gmem4_ARSIZE,
  output wire [1:0]                           m_axi_gmem4_ARBURST,
  output wire [1:0]                           m_axi_gmem4_ARLOCK,
  output wire [3:0]                           m_axi_gmem4_ARCACHE,
  output wire [2:0]                           m_axi_gmem4_ARPROT,
  output wire [3:0]                           m_axi_gmem4_ARQOS,
  output wire [3:0]                           m_axi_gmem4_ARREGION,
  input  wire                                 m_axi_gmem4_RVALID,
  output wire                                 m_axi_gmem4_RREADY,
  input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem4_RDATA,
  input  wire                                 m_axi_gmem4_RLAST,
  input  wire [C_M_AXI_GMEM_ID_WIDTH4 - 1:0]  m_axi_gmem4_RID,
  input  wire [1:0]                           m_axi_gmem4_RRESP,
  input  wire                                 m_axi_gmem4_BVALID,
  output wire                                 m_axi_gmem4_BREADY,
  input  wire [1:0]                           m_axi_gmem4_BRESP,
  input  wire [C_M_AXI_GMEM_ID_WIDTH4 - 1:0]  m_axi_gmem4_BID,

  // AXI4-Lite slave interface
  input  wire                                    s_axi_control_AWVALID,
  output wire                                    s_axi_control_AWREADY,
  input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_AWADDR,
  input  wire                                    s_axi_control_WVALID,
  output wire                                    s_axi_control_WREADY,
  input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_WDATA,
  input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_WSTRB,
  input  wire                                    s_axi_control_ARVALID,
  output wire                                    s_axi_control_ARREADY,
  input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_ARADDR,
  output wire                                    s_axi_control_RVALID,
  input  wire                                    s_axi_control_RREADY,
  output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_RDATA,
  output wire [1:0]                              s_axi_control_RRESP,
  output wire                                    s_axi_control_BVALID,
  input  wire                                    s_axi_control_BREADY,
  output wire [1:0]                              s_axi_control_BRESP,
  output wire                                    interrupt 
);
///////////////////////////////////////////////////////////////////////////////
// Local Parameters (constants)
///////////////////////////////////////////////////////////////////////////////
localparam integer LP_NUM_READ_CHANNELS0  = 1;
localparam integer LP_NUM_READ_CHANNELS1  = 1;
localparam integer LP_NUM_READ_CHANNELS2  = 1;
localparam integer LP_NUM_READ_CHANNELS3  = 1;
localparam integer LP_NUM_READ_CHANNELS4  = 1;
localparam integer LP_LENGTH_WIDTH       = 32;
localparam integer LP_DW_BYTES           = C_M_AXI_GMEM_DATA_WIDTH/8;//64
localparam integer LP_AXI_BURST_LEN      = 4096/LP_DW_BYTES < 256 ? 4096/LP_DW_BYTES : 256;//64
localparam integer LP_LOG_BURST_LEN      = $clog2(LP_AXI_BURST_LEN);
localparam integer LP_RD_MAX_OUTSTANDING = 7;
localparam integer LP_RD_FIFO_DEPTH      = LP_AXI_BURST_LEN*(LP_RD_MAX_OUTSTANDING + 1);//512
localparam integer LP_WR_FIFO_DEPTH      = LP_AXI_BURST_LEN;


///////////////////////////////////////////////////////////////////////////////
// Variables
///////////////////////////////////////////////////////////////////////////////
logic areset = 1'b0;  
logic ap_start;
logic ap_start_pulse;
`KEEP logic ap_start_pulse_slr0_dly4;
`KEEP logic ap_start_pulse_slr1_dly4;
`KEEP logic ap_start_pulse_slr2_dly4;
`KEEP logic ap_start_pulse_slr3_dly4;
logic ap_start_r;
logic ap_ready;
logic ap_done0; // mean kernel is passed
logic ap_done3;
logic ap_done4;
`KEEP logic ap_done0_slr0_dly4;
`KEEP logic ap_done3_slr2_dly4;
`KEEP logic ap_done4_slr3_dly4;
logic ap_idle = 1'b1;

// AXI_0
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in0; 	//or
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] out0; 	//ow
// AXI_1
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in1_0; 	//or
//logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] out1; 	//ow
// AXI_2
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in1_1; 	//or
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in1_2; 	//or
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] scalar_addr; 	//or
// AXI_3
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in2; 	//or
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] out2; 	//ow
// AXI_4
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in3; 	//or
logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] out3; 	//ow

`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in0_slr0_dly4;
`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] out0_slr0_dly4;

`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in1_0_slr1_dly4;
`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in1_1_slr1_dly4;

`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in2_slr2_dly4;
`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] out2_slr2_dly4;

`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] in3_slr3_dly4;
`KEEP logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] out3_slr3_dly4;

logic [LP_LENGTH_WIDTH-1:0]         length_r;
logic [LP_LENGTH_WIDTH-1:0]         scalar_flag;

logic read_done0;
logic [LP_NUM_READ_CHANNELS0-1:0] rd_tvalid0;
logic [LP_NUM_READ_CHANNELS0-1:0] rd_tready_n0; 
logic [LP_NUM_READ_CHANNELS0-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_tdata0;
logic [LP_NUM_READ_CHANNELS0-1:0] ctrl_rd_fifo_prog_full0;
logic [LP_NUM_READ_CHANNELS0-1:0] rd_fifo_tvalid_n0;
logic [LP_NUM_READ_CHANNELS0-1:0] rd_fifo_tready0; 
logic [LP_NUM_READ_CHANNELS0-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_fifo_tdata0;

logic read_done1;
logic [LP_NUM_READ_CHANNELS1-1:0] rd_tvalid1;
logic [LP_NUM_READ_CHANNELS1-1:0] rd_tready_n1; 
logic [LP_NUM_READ_CHANNELS1-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_tdata1;
logic [LP_NUM_READ_CHANNELS1-1:0] ctrl_rd_fifo_prog_full1;
logic [LP_NUM_READ_CHANNELS1-1:0] rd_fifo_tvalid_n1;
logic [LP_NUM_READ_CHANNELS1-1:0] rd_fifo_tready1; 
logic [LP_NUM_READ_CHANNELS1-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_fifo_tdata1;

logic read_done2;
logic [LP_NUM_READ_CHANNELS2-1:0] rd_tvalid2;
logic [LP_NUM_READ_CHANNELS2-1:0] rd_tready_n2; 
logic [LP_NUM_READ_CHANNELS2-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_tdata2;
logic [LP_NUM_READ_CHANNELS2-1:0] ctrl_rd_fifo_prog_full2;
logic [LP_NUM_READ_CHANNELS2-1:0] rd_fifo_tvalid_n2;
logic [LP_NUM_READ_CHANNELS2-1:0] rd_fifo_tready2; 
logic [LP_NUM_READ_CHANNELS2-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_fifo_tdata2;

logic read_done3;
logic [LP_NUM_READ_CHANNELS3-1:0] rd_tvalid3;
logic [LP_NUM_READ_CHANNELS3-1:0] rd_tready_n3; 
logic [LP_NUM_READ_CHANNELS3-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_tdata3;
logic [LP_NUM_READ_CHANNELS3-1:0] ctrl_rd_fifo_prog_full3;
logic [LP_NUM_READ_CHANNELS3-1:0] rd_fifo_tvalid_n3;
logic [LP_NUM_READ_CHANNELS3-1:0] rd_fifo_tready3; 
logic [LP_NUM_READ_CHANNELS3-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_fifo_tdata3;

logic read_done4;
logic [LP_NUM_READ_CHANNELS4-1:0] rd_tvalid4;
logic [LP_NUM_READ_CHANNELS4-1:0] rd_tready_n4; 
logic [LP_NUM_READ_CHANNELS4-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_tdata4;
logic [LP_NUM_READ_CHANNELS4-1:0] ctrl_rd_fifo_prog_full4;
logic [LP_NUM_READ_CHANNELS4-1:0] rd_fifo_tvalid_n4;
logic [LP_NUM_READ_CHANNELS4-1:0] rd_fifo_tready4; 
logic [LP_NUM_READ_CHANNELS4-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_fifo_tdata4;

logic                                   adder_tvalid0;
logic                                 adder_tready_n0; 
logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]      adder_tdata0;
logic                               wr_fifo_tvalid_n0;
logic                                 wr_fifo_tready0; 
logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]    wr_fifo_tdata0;

//logic                                   adder_tvalid1;
logic                                 adder_tready_n1; 
//logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]      adder_tdata1;
logic                               wr_fifo_tvalid_n1;
logic                                 wr_fifo_tready1; 
logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]    wr_fifo_tdata1;

//logic                                   adder_tvalid3;
logic                                 adder_tready_n3; 
//logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]      adder_tdata3;
logic                               wr_fifo_tvalid_n3;
logic                                 wr_fifo_tready3; 
logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]    wr_fifo_tdata3;

//logic                                   adder_tvalid4;
logic                                 adder_tready_n4; 
//logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]      adder_tdata4;
logic                               wr_fifo_tvalid_n4;
logic                                 wr_fifo_tready4; 
logic [C_M_AXI_GMEM_DATA_WIDTH-1:0]    wr_fifo_tdata4;

logic                                   point0_valid;
logic                                   point0_ready;
logic [512-1:0]                         point0_data;
logic                                   point1_valid;
logic                                   point1_ready;
logic [512-1:0]                         point1_data;
logic                                   point2_valid;
logic                                   point2_ready;
logic [512-1:0]                         point2_data;
logic                                   point3_valid;
logic                                   point3_ready;
logic [512-1:0]                         point3_data;

logic                                   scalar_valid;
logic                                   scalar_ready;
logic [512-1:0]                         scalar_data;

logic                                   res0_valid;
logic                                   res0_ready;
logic                                   res0_full;
logic [512-1:0]                         res0_data;
logic                                   res1_valid;
logic                                   res1_ready;
logic                                   res1_full;
logic [512-1:0]                         res1_data;
logic                                   res2_valid;
logic                                   res2_ready;
logic                                   res2_full;
logic [512-1:0]                         res2_data;
logic                                   res3_valid;
logic                                   res3_ready;
logic                                   res3_full;
logic [512-1:0]                         res3_data;

logic [32-1:0]                          msm_size;       //length_r from s_control_axi
logic [32-1:0]                          point_length;   //msm_size*1152/4/512
logic [32-1:0]                          scalar_length;  //(s_bit = 16)msm_size*256/512
`KEEP logic [32-1:0]                          point_length_slr0_dly4;
`KEEP logic [32-1:0]                          point_length_slr1_dly4;
`KEEP logic [32-1:0]                          point_length_slr2_dly4;
`KEEP logic [32-1:0]                          point_length_slr3_dly4;
`KEEP logic [32-1:0]                          scalar_length_slr1_dly4;

logic [32-1:0]                          res_length;      //256*1536/512

logic [24-1:0]                          msm_num;       //equal to msm_size - 1
`KEEP logic [24-1:0]                          msm_num0;
`KEEP logic [24-1:0]                          msm_num2;
`KEEP logic [24-1:0]                          msm_num3;

logic                                   real_ap_done;//pulse
logic [4-1:0]                           ap_done_latch;

logic clk;
logic rstN;
logic [63:0] zprize_debug_cnt;

///////////////////////////////////////////////////////////////////////////////
// RTL Logic 
///////////////////////////////////////////////////////////////////////////////
assign clk  = ap_clk;
assign rstN = ap_rst_n;

`Flop(zprize_debug_cnt, 1, zprize_debug_cnt+1)

// Tie-off unused AXI protocol features
assign m_axi_gmem0_AWID     = {C_M_AXI_GMEM_ID_WIDTH0{1'b0}};
assign m_axi_gmem0_AWBURST  = 2'b01;
assign m_axi_gmem0_AWLOCK   = 2'b00;
assign m_axi_gmem0_AWCACHE  = 4'b0011;
assign m_axi_gmem0_AWPROT   = 3'b000;
assign m_axi_gmem0_AWQOS    = 4'b0000;
assign m_axi_gmem0_AWREGION = 4'b0000;
assign m_axi_gmem0_ARBURST  = 2'b01;
assign m_axi_gmem0_ARLOCK   = 2'b00;
assign m_axi_gmem0_ARCACHE  = 4'b0011;
assign m_axi_gmem0_ARPROT   = 3'b000;
assign m_axi_gmem0_ARQOS    = 4'b0000;
assign m_axi_gmem0_ARREGION = 4'b0000;

assign m_axi_gmem1_AWID     = {C_M_AXI_GMEM_ID_WIDTH1{1'b0}};
assign m_axi_gmem1_AWBURST  = 2'b01;
assign m_axi_gmem1_AWLOCK   = 2'b00;
assign m_axi_gmem1_AWCACHE  = 4'b0011;
assign m_axi_gmem1_AWPROT   = 3'b000;
assign m_axi_gmem1_AWQOS    = 4'b0000;
assign m_axi_gmem1_AWREGION = 4'b0000;
assign m_axi_gmem1_ARBURST  = 2'b01;
assign m_axi_gmem1_ARLOCK   = 2'b00;
assign m_axi_gmem1_ARCACHE  = 4'b0011;
assign m_axi_gmem1_ARPROT   = 3'b000;
assign m_axi_gmem1_ARQOS    = 4'b0000;
assign m_axi_gmem1_ARREGION = 4'b0000;

assign m_axi_gmem2_AWID     = {C_M_AXI_GMEM_ID_WIDTH2{1'b0}};
assign m_axi_gmem2_AWBURST  = 2'b01;
assign m_axi_gmem2_AWLOCK   = 2'b00;
assign m_axi_gmem2_AWCACHE  = 4'b0011;
assign m_axi_gmem2_AWPROT   = 3'b000;
assign m_axi_gmem2_AWQOS    = 4'b0000;
assign m_axi_gmem2_AWREGION = 4'b0000;
assign m_axi_gmem2_ARBURST  = 2'b01;
assign m_axi_gmem2_ARLOCK   = 2'b00;
assign m_axi_gmem2_ARCACHE  = 4'b0011;
assign m_axi_gmem2_ARPROT   = 3'b000;
assign m_axi_gmem2_ARQOS    = 4'b0000;
assign m_axi_gmem2_ARREGION = 4'b0000;

assign m_axi_gmem3_AWID     = {C_M_AXI_GMEM_ID_WIDTH3{1'b0}};
assign m_axi_gmem3_AWBURST  = 2'b01;
assign m_axi_gmem3_AWLOCK   = 2'b00;
assign m_axi_gmem3_AWCACHE  = 4'b0011;
assign m_axi_gmem3_AWPROT   = 3'b000;
assign m_axi_gmem3_AWQOS    = 4'b0000;
assign m_axi_gmem3_AWREGION = 4'b0000;
assign m_axi_gmem3_ARBURST  = 2'b01;
assign m_axi_gmem3_ARLOCK   = 2'b00;
assign m_axi_gmem3_ARCACHE  = 4'b0011;
assign m_axi_gmem3_ARPROT   = 3'b000;
assign m_axi_gmem3_ARQOS    = 4'b0000;
assign m_axi_gmem3_ARREGION = 4'b0000;

assign m_axi_gmem4_AWID     = {C_M_AXI_GMEM_ID_WIDTH4{1'b0}};
assign m_axi_gmem4_AWBURST  = 2'b01;
assign m_axi_gmem4_AWLOCK   = 2'b00;
assign m_axi_gmem4_AWCACHE  = 4'b0011;
assign m_axi_gmem4_AWPROT   = 3'b000;
assign m_axi_gmem4_AWQOS    = 4'b0000;
assign m_axi_gmem4_AWREGION = 4'b0000;
assign m_axi_gmem4_ARBURST  = 2'b01;
assign m_axi_gmem4_ARLOCK   = 2'b00;
assign m_axi_gmem4_ARCACHE  = 4'b0011;
assign m_axi_gmem4_ARPROT   = 3'b000;
assign m_axi_gmem4_ARQOS    = 4'b0000;
assign m_axi_gmem4_ARREGION = 4'b0000;

// Register and invert reset signal for better timing.
always @(posedge ap_clk) begin 
  areset <= ~ap_rst_n; 
end

// create pulse when ap_start transitions to 1
always @(posedge ap_clk) begin 
  begin 
    ap_start_r <= ap_start;
  end
end

assign ap_start_pulse = ap_start & ~ap_start_r;
// `define PIPE(latency,width,out,in) hyperPipe #(\
`KEEP_HIE `PIPE(4,1,ap_start_pulse_slr0_dly4,ap_start_pulse)

`KEEP_HIE `PIPE(4,1,ap_start_pulse_slr1_dly4,ap_start_pulse)

`KEEP_HIE `PIPE(4,1,ap_start_pulse_slr2_dly4,ap_start_pulse)

`KEEP_HIE `PIPE(4,1,ap_start_pulse_slr3_dly4,ap_start_pulse)

`KEEP_HIE `PIPE(4,32,point_length_slr0_dly4,point_length)

`KEEP_HIE `PIPE(4,32,point_length_slr1_dly4,point_length)

`KEEP_HIE `PIPE(4,32,point_length_slr2_dly4,point_length)

`KEEP_HIE `PIPE(4,32,point_length_slr3_dly4,point_length)

`KEEP_HIE `PIPE(4,32,scalar_length_slr1_dly4,scalar_length)



// ap_idle is asserted when done is asserted, it is de-asserted when ap_start_pulse 
// is asserted
always @(posedge ap_clk) begin 
  if (areset) begin 
    ap_idle <= 1'b1;
  end
  else begin 
    ap_idle <= real_ap_done        ? 1'b1 : 
               ap_start_pulse_slr1_dly4 ? 1'b0 : 
                                ap_idle;
  end
end

assign ap_ready = real_ap_done;

`KEEP_HIE `PIPE(4,1,ap_done0_slr0_dly4,ap_done0)

`KEEP_HIE `PIPE(4,1,ap_done3_slr2_dly4,ap_done3)
`KEEP_HIE `PIPE(4,1,ap_done4_slr3_dly4,ap_done4)


//all ap_done, then real_ap_done;
always @(posedge ap_clk) begin 
  if (areset) begin 
    ap_done_latch[0] <= 1'b0;
  end
  else if(real_ap_done)begin 
    ap_done_latch[0] <= 1'b0;
  end
  else if(ap_done0_slr0_dly4)begin 
    ap_done_latch[0] <= 1'b1;
  end
end


always @(posedge ap_clk) begin 
  if (areset) begin 
    ap_done_latch[2] <= 1'b0;
  end
  else if(real_ap_done)begin 
    ap_done_latch[2] <= 1'b0;
  end
  else if(ap_done3_slr2_dly4)begin 
    ap_done_latch[2] <= 1'b1;
  end
end

always @(posedge ap_clk) begin 
  if (areset) begin 
    ap_done_latch[3] <= 1'b0;
  end
  else if(real_ap_done)begin 
    ap_done_latch[3] <= 1'b0;
  end
  else if(ap_done4_slr3_dly4)begin 
    ap_done_latch[3] <= 1'b1;
  end
end

//assign real_ap_done = ap_done_latch == 4'b1111;
generate 
if (kernel_core_num == 4) begin : msm_4core_3_done 
assign real_ap_done =   ap_done_latch[0] == 1'b1 &&
                        ap_done_latch[2] == 1'b1 &&
                        ap_done_latch[3] == 1'b1 &&
                        1'b1;
end else begin : msm_1core_done  
assign real_ap_done =   ap_done_latch[0] == 1'b1 &&
//                        ap_done_latch[2] == 1'b1 &&
//                        ap_done_latch[3] == 1'b1 &&
                        1'b1;
end 
endgenerate
//---------------------		ctrl	----------------------------
// AXI4-Lite slave
krnl_vadd_rtl_control_s_axi #(
  .C_DATA_WIDTH      ( C_M_AXI_GMEM_DATA_WIDTH 	  ) ,
  .C_S_AXI_ADDR_WIDTH( C_S_AXI_CONTROL_ADDR_WIDTH ),
  .C_S_AXI_DATA_WIDTH( C_S_AXI_CONTROL_DATA_WIDTH )
) 
inst_krnl_vadd_control_s_axi (
  .AWVALID   ( s_axi_control_AWVALID         ) ,
  .AWREADY   ( s_axi_control_AWREADY         ) ,
  .AWADDR    ( s_axi_control_AWADDR          ) ,
  .WVALID    ( s_axi_control_WVALID          ) ,
  .WREADY    ( s_axi_control_WREADY          ) ,
  .WDATA     ( s_axi_control_WDATA           ) ,
  .WSTRB     ( s_axi_control_WSTRB           ) ,
  .ARVALID   ( s_axi_control_ARVALID         ) ,
  .ARREADY   ( s_axi_control_ARREADY         ) ,
  .ARADDR    ( s_axi_control_ARADDR          ) ,
  .RVALID    ( s_axi_control_RVALID          ) ,
  .RREADY    ( s_axi_control_RREADY          ) ,
  .RDATA     ( s_axi_control_RDATA           ) ,
  .RRESP     ( s_axi_control_RRESP           ) ,
  .BVALID    ( s_axi_control_BVALID          ) ,
  .BREADY    ( s_axi_control_BREADY          ) ,
  .BRESP     ( s_axi_control_BRESP           ) ,
  .ACLK      ( ap_clk                        ) ,
  .ARESET    ( areset                        ) ,
  .ACLK_EN   ( 1'b1                          ) ,
  .ap_start  ( ap_start                      ) ,
  .interrupt ( interrupt                     ) ,
  .ap_ready  ( ap_ready                      ) ,
  .ap_done   ( real_ap_done                       ) ,
  .ap_idle   ( ap_idle                       ) ,
  .in0     	 (   in0[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .in1_0     ( in1_0[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .in1_1     ( in1_1[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .in2     	 (   in2[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .in3     	 (   in3[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .out0      (  out0[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  //.out1      (  out1[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .out2      (  out2[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .out3      (  out3[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .length_r  ( length_r[0+:LP_LENGTH_WIDTH]  ),
  .in1_2     ( in1_2[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
  .scalar_flag  ( scalar_flag[0+:LP_LENGTH_WIDTH]  ) 
);



`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,in0_slr0_dly4,in0)
`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,out0_slr0_dly4,out0)

`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,in1_0_slr1_dly4,in1_0)
assign scalar_addr = |scalar_flag ? in1_2 : in1_1;
`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,in1_1_slr1_dly4,scalar_addr)

`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,in2_slr2_dly4,in2)
`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,out2_slr2_dly4,out2)

`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,in3_slr3_dly4,in3)
`KEEP_HIE `PIPE(4,C_M_AXI_GMEM_ADDR_WIDTH,out3_slr3_dly4,out3)

assign msm_size         = length_r;
assign point_length     = (msm_size>>4)*9;//msm_size*1152/4/512;
assign scalar_length    = (msm_size>>3);//msm_size*256/512/4; // per wave, so /4
assign res_length       = 32'd768;//256*1536/512;
assign msm_num          = msm_size - 1;
`KEEP_HIE `PIPE(5,24,msm_num0,msm_num)
`KEEP_HIE `PIPE(5,24,msm_num2,msm_num)
`KEEP_HIE `PIPE(5,24,msm_num3,msm_num)
//---------------------		read	----------------------------
// AXI4 Read Master
krnl_vadd_rtl_axi_read_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_ID_WIDTH         ( C_M_AXI_GMEM_ID_WIDTH0   ) ,
  .C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS0    ) ,
  .C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
  .C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
)
inst_axi_read_master0 ( 
  .aclk           ( ap_clk                 ) ,
  .areset         ( areset                 ) ,

  .ctrl_start     ( ap_start_pulse_slr0_dly4         ) , // input 
  .ctrl_done      ( read_done0              ) , // output
  .ctrl_offset    ( {in0_slr0_dly4}                   ) , // input 
  .ctrl_length    ( point_length_slr0_dly4            ) , // input 
  .ctrl_prog_full ( ctrl_rd_fifo_prog_full0 ) , // input 

  .arvalid        ( m_axi_gmem0_ARVALID     ) ,
  .arready        ( m_axi_gmem0_ARREADY     ) ,
  .araddr         ( m_axi_gmem0_ARADDR      ) ,
  .arid           ( m_axi_gmem0_ARID        ) ,
  .arlen          ( m_axi_gmem0_ARLEN       ) ,
  .arsize         ( m_axi_gmem0_ARSIZE      ) ,
  .rvalid         ( m_axi_gmem0_RVALID      ) ,
  .rready         ( m_axi_gmem0_RREADY      ) ,
  .rdata          ( m_axi_gmem0_RDATA       ) ,
  .rlast          ( m_axi_gmem0_RLAST       ) ,
  .rid            ( m_axi_gmem0_RID         ) ,
  .rresp          ( m_axi_gmem0_RRESP       ) ,

  .m_tvalid       ( rd_tvalid0              ) , // output
  .m_tready       ( ~rd_tready_n0           ) , // input 
  .m_tdata        ( rd_tdata0               )   // output
);

krnl_vadd_rtl_axi_read_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_ID_WIDTH         ( C_M_AXI_GMEM_ID_WIDTH1   ) ,
  .C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS1    ) ,
  .C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
  .C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
)
inst_axi_read_master1 ( 
  .aclk           ( ap_clk                 ) ,
  .areset         ( areset                 ) ,

  .ctrl_start     ( ap_start_pulse_slr1_dly4         ) , // input 
  .ctrl_done      ( read_done1              ) , // output
  .ctrl_offset    ( {in1_0_slr1_dly4}                  ) , // input 
  .ctrl_length    ( point_length_slr1_dly4            ) , // input 
  .ctrl_prog_full ( ctrl_rd_fifo_prog_full1 ) , // input 
                                                        
  .arvalid        ( m_axi_gmem1_ARVALID     ) ,
  .arready        ( m_axi_gmem1_ARREADY     ) ,
  .araddr         ( m_axi_gmem1_ARADDR      ) ,
  .arid           ( m_axi_gmem1_ARID        ) ,
  .arlen          ( m_axi_gmem1_ARLEN       ) ,
  .arsize         ( m_axi_gmem1_ARSIZE      ) ,
  .rvalid         ( m_axi_gmem1_RVALID      ) ,
  .rready         ( m_axi_gmem1_RREADY      ) ,
  .rdata          ( m_axi_gmem1_RDATA       ) ,
  .rlast          ( m_axi_gmem1_RLAST       ) ,
  .rid            ( m_axi_gmem1_RID         ) ,
  .rresp          ( m_axi_gmem1_RRESP       ) ,
                                                        
  .m_tvalid       ( rd_tvalid1              ) , // output
  .m_tready       ( ~rd_tready_n1          ) , // input 
  .m_tdata        ( rd_tdata1               )   // output
);

krnl_vadd_rtl_axi_read_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_ID_WIDTH         ( C_M_AXI_GMEM_ID_WIDTH2   ) ,
  .C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS2    ) ,
  .C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
  .C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
)
inst_axi_read_master2 ( 
  .aclk           ( ap_clk                 ) ,
  .areset         ( areset                 ) ,

  .ctrl_start     ( ap_start_pulse_slr1_dly4         ) , // input 
  .ctrl_done      ( read_done2              ) , // output
  .ctrl_offset    ( {in1_1_slr1_dly4}                  ) , // input 
  .ctrl_length    ( scalar_length_slr1_dly4           ) , // input 
  .ctrl_prog_full ( ctrl_rd_fifo_prog_full2 ) , // input 
                                                        
  .arvalid        ( m_axi_gmem2_ARVALID     ) ,
  .arready        ( m_axi_gmem2_ARREADY     ) ,
  .araddr         ( m_axi_gmem2_ARADDR      ) ,
  .arid           ( m_axi_gmem2_ARID        ) ,
  .arlen          ( m_axi_gmem2_ARLEN       ) ,
  .arsize         ( m_axi_gmem2_ARSIZE      ) ,
  .rvalid         ( m_axi_gmem2_RVALID      ) ,
  .rready         ( m_axi_gmem2_RREADY      ) ,
  .rdata          ( m_axi_gmem2_RDATA       ) ,
  .rlast          ( m_axi_gmem2_RLAST       ) ,
  .rid            ( m_axi_gmem2_RID         ) ,
  .rresp          ( m_axi_gmem2_RRESP       ) ,
                                                        
  .m_tvalid       ( rd_tvalid2              ) , // output
  .m_tready       ( ~rd_tready_n2          ) , // input 
  .m_tdata        ( rd_tdata2               )   // output
);

krnl_vadd_rtl_axi_read_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_ID_WIDTH         ( C_M_AXI_GMEM_ID_WIDTH3   ) ,
  .C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS3    ) ,
  .C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
  .C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
)
inst_axi_read_master3 ( 
  .aclk           ( ap_clk                 ) ,
  .areset         ( areset                 ) ,

  .ctrl_start     ( ap_start_pulse_slr2_dly4         ) , // input 
  .ctrl_done      ( read_done3              ) , // output
  .ctrl_offset    ( {in2_slr2_dly4}                  ) , // input 
  .ctrl_length    ( point_length_slr3_dly4            ) , // input 
  .ctrl_prog_full ( ctrl_rd_fifo_prog_full3 ) , // input 
                                                        
  .arvalid        ( m_axi_gmem3_ARVALID     ) ,
  .arready        ( m_axi_gmem3_ARREADY     ) ,
  .araddr         ( m_axi_gmem3_ARADDR      ) ,
  .arid           ( m_axi_gmem3_ARID        ) ,
  .arlen          ( m_axi_gmem3_ARLEN       ) ,
  .arsize         ( m_axi_gmem3_ARSIZE      ) ,
  .rvalid         ( m_axi_gmem3_RVALID      ) ,
  .rready         ( m_axi_gmem3_RREADY      ) ,
  .rdata          ( m_axi_gmem3_RDATA       ) ,
  .rlast          ( m_axi_gmem3_RLAST       ) ,
  .rid            ( m_axi_gmem3_RID         ) ,
  .rresp          ( m_axi_gmem3_RRESP       ) ,
                                                        
  .m_tvalid       ( rd_tvalid3              ) , // output
  .m_tready       ( ~rd_tready_n3          ) , // input 
  .m_tdata        ( rd_tdata3               )   // output
);

krnl_vadd_rtl_axi_read_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_ID_WIDTH         ( C_M_AXI_GMEM_ID_WIDTH4   ) ,
  .C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS4    ) ,
  .C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
  .C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
)
inst_axi_read_master4 ( 
  .aclk           ( ap_clk                 ) ,
  .areset         ( areset                 ) ,

  .ctrl_start     ( ap_start_pulse_slr3_dly4         ) , // input 
  .ctrl_done      ( read_done4              ) , // output
  .ctrl_offset    ( {in3_slr3_dly4}                  ) , // input 
  .ctrl_length    ( point_length_slr3_dly4            ) , // input 
  .ctrl_prog_full ( ctrl_rd_fifo_prog_full4 ) , // input 
                                                        
  .arvalid        ( m_axi_gmem4_ARVALID     ) ,
  .arready        ( m_axi_gmem4_ARREADY     ) ,
  .araddr         ( m_axi_gmem4_ARADDR      ) ,
  .arid           ( m_axi_gmem4_ARID        ) ,
  .arlen          ( m_axi_gmem4_ARLEN       ) ,
  .arsize         ( m_axi_gmem4_ARSIZE      ) ,
  .rvalid         ( m_axi_gmem4_RVALID      ) ,
  .rready         ( m_axi_gmem4_RREADY      ) ,
  .rdata          ( m_axi_gmem4_RDATA       ) ,
  .rlast          ( m_axi_gmem4_RLAST       ) ,
  .rid            ( m_axi_gmem4_RID         ) ,
  .rresp          ( m_axi_gmem4_RRESP       ) ,
                                                        
  .m_tvalid       ( rd_tvalid4              ) , // output
  .m_tready       ( ~rd_tready_n4          ) , // input 
  .m_tdata        ( rd_tdata4               )   // output
);

//---------------------		rfifo	----------------------------
// xpm_fifo_sync: Synchronous FIFO
// Xilinx Parameterized Macro, Version 2016.4
xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),        //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
  .PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_rd_xpm_fifo_sync0[LP_NUM_READ_CHANNELS0-1:0] (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( rd_tvalid0        ) ,
  .din           ( rd_tdata0         ) ,
  .full          ( rd_tready_n0      ) ,
  .prog_full     ( ctrl_rd_fifo_prog_full0) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( rd_fifo_tready0   ) ,
  .dout          ( rd_fifo_tdata0    ) ,
  .empty         ( rd_fifo_tvalid_n0 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),        //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
  .PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_rd_xpm_fifo_sync1[LP_NUM_READ_CHANNELS1-1:0] (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( rd_tvalid1        ) ,
  .din           ( rd_tdata1         ) ,
  .full          ( rd_tready_n1      ) ,
  .prog_full     ( ctrl_rd_fifo_prog_full1) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( rd_fifo_tready1   ) ,
  .dout          ( rd_fifo_tdata1    ) ,
  .empty         ( rd_fifo_tvalid_n1 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),        //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
  .PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_rd_xpm_fifo_sync2[LP_NUM_READ_CHANNELS2-1:0] (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( rd_tvalid2        ) ,
  .din           ( rd_tdata2         ) ,
  .full          ( rd_tready_n2      ) ,
  .prog_full     ( ctrl_rd_fifo_prog_full2) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( rd_fifo_tready2   ) ,
  .dout          ( rd_fifo_tdata2    ) ,
  .empty         ( rd_fifo_tvalid_n2 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),        //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
  .PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_rd_xpm_fifo_sync3[LP_NUM_READ_CHANNELS3-1:0] (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( rd_tvalid3        ) ,
  .din           ( rd_tdata3         ) ,
  .full          ( rd_tready_n3      ) ,
  .prog_full     ( ctrl_rd_fifo_prog_full3) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( rd_fifo_tready3   ) ,
  .dout          ( rd_fifo_tdata3    ) ,
  .empty         ( rd_fifo_tvalid_n3 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),        //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
  .PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_rd_xpm_fifo_sync4[LP_NUM_READ_CHANNELS4-1:0] (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( rd_tvalid4        ) ,
  .din           ( rd_tdata4         ) ,
  .full          ( rd_tready_n4      ) ,
  .prog_full     ( ctrl_rd_fifo_prog_full4) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( rd_fifo_tready4   ) ,
  .dout          ( rd_fifo_tdata4    ) ,
  .empty         ( rd_fifo_tvalid_n4 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

//---------------------		func	----------------------------
//assign rd_fifo_tready1 = ~rd_fifo_tvalid_n1;
//assign rd_fifo_tready2 = ~rd_fifo_tvalid_n2;
//assign rd_fifo_tready3 = ~rd_fifo_tvalid_n3;
//assign rd_fifo_tready4 = ~rd_fifo_tvalid_n4;

assign point0_valid = ~rd_fifo_tvalid_n0;
assign point1_valid = ~rd_fifo_tvalid_n1;
assign scalar_valid = ~rd_fifo_tvalid_n2;
assign point2_valid = ~rd_fifo_tvalid_n3;
assign point3_valid = ~rd_fifo_tvalid_n4;

assign rd_fifo_tready0 = point0_ready;
assign rd_fifo_tready1 = point1_ready;
assign rd_fifo_tready2 = scalar_ready;
assign rd_fifo_tready3 = point2_ready;
assign rd_fifo_tready4 = point3_ready;

assign point0_data  = rd_fifo_tdata0;
assign point1_data  = rd_fifo_tdata1;
assign scalar_data  = rd_fifo_tdata2;
assign point2_data  = rd_fifo_tdata3;
assign point3_data  = rd_fifo_tdata4;

// msm

zprize_msm msm(
    .pointDdr0_valid (point0_valid),
    .pointDdr0_ready (point0_ready),
    .pointDdr0_data  (point0_data),
    .pointDdr1_valid (point1_valid),
    .pointDdr1_ready (point1_ready),
    .pointDdr1_data  (point1_data),
    .pointDdr2_valid (point2_valid),
    .pointDdr2_ready (point2_ready),
    .pointDdr2_data  (point2_data),
    .pointDdr3_valid (point3_valid),
    .pointDdr3_ready (point3_ready),
    .pointDdr3_data  (point3_data),
    .scalarDdr_valid (scalar_valid),
    .scalarDdr_ready (scalar_ready),
    .scalarDdr_data  (scalar_data),
    .msm_num0        (msm_num0),
    .msm_num2        (msm_num2),
    .msm_num3        (msm_num3),
    .returnDdr0_valid  (res0_valid),
    .returnDdr0_ready  (res0_ready),
    .returnDdr0_data   (res0_data),
    .returnDdr1_valid  (res1_valid),
    .returnDdr1_ready  (res1_ready),
    .returnDdr1_data   (res1_data),
    .returnDdr2_valid  (res2_valid),
    .returnDdr2_ready  (res2_ready),
    .returnDdr2_data   (res2_data),
    .returnDdr3_valid  (res3_valid),
    .returnDdr3_ready  (res3_ready),
    .returnDdr3_data   (res3_data),
    .ap_clk           (ap_clk),
    .ap_rst_n         (ap_rst_n)
);

assign res0_ready = !res0_full;
//assign res1_ready = !res1_full;
assign res1_ready = 1;
assign res2_ready = !res2_full;
assign res3_ready = !res3_full;

//---------------------		wfifo	----------------------------
// xpm_fifo_sync: Synchronous FIFO
// Xilinx Parameterized Macro, Version 2016.4
xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_WR_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, Not used
  .PROG_FULL_THRESH          (10),               //positive integer, Not used 
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_wr_xpm_fifo_sync0 (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( res0_valid       ) ,
  .din           ( res0_data        ) ,
  .full          ( res0_full        ) ,
  .prog_full     (                  ) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( wr_fifo_tready0   ) ,
  .dout          ( wr_fifo_tdata0    ) ,
  .empty         ( wr_fifo_tvalid_n0 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_WR_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, Not used
  .PROG_FULL_THRESH          (10),               //positive integer, Not used 
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_wr_xpm_fifo_sync3 (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( res2_valid       ) ,
  .din           ( res2_data        ) ,
  .full          ( res2_full        ) ,
  .prog_full     (                  ) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( wr_fifo_tready3   ) ,
  .dout          ( wr_fifo_tdata3    ) ,
  .empty         ( wr_fifo_tvalid_n3 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);

xpm_fifo_sync # (
  .FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
  .ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
  .FIFO_WRITE_DEPTH          (LP_WR_FIFO_DEPTH),   //positive integer
  .WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .WR_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, Not used
  .PROG_FULL_THRESH          (10),               //positive integer, Not used 
  .FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
  .READ_MODE                 ("fwft"),            //string; "std" or "fwft";
  .FIFO_READ_LATENCY         (1),                //positive integer;
  .READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
  .RD_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, not used
  .PROG_EMPTY_THRESH         (10),               //positive integer, not used 
  .DOUT_RESET_VALUE          ("0"),              //string, don't care
  .WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

) inst_wr_xpm_fifo_sync4 (
  .sleep         ( 1'b0             ) ,
  .rst           ( areset           ) ,
  .wr_clk        ( ap_clk           ) ,
  .wr_en         ( res3_valid       ) ,
  .din           ( res3_data        ) ,
  .full          ( res3_full        ) ,
  .prog_full     (                  ) ,
  .wr_data_count (                  ) ,
  .overflow      (                  ) ,
  .wr_rst_busy   (                  ) ,
  .rd_en         ( wr_fifo_tready4   ) ,
  .dout          ( wr_fifo_tdata4    ) ,
  .empty         ( wr_fifo_tvalid_n4 ) ,
  .prog_empty    (                  ) ,
  .rd_data_count (                  ) ,
  .underflow     (                  ) ,
  .rd_rst_busy   (                  ) ,
  .injectsbiterr ( 1'b0             ) ,
  .injectdbiterr ( 1'b0             ) ,
  .sbiterr       (                  ) ,
  .dbiterr       (                  ) 

);


//---------------------		write	----------------------------
// AXI4 Write Master
krnl_vadd_rtl_axi_write_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_MAX_LENGTH_WIDTH ( LP_LENGTH_WIDTH     ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) 
)
inst_axi_write_master0 ( 
  .aclk        ( ap_clk             ) ,
  .areset      ( areset             ) ,

  .ctrl_start  ( ap_start_pulse_slr0_dly4     ) , // input 
  .ctrl_offset ( out0_slr0_dly4               ) , // input 
  .ctrl_length ( res_length         ) , // input 
  .ctrl_done   ( ap_done0            ) , // output

  .awvalid     ( m_axi_gmem0_AWVALID ) ,
  .awready     ( m_axi_gmem0_AWREADY ) ,
  .awaddr      ( m_axi_gmem0_AWADDR  ) ,
  .awlen       ( m_axi_gmem0_AWLEN   ) ,
  .awsize      ( m_axi_gmem0_AWSIZE  ) ,

  .s_tvalid    ( ~wr_fifo_tvalid_n0   ) ,// input 
  .s_tready    ( wr_fifo_tready0     ) , // output 
  .s_tdata     ( wr_fifo_tdata0      ) , // input

  .wvalid      ( m_axi_gmem0_WVALID  ) ,
  .wready      ( m_axi_gmem0_WREADY  ) ,
  .wdata       ( m_axi_gmem0_WDATA   ) ,
  .wstrb       ( m_axi_gmem0_WSTRB   ) ,
  .wlast       ( m_axi_gmem0_WLAST   ) ,

  .bvalid      ( m_axi_gmem0_BVALID  ) ,
  .bready      ( m_axi_gmem0_BREADY  ) ,
  .bresp       ( m_axi_gmem0_BRESP   ) 
);


  assign m_axi_gmem1_AWVALID = 0; 
  assign m_axi_gmem1_AWADDR  = 0; 
  assign m_axi_gmem1_AWLEN   = 0; 
  assign m_axi_gmem1_AWSIZE  = 0; 

  assign m_axi_gmem1_WVALID  = 0;
  assign m_axi_gmem1_WDATA   = 0;
  assign m_axi_gmem1_WSTRB   = 0;
  assign m_axi_gmem1_WLAST   = 0;
  assign m_axi_gmem1_BREADY  = 1;

  assign m_axi_gmem2_AWVALID = 0; 
  assign m_axi_gmem2_AWADDR  = 0; 
  assign m_axi_gmem2_AWLEN   = 0; 
  assign m_axi_gmem2_AWSIZE  = 0; 
  
  assign m_axi_gmem2_WVALID  = 0;
  assign m_axi_gmem2_WDATA   = 0;
  assign m_axi_gmem2_WSTRB   = 0;
  assign m_axi_gmem2_WLAST   = 0;
  assign m_axi_gmem2_BREADY  = 1;





krnl_vadd_rtl_axi_write_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_MAX_LENGTH_WIDTH ( LP_LENGTH_WIDTH     ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) 
)
inst_axi_write_master3 ( 
  .aclk        ( ap_clk             ) ,
  .areset      ( areset             ) ,

  .ctrl_start  ( ap_start_pulse_slr2_dly4     ) , // input 
  .ctrl_offset ( out2_slr2_dly4               ) , // input 
  .ctrl_length ( res_length         ) , // input 
  .ctrl_done   ( ap_done3            ) , // output

  .awvalid     ( m_axi_gmem3_AWVALID ) ,
  .awready     ( m_axi_gmem3_AWREADY ) ,
  .awaddr      ( m_axi_gmem3_AWADDR  ) ,
  .awlen       ( m_axi_gmem3_AWLEN   ) ,
  .awsize      ( m_axi_gmem3_AWSIZE  ) ,

  .s_tvalid    ( ~wr_fifo_tvalid_n3   ) ,// input 
  .s_tready    ( wr_fifo_tready3     ) , // output 
  .s_tdata     ( wr_fifo_tdata3      ) , // input

  .wvalid      ( m_axi_gmem3_WVALID  ) ,
  .wready      ( m_axi_gmem3_WREADY  ) ,
  .wdata       ( m_axi_gmem3_WDATA   ) ,
  .wstrb       ( m_axi_gmem3_WSTRB   ) ,
  .wlast       ( m_axi_gmem3_WLAST   ) ,

  .bvalid      ( m_axi_gmem3_BVALID  ) ,
  .bready      ( m_axi_gmem3_BREADY  ) ,
  .bresp       ( m_axi_gmem3_BRESP   ) 
);

krnl_vadd_rtl_axi_write_master #( 
  .C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
  .C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
  .C_MAX_LENGTH_WIDTH ( LP_LENGTH_WIDTH     ) ,
  .C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
  .C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) 
)
inst_axi_write_master4 ( 
  .aclk        ( ap_clk             ) ,
  .areset      ( areset             ) ,

  .ctrl_start  ( ap_start_pulse_slr3_dly4     ) , // input 
  .ctrl_offset ( out3_slr3_dly4               ) , // input 
  .ctrl_length ( res_length         ) , // input 
  .ctrl_done   ( ap_done4            ) , // output

  .awvalid     ( m_axi_gmem4_AWVALID ) ,
  .awready     ( m_axi_gmem4_AWREADY ) ,
  .awaddr      ( m_axi_gmem4_AWADDR  ) ,
  .awlen       ( m_axi_gmem4_AWLEN   ) ,
  .awsize      ( m_axi_gmem4_AWSIZE  ) ,

  .s_tvalid    ( ~wr_fifo_tvalid_n4   ) ,// input 
  .s_tready    ( wr_fifo_tready4     ) , // output 
  .s_tdata     ( wr_fifo_tdata4      ) , // input

  .wvalid      ( m_axi_gmem4_WVALID  ) ,
  .wready      ( m_axi_gmem4_WREADY  ) ,
  .wdata       ( m_axi_gmem4_WDATA   ) ,
  .wstrb       ( m_axi_gmem4_WSTRB   ) ,
  .wlast       ( m_axi_gmem4_WLAST   ) ,

  .bvalid      ( m_axi_gmem4_BVALID  ) ,
  .bready      ( m_axi_gmem4_BREADY  ) ,
  .bresp       ( m_axi_gmem4_BRESP   ) 
);


`ifdef ZPRIZE_DEBUG
`endif

endmodule : krnl_vadd_rtl_int

`default_nettype wire
