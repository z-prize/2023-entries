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

`timescale 1ns/1ps
module krnl_vadd_rtl_control_s_axi
#(parameter
	C_DATA_WIDTH	   = 512, // use 512bits data_width
    C_S_AXI_ADDR_WIDTH = 6,
    C_S_AXI_DATA_WIDTH = 32
)(
    // axi4 lite slave signals
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire                          interrupt,
    // user signals
    output wire                          ap_start,
    input  wire                          ap_done,
    input  wire                          ap_ready,
    input  wire                          ap_idle,
    output wire [63:0]                   in0  ,
    output wire [63:0]                   in1_0,
    output wire [63:0]                   in1_1,
    output wire [63:0]                   in2  ,
    output wire [63:0]                   in3  ,
    output wire [63:0]                   out0 ,
    //output wire [63:0]                   out1 ,
    output wire [63:0]                   out2 ,
    output wire [63:0]                   out3 ,
    output wire [31:0]                   length_r,
    output wire [31:0]                   scalar_flag,
    output wire [63:0]                   in1_2
);
//------------------------Address Info-------------------
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of in0
//        bit 31~0 - in0[31:0] (Read/Write)
// 0x14 : Data signal of in0
//        bit 31~0 - in0[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of in1_0
//        bit 31~0 - in1_0[31:0] (Read/Write)
// 0x20 : Data signal of in1_0
//        bit 31~0 - in1_0[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of in1_1
//        bit 31~0 - in1_1[31:0] (Read/Write)
// 0x2c : Data signal of in1_1
//        bit 31~0 - in1_1[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of length_r
//        bit 31~0 - length_r[31:0] (Read/Write)
// 0x38 : reserved
// 0x3c : Data signal of in2
//        bit 31~0 - in2[31:0] (Read/Write)
// 0x40 : Data signal of in2
//        bit 31~0 - in2[63:32] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of in3
//        bit 31~0 - in3[31:0] (Read/Write)
// 0x4c : Data signal of in3
//        bit 31~0 - in3[63:32] (Read/Write)
// 0x50 : reserved
// 0x54 : Data signal of out0
//        bit 31~0 - out0[31:0] (Read/Write)
// 0x58 : Data signal of out0
//        bit 31~0 - out0[63:32] (Read/Write)
// 0x5c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL         = 8'h00,
    ADDR_GIE             = 8'h04,
    ADDR_IER             = 8'h08,
    ADDR_ISR             = 8'h0c,

    ADDR_IN0_DATA_0      = 8'h10,
    ADDR_IN0_DATA_1      = 8'h14,
    ADDR_IN0_CTRL        = 8'h18,

    ADDR_IN1_0_DATA_0    = 8'h1c,
    ADDR_IN1_0_DATA_1    = 8'h20,
    ADDR_IN1_0_CTRL      = 8'h24,

    ADDR_IN1_1_DATA_0    = 8'h28,
    ADDR_IN1_1_DATA_1    = 8'h2c,
    ADDR_IN1_1_CTRL      = 8'h30,

    ADDR_IN2_DATA_0      = 8'h34,
    ADDR_IN2_DATA_1      = 8'h38,
    ADDR_IN2_CTRL        = 8'h3c,

    ADDR_IN3_DATA_0      = 8'h40,
    ADDR_IN3_DATA_1      = 8'h44,
    ADDR_IN3_CTRL        = 8'h48,

    ADDR_OUT0_DATA_0     = 8'h4c,
    ADDR_OUT0_DATA_1     = 8'h50,
    ADDR_OUT0_CTRL       = 8'h54,

    // ADDR_OUT1_DATA_0     = 8'h58,
    // ADDR_OUT1_DATA_1     = 8'h5c,
    // ADDR_OUT1_CTRL       = 8'h60,

    ADDR_OUT2_DATA_0     = 8'h64,
    ADDR_OUT2_DATA_1     = 8'h68,
    ADDR_OUT2_CTRL       = 8'h6c,

    ADDR_OUT3_DATA_0     = 8'h70,
    ADDR_OUT3_DATA_1     = 8'h74,
    ADDR_OUT3_CTRL       = 8'h78,

    ADDR_LENGTH_R_DATA_0 = 8'h7c,
    ADDR_LENGTH_R_CTRL   = 8'h80,

    ADDR_IN1_2_DATA_0     = 8'h88,
    ADDR_IN1_2_DATA_1     = 8'h8c,
    ADDR_IN1_2_CTRL       = 8'h90,

    ADDR_SCALAR_FLAG_DATA = 8'h94,
    ADDR_SCALAR_FLAG_CTRL = 8'h98,
// add for zprize

    WRIDLE               = 2'd0,
    WRDATA               = 2'd1,
    WRRESP               = 2'd2,
    RDIDLE               = 2'd0,
    RDDATA               = 2'd1,
    ADDR_BITS         = 8;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRIDLE;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [31:0]                   wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDIDLE;
    reg  [1:0]                    rnext;
    reg  [31:0]                   rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    wire                          int_ap_idle;
    wire                          int_ap_ready;
    reg                           int_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           int_gie = 2'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [63:0]                   int_in0 	= 64'b0;
    reg  [63:0]                   int_in1_0 = 64'b0;
    reg  [63:0]                   int_in1_1 = 64'b0;
    reg  [63:0]                   int_in1_2 = 64'b0;
    reg  [63:0]                   int_in2 	= 64'b0;
    reg  [63:0]                   int_in3 	= 64'b0;
    reg  [63:0]                   int_out0 	= 64'b0;
    //reg  [63:0]                   int_out1 	= 64'b0;
    reg  [63:0]                   int_out2 	= 64'b0;
    reg  [63:0]                   int_out3 	= 64'b0;
    reg  [31:0]                   int_length_r = 32'b0;
    reg  [31:0]                   int_scalar_flag = 32'b0;

//------------------------Instantiation------------------

//------------------------AXI write fsm------------------
assign AWREADY = (~ARESET) & (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;


// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRIDLE;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= AWADDR[ADDR_BITS-1:0];
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (~ARESET) && (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDIDLE;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 1'b0;
            case (raddr)
                ADDR_AP_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
                ADDR_IN0_DATA_0: begin
                    rdata <= int_in0[31:0];
                end
                ADDR_IN0_DATA_1: begin
                    rdata <= int_in0[63:32];
                end
                ADDR_IN1_0_DATA_0: begin
                    rdata <= int_in1_0[31:0];
                end
                ADDR_IN1_0_DATA_1: begin
                    rdata <= int_in1_0[63:32];
                end
                ADDR_IN1_1_DATA_0: begin
                    rdata <= int_in1_1[31:0];
                end
                ADDR_IN1_1_DATA_1: begin
                    rdata <= int_in1_1[63:32];
                end
                ADDR_LENGTH_R_DATA_0: begin
                    rdata <= int_length_r[31:0];
                end
                ADDR_IN2_DATA_0: begin
                    rdata <= int_in2[31:0];
                end
                ADDR_IN2_DATA_1: begin
                    rdata <= int_in2[63:32];
                end
                ADDR_IN3_DATA_0: begin
                    rdata <= int_in3[31:0];
                end
                ADDR_IN3_DATA_1: begin
                    rdata <= int_in3[63:32];
                end
                ADDR_OUT0_DATA_0: begin
                    rdata <= int_out0[31:0];
                end
                ADDR_OUT0_DATA_1: begin
                    rdata <= int_out0[63:32];
                end
                //ADDR_OUT1_DATA_0: begin
                //    rdata <= int_out1[31:0];
                //end
                //ADDR_OUT1_DATA_1: begin
                //    rdata <= int_out1[63:32];
                //end
                ADDR_OUT2_DATA_0: begin
                    rdata <= int_out2[31:0];
                end
                ADDR_OUT2_DATA_1: begin
                    rdata <= int_out2[63:32];
                end
                ADDR_OUT3_DATA_0: begin
                    rdata <= int_out3[31:0];
                end
                ADDR_OUT3_DATA_1: begin
                    rdata <= int_out3[63:32];
                end
                ADDR_IN1_2_DATA_0: begin
                    rdata <= int_in1_2[31:0];
                end
                ADDR_IN1_2_DATA_1: begin
                    rdata <= int_in1_2[63:32];
                end
                ADDR_SCALAR_FLAG_DATA: begin
                    rdata <= int_scalar_flag[31:0];
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign interrupt    = int_gie & (|int_isr);
assign ap_start     = int_ap_start;
assign int_ap_idle  = ap_idle;
assign int_ap_ready = ap_ready;
assign in0         	= int_in0;
assign in1_0        = int_in1_0;
assign in1_1        = int_in1_1;
assign in1_2        = int_in1_2;
assign in2 			= int_in2;
assign in3 			= int_in3;
//assign length_r     = int_length_r * 32 / C_DATA_WIDTH; // host use 32bits width to sum lengeth
assign length_r     = int_length_r; //msm_size
assign scalar_flag  = int_scalar_flag; //msm_size
assign out0 		= int_out0;
//assign out1 		= int_out1;
assign out2 		= int_out2;
assign out3 		= int_out3;
// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (int_ap_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (ap_done)
            int_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_ap_done <= 1'b0; // clear on read
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <=  WDATA[7];
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & ap_done)
            int_isr[0] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[0] <= int_isr[0] ^ WDATA[0]; // toggle on write
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & ap_ready)
            int_isr[1] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[1] <= int_isr[1] ^ WDATA[1]; // toggle on write
    end
end

// int_in0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_in0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN0_DATA_0)
            int_in0[31:0] <= (WDATA[31:0] & wmask) | (int_in0[31:0] & ~wmask);
    end
end

// int_in0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_in0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN0_DATA_1)
            int_in0[63:32] <= (WDATA[31:0] & wmask) | (int_in0[63:32] & ~wmask);
    end
end

// int_in1_0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_in1_0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN1_0_DATA_0)
            int_in1_0[31:0] <= (WDATA[31:0] & wmask) | (int_in1_0[31:0] & ~wmask);
    end
end

// int_in1_0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_in1_0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN1_0_DATA_1)
            int_in1_0[63:32] <= (WDATA[31:0] & wmask) | (int_in1_0[63:32] & ~wmask);
    end
end

// int_in1_1[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_in1_1[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN1_1_DATA_0)
            int_in1_1[31:0] <= (WDATA[31:0] & wmask) | (int_in1_1[31:0] & ~wmask);
    end
end

// int_in1_1[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_in1_1[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN1_1_DATA_1)
            int_in1_1[63:32] <= (WDATA[31:0] & wmask) | (int_in1_1[63:32] & ~wmask);
    end
end

// int_length_r[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_length_r[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_LENGTH_R_DATA_0)
            int_length_r[31:0] <= (WDATA[31:0] & wmask) | (int_length_r[31:0] & ~wmask);
    end
end

// int_in2[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_in2[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN2_DATA_0)
            int_in2[31:0] <= (WDATA[31:0] & wmask) | (int_in2[31:0] & ~wmask);
    end
end

// int_in2[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_in2[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN2_DATA_1)
            int_in2[63:32] <= (WDATA[31:0] & wmask) | (int_in2[63:32] & ~wmask);
    end
end

// int_in3[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_in3[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN3_DATA_0)
            int_in3[31:0] <= (WDATA[31:0] & wmask) | (int_in3[31:0] & ~wmask);
    end
end

// int_in3[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_in3[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN3_DATA_1)
            int_in3[63:32] <= (WDATA[31:0] & wmask) | (int_in3[63:32] & ~wmask);
    end
end

// int_out0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_out0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_OUT0_DATA_0)
            int_out0[31:0] <= (WDATA[31:0] & wmask) | (int_out0[31:0] & ~wmask);
    end
end

// int_out0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_out0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_OUT0_DATA_1)
            int_out0[63:32] <= (WDATA[31:0] & wmask) | (int_out0[63:32] & ~wmask);
    end
end

//// int_out1[31:0]
//always @(posedge ACLK) begin
//    if (ARESET)
//        int_out1[31:0] <= 0;
//    else if (ACLK_EN) begin
//        if (w_hs && waddr == ADDR_OUT1_DATA_0)
//            int_out1[31:0] <= (WDATA[31:0] & wmask) | (int_out1[31:0] & ~wmask);
//    end
//end
//
//// int_out1[63:32]
//always @(posedge ACLK) begin
//    if (ARESET)
//        int_out1[63:32] <= 0;
//    else if (ACLK_EN) begin
//        if (w_hs && waddr == ADDR_OUT1_DATA_1)
//            int_out1[63:32] <= (WDATA[31:0] & wmask) | (int_out1[63:32] & ~wmask);
//    end
//end

// int_out2[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_out2[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_OUT2_DATA_0)
            int_out2[31:0] <= (WDATA[31:0] & wmask) | (int_out2[31:0] & ~wmask);
    end
end

// int_out2[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_out2[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_OUT2_DATA_1)
            int_out2[63:32] <= (WDATA[31:0] & wmask) | (int_out2[63:32] & ~wmask);
    end
end

// int_out3[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_out3[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_OUT3_DATA_0)
            int_out3[31:0] <= (WDATA[31:0] & wmask) | (int_out3[31:0] & ~wmask);
    end
end

// int_out3[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_out3[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_OUT3_DATA_1)
            int_out3[63:32] <= (WDATA[31:0] & wmask) | (int_out3[63:32] & ~wmask);
    end
end



// int_in1_2[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_in1_2[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN1_2_DATA_0)
            int_in1_2[31:0] <= (WDATA[31:0] & wmask) | (int_in1_2[31:0] & ~wmask);
    end
end

// int_in1_2[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_in1_2[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IN1_2_DATA_1)
            int_in1_2[63:32] <= (WDATA[31:0] & wmask) | (int_in1_2[63:32] & ~wmask);
    end
end

// int_length_r[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_scalar_flag[31:0] <= 0;
    else if (ap_done)
        int_scalar_flag[0] <= ~int_scalar_flag[0];
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SCALAR_FLAG_DATA)
            int_scalar_flag[31:0] <= (WDATA[31:0] & wmask) | (int_scalar_flag[31:0] & ~wmask);
    end
end
//------------------------Memory logic-------------------

endmodule
