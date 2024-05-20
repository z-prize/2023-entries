`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

module zprize_msm import zprize_param::* ; (
//point in        
    input                       pointDdr0_valid,
    output                      pointDdr0_ready,
    input   [512-1:0]       pointDdr0_data,
    input                       pointDdr1_valid,
    output                      pointDdr1_ready,
    input   [512-1:0]       pointDdr1_data,
    input                       pointDdr2_valid,
    output                      pointDdr2_ready,
    input   [512-1:0]       pointDdr2_data,
    input                       pointDdr3_valid,
    output                      pointDdr3_ready,
    input   [512-1:0]       pointDdr3_data,
//scalar in
    input                       scalarDdr_valid,
    output                      scalarDdr_ready,
    input   [512-1:0]       scalarDdr_data,
//msm_num
    input   [`LOG2((1<<24))-1:0]       msm_num0, // msm_numk
    input   [`LOG2((1<<24))-1:0]       msm_num2,
    input   [`LOG2((1<<24))-1:0]       msm_num3,
//return
///for k, 0 < 4
    output                      returnDdr0_valid,
    input                       returnDdr0_ready,
    output  [512-1:0]       returnDdr0_data,
    output                      returnDdr1_valid,
    input                       returnDdr1_ready,
    output  [512-1:0]       returnDdr1_data,
    output                      returnDdr2_valid,
    input                       returnDdr2_ready,
    output  [512-1:0]       returnDdr2_data,
    output                      returnDdr3_valid,
    input                       returnDdr3_ready,
    output  [512-1:0]       returnDdr3_data,
//global
    input                       ap_clk,
    input                       ap_rst_n            
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    logic                       clk;
    logic                       rstN;

    logic   [`LOG2(4)-1:0] switch;
    logic                       switch_ready;

    logic                       pointPipeValid;
    logic   [1152-1:0]     pointPipeData;
//k is index of SLR
    logic                       pointPipeValid0;
    logic   [1152-1:0]     pointPipeData0;
    logic                       real_pointPipeValid0;
    logic   [1152-1:0]     real_pointPipeData0;
    logic                       pointPipeValid1;
    logic   [1152-1:0]     pointPipeData1;
    logic                       real_pointPipeValid1;
    logic   [1152-1:0]     real_pointPipeData1;
    logic                       pointPipeValid2;
    logic   [1152-1:0]     pointPipeData2;
    logic                       real_pointPipeValid2;
    logic   [1152-1:0]     real_pointPipeData2;
    logic                       pointPipeValid3;
    logic   [1152-1:0]     pointPipeData3;
    logic                       real_pointPipeValid3;
    logic   [1152-1:0]     real_pointPipeData3;
    logic                       pointPipeValid1a;
    logic   [1152-1:0]     pointPipeData1a;
    logic                       pointPipeValid1b;
    logic   [1152-1:0]     pointPipeData1b;

    logic                       pointPipeValid2a;
    logic   [1152-1:0]     pointPipeData2a;

    logic                       scalarCreditOk0;
    logic                       pointCreditOk0;
    logic                       raw_scalarCreditOk0;
    logic                       raw_pointCreditOk0;
    logic                       scalarCreditOk0_0;
    logic                       pointCreditOk0_0;
    logic                       pointCreditOk0_0a;      //add one stage for timing
    logic                       pointRamBufferFifo0_CreditOk_0;
    logic                       scalarCreditOk0_1;
    logic                       pointCreditOk0_1;
    logic                       pointCreditOk0_1a;      //add one stage for timing
    logic                       pointRamBufferFifo1_CreditOk_0;
    logic                       scalarCreditOk0_2;
    logic                       pointCreditOk0_2;
    logic                       pointCreditOk0_2a;      //add one stage for timing
    logic                       pointRamBufferFifo2_CreditOk_0;
    logic                       scalarCreditOk0_3;
    logic                       pointCreditOk0_3;
    logic                       pointCreditOk0_3a;      //add one stage for timing
    logic                       pointRamBufferFifo3_CreditOk_0;
    logic                       scalarCreditOk1;
    logic                       pointCreditOk1;
    logic                       raw_scalarCreditOk1;
    logic                       raw_pointCreditOk1;
    logic                       scalarCreditOk1_0;
    logic                       pointCreditOk1_0;
    logic                       pointCreditOk1_0a;      //add one stage for timing
    logic                       pointRamBufferFifo0_CreditOk_1;
    logic                       scalarCreditOk1_1;
    logic                       pointCreditOk1_1;
    logic                       pointCreditOk1_1a;      //add one stage for timing
    logic                       pointRamBufferFifo1_CreditOk_1;
    logic                       scalarCreditOk1_2;
    logic                       pointCreditOk1_2;
    logic                       pointCreditOk1_2a;      //add one stage for timing
    logic                       pointRamBufferFifo2_CreditOk_1;
    logic                       scalarCreditOk1_3;
    logic                       pointCreditOk1_3;
    logic                       pointCreditOk1_3a;      //add one stage for timing
    logic                       pointRamBufferFifo3_CreditOk_1;
    logic                       scalarCreditOk2;
    logic                       pointCreditOk2;
    logic                       raw_scalarCreditOk2;
    logic                       raw_pointCreditOk2;
    logic                       scalarCreditOk2_0;
    logic                       pointCreditOk2_0;
    logic                       pointCreditOk2_0a;      //add one stage for timing
    logic                       pointRamBufferFifo0_CreditOk_2;
    logic                       scalarCreditOk2_1;
    logic                       pointCreditOk2_1;
    logic                       pointCreditOk2_1a;      //add one stage for timing
    logic                       pointRamBufferFifo1_CreditOk_2;
    logic                       scalarCreditOk2_2;
    logic                       pointCreditOk2_2;
    logic                       pointCreditOk2_2a;      //add one stage for timing
    logic                       pointRamBufferFifo2_CreditOk_2;
    logic                       scalarCreditOk2_3;
    logic                       pointCreditOk2_3;
    logic                       pointCreditOk2_3a;      //add one stage for timing
    logic                       pointRamBufferFifo3_CreditOk_2;
    logic                       scalarCreditOk3;
    logic                       pointCreditOk3;
    logic                       raw_scalarCreditOk3;
    logic                       raw_pointCreditOk3;
    logic                       scalarCreditOk3_0;
    logic                       pointCreditOk3_0;
    logic                       pointCreditOk3_0a;      //add one stage for timing
    logic                       pointRamBufferFifo0_CreditOk_3;
    logic                       scalarCreditOk3_1;
    logic                       pointCreditOk3_1;
    logic                       pointCreditOk3_1a;      //add one stage for timing
    logic                       pointRamBufferFifo1_CreditOk_3;
    logic                       scalarCreditOk3_2;
    logic                       pointCreditOk3_2;
    logic                       pointCreditOk3_2a;      //add one stage for timing
    logic                       pointRamBufferFifo2_CreditOk_3;
    logic                       scalarCreditOk3_3;
    logic                       pointCreditOk3_3;
    logic                       pointCreditOk3_3a;      //add one stage for timing
    logic                       pointRamBufferFifo3_CreditOk_3;

    logic                       pointRamFifo0_wvalid;
    logic                       pointRamFifo0_wready;
    logic   [1152-1:0]     pointRamFifo0_wdata;    
    logic                       pointRamFifo0_rvalid;
    logic                       pointRamFifo0_rready;
    logic   [1152-1:0]     pointRamFifo0_rdata;    

    logic                       pointPipeValid0_0;
    logic   [1152-1:0]     pointPipeData0_0;
    logic                       pointPipeValid0_1;
    logic   [1152-1:0]     pointPipeData0_1;
    logic                       pointPipeValid0_2;
    logic   [1152-1:0]     pointPipeData0_2;
    logic                       pointPipeValid0_3;
    logic   [1152-1:0]     pointPipeData0_3;
    logic                       pointPipeValid0_4;
    logic   [1152-1:0]     pointPipeData0_4;
    logic                       pointPipeValid0_5;
    logic   [1152-1:0]     pointPipeData0_5;
    logic                       pointPipeValid0_6;
    logic   [1152-1:0]     pointPipeData0_6;
    logic                       pointPipeValid0_last;
    logic   [1152-1:0]     pointPipeData0_last;

    logic                       pointRamBufferFifo0_wvalid;
    logic                       pointRamBufferFifo0_wready;
    logic   [1152-1:0]     pointRamBufferFifo0_wdata;    
    logic                       pointRamBufferFifo0_rvalid;
    logic                       pointRamBufferFifo0_rready;
    logic   [1152-1:0]     pointRamBufferFifo0_rdata;    
    logic   [`LOG2(point_ram_depth+1)-1:0] pointRamBufferFifo0_num;
    logic                       pointRamBufferFifo0_CreditOk;
    
    
    logic                       pointRamFifo1_wvalid;
    logic                       pointRamFifo1_wready;
    logic   [1152-1:0]     pointRamFifo1_wdata;    
    logic                       pointRamFifo1_rvalid;
    logic                       pointRamFifo1_rready;
    logic   [1152-1:0]     pointRamFifo1_rdata;    

    logic                       pointPipeValid1_0;
    logic   [1152-1:0]     pointPipeData1_0;
    logic                       pointPipeValid1_1;
    logic   [1152-1:0]     pointPipeData1_1;
    logic                       pointPipeValid1_2;
    logic   [1152-1:0]     pointPipeData1_2;
    logic                       pointPipeValid1_3;
    logic   [1152-1:0]     pointPipeData1_3;
    logic                       pointPipeValid1_4;
    logic   [1152-1:0]     pointPipeData1_4;
    logic                       pointPipeValid1_5;
    logic   [1152-1:0]     pointPipeData1_5;
    logic                       pointPipeValid1_6;
    logic   [1152-1:0]     pointPipeData1_6;
    logic                       pointPipeValid1_last;
    logic   [1152-1:0]     pointPipeData1_last;

    logic                       pointRamBufferFifo1_wvalid;
    logic                       pointRamBufferFifo1_wready;
    logic   [1152-1:0]     pointRamBufferFifo1_wdata;    
    logic                       pointRamBufferFifo1_rvalid;
    logic                       pointRamBufferFifo1_rready;
    logic   [1152-1:0]     pointRamBufferFifo1_rdata;    
    logic   [`LOG2(point_ram_depth+1)-1:0] pointRamBufferFifo1_num;
    logic                       pointRamBufferFifo1_CreditOk;
    
    
    logic                       pointRamFifo2_wvalid;
    logic                       pointRamFifo2_wready;
    logic   [1152-1:0]     pointRamFifo2_wdata;    
    logic                       pointRamFifo2_rvalid;
    logic                       pointRamFifo2_rready;
    logic   [1152-1:0]     pointRamFifo2_rdata;    

    logic                       pointPipeValid2_0;
    logic   [1152-1:0]     pointPipeData2_0;
    logic                       pointPipeValid2_1;
    logic   [1152-1:0]     pointPipeData2_1;
    logic                       pointPipeValid2_2;
    logic   [1152-1:0]     pointPipeData2_2;
    logic                       pointPipeValid2_3;
    logic   [1152-1:0]     pointPipeData2_3;
    logic                       pointPipeValid2_4;
    logic   [1152-1:0]     pointPipeData2_4;
    logic                       pointPipeValid2_5;
    logic   [1152-1:0]     pointPipeData2_5;
    logic                       pointPipeValid2_6;
    logic   [1152-1:0]     pointPipeData2_6;
    logic                       pointPipeValid2_last;
    logic   [1152-1:0]     pointPipeData2_last;

    logic                       pointRamBufferFifo2_wvalid;
    logic                       pointRamBufferFifo2_wready;
    logic   [1152-1:0]     pointRamBufferFifo2_wdata;    
    logic                       pointRamBufferFifo2_rvalid;
    logic                       pointRamBufferFifo2_rready;
    logic   [1152-1:0]     pointRamBufferFifo2_rdata;    
    logic   [`LOG2(point_ram_depth+1)-1:0] pointRamBufferFifo2_num;
    logic                       pointRamBufferFifo2_CreditOk;
    
    
    logic                       pointRamFifo3_wvalid;
    logic                       pointRamFifo3_wready;
    logic   [1152-1:0]     pointRamFifo3_wdata;    
    logic                       pointRamFifo3_rvalid;
    logic                       pointRamFifo3_rready;
    logic   [1152-1:0]     pointRamFifo3_rdata;    

    logic                       pointPipeValid3_0;
    logic   [1152-1:0]     pointPipeData3_0;
    logic                       pointPipeValid3_1;
    logic   [1152-1:0]     pointPipeData3_1;
    logic                       pointPipeValid3_2;
    logic   [1152-1:0]     pointPipeData3_2;
    logic                       pointPipeValid3_3;
    logic   [1152-1:0]     pointPipeData3_3;
    logic                       pointPipeValid3_4;
    logic   [1152-1:0]     pointPipeData3_4;
    logic                       pointPipeValid3_5;
    logic   [1152-1:0]     pointPipeData3_5;
    logic                       pointPipeValid3_6;
    logic   [1152-1:0]     pointPipeData3_6;
    logic                       pointPipeValid3_last;
    logic   [1152-1:0]     pointPipeData3_last;

    logic                       pointRamBufferFifo3_wvalid;
    logic                       pointRamBufferFifo3_wready;
    logic   [1152-1:0]     pointRamBufferFifo3_wdata;    
    logic                       pointRamBufferFifo3_rvalid;
    logic                       pointRamBufferFifo3_rready;
    logic   [1152-1:0]     pointRamBufferFifo3_rdata;    
    logic   [`LOG2(point_ram_depth+1)-1:0] pointRamBufferFifo3_num;
    logic                       pointRamBufferFifo3_CreditOk;
    
    

    logic                       scalarRamFifo_wvalid;
    logic                       scalarRamFifo_wready;
    logic   [64-1:0] scalarRamFifo_wdata;    
    logic                       scalarRamFifo_rvalid;
    logic                       scalarRamFifo_rready;
    logic   [64-1:0] scalarRamFifo_rdata;    
    logic                       scalarPipeValid;
    logic                       scalarPipeReady;
    logic   [64-1:0] scalarPipeData;    

    logic                       scalarPipeValid0;
    logic   [64-1:0] scalarPipeData0;    
    logic                       real_scalarPipeValid0;
    logic   [64-1:0] real_scalarPipeData0;    
    logic                       scalarPipeValid1;
    logic   [64-1:0] scalarPipeData1;    
    logic                       real_scalarPipeValid1;
    logic   [64-1:0] real_scalarPipeData1;    
    logic                       scalarPipeValid2;
    logic   [64-1:0] scalarPipeData2;    
    logic                       real_scalarPipeValid2;
    logic   [64-1:0] real_scalarPipeData2;    
    logic                       scalarPipeValid3;
    logic   [64-1:0] scalarPipeData3;    
    logic                       real_scalarPipeValid3;
    logic   [64-1:0] real_scalarPipeData3;    

    logic                           f2_result0_valid;
    logic                           f2_result0_ready;
    logic      [1536-1:0] f2_result0_data     ;

    logic                           real_f2_result0_valid;
    logic                           real_f2_result0_ready;
    logic      [1536-1:0] real_f2_result0_data     ;
    logic                           f2_result1_valid;
    logic                           f2_result1_ready;
    logic      [1536-1:0] f2_result1_data     ;

    logic                           real_f2_result1_valid;
    logic                           real_f2_result1_ready;
    logic      [1536-1:0] real_f2_result1_data     ;
    logic                           f2_result2_valid;
    logic                           f2_result2_ready;
    logic      [1536-1:0] f2_result2_data     ;

    logic                           real_f2_result2_valid;
    logic                           real_f2_result2_ready;
    logic      [1536-1:0] real_f2_result2_data     ;
    logic                           f2_result3_valid;
    logic                           f2_result3_ready;
    logic      [1536-1:0] f2_result3_data     ;

    logic                           real_f2_result3_valid;
    logic                           real_f2_result3_ready;
    logic      [1536-1:0] real_f2_result3_data     ;
//
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign clk      = ap_clk;
    assign rstN     = ap_rst_n;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pointShiftFifo
////////////////////////////////////////////////////////////////////////////////////////////////////////
`KEEP_HIE barrelShiftFifo_point pointShiftFifo0(
    .in_valid   (pointDdr0_valid),
    .in_ready   (pointDdr0_ready),
    .in_data    (pointDdr0_data),
    .out_valid  (pointRamFifo0_wvalid),
    .out_ready  (pointRamFifo0_wready),
    .out_data   (pointRamFifo0_wdata),
    .clk        (clk),
    .rstN       (rstN)
);
`KEEP_HIE barrelShiftFifo_point pointShiftFifo1(
    .in_valid   (pointDdr1_valid),
    .in_ready   (pointDdr1_ready),
    .in_data    (pointDdr1_data),
    .out_valid  (pointRamFifo1_wvalid),
    .out_ready  (pointRamFifo1_wready),
    .out_data   (pointRamFifo1_wdata),
    .clk        (clk),
    .rstN       (rstN)
);
`KEEP_HIE barrelShiftFifo_point pointShiftFifo2(
    .in_valid   (pointDdr2_valid),
    .in_ready   (pointDdr2_ready),
    .in_data    (pointDdr2_data),
    .out_valid  (pointRamFifo2_wvalid),
    .out_ready  (pointRamFifo2_wready),
    .out_data   (pointRamFifo2_wdata),
    .clk        (clk),
    .rstN       (rstN)
);
`KEEP_HIE barrelShiftFifo_point pointShiftFifo3(
    .in_valid   (pointDdr3_valid),
    .in_ready   (pointDdr3_ready),
    .in_data    (pointDdr3_data),
    .out_valid  (pointRamFifo3_wvalid),
    .out_ready  (pointRamFifo3_wready),
    .out_data   (pointRamFifo3_wdata),
    .clk        (clk),
    .rstN       (rstN)
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//pointRamFifo 
////////////////////////////////////////////////////////////////////////////////////////////////////////
`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamFifo0(
    .wDatValid      (pointRamFifo0_wvalid),
    .wDatReady      (pointRamFifo0_wready),
    .wDat           (pointRamFifo0_wdata),
    .rDatValid      (pointRamFifo0_rvalid),
    .rDatReady      (pointRamFifo0_rready),
    .rDat           (pointRamFifo0_rdata),
    .num            (),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);

assign pointRamFifo0_rready = pointRamBufferFifo0_CreditOk_0;

`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamFifo1(
    .wDatValid      (pointRamFifo1_wvalid),
    .wDatReady      (pointRamFifo1_wready),
    .wDat           (pointRamFifo1_wdata),
    .rDatValid      (pointRamFifo1_rvalid),
    .rDatReady      (pointRamFifo1_rready),
    .rDat           (pointRamFifo1_rdata),
    .num            (),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);

assign pointRamFifo1_rready = pointRamBufferFifo1_CreditOk_1;

`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamFifo2(
    .wDatValid      (pointRamFifo2_wvalid),
    .wDatReady      (pointRamFifo2_wready),
    .wDat           (pointRamFifo2_wdata),
    .rDatValid      (pointRamFifo2_rvalid),
    .rDatReady      (pointRamFifo2_rready),
    .rDat           (pointRamFifo2_rdata),
    .num            (),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);

assign pointRamFifo2_rready = pointRamBufferFifo2_CreditOk_2;

`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamFifo3(
    .wDatValid      (pointRamFifo3_wvalid),
    .wDatReady      (pointRamFifo3_wready),
    .wDat           (pointRamFifo3_wdata),
    .rDatValid      (pointRamFifo3_rvalid),
    .rDatReady      (pointRamFifo3_rready),
    .rDat           (pointRamFifo3_rdata),
    .num            (),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);

assign pointRamFifo3_rready = pointRamBufferFifo3_CreditOk_3;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline0_0, first 0: ddr channel 0. second 0: pipeline stage
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign pointPipeValid0_0 = pointRamFifo0_rvalid && pointRamFifo0_rready;
assign pointPipeData0_0  = pointRamFifo0_rdata;
onepipe #(.WIDTH(1152))u_pointPipe0_0(.valid_i(pointPipeValid0_0), .data_i(pointPipeData0_0), .valid_o(pointPipeValid0_1), .data_o(pointPipeData0_1), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe0_1(.valid_i(pointPipeValid0_1), .data_i(pointPipeData0_1), .valid_o(pointPipeValid0_2), .data_o(pointPipeData0_2), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe0_2(.valid_i(pointPipeValid0_2), .data_i(pointPipeData0_2), .valid_o(pointPipeValid0_3), .data_o(pointPipeData0_3), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe0_3(.valid_i(pointPipeValid0_3), .data_i(pointPipeData0_3), .valid_o(pointPipeValid0_4), .data_o(pointPipeData0_4), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe0_4(.valid_i(pointPipeValid0_4), .data_i(pointPipeData0_4), .valid_o(pointPipeValid0_5), .data_o(pointPipeData0_5), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe0_5(.valid_i(pointPipeValid0_5), .data_i(pointPipeData0_5), .valid_o(pointPipeValid0_6), .data_o(pointPipeData0_6), .clk(clk), .rstN(rstN));
assign pointPipeValid1_0 = pointRamFifo1_rvalid && pointRamFifo1_rready;
assign pointPipeData1_0  = pointRamFifo1_rdata;
onepipe #(.WIDTH(1152))u_pointPipe1_0(.valid_i(pointPipeValid1_0), .data_i(pointPipeData1_0), .valid_o(pointPipeValid1_1), .data_o(pointPipeData1_1), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe1_1(.valid_i(pointPipeValid1_1), .data_i(pointPipeData1_1), .valid_o(pointPipeValid1_2), .data_o(pointPipeData1_2), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe1_2(.valid_i(pointPipeValid1_2), .data_i(pointPipeData1_2), .valid_o(pointPipeValid1_3), .data_o(pointPipeData1_3), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe1_3(.valid_i(pointPipeValid1_3), .data_i(pointPipeData1_3), .valid_o(pointPipeValid1_4), .data_o(pointPipeData1_4), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe1_4(.valid_i(pointPipeValid1_4), .data_i(pointPipeData1_4), .valid_o(pointPipeValid1_5), .data_o(pointPipeData1_5), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe1_5(.valid_i(pointPipeValid1_5), .data_i(pointPipeData1_5), .valid_o(pointPipeValid1_6), .data_o(pointPipeData1_6), .clk(clk), .rstN(rstN));
assign pointPipeValid2_0 = pointRamFifo2_rvalid && pointRamFifo2_rready;
assign pointPipeData2_0  = pointRamFifo2_rdata;
onepipe #(.WIDTH(1152))u_pointPipe2_0(.valid_i(pointPipeValid2_0), .data_i(pointPipeData2_0), .valid_o(pointPipeValid2_1), .data_o(pointPipeData2_1), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe2_1(.valid_i(pointPipeValid2_1), .data_i(pointPipeData2_1), .valid_o(pointPipeValid2_2), .data_o(pointPipeData2_2), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe2_2(.valid_i(pointPipeValid2_2), .data_i(pointPipeData2_2), .valid_o(pointPipeValid2_3), .data_o(pointPipeData2_3), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe2_3(.valid_i(pointPipeValid2_3), .data_i(pointPipeData2_3), .valid_o(pointPipeValid2_4), .data_o(pointPipeData2_4), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe2_4(.valid_i(pointPipeValid2_4), .data_i(pointPipeData2_4), .valid_o(pointPipeValid2_5), .data_o(pointPipeData2_5), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe2_5(.valid_i(pointPipeValid2_5), .data_i(pointPipeData2_5), .valid_o(pointPipeValid2_6), .data_o(pointPipeData2_6), .clk(clk), .rstN(rstN));
assign pointPipeValid3_0 = pointRamFifo3_rvalid && pointRamFifo3_rready;
assign pointPipeData3_0  = pointRamFifo3_rdata;
onepipe #(.WIDTH(1152))u_pointPipe3_0(.valid_i(pointPipeValid3_0), .data_i(pointPipeData3_0), .valid_o(pointPipeValid3_1), .data_o(pointPipeData3_1), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe3_1(.valid_i(pointPipeValid3_1), .data_i(pointPipeData3_1), .valid_o(pointPipeValid3_2), .data_o(pointPipeData3_2), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe3_2(.valid_i(pointPipeValid3_2), .data_i(pointPipeData3_2), .valid_o(pointPipeValid3_3), .data_o(pointPipeData3_3), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe3_3(.valid_i(pointPipeValid3_3), .data_i(pointPipeData3_3), .valid_o(pointPipeValid3_4), .data_o(pointPipeData3_4), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe3_4(.valid_i(pointPipeValid3_4), .data_i(pointPipeData3_4), .valid_o(pointPipeValid3_5), .data_o(pointPipeData3_5), .clk(clk), .rstN(rstN));
onepipe #(.WIDTH(1152))u_pointPipe3_5(.valid_i(pointPipeValid3_5), .data_i(pointPipeData3_5), .valid_o(pointPipeValid3_6), .data_o(pointPipeData3_6), .clk(clk), .rstN(rstN));
assign pointPipeValid0_last = pointPipeValid0_4;
assign pointPipeValid1_last = pointPipeValid1_3;
assign pointPipeValid2_last = pointPipeValid2_4;
assign pointPipeValid3_last = pointPipeValid3_6;
assign pointPipeData0_last = pointPipeData0_4;
assign pointPipeData1_last = pointPipeData1_3;
assign pointPipeData2_last = pointPipeData2_4;
assign pointPipeData3_last = pointPipeData3_6;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//pointRamBufferFifo
////////////////////////////////////////////////////////////////////////////////////////////////////////
`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamBufferFifo0(
    .wDatValid      (pointRamBufferFifo0_wvalid),
    .wDatReady      (pointRamBufferFifo0_wready),
    .wDat           (pointRamBufferFifo0_wdata),
    .rDatValid      (pointRamBufferFifo0_rvalid),
    .rDatReady      (pointRamBufferFifo0_rready),
    .rDat           (pointRamBufferFifo0_rdata),
    .num            (pointRamBufferFifo0_num),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);
assign pointRamBufferFifo0_wvalid   = pointPipeValid0_last;
assign pointRamBufferFifo0_wdata    = pointPipeData0_last;
assign pointRamBufferFifo0_rready   = switch_ready && switch == 0;
assign pointRamBufferFifo0_CreditOk = pointRamBufferFifo0_num < point_ram_depth - 7; //tmp use

`assert_clk(pointRamBufferFifo0_wvalid |-> pointRamBufferFifo0_wready)
`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamBufferFifo1(
    .wDatValid      (pointRamBufferFifo1_wvalid),
    .wDatReady      (pointRamBufferFifo1_wready),
    .wDat           (pointRamBufferFifo1_wdata),
    .rDatValid      (pointRamBufferFifo1_rvalid),
    .rDatReady      (pointRamBufferFifo1_rready),
    .rDat           (pointRamBufferFifo1_rdata),
    .num            (pointRamBufferFifo1_num),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);
assign pointRamBufferFifo1_wvalid   = pointPipeValid1_last;
assign pointRamBufferFifo1_wdata    = pointPipeData1_last;
assign pointRamBufferFifo1_rready   = switch_ready && switch == 1;
assign pointRamBufferFifo1_CreditOk = pointRamBufferFifo1_num < point_ram_depth - 7; //tmp use

`assert_clk(pointRamBufferFifo1_wvalid |-> pointRamBufferFifo1_wready)
`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamBufferFifo2(
    .wDatValid      (pointRamBufferFifo2_wvalid),
    .wDatReady      (pointRamBufferFifo2_wready),
    .wDat           (pointRamBufferFifo2_wdata),
    .rDatValid      (pointRamBufferFifo2_rvalid),
    .rDatReady      (pointRamBufferFifo2_rready),
    .rDat           (pointRamBufferFifo2_rdata),
    .num            (pointRamBufferFifo2_num),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);
assign pointRamBufferFifo2_wvalid   = pointPipeValid2_last;
assign pointRamBufferFifo2_wdata    = pointPipeData2_last;
assign pointRamBufferFifo2_rready   = switch_ready && switch == 2;
assign pointRamBufferFifo2_CreditOk = pointRamBufferFifo2_num < point_ram_depth - 7; //tmp use

`assert_clk(pointRamBufferFifo2_wvalid |-> pointRamBufferFifo2_wready)
`KEEP_HIE flopOutSramFifo #(
    .DATA       (1152),
    .DEPTH      (point_ram_depth)
    ) pointRamBufferFifo3(
    .wDatValid      (pointRamBufferFifo3_wvalid),
    .wDatReady      (pointRamBufferFifo3_wready),
    .wDat           (pointRamBufferFifo3_wdata),
    .rDatValid      (pointRamBufferFifo3_rvalid),
    .rDatReady      (pointRamBufferFifo3_rready),
    .rDat           (pointRamBufferFifo3_rdata),
    .num            (pointRamBufferFifo3_num),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);
assign pointRamBufferFifo3_wvalid   = pointPipeValid3_last;
assign pointRamBufferFifo3_wdata    = pointPipeData3_last;
assign pointRamBufferFifo3_rready   = switch_ready && switch == 3;
assign pointRamBufferFifo3_CreditOk = pointRamBufferFifo3_num < point_ram_depth - 7; //tmp use

`assert_clk(pointRamBufferFifo3_wvalid |-> pointRamBufferFifo3_wready)
////////////////////////////////////////////////////////////////////////////////////////////////////////
//mux point from 4 channel into one
////////////////////////////////////////////////////////////////////////////////////////////////////////
zprize_msm_point_mux zprize_msm_point_mux(
    .pointCreditOk0_0          (pointCreditOk0_0      ),
    .pointCreditOk1_0          (pointCreditOk1_0      ),
    .pointCreditOk2_0          (pointCreditOk2_0      ),
    .pointCreditOk3_0          (pointCreditOk3_0      ),
    .pointRamBufferFifo0_rvalid    (pointRamBufferFifo0_rvalid),
    .pointRamBufferFifo0_rready    (pointRamBufferFifo0_rready),
    .pointRamBufferFifo0_rdata     (pointRamBufferFifo0_rdata ),
    .pointCreditOk0_1          (pointCreditOk0_1      ),
    .pointCreditOk1_1          (pointCreditOk1_1      ),
    .pointCreditOk2_1          (pointCreditOk2_1      ),
    .pointCreditOk3_1          (pointCreditOk3_1      ),
    .pointRamBufferFifo1_rvalid    (pointRamBufferFifo1_rvalid),
    .pointRamBufferFifo1_rready    (pointRamBufferFifo1_rready),
    .pointRamBufferFifo1_rdata     (pointRamBufferFifo1_rdata ),
    .pointCreditOk0_2          (pointCreditOk0_2      ),
    .pointCreditOk1_2          (pointCreditOk1_2      ),
    .pointCreditOk2_2          (pointCreditOk2_2      ),
    .pointCreditOk3_2          (pointCreditOk3_2      ),
    .pointRamBufferFifo2_rvalid    (pointRamBufferFifo2_rvalid),
    .pointRamBufferFifo2_rready    (pointRamBufferFifo2_rready),
    .pointRamBufferFifo2_rdata     (pointRamBufferFifo2_rdata ),
    .pointCreditOk0_3          (pointCreditOk0_3      ),
    .pointCreditOk1_3          (pointCreditOk1_3      ),
    .pointCreditOk2_3          (pointCreditOk2_3      ),
    .pointCreditOk3_3          (pointCreditOk3_3      ),
    .pointRamBufferFifo3_rvalid    (pointRamBufferFifo3_rvalid),
    .pointRamBufferFifo3_rready    (pointRamBufferFifo3_rready),
    .pointRamBufferFifo3_rdata     (pointRamBufferFifo3_rdata ),
    .switch          (switch          ),
    .switch_ready    (switch_ready    ),
    .pointPipeValid  (pointPipeValid  ),
    .pointPipeData   (pointPipeData   ),
    .clk             (clk             ),
    .rstN            (rstN            )
);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pointPipe2: SLR1
////////////////////////////////////////////////////////////////////////////////////////////////////////

//SLR3
`KEEP_HIE onepipe #(.WIDTH(1152))u_pointPipe3 (.valid_i(pointPipeValid2a), .data_i(pointPipeData2a), .valid_o(pointPipeValid3 ), .data_o(pointPipeData3 ), .clk(clk), .rstN(rstN));

//SLR2
`KEEP_HIE onepipe #(.WIDTH(1152))u_pointPipe2 (.valid_i(pointPipeValid1a), .data_i(pointPipeData1a), .valid_o(pointPipeValid2 ), .data_o(pointPipeData2 ), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe #(.WIDTH(1152))u_pointPipe2a(.valid_i(pointPipeValid2 ), .data_i(pointPipeData2 ), .valid_o(pointPipeValid2a), .data_o(pointPipeData2a), .clk(clk), .rstN(rstN));

//SLR1
`KEEP_HIE onepipe #(.WIDTH(1152))u_pointPipe1 (.valid_i(pointPipeValid  ), .data_i(pointPipeData  ), .valid_o(pointPipeValid1 ), .data_o(pointPipeData1 ), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe #(.WIDTH(1152))u_pointPipe1a(.valid_i(pointPipeValid1 ), .data_i(pointPipeData1 ), .valid_o(pointPipeValid1a), .data_o(pointPipeData1a), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe #(.WIDTH(1152))u_pointPipe1b(.valid_i(pointPipeValid1 ), .data_i(pointPipeData1 ), .valid_o(pointPipeValid1b), .data_o(pointPipeData1b), .clk(clk), .rstN(rstN));

//SLR0
`KEEP_HIE onepipe #(.WIDTH(1152))u_pointPipe0 (.valid_i(pointPipeValid1b), .data_i(pointPipeData1b), .valid_o(pointPipeValid0 ), .data_o(pointPipeData0 ), .clk(clk), .rstN(rstN));

////////////////////////////////////////////////////////////////////////////////////////////////////////
//scalarShiftFifo
////////////////////////////////////////////////////////////////////////////////////////////////////////
`KEEP_HIE barrelShiftFifo_scalar scalarShiftFifo(
    .in_valid   (scalarDdr_valid),
    .in_ready   (scalarDdr_ready),
    .in_data    (scalarDdr_data),
    .out_valid  (scalarRamFifo_wvalid),
    .out_ready  (scalarRamFifo_wready),
    .out_data   (scalarRamFifo_wdata),
    .clk        (clk),
    .rstN       (rstN)
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//scalarRamFifo 
////////////////////////////////////////////////////////////////////////////////////////////////////////
`KEEP_HIE flopOutSramFifo #(
    .DATA       (64),
    .DEPTH      (scalar_ram_depth)
    ) scalarRamFifo (
    .wDatValid      (scalarRamFifo_wvalid),
    .wDatReady      (scalarRamFifo_wready),
    .wDat           (scalarRamFifo_wdata),
    .rDatValid      (scalarRamFifo_rvalid),
    .rDatReady      (scalarPipeReady),
    .rDat           (scalarPipeData),
    .num            (),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);
assign scalarPipeValid = scalarRamFifo_rvalid && 
                    scalarCreditOk0_1 && 
                    scalarCreditOk1_1 && 
                    scalarCreditOk2_1 && 
                    scalarCreditOk3_1 && 
                    1;
assign scalarPipeReady =        //scalar Ram Fifo is at SLR1
                    scalarCreditOk0_1 && 
                    scalarCreditOk1_1 && 
                    scalarCreditOk2_1 && 
                    scalarCreditOk3_1 && 
                    1;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//scalarPipe1: SLR1
////////////////////////////////////////////////////////////////////////////////////////////////////////
`KEEP_HIE onepipe #(.WIDTH(64))u_scalarPipe1(.valid_i(scalarPipeValid ), .data_i(scalarPipeData ), .valid_o(scalarPipeValid1), .data_o(scalarPipeData1), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe #(.WIDTH(64))u_scalarPipe2(.valid_i(scalarPipeValid1), .data_i(scalarPipeData1), .valid_o(scalarPipeValid2), .data_o(scalarPipeData2), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe #(.WIDTH(64))u_scalarPipe3(.valid_i(scalarPipeValid2), .data_i(scalarPipeData2), .valid_o(scalarPipeValid3), .data_o(scalarPipeData3), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe #(.WIDTH(64))u_scalarPipe0(.valid_i(scalarPipeValid1), .data_i(scalarPipeData1), .valid_o(scalarPipeValid0), .data_o(scalarPipeData0), .clk(clk), .rstN(rstN));

////////////////////////////////////////////////////////////////////////////////////////////////////////
//creditOk
//latch from core to each DDR channel
////////////////////////////////////////////////////////////////////////////////////////////////////////
//0_0: first 0: core 0. second 0: SLR0
`KEEP_HIE onepipe u_scalarCreditOk0_0(.valid_i(scalarCreditOk0  ), .data_i(), .valid_o(scalarCreditOk0_0), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk0_1(.valid_i(scalarCreditOk0_0), .data_i(), .valid_o(scalarCreditOk0_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk0_2(.valid_i(scalarCreditOk0_1), .data_i(), .valid_o(scalarCreditOk0_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk0_3(.valid_i(scalarCreditOk0_2), .data_i(), .valid_o(scalarCreditOk0_3), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_scalarCreditOk1_1(.valid_i(scalarCreditOk1  ), .data_i(), .valid_o(scalarCreditOk1_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk1_0(.valid_i(scalarCreditOk1_1), .data_i(), .valid_o(scalarCreditOk1_0), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk1_2(.valid_i(scalarCreditOk1_1), .data_i(), .valid_o(scalarCreditOk1_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk1_3(.valid_i(scalarCreditOk1_2), .data_i(), .valid_o(scalarCreditOk1_3), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_scalarCreditOk2_2(.valid_i(scalarCreditOk2  ), .data_i(), .valid_o(scalarCreditOk2_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk2_1(.valid_i(scalarCreditOk2_2), .data_i(), .valid_o(scalarCreditOk2_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk2_0(.valid_i(scalarCreditOk2_2), .data_i(), .valid_o(scalarCreditOk2_0), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk2_3(.valid_i(scalarCreditOk2_2), .data_i(), .valid_o(scalarCreditOk2_3), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_scalarCreditOk3_3(.valid_i(scalarCreditOk3  ), .data_i(), .valid_o(scalarCreditOk3_3), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk3_2(.valid_i(scalarCreditOk3_3), .data_i(), .valid_o(scalarCreditOk3_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk3_1(.valid_i(scalarCreditOk3_2), .data_i(), .valid_o(scalarCreditOk3_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_scalarCreditOk3_0(.valid_i(scalarCreditOk3_1), .data_i(), .valid_o(scalarCreditOk3_0), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_pointCreditOk0_0 (.valid_i(pointCreditOk0  ), .data_i(), .valid_o(pointCreditOk0_0 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk0_1 (.valid_i(pointCreditOk0_0), .data_i(), .valid_o(pointCreditOk0_1 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk0_2 (.valid_i(pointCreditOk0_1), .data_i(), .valid_o(pointCreditOk0_2 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk0_3 (.valid_i(pointCreditOk0_2), .data_i(), .valid_o(pointCreditOk0_3 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk0_2a(.valid_i(pointCreditOk0_2), .data_i(), .valid_o(pointCreditOk0_2a), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_pointCreditOk1_1 (.valid_i(pointCreditOk1  ), .data_i(), .valid_o(pointCreditOk1_1 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk1_0 (.valid_i(pointCreditOk1_1), .data_i(), .valid_o(pointCreditOk1_0 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk1_2 (.valid_i(pointCreditOk1_1), .data_i(), .valid_o(pointCreditOk1_2 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk1_3 (.valid_i(pointCreditOk1_2), .data_i(), .valid_o(pointCreditOk1_3 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk1_2a(.valid_i(pointCreditOk1_2), .data_i(), .valid_o(pointCreditOk1_2a), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_pointCreditOk2_2 (.valid_i(pointCreditOk2  ), .data_i(), .valid_o(pointCreditOk2_2 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk2_1 (.valid_i(pointCreditOk2_2), .data_i(), .valid_o(pointCreditOk2_1 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk2_0 (.valid_i(pointCreditOk2_1), .data_i(), .valid_o(pointCreditOk2_0 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk2_3 (.valid_i(pointCreditOk2_2), .data_i(), .valid_o(pointCreditOk2_3 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk2_2a(.valid_i(pointCreditOk2_2), .data_i(), .valid_o(pointCreditOk2_2a), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_pointCreditOk3_3 (.valid_i(pointCreditOk3  ), .data_i(), .valid_o(pointCreditOk3_3 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk3_2 (.valid_i(pointCreditOk3_3), .data_i(), .valid_o(pointCreditOk3_2 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk3_1 (.valid_i(pointCreditOk3_2), .data_i(), .valid_o(pointCreditOk3_1 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk3_0 (.valid_i(pointCreditOk3_1), .data_i(), .valid_o(pointCreditOk3_0 ), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointCreditOk3_2a(.valid_i(pointCreditOk3_2), .data_i(), .valid_o(pointCreditOk3_2a), .data_o(), .clk(clk), .rstN(rstN));

`KEEP_HIE onepipe u_pointRamBuffer_CreditOk0_2(.valid_i(pointRamBufferFifo0_CreditOk  ), .data_i(), .valid_o(pointRamBufferFifo0_CreditOk_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk0_1(.valid_i(pointRamBufferFifo0_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo0_CreditOk_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk0_0(.valid_i(pointRamBufferFifo0_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo0_CreditOk_0), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk0_3(.valid_i(pointRamBufferFifo0_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo0_CreditOk_3), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk1_2(.valid_i(pointRamBufferFifo1_CreditOk  ), .data_i(), .valid_o(pointRamBufferFifo1_CreditOk_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk1_1(.valid_i(pointRamBufferFifo1_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo1_CreditOk_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk1_0(.valid_i(pointRamBufferFifo1_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo1_CreditOk_0), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk1_3(.valid_i(pointRamBufferFifo1_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo1_CreditOk_3), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk2_2(.valid_i(pointRamBufferFifo2_CreditOk  ), .data_i(), .valid_o(pointRamBufferFifo2_CreditOk_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk2_1(.valid_i(pointRamBufferFifo2_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo2_CreditOk_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk2_0(.valid_i(pointRamBufferFifo2_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo2_CreditOk_0), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk2_3(.valid_i(pointRamBufferFifo2_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo2_CreditOk_3), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk3_2(.valid_i(pointRamBufferFifo3_CreditOk  ), .data_i(), .valid_o(pointRamBufferFifo3_CreditOk_2), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk3_1(.valid_i(pointRamBufferFifo3_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo3_CreditOk_1), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk3_0(.valid_i(pointRamBufferFifo3_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo3_CreditOk_0), .data_o(), .clk(clk), .rstN(rstN));
`KEEP_HIE onepipe u_pointRamBuffer_CreditOk3_3(.valid_i(pointRamBufferFifo3_CreditOk_2), .data_i(), .valid_o(pointRamBufferFifo3_CreditOk_3), .data_o(), .clk(clk), .rstN(rstN));

////////////////////////////////////////////////////////////////////////////////////////////////////////
//core 
////////////////////////////////////////////////////////////////////////////////////////////////////////
generate 
if(0 != 1) begin: core0
`KEEP_HIE zprize_core zprize0(
    .point_valid_B    (real_pointPipeValid0),
    .point_ready_B    (),
    .point_data_B     (real_pointPipeData0),
    .scalar_valid_B   (real_scalarPipeValid0),
    .scalar_ready_B   (),
    .scalar_data_B    (real_scalarPipeData0[16*0+:16]),  //CVSBIT = 15, 64 = 64, [63] [47] [31] [15] bit is dummy
    .f2_result_valid    (f2_result0_valid    ),
    .f2_result_ready    (f2_result0_ready    ),
    .f2_result_data     (f2_result0_data          ),
    .scalarCreditOk       (raw_scalarCreditOk0),
    .affineCreditOk       (raw_pointCreditOk0),
    .msm_num            (msm_num0),
    .clk            (clk),
    .rstN           (rstN)
);
`assert_clk(zprize0.point_valid_B |-> zprize0.point_ready_B)
`assert_clk(zprize0.scalar_valid_B|-> zprize0.scalar_ready_B)
end 
else begin
end
endgenerate

//omit core1's output, so that core1 will be empty
assign real_pointPipeValid0  = 0 == 1 ? 0 : pointPipeValid0;
assign real_pointPipeData0   = 0 == 1 ? 0 : pointPipeData0;
assign real_scalarPipeValid0  = 0 == 1 ? 0 : scalarPipeValid0;
assign real_scalarPipeData0   = 0 == 1 ? 0 : scalarPipeData0;
assign real_f2_result0_valid = 0 == 1 ? 0 : f2_result0_valid;
assign real_f2_result0_data  = 0 == 1 ? 0 : f2_result0_data ;
assign f2_result0_ready      = real_f2_result0_ready;

assign scalarCreditOk0       = 0 == 1 ? 1 :  raw_scalarCreditOk0;
assign pointCreditOk0        = 0 == 1 ? 1 :  raw_pointCreditOk0;

generate 
if(1 != 1) begin: core1
`KEEP_HIE zprize_core zprize1(
    .point_valid_B    (real_pointPipeValid1),
    .point_ready_B    (),
    .point_data_B     (real_pointPipeData1),
    .scalar_valid_B   (real_scalarPipeValid1),
    .scalar_ready_B   (),
    .scalar_data_B    (real_scalarPipeData1[16*1+:16]),  //CVSBIT = 15, 64 = 64, [63] [47] [31] [15] bit is dummy
    .f2_result_valid    (f2_result1_valid    ),
    .f2_result_ready    (f2_result1_ready    ),
    .f2_result_data     (f2_result1_data          ),
    .scalarCreditOk       (raw_scalarCreditOk1),
    .affineCreditOk       (raw_pointCreditOk1),
    .msm_num            (msm_num1),
    .clk            (clk),
    .rstN           (rstN)
);
`assert_clk(zprize1.point_valid_B |-> zprize1.point_ready_B)
`assert_clk(zprize1.scalar_valid_B|-> zprize1.scalar_ready_B)
end 
else begin
end
endgenerate

//omit core1's output, so that core1 will be empty
assign real_pointPipeValid1  = 1 == 1 ? 0 : pointPipeValid1;
assign real_pointPipeData1   = 1 == 1 ? 0 : pointPipeData1;
assign real_scalarPipeValid1  = 1 == 1 ? 0 : scalarPipeValid1;
assign real_scalarPipeData1   = 1 == 1 ? 0 : scalarPipeData1;
assign real_f2_result1_valid = 1 == 1 ? 0 : f2_result1_valid;
assign real_f2_result1_data  = 1 == 1 ? 0 : f2_result1_data ;
assign f2_result1_ready      = real_f2_result1_ready;

assign scalarCreditOk1       = 1 == 1 ? 1 :  raw_scalarCreditOk1;
assign pointCreditOk1        = 1 == 1 ? 1 :  raw_pointCreditOk1;

generate 
if(2 != 1) begin: core2
`KEEP_HIE zprize_core zprize2(
    .point_valid_B    (real_pointPipeValid2),
    .point_ready_B    (),
    .point_data_B     (real_pointPipeData2),
    .scalar_valid_B   (real_scalarPipeValid2),
    .scalar_ready_B   (),
    .scalar_data_B    (real_scalarPipeData2[16*2+:16]),  //CVSBIT = 15, 64 = 64, [63] [47] [31] [15] bit is dummy
    .f2_result_valid    (f2_result2_valid    ),
    .f2_result_ready    (f2_result2_ready    ),
    .f2_result_data     (f2_result2_data          ),
    .scalarCreditOk       (raw_scalarCreditOk2),
    .affineCreditOk       (raw_pointCreditOk2),
    .msm_num            (msm_num2),
    .clk            (clk),
    .rstN           (rstN)
);
`assert_clk(zprize2.point_valid_B |-> zprize2.point_ready_B)
`assert_clk(zprize2.scalar_valid_B|-> zprize2.scalar_ready_B)
end 
else begin
end
endgenerate

//omit core1's output, so that core1 will be empty
assign real_pointPipeValid2  = 2 == 1 ? 0 : pointPipeValid2;
assign real_pointPipeData2   = 2 == 1 ? 0 : pointPipeData2;
assign real_scalarPipeValid2  = 2 == 1 ? 0 : scalarPipeValid2;
assign real_scalarPipeData2   = 2 == 1 ? 0 : scalarPipeData2;
assign real_f2_result2_valid = 2 == 1 ? 0 : f2_result2_valid;
assign real_f2_result2_data  = 2 == 1 ? 0 : f2_result2_data ;
assign f2_result2_ready      = real_f2_result2_ready;

assign scalarCreditOk2       = 2 == 1 ? 1 :  raw_scalarCreditOk2;
assign pointCreditOk2        = 2 == 1 ? 1 :  raw_pointCreditOk2;

generate 
if(3 != 1) begin: core3
`KEEP_HIE zprize_core zprize3(
    .point_valid_B    (real_pointPipeValid3),
    .point_ready_B    (),
    .point_data_B     (real_pointPipeData3),
    .scalar_valid_B   (real_scalarPipeValid3),
    .scalar_ready_B   (),
    .scalar_data_B    (real_scalarPipeData3[16*3+:16]),  //CVSBIT = 15, 64 = 64, [63] [47] [31] [15] bit is dummy
    .f2_result_valid    (f2_result3_valid    ),
    .f2_result_ready    (f2_result3_ready    ),
    .f2_result_data     (f2_result3_data          ),
    .scalarCreditOk       (raw_scalarCreditOk3),
    .affineCreditOk       (raw_pointCreditOk3),
    .msm_num            (msm_num3),
    .clk            (clk),
    .rstN           (rstN)
);
`assert_clk(zprize3.point_valid_B |-> zprize3.point_ready_B)
`assert_clk(zprize3.scalar_valid_B|-> zprize3.scalar_ready_B)
end 
else begin
end
endgenerate

//omit core1's output, so that core1 will be empty
assign real_pointPipeValid3  = 3 == 1 ? 0 : pointPipeValid3;
assign real_pointPipeData3   = 3 == 1 ? 0 : pointPipeData3;
assign real_scalarPipeValid3  = 3 == 1 ? 0 : scalarPipeValid3;
assign real_scalarPipeData3   = 3 == 1 ? 0 : scalarPipeData3;
assign real_f2_result3_valid = 3 == 1 ? 0 : f2_result3_valid;
assign real_f2_result3_data  = 3 == 1 ? 0 : f2_result3_data ;
assign f2_result3_ready      = real_f2_result3_ready;

assign scalarCreditOk3       = 3 == 1 ? 1 :  raw_scalarCreditOk3;
assign pointCreditOk3        = 3 == 1 ? 1 :  raw_pointCreditOk3;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////////////////////////////
`KEEP_HIE barrelShiftFifo_result resultShiftFifo0(
    .in_valid   (real_f2_result0_valid),
    .in_ready   (real_f2_result0_ready),
    .in_data    (real_f2_result0_data),
    .out_valid  (returnDdr0_valid),
    .out_ready  (returnDdr0_ready),
    .out_data   (returnDdr0_data),
    .clk        (clk),
    .rstN       (rstN)
);
`KEEP_HIE barrelShiftFifo_result resultShiftFifo1(
    .in_valid   (real_f2_result1_valid),
    .in_ready   (real_f2_result1_ready),
    .in_data    (real_f2_result1_data),
    .out_valid  (returnDdr1_valid),
    .out_ready  (returnDdr1_ready),
    .out_data   (returnDdr1_data),
    .clk        (clk),
    .rstN       (rstN)
);
`KEEP_HIE barrelShiftFifo_result resultShiftFifo2(
    .in_valid   (real_f2_result2_valid),
    .in_ready   (real_f2_result2_ready),
    .in_data    (real_f2_result2_data),
    .out_valid  (returnDdr2_valid),
    .out_ready  (returnDdr2_ready),
    .out_data   (returnDdr2_data),
    .clk        (clk),
    .rstN       (rstN)
);
`KEEP_HIE barrelShiftFifo_result resultShiftFifo3(
    .in_valid   (real_f2_result3_valid),
    .in_ready   (real_f2_result3_ready),
    .in_data    (real_f2_result3_data),
    .out_valid  (returnDdr3_valid),
    .out_ready  (returnDdr3_ready),
    .out_data   (returnDdr3_data),
    .clk        (clk),
    .rstN       (rstN)
);
endmodule
