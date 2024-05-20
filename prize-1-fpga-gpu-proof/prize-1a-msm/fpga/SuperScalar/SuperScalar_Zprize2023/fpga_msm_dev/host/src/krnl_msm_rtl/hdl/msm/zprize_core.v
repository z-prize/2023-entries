`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

module zprize_core import zprize_param::*;
(
//point in        
    input                       point_valid_B,
    output                      point_ready_B,
    input   [1152-1:0]     point_data_B,
//scalar in
    input                       scalar_valid_B,
    output                      scalar_ready_B,
    input   [16-1:0]       scalar_data_B,
//output result
    output                                   f2_result_valid,
    input                                  f2_result_ready,
    output              [1536-1:0] f2_result_data     ,
//global
    output                       scalarCreditOk, //have enough credit to receive input
    output                       affineCreditOk, //have enough credit to receive input
    input   [`LOG2((1<<24))-1:0] msm_num,
    input                        clk,
    input                        rstN            
);
//pipeline C: point/scalar input
//pipeline S: F3 output
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    logic   [`LOG2(scalar_ram_depth+1)-1:0] scalarFifoNum;
    logic   [`LOG2((1<<9)+1)-1:0]   affineIndexFifoNum;
    logic   [`LOG2((1<<9)+1)-1:0]   affineRamNum;
    logic                                   point_valid_r;
    logic   [`LOG2((1<<9))-1:0]    point_addr_r;
    logic   [1152-1:0]                 point_data_r;

    logic                      point_valid_B1;
    logic                      point_ready_B1;
    logic   [`LOG2((1<<9))-1:0]    point_addr_B1;
    logic  [1152-1:0]     point_data_B1;
    logic                       ready_B1;

    logic                                   affine_valid_B1;     //pipeline B1 affineValid has available item
    logic                                   affine_ready_B1;
    logic   [`LOG2((1<<9))-1:0]     affineIndex_B1;
    logic                                   ack_C;
    logic                                   valid_C;

    logic                       scalar_valid_C;
    logic                       scalar_ready_C;
    logic   [16-1:0]       scalar_data_C;
    logic                                   cancel_bucket0_C;

    logic                                   point_affine_valid_C;     
    logic                                   point_affine_ready_C;
    logic   [`LOG2((1<<9))-1:0]     affineIndex_C;

    logic                                   calc_bucket_idle;        //should remain high until valid_S && ready_S, should be 0 at the first cycle of valid_S
//scalar
    logic                                   calc_bucket_ready_C; 
    logic   [16-1:0]                   scalar_C;
//read pointRAM
    logic                                   point_re_I;
    logic   [`LOG2((1<<9))-1:0]    point_raddr_I;
    logic   [1152-1:0]            point_rdata_K;
//read/write accRAM 
    logic                                   acc_we_S;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S;
    logic [1536-1:0]              acc_wdata_S;
    logic                                   acc_re_G6;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6;
    logic                                   acc_re_G6_0;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_0;
    logic                                   acc_we_S_0;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_0;
    logic                                   acc_re_G6_1;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_1;
    logic                                   acc_we_S_1;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_1;
    logic                                   acc_re_G6_2;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_2;
    logic                                   acc_we_S_2;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_2;
    logic                                   acc_re_G6_3;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_3;
    logic                                   acc_we_S_3;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_3;
    logic                                   acc_re_G6_4;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_4;
    logic                                   acc_we_S_4;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_4;
    logic                                   acc_re_G6_5;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_5;
    logic                                   acc_we_S_5;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_5;
    logic                                   acc_re_G6_6;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_6;
    logic                                   acc_we_S_6;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_6;
    logic                                   acc_re_G6_7;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_G6_7;
    logic                                   acc_we_S_7;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S_7;
    `MAX_FANOUT32 logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_K1;
    logic [1536-1:0]              acc_rdata_K1;
    logic [1536-1:0]              acc_rdata_K;
    logic                                   acc_we_S_0;
    logic                                   acc_re_G6_0;
    logic                                   acc_re_G6_0_r;
    logic [1536-1:0]              acc_rdata_K1_0;
    logic                                   acc_we_S_1;
    logic                                   acc_re_G6_1;
    logic                                   acc_re_G6_1_r;
    logic [1536-1:0]              acc_rdata_K1_1;
    logic                                   acc_we_S_2;
    logic                                   acc_re_G6_2;
    logic                                   acc_re_G6_2_r;
    logic [1536-1:0]              acc_rdata_K1_2;
    logic                                   acc_we_S_3;
    logic                                   acc_re_G6_3;
    logic                                   acc_re_G6_3_r;
    logic [1536-1:0]              acc_rdata_K1_3;
    logic                                   acc_we_S_4;
    logic                                   acc_re_G6_4;
    logic                                   acc_re_G6_4_r;
    logic [1536-1:0]              acc_rdata_K1_4;
    logic                                   acc_we_S_5;
    logic                                   acc_re_G6_5;
    logic                                   acc_re_G6_5_r;
    logic [1536-1:0]              acc_rdata_K1_5;
    logic                                   acc_we_S_6;
    logic                                   acc_re_G6_6;
    logic                                   acc_re_G6_6_r;
    logic [1536-1:0]              acc_rdata_K1_6;
    logic                                   acc_we_S_7;
    logic                                   acc_re_G6_7;
    logic                                   acc_re_G6_7_r;
    logic [1536-1:0]              acc_rdata_K1_7;
    logic [1536-1:0]              acc_rdata_K2;

    logic   [(1<<9)-1:0]           affineValid_B;
    logic   [(1<<9)-1:0]           free_affineValid;
    logic   [(1<<9)-1:0]           free_affineValid_I;
    logic   [(1<<9)-1:0]           cancel_affineValid;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
flopOutFifo #(
    .DATA       (1152),
    .DEPTH      (4)
    ) pointFifo (
    .wDatValid      (point_valid_B),
    .wDatReady      (point_ready_B),
    .wDat           (point_data_B),
    .rDatValid      (point_valid_B1),
    .rDatReady      (point_ready_B1),
    .rDat           (point_data_B1),
    .num            (),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);

//    logic   [`LOG2((1<<24))-1:0]          N_count_B1;   //for debug
    logic   [`LOG2((1<<24)):0]          N_count_B1;   //for debug
    logic   [`LOG2((1<<24)):0]          N_count_C;   //for debug
`Flop(N_count_C, ack_C, N_count_C + 1)
`Flop(N_count_B1, point_valid_B1 && point_ready_B1, N_count_B1 + 1)
////////////////////////////////////////////////////////////////////////////////////////////////////////
//scalar FIFO
////////////////////////////////////////////////////////////////////////////////////////////////////////
flopOutSramFifo #(
    .DATA       (16),
    .DEPTH      (scalar_ram_depth)
    ) scalarFifo (
    .wDatValid      (scalar_valid_B),
    .wDatReady      (scalar_ready_B),
    .wDat           (scalar_data_B),
    .rDatValid      (scalar_valid_C),
    .rDatReady      (scalar_ready_C),
    .rDat           (scalar_data_C),
    .num            (scalarFifoNum),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);

    assign scalarCreditOk = scalarFifoNum < scalar_ram_depth - pipeLatency;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline control
////////////////////////////////////////////////////////////////////////////////////////////////////////
//both point and scalar are all allowed accepted, the ack can be true
assign ack_C = 
            scalar_valid_C && //scalar into calc_bucket
            point_affine_valid_C && //point data in and affineRAM has available item
            calc_bucket_ready_C &&      //calc_bucket ready
            1;
            
assign scalar_ready_C = ack_C;
assign point_affine_ready_C = ack_C;

assign valid_C = ack_C && !cancel_bucket0_C;
assign cancel_bucket0_C = ack_C && &scalar_data_C == 1'b1; //cancel all original bucket0, useless
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
zprize_so_pipe #(
    .so_or_sz    (0),
    .reset_value (0)
)affineRam_sz(
    .set_clr_onehot (free_affineValid),
    .valid_B1    (affine_valid_B1),
    .ready_B1    (affine_ready_B1),
    .index_B1    (affineIndex_B1),
    .clk        (clk),
    .rstN       (rstN)
);

assign point_addr_B1 = affineIndex_B1;

assign point_affine_ack_B1 = 
        affine_valid_B1 &&
        point_valid_B1 &&
        ready_B1 &&
        1;

assign affine_ready_B1 = point_affine_ack_B1;
assign point_ready_B1  = point_affine_ack_B1;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//affineIndexFifo
////////////////////////////////////////////////////////////////////////////////////////////////////////
//point_valid means point data from DDR
//affine_valid means affineRam has available item
flopOutSramFifo #(
    .DATA       (`LOG2((1<<9))),
    .DEPTH      ((1<<9))          //must equal to scalar_ram_depth, to align with scalar
    ) affineIndexFifo (
    .wDatValid      (point_affine_ack_B1),
    .wDatReady      (ready_B1),
    .wDat           (affineIndex_B1),
    .rDatValid      (point_affine_valid_C),
    .rDatReady      (point_affine_ready_C),
    .rDat           (affineIndex_C),
    .num            (affineIndexFifoNum),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
);
`assert_clk(affineIndexFifo.wDatReady == 1)     //affineIndexFifo 's depth is equal affineRAM, affineRam has credit, so affineIndex must has available item    
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
//free
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign free_affineValid_I = point_re_I ? (1 << point_raddr_I) : 0;
assign cancel_affineValid = cancel_bucket0_C ? (1 << affineIndex_C) : 0;

assign free_affineValid = cancel_affineValid | free_affineValid_I;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//affineRAM
////////////////////////////////////////////////////////////////////////////////////////////////////////
`Flop(point_valid_r, 1, point_valid_B1 && point_ready_B1)
`FlopN(point_addr_r, point_valid_B1 && point_ready_B1, point_addr_B1)
`FlopN(point_data_r, point_valid_B1 && point_ready_B1, point_data_B1)
ram_1r1w#(
    .NWORDS((1<<9)),      //must equal to scalar_ram_depth, to align with scalar
    .WORDSZ(1152),
    .WESZ  (1)
) affineRAM(
    .wclk (clk),
    .rclk (clk),
    .we   (point_valid_r),
    .wem  (1'b1),
    .waddr(point_addr_r),
    .wdata(point_data_r),
    .re   (point_re_I),
    .raddr(point_raddr_I),
    .rdata(point_rdata_K)
);

//`Flop(affineRamNum, point_valid_r != point_re_I, point_valid_r ? affineRamNum+1 : affineRamNum-1)
always@(posedge clk or negedge rstN)
        if(!rstN)
                affineRamNum <= `CQ 0;
        else
            case({point_valid_r,point_re_I,cancel_bucket0_C})
//             3'b000:     affineRamNum <= `CQ affineRamNum;
             3'b001:     affineRamNum <= `CQ affineRamNum - 1;
             3'b010:     affineRamNum <= `CQ affineRamNum - 1;
             3'b011:     affineRamNum <= `CQ affineRamNum - 2;
             3'b100:     affineRamNum <= `CQ affineRamNum + 1;
//             3'b101:     affineRamNum <= `CQ affineRamNum;
//             3'b110:     affineRamNum <= `CQ affineRamNum;
             3'b111:     affineRamNum <= `CQ affineRamNum - 1;
            default:    affineRamNum <= `CQ affineRamNum;
            endcase

assign affineCreditOk = affineRamNum < (1<<9) - pipeLatency;
`assert_clk(affineRamNum <=512)
////////////////////////////////////////////////////////////////////////////////////////////////////////
//accRAM
////////////////////////////////////////////////////////////////////////////////////////////////////////
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_0(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_0),
    .wem  (1'b1),
    .waddr(acc_waddr_S_0[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_0),
    .raddr(acc_raddr_G6_0[11:0]),
    .rdata(acc_rdata_K1_0)
);
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_1(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_1),
    .wem  (1'b1),
    .waddr(acc_waddr_S_1[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_1),
    .raddr(acc_raddr_G6_1[11:0]),
    .rdata(acc_rdata_K1_1)
);
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_2(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_2),
    .wem  (1'b1),
    .waddr(acc_waddr_S_2[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_2),
    .raddr(acc_raddr_G6_2[11:0]),
    .rdata(acc_rdata_K1_2)
);
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_3(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_3),
    .wem  (1'b1),
    .waddr(acc_waddr_S_3[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_3),
    .raddr(acc_raddr_G6_3[11:0]),
    .rdata(acc_rdata_K1_3)
);
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_4(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_4),
    .wem  (1'b1),
    .waddr(acc_waddr_S_4[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_4),
    .raddr(acc_raddr_G6_4[11:0]),
    .rdata(acc_rdata_K1_4)
);
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_5(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_5),
    .wem  (1'b1),
    .waddr(acc_waddr_S_5[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_5),
    .raddr(acc_raddr_G6_5[11:0]),
    .rdata(acc_rdata_K1_5)
);
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_6(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_6),
    .wem  (1'b1),
    .waddr(acc_waddr_S_6[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_6),
    .raddr(acc_raddr_G6_6[11:0]),
    .rdata(acc_rdata_K1_6)
);
ram_1r1w#(
    .FPGA_RAM_STYLE("ultra"),
    .NWORDS(4096),
    .WORDSZ(1536),
    .WESZ  (1)
) accRAM_7(
    .wclk (clk),
    .rclk (clk),
    .we   (acc_we_S_7),
    .wem  (1'b1),
    .waddr(acc_waddr_S_7[11:0]),
    .wdata(acc_wdata_S),
    .re   (acc_re_G6_7),
    .raddr(acc_raddr_G6_7[11:0]),
    .rdata(acc_rdata_K1_7)
);

//assign acc_we_S_0 = acc_we_S && acc_waddr_S[16-2:12] == 0;
//assign acc_re_G6_0 = acc_re_G6 && acc_raddr_G6[16-2:12] == 0;
//`Flop(acc_re_K1_0, 1, acc_re_G6_0)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)
//assign acc_we_S_1 = acc_we_S && acc_waddr_S[16-2:12] == 1;
//assign acc_re_G6_1 = acc_re_G6 && acc_raddr_G6[16-2:12] == 1;
//`Flop(acc_re_K1_1, 1, acc_re_G6_1)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)
//assign acc_we_S_2 = acc_we_S && acc_waddr_S[16-2:12] == 2;
//assign acc_re_G6_2 = acc_re_G6 && acc_raddr_G6[16-2:12] == 2;
//`Flop(acc_re_K1_2, 1, acc_re_G6_2)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)
//assign acc_we_S_3 = acc_we_S && acc_waddr_S[16-2:12] == 3;
//assign acc_re_G6_3 = acc_re_G6 && acc_raddr_G6[16-2:12] == 3;
//`Flop(acc_re_K1_3, 1, acc_re_G6_3)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)
//assign acc_we_S_4 = acc_we_S && acc_waddr_S[16-2:12] == 4;
//assign acc_re_G6_4 = acc_re_G6 && acc_raddr_G6[16-2:12] == 4;
//`Flop(acc_re_K1_4, 1, acc_re_G6_4)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)
//assign acc_we_S_5 = acc_we_S && acc_waddr_S[16-2:12] == 5;
//assign acc_re_G6_5 = acc_re_G6 && acc_raddr_G6[16-2:12] == 5;
//`Flop(acc_re_K1_5, 1, acc_re_G6_5)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)
//assign acc_we_S_6 = acc_we_S && acc_waddr_S[16-2:12] == 6;
//assign acc_re_G6_6 = acc_re_G6 && acc_raddr_G6[16-2:12] == 6;
//`Flop(acc_re_K1_6, 1, acc_re_G6_6)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)
//assign acc_we_S_7 = acc_we_S && acc_waddr_S[16-2:12] == 7;
//assign acc_re_G6_7 = acc_re_G6 && acc_raddr_G6[16-2:12] == 7;
//`Flop(acc_re_K1_7, 1, acc_re_G6_7)
`FlopN(acc_raddr_K1, acc_re_G6, acc_raddr_G6)

always@(posedge clk)
        case(acc_raddr_K1[16-2:12])
        0:          acc_rdata_K2 <= `CQ acc_rdata_K1_0;
        1:          acc_rdata_K2 <= `CQ acc_rdata_K1_1;
        2:          acc_rdata_K2 <= `CQ acc_rdata_K1_2;
        3:          acc_rdata_K2 <= `CQ acc_rdata_K1_3;
        4:          acc_rdata_K2 <= `CQ acc_rdata_K1_4;
        5:          acc_rdata_K2 <= `CQ acc_rdata_K1_5;
        6:          acc_rdata_K2 <= `CQ acc_rdata_K1_6;
        7:          acc_rdata_K2 <= `CQ acc_rdata_K1_7;
        default:    acc_rdata_K2 <= `CQ 'bx;
        endcase

 `PIPE(3,1536,acc_rdata_K,acc_rdata_K2)

////////////////////////////////////////////////////////////////////////////////////////////////////////
//calc_bucket
////////////////////////////////////////////////////////////////////////////////////////////////////////

zprize_calc_bucket #(
    .LAST_GROUP_REMAINDER (0)
) zprize_calc_bucket(
    .calc_bucket_idle   (),
//scalar                
    .valid_C            (valid_C            ),
    .ready_C            (calc_bucket_ready_C),
    .cancel_C           (cancel_bucket0_C   ),
    .scalar_C           (scalar_data_C      ),
    .affineIndex_C      (affineIndex_C      ),
//read pointRAM        
    .point_re_G6        (point_re_I    ),
    .point_raddr_G6     (point_raddr_I ),
    .point_rdata_H      (point_rdata_K ),
//read/write accRAM   
    .acc_we_r           (acc_we_S      ),
    .acc_waddr_r        (acc_waddr_S   ),
    .acc_wdata_r        (acc_wdata_S   ),
    .acc_re_r           (acc_re_G6     ),
    .acc_raddr_r        (acc_raddr_G6  ),
    .acc_rdata          (acc_rdata_K   ),
    .acc_re_r_0     (acc_re_G6_0   ),
    .acc_raddr_r_0  (acc_raddr_G6_0),
    .acc_we_r_0     (acc_we_S_0    ),
    .acc_waddr_r_0  (acc_waddr_S_0 ),
    .acc_re_r_1     (acc_re_G6_1   ),
    .acc_raddr_r_1  (acc_raddr_G6_1),
    .acc_we_r_1     (acc_we_S_1    ),
    .acc_waddr_r_1  (acc_waddr_S_1 ),
    .acc_re_r_2     (acc_re_G6_2   ),
    .acc_raddr_r_2  (acc_raddr_G6_2),
    .acc_we_r_2     (acc_we_S_2    ),
    .acc_waddr_r_2  (acc_waddr_S_2 ),
    .acc_re_r_3     (acc_re_G6_3   ),
    .acc_raddr_r_3  (acc_raddr_G6_3),
    .acc_we_r_3     (acc_we_S_3    ),
    .acc_waddr_r_3  (acc_waddr_S_3 ),
    .acc_re_r_4     (acc_re_G6_4   ),
    .acc_raddr_r_4  (acc_raddr_G6_4),
    .acc_we_r_4     (acc_we_S_4    ),
    .acc_waddr_r_4  (acc_waddr_S_4 ),
    .acc_re_r_5     (acc_re_G6_5   ),
    .acc_raddr_r_5  (acc_raddr_G6_5),
    .acc_we_r_5     (acc_we_S_5    ),
    .acc_waddr_r_5  (acc_waddr_S_5 ),
    .acc_re_r_6     (acc_re_G6_6   ),
    .acc_raddr_r_6  (acc_raddr_G6_6),
    .acc_we_r_6     (acc_we_S_6    ),
    .acc_waddr_r_6  (acc_waddr_S_6 ),
    .acc_re_r_7     (acc_re_G6_7   ),
    .acc_raddr_r_7  (acc_raddr_G6_7),
    .acc_we_r_7     (acc_we_S_7    ),
    .acc_waddr_r_7  (acc_waddr_S_7 ),
//output result
    .f2_result_valid    (f2_result_valid    ),
    .f2_result_ready    (f2_result_ready    ),
    .f2_result_data     (f2_result_data     ),
//Global
    .msm_num            (msm_num),
    .clk                (clk),
    .rstN               (rstN)
);

//for debug
    logic   [23:0]       point_num_in_calc_bucket;
    logic   [23:0]       point_num_G6;
`Flop(point_num_in_calc_bucket, valid_C && calc_bucket_ready_C != point_re_I, valid_C && calc_bucket_ready_C ? point_num_in_calc_bucket+1 : point_num_in_calc_bucket-1)
`Flop(point_num_G6, point_re_I , point_num_G6+1)

endmodule
