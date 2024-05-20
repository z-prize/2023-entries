`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

module zprize_calc_group import zprize_param::*; (
//control
    input                           start      ,
    output                          idle       ,        //after release F2_F3 pipeline, until output finish
    output                          release_S    ,      //release F2_F3 pipeline, so that calc_bucket can restart again
//read accRAM
    output                          acc_re     ,
    output[`LOG2((1<<(16-1)))-1:0]acc_raddr  ,
//to F2
    `MAX_FANOUT32 output reg                      valid_i    ,        //input to F2
    output[1536-1:0]      data0_i    ,        //tmp
    output                          data1_sel_i,        //0: sel sum, 1: sel accRAM
    output[1536-1:0]      data1_i    ,        //sum
//from F2
    input                           valid_o    ,        //output from F2
    input [1536-1:0]      data_o     ,
//output result
    output reg                      result_valid,
    input                           result_ready,
    output reg [1536-1:0] result_data ,
//Globcal
    input                           clk        ,
    input                           rstN       
);

//WORK pipeline
//A: read accRAM
//B: read tmpsumRAM
//C: tmpsumRAM return, bad timing, latch
//D: accRAM return, good timing, don latch. MUX acc_rdata and sum to F2. tmp to F2
//S: F2 return, write to tmpsumRAM

//OUTPUT pipeline
//A; nothing
//B: read tmpsumRAM
//C: tmpsumRAM return, bad timing latch
//result(NOT D!!!): output result

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
//for F2 data input
//sum=tmp    +  sum
//tmp=tmp    +  acc
//     ^         ^      
//    data0    data1   

localparam  F2_IDLE = 0;
localparam  F2_WORK = 1;
localparam  F2_WAIT_FINISH = 2;
localparam  F2_OUTPUT = 3;

    logic [2:0]                     state_A;
    logic                           valid_A;
    logic                           ready_A;
    logic                           finish_A;
    logic   [1:0]                   phase_A;        //when state == WORK 
                                                    //0: read sum
                                                    //1: read tmp, calculate sum=sum+tmp, writeback sum
                                                    //2: read accRAM
                                                    //3: read tmp, calculate tmp=acc+tmp, writeback tmp
                                                    //when stat == OUTPUT
                                                    //0: output sum
                                                    //1: output tmp
    logic   [`LOG2(128)-1:0]       section_A;
    logic   [`LOG2((1<<(16-1))/128)-1:0] bucket_in_section_A;
    logic                          release_A    ;      //begin output

    logic                           valid_B;
    logic                           ready_B;        //ready_B only useful for OUTPUT
    logic [2:0]                     state_B;
    logic   [1:0]                   phase_B;        
    logic   [`LOG2(128)-1:0]       section_B;
    logic   [`LOG2((1<<(16-1))/128)-1:0] bucket_in_section_B;
    logic                                   tmpsum_re_B;
    logic   [`LOG2(128*2)-1:0]     tmpsum_raddr_B;     //address[0]==0: sum, address[1]==1: tmp.
                                                                    //high bit: section
    logic [1536-1:0]      tmpsum_rdata_C; //rdata return at pipeline C

    `MAX_FANOUT32 logic                           valid_C;
    logic                           ready_C;
    `MAX_FANOUT32 logic [2:0]                     state_C;
    logic   [1:0]                   phase_C;
    logic   [`LOG2(128)-1:0]       section_C;
    logic   [`LOG2((1<<(16-1))/128)-1:0] bucket_in_section_C;
    logic                                   bucket_in_section_last_C;

    `MAX_FANOUT32 logic                           valid_D;
    `MAX_FANOUT32 logic   [1:0]                   phase_D;
    logic   [`LOG2(128)-1:0]       section_D;
    logic   [`LOG2((1<<(16-1))/128)-1:0] bucket_in_section_D;
    `MAX_FANOUT32 logic                                   bucket_in_section_last_D;

    logic [1536-1:0]      tmpsum_rdata_D; 
    logic [1536-1:0]      tmpsum_rdata_latch_D; 
    logic [1536-1:0]      real_tmpsum_rdata_D; 

    logic                           valid_S;
    logic   [1:0]                   phase_S;
    logic   [`LOG2(128)-1:0]       section_S;
    logic   [`LOG2((1<<(16-1))/128)-1:0] bucket_in_section_S;

    logic                                   tmpsum_we_S;
    logic   [`LOG2(128*2)-1:0]      tmpsum_waddr_S;     
    logic [1536-1:0]              tmpsum_wdata_S; 

    logic                           valid_AB_idle;
    logic                           valid_DS_idle;

    logic                           pipeline_idle;      //for wait finish
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign idle = state_A == F2_IDLE &&
                pipeline_idle &&
              1;

assign pipeline_idle = 
              valid_AB_idle &&
              !valid_C &&
              !valid_D &&
              valid_DS_idle && 
              1;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//state
////////////////////////////////////////////////////////////////////////////////////////////////////////
    always@(posedge clk or negedge rstN)
        if(!rstN)
            state_A <= `CQ F2_IDLE;
        else 
            case(state_A)
            F2_IDLE:
                if(start)
                    state_A <= `CQ F2_WORK;
            F2_WORK:
                if(release_A)
                    state_A <= `CQ F2_WAIT_FINISH;
            F2_WAIT_FINISH:
                if(pipeline_idle)       //wait pipeline idle
                    state_A <= `CQ F2_OUTPUT;
            F2_OUTPUT:
                if(finish_A) 
                    state_A <= `CQ F2_IDLE; 
            endcase 
//////////////////////////////////////////////////////////////////////////////////////////////////////// 
//A 
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign valid_A      =   state_A == F2_WORK || state_A == F2_OUTPUT;
assign release_A    =   state_A == F2_WORK &&
                        phase_A == 3 && 
                        section_A == 128 - 1 && 
                        bucket_in_section_A == 0;

assign finish_A     =   state_A == F2_OUTPUT && phase_A == 1 && section_A == 128 - 1 && ready_A;                

//combine WORK and OUTPUT, so that generate raddr of tmpsumRAM
`Flop(phase_A,              state_A == F2_WORK || ready_A && state_A == F2_OUTPUT, state_A == F2_WORK ? phase_A + 1 : phase_A == 1 ? 0 : phase_A + 1)
`Flop(section_A,            state_A == F2_WORK && phase_A == 3 || ready_A && state_A == F2_OUTPUT && phase_A == 1, section_A == 128 - 1 ? 0 : section_A + 1)
`FlopAll(bucket_in_section_A,  state_A == F2_WORK && phase_A == 3 && section_A == 128 - 1, bucket_in_section_A == 0 ? (1<<(16-1))/128 - 1 : bucket_in_section_A - 1, clk, rstN, (1<<(16-1))/128 - 1)

//accRAM's read latency maybe larger than tmpsumRAM, so actually read tmpsum occur after pipeline A
assign acc_re           = valid_A && phase_A == 2;
assign acc_raddr        = {section_A,bucket_in_section_A};

assign ready_A = valid_B ? ready_B : 1;//for output
////////////////////////////////////////////////////////////////////////////////////////////////////////
//B
////////////////////////////////////////////////////////////////////////////////////////////////////////

validpipe #(
    .CYCLES (3   )      //AB_lateceny = accRAM_latency - 2
)validpipe_AB(
	.din    (valid_A),
	.dout   (valid_B),
    .enable (ready_A),
    .idle   (valid_AB_idle),
	.clk    (clk),
    .rstN   (rstN) 
);
hyperpipe #(
    .CYCLES (3   ),
    .WIDTH  ($bits({state_A,phase_A,section_A,bucket_in_section_A}))
)hyperpipe_AC(
    .enable (ready_A),
	.din    ({state_A,phase_A,section_A,bucket_in_section_A}),
	.dout   ({state_B,phase_B,section_B,bucket_in_section_B}),
	.clk    (clk   )
);

assign tmpsum_re_B      = valid_B && 
                            (state_B == F2_WORK && (phase_B == 0 || phase_B == 1 || phase_B == 3) || 
                             state_B == F2_OUTPUT && ready_B);
assign tmpsum_raddr_B   = {section_B,phase_B[0]};
assign ready_B = valid_C ? ready_C : 1;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//tmpsumRAM
////////////////////////////////////////////////////////////////////////////////////////////////////////
ram_1r1w#(
    .NWORDS(128*2),
    .WORDSZ(1536),
    .WESZ  (1)
) tmpsumRAM (
    .wclk (clk),
    .rclk (clk),
    .we   (tmpsum_we_S),
    .wem  (1'b1),
    .waddr(tmpsum_waddr_S),
    .wdata(tmpsum_wdata_S),
    .re   (tmpsum_re_B),
    .raddr(tmpsum_raddr_B),
    .rdata(tmpsum_rdata_C)
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//C
////////////////////////////////////////////////////////////////////////////////////////////////////////

`FlopSC0(valid_C, valid_B && (state_B == F2_OUTPUT ? ready_B : 1), valid_C && (state_C == F2_OUTPUT ? ready_C : 1))
`FlopN({state_C,phase_C,section_C,bucket_in_section_C}, valid_B && (state_B == F2_OUTPUT ? ready_B : 1), {state_B,phase_B,section_B,bucket_in_section_B})

assign bucket_in_section_last_C = bucket_in_section_C == (1<<(16-1))/128 - 1;

assign ready_C = result_valid ? result_ready : 1;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//OUTPUT result
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopSC0(result_valid, valid_C && state_C == F2_OUTPUT && ready_C, result_valid && result_ready)
`FlopN(result_data   , valid_C && state_C == F2_OUTPUT && ready_C, tmpsum_rdata_C)

logic [8:0]     res_cnt;
`Flop(res_cnt, result_valid && result_ready, res_cnt + 1)

////////////////////////////////////////////////////////////////////////////////////////////////////////
//D
////////////////////////////////////////////////////////////////////////////////////////////////////////
`Flop(valid_D, 1, valid_C && state_C == F2_WORK)
`FlopN({phase_D,section_D,bucket_in_section_D,bucket_in_section_last_D}, valid_C && state_C == F2_WORK, {phase_C,section_C,bucket_in_section_C,bucket_in_section_last_C})
`FlopN(tmpsum_rdata_D, valid_C && state_C == F2_WORK, tmpsum_rdata_C)

assign real_tmpsum_rdata_D = bucket_in_section_last_D ? ZPRIZE_ACCRAM_INI : tmpsum_rdata_D;   //if first section, sum/tmp are 0

//send data to F2 at pipeline D
//sum is at phase0, so latch one cycle to sync with tmp at phase1
`FlopN(tmpsum_rdata_latch_D, valid_D && phase_D == 0, real_tmpsum_rdata_D)          

`Flop(valid_i, 1, valid_C && state_C == F2_WORK && (phase_C == 1 || phase_C == 3))     //flopout
//assign valid_i = valid_D && (phase_D == 1 || phase_D == 3);    //phase 1 and phase 3 calculate
assign data0_i = real_tmpsum_rdata_D;        //tmp
assign data1_sel_i = phase_D[1];        //phase == 1: sel sum. phase == 3: sel accRAM
assign data1_i = tmpsum_rdata_latch_D;  //sum

////////////////////////////////////////////////////////////////////////////////////////////////////////
//F2 output at pipeline S
////////////////////////////////////////////////////////////////////////////////////////////////////////
validpipe #(
    .CYCLES (39*2+((7+1+1)-1+(7+1+1)+1+1)*2+4+39+1)
)validpipe_DS(
	.din    (valid_D),
	.dout   (valid_S),
    .enable (1'b1),
    .idle   (valid_DS_idle),
	.clk    (clk),
    .rstN   (rstN) 
);

//reconstruct phase/section/bucket_in_section
`Flop(phase_S,              valid_S, phase_S + 1)
`Flop(section_S,            valid_S && phase_S == 3, section_S == 128 - 1 ? 0 : section_S + 1)
`FlopAll(bucket_in_section_S,  valid_S && phase_S == 3 && section_S == 128 - 1, bucket_in_section_S == 0 ? (1<<(16-1))/128 - 1 : bucket_in_section_S - 1, clk, rstN, (1<<(16-1))/128 - 1)

assign release_S    =   valid_S &&
                        phase_S == 3 && 
                        section_S == 128 - 1 && 
                        bucket_in_section_S == 0;

assign tmpsum_we_S = valid_S && (phase_S == 1 || phase_S == 3);
assign tmpsum_waddr_S = {section_S,phase_S[1]};
assign tmpsum_wdata_S = data_o;

`assert_clk(tmpsum_we_S == valid_o)

endmodule
