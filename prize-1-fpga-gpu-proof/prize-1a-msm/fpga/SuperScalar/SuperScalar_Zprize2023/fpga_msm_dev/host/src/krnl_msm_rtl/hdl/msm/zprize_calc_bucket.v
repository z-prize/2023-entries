`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

////////////////////////////////////////////////////////////////////////////////////////////////////////
//C     --  input (scalar)
//D     --  fifo, buffer between C~D
//E     --  
//E1    --  judge the last group and add sub_bucket by N_count
//F     --  backup buffer
//F1-G6 --  bucketControl, choose selectable bucket to F3
//   G6 --  accRAM_re_request, clear accValid
//
//I     --  pointRAM_re_request
//K     --  accRAM_rData & pointRAM_rData
//
//P     --  padd_F3 input
//S     --  padd_F3 output
//          write into accRAM according to group & bucket
//          set accValid
////////////////////////////////////////////////////////////////////////////////////////////////////////

module zprize_calc_bucket import zprize_param::*;
#(
    parameter LAST_GROUP_REMAINDER = 1
) (
    output                                  calc_bucket_idle,        //should remain high until valid_S && ready_S, should be 0 at the first cycle of valid_S
//scalar
    input                                   valid_C,
    output                                  ready_C,
    input                                   cancel_C,
    input   [16-1:0]                   scalar_C,
    input   [`LOG2(512)-1:0]     affineIndex_C,
//read pointRAM
    output                                    point_re_G6,
    output  [`LOG2(512)-1:0]      point_raddr_G6,
    input   [1152-1:0]                   point_rdata_H,
//read/write accRAM 
    output logic                                    acc_we_r,
    output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r,
    output logic[1536-1:0]                acc_wdata_r,
    output logic                                    acc_re_r,
    output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r,
    output logic                                    acc_re_r_0,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_0,
    output logic                                    acc_we_r_0,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_0,
    output logic                                    acc_re_r_1,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_1,
    output logic                                    acc_we_r_1,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_1,
    output logic                                    acc_re_r_2,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_2,
    output logic                                    acc_we_r_2,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_2,
    output logic                                    acc_re_r_3,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_3,
    output logic                                    acc_we_r_3,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_3,
    output logic                                    acc_re_r_4,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_4,
    output logic                                    acc_we_r_4,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_4,
    output logic                                    acc_re_r_5,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_5,
    output logic                                    acc_we_r_5,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_5,
    output logic                                    acc_re_r_6,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_6,
    output logic                                    acc_we_r_6,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_6,
    output logic                                    acc_re_r_7,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_raddr_r_7,
    output logic                                    acc_we_r_7,
    `KEEP output logic[`LOG2((1<<(16-1)))-1:0]      acc_waddr_r_7,
    input [1536-1:0]                acc_rdata,
//output result
    output                                   f2_result_valid,
    input                                  f2_result_ready,
    output              [1536-1:0] f2_result_data     ,
//Global
    input   [`LOG2((1<<24))-1:0] msm_num,
    input           clk,
    input           rstN

);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    genvar                                  i;

    wire                                    idle;                       //whole module idle
    logic                                   calc_bucket_start_C;
    `MAX_FANOUT32 reg   [4:0]                             state_C;                      //state of calc_bucket of one group

    logic   [`LOG2((1<<(16-1)))-1:0]      init_count;                 //init bucketRAM to 0

    logic                                   allow_C;
    logic                                   scalar_buffer_fifo_wvalid_C;
    logic                                   scalar_buffer_fifo_wready_C;

    logic                                   group_finish_C;               

    logic   [`LOG2(512)-1:0]     affineIndex_D;
    logic   [`LOG2((1<<24))-1:0]          N_count_D;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucket_D;
    logic                               valid_D;
    logic                               ready_D;

    logic   [`LOG2(512)-1:0]     affineIndex_E;
    logic   [`LOG2(512)-1:0]     affineIndex_E1;
    logic   [`LOG2(512)-1:0]     affineIndex_F;
    logic   [`LOG2(512)-1:0]     affineIndex_F1;
    logic   [`LOG2(512)-1:0]     affineIndex_F1_t;

    reg   [`LOG2((1<<24))-1:0]                N_count_C;                        
    logic [`LOG2((1<<24))-1:0]                N_count_E;                        
    reg   [`LOG2((1<<24))-1:0]                N_count_E1;                        
    logic [`LOG2((1<<24))-1:0]                N_count_F;                        
    logic [`LOG2((1<<24))-1:0]                N_count_F1;                       
    reg   [`LOG2((1<<24))-1:0]                N_count_K;                        
    wire  [`LOG2((1<<24))-1:0]                N_count_P;                        
    wire  [`LOG2((1<<24))-1:0]                N_count_S;                        

    
    wire                                valid_P_idle;

    logic [31:0]    group_count_bucket;
    logic                           signed_flag_C;
    logic                           signed_flag_D;
    logic                           signed_flag_E;
    logic                           signed_flag_E1;
    logic                           signed_flag_F;
    logic                           signed_flag_F1_t;
    logic                           signed_flag_G6;
    logic                           signed_flag_H;

    reg                                     valid_G4;
    logic [`LOG2((1<<(16-1)))-1:0]        bucket_G4;
    logic [`LOG2((1<<24))-1:0]                N_count_G4;                        //0~16383
    logic   [`LOG2(512)-1:0]     affineIndex_G4;

    reg                                     valid_G5;
    logic [`LOG2((1<<(16-1)))-1:0]        bucket_G5;
    logic [`LOG2((1<<24))-1:0]                N_count_G5;                        //0~16383
    logic   [`LOG2(512)-1:0]     affineIndex_G5;

    logic                                    valid_G6;
    logic [`LOG2((1<<(16-1)))-1:0]        bucket_G6;
    logic [`LOG2((1<<24))-1:0]                N_count_G6;                        //0~16383
    logic   [`LOG2(512)-1:0]     affineIndex_G6;

    logic [`LOG2((1<<(16-1)))-1:0]        bucket_K;
    wire  [`LOG2((1<<(16-1)))-1:0]        bucket_P;
    wire  [`LOG2((1<<(16-1)))-1:0]        bucket_S;
    wire  [`LOG2((1<<(16-1)))-1:0]        real_bucket_S;

    logic                                   valid_E;
    logic                                   valid_E1;
    logic                                   valid_F;
    logic                                   valid_F1;

    wire                                    ready_E;
    logic                                   ready_E1;
    logic                                   ready_F;
    logic                                   ready_F1;
    //asic_bucketControl
    logic                                   valid_F1_t;
    logic                                   ready_F1_t;
    logic [`LOG2((1<<(16-1)))-1:0]        bucket_F1_t;
    logic [`LOG2((1<<24))-1:0]                N_count_F1_t;                       

    reg                                     valid_K;

    wire [`LOG2((1<<(16-1)))-1:0]         last_group_remainder_E;
    wire [`LOG2((1<<(16-1)))-1:0]         div_sub_bucket_valid_E;

    reg   [(1<<(16-1))-1:0]                accValid;     //group's accValid

    wire  [(1<<(16-1))-1:0]                next_accValid;   //accValid's next cycle value

    wire  [`LOG2((1<<(16-1)))-1:0]        bucket_C;

    logic [`LOG2((1<<(16-1)))-1:0]        bucket_E;
    logic [`LOG2((1<<(16-1)))-1:0]        bucket_E1;
    logic [`LOG2((1<<(16-1)))-1:0]        bucket_F;
    logic [`LOG2((1<<(16-1)))-1:0]        bucket_F1;

    wire                                    valid_P;
    wire  [1536-1:0]                acc_rdata_P;
    wire  [1152-1:0]              point_rdata_P;

    wire                                    valid_S;
    wire  [1536-1:0]                data_S;

    wire                                    PADD_f2_f3_idle;
    reg   [4:0]                             PADD_f3_idle_count;
    reg                                     true_PADD_f3_idle;

    logic                           mode;
    logic                           f2_f3_valid_o;

    logic                           f2_start      ;
    logic                           f2_release    ;
    logic                           calc_group_idle       ;
//read accRAM
    logic                           f2_acc_re     ;
    logic [`LOG2((1<<(16-1)))-1:0]f2_acc_raddr  ;
//to F2
    logic                           f2_valid_i    ;        //F2's iput
    logic [1536-1:0]      f2_data0_i    ;        //tmp
    logic [1536-1:0]      f2_data1_i    ;        //sum
    logic [1536-1:0]      real_f2_data1_i    ;   //sum and acc
    logic                           f2_data1_sel_i;        //0: sel sum, 1: sel accRAM
//from F2
    logic                           f2_valid_o    ;        //F2's output
    logic [1536-1:0]      f2_data_o     ;

//logic  result

`ifndef FPGA
    reg   [31:0]                            statistic_point_num_in_bucket[(1<<(16-1))-1:0];
`endif
    wire                                    bucketControl_idle;
    wire                                    valid_K_idle;

    logic [1152-1:0]                   point_rdata_K;
    logic                                   acc_re_I;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr_I;
    logic [1536-1:0]              acc_rdata_K;
    logic                                   acc_we_S;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr_S;
    logic [1536-1:0]              acc_wdata_S;
    logic                                   acc_re;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_raddr;
    logic                                   acc_we;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_waddr;
    logic [1536-1:0]              acc_wdata;

    logic                                   acc_flush_en;
    logic [`LOG2((1<<(16-1)))-1:0]    acc_flush_addr;

    logic                                   valid_H;
    logic                                   idle_HK;
    logic [`LOG2((1<<(16-1)))-1:0]    bucket_H;
    logic [`LOG2((1<<24))-1:0]              N_count_H;

    logic                                   valid_I;
    logic [`LOG2((1<<(16-1)))-1:0]    bucket_I;
    logic [`LOG2((1<<24))-1:0]              N_count_I;                        
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
//group control
////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////
//idle
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign idle =   
                !valid_C &&                 //
                !valid_D &&                 //
                !valid_E &&                 //
                !valid_E1 &&                //
                !valid_F &&                 //
                bucketControl_idle &&       //F1_t-G6
                !valid_H &&                 //H
                idle_HK &&                  //H-K
                valid_P_idle &&             //K-P
                PADD_f2_f3_idle &&          //P-S
//                calc_group_idle &&
                1;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//state
////////////////////////////////////////////////////////////////////////////////////////////////////////
    always@(posedge clk or negedge rstN)
        if(!rstN)
            state_C <= `CQ INIT;
        else 
            case(state_C)
            INIT:
                if(init_count == (1<<(16-1))-1)
                    state_C <= `CQ READ_SCALAR_POINT;
            READ_SCALAR_POINT:
                if(group_finish_C)
                    state_C <= `CQ WAIT_FINISH;
            WAIT_FINISH:
                    if(idle && calc_group_idle)     //calc_group_idle means previous calc_group finish, because we only check f2_reease at the begin of READ_SCALAR_POINT, so we must check calc_group_idle before new calc_group
                    state_C <= `CQ CALC_GROUP;
            CALC_GROUP:
                if(f2_release) begin        //only check f2_reease, to restart READ_SCALAR_POINT early.
                    state_C <= `CQ READ_SCALAR_POINT;
                end
            endcase
`Flop(init_count, state_C[INITi], init_count + 1)

assign calc_bucket_idle = state_C[IDLEi];

assign allow_C = state_C[READ_SCALAR_POINTi];
////////////////////////////////////////////////////////////////////////////////////////////////////////
//scalar input
////////////////////////////////////////////////////////////////////////////////////////////////////////
`assert_clk(valid_C |-> scalar_C !== 'bx)
assign bucket_C      = scalar_C[16-2:0];
assign signed_flag_C = scalar_C[16-1];
`Flop(N_count_C, group_finish_C || (valid_C || cancel_C) && ready_C, group_finish_C ? 0 : N_count_C+1)
assign group_finish_C = valid_C && ready_C && N_count_C == msm_num;

`Flop(group_count_bucket, f2_start, group_count_bucket + 1'b1)
`ifndef FPGA
generate 
    for(i=0;i<(1<<(16-1));i=i+1) begin: b
`Flop(statistic_point_num_in_bucket[i], valid_C && ready_C && bucket_C == i, statistic_point_num_in_bucket[i] + 1)
    end
endgenerate
`endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
//D
////////////////////////////////////////////////////////////////////////////////////////////////////////
/////fifo,buffer,C~D
assign scalar_buffer_fifo_wvalid_C = valid_C && allow_C;
assign ready_C = scalar_buffer_fifo_wready_C && allow_C;

flopOut2Fifo #(
    .DATA       (`LOG2(512)+`LOG2((1<<24))+`LOG2((1<<(16-1)))+1),
    .DEPTH      (32)
    ) scalar_buffer_fifo(
    .wDatValid  (scalar_buffer_fifo_wvalid_C),
    .wDatReady  (scalar_buffer_fifo_wready_C),
    .wDatAlmostFull(),
    .wDat       ({affineIndex_C,signed_flag_C,bucket_C,N_count_C}),
    .rDatValid  (valid_D),
    .rDatReady  (ready_D),
    .rDat       ({affineIndex_D,signed_flag_D,bucket_D,N_count_D}),
    .num        (),
    .clk        (clk),
    .rstN       (rstN),
    .clear      (1'b0)
    );
assign ready_D = valid_E ? ready_E : 1;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//E
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopSC0(valid_E, valid_D && ready_D, valid_E && ready_E)
`FlopN({affineIndex_E,N_count_E,signed_flag_E,bucket_E}, valid_D && ready_D, {affineIndex_D,N_count_D,signed_flag_D,bucket_D})

assign ready_E = valid_E1 ? ready_E1 : 1;

assign last_group_remainder_E = LAST_GROUP_REMAINDER;
assign div_sub_bucket_valid_E = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline E1
//div sub_bucket--improve the last group
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopSC0(valid_E1, valid_E && ready_E                                                    , valid_E1 && ready_E1)

`FlopN({affineIndex_E1,N_count_E1,signed_flag_E1}, valid_E && ready_E, {affineIndex_E,N_count_E,signed_flag_E})

always@(posedge clk)
        if(valid_E && ready_E && div_sub_bucket_valid_E == 0)
            bucket_E1 <= `CQ bucket_E;
        else if(valid_E && ready_E && div_sub_bucket_valid_E == 1)
            bucket_E1 <= `CQ N_count_E[`LOG2((1<<(16-1)))-1:0];

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline F
//with backup buffer 
////////////////////////////////////////////////////////////////////////////////////////////////////////
backupPipe #(
    .WIDTH(`LOG2(512)+`LOG2((1<<24))+`LOG2((1<<(16-1))) + 1 )
)F_backuppipe(
    .valid_i            (valid_E1),
    .ready_i            (ready_E1),
    .data_i             ({affineIndex_E1,N_count_E1,signed_flag_E1,bucket_E1}),
    .valid_o            (valid_F),
    .ready_o            (ready_F),
    .data_o             ({affineIndex_F,N_count_F,signed_flag_F,bucket_F}),
    .clk                (clk),
    .rstN               (rstN)
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//pipeline F1
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopSC0(valid_F1_t, valid_F && ready_F, valid_F1_t && ready_F1_t)
`FlopN({affineIndex_F1_t,N_count_F1_t,signed_flag_F1_t,bucket_F1_t}, valid_F && ready_F, {affineIndex_F,N_count_F,signed_flag_F,bucket_F})
assign ready_F             = valid_F1_t ?  ready_F1_t : 1;

//
////////////////////////////////////////////////////////////////////////////////////////////////////////
//bucketControl
//F1
//F2
//F3
//G
//G1
//G2
//G3
//G4
//G5
//G6
////////////////////////////////////////////////////////////////////////////////////////////////////////
zprize_new_bucketControl bucketControl(
    .valid_F1_t                  (valid_F1_t                  ),
    .ready_F1_t                  (ready_F1_t ),
    .N_count_F1_t                (N_count_F1_t                ),
    .bucket_F1_t                 (bucket_F1_t         ),
    .signed_flag_F1_t            (signed_flag_F1_t    ),
    .affineIndex_F1_t           (affineIndex_F1_t),
    .bucket_L                 (bucket_G6                     ),
    .valid_L                  (valid_G6                              ),
    .signed_flag_L            (signed_flag_G6    ),
    .N_count_L                (N_count_G6                            ),
    .affineIndex_L            (affineIndex_G6             ),
    .valid_S                    (valid_S),
    .bucket_S                   (bucket_S),
    .real_bucket_S               (real_bucket_S),
    .state_C                   (state_C                   ),
    .init_count                (init_count              ),
    .bucketControl_idle        (bucketControl_idle  ),
	.clk    (clk   ),
    .rstN   (rstN)
);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//pointRAM read
assign point_re_G6    = valid_G6;
assign point_raddr_G6 = affineIndex_G6;

validpipe #(
    .CYCLES (1),
    .WIDTH  (1)
)validpipe_G6H(
    .enable (1'b1),
	.din    (valid_G6),
	.dout   (valid_H),
    .idle   (),
	.clk    (clk),
    .rstN   (rstN)
);
`Flop(signed_flag_H, 1, signed_flag_G6)

hyperpipe #(
    .CYCLES (1   ),
    .WIDTH  (`LOG2((1<<24))+`LOG2((1<<(16-1))))
)hyperpipe_G6H(
    .enable (1'b1   ),
	.din    ({N_count_G6,bucket_G6}),
	.dout   ({N_count_H,bucket_H}),
	.clk    (clk   )
);

zprize_neg_affine zprize_neg_affine(
    .valid_in           (valid_H       ),
    .neg_flag_in        (signed_flag_H ),
    .affine_in          (point_rdata_H ),
    .valid_out          (valid_K       ),
    .neg_flag_out       (signed_flag_K ),
    .affine_out         (point_rdata_K ),
    .idle               (idle_HK       ),
	.clk    (clk   ),
    .rstN   (rstN)
);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//accRAM read
    
validpipe #(
    .CYCLES (2),
    .WIDTH  (1)
)validpipe_HI(
    .enable (1'b1),
	.din    (valid_H),
	.dout   (valid_I),
    .idle   (),
	.clk    (clk),
    .rstN   (rstN)
);

hyperpipe #(
    .CYCLES (2),
    .WIDTH  (`LOG2((1<<24))+`LOG2((1<<(16-1))))
)hyperpipe_HI(
    .enable (1'b1   ),
	.din    ({N_count_H,bucket_H}),
	.dout   ({N_count_I,bucket_I}),
	.clk    (clk   )
);

assign acc_re_I    = valid_I;
assign acc_raddr_I = bucket_I;

`ifndef NEW_BUCKETCONTROL
////////////////////////////////////////////////////////////////////////////////////////////////////////
//accValid, record the item in accRAM is valid
//when current group finish, set accValid to all 1
//when projective point value writed into accRAM, set the item to 1, according to it's bucket num
//when accRAM read out, clear the item to 0, according to it's bucket num
////////////////////////////////////////////////////////////////////////////////////////////////////////
generate 
    for(i=0;i<(1<<(16-1));i=i+1) begin: a
    always@(posedge clk or negedge rstN)
        if(!rstN)
            accValid[i] <= `CQ 1;
        else if(valid_G6 && bucket_G6 == i)
            accValid[i] <= `CQ 0;
        else if(valid_S && acc_waddr_S[`LOG2((1<<(16-1)))-1:0] == i)
            accValid[i] <= `CQ 1;

    //next_accValid is used to generate bucketOK, so don't care initialize 
    //don't care group_finish to clear
    assign next_accValid[i] = 
                valid_G6 && bucket_G6 == i ? 0 :
                valid_S && acc_waddr_S[`LOG2((1<<(16-1)))-1:0] == i ? 1 :
                accValid[i];

    end
endgenerate

//fake positives
//`assert_clk(valid_G6 |-> accValid[bucket_G6] == 1)
`assert_clk(valid_S |-> accValid[acc_waddr_S[`LOG2((1<<(16-1)))-1:0]] == 0)
`endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
//accRAM write
////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign acc_we_S       = valid_S;

    assign acc_waddr_S    = real_bucket_S;
    assign acc_wdata_S    = data_S;
    //assign acc_wdata_S    = state_C[INITi] ? 'haaaaaaaaaaaaaaaaa55555555555555aaaaaaaaaa555555555: data_S;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Point ADD input
////////////////////////////////////////////////////////////////////////////////////////////////////////

hyperpipe #(
    .CYCLES (6   ),
    .WIDTH  (`LOG2((1<<24))+`LOG2((1<<(16-1))))
)hyperpipe_IK(
    .enable (1'b1   ),
	.din    ({N_count_I,bucket_I}),
	.dout   ({N_count_K,bucket_K}),
	.clk    (clk   )
);

hyperpipe #(
    .CYCLES (1   ),
    .WIDTH  (1152)
)hyperpipe_K1P(
    .enable (1'b1   ),
	.din    (point_rdata_K),
	.dout   (point_rdata_P),
	.clk    (clk   )
);

//delay 2 cycles for timing
validpipe #(
    .CYCLES (1   )
)validpipe_valid_KP(
	.din    (valid_K),
	.dout   (valid_P),
    .enable (1'b1),
    .idle   (valid_P_idle),
	.clk    (clk),
    .rstN   (rstN) 
);

hyperpipe #(
    .CYCLES (1   ),
    .WIDTH  (1536+`LOG2((1<<24))+`LOG2((1<<(16-1))))
)hyperpipe_KP(
    .enable (1'b1   ),
	.din    ({N_count_K,bucket_K,acc_rdata_K}),
	.dout   ({N_count_P,bucket_P,acc_rdata_P}),
	.clk    (clk   )
);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Point ADD 
////////////////////////////////////////////////////////////////////////////////////////////////////////
//exed_padd_f3 padd_f3(

assign mode = state_C[CALC_GROUPi];
assign real_f2_data1_i = f2_data1_sel_i == 1 ? acc_rdata_K : f2_data1_i;

`ifdef FAKE_F2_F3
        assign PADD_f2_f3_idle = 1;
fake_zprize_F2_F3_combine f2_f3(
`else
zprize_F2_F3_combine f2_f3(
`endif
    .mode       (mode),
//input
//f2
    .f2_in_valid  (f2_valid_i ),
    .f2_in_data1  (f2_data0_i ),        //rename 0 to 1
    .f2_in_data2  (real_f2_data1_i ),   //rename 1 to 2
//f3
    .f3_in_valid  (valid_P ),           
    .f3_in_data1  (acc_rdata_P ),           
    .f3_in_data2  (point_rdata_P ),           
//output             
    .out_valid (f2_f3_valid_o),
    .f2_out_valid(f2_valid_o),
    .f3_out_valid(valid_S),
    .out_data  (data_S ),
//Global            
    .idle           (PADD_f2_f3_idle  ),
    .clk            (clk           ),
    .rstN           (rstN          )
);

//assign valid_S = mode == 0 && f2_f3_valid_o;
//assign f2_valid_o = mode == 1 && f2_f3_valid_o;
assign f2_data_o = data_S;

//delay register according to padd_f3
hyperpipe #(
`ifdef FAKE_F2_F3
    .CYCLES (1   ),
`else
    .CYCLES (39*2+((7+1+1)-1+(7+1+1)+1+1)*2+4   ),
`endif
    .WIDTH  (`LOG2((1<<24))+`LOG2((1<<(16-1))))
)hyperpipe_padd_f3(
    .enable (1'b1   ),
	.din    ({N_count_P,bucket_P}),
	.dout   ({N_count_S,bucket_S}),
	.clk    (clk   )
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//calc_group
////////////////////////////////////////////////////////////////////////////////////////////////////////
//pulse start
`FlopSC0(f2_start, state_C[WAIT_FINISHi] && idle, f2_start)

zprize_calc_group zprize_calc_group(
    .start      (f2_start),
    .release_S  (f2_release),       //release accRAM
    .idle       (calc_group_idle),
    .acc_re     (f2_acc_re),
    .acc_raddr  (f2_acc_raddr),
    .valid_i    (f2_valid_i),
    .data0_i    (f2_data0_i),
    .data1_i    (f2_data1_i),
    .data1_sel_i(f2_data1_sel_i),
    .valid_o    (f2_valid_o),
    .data_o     (f2_data_o),
    .result_valid(f2_result_valid),
    .result_ready(f2_result_ready),
    .result_data (f2_result_data),
    .clk        (clk),
    .rstN       (rstN)
);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//flush_accRAM
////////////////////////////////////////////////////////////////////////////////////////////////////////
// flush all buckets 2 cycles after f2_acc_re for next group's calc_bucket;

validpipe #(
    .CYCLES (2),
    .WIDTH  (1)
)validpipe_acc_flush_en(
    .enable (1'b1),
	.din    (f2_acc_re),
	.dout   (acc_flush_en),
    .idle   (),
	.clk    (clk),
    .rstN   (rstN)
);

hyperpipe #(
    .CYCLES (2),
    .WIDTH  ($bits(acc_raddr))
)hyperpipe_acc_flush_addr(
    .enable (1'b1),
	.din    (f2_acc_raddr),
	.dout   (acc_flush_addr),
	.clk    (clk)
);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//accRAM_rw_mux
////////////////////////////////////////////////////////////////////////////////////////////////////////
//re
//from calc_bucket or calc_group
    assign acc_re           = acc_re_I || f2_acc_re;
    assign acc_raddr        = acc_re_I ? acc_raddr_I : f2_acc_raddr;
    assign acc_rdata_K      = acc_rdata;
//we
    assign acc_we           = acc_we_S || state_C[INITi] || acc_flush_en;
    assign acc_waddr        =
            {$bits(acc_waddr){acc_we_S      }} & acc_waddr_S    |
            {$bits(acc_waddr){state_C[INITi]}} & init_count     |
            {$bits(acc_waddr){acc_flush_en  }} & acc_flush_addr |
            0;
    assign acc_wdata        = acc_we_S ? data_S : 
                              ZPRIZE_ACCRAM_INI; 
//delay 1 cycle
`Flop (acc_re_r,        1'b1, acc_re)
`FlopN(acc_raddr_r,     1'b1, acc_raddr)

`Flop (acc_we_r,        1'b1, acc_we)
`FlopN(acc_waddr_r,     1'b1, acc_waddr)
`FlopN(acc_wdata_r,     1'b1, acc_wdata)
`Flop (acc_re_r_0,     1, acc_re && acc_raddr[16-2:12] == 0)
`FlopN(acc_raddr_r_0,  1, acc_raddr)
`Flop (acc_we_r_0,     1, acc_we && acc_waddr[16-2:12] == 0)
`FlopN(acc_waddr_r_0,  1, acc_waddr)
`Flop (acc_re_r_1,     1, acc_re && acc_raddr[16-2:12] == 1)
`FlopN(acc_raddr_r_1,  1, acc_raddr)
`Flop (acc_we_r_1,     1, acc_we && acc_waddr[16-2:12] == 1)
`FlopN(acc_waddr_r_1,  1, acc_waddr)
`Flop (acc_re_r_2,     1, acc_re && acc_raddr[16-2:12] == 2)
`FlopN(acc_raddr_r_2,  1, acc_raddr)
`Flop (acc_we_r_2,     1, acc_we && acc_waddr[16-2:12] == 2)
`FlopN(acc_waddr_r_2,  1, acc_waddr)
`Flop (acc_re_r_3,     1, acc_re && acc_raddr[16-2:12] == 3)
`FlopN(acc_raddr_r_3,  1, acc_raddr)
`Flop (acc_we_r_3,     1, acc_we && acc_waddr[16-2:12] == 3)
`FlopN(acc_waddr_r_3,  1, acc_waddr)
`Flop (acc_re_r_4,     1, acc_re && acc_raddr[16-2:12] == 4)
`FlopN(acc_raddr_r_4,  1, acc_raddr)
`Flop (acc_we_r_4,     1, acc_we && acc_waddr[16-2:12] == 4)
`FlopN(acc_waddr_r_4,  1, acc_waddr)
`Flop (acc_re_r_5,     1, acc_re && acc_raddr[16-2:12] == 5)
`FlopN(acc_raddr_r_5,  1, acc_raddr)
`Flop (acc_we_r_5,     1, acc_we && acc_waddr[16-2:12] == 5)
`FlopN(acc_waddr_r_5,  1, acc_waddr)
`Flop (acc_re_r_6,     1, acc_re && acc_raddr[16-2:12] == 6)
`FlopN(acc_raddr_r_6,  1, acc_raddr)
`Flop (acc_we_r_6,     1, acc_we && acc_waddr[16-2:12] == 6)
`FlopN(acc_waddr_r_6,  1, acc_waddr)
`Flop (acc_re_r_7,     1, acc_re && acc_raddr[16-2:12] == 7)
`FlopN(acc_raddr_r_7,  1, acc_raddr)
`Flop (acc_we_r_7,     1, acc_we && acc_waddr[16-2:12] == 7)
`FlopN(acc_waddr_r_7,  1, acc_waddr)

endmodule
