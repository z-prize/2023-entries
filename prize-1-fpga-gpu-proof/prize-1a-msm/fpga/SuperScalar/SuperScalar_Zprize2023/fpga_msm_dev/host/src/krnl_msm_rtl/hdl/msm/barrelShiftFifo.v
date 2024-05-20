`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so 64 = 64 for DDR alignment
///define 64       (16*4)

//m items in, n items out
//flop out

//(1152+512)/128        >= 512/128              
//(1152+512)/128        >= 1152/128          
//suggest (1152+512)/128        = 512/128               + 1152/128          
module barrelShiftFifo_point import zprize_param::*; (
    input                       in_valid,
    output                      in_ready,
    input   [128       *512/128              -1:0]   in_data,
    output                      out_valid,
    input                       out_ready,
    output  [128       *1152/128          -1:0]  out_data,
    input                       clk,
    input                       rstN
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    logic   [128       -1:0]         buffer0;
    logic   [128       -1:0]         buffer1;
    logic   [128       -1:0]         buffer2;
    logic   [128       -1:0]         buffer3;
    logic   [128       -1:0]         buffer4;
    logic   [128       -1:0]         buffer5;
    logic   [128       -1:0]         buffer6;
    logic   [128       -1:0]         buffer7;
    logic   [128       -1:0]         buffer8;
    logic   [128       -1:0]         buffer9;
    logic   [128       -1:0]         buffer10;
    logic   [128       -1:0]         buffer11;
    logic   [128       -1:0]         buffer12;
    logic   [128       -1:0]         buffer13;      //useless
    logic   [128       -1:0]         buffer14;      //useless
    logic   [128       -1:0]         buffer15;      //useless
    logic   [128       -1:0]         buffer16;      //useless
    logic   [128       -1:0]         buffer17;      //useless
    logic   [128       -1:0]         buffer18;      //useless
    logic   [128       -1:0]         buffer19;      //useless
    logic   [128       -1:0]         buffer20;      //useless
    logic   [128       -1:0]         buffer21;      //useless

    //to fix fanout, duplicate tail
  (* max_fanout=400 *)  logic   [`LOG2((1152+512)/128       )-1:0]  tail;   //point to first empty item

    logic   [`LOG2((1152+512)/128       )-1:0]  tail_0_0;   //duplicate
    logic                       in_ack_0_0   ;
    logic                       out_ack_0_0  ;
    logic                       in_ready_0_0 ;
    logic                       out_valid_0_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_0_1;   //duplicate
    logic                       in_ack_0_1   ;
    logic                       out_ack_0_1  ;
    logic                       in_ready_0_1 ;
    logic                       out_valid_0_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_0_2;   //duplicate
    logic                       in_ack_0_2   ;
    logic                       out_ack_0_2  ;
    logic                       in_ready_0_2 ;
    logic                       out_valid_0_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_0_3;   //duplicate
    logic                       in_ack_0_3   ;
    logic                       out_ack_0_3  ;
    logic                       in_ready_0_3 ;
    logic                       out_valid_0_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_1_0;   //duplicate
    logic                       in_ack_1_0   ;
    logic                       out_ack_1_0  ;
    logic                       in_ready_1_0 ;
    logic                       out_valid_1_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_1_1;   //duplicate
    logic                       in_ack_1_1   ;
    logic                       out_ack_1_1  ;
    logic                       in_ready_1_1 ;
    logic                       out_valid_1_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_1_2;   //duplicate
    logic                       in_ack_1_2   ;
    logic                       out_ack_1_2  ;
    logic                       in_ready_1_2 ;
    logic                       out_valid_1_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_1_3;   //duplicate
    logic                       in_ack_1_3   ;
    logic                       out_ack_1_3  ;
    logic                       in_ready_1_3 ;
    logic                       out_valid_1_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_2_0;   //duplicate
    logic                       in_ack_2_0   ;
    logic                       out_ack_2_0  ;
    logic                       in_ready_2_0 ;
    logic                       out_valid_2_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_2_1;   //duplicate
    logic                       in_ack_2_1   ;
    logic                       out_ack_2_1  ;
    logic                       in_ready_2_1 ;
    logic                       out_valid_2_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_2_2;   //duplicate
    logic                       in_ack_2_2   ;
    logic                       out_ack_2_2  ;
    logic                       in_ready_2_2 ;
    logic                       out_valid_2_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_2_3;   //duplicate
    logic                       in_ack_2_3   ;
    logic                       out_ack_2_3  ;
    logic                       in_ready_2_3 ;
    logic                       out_valid_2_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_3_0;   //duplicate
    logic                       in_ack_3_0   ;
    logic                       out_ack_3_0  ;
    logic                       in_ready_3_0 ;
    logic                       out_valid_3_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_3_1;   //duplicate
    logic                       in_ack_3_1   ;
    logic                       out_ack_3_1  ;
    logic                       in_ready_3_1 ;
    logic                       out_valid_3_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_3_2;   //duplicate
    logic                       in_ack_3_2   ;
    logic                       out_ack_3_2  ;
    logic                       in_ready_3_2 ;
    logic                       out_valid_3_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_3_3;   //duplicate
    logic                       in_ack_3_3   ;
    logic                       out_ack_3_3  ;
    logic                       in_ready_3_3 ;
    logic                       out_valid_3_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_4_0;   //duplicate
    logic                       in_ack_4_0   ;
    logic                       out_ack_4_0  ;
    logic                       in_ready_4_0 ;
    logic                       out_valid_4_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_4_1;   //duplicate
    logic                       in_ack_4_1   ;
    logic                       out_ack_4_1  ;
    logic                       in_ready_4_1 ;
    logic                       out_valid_4_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_4_2;   //duplicate
    logic                       in_ack_4_2   ;
    logic                       out_ack_4_2  ;
    logic                       in_ready_4_2 ;
    logic                       out_valid_4_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_4_3;   //duplicate
    logic                       in_ack_4_3   ;
    logic                       out_ack_4_3  ;
    logic                       in_ready_4_3 ;
    logic                       out_valid_4_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_5_0;   //duplicate
    logic                       in_ack_5_0   ;
    logic                       out_ack_5_0  ;
    logic                       in_ready_5_0 ;
    logic                       out_valid_5_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_5_1;   //duplicate
    logic                       in_ack_5_1   ;
    logic                       out_ack_5_1  ;
    logic                       in_ready_5_1 ;
    logic                       out_valid_5_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_5_2;   //duplicate
    logic                       in_ack_5_2   ;
    logic                       out_ack_5_2  ;
    logic                       in_ready_5_2 ;
    logic                       out_valid_5_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_5_3;   //duplicate
    logic                       in_ack_5_3   ;
    logic                       out_ack_5_3  ;
    logic                       in_ready_5_3 ;
    logic                       out_valid_5_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_6_0;   //duplicate
    logic                       in_ack_6_0   ;
    logic                       out_ack_6_0  ;
    logic                       in_ready_6_0 ;
    logic                       out_valid_6_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_6_1;   //duplicate
    logic                       in_ack_6_1   ;
    logic                       out_ack_6_1  ;
    logic                       in_ready_6_1 ;
    logic                       out_valid_6_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_6_2;   //duplicate
    logic                       in_ack_6_2   ;
    logic                       out_ack_6_2  ;
    logic                       in_ready_6_2 ;
    logic                       out_valid_6_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_6_3;   //duplicate
    logic                       in_ack_6_3   ;
    logic                       out_ack_6_3  ;
    logic                       in_ready_6_3 ;
    logic                       out_valid_6_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_7_0;   //duplicate
    logic                       in_ack_7_0   ;
    logic                       out_ack_7_0  ;
    logic                       in_ready_7_0 ;
    logic                       out_valid_7_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_7_1;   //duplicate
    logic                       in_ack_7_1   ;
    logic                       out_ack_7_1  ;
    logic                       in_ready_7_1 ;
    logic                       out_valid_7_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_7_2;   //duplicate
    logic                       in_ack_7_2   ;
    logic                       out_ack_7_2  ;
    logic                       in_ready_7_2 ;
    logic                       out_valid_7_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_7_3;   //duplicate
    logic                       in_ack_7_3   ;
    logic                       out_ack_7_3  ;
    logic                       in_ready_7_3 ;
    logic                       out_valid_7_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_8_0;   //duplicate
    logic                       in_ack_8_0   ;
    logic                       out_ack_8_0  ;
    logic                       in_ready_8_0 ;
    logic                       out_valid_8_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_8_1;   //duplicate
    logic                       in_ack_8_1   ;
    logic                       out_ack_8_1  ;
    logic                       in_ready_8_1 ;
    logic                       out_valid_8_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_8_2;   //duplicate
    logic                       in_ack_8_2   ;
    logic                       out_ack_8_2  ;
    logic                       in_ready_8_2 ;
    logic                       out_valid_8_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_8_3;   //duplicate
    logic                       in_ack_8_3   ;
    logic                       out_ack_8_3  ;
    logic                       in_ready_8_3 ;
    logic                       out_valid_8_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_9_0;   //duplicate
    logic                       in_ack_9_0   ;
    logic                       out_ack_9_0  ;
    logic                       in_ready_9_0 ;
    logic                       out_valid_9_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_9_1;   //duplicate
    logic                       in_ack_9_1   ;
    logic                       out_ack_9_1  ;
    logic                       in_ready_9_1 ;
    logic                       out_valid_9_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_9_2;   //duplicate
    logic                       in_ack_9_2   ;
    logic                       out_ack_9_2  ;
    logic                       in_ready_9_2 ;
    logic                       out_valid_9_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_9_3;   //duplicate
    logic                       in_ack_9_3   ;
    logic                       out_ack_9_3  ;
    logic                       in_ready_9_3 ;
    logic                       out_valid_9_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_10_0;   //duplicate
    logic                       in_ack_10_0   ;
    logic                       out_ack_10_0  ;
    logic                       in_ready_10_0 ;
    logic                       out_valid_10_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_10_1;   //duplicate
    logic                       in_ack_10_1   ;
    logic                       out_ack_10_1  ;
    logic                       in_ready_10_1 ;
    logic                       out_valid_10_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_10_2;   //duplicate
    logic                       in_ack_10_2   ;
    logic                       out_ack_10_2  ;
    logic                       in_ready_10_2 ;
    logic                       out_valid_10_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_10_3;   //duplicate
    logic                       in_ack_10_3   ;
    logic                       out_ack_10_3  ;
    logic                       in_ready_10_3 ;
    logic                       out_valid_10_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_11_0;   //duplicate
    logic                       in_ack_11_0   ;
    logic                       out_ack_11_0  ;
    logic                       in_ready_11_0 ;
    logic                       out_valid_11_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_11_1;   //duplicate
    logic                       in_ack_11_1   ;
    logic                       out_ack_11_1  ;
    logic                       in_ready_11_1 ;
    logic                       out_valid_11_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_11_2;   //duplicate
    logic                       in_ack_11_2   ;
    logic                       out_ack_11_2  ;
    logic                       in_ready_11_2 ;
    logic                       out_valid_11_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_11_3;   //duplicate
    logic                       in_ack_11_3   ;
    logic                       out_ack_11_3  ;
    logic                       in_ready_11_3 ;
    logic                       out_valid_11_3;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_12_0;   //duplicate
    logic                       in_ack_12_0   ;
    logic                       out_ack_12_0  ;
    logic                       in_ready_12_0 ;
    logic                       out_valid_12_0;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_12_1;   //duplicate
    logic                       in_ack_12_1   ;
    logic                       out_ack_12_1  ;
    logic                       in_ready_12_1 ;
    logic                       out_valid_12_1;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_12_2;   //duplicate
    logic                       in_ack_12_2   ;
    logic                       out_ack_12_2  ;
    logic                       in_ready_12_2 ;
    logic                       out_valid_12_2;
    logic   [`LOG2((1152+512)/128       )-1:0]  tail_12_3;   //duplicate
    logic                       in_ack_12_3   ;
    logic                       out_ack_12_3  ;
    logic                       in_ready_12_3 ;
    logic                       out_valid_12_3;

    logic                       in_ack;
    logic                       out_ack;
    genvar                      i, j, k;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign in_ack = in_valid && in_ready;
    assign out_ack = out_valid && out_ready;
    assign in_ready = tail + 512/128               <= (1152+512)/128       ;
    assign out_valid = tail >= 1152/128          ;

    assign in_ack_0_0   = in_valid && in_ready_0_0;
    assign out_ack_0_0  = out_valid_0_0 && out_ready;
    assign in_ready_0_0 = tail_0_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_0_0= tail_0_0 >= 1152/128          ;
    assign in_ack_0_1   = in_valid && in_ready_0_1;
    assign out_ack_0_1  = out_valid_0_1 && out_ready;
    assign in_ready_0_1 = tail_0_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_0_1= tail_0_1 >= 1152/128          ;
    assign in_ack_0_2   = in_valid && in_ready_0_2;
    assign out_ack_0_2  = out_valid_0_2 && out_ready;
    assign in_ready_0_2 = tail_0_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_0_2= tail_0_2 >= 1152/128          ;
    assign in_ack_0_3   = in_valid && in_ready_0_3;
    assign out_ack_0_3  = out_valid_0_3 && out_ready;
    assign in_ready_0_3 = tail_0_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_0_3= tail_0_3 >= 1152/128          ;
    assign in_ack_1_0   = in_valid && in_ready_1_0;
    assign out_ack_1_0  = out_valid_1_0 && out_ready;
    assign in_ready_1_0 = tail_1_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_1_0= tail_1_0 >= 1152/128          ;
    assign in_ack_1_1   = in_valid && in_ready_1_1;
    assign out_ack_1_1  = out_valid_1_1 && out_ready;
    assign in_ready_1_1 = tail_1_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_1_1= tail_1_1 >= 1152/128          ;
    assign in_ack_1_2   = in_valid && in_ready_1_2;
    assign out_ack_1_2  = out_valid_1_2 && out_ready;
    assign in_ready_1_2 = tail_1_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_1_2= tail_1_2 >= 1152/128          ;
    assign in_ack_1_3   = in_valid && in_ready_1_3;
    assign out_ack_1_3  = out_valid_1_3 && out_ready;
    assign in_ready_1_3 = tail_1_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_1_3= tail_1_3 >= 1152/128          ;
    assign in_ack_2_0   = in_valid && in_ready_2_0;
    assign out_ack_2_0  = out_valid_2_0 && out_ready;
    assign in_ready_2_0 = tail_2_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_2_0= tail_2_0 >= 1152/128          ;
    assign in_ack_2_1   = in_valid && in_ready_2_1;
    assign out_ack_2_1  = out_valid_2_1 && out_ready;
    assign in_ready_2_1 = tail_2_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_2_1= tail_2_1 >= 1152/128          ;
    assign in_ack_2_2   = in_valid && in_ready_2_2;
    assign out_ack_2_2  = out_valid_2_2 && out_ready;
    assign in_ready_2_2 = tail_2_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_2_2= tail_2_2 >= 1152/128          ;
    assign in_ack_2_3   = in_valid && in_ready_2_3;
    assign out_ack_2_3  = out_valid_2_3 && out_ready;
    assign in_ready_2_3 = tail_2_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_2_3= tail_2_3 >= 1152/128          ;
    assign in_ack_3_0   = in_valid && in_ready_3_0;
    assign out_ack_3_0  = out_valid_3_0 && out_ready;
    assign in_ready_3_0 = tail_3_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_3_0= tail_3_0 >= 1152/128          ;
    assign in_ack_3_1   = in_valid && in_ready_3_1;
    assign out_ack_3_1  = out_valid_3_1 && out_ready;
    assign in_ready_3_1 = tail_3_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_3_1= tail_3_1 >= 1152/128          ;
    assign in_ack_3_2   = in_valid && in_ready_3_2;
    assign out_ack_3_2  = out_valid_3_2 && out_ready;
    assign in_ready_3_2 = tail_3_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_3_2= tail_3_2 >= 1152/128          ;
    assign in_ack_3_3   = in_valid && in_ready_3_3;
    assign out_ack_3_3  = out_valid_3_3 && out_ready;
    assign in_ready_3_3 = tail_3_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_3_3= tail_3_3 >= 1152/128          ;
    assign in_ack_4_0   = in_valid && in_ready_4_0;
    assign out_ack_4_0  = out_valid_4_0 && out_ready;
    assign in_ready_4_0 = tail_4_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_4_0= tail_4_0 >= 1152/128          ;
    assign in_ack_4_1   = in_valid && in_ready_4_1;
    assign out_ack_4_1  = out_valid_4_1 && out_ready;
    assign in_ready_4_1 = tail_4_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_4_1= tail_4_1 >= 1152/128          ;
    assign in_ack_4_2   = in_valid && in_ready_4_2;
    assign out_ack_4_2  = out_valid_4_2 && out_ready;
    assign in_ready_4_2 = tail_4_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_4_2= tail_4_2 >= 1152/128          ;
    assign in_ack_4_3   = in_valid && in_ready_4_3;
    assign out_ack_4_3  = out_valid_4_3 && out_ready;
    assign in_ready_4_3 = tail_4_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_4_3= tail_4_3 >= 1152/128          ;
    assign in_ack_5_0   = in_valid && in_ready_5_0;
    assign out_ack_5_0  = out_valid_5_0 && out_ready;
    assign in_ready_5_0 = tail_5_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_5_0= tail_5_0 >= 1152/128          ;
    assign in_ack_5_1   = in_valid && in_ready_5_1;
    assign out_ack_5_1  = out_valid_5_1 && out_ready;
    assign in_ready_5_1 = tail_5_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_5_1= tail_5_1 >= 1152/128          ;
    assign in_ack_5_2   = in_valid && in_ready_5_2;
    assign out_ack_5_2  = out_valid_5_2 && out_ready;
    assign in_ready_5_2 = tail_5_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_5_2= tail_5_2 >= 1152/128          ;
    assign in_ack_5_3   = in_valid && in_ready_5_3;
    assign out_ack_5_3  = out_valid_5_3 && out_ready;
    assign in_ready_5_3 = tail_5_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_5_3= tail_5_3 >= 1152/128          ;
    assign in_ack_6_0   = in_valid && in_ready_6_0;
    assign out_ack_6_0  = out_valid_6_0 && out_ready;
    assign in_ready_6_0 = tail_6_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_6_0= tail_6_0 >= 1152/128          ;
    assign in_ack_6_1   = in_valid && in_ready_6_1;
    assign out_ack_6_1  = out_valid_6_1 && out_ready;
    assign in_ready_6_1 = tail_6_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_6_1= tail_6_1 >= 1152/128          ;
    assign in_ack_6_2   = in_valid && in_ready_6_2;
    assign out_ack_6_2  = out_valid_6_2 && out_ready;
    assign in_ready_6_2 = tail_6_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_6_2= tail_6_2 >= 1152/128          ;
    assign in_ack_6_3   = in_valid && in_ready_6_3;
    assign out_ack_6_3  = out_valid_6_3 && out_ready;
    assign in_ready_6_3 = tail_6_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_6_3= tail_6_3 >= 1152/128          ;
    assign in_ack_7_0   = in_valid && in_ready_7_0;
    assign out_ack_7_0  = out_valid_7_0 && out_ready;
    assign in_ready_7_0 = tail_7_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_7_0= tail_7_0 >= 1152/128          ;
    assign in_ack_7_1   = in_valid && in_ready_7_1;
    assign out_ack_7_1  = out_valid_7_1 && out_ready;
    assign in_ready_7_1 = tail_7_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_7_1= tail_7_1 >= 1152/128          ;
    assign in_ack_7_2   = in_valid && in_ready_7_2;
    assign out_ack_7_2  = out_valid_7_2 && out_ready;
    assign in_ready_7_2 = tail_7_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_7_2= tail_7_2 >= 1152/128          ;
    assign in_ack_7_3   = in_valid && in_ready_7_3;
    assign out_ack_7_3  = out_valid_7_3 && out_ready;
    assign in_ready_7_3 = tail_7_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_7_3= tail_7_3 >= 1152/128          ;
    assign in_ack_8_0   = in_valid && in_ready_8_0;
    assign out_ack_8_0  = out_valid_8_0 && out_ready;
    assign in_ready_8_0 = tail_8_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_8_0= tail_8_0 >= 1152/128          ;
    assign in_ack_8_1   = in_valid && in_ready_8_1;
    assign out_ack_8_1  = out_valid_8_1 && out_ready;
    assign in_ready_8_1 = tail_8_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_8_1= tail_8_1 >= 1152/128          ;
    assign in_ack_8_2   = in_valid && in_ready_8_2;
    assign out_ack_8_2  = out_valid_8_2 && out_ready;
    assign in_ready_8_2 = tail_8_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_8_2= tail_8_2 >= 1152/128          ;
    assign in_ack_8_3   = in_valid && in_ready_8_3;
    assign out_ack_8_3  = out_valid_8_3 && out_ready;
    assign in_ready_8_3 = tail_8_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_8_3= tail_8_3 >= 1152/128          ;
    assign in_ack_9_0   = in_valid && in_ready_9_0;
    assign out_ack_9_0  = out_valid_9_0 && out_ready;
    assign in_ready_9_0 = tail_9_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_9_0= tail_9_0 >= 1152/128          ;
    assign in_ack_9_1   = in_valid && in_ready_9_1;
    assign out_ack_9_1  = out_valid_9_1 && out_ready;
    assign in_ready_9_1 = tail_9_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_9_1= tail_9_1 >= 1152/128          ;
    assign in_ack_9_2   = in_valid && in_ready_9_2;
    assign out_ack_9_2  = out_valid_9_2 && out_ready;
    assign in_ready_9_2 = tail_9_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_9_2= tail_9_2 >= 1152/128          ;
    assign in_ack_9_3   = in_valid && in_ready_9_3;
    assign out_ack_9_3  = out_valid_9_3 && out_ready;
    assign in_ready_9_3 = tail_9_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_9_3= tail_9_3 >= 1152/128          ;
    assign in_ack_10_0   = in_valid && in_ready_10_0;
    assign out_ack_10_0  = out_valid_10_0 && out_ready;
    assign in_ready_10_0 = tail_10_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_10_0= tail_10_0 >= 1152/128          ;
    assign in_ack_10_1   = in_valid && in_ready_10_1;
    assign out_ack_10_1  = out_valid_10_1 && out_ready;
    assign in_ready_10_1 = tail_10_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_10_1= tail_10_1 >= 1152/128          ;
    assign in_ack_10_2   = in_valid && in_ready_10_2;
    assign out_ack_10_2  = out_valid_10_2 && out_ready;
    assign in_ready_10_2 = tail_10_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_10_2= tail_10_2 >= 1152/128          ;
    assign in_ack_10_3   = in_valid && in_ready_10_3;
    assign out_ack_10_3  = out_valid_10_3 && out_ready;
    assign in_ready_10_3 = tail_10_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_10_3= tail_10_3 >= 1152/128          ;
    assign in_ack_11_0   = in_valid && in_ready_11_0;
    assign out_ack_11_0  = out_valid_11_0 && out_ready;
    assign in_ready_11_0 = tail_11_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_11_0= tail_11_0 >= 1152/128          ;
    assign in_ack_11_1   = in_valid && in_ready_11_1;
    assign out_ack_11_1  = out_valid_11_1 && out_ready;
    assign in_ready_11_1 = tail_11_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_11_1= tail_11_1 >= 1152/128          ;
    assign in_ack_11_2   = in_valid && in_ready_11_2;
    assign out_ack_11_2  = out_valid_11_2 && out_ready;
    assign in_ready_11_2 = tail_11_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_11_2= tail_11_2 >= 1152/128          ;
    assign in_ack_11_3   = in_valid && in_ready_11_3;
    assign out_ack_11_3  = out_valid_11_3 && out_ready;
    assign in_ready_11_3 = tail_11_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_11_3= tail_11_3 >= 1152/128          ;
    assign in_ack_12_0   = in_valid && in_ready_12_0;
    assign out_ack_12_0  = out_valid_12_0 && out_ready;
    assign in_ready_12_0 = tail_12_0 + 512/128               <= (1152+512)/128       ;
    assign out_valid_12_0= tail_12_0 >= 1152/128          ;
    assign in_ack_12_1   = in_valid && in_ready_12_1;
    assign out_ack_12_1  = out_valid_12_1 && out_ready;
    assign in_ready_12_1 = tail_12_1 + 512/128               <= (1152+512)/128       ;
    assign out_valid_12_1= tail_12_1 >= 1152/128          ;
    assign in_ack_12_2   = in_valid && in_ready_12_2;
    assign out_ack_12_2  = out_valid_12_2 && out_ready;
    assign in_ready_12_2 = tail_12_2 + 512/128               <= (1152+512)/128       ;
    assign out_valid_12_2= tail_12_2 >= 1152/128          ;
    assign in_ack_12_3   = in_valid && in_ready_12_3;
    assign out_ack_12_3  = out_valid_12_3 && out_ready;
    assign in_ready_12_3 = tail_12_3 + 512/128               <= (1152+512)/128       ;
    assign out_valid_12_3= tail_12_3 >= 1152/128          ;

    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail <= `CQ 0;
        else if(in_ack && out_ack)
            tail <= `CQ tail + 512/128               - 1152/128          ;
        else if(in_ack)
            tail <= `CQ tail + 512/128              ;
        else if(out_ack)
            tail <= `CQ tail - 1152/128          ;

    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_0 <= `CQ 0;
        else if(in_ack_0_0&& out_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 + 512/128               - 1152/128          ;
        else if(in_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 + 512/128              ;
        else if(out_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_1 <= `CQ 0;
        else if(in_ack_0_1&& out_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 + 512/128               - 1152/128          ;
        else if(in_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 + 512/128              ;
        else if(out_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_2 <= `CQ 0;
        else if(in_ack_0_2&& out_ack_0_2)
            tail_0_2 <= `CQ tail_0_2 + 512/128               - 1152/128          ;
        else if(in_ack_0_2)
            tail_0_2 <= `CQ tail_0_2 + 512/128              ;
        else if(out_ack_0_2)
            tail_0_2 <= `CQ tail_0_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_3 <= `CQ 0;
        else if(in_ack_0_3&& out_ack_0_3)
            tail_0_3 <= `CQ tail_0_3 + 512/128               - 1152/128          ;
        else if(in_ack_0_3)
            tail_0_3 <= `CQ tail_0_3 + 512/128              ;
        else if(out_ack_0_3)
            tail_0_3 <= `CQ tail_0_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_0 <= `CQ 0;
        else if(in_ack_1_0&& out_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 + 512/128               - 1152/128          ;
        else if(in_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 + 512/128              ;
        else if(out_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_1 <= `CQ 0;
        else if(in_ack_1_1&& out_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 + 512/128               - 1152/128          ;
        else if(in_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 + 512/128              ;
        else if(out_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_2 <= `CQ 0;
        else if(in_ack_1_2&& out_ack_1_2)
            tail_1_2 <= `CQ tail_1_2 + 512/128               - 1152/128          ;
        else if(in_ack_1_2)
            tail_1_2 <= `CQ tail_1_2 + 512/128              ;
        else if(out_ack_1_2)
            tail_1_2 <= `CQ tail_1_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_3 <= `CQ 0;
        else if(in_ack_1_3&& out_ack_1_3)
            tail_1_3 <= `CQ tail_1_3 + 512/128               - 1152/128          ;
        else if(in_ack_1_3)
            tail_1_3 <= `CQ tail_1_3 + 512/128              ;
        else if(out_ack_1_3)
            tail_1_3 <= `CQ tail_1_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_0 <= `CQ 0;
        else if(in_ack_2_0&& out_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 + 512/128               - 1152/128          ;
        else if(in_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 + 512/128              ;
        else if(out_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_1 <= `CQ 0;
        else if(in_ack_2_1&& out_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 + 512/128               - 1152/128          ;
        else if(in_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 + 512/128              ;
        else if(out_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_2 <= `CQ 0;
        else if(in_ack_2_2&& out_ack_2_2)
            tail_2_2 <= `CQ tail_2_2 + 512/128               - 1152/128          ;
        else if(in_ack_2_2)
            tail_2_2 <= `CQ tail_2_2 + 512/128              ;
        else if(out_ack_2_2)
            tail_2_2 <= `CQ tail_2_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_3 <= `CQ 0;
        else if(in_ack_2_3&& out_ack_2_3)
            tail_2_3 <= `CQ tail_2_3 + 512/128               - 1152/128          ;
        else if(in_ack_2_3)
            tail_2_3 <= `CQ tail_2_3 + 512/128              ;
        else if(out_ack_2_3)
            tail_2_3 <= `CQ tail_2_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_0 <= `CQ 0;
        else if(in_ack_3_0&& out_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 + 512/128               - 1152/128          ;
        else if(in_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 + 512/128              ;
        else if(out_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_1 <= `CQ 0;
        else if(in_ack_3_1&& out_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 + 512/128               - 1152/128          ;
        else if(in_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 + 512/128              ;
        else if(out_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_2 <= `CQ 0;
        else if(in_ack_3_2&& out_ack_3_2)
            tail_3_2 <= `CQ tail_3_2 + 512/128               - 1152/128          ;
        else if(in_ack_3_2)
            tail_3_2 <= `CQ tail_3_2 + 512/128              ;
        else if(out_ack_3_2)
            tail_3_2 <= `CQ tail_3_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_3 <= `CQ 0;
        else if(in_ack_3_3&& out_ack_3_3)
            tail_3_3 <= `CQ tail_3_3 + 512/128               - 1152/128          ;
        else if(in_ack_3_3)
            tail_3_3 <= `CQ tail_3_3 + 512/128              ;
        else if(out_ack_3_3)
            tail_3_3 <= `CQ tail_3_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_4_0 <= `CQ 0;
        else if(in_ack_4_0&& out_ack_4_0)
            tail_4_0 <= `CQ tail_4_0 + 512/128               - 1152/128          ;
        else if(in_ack_4_0)
            tail_4_0 <= `CQ tail_4_0 + 512/128              ;
        else if(out_ack_4_0)
            tail_4_0 <= `CQ tail_4_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_4_1 <= `CQ 0;
        else if(in_ack_4_1&& out_ack_4_1)
            tail_4_1 <= `CQ tail_4_1 + 512/128               - 1152/128          ;
        else if(in_ack_4_1)
            tail_4_1 <= `CQ tail_4_1 + 512/128              ;
        else if(out_ack_4_1)
            tail_4_1 <= `CQ tail_4_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_4_2 <= `CQ 0;
        else if(in_ack_4_2&& out_ack_4_2)
            tail_4_2 <= `CQ tail_4_2 + 512/128               - 1152/128          ;
        else if(in_ack_4_2)
            tail_4_2 <= `CQ tail_4_2 + 512/128              ;
        else if(out_ack_4_2)
            tail_4_2 <= `CQ tail_4_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_4_3 <= `CQ 0;
        else if(in_ack_4_3&& out_ack_4_3)
            tail_4_3 <= `CQ tail_4_3 + 512/128               - 1152/128          ;
        else if(in_ack_4_3)
            tail_4_3 <= `CQ tail_4_3 + 512/128              ;
        else if(out_ack_4_3)
            tail_4_3 <= `CQ tail_4_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_5_0 <= `CQ 0;
        else if(in_ack_5_0&& out_ack_5_0)
            tail_5_0 <= `CQ tail_5_0 + 512/128               - 1152/128          ;
        else if(in_ack_5_0)
            tail_5_0 <= `CQ tail_5_0 + 512/128              ;
        else if(out_ack_5_0)
            tail_5_0 <= `CQ tail_5_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_5_1 <= `CQ 0;
        else if(in_ack_5_1&& out_ack_5_1)
            tail_5_1 <= `CQ tail_5_1 + 512/128               - 1152/128          ;
        else if(in_ack_5_1)
            tail_5_1 <= `CQ tail_5_1 + 512/128              ;
        else if(out_ack_5_1)
            tail_5_1 <= `CQ tail_5_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_5_2 <= `CQ 0;
        else if(in_ack_5_2&& out_ack_5_2)
            tail_5_2 <= `CQ tail_5_2 + 512/128               - 1152/128          ;
        else if(in_ack_5_2)
            tail_5_2 <= `CQ tail_5_2 + 512/128              ;
        else if(out_ack_5_2)
            tail_5_2 <= `CQ tail_5_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_5_3 <= `CQ 0;
        else if(in_ack_5_3&& out_ack_5_3)
            tail_5_3 <= `CQ tail_5_3 + 512/128               - 1152/128          ;
        else if(in_ack_5_3)
            tail_5_3 <= `CQ tail_5_3 + 512/128              ;
        else if(out_ack_5_3)
            tail_5_3 <= `CQ tail_5_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_6_0 <= `CQ 0;
        else if(in_ack_6_0&& out_ack_6_0)
            tail_6_0 <= `CQ tail_6_0 + 512/128               - 1152/128          ;
        else if(in_ack_6_0)
            tail_6_0 <= `CQ tail_6_0 + 512/128              ;
        else if(out_ack_6_0)
            tail_6_0 <= `CQ tail_6_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_6_1 <= `CQ 0;
        else if(in_ack_6_1&& out_ack_6_1)
            tail_6_1 <= `CQ tail_6_1 + 512/128               - 1152/128          ;
        else if(in_ack_6_1)
            tail_6_1 <= `CQ tail_6_1 + 512/128              ;
        else if(out_ack_6_1)
            tail_6_1 <= `CQ tail_6_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_6_2 <= `CQ 0;
        else if(in_ack_6_2&& out_ack_6_2)
            tail_6_2 <= `CQ tail_6_2 + 512/128               - 1152/128          ;
        else if(in_ack_6_2)
            tail_6_2 <= `CQ tail_6_2 + 512/128              ;
        else if(out_ack_6_2)
            tail_6_2 <= `CQ tail_6_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_6_3 <= `CQ 0;
        else if(in_ack_6_3&& out_ack_6_3)
            tail_6_3 <= `CQ tail_6_3 + 512/128               - 1152/128          ;
        else if(in_ack_6_3)
            tail_6_3 <= `CQ tail_6_3 + 512/128              ;
        else if(out_ack_6_3)
            tail_6_3 <= `CQ tail_6_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_7_0 <= `CQ 0;
        else if(in_ack_7_0&& out_ack_7_0)
            tail_7_0 <= `CQ tail_7_0 + 512/128               - 1152/128          ;
        else if(in_ack_7_0)
            tail_7_0 <= `CQ tail_7_0 + 512/128              ;
        else if(out_ack_7_0)
            tail_7_0 <= `CQ tail_7_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_7_1 <= `CQ 0;
        else if(in_ack_7_1&& out_ack_7_1)
            tail_7_1 <= `CQ tail_7_1 + 512/128               - 1152/128          ;
        else if(in_ack_7_1)
            tail_7_1 <= `CQ tail_7_1 + 512/128              ;
        else if(out_ack_7_1)
            tail_7_1 <= `CQ tail_7_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_7_2 <= `CQ 0;
        else if(in_ack_7_2&& out_ack_7_2)
            tail_7_2 <= `CQ tail_7_2 + 512/128               - 1152/128          ;
        else if(in_ack_7_2)
            tail_7_2 <= `CQ tail_7_2 + 512/128              ;
        else if(out_ack_7_2)
            tail_7_2 <= `CQ tail_7_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_7_3 <= `CQ 0;
        else if(in_ack_7_3&& out_ack_7_3)
            tail_7_3 <= `CQ tail_7_3 + 512/128               - 1152/128          ;
        else if(in_ack_7_3)
            tail_7_3 <= `CQ tail_7_3 + 512/128              ;
        else if(out_ack_7_3)
            tail_7_3 <= `CQ tail_7_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_8_0 <= `CQ 0;
        else if(in_ack_8_0&& out_ack_8_0)
            tail_8_0 <= `CQ tail_8_0 + 512/128               - 1152/128          ;
        else if(in_ack_8_0)
            tail_8_0 <= `CQ tail_8_0 + 512/128              ;
        else if(out_ack_8_0)
            tail_8_0 <= `CQ tail_8_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_8_1 <= `CQ 0;
        else if(in_ack_8_1&& out_ack_8_1)
            tail_8_1 <= `CQ tail_8_1 + 512/128               - 1152/128          ;
        else if(in_ack_8_1)
            tail_8_1 <= `CQ tail_8_1 + 512/128              ;
        else if(out_ack_8_1)
            tail_8_1 <= `CQ tail_8_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_8_2 <= `CQ 0;
        else if(in_ack_8_2&& out_ack_8_2)
            tail_8_2 <= `CQ tail_8_2 + 512/128               - 1152/128          ;
        else if(in_ack_8_2)
            tail_8_2 <= `CQ tail_8_2 + 512/128              ;
        else if(out_ack_8_2)
            tail_8_2 <= `CQ tail_8_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_8_3 <= `CQ 0;
        else if(in_ack_8_3&& out_ack_8_3)
            tail_8_3 <= `CQ tail_8_3 + 512/128               - 1152/128          ;
        else if(in_ack_8_3)
            tail_8_3 <= `CQ tail_8_3 + 512/128              ;
        else if(out_ack_8_3)
            tail_8_3 <= `CQ tail_8_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_9_0 <= `CQ 0;
        else if(in_ack_9_0&& out_ack_9_0)
            tail_9_0 <= `CQ tail_9_0 + 512/128               - 1152/128          ;
        else if(in_ack_9_0)
            tail_9_0 <= `CQ tail_9_0 + 512/128              ;
        else if(out_ack_9_0)
            tail_9_0 <= `CQ tail_9_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_9_1 <= `CQ 0;
        else if(in_ack_9_1&& out_ack_9_1)
            tail_9_1 <= `CQ tail_9_1 + 512/128               - 1152/128          ;
        else if(in_ack_9_1)
            tail_9_1 <= `CQ tail_9_1 + 512/128              ;
        else if(out_ack_9_1)
            tail_9_1 <= `CQ tail_9_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_9_2 <= `CQ 0;
        else if(in_ack_9_2&& out_ack_9_2)
            tail_9_2 <= `CQ tail_9_2 + 512/128               - 1152/128          ;
        else if(in_ack_9_2)
            tail_9_2 <= `CQ tail_9_2 + 512/128              ;
        else if(out_ack_9_2)
            tail_9_2 <= `CQ tail_9_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_9_3 <= `CQ 0;
        else if(in_ack_9_3&& out_ack_9_3)
            tail_9_3 <= `CQ tail_9_3 + 512/128               - 1152/128          ;
        else if(in_ack_9_3)
            tail_9_3 <= `CQ tail_9_3 + 512/128              ;
        else if(out_ack_9_3)
            tail_9_3 <= `CQ tail_9_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_10_0 <= `CQ 0;
        else if(in_ack_10_0&& out_ack_10_0)
            tail_10_0 <= `CQ tail_10_0 + 512/128               - 1152/128          ;
        else if(in_ack_10_0)
            tail_10_0 <= `CQ tail_10_0 + 512/128              ;
        else if(out_ack_10_0)
            tail_10_0 <= `CQ tail_10_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_10_1 <= `CQ 0;
        else if(in_ack_10_1&& out_ack_10_1)
            tail_10_1 <= `CQ tail_10_1 + 512/128               - 1152/128          ;
        else if(in_ack_10_1)
            tail_10_1 <= `CQ tail_10_1 + 512/128              ;
        else if(out_ack_10_1)
            tail_10_1 <= `CQ tail_10_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_10_2 <= `CQ 0;
        else if(in_ack_10_2&& out_ack_10_2)
            tail_10_2 <= `CQ tail_10_2 + 512/128               - 1152/128          ;
        else if(in_ack_10_2)
            tail_10_2 <= `CQ tail_10_2 + 512/128              ;
        else if(out_ack_10_2)
            tail_10_2 <= `CQ tail_10_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_10_3 <= `CQ 0;
        else if(in_ack_10_3&& out_ack_10_3)
            tail_10_3 <= `CQ tail_10_3 + 512/128               - 1152/128          ;
        else if(in_ack_10_3)
            tail_10_3 <= `CQ tail_10_3 + 512/128              ;
        else if(out_ack_10_3)
            tail_10_3 <= `CQ tail_10_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_11_0 <= `CQ 0;
        else if(in_ack_11_0&& out_ack_11_0)
            tail_11_0 <= `CQ tail_11_0 + 512/128               - 1152/128          ;
        else if(in_ack_11_0)
            tail_11_0 <= `CQ tail_11_0 + 512/128              ;
        else if(out_ack_11_0)
            tail_11_0 <= `CQ tail_11_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_11_1 <= `CQ 0;
        else if(in_ack_11_1&& out_ack_11_1)
            tail_11_1 <= `CQ tail_11_1 + 512/128               - 1152/128          ;
        else if(in_ack_11_1)
            tail_11_1 <= `CQ tail_11_1 + 512/128              ;
        else if(out_ack_11_1)
            tail_11_1 <= `CQ tail_11_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_11_2 <= `CQ 0;
        else if(in_ack_11_2&& out_ack_11_2)
            tail_11_2 <= `CQ tail_11_2 + 512/128               - 1152/128          ;
        else if(in_ack_11_2)
            tail_11_2 <= `CQ tail_11_2 + 512/128              ;
        else if(out_ack_11_2)
            tail_11_2 <= `CQ tail_11_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_11_3 <= `CQ 0;
        else if(in_ack_11_3&& out_ack_11_3)
            tail_11_3 <= `CQ tail_11_3 + 512/128               - 1152/128          ;
        else if(in_ack_11_3)
            tail_11_3 <= `CQ tail_11_3 + 512/128              ;
        else if(out_ack_11_3)
            tail_11_3 <= `CQ tail_11_3 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_12_0 <= `CQ 0;
        else if(in_ack_12_0&& out_ack_12_0)
            tail_12_0 <= `CQ tail_12_0 + 512/128               - 1152/128          ;
        else if(in_ack_12_0)
            tail_12_0 <= `CQ tail_12_0 + 512/128              ;
        else if(out_ack_12_0)
            tail_12_0 <= `CQ tail_12_0 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_12_1 <= `CQ 0;
        else if(in_ack_12_1&& out_ack_12_1)
            tail_12_1 <= `CQ tail_12_1 + 512/128               - 1152/128          ;
        else if(in_ack_12_1)
            tail_12_1 <= `CQ tail_12_1 + 512/128              ;
        else if(out_ack_12_1)
            tail_12_1 <= `CQ tail_12_1 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_12_2 <= `CQ 0;
        else if(in_ack_12_2&& out_ack_12_2)
            tail_12_2 <= `CQ tail_12_2 + 512/128               - 1152/128          ;
        else if(in_ack_12_2)
            tail_12_2 <= `CQ tail_12_2 + 512/128              ;
        else if(out_ack_12_2)
            tail_12_2 <= `CQ tail_12_2 - 1152/128          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_12_3 <= `CQ 0;
        else if(in_ack_12_3&& out_ack_12_3)
            tail_12_3 <= `CQ tail_12_3 + 512/128               - 1152/128          ;
        else if(in_ack_12_3)
            tail_12_3 <= `CQ tail_12_3 + 512/128              ;
        else if(out_ack_12_3)
            tail_12_3 <= `CQ tail_12_3 - 1152/128          ;

//split bufferk to 32bit, for fanout timing

    always@(posedge clk)
        if(in_ack_0_0 && out_ack_0_0) begin
            if(0+1152/128           < tail_0_0)
                buffer0[32*0+:32] <= `CQ buffer9[32*0+:32];
            if(0+1152/128           >= tail_0_0+0 && 0+1152/128           < tail_0_0+0+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(0+1152/128           >= tail_0_0+1 && 0+1152/128           < tail_0_0+1+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(0+1152/128           >= tail_0_0+2 && 0+1152/128           < tail_0_0+2+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(0+1152/128           >= tail_0_0+3 && 0+1152/128           < tail_0_0+3+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_0_0) begin
            if(0>= tail_0_0+0 && 0 < tail_0_0+0+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(0>= tail_0_0+1 && 0 < tail_0_0+1+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(0>= tail_0_0+2 && 0 < tail_0_0+2+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(0>= tail_0_0+3 && 0 < tail_0_0+3+1)
                buffer0[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_0_0) begin
            if(0+1152/128           < tail_0_0)
                buffer0[32*0+:32] <= `CQ buffer9[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_1_0 && out_ack_1_0) begin
            if(1+1152/128           < tail_1_0)
                buffer1[32*0+:32] <= `CQ buffer10[32*0+:32];
            if(1+1152/128           >= tail_1_0+0 && 1+1152/128           < tail_1_0+0+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(1+1152/128           >= tail_1_0+1 && 1+1152/128           < tail_1_0+1+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(1+1152/128           >= tail_1_0+2 && 1+1152/128           < tail_1_0+2+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(1+1152/128           >= tail_1_0+3 && 1+1152/128           < tail_1_0+3+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_1_0) begin
            if(1>= tail_1_0+0 && 1 < tail_1_0+0+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(1>= tail_1_0+1 && 1 < tail_1_0+1+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(1>= tail_1_0+2 && 1 < tail_1_0+2+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(1>= tail_1_0+3 && 1 < tail_1_0+3+1)
                buffer1[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_1_0) begin
            if(1+1152/128           < tail_1_0)
                buffer1[32*0+:32] <= `CQ buffer10[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_2_0 && out_ack_2_0) begin
            if(2+1152/128           < tail_2_0)
                buffer2[32*0+:32] <= `CQ buffer11[32*0+:32];
            if(2+1152/128           >= tail_2_0+0 && 2+1152/128           < tail_2_0+0+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(2+1152/128           >= tail_2_0+1 && 2+1152/128           < tail_2_0+1+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(2+1152/128           >= tail_2_0+2 && 2+1152/128           < tail_2_0+2+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(2+1152/128           >= tail_2_0+3 && 2+1152/128           < tail_2_0+3+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_2_0) begin
            if(2>= tail_2_0+0 && 2 < tail_2_0+0+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(2>= tail_2_0+1 && 2 < tail_2_0+1+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(2>= tail_2_0+2 && 2 < tail_2_0+2+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(2>= tail_2_0+3 && 2 < tail_2_0+3+1)
                buffer2[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_2_0) begin
            if(2+1152/128           < tail_2_0)
                buffer2[32*0+:32] <= `CQ buffer11[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_3_0 && out_ack_3_0) begin
            if(3+1152/128           < tail_3_0)
                buffer3[32*0+:32] <= `CQ buffer12[32*0+:32];
            if(3+1152/128           >= tail_3_0+0 && 3+1152/128           < tail_3_0+0+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(3+1152/128           >= tail_3_0+1 && 3+1152/128           < tail_3_0+1+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(3+1152/128           >= tail_3_0+2 && 3+1152/128           < tail_3_0+2+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(3+1152/128           >= tail_3_0+3 && 3+1152/128           < tail_3_0+3+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_3_0) begin
            if(3>= tail_3_0+0 && 3 < tail_3_0+0+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(3>= tail_3_0+1 && 3 < tail_3_0+1+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(3>= tail_3_0+2 && 3 < tail_3_0+2+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(3>= tail_3_0+3 && 3 < tail_3_0+3+1)
                buffer3[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_3_0) begin
            if(3+1152/128           < tail_3_0)
                buffer3[32*0+:32] <= `CQ buffer12[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_4_0 && out_ack_4_0) begin
            if(4+1152/128           < tail_4_0)
                buffer4[32*0+:32] <= `CQ buffer13[32*0+:32];
            if(4+1152/128           >= tail_4_0+0 && 4+1152/128           < tail_4_0+0+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(4+1152/128           >= tail_4_0+1 && 4+1152/128           < tail_4_0+1+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(4+1152/128           >= tail_4_0+2 && 4+1152/128           < tail_4_0+2+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(4+1152/128           >= tail_4_0+3 && 4+1152/128           < tail_4_0+3+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_4_0) begin
            if(4>= tail_4_0+0 && 4 < tail_4_0+0+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(4>= tail_4_0+1 && 4 < tail_4_0+1+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(4>= tail_4_0+2 && 4 < tail_4_0+2+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(4>= tail_4_0+3 && 4 < tail_4_0+3+1)
                buffer4[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_4_0) begin
            if(4+1152/128           < tail_4_0)
                buffer4[32*0+:32] <= `CQ buffer13[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_5_0 && out_ack_5_0) begin
            if(5+1152/128           < tail_5_0)
                buffer5[32*0+:32] <= `CQ buffer14[32*0+:32];
            if(5+1152/128           >= tail_5_0+0 && 5+1152/128           < tail_5_0+0+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(5+1152/128           >= tail_5_0+1 && 5+1152/128           < tail_5_0+1+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(5+1152/128           >= tail_5_0+2 && 5+1152/128           < tail_5_0+2+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(5+1152/128           >= tail_5_0+3 && 5+1152/128           < tail_5_0+3+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_5_0) begin
            if(5>= tail_5_0+0 && 5 < tail_5_0+0+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(5>= tail_5_0+1 && 5 < tail_5_0+1+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(5>= tail_5_0+2 && 5 < tail_5_0+2+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(5>= tail_5_0+3 && 5 < tail_5_0+3+1)
                buffer5[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_5_0) begin
            if(5+1152/128           < tail_5_0)
                buffer5[32*0+:32] <= `CQ buffer14[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_6_0 && out_ack_6_0) begin
            if(6+1152/128           < tail_6_0)
                buffer6[32*0+:32] <= `CQ buffer15[32*0+:32];
            if(6+1152/128           >= tail_6_0+0 && 6+1152/128           < tail_6_0+0+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(6+1152/128           >= tail_6_0+1 && 6+1152/128           < tail_6_0+1+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(6+1152/128           >= tail_6_0+2 && 6+1152/128           < tail_6_0+2+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(6+1152/128           >= tail_6_0+3 && 6+1152/128           < tail_6_0+3+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_6_0) begin
            if(6>= tail_6_0+0 && 6 < tail_6_0+0+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(6>= tail_6_0+1 && 6 < tail_6_0+1+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(6>= tail_6_0+2 && 6 < tail_6_0+2+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(6>= tail_6_0+3 && 6 < tail_6_0+3+1)
                buffer6[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_6_0) begin
            if(6+1152/128           < tail_6_0)
                buffer6[32*0+:32] <= `CQ buffer15[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_7_0 && out_ack_7_0) begin
            if(7+1152/128           < tail_7_0)
                buffer7[32*0+:32] <= `CQ buffer16[32*0+:32];
            if(7+1152/128           >= tail_7_0+0 && 7+1152/128           < tail_7_0+0+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(7+1152/128           >= tail_7_0+1 && 7+1152/128           < tail_7_0+1+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(7+1152/128           >= tail_7_0+2 && 7+1152/128           < tail_7_0+2+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(7+1152/128           >= tail_7_0+3 && 7+1152/128           < tail_7_0+3+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_7_0) begin
            if(7>= tail_7_0+0 && 7 < tail_7_0+0+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(7>= tail_7_0+1 && 7 < tail_7_0+1+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(7>= tail_7_0+2 && 7 < tail_7_0+2+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(7>= tail_7_0+3 && 7 < tail_7_0+3+1)
                buffer7[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_7_0) begin
            if(7+1152/128           < tail_7_0)
                buffer7[32*0+:32] <= `CQ buffer16[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_8_0 && out_ack_8_0) begin
            if(8+1152/128           < tail_8_0)
                buffer8[32*0+:32] <= `CQ buffer17[32*0+:32];
            if(8+1152/128           >= tail_8_0+0 && 8+1152/128           < tail_8_0+0+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(8+1152/128           >= tail_8_0+1 && 8+1152/128           < tail_8_0+1+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(8+1152/128           >= tail_8_0+2 && 8+1152/128           < tail_8_0+2+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(8+1152/128           >= tail_8_0+3 && 8+1152/128           < tail_8_0+3+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_8_0) begin
            if(8>= tail_8_0+0 && 8 < tail_8_0+0+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(8>= tail_8_0+1 && 8 < tail_8_0+1+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(8>= tail_8_0+2 && 8 < tail_8_0+2+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(8>= tail_8_0+3 && 8 < tail_8_0+3+1)
                buffer8[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_8_0) begin
            if(8+1152/128           < tail_8_0)
                buffer8[32*0+:32] <= `CQ buffer17[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_9_0 && out_ack_9_0) begin
            if(9+1152/128           < tail_9_0)
                buffer9[32*0+:32] <= `CQ buffer18[32*0+:32];
            if(9+1152/128           >= tail_9_0+0 && 9+1152/128           < tail_9_0+0+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(9+1152/128           >= tail_9_0+1 && 9+1152/128           < tail_9_0+1+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(9+1152/128           >= tail_9_0+2 && 9+1152/128           < tail_9_0+2+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(9+1152/128           >= tail_9_0+3 && 9+1152/128           < tail_9_0+3+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_9_0) begin
            if(9>= tail_9_0+0 && 9 < tail_9_0+0+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(9>= tail_9_0+1 && 9 < tail_9_0+1+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(9>= tail_9_0+2 && 9 < tail_9_0+2+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(9>= tail_9_0+3 && 9 < tail_9_0+3+1)
                buffer9[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_9_0) begin
            if(9+1152/128           < tail_9_0)
                buffer9[32*0+:32] <= `CQ buffer18[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_10_0 && out_ack_10_0) begin
            if(10+1152/128           < tail_10_0)
                buffer10[32*0+:32] <= `CQ buffer19[32*0+:32];
            if(10+1152/128           >= tail_10_0+0 && 10+1152/128           < tail_10_0+0+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(10+1152/128           >= tail_10_0+1 && 10+1152/128           < tail_10_0+1+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(10+1152/128           >= tail_10_0+2 && 10+1152/128           < tail_10_0+2+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(10+1152/128           >= tail_10_0+3 && 10+1152/128           < tail_10_0+3+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_10_0) begin
            if(10>= tail_10_0+0 && 10 < tail_10_0+0+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(10>= tail_10_0+1 && 10 < tail_10_0+1+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(10>= tail_10_0+2 && 10 < tail_10_0+2+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(10>= tail_10_0+3 && 10 < tail_10_0+3+1)
                buffer10[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_10_0) begin
            if(10+1152/128           < tail_10_0)
                buffer10[32*0+:32] <= `CQ buffer19[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_11_0 && out_ack_11_0) begin
            if(11+1152/128           < tail_11_0)
                buffer11[32*0+:32] <= `CQ buffer20[32*0+:32];
            if(11+1152/128           >= tail_11_0+0 && 11+1152/128           < tail_11_0+0+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(11+1152/128           >= tail_11_0+1 && 11+1152/128           < tail_11_0+1+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(11+1152/128           >= tail_11_0+2 && 11+1152/128           < tail_11_0+2+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(11+1152/128           >= tail_11_0+3 && 11+1152/128           < tail_11_0+3+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_11_0) begin
            if(11>= tail_11_0+0 && 11 < tail_11_0+0+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(11>= tail_11_0+1 && 11 < tail_11_0+1+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(11>= tail_11_0+2 && 11 < tail_11_0+2+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(11>= tail_11_0+3 && 11 < tail_11_0+3+1)
                buffer11[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_11_0) begin
            if(11+1152/128           < tail_11_0)
                buffer11[32*0+:32] <= `CQ buffer20[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_12_0 && out_ack_12_0) begin
            if(12+1152/128           < tail_12_0)
                buffer12[32*0+:32] <= `CQ buffer21[32*0+:32];
            if(12+1152/128           >= tail_12_0+0 && 12+1152/128           < tail_12_0+0+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(12+1152/128           >= tail_12_0+1 && 12+1152/128           < tail_12_0+1+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(12+1152/128           >= tail_12_0+2 && 12+1152/128           < tail_12_0+2+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(12+1152/128           >= tail_12_0+3 && 12+1152/128           < tail_12_0+3+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(in_ack_12_0) begin
            if(12>= tail_12_0+0 && 12 < tail_12_0+0+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *0+32*0+:32];
            if(12>= tail_12_0+1 && 12 < tail_12_0+1+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *1+32*0+:32];
            if(12>= tail_12_0+2 && 12 < tail_12_0+2+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *2+32*0+:32];
            if(12>= tail_12_0+3 && 12 < tail_12_0+3+1)
                buffer12[32*0+:32] <= `CQ in_data[128       *3+32*0+:32];
        end
        else if(out_ack_12_0) begin
            if(12+1152/128           < tail_12_0)
                buffer12[32*0+:32] <= `CQ buffer21[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_0_1 && out_ack_0_1) begin
            if(0+1152/128           < tail_0_1)
                buffer0[32*1+:32] <= `CQ buffer9[32*1+:32];
            if(0+1152/128           >= tail_0_1+0 && 0+1152/128           < tail_0_1+0+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(0+1152/128           >= tail_0_1+1 && 0+1152/128           < tail_0_1+1+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(0+1152/128           >= tail_0_1+2 && 0+1152/128           < tail_0_1+2+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(0+1152/128           >= tail_0_1+3 && 0+1152/128           < tail_0_1+3+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_0_1) begin
            if(0>= tail_0_1+0 && 0 < tail_0_1+0+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(0>= tail_0_1+1 && 0 < tail_0_1+1+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(0>= tail_0_1+2 && 0 < tail_0_1+2+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(0>= tail_0_1+3 && 0 < tail_0_1+3+1)
                buffer0[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_0_1) begin
            if(0+1152/128           < tail_0_1)
                buffer0[32*1+:32] <= `CQ buffer9[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_1_1 && out_ack_1_1) begin
            if(1+1152/128           < tail_1_1)
                buffer1[32*1+:32] <= `CQ buffer10[32*1+:32];
            if(1+1152/128           >= tail_1_1+0 && 1+1152/128           < tail_1_1+0+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(1+1152/128           >= tail_1_1+1 && 1+1152/128           < tail_1_1+1+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(1+1152/128           >= tail_1_1+2 && 1+1152/128           < tail_1_1+2+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(1+1152/128           >= tail_1_1+3 && 1+1152/128           < tail_1_1+3+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_1_1) begin
            if(1>= tail_1_1+0 && 1 < tail_1_1+0+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(1>= tail_1_1+1 && 1 < tail_1_1+1+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(1>= tail_1_1+2 && 1 < tail_1_1+2+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(1>= tail_1_1+3 && 1 < tail_1_1+3+1)
                buffer1[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_1_1) begin
            if(1+1152/128           < tail_1_1)
                buffer1[32*1+:32] <= `CQ buffer10[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_2_1 && out_ack_2_1) begin
            if(2+1152/128           < tail_2_1)
                buffer2[32*1+:32] <= `CQ buffer11[32*1+:32];
            if(2+1152/128           >= tail_2_1+0 && 2+1152/128           < tail_2_1+0+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(2+1152/128           >= tail_2_1+1 && 2+1152/128           < tail_2_1+1+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(2+1152/128           >= tail_2_1+2 && 2+1152/128           < tail_2_1+2+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(2+1152/128           >= tail_2_1+3 && 2+1152/128           < tail_2_1+3+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_2_1) begin
            if(2>= tail_2_1+0 && 2 < tail_2_1+0+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(2>= tail_2_1+1 && 2 < tail_2_1+1+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(2>= tail_2_1+2 && 2 < tail_2_1+2+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(2>= tail_2_1+3 && 2 < tail_2_1+3+1)
                buffer2[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_2_1) begin
            if(2+1152/128           < tail_2_1)
                buffer2[32*1+:32] <= `CQ buffer11[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_3_1 && out_ack_3_1) begin
            if(3+1152/128           < tail_3_1)
                buffer3[32*1+:32] <= `CQ buffer12[32*1+:32];
            if(3+1152/128           >= tail_3_1+0 && 3+1152/128           < tail_3_1+0+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(3+1152/128           >= tail_3_1+1 && 3+1152/128           < tail_3_1+1+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(3+1152/128           >= tail_3_1+2 && 3+1152/128           < tail_3_1+2+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(3+1152/128           >= tail_3_1+3 && 3+1152/128           < tail_3_1+3+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_3_1) begin
            if(3>= tail_3_1+0 && 3 < tail_3_1+0+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(3>= tail_3_1+1 && 3 < tail_3_1+1+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(3>= tail_3_1+2 && 3 < tail_3_1+2+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(3>= tail_3_1+3 && 3 < tail_3_1+3+1)
                buffer3[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_3_1) begin
            if(3+1152/128           < tail_3_1)
                buffer3[32*1+:32] <= `CQ buffer12[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_4_1 && out_ack_4_1) begin
            if(4+1152/128           < tail_4_1)
                buffer4[32*1+:32] <= `CQ buffer13[32*1+:32];
            if(4+1152/128           >= tail_4_1+0 && 4+1152/128           < tail_4_1+0+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(4+1152/128           >= tail_4_1+1 && 4+1152/128           < tail_4_1+1+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(4+1152/128           >= tail_4_1+2 && 4+1152/128           < tail_4_1+2+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(4+1152/128           >= tail_4_1+3 && 4+1152/128           < tail_4_1+3+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_4_1) begin
            if(4>= tail_4_1+0 && 4 < tail_4_1+0+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(4>= tail_4_1+1 && 4 < tail_4_1+1+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(4>= tail_4_1+2 && 4 < tail_4_1+2+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(4>= tail_4_1+3 && 4 < tail_4_1+3+1)
                buffer4[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_4_1) begin
            if(4+1152/128           < tail_4_1)
                buffer4[32*1+:32] <= `CQ buffer13[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_5_1 && out_ack_5_1) begin
            if(5+1152/128           < tail_5_1)
                buffer5[32*1+:32] <= `CQ buffer14[32*1+:32];
            if(5+1152/128           >= tail_5_1+0 && 5+1152/128           < tail_5_1+0+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(5+1152/128           >= tail_5_1+1 && 5+1152/128           < tail_5_1+1+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(5+1152/128           >= tail_5_1+2 && 5+1152/128           < tail_5_1+2+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(5+1152/128           >= tail_5_1+3 && 5+1152/128           < tail_5_1+3+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_5_1) begin
            if(5>= tail_5_1+0 && 5 < tail_5_1+0+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(5>= tail_5_1+1 && 5 < tail_5_1+1+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(5>= tail_5_1+2 && 5 < tail_5_1+2+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(5>= tail_5_1+3 && 5 < tail_5_1+3+1)
                buffer5[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_5_1) begin
            if(5+1152/128           < tail_5_1)
                buffer5[32*1+:32] <= `CQ buffer14[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_6_1 && out_ack_6_1) begin
            if(6+1152/128           < tail_6_1)
                buffer6[32*1+:32] <= `CQ buffer15[32*1+:32];
            if(6+1152/128           >= tail_6_1+0 && 6+1152/128           < tail_6_1+0+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(6+1152/128           >= tail_6_1+1 && 6+1152/128           < tail_6_1+1+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(6+1152/128           >= tail_6_1+2 && 6+1152/128           < tail_6_1+2+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(6+1152/128           >= tail_6_1+3 && 6+1152/128           < tail_6_1+3+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_6_1) begin
            if(6>= tail_6_1+0 && 6 < tail_6_1+0+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(6>= tail_6_1+1 && 6 < tail_6_1+1+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(6>= tail_6_1+2 && 6 < tail_6_1+2+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(6>= tail_6_1+3 && 6 < tail_6_1+3+1)
                buffer6[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_6_1) begin
            if(6+1152/128           < tail_6_1)
                buffer6[32*1+:32] <= `CQ buffer15[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_7_1 && out_ack_7_1) begin
            if(7+1152/128           < tail_7_1)
                buffer7[32*1+:32] <= `CQ buffer16[32*1+:32];
            if(7+1152/128           >= tail_7_1+0 && 7+1152/128           < tail_7_1+0+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(7+1152/128           >= tail_7_1+1 && 7+1152/128           < tail_7_1+1+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(7+1152/128           >= tail_7_1+2 && 7+1152/128           < tail_7_1+2+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(7+1152/128           >= tail_7_1+3 && 7+1152/128           < tail_7_1+3+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_7_1) begin
            if(7>= tail_7_1+0 && 7 < tail_7_1+0+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(7>= tail_7_1+1 && 7 < tail_7_1+1+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(7>= tail_7_1+2 && 7 < tail_7_1+2+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(7>= tail_7_1+3 && 7 < tail_7_1+3+1)
                buffer7[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_7_1) begin
            if(7+1152/128           < tail_7_1)
                buffer7[32*1+:32] <= `CQ buffer16[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_8_1 && out_ack_8_1) begin
            if(8+1152/128           < tail_8_1)
                buffer8[32*1+:32] <= `CQ buffer17[32*1+:32];
            if(8+1152/128           >= tail_8_1+0 && 8+1152/128           < tail_8_1+0+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(8+1152/128           >= tail_8_1+1 && 8+1152/128           < tail_8_1+1+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(8+1152/128           >= tail_8_1+2 && 8+1152/128           < tail_8_1+2+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(8+1152/128           >= tail_8_1+3 && 8+1152/128           < tail_8_1+3+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_8_1) begin
            if(8>= tail_8_1+0 && 8 < tail_8_1+0+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(8>= tail_8_1+1 && 8 < tail_8_1+1+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(8>= tail_8_1+2 && 8 < tail_8_1+2+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(8>= tail_8_1+3 && 8 < tail_8_1+3+1)
                buffer8[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_8_1) begin
            if(8+1152/128           < tail_8_1)
                buffer8[32*1+:32] <= `CQ buffer17[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_9_1 && out_ack_9_1) begin
            if(9+1152/128           < tail_9_1)
                buffer9[32*1+:32] <= `CQ buffer18[32*1+:32];
            if(9+1152/128           >= tail_9_1+0 && 9+1152/128           < tail_9_1+0+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(9+1152/128           >= tail_9_1+1 && 9+1152/128           < tail_9_1+1+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(9+1152/128           >= tail_9_1+2 && 9+1152/128           < tail_9_1+2+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(9+1152/128           >= tail_9_1+3 && 9+1152/128           < tail_9_1+3+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_9_1) begin
            if(9>= tail_9_1+0 && 9 < tail_9_1+0+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(9>= tail_9_1+1 && 9 < tail_9_1+1+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(9>= tail_9_1+2 && 9 < tail_9_1+2+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(9>= tail_9_1+3 && 9 < tail_9_1+3+1)
                buffer9[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_9_1) begin
            if(9+1152/128           < tail_9_1)
                buffer9[32*1+:32] <= `CQ buffer18[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_10_1 && out_ack_10_1) begin
            if(10+1152/128           < tail_10_1)
                buffer10[32*1+:32] <= `CQ buffer19[32*1+:32];
            if(10+1152/128           >= tail_10_1+0 && 10+1152/128           < tail_10_1+0+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(10+1152/128           >= tail_10_1+1 && 10+1152/128           < tail_10_1+1+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(10+1152/128           >= tail_10_1+2 && 10+1152/128           < tail_10_1+2+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(10+1152/128           >= tail_10_1+3 && 10+1152/128           < tail_10_1+3+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_10_1) begin
            if(10>= tail_10_1+0 && 10 < tail_10_1+0+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(10>= tail_10_1+1 && 10 < tail_10_1+1+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(10>= tail_10_1+2 && 10 < tail_10_1+2+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(10>= tail_10_1+3 && 10 < tail_10_1+3+1)
                buffer10[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_10_1) begin
            if(10+1152/128           < tail_10_1)
                buffer10[32*1+:32] <= `CQ buffer19[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_11_1 && out_ack_11_1) begin
            if(11+1152/128           < tail_11_1)
                buffer11[32*1+:32] <= `CQ buffer20[32*1+:32];
            if(11+1152/128           >= tail_11_1+0 && 11+1152/128           < tail_11_1+0+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(11+1152/128           >= tail_11_1+1 && 11+1152/128           < tail_11_1+1+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(11+1152/128           >= tail_11_1+2 && 11+1152/128           < tail_11_1+2+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(11+1152/128           >= tail_11_1+3 && 11+1152/128           < tail_11_1+3+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_11_1) begin
            if(11>= tail_11_1+0 && 11 < tail_11_1+0+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(11>= tail_11_1+1 && 11 < tail_11_1+1+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(11>= tail_11_1+2 && 11 < tail_11_1+2+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(11>= tail_11_1+3 && 11 < tail_11_1+3+1)
                buffer11[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_11_1) begin
            if(11+1152/128           < tail_11_1)
                buffer11[32*1+:32] <= `CQ buffer20[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_12_1 && out_ack_12_1) begin
            if(12+1152/128           < tail_12_1)
                buffer12[32*1+:32] <= `CQ buffer21[32*1+:32];
            if(12+1152/128           >= tail_12_1+0 && 12+1152/128           < tail_12_1+0+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(12+1152/128           >= tail_12_1+1 && 12+1152/128           < tail_12_1+1+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(12+1152/128           >= tail_12_1+2 && 12+1152/128           < tail_12_1+2+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(12+1152/128           >= tail_12_1+3 && 12+1152/128           < tail_12_1+3+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(in_ack_12_1) begin
            if(12>= tail_12_1+0 && 12 < tail_12_1+0+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *0+32*1+:32];
            if(12>= tail_12_1+1 && 12 < tail_12_1+1+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *1+32*1+:32];
            if(12>= tail_12_1+2 && 12 < tail_12_1+2+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *2+32*1+:32];
            if(12>= tail_12_1+3 && 12 < tail_12_1+3+1)
                buffer12[32*1+:32] <= `CQ in_data[128       *3+32*1+:32];
        end
        else if(out_ack_12_1) begin
            if(12+1152/128           < tail_12_1)
                buffer12[32*1+:32] <= `CQ buffer21[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_0_2 && out_ack_0_2) begin
            if(0+1152/128           < tail_0_2)
                buffer0[32*2+:32] <= `CQ buffer9[32*2+:32];
            if(0+1152/128           >= tail_0_2+0 && 0+1152/128           < tail_0_2+0+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(0+1152/128           >= tail_0_2+1 && 0+1152/128           < tail_0_2+1+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(0+1152/128           >= tail_0_2+2 && 0+1152/128           < tail_0_2+2+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(0+1152/128           >= tail_0_2+3 && 0+1152/128           < tail_0_2+3+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_0_2) begin
            if(0>= tail_0_2+0 && 0 < tail_0_2+0+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(0>= tail_0_2+1 && 0 < tail_0_2+1+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(0>= tail_0_2+2 && 0 < tail_0_2+2+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(0>= tail_0_2+3 && 0 < tail_0_2+3+1)
                buffer0[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_0_2) begin
            if(0+1152/128           < tail_0_2)
                buffer0[32*2+:32] <= `CQ buffer9[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_1_2 && out_ack_1_2) begin
            if(1+1152/128           < tail_1_2)
                buffer1[32*2+:32] <= `CQ buffer10[32*2+:32];
            if(1+1152/128           >= tail_1_2+0 && 1+1152/128           < tail_1_2+0+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(1+1152/128           >= tail_1_2+1 && 1+1152/128           < tail_1_2+1+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(1+1152/128           >= tail_1_2+2 && 1+1152/128           < tail_1_2+2+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(1+1152/128           >= tail_1_2+3 && 1+1152/128           < tail_1_2+3+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_1_2) begin
            if(1>= tail_1_2+0 && 1 < tail_1_2+0+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(1>= tail_1_2+1 && 1 < tail_1_2+1+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(1>= tail_1_2+2 && 1 < tail_1_2+2+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(1>= tail_1_2+3 && 1 < tail_1_2+3+1)
                buffer1[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_1_2) begin
            if(1+1152/128           < tail_1_2)
                buffer1[32*2+:32] <= `CQ buffer10[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_2_2 && out_ack_2_2) begin
            if(2+1152/128           < tail_2_2)
                buffer2[32*2+:32] <= `CQ buffer11[32*2+:32];
            if(2+1152/128           >= tail_2_2+0 && 2+1152/128           < tail_2_2+0+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(2+1152/128           >= tail_2_2+1 && 2+1152/128           < tail_2_2+1+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(2+1152/128           >= tail_2_2+2 && 2+1152/128           < tail_2_2+2+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(2+1152/128           >= tail_2_2+3 && 2+1152/128           < tail_2_2+3+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_2_2) begin
            if(2>= tail_2_2+0 && 2 < tail_2_2+0+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(2>= tail_2_2+1 && 2 < tail_2_2+1+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(2>= tail_2_2+2 && 2 < tail_2_2+2+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(2>= tail_2_2+3 && 2 < tail_2_2+3+1)
                buffer2[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_2_2) begin
            if(2+1152/128           < tail_2_2)
                buffer2[32*2+:32] <= `CQ buffer11[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_3_2 && out_ack_3_2) begin
            if(3+1152/128           < tail_3_2)
                buffer3[32*2+:32] <= `CQ buffer12[32*2+:32];
            if(3+1152/128           >= tail_3_2+0 && 3+1152/128           < tail_3_2+0+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(3+1152/128           >= tail_3_2+1 && 3+1152/128           < tail_3_2+1+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(3+1152/128           >= tail_3_2+2 && 3+1152/128           < tail_3_2+2+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(3+1152/128           >= tail_3_2+3 && 3+1152/128           < tail_3_2+3+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_3_2) begin
            if(3>= tail_3_2+0 && 3 < tail_3_2+0+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(3>= tail_3_2+1 && 3 < tail_3_2+1+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(3>= tail_3_2+2 && 3 < tail_3_2+2+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(3>= tail_3_2+3 && 3 < tail_3_2+3+1)
                buffer3[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_3_2) begin
            if(3+1152/128           < tail_3_2)
                buffer3[32*2+:32] <= `CQ buffer12[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_4_2 && out_ack_4_2) begin
            if(4+1152/128           < tail_4_2)
                buffer4[32*2+:32] <= `CQ buffer13[32*2+:32];
            if(4+1152/128           >= tail_4_2+0 && 4+1152/128           < tail_4_2+0+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(4+1152/128           >= tail_4_2+1 && 4+1152/128           < tail_4_2+1+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(4+1152/128           >= tail_4_2+2 && 4+1152/128           < tail_4_2+2+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(4+1152/128           >= tail_4_2+3 && 4+1152/128           < tail_4_2+3+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_4_2) begin
            if(4>= tail_4_2+0 && 4 < tail_4_2+0+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(4>= tail_4_2+1 && 4 < tail_4_2+1+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(4>= tail_4_2+2 && 4 < tail_4_2+2+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(4>= tail_4_2+3 && 4 < tail_4_2+3+1)
                buffer4[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_4_2) begin
            if(4+1152/128           < tail_4_2)
                buffer4[32*2+:32] <= `CQ buffer13[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_5_2 && out_ack_5_2) begin
            if(5+1152/128           < tail_5_2)
                buffer5[32*2+:32] <= `CQ buffer14[32*2+:32];
            if(5+1152/128           >= tail_5_2+0 && 5+1152/128           < tail_5_2+0+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(5+1152/128           >= tail_5_2+1 && 5+1152/128           < tail_5_2+1+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(5+1152/128           >= tail_5_2+2 && 5+1152/128           < tail_5_2+2+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(5+1152/128           >= tail_5_2+3 && 5+1152/128           < tail_5_2+3+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_5_2) begin
            if(5>= tail_5_2+0 && 5 < tail_5_2+0+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(5>= tail_5_2+1 && 5 < tail_5_2+1+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(5>= tail_5_2+2 && 5 < tail_5_2+2+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(5>= tail_5_2+3 && 5 < tail_5_2+3+1)
                buffer5[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_5_2) begin
            if(5+1152/128           < tail_5_2)
                buffer5[32*2+:32] <= `CQ buffer14[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_6_2 && out_ack_6_2) begin
            if(6+1152/128           < tail_6_2)
                buffer6[32*2+:32] <= `CQ buffer15[32*2+:32];
            if(6+1152/128           >= tail_6_2+0 && 6+1152/128           < tail_6_2+0+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(6+1152/128           >= tail_6_2+1 && 6+1152/128           < tail_6_2+1+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(6+1152/128           >= tail_6_2+2 && 6+1152/128           < tail_6_2+2+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(6+1152/128           >= tail_6_2+3 && 6+1152/128           < tail_6_2+3+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_6_2) begin
            if(6>= tail_6_2+0 && 6 < tail_6_2+0+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(6>= tail_6_2+1 && 6 < tail_6_2+1+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(6>= tail_6_2+2 && 6 < tail_6_2+2+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(6>= tail_6_2+3 && 6 < tail_6_2+3+1)
                buffer6[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_6_2) begin
            if(6+1152/128           < tail_6_2)
                buffer6[32*2+:32] <= `CQ buffer15[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_7_2 && out_ack_7_2) begin
            if(7+1152/128           < tail_7_2)
                buffer7[32*2+:32] <= `CQ buffer16[32*2+:32];
            if(7+1152/128           >= tail_7_2+0 && 7+1152/128           < tail_7_2+0+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(7+1152/128           >= tail_7_2+1 && 7+1152/128           < tail_7_2+1+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(7+1152/128           >= tail_7_2+2 && 7+1152/128           < tail_7_2+2+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(7+1152/128           >= tail_7_2+3 && 7+1152/128           < tail_7_2+3+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_7_2) begin
            if(7>= tail_7_2+0 && 7 < tail_7_2+0+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(7>= tail_7_2+1 && 7 < tail_7_2+1+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(7>= tail_7_2+2 && 7 < tail_7_2+2+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(7>= tail_7_2+3 && 7 < tail_7_2+3+1)
                buffer7[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_7_2) begin
            if(7+1152/128           < tail_7_2)
                buffer7[32*2+:32] <= `CQ buffer16[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_8_2 && out_ack_8_2) begin
            if(8+1152/128           < tail_8_2)
                buffer8[32*2+:32] <= `CQ buffer17[32*2+:32];
            if(8+1152/128           >= tail_8_2+0 && 8+1152/128           < tail_8_2+0+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(8+1152/128           >= tail_8_2+1 && 8+1152/128           < tail_8_2+1+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(8+1152/128           >= tail_8_2+2 && 8+1152/128           < tail_8_2+2+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(8+1152/128           >= tail_8_2+3 && 8+1152/128           < tail_8_2+3+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_8_2) begin
            if(8>= tail_8_2+0 && 8 < tail_8_2+0+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(8>= tail_8_2+1 && 8 < tail_8_2+1+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(8>= tail_8_2+2 && 8 < tail_8_2+2+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(8>= tail_8_2+3 && 8 < tail_8_2+3+1)
                buffer8[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_8_2) begin
            if(8+1152/128           < tail_8_2)
                buffer8[32*2+:32] <= `CQ buffer17[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_9_2 && out_ack_9_2) begin
            if(9+1152/128           < tail_9_2)
                buffer9[32*2+:32] <= `CQ buffer18[32*2+:32];
            if(9+1152/128           >= tail_9_2+0 && 9+1152/128           < tail_9_2+0+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(9+1152/128           >= tail_9_2+1 && 9+1152/128           < tail_9_2+1+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(9+1152/128           >= tail_9_2+2 && 9+1152/128           < tail_9_2+2+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(9+1152/128           >= tail_9_2+3 && 9+1152/128           < tail_9_2+3+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_9_2) begin
            if(9>= tail_9_2+0 && 9 < tail_9_2+0+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(9>= tail_9_2+1 && 9 < tail_9_2+1+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(9>= tail_9_2+2 && 9 < tail_9_2+2+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(9>= tail_9_2+3 && 9 < tail_9_2+3+1)
                buffer9[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_9_2) begin
            if(9+1152/128           < tail_9_2)
                buffer9[32*2+:32] <= `CQ buffer18[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_10_2 && out_ack_10_2) begin
            if(10+1152/128           < tail_10_2)
                buffer10[32*2+:32] <= `CQ buffer19[32*2+:32];
            if(10+1152/128           >= tail_10_2+0 && 10+1152/128           < tail_10_2+0+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(10+1152/128           >= tail_10_2+1 && 10+1152/128           < tail_10_2+1+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(10+1152/128           >= tail_10_2+2 && 10+1152/128           < tail_10_2+2+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(10+1152/128           >= tail_10_2+3 && 10+1152/128           < tail_10_2+3+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_10_2) begin
            if(10>= tail_10_2+0 && 10 < tail_10_2+0+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(10>= tail_10_2+1 && 10 < tail_10_2+1+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(10>= tail_10_2+2 && 10 < tail_10_2+2+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(10>= tail_10_2+3 && 10 < tail_10_2+3+1)
                buffer10[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_10_2) begin
            if(10+1152/128           < tail_10_2)
                buffer10[32*2+:32] <= `CQ buffer19[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_11_2 && out_ack_11_2) begin
            if(11+1152/128           < tail_11_2)
                buffer11[32*2+:32] <= `CQ buffer20[32*2+:32];
            if(11+1152/128           >= tail_11_2+0 && 11+1152/128           < tail_11_2+0+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(11+1152/128           >= tail_11_2+1 && 11+1152/128           < tail_11_2+1+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(11+1152/128           >= tail_11_2+2 && 11+1152/128           < tail_11_2+2+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(11+1152/128           >= tail_11_2+3 && 11+1152/128           < tail_11_2+3+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_11_2) begin
            if(11>= tail_11_2+0 && 11 < tail_11_2+0+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(11>= tail_11_2+1 && 11 < tail_11_2+1+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(11>= tail_11_2+2 && 11 < tail_11_2+2+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(11>= tail_11_2+3 && 11 < tail_11_2+3+1)
                buffer11[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_11_2) begin
            if(11+1152/128           < tail_11_2)
                buffer11[32*2+:32] <= `CQ buffer20[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_12_2 && out_ack_12_2) begin
            if(12+1152/128           < tail_12_2)
                buffer12[32*2+:32] <= `CQ buffer21[32*2+:32];
            if(12+1152/128           >= tail_12_2+0 && 12+1152/128           < tail_12_2+0+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(12+1152/128           >= tail_12_2+1 && 12+1152/128           < tail_12_2+1+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(12+1152/128           >= tail_12_2+2 && 12+1152/128           < tail_12_2+2+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(12+1152/128           >= tail_12_2+3 && 12+1152/128           < tail_12_2+3+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(in_ack_12_2) begin
            if(12>= tail_12_2+0 && 12 < tail_12_2+0+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *0+32*2+:32];
            if(12>= tail_12_2+1 && 12 < tail_12_2+1+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *1+32*2+:32];
            if(12>= tail_12_2+2 && 12 < tail_12_2+2+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *2+32*2+:32];
            if(12>= tail_12_2+3 && 12 < tail_12_2+3+1)
                buffer12[32*2+:32] <= `CQ in_data[128       *3+32*2+:32];
        end
        else if(out_ack_12_2) begin
            if(12+1152/128           < tail_12_2)
                buffer12[32*2+:32] <= `CQ buffer21[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_0_3 && out_ack_0_3) begin
            if(0+1152/128           < tail_0_3)
                buffer0[32*3+:32] <= `CQ buffer9[32*3+:32];
            if(0+1152/128           >= tail_0_3+0 && 0+1152/128           < tail_0_3+0+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(0+1152/128           >= tail_0_3+1 && 0+1152/128           < tail_0_3+1+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(0+1152/128           >= tail_0_3+2 && 0+1152/128           < tail_0_3+2+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(0+1152/128           >= tail_0_3+3 && 0+1152/128           < tail_0_3+3+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_0_3) begin
            if(0>= tail_0_3+0 && 0 < tail_0_3+0+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(0>= tail_0_3+1 && 0 < tail_0_3+1+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(0>= tail_0_3+2 && 0 < tail_0_3+2+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(0>= tail_0_3+3 && 0 < tail_0_3+3+1)
                buffer0[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_0_3) begin
            if(0+1152/128           < tail_0_3)
                buffer0[32*3+:32] <= `CQ buffer9[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_1_3 && out_ack_1_3) begin
            if(1+1152/128           < tail_1_3)
                buffer1[32*3+:32] <= `CQ buffer10[32*3+:32];
            if(1+1152/128           >= tail_1_3+0 && 1+1152/128           < tail_1_3+0+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(1+1152/128           >= tail_1_3+1 && 1+1152/128           < tail_1_3+1+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(1+1152/128           >= tail_1_3+2 && 1+1152/128           < tail_1_3+2+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(1+1152/128           >= tail_1_3+3 && 1+1152/128           < tail_1_3+3+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_1_3) begin
            if(1>= tail_1_3+0 && 1 < tail_1_3+0+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(1>= tail_1_3+1 && 1 < tail_1_3+1+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(1>= tail_1_3+2 && 1 < tail_1_3+2+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(1>= tail_1_3+3 && 1 < tail_1_3+3+1)
                buffer1[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_1_3) begin
            if(1+1152/128           < tail_1_3)
                buffer1[32*3+:32] <= `CQ buffer10[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_2_3 && out_ack_2_3) begin
            if(2+1152/128           < tail_2_3)
                buffer2[32*3+:32] <= `CQ buffer11[32*3+:32];
            if(2+1152/128           >= tail_2_3+0 && 2+1152/128           < tail_2_3+0+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(2+1152/128           >= tail_2_3+1 && 2+1152/128           < tail_2_3+1+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(2+1152/128           >= tail_2_3+2 && 2+1152/128           < tail_2_3+2+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(2+1152/128           >= tail_2_3+3 && 2+1152/128           < tail_2_3+3+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_2_3) begin
            if(2>= tail_2_3+0 && 2 < tail_2_3+0+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(2>= tail_2_3+1 && 2 < tail_2_3+1+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(2>= tail_2_3+2 && 2 < tail_2_3+2+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(2>= tail_2_3+3 && 2 < tail_2_3+3+1)
                buffer2[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_2_3) begin
            if(2+1152/128           < tail_2_3)
                buffer2[32*3+:32] <= `CQ buffer11[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_3_3 && out_ack_3_3) begin
            if(3+1152/128           < tail_3_3)
                buffer3[32*3+:32] <= `CQ buffer12[32*3+:32];
            if(3+1152/128           >= tail_3_3+0 && 3+1152/128           < tail_3_3+0+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(3+1152/128           >= tail_3_3+1 && 3+1152/128           < tail_3_3+1+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(3+1152/128           >= tail_3_3+2 && 3+1152/128           < tail_3_3+2+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(3+1152/128           >= tail_3_3+3 && 3+1152/128           < tail_3_3+3+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_3_3) begin
            if(3>= tail_3_3+0 && 3 < tail_3_3+0+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(3>= tail_3_3+1 && 3 < tail_3_3+1+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(3>= tail_3_3+2 && 3 < tail_3_3+2+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(3>= tail_3_3+3 && 3 < tail_3_3+3+1)
                buffer3[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_3_3) begin
            if(3+1152/128           < tail_3_3)
                buffer3[32*3+:32] <= `CQ buffer12[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_4_3 && out_ack_4_3) begin
            if(4+1152/128           < tail_4_3)
                buffer4[32*3+:32] <= `CQ buffer13[32*3+:32];
            if(4+1152/128           >= tail_4_3+0 && 4+1152/128           < tail_4_3+0+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(4+1152/128           >= tail_4_3+1 && 4+1152/128           < tail_4_3+1+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(4+1152/128           >= tail_4_3+2 && 4+1152/128           < tail_4_3+2+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(4+1152/128           >= tail_4_3+3 && 4+1152/128           < tail_4_3+3+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_4_3) begin
            if(4>= tail_4_3+0 && 4 < tail_4_3+0+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(4>= tail_4_3+1 && 4 < tail_4_3+1+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(4>= tail_4_3+2 && 4 < tail_4_3+2+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(4>= tail_4_3+3 && 4 < tail_4_3+3+1)
                buffer4[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_4_3) begin
            if(4+1152/128           < tail_4_3)
                buffer4[32*3+:32] <= `CQ buffer13[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_5_3 && out_ack_5_3) begin
            if(5+1152/128           < tail_5_3)
                buffer5[32*3+:32] <= `CQ buffer14[32*3+:32];
            if(5+1152/128           >= tail_5_3+0 && 5+1152/128           < tail_5_3+0+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(5+1152/128           >= tail_5_3+1 && 5+1152/128           < tail_5_3+1+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(5+1152/128           >= tail_5_3+2 && 5+1152/128           < tail_5_3+2+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(5+1152/128           >= tail_5_3+3 && 5+1152/128           < tail_5_3+3+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_5_3) begin
            if(5>= tail_5_3+0 && 5 < tail_5_3+0+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(5>= tail_5_3+1 && 5 < tail_5_3+1+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(5>= tail_5_3+2 && 5 < tail_5_3+2+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(5>= tail_5_3+3 && 5 < tail_5_3+3+1)
                buffer5[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_5_3) begin
            if(5+1152/128           < tail_5_3)
                buffer5[32*3+:32] <= `CQ buffer14[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_6_3 && out_ack_6_3) begin
            if(6+1152/128           < tail_6_3)
                buffer6[32*3+:32] <= `CQ buffer15[32*3+:32];
            if(6+1152/128           >= tail_6_3+0 && 6+1152/128           < tail_6_3+0+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(6+1152/128           >= tail_6_3+1 && 6+1152/128           < tail_6_3+1+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(6+1152/128           >= tail_6_3+2 && 6+1152/128           < tail_6_3+2+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(6+1152/128           >= tail_6_3+3 && 6+1152/128           < tail_6_3+3+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_6_3) begin
            if(6>= tail_6_3+0 && 6 < tail_6_3+0+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(6>= tail_6_3+1 && 6 < tail_6_3+1+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(6>= tail_6_3+2 && 6 < tail_6_3+2+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(6>= tail_6_3+3 && 6 < tail_6_3+3+1)
                buffer6[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_6_3) begin
            if(6+1152/128           < tail_6_3)
                buffer6[32*3+:32] <= `CQ buffer15[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_7_3 && out_ack_7_3) begin
            if(7+1152/128           < tail_7_3)
                buffer7[32*3+:32] <= `CQ buffer16[32*3+:32];
            if(7+1152/128           >= tail_7_3+0 && 7+1152/128           < tail_7_3+0+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(7+1152/128           >= tail_7_3+1 && 7+1152/128           < tail_7_3+1+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(7+1152/128           >= tail_7_3+2 && 7+1152/128           < tail_7_3+2+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(7+1152/128           >= tail_7_3+3 && 7+1152/128           < tail_7_3+3+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_7_3) begin
            if(7>= tail_7_3+0 && 7 < tail_7_3+0+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(7>= tail_7_3+1 && 7 < tail_7_3+1+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(7>= tail_7_3+2 && 7 < tail_7_3+2+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(7>= tail_7_3+3 && 7 < tail_7_3+3+1)
                buffer7[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_7_3) begin
            if(7+1152/128           < tail_7_3)
                buffer7[32*3+:32] <= `CQ buffer16[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_8_3 && out_ack_8_3) begin
            if(8+1152/128           < tail_8_3)
                buffer8[32*3+:32] <= `CQ buffer17[32*3+:32];
            if(8+1152/128           >= tail_8_3+0 && 8+1152/128           < tail_8_3+0+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(8+1152/128           >= tail_8_3+1 && 8+1152/128           < tail_8_3+1+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(8+1152/128           >= tail_8_3+2 && 8+1152/128           < tail_8_3+2+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(8+1152/128           >= tail_8_3+3 && 8+1152/128           < tail_8_3+3+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_8_3) begin
            if(8>= tail_8_3+0 && 8 < tail_8_3+0+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(8>= tail_8_3+1 && 8 < tail_8_3+1+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(8>= tail_8_3+2 && 8 < tail_8_3+2+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(8>= tail_8_3+3 && 8 < tail_8_3+3+1)
                buffer8[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_8_3) begin
            if(8+1152/128           < tail_8_3)
                buffer8[32*3+:32] <= `CQ buffer17[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_9_3 && out_ack_9_3) begin
            if(9+1152/128           < tail_9_3)
                buffer9[32*3+:32] <= `CQ buffer18[32*3+:32];
            if(9+1152/128           >= tail_9_3+0 && 9+1152/128           < tail_9_3+0+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(9+1152/128           >= tail_9_3+1 && 9+1152/128           < tail_9_3+1+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(9+1152/128           >= tail_9_3+2 && 9+1152/128           < tail_9_3+2+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(9+1152/128           >= tail_9_3+3 && 9+1152/128           < tail_9_3+3+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_9_3) begin
            if(9>= tail_9_3+0 && 9 < tail_9_3+0+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(9>= tail_9_3+1 && 9 < tail_9_3+1+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(9>= tail_9_3+2 && 9 < tail_9_3+2+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(9>= tail_9_3+3 && 9 < tail_9_3+3+1)
                buffer9[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_9_3) begin
            if(9+1152/128           < tail_9_3)
                buffer9[32*3+:32] <= `CQ buffer18[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_10_3 && out_ack_10_3) begin
            if(10+1152/128           < tail_10_3)
                buffer10[32*3+:32] <= `CQ buffer19[32*3+:32];
            if(10+1152/128           >= tail_10_3+0 && 10+1152/128           < tail_10_3+0+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(10+1152/128           >= tail_10_3+1 && 10+1152/128           < tail_10_3+1+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(10+1152/128           >= tail_10_3+2 && 10+1152/128           < tail_10_3+2+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(10+1152/128           >= tail_10_3+3 && 10+1152/128           < tail_10_3+3+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_10_3) begin
            if(10>= tail_10_3+0 && 10 < tail_10_3+0+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(10>= tail_10_3+1 && 10 < tail_10_3+1+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(10>= tail_10_3+2 && 10 < tail_10_3+2+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(10>= tail_10_3+3 && 10 < tail_10_3+3+1)
                buffer10[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_10_3) begin
            if(10+1152/128           < tail_10_3)
                buffer10[32*3+:32] <= `CQ buffer19[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_11_3 && out_ack_11_3) begin
            if(11+1152/128           < tail_11_3)
                buffer11[32*3+:32] <= `CQ buffer20[32*3+:32];
            if(11+1152/128           >= tail_11_3+0 && 11+1152/128           < tail_11_3+0+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(11+1152/128           >= tail_11_3+1 && 11+1152/128           < tail_11_3+1+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(11+1152/128           >= tail_11_3+2 && 11+1152/128           < tail_11_3+2+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(11+1152/128           >= tail_11_3+3 && 11+1152/128           < tail_11_3+3+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_11_3) begin
            if(11>= tail_11_3+0 && 11 < tail_11_3+0+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(11>= tail_11_3+1 && 11 < tail_11_3+1+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(11>= tail_11_3+2 && 11 < tail_11_3+2+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(11>= tail_11_3+3 && 11 < tail_11_3+3+1)
                buffer11[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_11_3) begin
            if(11+1152/128           < tail_11_3)
                buffer11[32*3+:32] <= `CQ buffer20[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_12_3 && out_ack_12_3) begin
            if(12+1152/128           < tail_12_3)
                buffer12[32*3+:32] <= `CQ buffer21[32*3+:32];
            if(12+1152/128           >= tail_12_3+0 && 12+1152/128           < tail_12_3+0+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(12+1152/128           >= tail_12_3+1 && 12+1152/128           < tail_12_3+1+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(12+1152/128           >= tail_12_3+2 && 12+1152/128           < tail_12_3+2+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(12+1152/128           >= tail_12_3+3 && 12+1152/128           < tail_12_3+3+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(in_ack_12_3) begin
            if(12>= tail_12_3+0 && 12 < tail_12_3+0+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *0+32*3+:32];
            if(12>= tail_12_3+1 && 12 < tail_12_3+1+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *1+32*3+:32];
            if(12>= tail_12_3+2 && 12 < tail_12_3+2+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *2+32*3+:32];
            if(12>= tail_12_3+3 && 12 < tail_12_3+3+1)
                buffer12[32*3+:32] <= `CQ in_data[128       *3+32*3+:32];
        end
        else if(out_ack_12_3) begin
            if(12+1152/128           < tail_12_3)
                buffer12[32*3+:32] <= `CQ buffer21[32*3+:32];
        end

    assign out_data[128       *0+:128       ] = buffer0;        
    assign out_data[128       *1+:128       ] = buffer1;        
    assign out_data[128       *2+:128       ] = buffer2;        
    assign out_data[128       *3+:128       ] = buffer3;        
    assign out_data[128       *4+:128       ] = buffer4;        
    assign out_data[128       *5+:128       ] = buffer5;        
    assign out_data[128       *6+:128       ] = buffer6;        
    assign out_data[128       *7+:128       ] = buffer7;        
    assign out_data[128       *8+:128       ] = buffer8;        

endmodule

`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so 64 = 64 for DDR alignment
///define 64       (16*4)

//m items in, n items out
//flop out

//(64+512)/64        >= 512/64              
//(64+512)/64        >= 64/64          
//suggest (64+512)/64        = 512/64               + 64/64          
module barrelShiftFifo_scalar import zprize_param::*; (
    input                       in_valid,
    output                      in_ready,
    input   [64       *512/64              -1:0]   in_data,
    output                      out_valid,
    input                       out_ready,
    output  [64       *64/64          -1:0]  out_data,
    input                       clk,
    input                       rstN
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    logic   [64       -1:0]         buffer0;
    logic   [64       -1:0]         buffer1;
    logic   [64       -1:0]         buffer2;
    logic   [64       -1:0]         buffer3;
    logic   [64       -1:0]         buffer4;
    logic   [64       -1:0]         buffer5;
    logic   [64       -1:0]         buffer6;
    logic   [64       -1:0]         buffer7;
    logic   [64       -1:0]         buffer8;
    logic   [64       -1:0]         buffer9;      //useless

    //to fix fanout, duplicate tail
  (* max_fanout=400 *)  logic   [`LOG2((64+512)/64       )-1:0]  tail;   //point to first empty item

    logic   [`LOG2((64+512)/64       )-1:0]  tail_0_0;   //duplicate
    logic                       in_ack_0_0   ;
    logic                       out_ack_0_0  ;
    logic                       in_ready_0_0 ;
    logic                       out_valid_0_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_0_1;   //duplicate
    logic                       in_ack_0_1   ;
    logic                       out_ack_0_1  ;
    logic                       in_ready_0_1 ;
    logic                       out_valid_0_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_1_0;   //duplicate
    logic                       in_ack_1_0   ;
    logic                       out_ack_1_0  ;
    logic                       in_ready_1_0 ;
    logic                       out_valid_1_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_1_1;   //duplicate
    logic                       in_ack_1_1   ;
    logic                       out_ack_1_1  ;
    logic                       in_ready_1_1 ;
    logic                       out_valid_1_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_2_0;   //duplicate
    logic                       in_ack_2_0   ;
    logic                       out_ack_2_0  ;
    logic                       in_ready_2_0 ;
    logic                       out_valid_2_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_2_1;   //duplicate
    logic                       in_ack_2_1   ;
    logic                       out_ack_2_1  ;
    logic                       in_ready_2_1 ;
    logic                       out_valid_2_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_3_0;   //duplicate
    logic                       in_ack_3_0   ;
    logic                       out_ack_3_0  ;
    logic                       in_ready_3_0 ;
    logic                       out_valid_3_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_3_1;   //duplicate
    logic                       in_ack_3_1   ;
    logic                       out_ack_3_1  ;
    logic                       in_ready_3_1 ;
    logic                       out_valid_3_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_4_0;   //duplicate
    logic                       in_ack_4_0   ;
    logic                       out_ack_4_0  ;
    logic                       in_ready_4_0 ;
    logic                       out_valid_4_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_4_1;   //duplicate
    logic                       in_ack_4_1   ;
    logic                       out_ack_4_1  ;
    logic                       in_ready_4_1 ;
    logic                       out_valid_4_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_5_0;   //duplicate
    logic                       in_ack_5_0   ;
    logic                       out_ack_5_0  ;
    logic                       in_ready_5_0 ;
    logic                       out_valid_5_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_5_1;   //duplicate
    logic                       in_ack_5_1   ;
    logic                       out_ack_5_1  ;
    logic                       in_ready_5_1 ;
    logic                       out_valid_5_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_6_0;   //duplicate
    logic                       in_ack_6_0   ;
    logic                       out_ack_6_0  ;
    logic                       in_ready_6_0 ;
    logic                       out_valid_6_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_6_1;   //duplicate
    logic                       in_ack_6_1   ;
    logic                       out_ack_6_1  ;
    logic                       in_ready_6_1 ;
    logic                       out_valid_6_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_7_0;   //duplicate
    logic                       in_ack_7_0   ;
    logic                       out_ack_7_0  ;
    logic                       in_ready_7_0 ;
    logic                       out_valid_7_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_7_1;   //duplicate
    logic                       in_ack_7_1   ;
    logic                       out_ack_7_1  ;
    logic                       in_ready_7_1 ;
    logic                       out_valid_7_1;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_8_0;   //duplicate
    logic                       in_ack_8_0   ;
    logic                       out_ack_8_0  ;
    logic                       in_ready_8_0 ;
    logic                       out_valid_8_0;
    logic   [`LOG2((64+512)/64       )-1:0]  tail_8_1;   //duplicate
    logic                       in_ack_8_1   ;
    logic                       out_ack_8_1  ;
    logic                       in_ready_8_1 ;
    logic                       out_valid_8_1;

    logic                       in_ack;
    logic                       out_ack;
    genvar                      i, j, k;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign in_ack = in_valid && in_ready;
    assign out_ack = out_valid && out_ready;
    assign in_ready = tail + 512/64               <= (64+512)/64       ;
    assign out_valid = tail >= 64/64          ;

    assign in_ack_0_0   = in_valid && in_ready_0_0;
    assign out_ack_0_0  = out_valid_0_0 && out_ready;
    assign in_ready_0_0 = tail_0_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_0_0= tail_0_0 >= 64/64          ;
    assign in_ack_0_1   = in_valid && in_ready_0_1;
    assign out_ack_0_1  = out_valid_0_1 && out_ready;
    assign in_ready_0_1 = tail_0_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_0_1= tail_0_1 >= 64/64          ;
    assign in_ack_1_0   = in_valid && in_ready_1_0;
    assign out_ack_1_0  = out_valid_1_0 && out_ready;
    assign in_ready_1_0 = tail_1_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_1_0= tail_1_0 >= 64/64          ;
    assign in_ack_1_1   = in_valid && in_ready_1_1;
    assign out_ack_1_1  = out_valid_1_1 && out_ready;
    assign in_ready_1_1 = tail_1_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_1_1= tail_1_1 >= 64/64          ;
    assign in_ack_2_0   = in_valid && in_ready_2_0;
    assign out_ack_2_0  = out_valid_2_0 && out_ready;
    assign in_ready_2_0 = tail_2_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_2_0= tail_2_0 >= 64/64          ;
    assign in_ack_2_1   = in_valid && in_ready_2_1;
    assign out_ack_2_1  = out_valid_2_1 && out_ready;
    assign in_ready_2_1 = tail_2_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_2_1= tail_2_1 >= 64/64          ;
    assign in_ack_3_0   = in_valid && in_ready_3_0;
    assign out_ack_3_0  = out_valid_3_0 && out_ready;
    assign in_ready_3_0 = tail_3_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_3_0= tail_3_0 >= 64/64          ;
    assign in_ack_3_1   = in_valid && in_ready_3_1;
    assign out_ack_3_1  = out_valid_3_1 && out_ready;
    assign in_ready_3_1 = tail_3_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_3_1= tail_3_1 >= 64/64          ;
    assign in_ack_4_0   = in_valid && in_ready_4_0;
    assign out_ack_4_0  = out_valid_4_0 && out_ready;
    assign in_ready_4_0 = tail_4_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_4_0= tail_4_0 >= 64/64          ;
    assign in_ack_4_1   = in_valid && in_ready_4_1;
    assign out_ack_4_1  = out_valid_4_1 && out_ready;
    assign in_ready_4_1 = tail_4_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_4_1= tail_4_1 >= 64/64          ;
    assign in_ack_5_0   = in_valid && in_ready_5_0;
    assign out_ack_5_0  = out_valid_5_0 && out_ready;
    assign in_ready_5_0 = tail_5_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_5_0= tail_5_0 >= 64/64          ;
    assign in_ack_5_1   = in_valid && in_ready_5_1;
    assign out_ack_5_1  = out_valid_5_1 && out_ready;
    assign in_ready_5_1 = tail_5_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_5_1= tail_5_1 >= 64/64          ;
    assign in_ack_6_0   = in_valid && in_ready_6_0;
    assign out_ack_6_0  = out_valid_6_0 && out_ready;
    assign in_ready_6_0 = tail_6_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_6_0= tail_6_0 >= 64/64          ;
    assign in_ack_6_1   = in_valid && in_ready_6_1;
    assign out_ack_6_1  = out_valid_6_1 && out_ready;
    assign in_ready_6_1 = tail_6_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_6_1= tail_6_1 >= 64/64          ;
    assign in_ack_7_0   = in_valid && in_ready_7_0;
    assign out_ack_7_0  = out_valid_7_0 && out_ready;
    assign in_ready_7_0 = tail_7_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_7_0= tail_7_0 >= 64/64          ;
    assign in_ack_7_1   = in_valid && in_ready_7_1;
    assign out_ack_7_1  = out_valid_7_1 && out_ready;
    assign in_ready_7_1 = tail_7_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_7_1= tail_7_1 >= 64/64          ;
    assign in_ack_8_0   = in_valid && in_ready_8_0;
    assign out_ack_8_0  = out_valid_8_0 && out_ready;
    assign in_ready_8_0 = tail_8_0 + 512/64               <= (64+512)/64       ;
    assign out_valid_8_0= tail_8_0 >= 64/64          ;
    assign in_ack_8_1   = in_valid && in_ready_8_1;
    assign out_ack_8_1  = out_valid_8_1 && out_ready;
    assign in_ready_8_1 = tail_8_1 + 512/64               <= (64+512)/64       ;
    assign out_valid_8_1= tail_8_1 >= 64/64          ;

    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail <= `CQ 0;
        else if(in_ack && out_ack)
            tail <= `CQ tail + 512/64               - 64/64          ;
        else if(in_ack)
            tail <= `CQ tail + 512/64              ;
        else if(out_ack)
            tail <= `CQ tail - 64/64          ;

    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_0 <= `CQ 0;
        else if(in_ack_0_0&& out_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 + 512/64               - 64/64          ;
        else if(in_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 + 512/64              ;
        else if(out_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_1 <= `CQ 0;
        else if(in_ack_0_1&& out_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 + 512/64               - 64/64          ;
        else if(in_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 + 512/64              ;
        else if(out_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_0 <= `CQ 0;
        else if(in_ack_1_0&& out_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 + 512/64               - 64/64          ;
        else if(in_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 + 512/64              ;
        else if(out_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_1 <= `CQ 0;
        else if(in_ack_1_1&& out_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 + 512/64               - 64/64          ;
        else if(in_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 + 512/64              ;
        else if(out_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_0 <= `CQ 0;
        else if(in_ack_2_0&& out_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 + 512/64               - 64/64          ;
        else if(in_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 + 512/64              ;
        else if(out_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_1 <= `CQ 0;
        else if(in_ack_2_1&& out_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 + 512/64               - 64/64          ;
        else if(in_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 + 512/64              ;
        else if(out_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_0 <= `CQ 0;
        else if(in_ack_3_0&& out_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 + 512/64               - 64/64          ;
        else if(in_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 + 512/64              ;
        else if(out_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_1 <= `CQ 0;
        else if(in_ack_3_1&& out_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 + 512/64               - 64/64          ;
        else if(in_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 + 512/64              ;
        else if(out_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_4_0 <= `CQ 0;
        else if(in_ack_4_0&& out_ack_4_0)
            tail_4_0 <= `CQ tail_4_0 + 512/64               - 64/64          ;
        else if(in_ack_4_0)
            tail_4_0 <= `CQ tail_4_0 + 512/64              ;
        else if(out_ack_4_0)
            tail_4_0 <= `CQ tail_4_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_4_1 <= `CQ 0;
        else if(in_ack_4_1&& out_ack_4_1)
            tail_4_1 <= `CQ tail_4_1 + 512/64               - 64/64          ;
        else if(in_ack_4_1)
            tail_4_1 <= `CQ tail_4_1 + 512/64              ;
        else if(out_ack_4_1)
            tail_4_1 <= `CQ tail_4_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_5_0 <= `CQ 0;
        else if(in_ack_5_0&& out_ack_5_0)
            tail_5_0 <= `CQ tail_5_0 + 512/64               - 64/64          ;
        else if(in_ack_5_0)
            tail_5_0 <= `CQ tail_5_0 + 512/64              ;
        else if(out_ack_5_0)
            tail_5_0 <= `CQ tail_5_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_5_1 <= `CQ 0;
        else if(in_ack_5_1&& out_ack_5_1)
            tail_5_1 <= `CQ tail_5_1 + 512/64               - 64/64          ;
        else if(in_ack_5_1)
            tail_5_1 <= `CQ tail_5_1 + 512/64              ;
        else if(out_ack_5_1)
            tail_5_1 <= `CQ tail_5_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_6_0 <= `CQ 0;
        else if(in_ack_6_0&& out_ack_6_0)
            tail_6_0 <= `CQ tail_6_0 + 512/64               - 64/64          ;
        else if(in_ack_6_0)
            tail_6_0 <= `CQ tail_6_0 + 512/64              ;
        else if(out_ack_6_0)
            tail_6_0 <= `CQ tail_6_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_6_1 <= `CQ 0;
        else if(in_ack_6_1&& out_ack_6_1)
            tail_6_1 <= `CQ tail_6_1 + 512/64               - 64/64          ;
        else if(in_ack_6_1)
            tail_6_1 <= `CQ tail_6_1 + 512/64              ;
        else if(out_ack_6_1)
            tail_6_1 <= `CQ tail_6_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_7_0 <= `CQ 0;
        else if(in_ack_7_0&& out_ack_7_0)
            tail_7_0 <= `CQ tail_7_0 + 512/64               - 64/64          ;
        else if(in_ack_7_0)
            tail_7_0 <= `CQ tail_7_0 + 512/64              ;
        else if(out_ack_7_0)
            tail_7_0 <= `CQ tail_7_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_7_1 <= `CQ 0;
        else if(in_ack_7_1&& out_ack_7_1)
            tail_7_1 <= `CQ tail_7_1 + 512/64               - 64/64          ;
        else if(in_ack_7_1)
            tail_7_1 <= `CQ tail_7_1 + 512/64              ;
        else if(out_ack_7_1)
            tail_7_1 <= `CQ tail_7_1 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_8_0 <= `CQ 0;
        else if(in_ack_8_0&& out_ack_8_0)
            tail_8_0 <= `CQ tail_8_0 + 512/64               - 64/64          ;
        else if(in_ack_8_0)
            tail_8_0 <= `CQ tail_8_0 + 512/64              ;
        else if(out_ack_8_0)
            tail_8_0 <= `CQ tail_8_0 - 64/64          ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_8_1 <= `CQ 0;
        else if(in_ack_8_1&& out_ack_8_1)
            tail_8_1 <= `CQ tail_8_1 + 512/64               - 64/64          ;
        else if(in_ack_8_1)
            tail_8_1 <= `CQ tail_8_1 + 512/64              ;
        else if(out_ack_8_1)
            tail_8_1 <= `CQ tail_8_1 - 64/64          ;

//split bufferk to 32bit, for fanout timing

    always@(posedge clk)
        if(in_ack_0_0 && out_ack_0_0) begin
            if(0+64/64           < tail_0_0)
                buffer0[32*0+:32] <= `CQ buffer1[32*0+:32];
            if(0+64/64           >= tail_0_0+0 && 0+64/64           < tail_0_0+0+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(0+64/64           >= tail_0_0+1 && 0+64/64           < tail_0_0+1+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(0+64/64           >= tail_0_0+2 && 0+64/64           < tail_0_0+2+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(0+64/64           >= tail_0_0+3 && 0+64/64           < tail_0_0+3+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(0+64/64           >= tail_0_0+4 && 0+64/64           < tail_0_0+4+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(0+64/64           >= tail_0_0+5 && 0+64/64           < tail_0_0+5+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(0+64/64           >= tail_0_0+6 && 0+64/64           < tail_0_0+6+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(0+64/64           >= tail_0_0+7 && 0+64/64           < tail_0_0+7+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_0_0) begin
            if(0>= tail_0_0+0 && 0 < tail_0_0+0+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(0>= tail_0_0+1 && 0 < tail_0_0+1+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(0>= tail_0_0+2 && 0 < tail_0_0+2+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(0>= tail_0_0+3 && 0 < tail_0_0+3+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(0>= tail_0_0+4 && 0 < tail_0_0+4+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(0>= tail_0_0+5 && 0 < tail_0_0+5+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(0>= tail_0_0+6 && 0 < tail_0_0+6+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(0>= tail_0_0+7 && 0 < tail_0_0+7+1)
                buffer0[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_0_0) begin
            if(0+64/64           < tail_0_0)
                buffer0[32*0+:32] <= `CQ buffer1[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_1_0 && out_ack_1_0) begin
            if(1+64/64           < tail_1_0)
                buffer1[32*0+:32] <= `CQ buffer2[32*0+:32];
            if(1+64/64           >= tail_1_0+0 && 1+64/64           < tail_1_0+0+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(1+64/64           >= tail_1_0+1 && 1+64/64           < tail_1_0+1+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(1+64/64           >= tail_1_0+2 && 1+64/64           < tail_1_0+2+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(1+64/64           >= tail_1_0+3 && 1+64/64           < tail_1_0+3+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(1+64/64           >= tail_1_0+4 && 1+64/64           < tail_1_0+4+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(1+64/64           >= tail_1_0+5 && 1+64/64           < tail_1_0+5+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(1+64/64           >= tail_1_0+6 && 1+64/64           < tail_1_0+6+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(1+64/64           >= tail_1_0+7 && 1+64/64           < tail_1_0+7+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_1_0) begin
            if(1>= tail_1_0+0 && 1 < tail_1_0+0+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(1>= tail_1_0+1 && 1 < tail_1_0+1+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(1>= tail_1_0+2 && 1 < tail_1_0+2+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(1>= tail_1_0+3 && 1 < tail_1_0+3+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(1>= tail_1_0+4 && 1 < tail_1_0+4+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(1>= tail_1_0+5 && 1 < tail_1_0+5+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(1>= tail_1_0+6 && 1 < tail_1_0+6+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(1>= tail_1_0+7 && 1 < tail_1_0+7+1)
                buffer1[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_1_0) begin
            if(1+64/64           < tail_1_0)
                buffer1[32*0+:32] <= `CQ buffer2[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_2_0 && out_ack_2_0) begin
            if(2+64/64           < tail_2_0)
                buffer2[32*0+:32] <= `CQ buffer3[32*0+:32];
            if(2+64/64           >= tail_2_0+0 && 2+64/64           < tail_2_0+0+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(2+64/64           >= tail_2_0+1 && 2+64/64           < tail_2_0+1+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(2+64/64           >= tail_2_0+2 && 2+64/64           < tail_2_0+2+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(2+64/64           >= tail_2_0+3 && 2+64/64           < tail_2_0+3+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(2+64/64           >= tail_2_0+4 && 2+64/64           < tail_2_0+4+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(2+64/64           >= tail_2_0+5 && 2+64/64           < tail_2_0+5+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(2+64/64           >= tail_2_0+6 && 2+64/64           < tail_2_0+6+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(2+64/64           >= tail_2_0+7 && 2+64/64           < tail_2_0+7+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_2_0) begin
            if(2>= tail_2_0+0 && 2 < tail_2_0+0+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(2>= tail_2_0+1 && 2 < tail_2_0+1+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(2>= tail_2_0+2 && 2 < tail_2_0+2+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(2>= tail_2_0+3 && 2 < tail_2_0+3+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(2>= tail_2_0+4 && 2 < tail_2_0+4+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(2>= tail_2_0+5 && 2 < tail_2_0+5+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(2>= tail_2_0+6 && 2 < tail_2_0+6+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(2>= tail_2_0+7 && 2 < tail_2_0+7+1)
                buffer2[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_2_0) begin
            if(2+64/64           < tail_2_0)
                buffer2[32*0+:32] <= `CQ buffer3[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_3_0 && out_ack_3_0) begin
            if(3+64/64           < tail_3_0)
                buffer3[32*0+:32] <= `CQ buffer4[32*0+:32];
            if(3+64/64           >= tail_3_0+0 && 3+64/64           < tail_3_0+0+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(3+64/64           >= tail_3_0+1 && 3+64/64           < tail_3_0+1+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(3+64/64           >= tail_3_0+2 && 3+64/64           < tail_3_0+2+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(3+64/64           >= tail_3_0+3 && 3+64/64           < tail_3_0+3+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(3+64/64           >= tail_3_0+4 && 3+64/64           < tail_3_0+4+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(3+64/64           >= tail_3_0+5 && 3+64/64           < tail_3_0+5+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(3+64/64           >= tail_3_0+6 && 3+64/64           < tail_3_0+6+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(3+64/64           >= tail_3_0+7 && 3+64/64           < tail_3_0+7+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_3_0) begin
            if(3>= tail_3_0+0 && 3 < tail_3_0+0+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(3>= tail_3_0+1 && 3 < tail_3_0+1+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(3>= tail_3_0+2 && 3 < tail_3_0+2+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(3>= tail_3_0+3 && 3 < tail_3_0+3+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(3>= tail_3_0+4 && 3 < tail_3_0+4+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(3>= tail_3_0+5 && 3 < tail_3_0+5+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(3>= tail_3_0+6 && 3 < tail_3_0+6+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(3>= tail_3_0+7 && 3 < tail_3_0+7+1)
                buffer3[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_3_0) begin
            if(3+64/64           < tail_3_0)
                buffer3[32*0+:32] <= `CQ buffer4[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_4_0 && out_ack_4_0) begin
            if(4+64/64           < tail_4_0)
                buffer4[32*0+:32] <= `CQ buffer5[32*0+:32];
            if(4+64/64           >= tail_4_0+0 && 4+64/64           < tail_4_0+0+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(4+64/64           >= tail_4_0+1 && 4+64/64           < tail_4_0+1+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(4+64/64           >= tail_4_0+2 && 4+64/64           < tail_4_0+2+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(4+64/64           >= tail_4_0+3 && 4+64/64           < tail_4_0+3+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(4+64/64           >= tail_4_0+4 && 4+64/64           < tail_4_0+4+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(4+64/64           >= tail_4_0+5 && 4+64/64           < tail_4_0+5+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(4+64/64           >= tail_4_0+6 && 4+64/64           < tail_4_0+6+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(4+64/64           >= tail_4_0+7 && 4+64/64           < tail_4_0+7+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_4_0) begin
            if(4>= tail_4_0+0 && 4 < tail_4_0+0+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(4>= tail_4_0+1 && 4 < tail_4_0+1+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(4>= tail_4_0+2 && 4 < tail_4_0+2+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(4>= tail_4_0+3 && 4 < tail_4_0+3+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(4>= tail_4_0+4 && 4 < tail_4_0+4+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(4>= tail_4_0+5 && 4 < tail_4_0+5+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(4>= tail_4_0+6 && 4 < tail_4_0+6+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(4>= tail_4_0+7 && 4 < tail_4_0+7+1)
                buffer4[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_4_0) begin
            if(4+64/64           < tail_4_0)
                buffer4[32*0+:32] <= `CQ buffer5[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_5_0 && out_ack_5_0) begin
            if(5+64/64           < tail_5_0)
                buffer5[32*0+:32] <= `CQ buffer6[32*0+:32];
            if(5+64/64           >= tail_5_0+0 && 5+64/64           < tail_5_0+0+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(5+64/64           >= tail_5_0+1 && 5+64/64           < tail_5_0+1+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(5+64/64           >= tail_5_0+2 && 5+64/64           < tail_5_0+2+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(5+64/64           >= tail_5_0+3 && 5+64/64           < tail_5_0+3+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(5+64/64           >= tail_5_0+4 && 5+64/64           < tail_5_0+4+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(5+64/64           >= tail_5_0+5 && 5+64/64           < tail_5_0+5+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(5+64/64           >= tail_5_0+6 && 5+64/64           < tail_5_0+6+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(5+64/64           >= tail_5_0+7 && 5+64/64           < tail_5_0+7+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_5_0) begin
            if(5>= tail_5_0+0 && 5 < tail_5_0+0+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(5>= tail_5_0+1 && 5 < tail_5_0+1+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(5>= tail_5_0+2 && 5 < tail_5_0+2+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(5>= tail_5_0+3 && 5 < tail_5_0+3+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(5>= tail_5_0+4 && 5 < tail_5_0+4+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(5>= tail_5_0+5 && 5 < tail_5_0+5+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(5>= tail_5_0+6 && 5 < tail_5_0+6+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(5>= tail_5_0+7 && 5 < tail_5_0+7+1)
                buffer5[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_5_0) begin
            if(5+64/64           < tail_5_0)
                buffer5[32*0+:32] <= `CQ buffer6[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_6_0 && out_ack_6_0) begin
            if(6+64/64           < tail_6_0)
                buffer6[32*0+:32] <= `CQ buffer7[32*0+:32];
            if(6+64/64           >= tail_6_0+0 && 6+64/64           < tail_6_0+0+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(6+64/64           >= tail_6_0+1 && 6+64/64           < tail_6_0+1+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(6+64/64           >= tail_6_0+2 && 6+64/64           < tail_6_0+2+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(6+64/64           >= tail_6_0+3 && 6+64/64           < tail_6_0+3+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(6+64/64           >= tail_6_0+4 && 6+64/64           < tail_6_0+4+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(6+64/64           >= tail_6_0+5 && 6+64/64           < tail_6_0+5+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(6+64/64           >= tail_6_0+6 && 6+64/64           < tail_6_0+6+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(6+64/64           >= tail_6_0+7 && 6+64/64           < tail_6_0+7+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_6_0) begin
            if(6>= tail_6_0+0 && 6 < tail_6_0+0+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(6>= tail_6_0+1 && 6 < tail_6_0+1+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(6>= tail_6_0+2 && 6 < tail_6_0+2+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(6>= tail_6_0+3 && 6 < tail_6_0+3+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(6>= tail_6_0+4 && 6 < tail_6_0+4+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(6>= tail_6_0+5 && 6 < tail_6_0+5+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(6>= tail_6_0+6 && 6 < tail_6_0+6+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(6>= tail_6_0+7 && 6 < tail_6_0+7+1)
                buffer6[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_6_0) begin
            if(6+64/64           < tail_6_0)
                buffer6[32*0+:32] <= `CQ buffer7[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_7_0 && out_ack_7_0) begin
            if(7+64/64           < tail_7_0)
                buffer7[32*0+:32] <= `CQ buffer8[32*0+:32];
            if(7+64/64           >= tail_7_0+0 && 7+64/64           < tail_7_0+0+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(7+64/64           >= tail_7_0+1 && 7+64/64           < tail_7_0+1+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(7+64/64           >= tail_7_0+2 && 7+64/64           < tail_7_0+2+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(7+64/64           >= tail_7_0+3 && 7+64/64           < tail_7_0+3+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(7+64/64           >= tail_7_0+4 && 7+64/64           < tail_7_0+4+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(7+64/64           >= tail_7_0+5 && 7+64/64           < tail_7_0+5+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(7+64/64           >= tail_7_0+6 && 7+64/64           < tail_7_0+6+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(7+64/64           >= tail_7_0+7 && 7+64/64           < tail_7_0+7+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_7_0) begin
            if(7>= tail_7_0+0 && 7 < tail_7_0+0+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(7>= tail_7_0+1 && 7 < tail_7_0+1+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(7>= tail_7_0+2 && 7 < tail_7_0+2+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(7>= tail_7_0+3 && 7 < tail_7_0+3+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(7>= tail_7_0+4 && 7 < tail_7_0+4+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(7>= tail_7_0+5 && 7 < tail_7_0+5+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(7>= tail_7_0+6 && 7 < tail_7_0+6+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(7>= tail_7_0+7 && 7 < tail_7_0+7+1)
                buffer7[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_7_0) begin
            if(7+64/64           < tail_7_0)
                buffer7[32*0+:32] <= `CQ buffer8[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_8_0 && out_ack_8_0) begin
            if(8+64/64           < tail_8_0)
                buffer8[32*0+:32] <= `CQ buffer9[32*0+:32];
            if(8+64/64           >= tail_8_0+0 && 8+64/64           < tail_8_0+0+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(8+64/64           >= tail_8_0+1 && 8+64/64           < tail_8_0+1+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(8+64/64           >= tail_8_0+2 && 8+64/64           < tail_8_0+2+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(8+64/64           >= tail_8_0+3 && 8+64/64           < tail_8_0+3+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(8+64/64           >= tail_8_0+4 && 8+64/64           < tail_8_0+4+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(8+64/64           >= tail_8_0+5 && 8+64/64           < tail_8_0+5+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(8+64/64           >= tail_8_0+6 && 8+64/64           < tail_8_0+6+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(8+64/64           >= tail_8_0+7 && 8+64/64           < tail_8_0+7+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(in_ack_8_0) begin
            if(8>= tail_8_0+0 && 8 < tail_8_0+0+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *0+32*0+:32];
            if(8>= tail_8_0+1 && 8 < tail_8_0+1+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *1+32*0+:32];
            if(8>= tail_8_0+2 && 8 < tail_8_0+2+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *2+32*0+:32];
            if(8>= tail_8_0+3 && 8 < tail_8_0+3+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *3+32*0+:32];
            if(8>= tail_8_0+4 && 8 < tail_8_0+4+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *4+32*0+:32];
            if(8>= tail_8_0+5 && 8 < tail_8_0+5+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *5+32*0+:32];
            if(8>= tail_8_0+6 && 8 < tail_8_0+6+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *6+32*0+:32];
            if(8>= tail_8_0+7 && 8 < tail_8_0+7+1)
                buffer8[32*0+:32] <= `CQ in_data[64       *7+32*0+:32];
        end
        else if(out_ack_8_0) begin
            if(8+64/64           < tail_8_0)
                buffer8[32*0+:32] <= `CQ buffer9[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_0_1 && out_ack_0_1) begin
            if(0+64/64           < tail_0_1)
                buffer0[32*1+:32] <= `CQ buffer1[32*1+:32];
            if(0+64/64           >= tail_0_1+0 && 0+64/64           < tail_0_1+0+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(0+64/64           >= tail_0_1+1 && 0+64/64           < tail_0_1+1+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(0+64/64           >= tail_0_1+2 && 0+64/64           < tail_0_1+2+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(0+64/64           >= tail_0_1+3 && 0+64/64           < tail_0_1+3+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(0+64/64           >= tail_0_1+4 && 0+64/64           < tail_0_1+4+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(0+64/64           >= tail_0_1+5 && 0+64/64           < tail_0_1+5+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(0+64/64           >= tail_0_1+6 && 0+64/64           < tail_0_1+6+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(0+64/64           >= tail_0_1+7 && 0+64/64           < tail_0_1+7+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_0_1) begin
            if(0>= tail_0_1+0 && 0 < tail_0_1+0+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(0>= tail_0_1+1 && 0 < tail_0_1+1+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(0>= tail_0_1+2 && 0 < tail_0_1+2+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(0>= tail_0_1+3 && 0 < tail_0_1+3+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(0>= tail_0_1+4 && 0 < tail_0_1+4+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(0>= tail_0_1+5 && 0 < tail_0_1+5+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(0>= tail_0_1+6 && 0 < tail_0_1+6+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(0>= tail_0_1+7 && 0 < tail_0_1+7+1)
                buffer0[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_0_1) begin
            if(0+64/64           < tail_0_1)
                buffer0[32*1+:32] <= `CQ buffer1[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_1_1 && out_ack_1_1) begin
            if(1+64/64           < tail_1_1)
                buffer1[32*1+:32] <= `CQ buffer2[32*1+:32];
            if(1+64/64           >= tail_1_1+0 && 1+64/64           < tail_1_1+0+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(1+64/64           >= tail_1_1+1 && 1+64/64           < tail_1_1+1+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(1+64/64           >= tail_1_1+2 && 1+64/64           < tail_1_1+2+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(1+64/64           >= tail_1_1+3 && 1+64/64           < tail_1_1+3+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(1+64/64           >= tail_1_1+4 && 1+64/64           < tail_1_1+4+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(1+64/64           >= tail_1_1+5 && 1+64/64           < tail_1_1+5+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(1+64/64           >= tail_1_1+6 && 1+64/64           < tail_1_1+6+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(1+64/64           >= tail_1_1+7 && 1+64/64           < tail_1_1+7+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_1_1) begin
            if(1>= tail_1_1+0 && 1 < tail_1_1+0+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(1>= tail_1_1+1 && 1 < tail_1_1+1+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(1>= tail_1_1+2 && 1 < tail_1_1+2+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(1>= tail_1_1+3 && 1 < tail_1_1+3+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(1>= tail_1_1+4 && 1 < tail_1_1+4+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(1>= tail_1_1+5 && 1 < tail_1_1+5+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(1>= tail_1_1+6 && 1 < tail_1_1+6+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(1>= tail_1_1+7 && 1 < tail_1_1+7+1)
                buffer1[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_1_1) begin
            if(1+64/64           < tail_1_1)
                buffer1[32*1+:32] <= `CQ buffer2[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_2_1 && out_ack_2_1) begin
            if(2+64/64           < tail_2_1)
                buffer2[32*1+:32] <= `CQ buffer3[32*1+:32];
            if(2+64/64           >= tail_2_1+0 && 2+64/64           < tail_2_1+0+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(2+64/64           >= tail_2_1+1 && 2+64/64           < tail_2_1+1+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(2+64/64           >= tail_2_1+2 && 2+64/64           < tail_2_1+2+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(2+64/64           >= tail_2_1+3 && 2+64/64           < tail_2_1+3+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(2+64/64           >= tail_2_1+4 && 2+64/64           < tail_2_1+4+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(2+64/64           >= tail_2_1+5 && 2+64/64           < tail_2_1+5+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(2+64/64           >= tail_2_1+6 && 2+64/64           < tail_2_1+6+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(2+64/64           >= tail_2_1+7 && 2+64/64           < tail_2_1+7+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_2_1) begin
            if(2>= tail_2_1+0 && 2 < tail_2_1+0+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(2>= tail_2_1+1 && 2 < tail_2_1+1+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(2>= tail_2_1+2 && 2 < tail_2_1+2+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(2>= tail_2_1+3 && 2 < tail_2_1+3+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(2>= tail_2_1+4 && 2 < tail_2_1+4+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(2>= tail_2_1+5 && 2 < tail_2_1+5+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(2>= tail_2_1+6 && 2 < tail_2_1+6+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(2>= tail_2_1+7 && 2 < tail_2_1+7+1)
                buffer2[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_2_1) begin
            if(2+64/64           < tail_2_1)
                buffer2[32*1+:32] <= `CQ buffer3[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_3_1 && out_ack_3_1) begin
            if(3+64/64           < tail_3_1)
                buffer3[32*1+:32] <= `CQ buffer4[32*1+:32];
            if(3+64/64           >= tail_3_1+0 && 3+64/64           < tail_3_1+0+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(3+64/64           >= tail_3_1+1 && 3+64/64           < tail_3_1+1+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(3+64/64           >= tail_3_1+2 && 3+64/64           < tail_3_1+2+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(3+64/64           >= tail_3_1+3 && 3+64/64           < tail_3_1+3+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(3+64/64           >= tail_3_1+4 && 3+64/64           < tail_3_1+4+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(3+64/64           >= tail_3_1+5 && 3+64/64           < tail_3_1+5+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(3+64/64           >= tail_3_1+6 && 3+64/64           < tail_3_1+6+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(3+64/64           >= tail_3_1+7 && 3+64/64           < tail_3_1+7+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_3_1) begin
            if(3>= tail_3_1+0 && 3 < tail_3_1+0+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(3>= tail_3_1+1 && 3 < tail_3_1+1+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(3>= tail_3_1+2 && 3 < tail_3_1+2+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(3>= tail_3_1+3 && 3 < tail_3_1+3+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(3>= tail_3_1+4 && 3 < tail_3_1+4+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(3>= tail_3_1+5 && 3 < tail_3_1+5+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(3>= tail_3_1+6 && 3 < tail_3_1+6+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(3>= tail_3_1+7 && 3 < tail_3_1+7+1)
                buffer3[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_3_1) begin
            if(3+64/64           < tail_3_1)
                buffer3[32*1+:32] <= `CQ buffer4[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_4_1 && out_ack_4_1) begin
            if(4+64/64           < tail_4_1)
                buffer4[32*1+:32] <= `CQ buffer5[32*1+:32];
            if(4+64/64           >= tail_4_1+0 && 4+64/64           < tail_4_1+0+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(4+64/64           >= tail_4_1+1 && 4+64/64           < tail_4_1+1+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(4+64/64           >= tail_4_1+2 && 4+64/64           < tail_4_1+2+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(4+64/64           >= tail_4_1+3 && 4+64/64           < tail_4_1+3+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(4+64/64           >= tail_4_1+4 && 4+64/64           < tail_4_1+4+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(4+64/64           >= tail_4_1+5 && 4+64/64           < tail_4_1+5+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(4+64/64           >= tail_4_1+6 && 4+64/64           < tail_4_1+6+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(4+64/64           >= tail_4_1+7 && 4+64/64           < tail_4_1+7+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_4_1) begin
            if(4>= tail_4_1+0 && 4 < tail_4_1+0+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(4>= tail_4_1+1 && 4 < tail_4_1+1+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(4>= tail_4_1+2 && 4 < tail_4_1+2+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(4>= tail_4_1+3 && 4 < tail_4_1+3+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(4>= tail_4_1+4 && 4 < tail_4_1+4+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(4>= tail_4_1+5 && 4 < tail_4_1+5+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(4>= tail_4_1+6 && 4 < tail_4_1+6+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(4>= tail_4_1+7 && 4 < tail_4_1+7+1)
                buffer4[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_4_1) begin
            if(4+64/64           < tail_4_1)
                buffer4[32*1+:32] <= `CQ buffer5[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_5_1 && out_ack_5_1) begin
            if(5+64/64           < tail_5_1)
                buffer5[32*1+:32] <= `CQ buffer6[32*1+:32];
            if(5+64/64           >= tail_5_1+0 && 5+64/64           < tail_5_1+0+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(5+64/64           >= tail_5_1+1 && 5+64/64           < tail_5_1+1+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(5+64/64           >= tail_5_1+2 && 5+64/64           < tail_5_1+2+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(5+64/64           >= tail_5_1+3 && 5+64/64           < tail_5_1+3+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(5+64/64           >= tail_5_1+4 && 5+64/64           < tail_5_1+4+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(5+64/64           >= tail_5_1+5 && 5+64/64           < tail_5_1+5+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(5+64/64           >= tail_5_1+6 && 5+64/64           < tail_5_1+6+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(5+64/64           >= tail_5_1+7 && 5+64/64           < tail_5_1+7+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_5_1) begin
            if(5>= tail_5_1+0 && 5 < tail_5_1+0+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(5>= tail_5_1+1 && 5 < tail_5_1+1+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(5>= tail_5_1+2 && 5 < tail_5_1+2+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(5>= tail_5_1+3 && 5 < tail_5_1+3+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(5>= tail_5_1+4 && 5 < tail_5_1+4+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(5>= tail_5_1+5 && 5 < tail_5_1+5+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(5>= tail_5_1+6 && 5 < tail_5_1+6+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(5>= tail_5_1+7 && 5 < tail_5_1+7+1)
                buffer5[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_5_1) begin
            if(5+64/64           < tail_5_1)
                buffer5[32*1+:32] <= `CQ buffer6[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_6_1 && out_ack_6_1) begin
            if(6+64/64           < tail_6_1)
                buffer6[32*1+:32] <= `CQ buffer7[32*1+:32];
            if(6+64/64           >= tail_6_1+0 && 6+64/64           < tail_6_1+0+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(6+64/64           >= tail_6_1+1 && 6+64/64           < tail_6_1+1+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(6+64/64           >= tail_6_1+2 && 6+64/64           < tail_6_1+2+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(6+64/64           >= tail_6_1+3 && 6+64/64           < tail_6_1+3+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(6+64/64           >= tail_6_1+4 && 6+64/64           < tail_6_1+4+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(6+64/64           >= tail_6_1+5 && 6+64/64           < tail_6_1+5+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(6+64/64           >= tail_6_1+6 && 6+64/64           < tail_6_1+6+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(6+64/64           >= tail_6_1+7 && 6+64/64           < tail_6_1+7+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_6_1) begin
            if(6>= tail_6_1+0 && 6 < tail_6_1+0+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(6>= tail_6_1+1 && 6 < tail_6_1+1+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(6>= tail_6_1+2 && 6 < tail_6_1+2+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(6>= tail_6_1+3 && 6 < tail_6_1+3+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(6>= tail_6_1+4 && 6 < tail_6_1+4+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(6>= tail_6_1+5 && 6 < tail_6_1+5+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(6>= tail_6_1+6 && 6 < tail_6_1+6+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(6>= tail_6_1+7 && 6 < tail_6_1+7+1)
                buffer6[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_6_1) begin
            if(6+64/64           < tail_6_1)
                buffer6[32*1+:32] <= `CQ buffer7[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_7_1 && out_ack_7_1) begin
            if(7+64/64           < tail_7_1)
                buffer7[32*1+:32] <= `CQ buffer8[32*1+:32];
            if(7+64/64           >= tail_7_1+0 && 7+64/64           < tail_7_1+0+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(7+64/64           >= tail_7_1+1 && 7+64/64           < tail_7_1+1+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(7+64/64           >= tail_7_1+2 && 7+64/64           < tail_7_1+2+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(7+64/64           >= tail_7_1+3 && 7+64/64           < tail_7_1+3+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(7+64/64           >= tail_7_1+4 && 7+64/64           < tail_7_1+4+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(7+64/64           >= tail_7_1+5 && 7+64/64           < tail_7_1+5+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(7+64/64           >= tail_7_1+6 && 7+64/64           < tail_7_1+6+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(7+64/64           >= tail_7_1+7 && 7+64/64           < tail_7_1+7+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_7_1) begin
            if(7>= tail_7_1+0 && 7 < tail_7_1+0+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(7>= tail_7_1+1 && 7 < tail_7_1+1+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(7>= tail_7_1+2 && 7 < tail_7_1+2+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(7>= tail_7_1+3 && 7 < tail_7_1+3+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(7>= tail_7_1+4 && 7 < tail_7_1+4+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(7>= tail_7_1+5 && 7 < tail_7_1+5+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(7>= tail_7_1+6 && 7 < tail_7_1+6+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(7>= tail_7_1+7 && 7 < tail_7_1+7+1)
                buffer7[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_7_1) begin
            if(7+64/64           < tail_7_1)
                buffer7[32*1+:32] <= `CQ buffer8[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_8_1 && out_ack_8_1) begin
            if(8+64/64           < tail_8_1)
                buffer8[32*1+:32] <= `CQ buffer9[32*1+:32];
            if(8+64/64           >= tail_8_1+0 && 8+64/64           < tail_8_1+0+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(8+64/64           >= tail_8_1+1 && 8+64/64           < tail_8_1+1+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(8+64/64           >= tail_8_1+2 && 8+64/64           < tail_8_1+2+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(8+64/64           >= tail_8_1+3 && 8+64/64           < tail_8_1+3+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(8+64/64           >= tail_8_1+4 && 8+64/64           < tail_8_1+4+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(8+64/64           >= tail_8_1+5 && 8+64/64           < tail_8_1+5+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(8+64/64           >= tail_8_1+6 && 8+64/64           < tail_8_1+6+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(8+64/64           >= tail_8_1+7 && 8+64/64           < tail_8_1+7+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(in_ack_8_1) begin
            if(8>= tail_8_1+0 && 8 < tail_8_1+0+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *0+32*1+:32];
            if(8>= tail_8_1+1 && 8 < tail_8_1+1+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *1+32*1+:32];
            if(8>= tail_8_1+2 && 8 < tail_8_1+2+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *2+32*1+:32];
            if(8>= tail_8_1+3 && 8 < tail_8_1+3+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *3+32*1+:32];
            if(8>= tail_8_1+4 && 8 < tail_8_1+4+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *4+32*1+:32];
            if(8>= tail_8_1+5 && 8 < tail_8_1+5+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *5+32*1+:32];
            if(8>= tail_8_1+6 && 8 < tail_8_1+6+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *6+32*1+:32];
            if(8>= tail_8_1+7 && 8 < tail_8_1+7+1)
                buffer8[32*1+:32] <= `CQ in_data[64       *7+32*1+:32];
        end
        else if(out_ack_8_1) begin
            if(8+64/64           < tail_8_1)
                buffer8[32*1+:32] <= `CQ buffer9[32*1+:32];
        end

    assign out_data[64       *0+:64       ] = buffer0;        

endmodule

`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so 64 = 64 for DDR alignment
///define 64       (16*4)

//m items in, n items out
//flop out

//(1536+512)/512        >= 1536/512          
//(1536+512)/512        >= 512/512              
//suggest (1536+512)/512        = 1536/512           + 512/512              
module barrelShiftFifo_result import zprize_param::*; (
    input                       in_valid,
    output                      in_ready,
    input   [512       *1536/512          -1:0]   in_data,
    output                      out_valid,
    input                       out_ready,
    output  [512       *512/512              -1:0]  out_data,
    input                       clk,
    input                       rstN
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    logic   [512       -1:0]         buffer0;
    logic   [512       -1:0]         buffer1;
    logic   [512       -1:0]         buffer2;
    logic   [512       -1:0]         buffer3;
    logic   [512       -1:0]         buffer4;      //useless

    //to fix fanout, duplicate tail
  (* max_fanout=400 *)  logic   [`LOG2((1536+512)/512       )-1:0]  tail;   //point to first empty item

    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_0;   //duplicate
    logic                       in_ack_0_0   ;
    logic                       out_ack_0_0  ;
    logic                       in_ready_0_0 ;
    logic                       out_valid_0_0;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_1;   //duplicate
    logic                       in_ack_0_1   ;
    logic                       out_ack_0_1  ;
    logic                       in_ready_0_1 ;
    logic                       out_valid_0_1;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_2;   //duplicate
    logic                       in_ack_0_2   ;
    logic                       out_ack_0_2  ;
    logic                       in_ready_0_2 ;
    logic                       out_valid_0_2;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_3;   //duplicate
    logic                       in_ack_0_3   ;
    logic                       out_ack_0_3  ;
    logic                       in_ready_0_3 ;
    logic                       out_valid_0_3;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_4;   //duplicate
    logic                       in_ack_0_4   ;
    logic                       out_ack_0_4  ;
    logic                       in_ready_0_4 ;
    logic                       out_valid_0_4;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_5;   //duplicate
    logic                       in_ack_0_5   ;
    logic                       out_ack_0_5  ;
    logic                       in_ready_0_5 ;
    logic                       out_valid_0_5;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_6;   //duplicate
    logic                       in_ack_0_6   ;
    logic                       out_ack_0_6  ;
    logic                       in_ready_0_6 ;
    logic                       out_valid_0_6;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_7;   //duplicate
    logic                       in_ack_0_7   ;
    logic                       out_ack_0_7  ;
    logic                       in_ready_0_7 ;
    logic                       out_valid_0_7;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_8;   //duplicate
    logic                       in_ack_0_8   ;
    logic                       out_ack_0_8  ;
    logic                       in_ready_0_8 ;
    logic                       out_valid_0_8;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_9;   //duplicate
    logic                       in_ack_0_9   ;
    logic                       out_ack_0_9  ;
    logic                       in_ready_0_9 ;
    logic                       out_valid_0_9;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_10;   //duplicate
    logic                       in_ack_0_10   ;
    logic                       out_ack_0_10  ;
    logic                       in_ready_0_10 ;
    logic                       out_valid_0_10;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_11;   //duplicate
    logic                       in_ack_0_11   ;
    logic                       out_ack_0_11  ;
    logic                       in_ready_0_11 ;
    logic                       out_valid_0_11;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_12;   //duplicate
    logic                       in_ack_0_12   ;
    logic                       out_ack_0_12  ;
    logic                       in_ready_0_12 ;
    logic                       out_valid_0_12;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_13;   //duplicate
    logic                       in_ack_0_13   ;
    logic                       out_ack_0_13  ;
    logic                       in_ready_0_13 ;
    logic                       out_valid_0_13;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_14;   //duplicate
    logic                       in_ack_0_14   ;
    logic                       out_ack_0_14  ;
    logic                       in_ready_0_14 ;
    logic                       out_valid_0_14;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_0_15;   //duplicate
    logic                       in_ack_0_15   ;
    logic                       out_ack_0_15  ;
    logic                       in_ready_0_15 ;
    logic                       out_valid_0_15;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_0;   //duplicate
    logic                       in_ack_1_0   ;
    logic                       out_ack_1_0  ;
    logic                       in_ready_1_0 ;
    logic                       out_valid_1_0;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_1;   //duplicate
    logic                       in_ack_1_1   ;
    logic                       out_ack_1_1  ;
    logic                       in_ready_1_1 ;
    logic                       out_valid_1_1;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_2;   //duplicate
    logic                       in_ack_1_2   ;
    logic                       out_ack_1_2  ;
    logic                       in_ready_1_2 ;
    logic                       out_valid_1_2;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_3;   //duplicate
    logic                       in_ack_1_3   ;
    logic                       out_ack_1_3  ;
    logic                       in_ready_1_3 ;
    logic                       out_valid_1_3;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_4;   //duplicate
    logic                       in_ack_1_4   ;
    logic                       out_ack_1_4  ;
    logic                       in_ready_1_4 ;
    logic                       out_valid_1_4;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_5;   //duplicate
    logic                       in_ack_1_5   ;
    logic                       out_ack_1_5  ;
    logic                       in_ready_1_5 ;
    logic                       out_valid_1_5;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_6;   //duplicate
    logic                       in_ack_1_6   ;
    logic                       out_ack_1_6  ;
    logic                       in_ready_1_6 ;
    logic                       out_valid_1_6;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_7;   //duplicate
    logic                       in_ack_1_7   ;
    logic                       out_ack_1_7  ;
    logic                       in_ready_1_7 ;
    logic                       out_valid_1_7;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_8;   //duplicate
    logic                       in_ack_1_8   ;
    logic                       out_ack_1_8  ;
    logic                       in_ready_1_8 ;
    logic                       out_valid_1_8;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_9;   //duplicate
    logic                       in_ack_1_9   ;
    logic                       out_ack_1_9  ;
    logic                       in_ready_1_9 ;
    logic                       out_valid_1_9;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_10;   //duplicate
    logic                       in_ack_1_10   ;
    logic                       out_ack_1_10  ;
    logic                       in_ready_1_10 ;
    logic                       out_valid_1_10;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_11;   //duplicate
    logic                       in_ack_1_11   ;
    logic                       out_ack_1_11  ;
    logic                       in_ready_1_11 ;
    logic                       out_valid_1_11;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_12;   //duplicate
    logic                       in_ack_1_12   ;
    logic                       out_ack_1_12  ;
    logic                       in_ready_1_12 ;
    logic                       out_valid_1_12;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_13;   //duplicate
    logic                       in_ack_1_13   ;
    logic                       out_ack_1_13  ;
    logic                       in_ready_1_13 ;
    logic                       out_valid_1_13;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_14;   //duplicate
    logic                       in_ack_1_14   ;
    logic                       out_ack_1_14  ;
    logic                       in_ready_1_14 ;
    logic                       out_valid_1_14;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_1_15;   //duplicate
    logic                       in_ack_1_15   ;
    logic                       out_ack_1_15  ;
    logic                       in_ready_1_15 ;
    logic                       out_valid_1_15;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_0;   //duplicate
    logic                       in_ack_2_0   ;
    logic                       out_ack_2_0  ;
    logic                       in_ready_2_0 ;
    logic                       out_valid_2_0;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_1;   //duplicate
    logic                       in_ack_2_1   ;
    logic                       out_ack_2_1  ;
    logic                       in_ready_2_1 ;
    logic                       out_valid_2_1;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_2;   //duplicate
    logic                       in_ack_2_2   ;
    logic                       out_ack_2_2  ;
    logic                       in_ready_2_2 ;
    logic                       out_valid_2_2;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_3;   //duplicate
    logic                       in_ack_2_3   ;
    logic                       out_ack_2_3  ;
    logic                       in_ready_2_3 ;
    logic                       out_valid_2_3;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_4;   //duplicate
    logic                       in_ack_2_4   ;
    logic                       out_ack_2_4  ;
    logic                       in_ready_2_4 ;
    logic                       out_valid_2_4;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_5;   //duplicate
    logic                       in_ack_2_5   ;
    logic                       out_ack_2_5  ;
    logic                       in_ready_2_5 ;
    logic                       out_valid_2_5;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_6;   //duplicate
    logic                       in_ack_2_6   ;
    logic                       out_ack_2_6  ;
    logic                       in_ready_2_6 ;
    logic                       out_valid_2_6;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_7;   //duplicate
    logic                       in_ack_2_7   ;
    logic                       out_ack_2_7  ;
    logic                       in_ready_2_7 ;
    logic                       out_valid_2_7;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_8;   //duplicate
    logic                       in_ack_2_8   ;
    logic                       out_ack_2_8  ;
    logic                       in_ready_2_8 ;
    logic                       out_valid_2_8;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_9;   //duplicate
    logic                       in_ack_2_9   ;
    logic                       out_ack_2_9  ;
    logic                       in_ready_2_9 ;
    logic                       out_valid_2_9;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_10;   //duplicate
    logic                       in_ack_2_10   ;
    logic                       out_ack_2_10  ;
    logic                       in_ready_2_10 ;
    logic                       out_valid_2_10;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_11;   //duplicate
    logic                       in_ack_2_11   ;
    logic                       out_ack_2_11  ;
    logic                       in_ready_2_11 ;
    logic                       out_valid_2_11;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_12;   //duplicate
    logic                       in_ack_2_12   ;
    logic                       out_ack_2_12  ;
    logic                       in_ready_2_12 ;
    logic                       out_valid_2_12;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_13;   //duplicate
    logic                       in_ack_2_13   ;
    logic                       out_ack_2_13  ;
    logic                       in_ready_2_13 ;
    logic                       out_valid_2_13;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_14;   //duplicate
    logic                       in_ack_2_14   ;
    logic                       out_ack_2_14  ;
    logic                       in_ready_2_14 ;
    logic                       out_valid_2_14;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_2_15;   //duplicate
    logic                       in_ack_2_15   ;
    logic                       out_ack_2_15  ;
    logic                       in_ready_2_15 ;
    logic                       out_valid_2_15;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_0;   //duplicate
    logic                       in_ack_3_0   ;
    logic                       out_ack_3_0  ;
    logic                       in_ready_3_0 ;
    logic                       out_valid_3_0;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_1;   //duplicate
    logic                       in_ack_3_1   ;
    logic                       out_ack_3_1  ;
    logic                       in_ready_3_1 ;
    logic                       out_valid_3_1;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_2;   //duplicate
    logic                       in_ack_3_2   ;
    logic                       out_ack_3_2  ;
    logic                       in_ready_3_2 ;
    logic                       out_valid_3_2;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_3;   //duplicate
    logic                       in_ack_3_3   ;
    logic                       out_ack_3_3  ;
    logic                       in_ready_3_3 ;
    logic                       out_valid_3_3;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_4;   //duplicate
    logic                       in_ack_3_4   ;
    logic                       out_ack_3_4  ;
    logic                       in_ready_3_4 ;
    logic                       out_valid_3_4;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_5;   //duplicate
    logic                       in_ack_3_5   ;
    logic                       out_ack_3_5  ;
    logic                       in_ready_3_5 ;
    logic                       out_valid_3_5;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_6;   //duplicate
    logic                       in_ack_3_6   ;
    logic                       out_ack_3_6  ;
    logic                       in_ready_3_6 ;
    logic                       out_valid_3_6;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_7;   //duplicate
    logic                       in_ack_3_7   ;
    logic                       out_ack_3_7  ;
    logic                       in_ready_3_7 ;
    logic                       out_valid_3_7;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_8;   //duplicate
    logic                       in_ack_3_8   ;
    logic                       out_ack_3_8  ;
    logic                       in_ready_3_8 ;
    logic                       out_valid_3_8;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_9;   //duplicate
    logic                       in_ack_3_9   ;
    logic                       out_ack_3_9  ;
    logic                       in_ready_3_9 ;
    logic                       out_valid_3_9;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_10;   //duplicate
    logic                       in_ack_3_10   ;
    logic                       out_ack_3_10  ;
    logic                       in_ready_3_10 ;
    logic                       out_valid_3_10;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_11;   //duplicate
    logic                       in_ack_3_11   ;
    logic                       out_ack_3_11  ;
    logic                       in_ready_3_11 ;
    logic                       out_valid_3_11;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_12;   //duplicate
    logic                       in_ack_3_12   ;
    logic                       out_ack_3_12  ;
    logic                       in_ready_3_12 ;
    logic                       out_valid_3_12;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_13;   //duplicate
    logic                       in_ack_3_13   ;
    logic                       out_ack_3_13  ;
    logic                       in_ready_3_13 ;
    logic                       out_valid_3_13;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_14;   //duplicate
    logic                       in_ack_3_14   ;
    logic                       out_ack_3_14  ;
    logic                       in_ready_3_14 ;
    logic                       out_valid_3_14;
    logic   [`LOG2((1536+512)/512       )-1:0]  tail_3_15;   //duplicate
    logic                       in_ack_3_15   ;
    logic                       out_ack_3_15  ;
    logic                       in_ready_3_15 ;
    logic                       out_valid_3_15;

    logic                       in_ack;
    logic                       out_ack;
    genvar                      i, j, k;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign in_ack = in_valid && in_ready;
    assign out_ack = out_valid && out_ready;
    assign in_ready = tail + 1536/512           <= (1536+512)/512       ;
    assign out_valid = tail >= 512/512              ;

    assign in_ack_0_0   = in_valid && in_ready_0_0;
    assign out_ack_0_0  = out_valid_0_0 && out_ready;
    assign in_ready_0_0 = tail_0_0 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_0= tail_0_0 >= 512/512              ;
    assign in_ack_0_1   = in_valid && in_ready_0_1;
    assign out_ack_0_1  = out_valid_0_1 && out_ready;
    assign in_ready_0_1 = tail_0_1 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_1= tail_0_1 >= 512/512              ;
    assign in_ack_0_2   = in_valid && in_ready_0_2;
    assign out_ack_0_2  = out_valid_0_2 && out_ready;
    assign in_ready_0_2 = tail_0_2 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_2= tail_0_2 >= 512/512              ;
    assign in_ack_0_3   = in_valid && in_ready_0_3;
    assign out_ack_0_3  = out_valid_0_3 && out_ready;
    assign in_ready_0_3 = tail_0_3 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_3= tail_0_3 >= 512/512              ;
    assign in_ack_0_4   = in_valid && in_ready_0_4;
    assign out_ack_0_4  = out_valid_0_4 && out_ready;
    assign in_ready_0_4 = tail_0_4 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_4= tail_0_4 >= 512/512              ;
    assign in_ack_0_5   = in_valid && in_ready_0_5;
    assign out_ack_0_5  = out_valid_0_5 && out_ready;
    assign in_ready_0_5 = tail_0_5 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_5= tail_0_5 >= 512/512              ;
    assign in_ack_0_6   = in_valid && in_ready_0_6;
    assign out_ack_0_6  = out_valid_0_6 && out_ready;
    assign in_ready_0_6 = tail_0_6 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_6= tail_0_6 >= 512/512              ;
    assign in_ack_0_7   = in_valid && in_ready_0_7;
    assign out_ack_0_7  = out_valid_0_7 && out_ready;
    assign in_ready_0_7 = tail_0_7 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_7= tail_0_7 >= 512/512              ;
    assign in_ack_0_8   = in_valid && in_ready_0_8;
    assign out_ack_0_8  = out_valid_0_8 && out_ready;
    assign in_ready_0_8 = tail_0_8 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_8= tail_0_8 >= 512/512              ;
    assign in_ack_0_9   = in_valid && in_ready_0_9;
    assign out_ack_0_9  = out_valid_0_9 && out_ready;
    assign in_ready_0_9 = tail_0_9 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_9= tail_0_9 >= 512/512              ;
    assign in_ack_0_10   = in_valid && in_ready_0_10;
    assign out_ack_0_10  = out_valid_0_10 && out_ready;
    assign in_ready_0_10 = tail_0_10 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_10= tail_0_10 >= 512/512              ;
    assign in_ack_0_11   = in_valid && in_ready_0_11;
    assign out_ack_0_11  = out_valid_0_11 && out_ready;
    assign in_ready_0_11 = tail_0_11 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_11= tail_0_11 >= 512/512              ;
    assign in_ack_0_12   = in_valid && in_ready_0_12;
    assign out_ack_0_12  = out_valid_0_12 && out_ready;
    assign in_ready_0_12 = tail_0_12 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_12= tail_0_12 >= 512/512              ;
    assign in_ack_0_13   = in_valid && in_ready_0_13;
    assign out_ack_0_13  = out_valid_0_13 && out_ready;
    assign in_ready_0_13 = tail_0_13 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_13= tail_0_13 >= 512/512              ;
    assign in_ack_0_14   = in_valid && in_ready_0_14;
    assign out_ack_0_14  = out_valid_0_14 && out_ready;
    assign in_ready_0_14 = tail_0_14 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_14= tail_0_14 >= 512/512              ;
    assign in_ack_0_15   = in_valid && in_ready_0_15;
    assign out_ack_0_15  = out_valid_0_15 && out_ready;
    assign in_ready_0_15 = tail_0_15 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_0_15= tail_0_15 >= 512/512              ;
    assign in_ack_1_0   = in_valid && in_ready_1_0;
    assign out_ack_1_0  = out_valid_1_0 && out_ready;
    assign in_ready_1_0 = tail_1_0 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_0= tail_1_0 >= 512/512              ;
    assign in_ack_1_1   = in_valid && in_ready_1_1;
    assign out_ack_1_1  = out_valid_1_1 && out_ready;
    assign in_ready_1_1 = tail_1_1 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_1= tail_1_1 >= 512/512              ;
    assign in_ack_1_2   = in_valid && in_ready_1_2;
    assign out_ack_1_2  = out_valid_1_2 && out_ready;
    assign in_ready_1_2 = tail_1_2 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_2= tail_1_2 >= 512/512              ;
    assign in_ack_1_3   = in_valid && in_ready_1_3;
    assign out_ack_1_3  = out_valid_1_3 && out_ready;
    assign in_ready_1_3 = tail_1_3 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_3= tail_1_3 >= 512/512              ;
    assign in_ack_1_4   = in_valid && in_ready_1_4;
    assign out_ack_1_4  = out_valid_1_4 && out_ready;
    assign in_ready_1_4 = tail_1_4 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_4= tail_1_4 >= 512/512              ;
    assign in_ack_1_5   = in_valid && in_ready_1_5;
    assign out_ack_1_5  = out_valid_1_5 && out_ready;
    assign in_ready_1_5 = tail_1_5 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_5= tail_1_5 >= 512/512              ;
    assign in_ack_1_6   = in_valid && in_ready_1_6;
    assign out_ack_1_6  = out_valid_1_6 && out_ready;
    assign in_ready_1_6 = tail_1_6 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_6= tail_1_6 >= 512/512              ;
    assign in_ack_1_7   = in_valid && in_ready_1_7;
    assign out_ack_1_7  = out_valid_1_7 && out_ready;
    assign in_ready_1_7 = tail_1_7 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_7= tail_1_7 >= 512/512              ;
    assign in_ack_1_8   = in_valid && in_ready_1_8;
    assign out_ack_1_8  = out_valid_1_8 && out_ready;
    assign in_ready_1_8 = tail_1_8 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_8= tail_1_8 >= 512/512              ;
    assign in_ack_1_9   = in_valid && in_ready_1_9;
    assign out_ack_1_9  = out_valid_1_9 && out_ready;
    assign in_ready_1_9 = tail_1_9 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_9= tail_1_9 >= 512/512              ;
    assign in_ack_1_10   = in_valid && in_ready_1_10;
    assign out_ack_1_10  = out_valid_1_10 && out_ready;
    assign in_ready_1_10 = tail_1_10 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_10= tail_1_10 >= 512/512              ;
    assign in_ack_1_11   = in_valid && in_ready_1_11;
    assign out_ack_1_11  = out_valid_1_11 && out_ready;
    assign in_ready_1_11 = tail_1_11 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_11= tail_1_11 >= 512/512              ;
    assign in_ack_1_12   = in_valid && in_ready_1_12;
    assign out_ack_1_12  = out_valid_1_12 && out_ready;
    assign in_ready_1_12 = tail_1_12 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_12= tail_1_12 >= 512/512              ;
    assign in_ack_1_13   = in_valid && in_ready_1_13;
    assign out_ack_1_13  = out_valid_1_13 && out_ready;
    assign in_ready_1_13 = tail_1_13 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_13= tail_1_13 >= 512/512              ;
    assign in_ack_1_14   = in_valid && in_ready_1_14;
    assign out_ack_1_14  = out_valid_1_14 && out_ready;
    assign in_ready_1_14 = tail_1_14 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_14= tail_1_14 >= 512/512              ;
    assign in_ack_1_15   = in_valid && in_ready_1_15;
    assign out_ack_1_15  = out_valid_1_15 && out_ready;
    assign in_ready_1_15 = tail_1_15 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_1_15= tail_1_15 >= 512/512              ;
    assign in_ack_2_0   = in_valid && in_ready_2_0;
    assign out_ack_2_0  = out_valid_2_0 && out_ready;
    assign in_ready_2_0 = tail_2_0 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_0= tail_2_0 >= 512/512              ;
    assign in_ack_2_1   = in_valid && in_ready_2_1;
    assign out_ack_2_1  = out_valid_2_1 && out_ready;
    assign in_ready_2_1 = tail_2_1 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_1= tail_2_1 >= 512/512              ;
    assign in_ack_2_2   = in_valid && in_ready_2_2;
    assign out_ack_2_2  = out_valid_2_2 && out_ready;
    assign in_ready_2_2 = tail_2_2 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_2= tail_2_2 >= 512/512              ;
    assign in_ack_2_3   = in_valid && in_ready_2_3;
    assign out_ack_2_3  = out_valid_2_3 && out_ready;
    assign in_ready_2_3 = tail_2_3 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_3= tail_2_3 >= 512/512              ;
    assign in_ack_2_4   = in_valid && in_ready_2_4;
    assign out_ack_2_4  = out_valid_2_4 && out_ready;
    assign in_ready_2_4 = tail_2_4 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_4= tail_2_4 >= 512/512              ;
    assign in_ack_2_5   = in_valid && in_ready_2_5;
    assign out_ack_2_5  = out_valid_2_5 && out_ready;
    assign in_ready_2_5 = tail_2_5 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_5= tail_2_5 >= 512/512              ;
    assign in_ack_2_6   = in_valid && in_ready_2_6;
    assign out_ack_2_6  = out_valid_2_6 && out_ready;
    assign in_ready_2_6 = tail_2_6 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_6= tail_2_6 >= 512/512              ;
    assign in_ack_2_7   = in_valid && in_ready_2_7;
    assign out_ack_2_7  = out_valid_2_7 && out_ready;
    assign in_ready_2_7 = tail_2_7 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_7= tail_2_7 >= 512/512              ;
    assign in_ack_2_8   = in_valid && in_ready_2_8;
    assign out_ack_2_8  = out_valid_2_8 && out_ready;
    assign in_ready_2_8 = tail_2_8 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_8= tail_2_8 >= 512/512              ;
    assign in_ack_2_9   = in_valid && in_ready_2_9;
    assign out_ack_2_9  = out_valid_2_9 && out_ready;
    assign in_ready_2_9 = tail_2_9 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_9= tail_2_9 >= 512/512              ;
    assign in_ack_2_10   = in_valid && in_ready_2_10;
    assign out_ack_2_10  = out_valid_2_10 && out_ready;
    assign in_ready_2_10 = tail_2_10 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_10= tail_2_10 >= 512/512              ;
    assign in_ack_2_11   = in_valid && in_ready_2_11;
    assign out_ack_2_11  = out_valid_2_11 && out_ready;
    assign in_ready_2_11 = tail_2_11 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_11= tail_2_11 >= 512/512              ;
    assign in_ack_2_12   = in_valid && in_ready_2_12;
    assign out_ack_2_12  = out_valid_2_12 && out_ready;
    assign in_ready_2_12 = tail_2_12 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_12= tail_2_12 >= 512/512              ;
    assign in_ack_2_13   = in_valid && in_ready_2_13;
    assign out_ack_2_13  = out_valid_2_13 && out_ready;
    assign in_ready_2_13 = tail_2_13 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_13= tail_2_13 >= 512/512              ;
    assign in_ack_2_14   = in_valid && in_ready_2_14;
    assign out_ack_2_14  = out_valid_2_14 && out_ready;
    assign in_ready_2_14 = tail_2_14 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_14= tail_2_14 >= 512/512              ;
    assign in_ack_2_15   = in_valid && in_ready_2_15;
    assign out_ack_2_15  = out_valid_2_15 && out_ready;
    assign in_ready_2_15 = tail_2_15 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_2_15= tail_2_15 >= 512/512              ;
    assign in_ack_3_0   = in_valid && in_ready_3_0;
    assign out_ack_3_0  = out_valid_3_0 && out_ready;
    assign in_ready_3_0 = tail_3_0 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_0= tail_3_0 >= 512/512              ;
    assign in_ack_3_1   = in_valid && in_ready_3_1;
    assign out_ack_3_1  = out_valid_3_1 && out_ready;
    assign in_ready_3_1 = tail_3_1 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_1= tail_3_1 >= 512/512              ;
    assign in_ack_3_2   = in_valid && in_ready_3_2;
    assign out_ack_3_2  = out_valid_3_2 && out_ready;
    assign in_ready_3_2 = tail_3_2 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_2= tail_3_2 >= 512/512              ;
    assign in_ack_3_3   = in_valid && in_ready_3_3;
    assign out_ack_3_3  = out_valid_3_3 && out_ready;
    assign in_ready_3_3 = tail_3_3 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_3= tail_3_3 >= 512/512              ;
    assign in_ack_3_4   = in_valid && in_ready_3_4;
    assign out_ack_3_4  = out_valid_3_4 && out_ready;
    assign in_ready_3_4 = tail_3_4 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_4= tail_3_4 >= 512/512              ;
    assign in_ack_3_5   = in_valid && in_ready_3_5;
    assign out_ack_3_5  = out_valid_3_5 && out_ready;
    assign in_ready_3_5 = tail_3_5 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_5= tail_3_5 >= 512/512              ;
    assign in_ack_3_6   = in_valid && in_ready_3_6;
    assign out_ack_3_6  = out_valid_3_6 && out_ready;
    assign in_ready_3_6 = tail_3_6 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_6= tail_3_6 >= 512/512              ;
    assign in_ack_3_7   = in_valid && in_ready_3_7;
    assign out_ack_3_7  = out_valid_3_7 && out_ready;
    assign in_ready_3_7 = tail_3_7 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_7= tail_3_7 >= 512/512              ;
    assign in_ack_3_8   = in_valid && in_ready_3_8;
    assign out_ack_3_8  = out_valid_3_8 && out_ready;
    assign in_ready_3_8 = tail_3_8 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_8= tail_3_8 >= 512/512              ;
    assign in_ack_3_9   = in_valid && in_ready_3_9;
    assign out_ack_3_9  = out_valid_3_9 && out_ready;
    assign in_ready_3_9 = tail_3_9 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_9= tail_3_9 >= 512/512              ;
    assign in_ack_3_10   = in_valid && in_ready_3_10;
    assign out_ack_3_10  = out_valid_3_10 && out_ready;
    assign in_ready_3_10 = tail_3_10 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_10= tail_3_10 >= 512/512              ;
    assign in_ack_3_11   = in_valid && in_ready_3_11;
    assign out_ack_3_11  = out_valid_3_11 && out_ready;
    assign in_ready_3_11 = tail_3_11 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_11= tail_3_11 >= 512/512              ;
    assign in_ack_3_12   = in_valid && in_ready_3_12;
    assign out_ack_3_12  = out_valid_3_12 && out_ready;
    assign in_ready_3_12 = tail_3_12 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_12= tail_3_12 >= 512/512              ;
    assign in_ack_3_13   = in_valid && in_ready_3_13;
    assign out_ack_3_13  = out_valid_3_13 && out_ready;
    assign in_ready_3_13 = tail_3_13 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_13= tail_3_13 >= 512/512              ;
    assign in_ack_3_14   = in_valid && in_ready_3_14;
    assign out_ack_3_14  = out_valid_3_14 && out_ready;
    assign in_ready_3_14 = tail_3_14 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_14= tail_3_14 >= 512/512              ;
    assign in_ack_3_15   = in_valid && in_ready_3_15;
    assign out_ack_3_15  = out_valid_3_15 && out_ready;
    assign in_ready_3_15 = tail_3_15 + 1536/512           <= (1536+512)/512       ;
    assign out_valid_3_15= tail_3_15 >= 512/512              ;

    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail <= `CQ 0;
        else if(in_ack && out_ack)
            tail <= `CQ tail + 1536/512           - 512/512              ;
        else if(in_ack)
            tail <= `CQ tail + 1536/512          ;
        else if(out_ack)
            tail <= `CQ tail - 512/512              ;

    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_0 <= `CQ 0;
        else if(in_ack_0_0&& out_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 + 1536/512           - 512/512              ;
        else if(in_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 + 1536/512          ;
        else if(out_ack_0_0)
            tail_0_0 <= `CQ tail_0_0 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_1 <= `CQ 0;
        else if(in_ack_0_1&& out_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 + 1536/512           - 512/512              ;
        else if(in_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 + 1536/512          ;
        else if(out_ack_0_1)
            tail_0_1 <= `CQ tail_0_1 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_2 <= `CQ 0;
        else if(in_ack_0_2&& out_ack_0_2)
            tail_0_2 <= `CQ tail_0_2 + 1536/512           - 512/512              ;
        else if(in_ack_0_2)
            tail_0_2 <= `CQ tail_0_2 + 1536/512          ;
        else if(out_ack_0_2)
            tail_0_2 <= `CQ tail_0_2 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_3 <= `CQ 0;
        else if(in_ack_0_3&& out_ack_0_3)
            tail_0_3 <= `CQ tail_0_3 + 1536/512           - 512/512              ;
        else if(in_ack_0_3)
            tail_0_3 <= `CQ tail_0_3 + 1536/512          ;
        else if(out_ack_0_3)
            tail_0_3 <= `CQ tail_0_3 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_4 <= `CQ 0;
        else if(in_ack_0_4&& out_ack_0_4)
            tail_0_4 <= `CQ tail_0_4 + 1536/512           - 512/512              ;
        else if(in_ack_0_4)
            tail_0_4 <= `CQ tail_0_4 + 1536/512          ;
        else if(out_ack_0_4)
            tail_0_4 <= `CQ tail_0_4 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_5 <= `CQ 0;
        else if(in_ack_0_5&& out_ack_0_5)
            tail_0_5 <= `CQ tail_0_5 + 1536/512           - 512/512              ;
        else if(in_ack_0_5)
            tail_0_5 <= `CQ tail_0_5 + 1536/512          ;
        else if(out_ack_0_5)
            tail_0_5 <= `CQ tail_0_5 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_6 <= `CQ 0;
        else if(in_ack_0_6&& out_ack_0_6)
            tail_0_6 <= `CQ tail_0_6 + 1536/512           - 512/512              ;
        else if(in_ack_0_6)
            tail_0_6 <= `CQ tail_0_6 + 1536/512          ;
        else if(out_ack_0_6)
            tail_0_6 <= `CQ tail_0_6 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_7 <= `CQ 0;
        else if(in_ack_0_7&& out_ack_0_7)
            tail_0_7 <= `CQ tail_0_7 + 1536/512           - 512/512              ;
        else if(in_ack_0_7)
            tail_0_7 <= `CQ tail_0_7 + 1536/512          ;
        else if(out_ack_0_7)
            tail_0_7 <= `CQ tail_0_7 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_8 <= `CQ 0;
        else if(in_ack_0_8&& out_ack_0_8)
            tail_0_8 <= `CQ tail_0_8 + 1536/512           - 512/512              ;
        else if(in_ack_0_8)
            tail_0_8 <= `CQ tail_0_8 + 1536/512          ;
        else if(out_ack_0_8)
            tail_0_8 <= `CQ tail_0_8 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_9 <= `CQ 0;
        else if(in_ack_0_9&& out_ack_0_9)
            tail_0_9 <= `CQ tail_0_9 + 1536/512           - 512/512              ;
        else if(in_ack_0_9)
            tail_0_9 <= `CQ tail_0_9 + 1536/512          ;
        else if(out_ack_0_9)
            tail_0_9 <= `CQ tail_0_9 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_10 <= `CQ 0;
        else if(in_ack_0_10&& out_ack_0_10)
            tail_0_10 <= `CQ tail_0_10 + 1536/512           - 512/512              ;
        else if(in_ack_0_10)
            tail_0_10 <= `CQ tail_0_10 + 1536/512          ;
        else if(out_ack_0_10)
            tail_0_10 <= `CQ tail_0_10 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_11 <= `CQ 0;
        else if(in_ack_0_11&& out_ack_0_11)
            tail_0_11 <= `CQ tail_0_11 + 1536/512           - 512/512              ;
        else if(in_ack_0_11)
            tail_0_11 <= `CQ tail_0_11 + 1536/512          ;
        else if(out_ack_0_11)
            tail_0_11 <= `CQ tail_0_11 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_12 <= `CQ 0;
        else if(in_ack_0_12&& out_ack_0_12)
            tail_0_12 <= `CQ tail_0_12 + 1536/512           - 512/512              ;
        else if(in_ack_0_12)
            tail_0_12 <= `CQ tail_0_12 + 1536/512          ;
        else if(out_ack_0_12)
            tail_0_12 <= `CQ tail_0_12 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_13 <= `CQ 0;
        else if(in_ack_0_13&& out_ack_0_13)
            tail_0_13 <= `CQ tail_0_13 + 1536/512           - 512/512              ;
        else if(in_ack_0_13)
            tail_0_13 <= `CQ tail_0_13 + 1536/512          ;
        else if(out_ack_0_13)
            tail_0_13 <= `CQ tail_0_13 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_14 <= `CQ 0;
        else if(in_ack_0_14&& out_ack_0_14)
            tail_0_14 <= `CQ tail_0_14 + 1536/512           - 512/512              ;
        else if(in_ack_0_14)
            tail_0_14 <= `CQ tail_0_14 + 1536/512          ;
        else if(out_ack_0_14)
            tail_0_14 <= `CQ tail_0_14 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_0_15 <= `CQ 0;
        else if(in_ack_0_15&& out_ack_0_15)
            tail_0_15 <= `CQ tail_0_15 + 1536/512           - 512/512              ;
        else if(in_ack_0_15)
            tail_0_15 <= `CQ tail_0_15 + 1536/512          ;
        else if(out_ack_0_15)
            tail_0_15 <= `CQ tail_0_15 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_0 <= `CQ 0;
        else if(in_ack_1_0&& out_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 + 1536/512           - 512/512              ;
        else if(in_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 + 1536/512          ;
        else if(out_ack_1_0)
            tail_1_0 <= `CQ tail_1_0 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_1 <= `CQ 0;
        else if(in_ack_1_1&& out_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 + 1536/512           - 512/512              ;
        else if(in_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 + 1536/512          ;
        else if(out_ack_1_1)
            tail_1_1 <= `CQ tail_1_1 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_2 <= `CQ 0;
        else if(in_ack_1_2&& out_ack_1_2)
            tail_1_2 <= `CQ tail_1_2 + 1536/512           - 512/512              ;
        else if(in_ack_1_2)
            tail_1_2 <= `CQ tail_1_2 + 1536/512          ;
        else if(out_ack_1_2)
            tail_1_2 <= `CQ tail_1_2 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_3 <= `CQ 0;
        else if(in_ack_1_3&& out_ack_1_3)
            tail_1_3 <= `CQ tail_1_3 + 1536/512           - 512/512              ;
        else if(in_ack_1_3)
            tail_1_3 <= `CQ tail_1_3 + 1536/512          ;
        else if(out_ack_1_3)
            tail_1_3 <= `CQ tail_1_3 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_4 <= `CQ 0;
        else if(in_ack_1_4&& out_ack_1_4)
            tail_1_4 <= `CQ tail_1_4 + 1536/512           - 512/512              ;
        else if(in_ack_1_4)
            tail_1_4 <= `CQ tail_1_4 + 1536/512          ;
        else if(out_ack_1_4)
            tail_1_4 <= `CQ tail_1_4 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_5 <= `CQ 0;
        else if(in_ack_1_5&& out_ack_1_5)
            tail_1_5 <= `CQ tail_1_5 + 1536/512           - 512/512              ;
        else if(in_ack_1_5)
            tail_1_5 <= `CQ tail_1_5 + 1536/512          ;
        else if(out_ack_1_5)
            tail_1_5 <= `CQ tail_1_5 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_6 <= `CQ 0;
        else if(in_ack_1_6&& out_ack_1_6)
            tail_1_6 <= `CQ tail_1_6 + 1536/512           - 512/512              ;
        else if(in_ack_1_6)
            tail_1_6 <= `CQ tail_1_6 + 1536/512          ;
        else if(out_ack_1_6)
            tail_1_6 <= `CQ tail_1_6 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_7 <= `CQ 0;
        else if(in_ack_1_7&& out_ack_1_7)
            tail_1_7 <= `CQ tail_1_7 + 1536/512           - 512/512              ;
        else if(in_ack_1_7)
            tail_1_7 <= `CQ tail_1_7 + 1536/512          ;
        else if(out_ack_1_7)
            tail_1_7 <= `CQ tail_1_7 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_8 <= `CQ 0;
        else if(in_ack_1_8&& out_ack_1_8)
            tail_1_8 <= `CQ tail_1_8 + 1536/512           - 512/512              ;
        else if(in_ack_1_8)
            tail_1_8 <= `CQ tail_1_8 + 1536/512          ;
        else if(out_ack_1_8)
            tail_1_8 <= `CQ tail_1_8 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_9 <= `CQ 0;
        else if(in_ack_1_9&& out_ack_1_9)
            tail_1_9 <= `CQ tail_1_9 + 1536/512           - 512/512              ;
        else if(in_ack_1_9)
            tail_1_9 <= `CQ tail_1_9 + 1536/512          ;
        else if(out_ack_1_9)
            tail_1_9 <= `CQ tail_1_9 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_10 <= `CQ 0;
        else if(in_ack_1_10&& out_ack_1_10)
            tail_1_10 <= `CQ tail_1_10 + 1536/512           - 512/512              ;
        else if(in_ack_1_10)
            tail_1_10 <= `CQ tail_1_10 + 1536/512          ;
        else if(out_ack_1_10)
            tail_1_10 <= `CQ tail_1_10 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_11 <= `CQ 0;
        else if(in_ack_1_11&& out_ack_1_11)
            tail_1_11 <= `CQ tail_1_11 + 1536/512           - 512/512              ;
        else if(in_ack_1_11)
            tail_1_11 <= `CQ tail_1_11 + 1536/512          ;
        else if(out_ack_1_11)
            tail_1_11 <= `CQ tail_1_11 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_12 <= `CQ 0;
        else if(in_ack_1_12&& out_ack_1_12)
            tail_1_12 <= `CQ tail_1_12 + 1536/512           - 512/512              ;
        else if(in_ack_1_12)
            tail_1_12 <= `CQ tail_1_12 + 1536/512          ;
        else if(out_ack_1_12)
            tail_1_12 <= `CQ tail_1_12 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_13 <= `CQ 0;
        else if(in_ack_1_13&& out_ack_1_13)
            tail_1_13 <= `CQ tail_1_13 + 1536/512           - 512/512              ;
        else if(in_ack_1_13)
            tail_1_13 <= `CQ tail_1_13 + 1536/512          ;
        else if(out_ack_1_13)
            tail_1_13 <= `CQ tail_1_13 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_14 <= `CQ 0;
        else if(in_ack_1_14&& out_ack_1_14)
            tail_1_14 <= `CQ tail_1_14 + 1536/512           - 512/512              ;
        else if(in_ack_1_14)
            tail_1_14 <= `CQ tail_1_14 + 1536/512          ;
        else if(out_ack_1_14)
            tail_1_14 <= `CQ tail_1_14 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_1_15 <= `CQ 0;
        else if(in_ack_1_15&& out_ack_1_15)
            tail_1_15 <= `CQ tail_1_15 + 1536/512           - 512/512              ;
        else if(in_ack_1_15)
            tail_1_15 <= `CQ tail_1_15 + 1536/512          ;
        else if(out_ack_1_15)
            tail_1_15 <= `CQ tail_1_15 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_0 <= `CQ 0;
        else if(in_ack_2_0&& out_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 + 1536/512           - 512/512              ;
        else if(in_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 + 1536/512          ;
        else if(out_ack_2_0)
            tail_2_0 <= `CQ tail_2_0 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_1 <= `CQ 0;
        else if(in_ack_2_1&& out_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 + 1536/512           - 512/512              ;
        else if(in_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 + 1536/512          ;
        else if(out_ack_2_1)
            tail_2_1 <= `CQ tail_2_1 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_2 <= `CQ 0;
        else if(in_ack_2_2&& out_ack_2_2)
            tail_2_2 <= `CQ tail_2_2 + 1536/512           - 512/512              ;
        else if(in_ack_2_2)
            tail_2_2 <= `CQ tail_2_2 + 1536/512          ;
        else if(out_ack_2_2)
            tail_2_2 <= `CQ tail_2_2 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_3 <= `CQ 0;
        else if(in_ack_2_3&& out_ack_2_3)
            tail_2_3 <= `CQ tail_2_3 + 1536/512           - 512/512              ;
        else if(in_ack_2_3)
            tail_2_3 <= `CQ tail_2_3 + 1536/512          ;
        else if(out_ack_2_3)
            tail_2_3 <= `CQ tail_2_3 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_4 <= `CQ 0;
        else if(in_ack_2_4&& out_ack_2_4)
            tail_2_4 <= `CQ tail_2_4 + 1536/512           - 512/512              ;
        else if(in_ack_2_4)
            tail_2_4 <= `CQ tail_2_4 + 1536/512          ;
        else if(out_ack_2_4)
            tail_2_4 <= `CQ tail_2_4 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_5 <= `CQ 0;
        else if(in_ack_2_5&& out_ack_2_5)
            tail_2_5 <= `CQ tail_2_5 + 1536/512           - 512/512              ;
        else if(in_ack_2_5)
            tail_2_5 <= `CQ tail_2_5 + 1536/512          ;
        else if(out_ack_2_5)
            tail_2_5 <= `CQ tail_2_5 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_6 <= `CQ 0;
        else if(in_ack_2_6&& out_ack_2_6)
            tail_2_6 <= `CQ tail_2_6 + 1536/512           - 512/512              ;
        else if(in_ack_2_6)
            tail_2_6 <= `CQ tail_2_6 + 1536/512          ;
        else if(out_ack_2_6)
            tail_2_6 <= `CQ tail_2_6 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_7 <= `CQ 0;
        else if(in_ack_2_7&& out_ack_2_7)
            tail_2_7 <= `CQ tail_2_7 + 1536/512           - 512/512              ;
        else if(in_ack_2_7)
            tail_2_7 <= `CQ tail_2_7 + 1536/512          ;
        else if(out_ack_2_7)
            tail_2_7 <= `CQ tail_2_7 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_8 <= `CQ 0;
        else if(in_ack_2_8&& out_ack_2_8)
            tail_2_8 <= `CQ tail_2_8 + 1536/512           - 512/512              ;
        else if(in_ack_2_8)
            tail_2_8 <= `CQ tail_2_8 + 1536/512          ;
        else if(out_ack_2_8)
            tail_2_8 <= `CQ tail_2_8 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_9 <= `CQ 0;
        else if(in_ack_2_9&& out_ack_2_9)
            tail_2_9 <= `CQ tail_2_9 + 1536/512           - 512/512              ;
        else if(in_ack_2_9)
            tail_2_9 <= `CQ tail_2_9 + 1536/512          ;
        else if(out_ack_2_9)
            tail_2_9 <= `CQ tail_2_9 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_10 <= `CQ 0;
        else if(in_ack_2_10&& out_ack_2_10)
            tail_2_10 <= `CQ tail_2_10 + 1536/512           - 512/512              ;
        else if(in_ack_2_10)
            tail_2_10 <= `CQ tail_2_10 + 1536/512          ;
        else if(out_ack_2_10)
            tail_2_10 <= `CQ tail_2_10 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_11 <= `CQ 0;
        else if(in_ack_2_11&& out_ack_2_11)
            tail_2_11 <= `CQ tail_2_11 + 1536/512           - 512/512              ;
        else if(in_ack_2_11)
            tail_2_11 <= `CQ tail_2_11 + 1536/512          ;
        else if(out_ack_2_11)
            tail_2_11 <= `CQ tail_2_11 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_12 <= `CQ 0;
        else if(in_ack_2_12&& out_ack_2_12)
            tail_2_12 <= `CQ tail_2_12 + 1536/512           - 512/512              ;
        else if(in_ack_2_12)
            tail_2_12 <= `CQ tail_2_12 + 1536/512          ;
        else if(out_ack_2_12)
            tail_2_12 <= `CQ tail_2_12 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_13 <= `CQ 0;
        else if(in_ack_2_13&& out_ack_2_13)
            tail_2_13 <= `CQ tail_2_13 + 1536/512           - 512/512              ;
        else if(in_ack_2_13)
            tail_2_13 <= `CQ tail_2_13 + 1536/512          ;
        else if(out_ack_2_13)
            tail_2_13 <= `CQ tail_2_13 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_14 <= `CQ 0;
        else if(in_ack_2_14&& out_ack_2_14)
            tail_2_14 <= `CQ tail_2_14 + 1536/512           - 512/512              ;
        else if(in_ack_2_14)
            tail_2_14 <= `CQ tail_2_14 + 1536/512          ;
        else if(out_ack_2_14)
            tail_2_14 <= `CQ tail_2_14 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_2_15 <= `CQ 0;
        else if(in_ack_2_15&& out_ack_2_15)
            tail_2_15 <= `CQ tail_2_15 + 1536/512           - 512/512              ;
        else if(in_ack_2_15)
            tail_2_15 <= `CQ tail_2_15 + 1536/512          ;
        else if(out_ack_2_15)
            tail_2_15 <= `CQ tail_2_15 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_0 <= `CQ 0;
        else if(in_ack_3_0&& out_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 + 1536/512           - 512/512              ;
        else if(in_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 + 1536/512          ;
        else if(out_ack_3_0)
            tail_3_0 <= `CQ tail_3_0 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_1 <= `CQ 0;
        else if(in_ack_3_1&& out_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 + 1536/512           - 512/512              ;
        else if(in_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 + 1536/512          ;
        else if(out_ack_3_1)
            tail_3_1 <= `CQ tail_3_1 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_2 <= `CQ 0;
        else if(in_ack_3_2&& out_ack_3_2)
            tail_3_2 <= `CQ tail_3_2 + 1536/512           - 512/512              ;
        else if(in_ack_3_2)
            tail_3_2 <= `CQ tail_3_2 + 1536/512          ;
        else if(out_ack_3_2)
            tail_3_2 <= `CQ tail_3_2 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_3 <= `CQ 0;
        else if(in_ack_3_3&& out_ack_3_3)
            tail_3_3 <= `CQ tail_3_3 + 1536/512           - 512/512              ;
        else if(in_ack_3_3)
            tail_3_3 <= `CQ tail_3_3 + 1536/512          ;
        else if(out_ack_3_3)
            tail_3_3 <= `CQ tail_3_3 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_4 <= `CQ 0;
        else if(in_ack_3_4&& out_ack_3_4)
            tail_3_4 <= `CQ tail_3_4 + 1536/512           - 512/512              ;
        else if(in_ack_3_4)
            tail_3_4 <= `CQ tail_3_4 + 1536/512          ;
        else if(out_ack_3_4)
            tail_3_4 <= `CQ tail_3_4 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_5 <= `CQ 0;
        else if(in_ack_3_5&& out_ack_3_5)
            tail_3_5 <= `CQ tail_3_5 + 1536/512           - 512/512              ;
        else if(in_ack_3_5)
            tail_3_5 <= `CQ tail_3_5 + 1536/512          ;
        else if(out_ack_3_5)
            tail_3_5 <= `CQ tail_3_5 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_6 <= `CQ 0;
        else if(in_ack_3_6&& out_ack_3_6)
            tail_3_6 <= `CQ tail_3_6 + 1536/512           - 512/512              ;
        else if(in_ack_3_6)
            tail_3_6 <= `CQ tail_3_6 + 1536/512          ;
        else if(out_ack_3_6)
            tail_3_6 <= `CQ tail_3_6 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_7 <= `CQ 0;
        else if(in_ack_3_7&& out_ack_3_7)
            tail_3_7 <= `CQ tail_3_7 + 1536/512           - 512/512              ;
        else if(in_ack_3_7)
            tail_3_7 <= `CQ tail_3_7 + 1536/512          ;
        else if(out_ack_3_7)
            tail_3_7 <= `CQ tail_3_7 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_8 <= `CQ 0;
        else if(in_ack_3_8&& out_ack_3_8)
            tail_3_8 <= `CQ tail_3_8 + 1536/512           - 512/512              ;
        else if(in_ack_3_8)
            tail_3_8 <= `CQ tail_3_8 + 1536/512          ;
        else if(out_ack_3_8)
            tail_3_8 <= `CQ tail_3_8 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_9 <= `CQ 0;
        else if(in_ack_3_9&& out_ack_3_9)
            tail_3_9 <= `CQ tail_3_9 + 1536/512           - 512/512              ;
        else if(in_ack_3_9)
            tail_3_9 <= `CQ tail_3_9 + 1536/512          ;
        else if(out_ack_3_9)
            tail_3_9 <= `CQ tail_3_9 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_10 <= `CQ 0;
        else if(in_ack_3_10&& out_ack_3_10)
            tail_3_10 <= `CQ tail_3_10 + 1536/512           - 512/512              ;
        else if(in_ack_3_10)
            tail_3_10 <= `CQ tail_3_10 + 1536/512          ;
        else if(out_ack_3_10)
            tail_3_10 <= `CQ tail_3_10 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_11 <= `CQ 0;
        else if(in_ack_3_11&& out_ack_3_11)
            tail_3_11 <= `CQ tail_3_11 + 1536/512           - 512/512              ;
        else if(in_ack_3_11)
            tail_3_11 <= `CQ tail_3_11 + 1536/512          ;
        else if(out_ack_3_11)
            tail_3_11 <= `CQ tail_3_11 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_12 <= `CQ 0;
        else if(in_ack_3_12&& out_ack_3_12)
            tail_3_12 <= `CQ tail_3_12 + 1536/512           - 512/512              ;
        else if(in_ack_3_12)
            tail_3_12 <= `CQ tail_3_12 + 1536/512          ;
        else if(out_ack_3_12)
            tail_3_12 <= `CQ tail_3_12 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_13 <= `CQ 0;
        else if(in_ack_3_13&& out_ack_3_13)
            tail_3_13 <= `CQ tail_3_13 + 1536/512           - 512/512              ;
        else if(in_ack_3_13)
            tail_3_13 <= `CQ tail_3_13 + 1536/512          ;
        else if(out_ack_3_13)
            tail_3_13 <= `CQ tail_3_13 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_14 <= `CQ 0;
        else if(in_ack_3_14&& out_ack_3_14)
            tail_3_14 <= `CQ tail_3_14 + 1536/512           - 512/512              ;
        else if(in_ack_3_14)
            tail_3_14 <= `CQ tail_3_14 + 1536/512          ;
        else if(out_ack_3_14)
            tail_3_14 <= `CQ tail_3_14 - 512/512              ;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            tail_3_15 <= `CQ 0;
        else if(in_ack_3_15&& out_ack_3_15)
            tail_3_15 <= `CQ tail_3_15 + 1536/512           - 512/512              ;
        else if(in_ack_3_15)
            tail_3_15 <= `CQ tail_3_15 + 1536/512          ;
        else if(out_ack_3_15)
            tail_3_15 <= `CQ tail_3_15 - 512/512              ;

//split bufferk to 32bit, for fanout timing

    always@(posedge clk)
        if(in_ack_0_0 && out_ack_0_0) begin
            if(0+512/512               < tail_0_0)
                buffer0[32*0+:32] <= `CQ buffer1[32*0+:32];
            if(0+512/512               >= tail_0_0+0 && 0+512/512               < tail_0_0+0+1)
                buffer0[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(0+512/512               >= tail_0_0+1 && 0+512/512               < tail_0_0+1+1)
                buffer0[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(0+512/512               >= tail_0_0+2 && 0+512/512               < tail_0_0+2+1)
                buffer0[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(in_ack_0_0) begin
            if(0>= tail_0_0+0 && 0 < tail_0_0+0+1)
                buffer0[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(0>= tail_0_0+1 && 0 < tail_0_0+1+1)
                buffer0[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(0>= tail_0_0+2 && 0 < tail_0_0+2+1)
                buffer0[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(out_ack_0_0) begin
            if(0+512/512               < tail_0_0)
                buffer0[32*0+:32] <= `CQ buffer1[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_1_0 && out_ack_1_0) begin
            if(1+512/512               < tail_1_0)
                buffer1[32*0+:32] <= `CQ buffer2[32*0+:32];
            if(1+512/512               >= tail_1_0+0 && 1+512/512               < tail_1_0+0+1)
                buffer1[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(1+512/512               >= tail_1_0+1 && 1+512/512               < tail_1_0+1+1)
                buffer1[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(1+512/512               >= tail_1_0+2 && 1+512/512               < tail_1_0+2+1)
                buffer1[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(in_ack_1_0) begin
            if(1>= tail_1_0+0 && 1 < tail_1_0+0+1)
                buffer1[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(1>= tail_1_0+1 && 1 < tail_1_0+1+1)
                buffer1[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(1>= tail_1_0+2 && 1 < tail_1_0+2+1)
                buffer1[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(out_ack_1_0) begin
            if(1+512/512               < tail_1_0)
                buffer1[32*0+:32] <= `CQ buffer2[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_2_0 && out_ack_2_0) begin
            if(2+512/512               < tail_2_0)
                buffer2[32*0+:32] <= `CQ buffer3[32*0+:32];
            if(2+512/512               >= tail_2_0+0 && 2+512/512               < tail_2_0+0+1)
                buffer2[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(2+512/512               >= tail_2_0+1 && 2+512/512               < tail_2_0+1+1)
                buffer2[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(2+512/512               >= tail_2_0+2 && 2+512/512               < tail_2_0+2+1)
                buffer2[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(in_ack_2_0) begin
            if(2>= tail_2_0+0 && 2 < tail_2_0+0+1)
                buffer2[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(2>= tail_2_0+1 && 2 < tail_2_0+1+1)
                buffer2[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(2>= tail_2_0+2 && 2 < tail_2_0+2+1)
                buffer2[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(out_ack_2_0) begin
            if(2+512/512               < tail_2_0)
                buffer2[32*0+:32] <= `CQ buffer3[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_3_0 && out_ack_3_0) begin
            if(3+512/512               < tail_3_0)
                buffer3[32*0+:32] <= `CQ buffer4[32*0+:32];
            if(3+512/512               >= tail_3_0+0 && 3+512/512               < tail_3_0+0+1)
                buffer3[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(3+512/512               >= tail_3_0+1 && 3+512/512               < tail_3_0+1+1)
                buffer3[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(3+512/512               >= tail_3_0+2 && 3+512/512               < tail_3_0+2+1)
                buffer3[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(in_ack_3_0) begin
            if(3>= tail_3_0+0 && 3 < tail_3_0+0+1)
                buffer3[32*0+:32] <= `CQ in_data[512       *0+32*0+:32];
            if(3>= tail_3_0+1 && 3 < tail_3_0+1+1)
                buffer3[32*0+:32] <= `CQ in_data[512       *1+32*0+:32];
            if(3>= tail_3_0+2 && 3 < tail_3_0+2+1)
                buffer3[32*0+:32] <= `CQ in_data[512       *2+32*0+:32];
        end
        else if(out_ack_3_0) begin
            if(3+512/512               < tail_3_0)
                buffer3[32*0+:32] <= `CQ buffer4[32*0+:32];
        end
    always@(posedge clk)
        if(in_ack_0_1 && out_ack_0_1) begin
            if(0+512/512               < tail_0_1)
                buffer0[32*1+:32] <= `CQ buffer1[32*1+:32];
            if(0+512/512               >= tail_0_1+0 && 0+512/512               < tail_0_1+0+1)
                buffer0[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(0+512/512               >= tail_0_1+1 && 0+512/512               < tail_0_1+1+1)
                buffer0[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(0+512/512               >= tail_0_1+2 && 0+512/512               < tail_0_1+2+1)
                buffer0[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(in_ack_0_1) begin
            if(0>= tail_0_1+0 && 0 < tail_0_1+0+1)
                buffer0[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(0>= tail_0_1+1 && 0 < tail_0_1+1+1)
                buffer0[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(0>= tail_0_1+2 && 0 < tail_0_1+2+1)
                buffer0[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(out_ack_0_1) begin
            if(0+512/512               < tail_0_1)
                buffer0[32*1+:32] <= `CQ buffer1[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_1_1 && out_ack_1_1) begin
            if(1+512/512               < tail_1_1)
                buffer1[32*1+:32] <= `CQ buffer2[32*1+:32];
            if(1+512/512               >= tail_1_1+0 && 1+512/512               < tail_1_1+0+1)
                buffer1[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(1+512/512               >= tail_1_1+1 && 1+512/512               < tail_1_1+1+1)
                buffer1[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(1+512/512               >= tail_1_1+2 && 1+512/512               < tail_1_1+2+1)
                buffer1[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(in_ack_1_1) begin
            if(1>= tail_1_1+0 && 1 < tail_1_1+0+1)
                buffer1[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(1>= tail_1_1+1 && 1 < tail_1_1+1+1)
                buffer1[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(1>= tail_1_1+2 && 1 < tail_1_1+2+1)
                buffer1[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(out_ack_1_1) begin
            if(1+512/512               < tail_1_1)
                buffer1[32*1+:32] <= `CQ buffer2[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_2_1 && out_ack_2_1) begin
            if(2+512/512               < tail_2_1)
                buffer2[32*1+:32] <= `CQ buffer3[32*1+:32];
            if(2+512/512               >= tail_2_1+0 && 2+512/512               < tail_2_1+0+1)
                buffer2[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(2+512/512               >= tail_2_1+1 && 2+512/512               < tail_2_1+1+1)
                buffer2[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(2+512/512               >= tail_2_1+2 && 2+512/512               < tail_2_1+2+1)
                buffer2[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(in_ack_2_1) begin
            if(2>= tail_2_1+0 && 2 < tail_2_1+0+1)
                buffer2[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(2>= tail_2_1+1 && 2 < tail_2_1+1+1)
                buffer2[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(2>= tail_2_1+2 && 2 < tail_2_1+2+1)
                buffer2[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(out_ack_2_1) begin
            if(2+512/512               < tail_2_1)
                buffer2[32*1+:32] <= `CQ buffer3[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_3_1 && out_ack_3_1) begin
            if(3+512/512               < tail_3_1)
                buffer3[32*1+:32] <= `CQ buffer4[32*1+:32];
            if(3+512/512               >= tail_3_1+0 && 3+512/512               < tail_3_1+0+1)
                buffer3[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(3+512/512               >= tail_3_1+1 && 3+512/512               < tail_3_1+1+1)
                buffer3[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(3+512/512               >= tail_3_1+2 && 3+512/512               < tail_3_1+2+1)
                buffer3[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(in_ack_3_1) begin
            if(3>= tail_3_1+0 && 3 < tail_3_1+0+1)
                buffer3[32*1+:32] <= `CQ in_data[512       *0+32*1+:32];
            if(3>= tail_3_1+1 && 3 < tail_3_1+1+1)
                buffer3[32*1+:32] <= `CQ in_data[512       *1+32*1+:32];
            if(3>= tail_3_1+2 && 3 < tail_3_1+2+1)
                buffer3[32*1+:32] <= `CQ in_data[512       *2+32*1+:32];
        end
        else if(out_ack_3_1) begin
            if(3+512/512               < tail_3_1)
                buffer3[32*1+:32] <= `CQ buffer4[32*1+:32];
        end
    always@(posedge clk)
        if(in_ack_0_2 && out_ack_0_2) begin
            if(0+512/512               < tail_0_2)
                buffer0[32*2+:32] <= `CQ buffer1[32*2+:32];
            if(0+512/512               >= tail_0_2+0 && 0+512/512               < tail_0_2+0+1)
                buffer0[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(0+512/512               >= tail_0_2+1 && 0+512/512               < tail_0_2+1+1)
                buffer0[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(0+512/512               >= tail_0_2+2 && 0+512/512               < tail_0_2+2+1)
                buffer0[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(in_ack_0_2) begin
            if(0>= tail_0_2+0 && 0 < tail_0_2+0+1)
                buffer0[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(0>= tail_0_2+1 && 0 < tail_0_2+1+1)
                buffer0[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(0>= tail_0_2+2 && 0 < tail_0_2+2+1)
                buffer0[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(out_ack_0_2) begin
            if(0+512/512               < tail_0_2)
                buffer0[32*2+:32] <= `CQ buffer1[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_1_2 && out_ack_1_2) begin
            if(1+512/512               < tail_1_2)
                buffer1[32*2+:32] <= `CQ buffer2[32*2+:32];
            if(1+512/512               >= tail_1_2+0 && 1+512/512               < tail_1_2+0+1)
                buffer1[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(1+512/512               >= tail_1_2+1 && 1+512/512               < tail_1_2+1+1)
                buffer1[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(1+512/512               >= tail_1_2+2 && 1+512/512               < tail_1_2+2+1)
                buffer1[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(in_ack_1_2) begin
            if(1>= tail_1_2+0 && 1 < tail_1_2+0+1)
                buffer1[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(1>= tail_1_2+1 && 1 < tail_1_2+1+1)
                buffer1[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(1>= tail_1_2+2 && 1 < tail_1_2+2+1)
                buffer1[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(out_ack_1_2) begin
            if(1+512/512               < tail_1_2)
                buffer1[32*2+:32] <= `CQ buffer2[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_2_2 && out_ack_2_2) begin
            if(2+512/512               < tail_2_2)
                buffer2[32*2+:32] <= `CQ buffer3[32*2+:32];
            if(2+512/512               >= tail_2_2+0 && 2+512/512               < tail_2_2+0+1)
                buffer2[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(2+512/512               >= tail_2_2+1 && 2+512/512               < tail_2_2+1+1)
                buffer2[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(2+512/512               >= tail_2_2+2 && 2+512/512               < tail_2_2+2+1)
                buffer2[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(in_ack_2_2) begin
            if(2>= tail_2_2+0 && 2 < tail_2_2+0+1)
                buffer2[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(2>= tail_2_2+1 && 2 < tail_2_2+1+1)
                buffer2[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(2>= tail_2_2+2 && 2 < tail_2_2+2+1)
                buffer2[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(out_ack_2_2) begin
            if(2+512/512               < tail_2_2)
                buffer2[32*2+:32] <= `CQ buffer3[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_3_2 && out_ack_3_2) begin
            if(3+512/512               < tail_3_2)
                buffer3[32*2+:32] <= `CQ buffer4[32*2+:32];
            if(3+512/512               >= tail_3_2+0 && 3+512/512               < tail_3_2+0+1)
                buffer3[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(3+512/512               >= tail_3_2+1 && 3+512/512               < tail_3_2+1+1)
                buffer3[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(3+512/512               >= tail_3_2+2 && 3+512/512               < tail_3_2+2+1)
                buffer3[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(in_ack_3_2) begin
            if(3>= tail_3_2+0 && 3 < tail_3_2+0+1)
                buffer3[32*2+:32] <= `CQ in_data[512       *0+32*2+:32];
            if(3>= tail_3_2+1 && 3 < tail_3_2+1+1)
                buffer3[32*2+:32] <= `CQ in_data[512       *1+32*2+:32];
            if(3>= tail_3_2+2 && 3 < tail_3_2+2+1)
                buffer3[32*2+:32] <= `CQ in_data[512       *2+32*2+:32];
        end
        else if(out_ack_3_2) begin
            if(3+512/512               < tail_3_2)
                buffer3[32*2+:32] <= `CQ buffer4[32*2+:32];
        end
    always@(posedge clk)
        if(in_ack_0_3 && out_ack_0_3) begin
            if(0+512/512               < tail_0_3)
                buffer0[32*3+:32] <= `CQ buffer1[32*3+:32];
            if(0+512/512               >= tail_0_3+0 && 0+512/512               < tail_0_3+0+1)
                buffer0[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(0+512/512               >= tail_0_3+1 && 0+512/512               < tail_0_3+1+1)
                buffer0[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(0+512/512               >= tail_0_3+2 && 0+512/512               < tail_0_3+2+1)
                buffer0[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(in_ack_0_3) begin
            if(0>= tail_0_3+0 && 0 < tail_0_3+0+1)
                buffer0[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(0>= tail_0_3+1 && 0 < tail_0_3+1+1)
                buffer0[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(0>= tail_0_3+2 && 0 < tail_0_3+2+1)
                buffer0[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(out_ack_0_3) begin
            if(0+512/512               < tail_0_3)
                buffer0[32*3+:32] <= `CQ buffer1[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_1_3 && out_ack_1_3) begin
            if(1+512/512               < tail_1_3)
                buffer1[32*3+:32] <= `CQ buffer2[32*3+:32];
            if(1+512/512               >= tail_1_3+0 && 1+512/512               < tail_1_3+0+1)
                buffer1[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(1+512/512               >= tail_1_3+1 && 1+512/512               < tail_1_3+1+1)
                buffer1[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(1+512/512               >= tail_1_3+2 && 1+512/512               < tail_1_3+2+1)
                buffer1[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(in_ack_1_3) begin
            if(1>= tail_1_3+0 && 1 < tail_1_3+0+1)
                buffer1[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(1>= tail_1_3+1 && 1 < tail_1_3+1+1)
                buffer1[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(1>= tail_1_3+2 && 1 < tail_1_3+2+1)
                buffer1[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(out_ack_1_3) begin
            if(1+512/512               < tail_1_3)
                buffer1[32*3+:32] <= `CQ buffer2[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_2_3 && out_ack_2_3) begin
            if(2+512/512               < tail_2_3)
                buffer2[32*3+:32] <= `CQ buffer3[32*3+:32];
            if(2+512/512               >= tail_2_3+0 && 2+512/512               < tail_2_3+0+1)
                buffer2[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(2+512/512               >= tail_2_3+1 && 2+512/512               < tail_2_3+1+1)
                buffer2[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(2+512/512               >= tail_2_3+2 && 2+512/512               < tail_2_3+2+1)
                buffer2[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(in_ack_2_3) begin
            if(2>= tail_2_3+0 && 2 < tail_2_3+0+1)
                buffer2[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(2>= tail_2_3+1 && 2 < tail_2_3+1+1)
                buffer2[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(2>= tail_2_3+2 && 2 < tail_2_3+2+1)
                buffer2[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(out_ack_2_3) begin
            if(2+512/512               < tail_2_3)
                buffer2[32*3+:32] <= `CQ buffer3[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_3_3 && out_ack_3_3) begin
            if(3+512/512               < tail_3_3)
                buffer3[32*3+:32] <= `CQ buffer4[32*3+:32];
            if(3+512/512               >= tail_3_3+0 && 3+512/512               < tail_3_3+0+1)
                buffer3[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(3+512/512               >= tail_3_3+1 && 3+512/512               < tail_3_3+1+1)
                buffer3[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(3+512/512               >= tail_3_3+2 && 3+512/512               < tail_3_3+2+1)
                buffer3[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(in_ack_3_3) begin
            if(3>= tail_3_3+0 && 3 < tail_3_3+0+1)
                buffer3[32*3+:32] <= `CQ in_data[512       *0+32*3+:32];
            if(3>= tail_3_3+1 && 3 < tail_3_3+1+1)
                buffer3[32*3+:32] <= `CQ in_data[512       *1+32*3+:32];
            if(3>= tail_3_3+2 && 3 < tail_3_3+2+1)
                buffer3[32*3+:32] <= `CQ in_data[512       *2+32*3+:32];
        end
        else if(out_ack_3_3) begin
            if(3+512/512               < tail_3_3)
                buffer3[32*3+:32] <= `CQ buffer4[32*3+:32];
        end
    always@(posedge clk)
        if(in_ack_0_4 && out_ack_0_4) begin
            if(0+512/512               < tail_0_4)
                buffer0[32*4+:32] <= `CQ buffer1[32*4+:32];
            if(0+512/512               >= tail_0_4+0 && 0+512/512               < tail_0_4+0+1)
                buffer0[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(0+512/512               >= tail_0_4+1 && 0+512/512               < tail_0_4+1+1)
                buffer0[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(0+512/512               >= tail_0_4+2 && 0+512/512               < tail_0_4+2+1)
                buffer0[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(in_ack_0_4) begin
            if(0>= tail_0_4+0 && 0 < tail_0_4+0+1)
                buffer0[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(0>= tail_0_4+1 && 0 < tail_0_4+1+1)
                buffer0[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(0>= tail_0_4+2 && 0 < tail_0_4+2+1)
                buffer0[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(out_ack_0_4) begin
            if(0+512/512               < tail_0_4)
                buffer0[32*4+:32] <= `CQ buffer1[32*4+:32];
        end
    always@(posedge clk)
        if(in_ack_1_4 && out_ack_1_4) begin
            if(1+512/512               < tail_1_4)
                buffer1[32*4+:32] <= `CQ buffer2[32*4+:32];
            if(1+512/512               >= tail_1_4+0 && 1+512/512               < tail_1_4+0+1)
                buffer1[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(1+512/512               >= tail_1_4+1 && 1+512/512               < tail_1_4+1+1)
                buffer1[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(1+512/512               >= tail_1_4+2 && 1+512/512               < tail_1_4+2+1)
                buffer1[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(in_ack_1_4) begin
            if(1>= tail_1_4+0 && 1 < tail_1_4+0+1)
                buffer1[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(1>= tail_1_4+1 && 1 < tail_1_4+1+1)
                buffer1[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(1>= tail_1_4+2 && 1 < tail_1_4+2+1)
                buffer1[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(out_ack_1_4) begin
            if(1+512/512               < tail_1_4)
                buffer1[32*4+:32] <= `CQ buffer2[32*4+:32];
        end
    always@(posedge clk)
        if(in_ack_2_4 && out_ack_2_4) begin
            if(2+512/512               < tail_2_4)
                buffer2[32*4+:32] <= `CQ buffer3[32*4+:32];
            if(2+512/512               >= tail_2_4+0 && 2+512/512               < tail_2_4+0+1)
                buffer2[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(2+512/512               >= tail_2_4+1 && 2+512/512               < tail_2_4+1+1)
                buffer2[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(2+512/512               >= tail_2_4+2 && 2+512/512               < tail_2_4+2+1)
                buffer2[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(in_ack_2_4) begin
            if(2>= tail_2_4+0 && 2 < tail_2_4+0+1)
                buffer2[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(2>= tail_2_4+1 && 2 < tail_2_4+1+1)
                buffer2[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(2>= tail_2_4+2 && 2 < tail_2_4+2+1)
                buffer2[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(out_ack_2_4) begin
            if(2+512/512               < tail_2_4)
                buffer2[32*4+:32] <= `CQ buffer3[32*4+:32];
        end
    always@(posedge clk)
        if(in_ack_3_4 && out_ack_3_4) begin
            if(3+512/512               < tail_3_4)
                buffer3[32*4+:32] <= `CQ buffer4[32*4+:32];
            if(3+512/512               >= tail_3_4+0 && 3+512/512               < tail_3_4+0+1)
                buffer3[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(3+512/512               >= tail_3_4+1 && 3+512/512               < tail_3_4+1+1)
                buffer3[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(3+512/512               >= tail_3_4+2 && 3+512/512               < tail_3_4+2+1)
                buffer3[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(in_ack_3_4) begin
            if(3>= tail_3_4+0 && 3 < tail_3_4+0+1)
                buffer3[32*4+:32] <= `CQ in_data[512       *0+32*4+:32];
            if(3>= tail_3_4+1 && 3 < tail_3_4+1+1)
                buffer3[32*4+:32] <= `CQ in_data[512       *1+32*4+:32];
            if(3>= tail_3_4+2 && 3 < tail_3_4+2+1)
                buffer3[32*4+:32] <= `CQ in_data[512       *2+32*4+:32];
        end
        else if(out_ack_3_4) begin
            if(3+512/512               < tail_3_4)
                buffer3[32*4+:32] <= `CQ buffer4[32*4+:32];
        end
    always@(posedge clk)
        if(in_ack_0_5 && out_ack_0_5) begin
            if(0+512/512               < tail_0_5)
                buffer0[32*5+:32] <= `CQ buffer1[32*5+:32];
            if(0+512/512               >= tail_0_5+0 && 0+512/512               < tail_0_5+0+1)
                buffer0[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(0+512/512               >= tail_0_5+1 && 0+512/512               < tail_0_5+1+1)
                buffer0[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(0+512/512               >= tail_0_5+2 && 0+512/512               < tail_0_5+2+1)
                buffer0[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(in_ack_0_5) begin
            if(0>= tail_0_5+0 && 0 < tail_0_5+0+1)
                buffer0[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(0>= tail_0_5+1 && 0 < tail_0_5+1+1)
                buffer0[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(0>= tail_0_5+2 && 0 < tail_0_5+2+1)
                buffer0[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(out_ack_0_5) begin
            if(0+512/512               < tail_0_5)
                buffer0[32*5+:32] <= `CQ buffer1[32*5+:32];
        end
    always@(posedge clk)
        if(in_ack_1_5 && out_ack_1_5) begin
            if(1+512/512               < tail_1_5)
                buffer1[32*5+:32] <= `CQ buffer2[32*5+:32];
            if(1+512/512               >= tail_1_5+0 && 1+512/512               < tail_1_5+0+1)
                buffer1[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(1+512/512               >= tail_1_5+1 && 1+512/512               < tail_1_5+1+1)
                buffer1[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(1+512/512               >= tail_1_5+2 && 1+512/512               < tail_1_5+2+1)
                buffer1[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(in_ack_1_5) begin
            if(1>= tail_1_5+0 && 1 < tail_1_5+0+1)
                buffer1[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(1>= tail_1_5+1 && 1 < tail_1_5+1+1)
                buffer1[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(1>= tail_1_5+2 && 1 < tail_1_5+2+1)
                buffer1[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(out_ack_1_5) begin
            if(1+512/512               < tail_1_5)
                buffer1[32*5+:32] <= `CQ buffer2[32*5+:32];
        end
    always@(posedge clk)
        if(in_ack_2_5 && out_ack_2_5) begin
            if(2+512/512               < tail_2_5)
                buffer2[32*5+:32] <= `CQ buffer3[32*5+:32];
            if(2+512/512               >= tail_2_5+0 && 2+512/512               < tail_2_5+0+1)
                buffer2[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(2+512/512               >= tail_2_5+1 && 2+512/512               < tail_2_5+1+1)
                buffer2[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(2+512/512               >= tail_2_5+2 && 2+512/512               < tail_2_5+2+1)
                buffer2[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(in_ack_2_5) begin
            if(2>= tail_2_5+0 && 2 < tail_2_5+0+1)
                buffer2[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(2>= tail_2_5+1 && 2 < tail_2_5+1+1)
                buffer2[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(2>= tail_2_5+2 && 2 < tail_2_5+2+1)
                buffer2[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(out_ack_2_5) begin
            if(2+512/512               < tail_2_5)
                buffer2[32*5+:32] <= `CQ buffer3[32*5+:32];
        end
    always@(posedge clk)
        if(in_ack_3_5 && out_ack_3_5) begin
            if(3+512/512               < tail_3_5)
                buffer3[32*5+:32] <= `CQ buffer4[32*5+:32];
            if(3+512/512               >= tail_3_5+0 && 3+512/512               < tail_3_5+0+1)
                buffer3[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(3+512/512               >= tail_3_5+1 && 3+512/512               < tail_3_5+1+1)
                buffer3[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(3+512/512               >= tail_3_5+2 && 3+512/512               < tail_3_5+2+1)
                buffer3[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(in_ack_3_5) begin
            if(3>= tail_3_5+0 && 3 < tail_3_5+0+1)
                buffer3[32*5+:32] <= `CQ in_data[512       *0+32*5+:32];
            if(3>= tail_3_5+1 && 3 < tail_3_5+1+1)
                buffer3[32*5+:32] <= `CQ in_data[512       *1+32*5+:32];
            if(3>= tail_3_5+2 && 3 < tail_3_5+2+1)
                buffer3[32*5+:32] <= `CQ in_data[512       *2+32*5+:32];
        end
        else if(out_ack_3_5) begin
            if(3+512/512               < tail_3_5)
                buffer3[32*5+:32] <= `CQ buffer4[32*5+:32];
        end
    always@(posedge clk)
        if(in_ack_0_6 && out_ack_0_6) begin
            if(0+512/512               < tail_0_6)
                buffer0[32*6+:32] <= `CQ buffer1[32*6+:32];
            if(0+512/512               >= tail_0_6+0 && 0+512/512               < tail_0_6+0+1)
                buffer0[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(0+512/512               >= tail_0_6+1 && 0+512/512               < tail_0_6+1+1)
                buffer0[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(0+512/512               >= tail_0_6+2 && 0+512/512               < tail_0_6+2+1)
                buffer0[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(in_ack_0_6) begin
            if(0>= tail_0_6+0 && 0 < tail_0_6+0+1)
                buffer0[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(0>= tail_0_6+1 && 0 < tail_0_6+1+1)
                buffer0[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(0>= tail_0_6+2 && 0 < tail_0_6+2+1)
                buffer0[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(out_ack_0_6) begin
            if(0+512/512               < tail_0_6)
                buffer0[32*6+:32] <= `CQ buffer1[32*6+:32];
        end
    always@(posedge clk)
        if(in_ack_1_6 && out_ack_1_6) begin
            if(1+512/512               < tail_1_6)
                buffer1[32*6+:32] <= `CQ buffer2[32*6+:32];
            if(1+512/512               >= tail_1_6+0 && 1+512/512               < tail_1_6+0+1)
                buffer1[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(1+512/512               >= tail_1_6+1 && 1+512/512               < tail_1_6+1+1)
                buffer1[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(1+512/512               >= tail_1_6+2 && 1+512/512               < tail_1_6+2+1)
                buffer1[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(in_ack_1_6) begin
            if(1>= tail_1_6+0 && 1 < tail_1_6+0+1)
                buffer1[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(1>= tail_1_6+1 && 1 < tail_1_6+1+1)
                buffer1[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(1>= tail_1_6+2 && 1 < tail_1_6+2+1)
                buffer1[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(out_ack_1_6) begin
            if(1+512/512               < tail_1_6)
                buffer1[32*6+:32] <= `CQ buffer2[32*6+:32];
        end
    always@(posedge clk)
        if(in_ack_2_6 && out_ack_2_6) begin
            if(2+512/512               < tail_2_6)
                buffer2[32*6+:32] <= `CQ buffer3[32*6+:32];
            if(2+512/512               >= tail_2_6+0 && 2+512/512               < tail_2_6+0+1)
                buffer2[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(2+512/512               >= tail_2_6+1 && 2+512/512               < tail_2_6+1+1)
                buffer2[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(2+512/512               >= tail_2_6+2 && 2+512/512               < tail_2_6+2+1)
                buffer2[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(in_ack_2_6) begin
            if(2>= tail_2_6+0 && 2 < tail_2_6+0+1)
                buffer2[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(2>= tail_2_6+1 && 2 < tail_2_6+1+1)
                buffer2[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(2>= tail_2_6+2 && 2 < tail_2_6+2+1)
                buffer2[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(out_ack_2_6) begin
            if(2+512/512               < tail_2_6)
                buffer2[32*6+:32] <= `CQ buffer3[32*6+:32];
        end
    always@(posedge clk)
        if(in_ack_3_6 && out_ack_3_6) begin
            if(3+512/512               < tail_3_6)
                buffer3[32*6+:32] <= `CQ buffer4[32*6+:32];
            if(3+512/512               >= tail_3_6+0 && 3+512/512               < tail_3_6+0+1)
                buffer3[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(3+512/512               >= tail_3_6+1 && 3+512/512               < tail_3_6+1+1)
                buffer3[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(3+512/512               >= tail_3_6+2 && 3+512/512               < tail_3_6+2+1)
                buffer3[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(in_ack_3_6) begin
            if(3>= tail_3_6+0 && 3 < tail_3_6+0+1)
                buffer3[32*6+:32] <= `CQ in_data[512       *0+32*6+:32];
            if(3>= tail_3_6+1 && 3 < tail_3_6+1+1)
                buffer3[32*6+:32] <= `CQ in_data[512       *1+32*6+:32];
            if(3>= tail_3_6+2 && 3 < tail_3_6+2+1)
                buffer3[32*6+:32] <= `CQ in_data[512       *2+32*6+:32];
        end
        else if(out_ack_3_6) begin
            if(3+512/512               < tail_3_6)
                buffer3[32*6+:32] <= `CQ buffer4[32*6+:32];
        end
    always@(posedge clk)
        if(in_ack_0_7 && out_ack_0_7) begin
            if(0+512/512               < tail_0_7)
                buffer0[32*7+:32] <= `CQ buffer1[32*7+:32];
            if(0+512/512               >= tail_0_7+0 && 0+512/512               < tail_0_7+0+1)
                buffer0[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(0+512/512               >= tail_0_7+1 && 0+512/512               < tail_0_7+1+1)
                buffer0[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(0+512/512               >= tail_0_7+2 && 0+512/512               < tail_0_7+2+1)
                buffer0[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(in_ack_0_7) begin
            if(0>= tail_0_7+0 && 0 < tail_0_7+0+1)
                buffer0[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(0>= tail_0_7+1 && 0 < tail_0_7+1+1)
                buffer0[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(0>= tail_0_7+2 && 0 < tail_0_7+2+1)
                buffer0[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(out_ack_0_7) begin
            if(0+512/512               < tail_0_7)
                buffer0[32*7+:32] <= `CQ buffer1[32*7+:32];
        end
    always@(posedge clk)
        if(in_ack_1_7 && out_ack_1_7) begin
            if(1+512/512               < tail_1_7)
                buffer1[32*7+:32] <= `CQ buffer2[32*7+:32];
            if(1+512/512               >= tail_1_7+0 && 1+512/512               < tail_1_7+0+1)
                buffer1[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(1+512/512               >= tail_1_7+1 && 1+512/512               < tail_1_7+1+1)
                buffer1[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(1+512/512               >= tail_1_7+2 && 1+512/512               < tail_1_7+2+1)
                buffer1[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(in_ack_1_7) begin
            if(1>= tail_1_7+0 && 1 < tail_1_7+0+1)
                buffer1[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(1>= tail_1_7+1 && 1 < tail_1_7+1+1)
                buffer1[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(1>= tail_1_7+2 && 1 < tail_1_7+2+1)
                buffer1[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(out_ack_1_7) begin
            if(1+512/512               < tail_1_7)
                buffer1[32*7+:32] <= `CQ buffer2[32*7+:32];
        end
    always@(posedge clk)
        if(in_ack_2_7 && out_ack_2_7) begin
            if(2+512/512               < tail_2_7)
                buffer2[32*7+:32] <= `CQ buffer3[32*7+:32];
            if(2+512/512               >= tail_2_7+0 && 2+512/512               < tail_2_7+0+1)
                buffer2[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(2+512/512               >= tail_2_7+1 && 2+512/512               < tail_2_7+1+1)
                buffer2[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(2+512/512               >= tail_2_7+2 && 2+512/512               < tail_2_7+2+1)
                buffer2[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(in_ack_2_7) begin
            if(2>= tail_2_7+0 && 2 < tail_2_7+0+1)
                buffer2[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(2>= tail_2_7+1 && 2 < tail_2_7+1+1)
                buffer2[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(2>= tail_2_7+2 && 2 < tail_2_7+2+1)
                buffer2[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(out_ack_2_7) begin
            if(2+512/512               < tail_2_7)
                buffer2[32*7+:32] <= `CQ buffer3[32*7+:32];
        end
    always@(posedge clk)
        if(in_ack_3_7 && out_ack_3_7) begin
            if(3+512/512               < tail_3_7)
                buffer3[32*7+:32] <= `CQ buffer4[32*7+:32];
            if(3+512/512               >= tail_3_7+0 && 3+512/512               < tail_3_7+0+1)
                buffer3[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(3+512/512               >= tail_3_7+1 && 3+512/512               < tail_3_7+1+1)
                buffer3[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(3+512/512               >= tail_3_7+2 && 3+512/512               < tail_3_7+2+1)
                buffer3[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(in_ack_3_7) begin
            if(3>= tail_3_7+0 && 3 < tail_3_7+0+1)
                buffer3[32*7+:32] <= `CQ in_data[512       *0+32*7+:32];
            if(3>= tail_3_7+1 && 3 < tail_3_7+1+1)
                buffer3[32*7+:32] <= `CQ in_data[512       *1+32*7+:32];
            if(3>= tail_3_7+2 && 3 < tail_3_7+2+1)
                buffer3[32*7+:32] <= `CQ in_data[512       *2+32*7+:32];
        end
        else if(out_ack_3_7) begin
            if(3+512/512               < tail_3_7)
                buffer3[32*7+:32] <= `CQ buffer4[32*7+:32];
        end
    always@(posedge clk)
        if(in_ack_0_8 && out_ack_0_8) begin
            if(0+512/512               < tail_0_8)
                buffer0[32*8+:32] <= `CQ buffer1[32*8+:32];
            if(0+512/512               >= tail_0_8+0 && 0+512/512               < tail_0_8+0+1)
                buffer0[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(0+512/512               >= tail_0_8+1 && 0+512/512               < tail_0_8+1+1)
                buffer0[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(0+512/512               >= tail_0_8+2 && 0+512/512               < tail_0_8+2+1)
                buffer0[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(in_ack_0_8) begin
            if(0>= tail_0_8+0 && 0 < tail_0_8+0+1)
                buffer0[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(0>= tail_0_8+1 && 0 < tail_0_8+1+1)
                buffer0[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(0>= tail_0_8+2 && 0 < tail_0_8+2+1)
                buffer0[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(out_ack_0_8) begin
            if(0+512/512               < tail_0_8)
                buffer0[32*8+:32] <= `CQ buffer1[32*8+:32];
        end
    always@(posedge clk)
        if(in_ack_1_8 && out_ack_1_8) begin
            if(1+512/512               < tail_1_8)
                buffer1[32*8+:32] <= `CQ buffer2[32*8+:32];
            if(1+512/512               >= tail_1_8+0 && 1+512/512               < tail_1_8+0+1)
                buffer1[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(1+512/512               >= tail_1_8+1 && 1+512/512               < tail_1_8+1+1)
                buffer1[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(1+512/512               >= tail_1_8+2 && 1+512/512               < tail_1_8+2+1)
                buffer1[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(in_ack_1_8) begin
            if(1>= tail_1_8+0 && 1 < tail_1_8+0+1)
                buffer1[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(1>= tail_1_8+1 && 1 < tail_1_8+1+1)
                buffer1[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(1>= tail_1_8+2 && 1 < tail_1_8+2+1)
                buffer1[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(out_ack_1_8) begin
            if(1+512/512               < tail_1_8)
                buffer1[32*8+:32] <= `CQ buffer2[32*8+:32];
        end
    always@(posedge clk)
        if(in_ack_2_8 && out_ack_2_8) begin
            if(2+512/512               < tail_2_8)
                buffer2[32*8+:32] <= `CQ buffer3[32*8+:32];
            if(2+512/512               >= tail_2_8+0 && 2+512/512               < tail_2_8+0+1)
                buffer2[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(2+512/512               >= tail_2_8+1 && 2+512/512               < tail_2_8+1+1)
                buffer2[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(2+512/512               >= tail_2_8+2 && 2+512/512               < tail_2_8+2+1)
                buffer2[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(in_ack_2_8) begin
            if(2>= tail_2_8+0 && 2 < tail_2_8+0+1)
                buffer2[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(2>= tail_2_8+1 && 2 < tail_2_8+1+1)
                buffer2[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(2>= tail_2_8+2 && 2 < tail_2_8+2+1)
                buffer2[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(out_ack_2_8) begin
            if(2+512/512               < tail_2_8)
                buffer2[32*8+:32] <= `CQ buffer3[32*8+:32];
        end
    always@(posedge clk)
        if(in_ack_3_8 && out_ack_3_8) begin
            if(3+512/512               < tail_3_8)
                buffer3[32*8+:32] <= `CQ buffer4[32*8+:32];
            if(3+512/512               >= tail_3_8+0 && 3+512/512               < tail_3_8+0+1)
                buffer3[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(3+512/512               >= tail_3_8+1 && 3+512/512               < tail_3_8+1+1)
                buffer3[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(3+512/512               >= tail_3_8+2 && 3+512/512               < tail_3_8+2+1)
                buffer3[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(in_ack_3_8) begin
            if(3>= tail_3_8+0 && 3 < tail_3_8+0+1)
                buffer3[32*8+:32] <= `CQ in_data[512       *0+32*8+:32];
            if(3>= tail_3_8+1 && 3 < tail_3_8+1+1)
                buffer3[32*8+:32] <= `CQ in_data[512       *1+32*8+:32];
            if(3>= tail_3_8+2 && 3 < tail_3_8+2+1)
                buffer3[32*8+:32] <= `CQ in_data[512       *2+32*8+:32];
        end
        else if(out_ack_3_8) begin
            if(3+512/512               < tail_3_8)
                buffer3[32*8+:32] <= `CQ buffer4[32*8+:32];
        end
    always@(posedge clk)
        if(in_ack_0_9 && out_ack_0_9) begin
            if(0+512/512               < tail_0_9)
                buffer0[32*9+:32] <= `CQ buffer1[32*9+:32];
            if(0+512/512               >= tail_0_9+0 && 0+512/512               < tail_0_9+0+1)
                buffer0[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(0+512/512               >= tail_0_9+1 && 0+512/512               < tail_0_9+1+1)
                buffer0[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(0+512/512               >= tail_0_9+2 && 0+512/512               < tail_0_9+2+1)
                buffer0[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(in_ack_0_9) begin
            if(0>= tail_0_9+0 && 0 < tail_0_9+0+1)
                buffer0[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(0>= tail_0_9+1 && 0 < tail_0_9+1+1)
                buffer0[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(0>= tail_0_9+2 && 0 < tail_0_9+2+1)
                buffer0[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(out_ack_0_9) begin
            if(0+512/512               < tail_0_9)
                buffer0[32*9+:32] <= `CQ buffer1[32*9+:32];
        end
    always@(posedge clk)
        if(in_ack_1_9 && out_ack_1_9) begin
            if(1+512/512               < tail_1_9)
                buffer1[32*9+:32] <= `CQ buffer2[32*9+:32];
            if(1+512/512               >= tail_1_9+0 && 1+512/512               < tail_1_9+0+1)
                buffer1[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(1+512/512               >= tail_1_9+1 && 1+512/512               < tail_1_9+1+1)
                buffer1[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(1+512/512               >= tail_1_9+2 && 1+512/512               < tail_1_9+2+1)
                buffer1[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(in_ack_1_9) begin
            if(1>= tail_1_9+0 && 1 < tail_1_9+0+1)
                buffer1[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(1>= tail_1_9+1 && 1 < tail_1_9+1+1)
                buffer1[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(1>= tail_1_9+2 && 1 < tail_1_9+2+1)
                buffer1[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(out_ack_1_9) begin
            if(1+512/512               < tail_1_9)
                buffer1[32*9+:32] <= `CQ buffer2[32*9+:32];
        end
    always@(posedge clk)
        if(in_ack_2_9 && out_ack_2_9) begin
            if(2+512/512               < tail_2_9)
                buffer2[32*9+:32] <= `CQ buffer3[32*9+:32];
            if(2+512/512               >= tail_2_9+0 && 2+512/512               < tail_2_9+0+1)
                buffer2[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(2+512/512               >= tail_2_9+1 && 2+512/512               < tail_2_9+1+1)
                buffer2[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(2+512/512               >= tail_2_9+2 && 2+512/512               < tail_2_9+2+1)
                buffer2[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(in_ack_2_9) begin
            if(2>= tail_2_9+0 && 2 < tail_2_9+0+1)
                buffer2[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(2>= tail_2_9+1 && 2 < tail_2_9+1+1)
                buffer2[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(2>= tail_2_9+2 && 2 < tail_2_9+2+1)
                buffer2[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(out_ack_2_9) begin
            if(2+512/512               < tail_2_9)
                buffer2[32*9+:32] <= `CQ buffer3[32*9+:32];
        end
    always@(posedge clk)
        if(in_ack_3_9 && out_ack_3_9) begin
            if(3+512/512               < tail_3_9)
                buffer3[32*9+:32] <= `CQ buffer4[32*9+:32];
            if(3+512/512               >= tail_3_9+0 && 3+512/512               < tail_3_9+0+1)
                buffer3[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(3+512/512               >= tail_3_9+1 && 3+512/512               < tail_3_9+1+1)
                buffer3[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(3+512/512               >= tail_3_9+2 && 3+512/512               < tail_3_9+2+1)
                buffer3[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(in_ack_3_9) begin
            if(3>= tail_3_9+0 && 3 < tail_3_9+0+1)
                buffer3[32*9+:32] <= `CQ in_data[512       *0+32*9+:32];
            if(3>= tail_3_9+1 && 3 < tail_3_9+1+1)
                buffer3[32*9+:32] <= `CQ in_data[512       *1+32*9+:32];
            if(3>= tail_3_9+2 && 3 < tail_3_9+2+1)
                buffer3[32*9+:32] <= `CQ in_data[512       *2+32*9+:32];
        end
        else if(out_ack_3_9) begin
            if(3+512/512               < tail_3_9)
                buffer3[32*9+:32] <= `CQ buffer4[32*9+:32];
        end
    always@(posedge clk)
        if(in_ack_0_10 && out_ack_0_10) begin
            if(0+512/512               < tail_0_10)
                buffer0[32*10+:32] <= `CQ buffer1[32*10+:32];
            if(0+512/512               >= tail_0_10+0 && 0+512/512               < tail_0_10+0+1)
                buffer0[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(0+512/512               >= tail_0_10+1 && 0+512/512               < tail_0_10+1+1)
                buffer0[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(0+512/512               >= tail_0_10+2 && 0+512/512               < tail_0_10+2+1)
                buffer0[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(in_ack_0_10) begin
            if(0>= tail_0_10+0 && 0 < tail_0_10+0+1)
                buffer0[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(0>= tail_0_10+1 && 0 < tail_0_10+1+1)
                buffer0[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(0>= tail_0_10+2 && 0 < tail_0_10+2+1)
                buffer0[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(out_ack_0_10) begin
            if(0+512/512               < tail_0_10)
                buffer0[32*10+:32] <= `CQ buffer1[32*10+:32];
        end
    always@(posedge clk)
        if(in_ack_1_10 && out_ack_1_10) begin
            if(1+512/512               < tail_1_10)
                buffer1[32*10+:32] <= `CQ buffer2[32*10+:32];
            if(1+512/512               >= tail_1_10+0 && 1+512/512               < tail_1_10+0+1)
                buffer1[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(1+512/512               >= tail_1_10+1 && 1+512/512               < tail_1_10+1+1)
                buffer1[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(1+512/512               >= tail_1_10+2 && 1+512/512               < tail_1_10+2+1)
                buffer1[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(in_ack_1_10) begin
            if(1>= tail_1_10+0 && 1 < tail_1_10+0+1)
                buffer1[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(1>= tail_1_10+1 && 1 < tail_1_10+1+1)
                buffer1[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(1>= tail_1_10+2 && 1 < tail_1_10+2+1)
                buffer1[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(out_ack_1_10) begin
            if(1+512/512               < tail_1_10)
                buffer1[32*10+:32] <= `CQ buffer2[32*10+:32];
        end
    always@(posedge clk)
        if(in_ack_2_10 && out_ack_2_10) begin
            if(2+512/512               < tail_2_10)
                buffer2[32*10+:32] <= `CQ buffer3[32*10+:32];
            if(2+512/512               >= tail_2_10+0 && 2+512/512               < tail_2_10+0+1)
                buffer2[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(2+512/512               >= tail_2_10+1 && 2+512/512               < tail_2_10+1+1)
                buffer2[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(2+512/512               >= tail_2_10+2 && 2+512/512               < tail_2_10+2+1)
                buffer2[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(in_ack_2_10) begin
            if(2>= tail_2_10+0 && 2 < tail_2_10+0+1)
                buffer2[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(2>= tail_2_10+1 && 2 < tail_2_10+1+1)
                buffer2[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(2>= tail_2_10+2 && 2 < tail_2_10+2+1)
                buffer2[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(out_ack_2_10) begin
            if(2+512/512               < tail_2_10)
                buffer2[32*10+:32] <= `CQ buffer3[32*10+:32];
        end
    always@(posedge clk)
        if(in_ack_3_10 && out_ack_3_10) begin
            if(3+512/512               < tail_3_10)
                buffer3[32*10+:32] <= `CQ buffer4[32*10+:32];
            if(3+512/512               >= tail_3_10+0 && 3+512/512               < tail_3_10+0+1)
                buffer3[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(3+512/512               >= tail_3_10+1 && 3+512/512               < tail_3_10+1+1)
                buffer3[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(3+512/512               >= tail_3_10+2 && 3+512/512               < tail_3_10+2+1)
                buffer3[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(in_ack_3_10) begin
            if(3>= tail_3_10+0 && 3 < tail_3_10+0+1)
                buffer3[32*10+:32] <= `CQ in_data[512       *0+32*10+:32];
            if(3>= tail_3_10+1 && 3 < tail_3_10+1+1)
                buffer3[32*10+:32] <= `CQ in_data[512       *1+32*10+:32];
            if(3>= tail_3_10+2 && 3 < tail_3_10+2+1)
                buffer3[32*10+:32] <= `CQ in_data[512       *2+32*10+:32];
        end
        else if(out_ack_3_10) begin
            if(3+512/512               < tail_3_10)
                buffer3[32*10+:32] <= `CQ buffer4[32*10+:32];
        end
    always@(posedge clk)
        if(in_ack_0_11 && out_ack_0_11) begin
            if(0+512/512               < tail_0_11)
                buffer0[32*11+:32] <= `CQ buffer1[32*11+:32];
            if(0+512/512               >= tail_0_11+0 && 0+512/512               < tail_0_11+0+1)
                buffer0[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(0+512/512               >= tail_0_11+1 && 0+512/512               < tail_0_11+1+1)
                buffer0[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(0+512/512               >= tail_0_11+2 && 0+512/512               < tail_0_11+2+1)
                buffer0[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(in_ack_0_11) begin
            if(0>= tail_0_11+0 && 0 < tail_0_11+0+1)
                buffer0[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(0>= tail_0_11+1 && 0 < tail_0_11+1+1)
                buffer0[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(0>= tail_0_11+2 && 0 < tail_0_11+2+1)
                buffer0[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(out_ack_0_11) begin
            if(0+512/512               < tail_0_11)
                buffer0[32*11+:32] <= `CQ buffer1[32*11+:32];
        end
    always@(posedge clk)
        if(in_ack_1_11 && out_ack_1_11) begin
            if(1+512/512               < tail_1_11)
                buffer1[32*11+:32] <= `CQ buffer2[32*11+:32];
            if(1+512/512               >= tail_1_11+0 && 1+512/512               < tail_1_11+0+1)
                buffer1[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(1+512/512               >= tail_1_11+1 && 1+512/512               < tail_1_11+1+1)
                buffer1[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(1+512/512               >= tail_1_11+2 && 1+512/512               < tail_1_11+2+1)
                buffer1[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(in_ack_1_11) begin
            if(1>= tail_1_11+0 && 1 < tail_1_11+0+1)
                buffer1[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(1>= tail_1_11+1 && 1 < tail_1_11+1+1)
                buffer1[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(1>= tail_1_11+2 && 1 < tail_1_11+2+1)
                buffer1[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(out_ack_1_11) begin
            if(1+512/512               < tail_1_11)
                buffer1[32*11+:32] <= `CQ buffer2[32*11+:32];
        end
    always@(posedge clk)
        if(in_ack_2_11 && out_ack_2_11) begin
            if(2+512/512               < tail_2_11)
                buffer2[32*11+:32] <= `CQ buffer3[32*11+:32];
            if(2+512/512               >= tail_2_11+0 && 2+512/512               < tail_2_11+0+1)
                buffer2[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(2+512/512               >= tail_2_11+1 && 2+512/512               < tail_2_11+1+1)
                buffer2[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(2+512/512               >= tail_2_11+2 && 2+512/512               < tail_2_11+2+1)
                buffer2[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(in_ack_2_11) begin
            if(2>= tail_2_11+0 && 2 < tail_2_11+0+1)
                buffer2[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(2>= tail_2_11+1 && 2 < tail_2_11+1+1)
                buffer2[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(2>= tail_2_11+2 && 2 < tail_2_11+2+1)
                buffer2[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(out_ack_2_11) begin
            if(2+512/512               < tail_2_11)
                buffer2[32*11+:32] <= `CQ buffer3[32*11+:32];
        end
    always@(posedge clk)
        if(in_ack_3_11 && out_ack_3_11) begin
            if(3+512/512               < tail_3_11)
                buffer3[32*11+:32] <= `CQ buffer4[32*11+:32];
            if(3+512/512               >= tail_3_11+0 && 3+512/512               < tail_3_11+0+1)
                buffer3[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(3+512/512               >= tail_3_11+1 && 3+512/512               < tail_3_11+1+1)
                buffer3[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(3+512/512               >= tail_3_11+2 && 3+512/512               < tail_3_11+2+1)
                buffer3[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(in_ack_3_11) begin
            if(3>= tail_3_11+0 && 3 < tail_3_11+0+1)
                buffer3[32*11+:32] <= `CQ in_data[512       *0+32*11+:32];
            if(3>= tail_3_11+1 && 3 < tail_3_11+1+1)
                buffer3[32*11+:32] <= `CQ in_data[512       *1+32*11+:32];
            if(3>= tail_3_11+2 && 3 < tail_3_11+2+1)
                buffer3[32*11+:32] <= `CQ in_data[512       *2+32*11+:32];
        end
        else if(out_ack_3_11) begin
            if(3+512/512               < tail_3_11)
                buffer3[32*11+:32] <= `CQ buffer4[32*11+:32];
        end
    always@(posedge clk)
        if(in_ack_0_12 && out_ack_0_12) begin
            if(0+512/512               < tail_0_12)
                buffer0[32*12+:32] <= `CQ buffer1[32*12+:32];
            if(0+512/512               >= tail_0_12+0 && 0+512/512               < tail_0_12+0+1)
                buffer0[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(0+512/512               >= tail_0_12+1 && 0+512/512               < tail_0_12+1+1)
                buffer0[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(0+512/512               >= tail_0_12+2 && 0+512/512               < tail_0_12+2+1)
                buffer0[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(in_ack_0_12) begin
            if(0>= tail_0_12+0 && 0 < tail_0_12+0+1)
                buffer0[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(0>= tail_0_12+1 && 0 < tail_0_12+1+1)
                buffer0[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(0>= tail_0_12+2 && 0 < tail_0_12+2+1)
                buffer0[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(out_ack_0_12) begin
            if(0+512/512               < tail_0_12)
                buffer0[32*12+:32] <= `CQ buffer1[32*12+:32];
        end
    always@(posedge clk)
        if(in_ack_1_12 && out_ack_1_12) begin
            if(1+512/512               < tail_1_12)
                buffer1[32*12+:32] <= `CQ buffer2[32*12+:32];
            if(1+512/512               >= tail_1_12+0 && 1+512/512               < tail_1_12+0+1)
                buffer1[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(1+512/512               >= tail_1_12+1 && 1+512/512               < tail_1_12+1+1)
                buffer1[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(1+512/512               >= tail_1_12+2 && 1+512/512               < tail_1_12+2+1)
                buffer1[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(in_ack_1_12) begin
            if(1>= tail_1_12+0 && 1 < tail_1_12+0+1)
                buffer1[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(1>= tail_1_12+1 && 1 < tail_1_12+1+1)
                buffer1[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(1>= tail_1_12+2 && 1 < tail_1_12+2+1)
                buffer1[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(out_ack_1_12) begin
            if(1+512/512               < tail_1_12)
                buffer1[32*12+:32] <= `CQ buffer2[32*12+:32];
        end
    always@(posedge clk)
        if(in_ack_2_12 && out_ack_2_12) begin
            if(2+512/512               < tail_2_12)
                buffer2[32*12+:32] <= `CQ buffer3[32*12+:32];
            if(2+512/512               >= tail_2_12+0 && 2+512/512               < tail_2_12+0+1)
                buffer2[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(2+512/512               >= tail_2_12+1 && 2+512/512               < tail_2_12+1+1)
                buffer2[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(2+512/512               >= tail_2_12+2 && 2+512/512               < tail_2_12+2+1)
                buffer2[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(in_ack_2_12) begin
            if(2>= tail_2_12+0 && 2 < tail_2_12+0+1)
                buffer2[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(2>= tail_2_12+1 && 2 < tail_2_12+1+1)
                buffer2[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(2>= tail_2_12+2 && 2 < tail_2_12+2+1)
                buffer2[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(out_ack_2_12) begin
            if(2+512/512               < tail_2_12)
                buffer2[32*12+:32] <= `CQ buffer3[32*12+:32];
        end
    always@(posedge clk)
        if(in_ack_3_12 && out_ack_3_12) begin
            if(3+512/512               < tail_3_12)
                buffer3[32*12+:32] <= `CQ buffer4[32*12+:32];
            if(3+512/512               >= tail_3_12+0 && 3+512/512               < tail_3_12+0+1)
                buffer3[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(3+512/512               >= tail_3_12+1 && 3+512/512               < tail_3_12+1+1)
                buffer3[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(3+512/512               >= tail_3_12+2 && 3+512/512               < tail_3_12+2+1)
                buffer3[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(in_ack_3_12) begin
            if(3>= tail_3_12+0 && 3 < tail_3_12+0+1)
                buffer3[32*12+:32] <= `CQ in_data[512       *0+32*12+:32];
            if(3>= tail_3_12+1 && 3 < tail_3_12+1+1)
                buffer3[32*12+:32] <= `CQ in_data[512       *1+32*12+:32];
            if(3>= tail_3_12+2 && 3 < tail_3_12+2+1)
                buffer3[32*12+:32] <= `CQ in_data[512       *2+32*12+:32];
        end
        else if(out_ack_3_12) begin
            if(3+512/512               < tail_3_12)
                buffer3[32*12+:32] <= `CQ buffer4[32*12+:32];
        end
    always@(posedge clk)
        if(in_ack_0_13 && out_ack_0_13) begin
            if(0+512/512               < tail_0_13)
                buffer0[32*13+:32] <= `CQ buffer1[32*13+:32];
            if(0+512/512               >= tail_0_13+0 && 0+512/512               < tail_0_13+0+1)
                buffer0[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(0+512/512               >= tail_0_13+1 && 0+512/512               < tail_0_13+1+1)
                buffer0[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(0+512/512               >= tail_0_13+2 && 0+512/512               < tail_0_13+2+1)
                buffer0[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(in_ack_0_13) begin
            if(0>= tail_0_13+0 && 0 < tail_0_13+0+1)
                buffer0[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(0>= tail_0_13+1 && 0 < tail_0_13+1+1)
                buffer0[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(0>= tail_0_13+2 && 0 < tail_0_13+2+1)
                buffer0[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(out_ack_0_13) begin
            if(0+512/512               < tail_0_13)
                buffer0[32*13+:32] <= `CQ buffer1[32*13+:32];
        end
    always@(posedge clk)
        if(in_ack_1_13 && out_ack_1_13) begin
            if(1+512/512               < tail_1_13)
                buffer1[32*13+:32] <= `CQ buffer2[32*13+:32];
            if(1+512/512               >= tail_1_13+0 && 1+512/512               < tail_1_13+0+1)
                buffer1[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(1+512/512               >= tail_1_13+1 && 1+512/512               < tail_1_13+1+1)
                buffer1[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(1+512/512               >= tail_1_13+2 && 1+512/512               < tail_1_13+2+1)
                buffer1[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(in_ack_1_13) begin
            if(1>= tail_1_13+0 && 1 < tail_1_13+0+1)
                buffer1[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(1>= tail_1_13+1 && 1 < tail_1_13+1+1)
                buffer1[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(1>= tail_1_13+2 && 1 < tail_1_13+2+1)
                buffer1[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(out_ack_1_13) begin
            if(1+512/512               < tail_1_13)
                buffer1[32*13+:32] <= `CQ buffer2[32*13+:32];
        end
    always@(posedge clk)
        if(in_ack_2_13 && out_ack_2_13) begin
            if(2+512/512               < tail_2_13)
                buffer2[32*13+:32] <= `CQ buffer3[32*13+:32];
            if(2+512/512               >= tail_2_13+0 && 2+512/512               < tail_2_13+0+1)
                buffer2[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(2+512/512               >= tail_2_13+1 && 2+512/512               < tail_2_13+1+1)
                buffer2[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(2+512/512               >= tail_2_13+2 && 2+512/512               < tail_2_13+2+1)
                buffer2[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(in_ack_2_13) begin
            if(2>= tail_2_13+0 && 2 < tail_2_13+0+1)
                buffer2[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(2>= tail_2_13+1 && 2 < tail_2_13+1+1)
                buffer2[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(2>= tail_2_13+2 && 2 < tail_2_13+2+1)
                buffer2[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(out_ack_2_13) begin
            if(2+512/512               < tail_2_13)
                buffer2[32*13+:32] <= `CQ buffer3[32*13+:32];
        end
    always@(posedge clk)
        if(in_ack_3_13 && out_ack_3_13) begin
            if(3+512/512               < tail_3_13)
                buffer3[32*13+:32] <= `CQ buffer4[32*13+:32];
            if(3+512/512               >= tail_3_13+0 && 3+512/512               < tail_3_13+0+1)
                buffer3[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(3+512/512               >= tail_3_13+1 && 3+512/512               < tail_3_13+1+1)
                buffer3[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(3+512/512               >= tail_3_13+2 && 3+512/512               < tail_3_13+2+1)
                buffer3[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(in_ack_3_13) begin
            if(3>= tail_3_13+0 && 3 < tail_3_13+0+1)
                buffer3[32*13+:32] <= `CQ in_data[512       *0+32*13+:32];
            if(3>= tail_3_13+1 && 3 < tail_3_13+1+1)
                buffer3[32*13+:32] <= `CQ in_data[512       *1+32*13+:32];
            if(3>= tail_3_13+2 && 3 < tail_3_13+2+1)
                buffer3[32*13+:32] <= `CQ in_data[512       *2+32*13+:32];
        end
        else if(out_ack_3_13) begin
            if(3+512/512               < tail_3_13)
                buffer3[32*13+:32] <= `CQ buffer4[32*13+:32];
        end
    always@(posedge clk)
        if(in_ack_0_14 && out_ack_0_14) begin
            if(0+512/512               < tail_0_14)
                buffer0[32*14+:32] <= `CQ buffer1[32*14+:32];
            if(0+512/512               >= tail_0_14+0 && 0+512/512               < tail_0_14+0+1)
                buffer0[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(0+512/512               >= tail_0_14+1 && 0+512/512               < tail_0_14+1+1)
                buffer0[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(0+512/512               >= tail_0_14+2 && 0+512/512               < tail_0_14+2+1)
                buffer0[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(in_ack_0_14) begin
            if(0>= tail_0_14+0 && 0 < tail_0_14+0+1)
                buffer0[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(0>= tail_0_14+1 && 0 < tail_0_14+1+1)
                buffer0[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(0>= tail_0_14+2 && 0 < tail_0_14+2+1)
                buffer0[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(out_ack_0_14) begin
            if(0+512/512               < tail_0_14)
                buffer0[32*14+:32] <= `CQ buffer1[32*14+:32];
        end
    always@(posedge clk)
        if(in_ack_1_14 && out_ack_1_14) begin
            if(1+512/512               < tail_1_14)
                buffer1[32*14+:32] <= `CQ buffer2[32*14+:32];
            if(1+512/512               >= tail_1_14+0 && 1+512/512               < tail_1_14+0+1)
                buffer1[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(1+512/512               >= tail_1_14+1 && 1+512/512               < tail_1_14+1+1)
                buffer1[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(1+512/512               >= tail_1_14+2 && 1+512/512               < tail_1_14+2+1)
                buffer1[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(in_ack_1_14) begin
            if(1>= tail_1_14+0 && 1 < tail_1_14+0+1)
                buffer1[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(1>= tail_1_14+1 && 1 < tail_1_14+1+1)
                buffer1[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(1>= tail_1_14+2 && 1 < tail_1_14+2+1)
                buffer1[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(out_ack_1_14) begin
            if(1+512/512               < tail_1_14)
                buffer1[32*14+:32] <= `CQ buffer2[32*14+:32];
        end
    always@(posedge clk)
        if(in_ack_2_14 && out_ack_2_14) begin
            if(2+512/512               < tail_2_14)
                buffer2[32*14+:32] <= `CQ buffer3[32*14+:32];
            if(2+512/512               >= tail_2_14+0 && 2+512/512               < tail_2_14+0+1)
                buffer2[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(2+512/512               >= tail_2_14+1 && 2+512/512               < tail_2_14+1+1)
                buffer2[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(2+512/512               >= tail_2_14+2 && 2+512/512               < tail_2_14+2+1)
                buffer2[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(in_ack_2_14) begin
            if(2>= tail_2_14+0 && 2 < tail_2_14+0+1)
                buffer2[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(2>= tail_2_14+1 && 2 < tail_2_14+1+1)
                buffer2[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(2>= tail_2_14+2 && 2 < tail_2_14+2+1)
                buffer2[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(out_ack_2_14) begin
            if(2+512/512               < tail_2_14)
                buffer2[32*14+:32] <= `CQ buffer3[32*14+:32];
        end
    always@(posedge clk)
        if(in_ack_3_14 && out_ack_3_14) begin
            if(3+512/512               < tail_3_14)
                buffer3[32*14+:32] <= `CQ buffer4[32*14+:32];
            if(3+512/512               >= tail_3_14+0 && 3+512/512               < tail_3_14+0+1)
                buffer3[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(3+512/512               >= tail_3_14+1 && 3+512/512               < tail_3_14+1+1)
                buffer3[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(3+512/512               >= tail_3_14+2 && 3+512/512               < tail_3_14+2+1)
                buffer3[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(in_ack_3_14) begin
            if(3>= tail_3_14+0 && 3 < tail_3_14+0+1)
                buffer3[32*14+:32] <= `CQ in_data[512       *0+32*14+:32];
            if(3>= tail_3_14+1 && 3 < tail_3_14+1+1)
                buffer3[32*14+:32] <= `CQ in_data[512       *1+32*14+:32];
            if(3>= tail_3_14+2 && 3 < tail_3_14+2+1)
                buffer3[32*14+:32] <= `CQ in_data[512       *2+32*14+:32];
        end
        else if(out_ack_3_14) begin
            if(3+512/512               < tail_3_14)
                buffer3[32*14+:32] <= `CQ buffer4[32*14+:32];
        end
    always@(posedge clk)
        if(in_ack_0_15 && out_ack_0_15) begin
            if(0+512/512               < tail_0_15)
                buffer0[32*15+:32] <= `CQ buffer1[32*15+:32];
            if(0+512/512               >= tail_0_15+0 && 0+512/512               < tail_0_15+0+1)
                buffer0[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(0+512/512               >= tail_0_15+1 && 0+512/512               < tail_0_15+1+1)
                buffer0[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(0+512/512               >= tail_0_15+2 && 0+512/512               < tail_0_15+2+1)
                buffer0[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(in_ack_0_15) begin
            if(0>= tail_0_15+0 && 0 < tail_0_15+0+1)
                buffer0[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(0>= tail_0_15+1 && 0 < tail_0_15+1+1)
                buffer0[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(0>= tail_0_15+2 && 0 < tail_0_15+2+1)
                buffer0[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(out_ack_0_15) begin
            if(0+512/512               < tail_0_15)
                buffer0[32*15+:32] <= `CQ buffer1[32*15+:32];
        end
    always@(posedge clk)
        if(in_ack_1_15 && out_ack_1_15) begin
            if(1+512/512               < tail_1_15)
                buffer1[32*15+:32] <= `CQ buffer2[32*15+:32];
            if(1+512/512               >= tail_1_15+0 && 1+512/512               < tail_1_15+0+1)
                buffer1[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(1+512/512               >= tail_1_15+1 && 1+512/512               < tail_1_15+1+1)
                buffer1[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(1+512/512               >= tail_1_15+2 && 1+512/512               < tail_1_15+2+1)
                buffer1[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(in_ack_1_15) begin
            if(1>= tail_1_15+0 && 1 < tail_1_15+0+1)
                buffer1[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(1>= tail_1_15+1 && 1 < tail_1_15+1+1)
                buffer1[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(1>= tail_1_15+2 && 1 < tail_1_15+2+1)
                buffer1[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(out_ack_1_15) begin
            if(1+512/512               < tail_1_15)
                buffer1[32*15+:32] <= `CQ buffer2[32*15+:32];
        end
    always@(posedge clk)
        if(in_ack_2_15 && out_ack_2_15) begin
            if(2+512/512               < tail_2_15)
                buffer2[32*15+:32] <= `CQ buffer3[32*15+:32];
            if(2+512/512               >= tail_2_15+0 && 2+512/512               < tail_2_15+0+1)
                buffer2[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(2+512/512               >= tail_2_15+1 && 2+512/512               < tail_2_15+1+1)
                buffer2[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(2+512/512               >= tail_2_15+2 && 2+512/512               < tail_2_15+2+1)
                buffer2[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(in_ack_2_15) begin
            if(2>= tail_2_15+0 && 2 < tail_2_15+0+1)
                buffer2[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(2>= tail_2_15+1 && 2 < tail_2_15+1+1)
                buffer2[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(2>= tail_2_15+2 && 2 < tail_2_15+2+1)
                buffer2[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(out_ack_2_15) begin
            if(2+512/512               < tail_2_15)
                buffer2[32*15+:32] <= `CQ buffer3[32*15+:32];
        end
    always@(posedge clk)
        if(in_ack_3_15 && out_ack_3_15) begin
            if(3+512/512               < tail_3_15)
                buffer3[32*15+:32] <= `CQ buffer4[32*15+:32];
            if(3+512/512               >= tail_3_15+0 && 3+512/512               < tail_3_15+0+1)
                buffer3[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(3+512/512               >= tail_3_15+1 && 3+512/512               < tail_3_15+1+1)
                buffer3[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(3+512/512               >= tail_3_15+2 && 3+512/512               < tail_3_15+2+1)
                buffer3[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(in_ack_3_15) begin
            if(3>= tail_3_15+0 && 3 < tail_3_15+0+1)
                buffer3[32*15+:32] <= `CQ in_data[512       *0+32*15+:32];
            if(3>= tail_3_15+1 && 3 < tail_3_15+1+1)
                buffer3[32*15+:32] <= `CQ in_data[512       *1+32*15+:32];
            if(3>= tail_3_15+2 && 3 < tail_3_15+2+1)
                buffer3[32*15+:32] <= `CQ in_data[512       *2+32*15+:32];
        end
        else if(out_ack_3_15) begin
            if(3+512/512               < tail_3_15)
                buffer3[32*15+:32] <= `CQ buffer4[32*15+:32];
        end

    assign out_data[512       *0+:512       ] = buffer0;        

endmodule

