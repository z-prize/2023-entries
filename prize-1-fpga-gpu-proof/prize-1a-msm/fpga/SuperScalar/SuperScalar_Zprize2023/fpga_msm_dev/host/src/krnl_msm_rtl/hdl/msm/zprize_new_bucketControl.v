`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

///////////////////////////////////////////////////////////////////////////////////////////////////////
//F1_t  --  input 
//I     --  check
//J     --  into waitBuffer or into L
//L     --  to F3
//
//waitBufferOut -- sync with J
//////////////////////////////////////////////////////////////////////////////////////////////////////

module zprize_new_bucketControl import zprize_param::*;(
//input        
    input                                   valid_F1_t,
    output                                  ready_F1_t,     
    input [`LOG2((1<<24))-1:0]              N_count_F1_t,                        //0~16383
    input [`LOG2((1<<(16-1)))-1:0]    bucket_F1_t,
    input                                   signed_flag_F1_t,
    input [`LOG2(512)-1:0]      affineIndex_F1_t,
    //input [(1<<(16-1))-1:0]               next_accValid,   //accValid's next cycle value
//output
    `MAX_FANOUT32 output reg                valid_L,
    output reg   [`LOG2((1<<(16-1)))-1:0] bucket_L,
    output reg[`LOG2((1<<24))-1:0]          N_count_L,                        //0~16383
    output reg[`LOG2(512)-1:0]  affineIndex_L,
    output reg                              signed_flag_L,
    output                                  bucketControl_idle,     
//F3 output
    input                                       valid_S,
    input     [`LOG2((1<<(16-1)))-1:0] bucket_S,
    output  [`LOG2((1<<(16-1)))-1:0]  real_bucket_S,
//Global
    input   [4:0]                           state_C,
    input   [`LOG2((1<<(16-1)))-1:0]      init_count,                 //init bucketRAM to 0
    input           clk,
    input           rstN
);
`ifdef FAKE_F2_F3
localparam bucket_hp_length = 1+2+6+1+1;
`else
localparam bucket_hp_length = 1+2+6+1+39*2+((7+1+1)-1+(7+1+1)+1+1)*2+4;
`endif
/*
F1_t: input 
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////
    genvar                      i,j;

//inFifo
    

    logic                                   valid_I;
    logic                                   ready_I;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucket_I;
    logic                                   signed_flag_I;
    logic   [`LOG2((1<<24))-1:0]            N_count_I;
    logic   [`LOG2(512)-1:0]    affineIndex_I;
    logic   [bucket_hp_length-1:0]             conflictInOtdBuffer_I;
    logic                                   conflictInJ_I;
    logic                                   conflictInL_I;
    logic                                   conflict_I;
    logic                                   stop_I;
    logic                                   bucketShiftBufferValid_I_0;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_0;
    logic                                   bucketShiftBufferValid_I_1;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_1;
    logic                                   bucketShiftBufferValid_I_2;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_2;
    logic                                   bucketShiftBufferValid_I_3;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_3;
    logic                                   bucketShiftBufferValid_I_4;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_4;
    logic                                   bucketShiftBufferValid_I_5;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_5;
    logic                                   bucketShiftBufferValid_I_6;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_6;
    logic                                   bucketShiftBufferValid_I_7;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_7;
    logic                                   bucketShiftBufferValid_I_8;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_8;
    logic                                   bucketShiftBufferValid_I_9;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucketShiftBuffer_I_9;

    logic                                   valid_J;
    logic                                   ready_J;
    logic                                   conflict_J;
    `MAX_FANOUT32 logic   [`LOG2((1<<(16-1)))-1:0]  bucket_J;
    logic                                   signed_flag_J;
    logic   [`LOG2((1<<24))-1:0]            N_count_J;
    logic   [`LOG2(512)-1:0]    affineIndex_J;

    logic   [bucket_hp_length-1:0]             conflictInOtdBuffer_J;
    logic                                   conflict_forward_S_J;
    logic                                   conflictInL_J;
    logic                                   next_conflict_J;
    logic                                   conflict_regen_J;
    logic                                   conflict_fromI_J;
    logic                                   ready_J_r;

    logic   [bucket_hp_length-1:0]             conflictInOtdBuffer_L;

    logic                                   mux_valid_J;
    logic                                   mux_ready_J;
    logic   [`LOG2((1<<(16-1)))-1:0]  mux_bucket_J;
    logic                                   mux_signed_flag_I;
    logic   [`LOG2((1<<24))-1:0]            mux_N_count_J;
    logic   [`LOG2(512)-1:0]    mux_affineIndex_J;

    logic                                   mux_J;
    logic                                   mux_priority;

    logic                                   wbOut_valid_J;
    logic                                   wbOut_ready_J;
    logic   [`LOG2((1<<(16-1)))-1:0]  wbOut_bucket_J;
    logic                                   wbOut_signed_flag_J;
    logic   [`LOG2((1<<24))-1:0]            wbOut_N_count_J;
    logic   [`LOG2(512)-1:0]    wbOut_affineIndex_J;
    logic   [16-1:0]            wbOutOnehot_J;
//cancel wbOut_valid_J
    logic                                   wbOut_conflict_J;           //the same like 

//wait buffer
    logic   [1:0]                           wbState0;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket0;
    logic                                   wb_signed_flag0;
    logic   [`LOG2(512)-1:0]    wb_affineIndex0;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count0;
    logic   [1:0]                           wbState1;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket1;
    logic                                   wb_signed_flag1;
    logic   [`LOG2(512)-1:0]    wb_affineIndex1;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count1;
    logic   [1:0]                           wbState2;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket2;
    logic                                   wb_signed_flag2;
    logic   [`LOG2(512)-1:0]    wb_affineIndex2;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count2;
    logic   [1:0]                           wbState3;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket3;
    logic                                   wb_signed_flag3;
    logic   [`LOG2(512)-1:0]    wb_affineIndex3;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count3;
    logic   [1:0]                           wbState4;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket4;
    logic                                   wb_signed_flag4;
    logic   [`LOG2(512)-1:0]    wb_affineIndex4;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count4;
    logic   [1:0]                           wbState5;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket5;
    logic                                   wb_signed_flag5;
    logic   [`LOG2(512)-1:0]    wb_affineIndex5;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count5;
    logic   [1:0]                           wbState6;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket6;
    logic                                   wb_signed_flag6;
    logic   [`LOG2(512)-1:0]    wb_affineIndex6;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count6;
    logic   [1:0]                           wbState7;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket7;
    logic                                   wb_signed_flag7;
    logic   [`LOG2(512)-1:0]    wb_affineIndex7;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count7;
    logic   [1:0]                           wbState8;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket8;
    logic                                   wb_signed_flag8;
    logic   [`LOG2(512)-1:0]    wb_affineIndex8;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count8;
    logic   [1:0]                           wbState9;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket9;
    logic                                   wb_signed_flag9;
    logic   [`LOG2(512)-1:0]    wb_affineIndex9;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count9;
    logic   [1:0]                           wbState10;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket10;
    logic                                   wb_signed_flag10;
    logic   [`LOG2(512)-1:0]    wb_affineIndex10;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count10;
    logic   [1:0]                           wbState11;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket11;
    logic                                   wb_signed_flag11;
    logic   [`LOG2(512)-1:0]    wb_affineIndex11;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count11;
    logic   [1:0]                           wbState12;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket12;
    logic                                   wb_signed_flag12;
    logic   [`LOG2(512)-1:0]    wb_affineIndex12;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count12;
    logic   [1:0]                           wbState13;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket13;
    logic                                   wb_signed_flag13;
    logic   [`LOG2(512)-1:0]    wb_affineIndex13;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count13;
    logic   [1:0]                           wbState14;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket14;
    logic                                   wb_signed_flag14;
    logic   [`LOG2(512)-1:0]    wb_affineIndex14;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count14;
    logic   [1:0]                           wbState15;
    logic   [`LOG2((1<<(16-1)))-1:0]  wb_bucket15;
    logic                                   wb_signed_flag15;
    logic   [`LOG2(512)-1:0]    wb_affineIndex15;
    logic   [`LOG2((1<<24))-1:0]            wb_N_count15;
    logic   [16-1:0]            wbEmpty;
    logic   [16-1:0]            wbValid;
    logic   [16-1:0]            wbValid_fast;
    logic   [16-1:0]            wbInOnehot;
    logic   [16-1:0]            wbOutOnehot;
    logic   [16-1:0]            wbOutOnehot_fast;
    logic   [`LOG2(16+1)-1:0]   wbNum;
    logic                                   wbReady;
    logic                                   wbNumAdd;
    logic                                   wbNumSub;
    
    logic   [`LOG2((1<<(16-1)))-1:0]  wbOut_bucket;

localparam WAITBUFFER_IDLE = 0;
localparam WAITBUFFER_WAIT = 1;
localparam WAITBUFFER_READY = 2;
localparam WAITBUFFER_OUT   = 3;

//outstanding buffer
    logic   [132-1:0]             otdbValid;
    logic   [`LOG2(132+1)-1:0]    otdbNum;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket0;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex0;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count0;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket1;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex1;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count1;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket2;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex2;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count2;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket3;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex3;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count3;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket4;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex4;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count4;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket5;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex5;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count5;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket6;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex6;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count6;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket7;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex7;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count7;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket8;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex8;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count8;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket9;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex9;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count9;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket10;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex10;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count10;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket11;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex11;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count11;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket12;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex12;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count12;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket13;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex13;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count13;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket14;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex14;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count14;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket15;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex15;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count15;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket16;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex16;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count16;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket17;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex17;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count17;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket18;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex18;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count18;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket19;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex19;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count19;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket20;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex20;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count20;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket21;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex21;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count21;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket22;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex22;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count22;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket23;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex23;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count23;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket24;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex24;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count24;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket25;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex25;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count25;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket26;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex26;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count26;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket27;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex27;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count27;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket28;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex28;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count28;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket29;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex29;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count29;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket30;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex30;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count30;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket31;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex31;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count31;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket32;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex32;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count32;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket33;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex33;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count33;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket34;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex34;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count34;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket35;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex35;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count35;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket36;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex36;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count36;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket37;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex37;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count37;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket38;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex38;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count38;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket39;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex39;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count39;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket40;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex40;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count40;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket41;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex41;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count41;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket42;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex42;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count42;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket43;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex43;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count43;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket44;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex44;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count44;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket45;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex45;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count45;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket46;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex46;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count46;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket47;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex47;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count47;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket48;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex48;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count48;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket49;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex49;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count49;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket50;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex50;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count50;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket51;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex51;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count51;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket52;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex52;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count52;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket53;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex53;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count53;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket54;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex54;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count54;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket55;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex55;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count55;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket56;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex56;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count56;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket57;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex57;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count57;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket58;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex58;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count58;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket59;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex59;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count59;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket60;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex60;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count60;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket61;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex61;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count61;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket62;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex62;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count62;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket63;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex63;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count63;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket64;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex64;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count64;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket65;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex65;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count65;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket66;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex66;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count66;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket67;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex67;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count67;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket68;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex68;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count68;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket69;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex69;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count69;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket70;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex70;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count70;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket71;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex71;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count71;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket72;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex72;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count72;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket73;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex73;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count73;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket74;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex74;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count74;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket75;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex75;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count75;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket76;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex76;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count76;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket77;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex77;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count77;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket78;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex78;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count78;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket79;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex79;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count79;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket80;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex80;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count80;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket81;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex81;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count81;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket82;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex82;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count82;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket83;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex83;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count83;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket84;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex84;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count84;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket85;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex85;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count85;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket86;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex86;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count86;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket87;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex87;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count87;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket88;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex88;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count88;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket89;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex89;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count89;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket90;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex90;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count90;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket91;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex91;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count91;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket92;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex92;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count92;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket93;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex93;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count93;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket94;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex94;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count94;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket95;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex95;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count95;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket96;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex96;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count96;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket97;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex97;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count97;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket98;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex98;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count98;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket99;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex99;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count99;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket100;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex100;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count100;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket101;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex101;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count101;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket102;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex102;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count102;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket103;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex103;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count103;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket104;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex104;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count104;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket105;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex105;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count105;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket106;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex106;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count106;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket107;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex107;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count107;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket108;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex108;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count108;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket109;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex109;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count109;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket110;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex110;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count110;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket111;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex111;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count111;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket112;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex112;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count112;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket113;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex113;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count113;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket114;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex114;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count114;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket115;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex115;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count115;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket116;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex116;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count116;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket117;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex117;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count117;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket118;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex118;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count118;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket119;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex119;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count119;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket120;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex120;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count120;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket121;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex121;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count121;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket122;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex122;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count122;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket123;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex123;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count123;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket124;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex124;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count124;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket125;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex125;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count125;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket126;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex126;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count126;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket127;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex127;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count127;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket128;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex128;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count128;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket129;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex129;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count129;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket130;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex130;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count130;
    logic   [`LOG2((1<<(16-1)))-1:0]  otdb_bucket131;
    logic   [`LOG2(512)-1:0]  otdb_affineIndex131;
    logic   [`LOG2((1<<24))-1:0]            otdb_N_count131;
    logic   [132-1:0]             otdbOnehot;

logic   [`LOG2((1<<(16-1)))-1:0]  bucket_hyperpipe[bucket_hp_length-1:0];
logic                                   bucket_hyperpipe_valid[bucket_hp_length-1:0];
//to F3

    logic                                   valid_S_r;
    logic   [`LOG2((1<<(16-1)))-1:0]  bucket_S_r;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////
//

assign bucketControl_idle = 
                !valid_F1_t &&
                !valid_I &&
                !valid_J &&
                !valid_L &&
                otdbNum == 0 &&
                wbNum == 0 &&
                1;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//inputFifo
////////////////////////////////////////////////////////////////////////////////////////////////////////
flopOutSramFifo #(
    .DATA       ($bits({signed_flag_F1_t,bucket_F1_t,N_count_F1_t,affineIndex_F1_t})),
    .DEPTH      (INFIFO_DEPTH)
    ) inFifo(
    .wDatValid      (valid_F1_t),
    .wDatReady      (ready_F1_t),
    .wDat           ({signed_flag_F1_t,bucket_F1_t,N_count_F1_t,affineIndex_F1_t}),
    .rDatValid      (valid_I),
    .rDatReady      (ready_I),
    .rDat           ({signed_flag_I,bucket_I,N_count_I,affineIndex_I}),
    .num            (),
    .clear          (1'b0),
    .clk            (clk),
    .rstN           (rstN)
    );
assign ready_I = (valid_J ? ready_J : 1) && !stop_I;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//confilict checking
////////////////////////////////////////////////////////////////////////////////////////////////////////
generate 
for(i=0;i<bucket_hp_length-2;i=i+1)begin        
    assign conflictInOtdBuffer_I[i] = bucket_hyperpipe_valid[i] && bucket_I == bucket_hyperpipe[i];
end
endgenerate
    assign conflictInOtdBuffer_I[bucket_hp_length-2] = 0;       //exclude the second last item because conflictInOtdBuffer_I is latch at previous cycle
    assign conflictInOtdBuffer_I[bucket_hp_length-1] = 0;       //exclude the last item because forward S pipe.

    //assign conflictInJ_I = mux_valid_J && bucket_I == mux_bucket_J;
    assign conflictInJ_I =  //should consider both J and wbOut_J, so timing
                            valid_J && bucket_I == bucket_J ||
//                            wbOut_valid_J && bucket_I == wbOut_bucket_J ||
                            0;
    assign conflictInL_I = valid_L && bucket_I == bucket_L;
    assign conflict_I = (|conflictInOtdBuffer_I) || conflictInJ_I || conflictInL_I;

//conflict_J re_check
//when J was stuck, re_check conflict rather than use conflict_I
generate 
for(i=0;i<bucket_hp_length-2;i=i+1)begin        
    assign conflictInOtdBuffer_J[i] = bucket_hyperpipe_valid[i] && bucket_J == bucket_hyperpipe[i];
end
endgenerate
    assign conflictInOtdBuffer_J[bucket_hp_length-2] = 0;       //exclude the second last item because conflictInOtdBuffer_I is latch at previous cycle
    assign conflictInOtdBuffer_J[bucket_hp_length-1] = 0;       //exclude the last item because forward S pipe.
//
    assign conflict_forward_S_J = valid_S ? bucket_J != bucket_S : 1;
    assign conflictInL_J = valid_L && bucket_J == bucket_L;
    assign next_conflict_J = (conflict_forward_S_J && |conflictInOtdBuffer_J) || conflictInL_J;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//bucketShiftBuffer_I
//if there are the same bucket at previous 10 cycle, stop I, to avoid contionous same bucket in J/L
////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign stop_I = valid_I && 
                bucketShiftBufferValid_I_0 && bucket_I == bucketShiftBuffer_I_0 ||
                bucketShiftBufferValid_I_1 && bucket_I == bucketShiftBuffer_I_1 ||
                bucketShiftBufferValid_I_2 && bucket_I == bucketShiftBuffer_I_2 ||
                bucketShiftBufferValid_I_3 && bucket_I == bucketShiftBuffer_I_3 ||
                bucketShiftBufferValid_I_4 && bucket_I == bucketShiftBuffer_I_4 ||
                bucketShiftBufferValid_I_5 && bucket_I == bucketShiftBuffer_I_5 ||
                bucketShiftBufferValid_I_6 && bucket_I == bucketShiftBuffer_I_6 ||
                bucketShiftBufferValid_I_7 && bucket_I == bucketShiftBuffer_I_7 ||
                bucketShiftBufferValid_I_8 && bucket_I == bucketShiftBuffer_I_8 ||
                bucketShiftBufferValid_I_9 && bucket_I == bucketShiftBuffer_I_9 ||
                0;

`Flop(bucketShiftBufferValid_I_0, 1, valid_I && ready_I)
`Flop(bucketShiftBuffer_I_0,      1, bucket_I)
`Flop(bucketShiftBufferValid_I_1, 1, bucketShiftBufferValid_I_0)
`Flop(bucketShiftBuffer_I_1,      1, bucketShiftBuffer_I_0)
`Flop(bucketShiftBufferValid_I_2, 1, bucketShiftBufferValid_I_1)
`Flop(bucketShiftBuffer_I_2,      1, bucketShiftBuffer_I_1)
`Flop(bucketShiftBufferValid_I_3, 1, bucketShiftBufferValid_I_2)
`Flop(bucketShiftBuffer_I_3,      1, bucketShiftBuffer_I_2)
`Flop(bucketShiftBufferValid_I_4, 1, bucketShiftBufferValid_I_3)
`Flop(bucketShiftBuffer_I_4,      1, bucketShiftBuffer_I_3)
`Flop(bucketShiftBufferValid_I_5, 1, bucketShiftBufferValid_I_4)
`Flop(bucketShiftBuffer_I_5,      1, bucketShiftBuffer_I_4)
`Flop(bucketShiftBufferValid_I_6, 1, bucketShiftBufferValid_I_5)
`Flop(bucketShiftBuffer_I_6,      1, bucketShiftBuffer_I_5)
`Flop(bucketShiftBufferValid_I_7, 1, bucketShiftBufferValid_I_6)
`Flop(bucketShiftBuffer_I_7,      1, bucketShiftBuffer_I_6)
`Flop(bucketShiftBufferValid_I_8, 1, bucketShiftBufferValid_I_7)
`Flop(bucketShiftBuffer_I_8,      1, bucketShiftBuffer_I_7)
`Flop(bucketShiftBufferValid_I_9, 1, bucketShiftBufferValid_I_8)
`Flop(bucketShiftBuffer_I_9,      1, bucketShiftBuffer_I_8)
////////////////////////////////////////////////////////////////////////////////////////////////////////
//J
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopSC0(valid_J, valid_I && ready_I, valid_J && ready_J)
`FlopN({signed_flag_J,bucket_J,N_count_J,affineIndex_J}, valid_I && ready_I, {signed_flag_I,bucket_I,N_count_I,affineIndex_I})
//`FlopN(conflict_fromI_J, valid_I && ready_I, conflict_I)
`FlopN(conflict_fromI_J, 1'b1, conflict_I)// conflict_I will expire after 1 cycle
assign ready_J = valid_J && (!conflict_J ? !mux_J && mux_ready_J : wbNum < 16);
//assign ready_J = valid_J && conflict_J ? wbNum < 16 : !mux_J && mux_ready_J;
//assign ready_J = valid_J && conflict_J ? wbNum < 16 : !(conflict_J || !valid_J) && mux_ready_J;

//gen real_conflict_J
// if valid_J && !ready_J && conflict_J, than newest_conflict_J need regen
// if !conflict_J, don't need regen
`FlopN(conflict_regen_J, 1, next_conflict_J)
`FlopN(ready_J_r, 1, valid_J ? ready_J : 1)
assign conflict_J = ready_J_r ? conflict_fromI_J : conflict_regen_J;
//conflict_J not include the cycle that bucket_S

////////////////////////////////////////////////////////////////////////////////////////////////////////
//wait buffer out at J
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopSC0(wbOut_valid_J, (|wbValid) && wbReady, wbOut_valid_J && wbOut_ready_J || wbOut_conflict_J)
`FlopN(wbOutOnehot_J, (|wbValid) && wbReady, wbOutOnehot)
assign {wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J} =
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[0]}} & {wb_signed_flag0,wb_bucket0,wb_N_count0,wb_affineIndex0} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[1]}} & {wb_signed_flag1,wb_bucket1,wb_N_count1,wb_affineIndex1} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[2]}} & {wb_signed_flag2,wb_bucket2,wb_N_count2,wb_affineIndex2} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[3]}} & {wb_signed_flag3,wb_bucket3,wb_N_count3,wb_affineIndex3} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[4]}} & {wb_signed_flag4,wb_bucket4,wb_N_count4,wb_affineIndex4} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[5]}} & {wb_signed_flag5,wb_bucket5,wb_N_count5,wb_affineIndex5} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[6]}} & {wb_signed_flag6,wb_bucket6,wb_N_count6,wb_affineIndex6} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[7]}} & {wb_signed_flag7,wb_bucket7,wb_N_count7,wb_affineIndex7} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[8]}} & {wb_signed_flag8,wb_bucket8,wb_N_count8,wb_affineIndex8} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[9]}} & {wb_signed_flag9,wb_bucket9,wb_N_count9,wb_affineIndex9} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[10]}} & {wb_signed_flag10,wb_bucket10,wb_N_count10,wb_affineIndex10} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[11]}} & {wb_signed_flag11,wb_bucket11,wb_N_count11,wb_affineIndex11} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[12]}} & {wb_signed_flag12,wb_bucket12,wb_N_count12,wb_affineIndex12} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[13]}} & {wb_signed_flag13,wb_bucket13,wb_N_count13,wb_affineIndex13} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[14]}} & {wb_signed_flag14,wb_bucket14,wb_N_count14,wb_affineIndex14} |
        {$bits({wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J}){wbOutOnehot_J[15]}} & {wb_signed_flag15,wb_bucket15,wb_N_count15,wb_affineIndex15} |
        0;

assign wbOut_ready_J = mux_J && mux_ready_J;

assign wbOut_bucket = 
        {$bits(wbOut_bucket){wbOutOnehot[0]}} & wb_bucket0 |
        {$bits(wbOut_bucket){wbOutOnehot[1]}} & wb_bucket1 |
        {$bits(wbOut_bucket){wbOutOnehot[2]}} & wb_bucket2 |
        {$bits(wbOut_bucket){wbOutOnehot[3]}} & wb_bucket3 |
        {$bits(wbOut_bucket){wbOutOnehot[4]}} & wb_bucket4 |
        {$bits(wbOut_bucket){wbOutOnehot[5]}} & wb_bucket5 |
        {$bits(wbOut_bucket){wbOutOnehot[6]}} & wb_bucket6 |
        {$bits(wbOut_bucket){wbOutOnehot[7]}} & wb_bucket7 |
        {$bits(wbOut_bucket){wbOutOnehot[8]}} & wb_bucket8 |
        {$bits(wbOut_bucket){wbOutOnehot[9]}} & wb_bucket9 |
        {$bits(wbOut_bucket){wbOutOnehot[10]}} & wb_bucket10 |
        {$bits(wbOut_bucket){wbOutOnehot[11]}} & wb_bucket11 |
        {$bits(wbOut_bucket){wbOutOnehot[12]}} & wb_bucket12 |
        {$bits(wbOut_bucket){wbOutOnehot[13]}} & wb_bucket13 |
        {$bits(wbOut_bucket){wbOutOnehot[14]}} & wb_bucket14 |
        {$bits(wbOut_bucket){wbOutOnehot[15]}} & wb_bucket15 |
        0;

assign wbOut_conflict_J = wbOut_valid_J && (
                            valid_I && bucket_I == wbOut_bucket_J ||
                            valid_J && bucket_J == wbOut_bucket_J ||
                            valid_L && bucket_L == wbOut_bucket_J ||
                            0);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//J mux
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign mux_valid_J = valid_J && !conflict_J || wbOut_valid_J && !wbOut_conflict_J;
assign mux_ready_J = 1;
//mux_priority is 0 for temp; so that mux_J = conflict_J || !valid_J;
assign mux_J = mux_priority && wbOut_valid_J || !mux_priority && !(valid_J && !conflict_J);       //1 from wbOut, 0 from pipe J

assign {mux_signed_flag_J,mux_bucket_J,mux_N_count_J,mux_affineIndex_J} = 
            mux_J ? {wbOut_signed_flag_J,wbOut_bucket_J,wbOut_N_count_J,wbOut_affineIndex_J} :
                  {signed_flag_J,bucket_J,N_count_J,affineIndex_J};
////////////////////////////////////////////////////////////////////////////////////////////////////////
//L
////////////////////////////////////////////////////////////////////////////////////////////////////////
`Flop(valid_L, 1, mux_valid_J && mux_ready_J)
`FlopN({signed_flag_L,bucket_L,N_count_L,affineIndex_L}, mux_valid_J && mux_ready_J, {mux_signed_flag_J,mux_bucket_J,mux_N_count_J,mux_affineIndex_J})
////////////////////////////////////////////////////////////////////////////////////////////////////////
//wait buffer
////////////////////////////////////////////////////////////////////////////////////////////////////////
assign wbReady = wbOut_valid_J ? wbOut_ready_J : 1;

assign wbEmpty[0] = wbState0 == WAITBUFFER_IDLE; 
assign wbValid[0] = wbState0 == WAITBUFFER_READY;
assign wbValid_fast[0] = wbState0 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState0 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState0)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[0]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState0 <= `CQ WAITBUFFER_READY;
                    else
                        wbState0 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket0)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket0)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket0)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket0)
                    wbState0 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket0)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket0)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket0)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[0] && wbReady)
                    wbState0 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket0)
//                else if(wbReady && conflict_vector[0])
 //                   wbState0 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState0 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[0])
                    wbState0 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag0,wb_bucket0,wb_affineIndex0,wb_N_count0}, valid_J && conflict_J && wbInOnehot[0], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState0 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket0 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket0 == wbOut_bucket_J)
//`assert_clk(wbState0 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket0 |->  conflict_J || wb_bucket0 == wbOut_bucket_J || wb_bucket0 == wbOut_bucket)
`assert_clk(wbState0 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket0 |-> !conflict_J)

assign wbEmpty[1] = wbState1 == WAITBUFFER_IDLE; 
assign wbValid[1] = wbState1 == WAITBUFFER_READY;
assign wbValid_fast[1] = wbState1 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState1 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState1)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[1]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState1 <= `CQ WAITBUFFER_READY;
                    else
                        wbState1 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket1)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket1)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket1)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket1)
                    wbState1 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket1)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket1)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket1)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[1] && wbReady)
                    wbState1 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket1)
//                else if(wbReady && conflict_vector[1])
 //                   wbState1 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState1 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[1])
                    wbState1 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag1,wb_bucket1,wb_affineIndex1,wb_N_count1}, valid_J && conflict_J && wbInOnehot[1], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState1 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket1 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket1 == wbOut_bucket_J)
//`assert_clk(wbState1 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket1 |->  conflict_J || wb_bucket1 == wbOut_bucket_J || wb_bucket1 == wbOut_bucket)
`assert_clk(wbState1 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket1 |-> !conflict_J)

assign wbEmpty[2] = wbState2 == WAITBUFFER_IDLE; 
assign wbValid[2] = wbState2 == WAITBUFFER_READY;
assign wbValid_fast[2] = wbState2 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState2 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState2)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[2]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState2 <= `CQ WAITBUFFER_READY;
                    else
                        wbState2 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket2)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket2)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket2)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket2)
                    wbState2 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket2)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket2)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket2)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[2] && wbReady)
                    wbState2 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket2)
//                else if(wbReady && conflict_vector[2])
 //                   wbState2 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState2 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[2])
                    wbState2 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag2,wb_bucket2,wb_affineIndex2,wb_N_count2}, valid_J && conflict_J && wbInOnehot[2], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState2 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket2 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket2 == wbOut_bucket_J)
//`assert_clk(wbState2 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket2 |->  conflict_J || wb_bucket2 == wbOut_bucket_J || wb_bucket2 == wbOut_bucket)
`assert_clk(wbState2 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket2 |-> !conflict_J)

assign wbEmpty[3] = wbState3 == WAITBUFFER_IDLE; 
assign wbValid[3] = wbState3 == WAITBUFFER_READY;
assign wbValid_fast[3] = wbState3 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState3 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState3)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[3]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState3 <= `CQ WAITBUFFER_READY;
                    else
                        wbState3 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket3)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket3)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket3)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket3)
                    wbState3 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket3)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket3)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket3)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[3] && wbReady)
                    wbState3 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket3)
//                else if(wbReady && conflict_vector[3])
 //                   wbState3 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState3 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[3])
                    wbState3 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag3,wb_bucket3,wb_affineIndex3,wb_N_count3}, valid_J && conflict_J && wbInOnehot[3], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState3 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket3 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket3 == wbOut_bucket_J)
//`assert_clk(wbState3 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket3 |->  conflict_J || wb_bucket3 == wbOut_bucket_J || wb_bucket3 == wbOut_bucket)
`assert_clk(wbState3 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket3 |-> !conflict_J)

assign wbEmpty[4] = wbState4 == WAITBUFFER_IDLE; 
assign wbValid[4] = wbState4 == WAITBUFFER_READY;
assign wbValid_fast[4] = wbState4 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState4 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState4)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[4]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState4 <= `CQ WAITBUFFER_READY;
                    else
                        wbState4 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket4)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket4)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket4)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket4)
                    wbState4 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket4)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket4)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket4)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[4] && wbReady)
                    wbState4 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket4)
//                else if(wbReady && conflict_vector[4])
 //                   wbState4 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState4 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[4])
                    wbState4 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag4,wb_bucket4,wb_affineIndex4,wb_N_count4}, valid_J && conflict_J && wbInOnehot[4], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState4 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket4 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket4 == wbOut_bucket_J)
//`assert_clk(wbState4 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket4 |->  conflict_J || wb_bucket4 == wbOut_bucket_J || wb_bucket4 == wbOut_bucket)
`assert_clk(wbState4 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket4 |-> !conflict_J)

assign wbEmpty[5] = wbState5 == WAITBUFFER_IDLE; 
assign wbValid[5] = wbState5 == WAITBUFFER_READY;
assign wbValid_fast[5] = wbState5 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState5 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState5)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[5]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState5 <= `CQ WAITBUFFER_READY;
                    else
                        wbState5 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket5)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket5)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket5)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket5)
                    wbState5 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket5)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket5)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket5)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[5] && wbReady)
                    wbState5 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket5)
//                else if(wbReady && conflict_vector[5])
 //                   wbState5 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState5 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[5])
                    wbState5 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag5,wb_bucket5,wb_affineIndex5,wb_N_count5}, valid_J && conflict_J && wbInOnehot[5], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState5 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket5 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket5 == wbOut_bucket_J)
//`assert_clk(wbState5 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket5 |->  conflict_J || wb_bucket5 == wbOut_bucket_J || wb_bucket5 == wbOut_bucket)
`assert_clk(wbState5 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket5 |-> !conflict_J)

assign wbEmpty[6] = wbState6 == WAITBUFFER_IDLE; 
assign wbValid[6] = wbState6 == WAITBUFFER_READY;
assign wbValid_fast[6] = wbState6 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState6 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState6)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[6]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState6 <= `CQ WAITBUFFER_READY;
                    else
                        wbState6 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket6)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket6)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket6)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket6)
                    wbState6 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket6)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket6)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket6)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[6] && wbReady)
                    wbState6 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket6)
//                else if(wbReady && conflict_vector[6])
 //                   wbState6 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState6 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[6])
                    wbState6 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag6,wb_bucket6,wb_affineIndex6,wb_N_count6}, valid_J && conflict_J && wbInOnehot[6], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState6 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket6 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket6 == wbOut_bucket_J)
//`assert_clk(wbState6 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket6 |->  conflict_J || wb_bucket6 == wbOut_bucket_J || wb_bucket6 == wbOut_bucket)
`assert_clk(wbState6 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket6 |-> !conflict_J)

assign wbEmpty[7] = wbState7 == WAITBUFFER_IDLE; 
assign wbValid[7] = wbState7 == WAITBUFFER_READY;
assign wbValid_fast[7] = wbState7 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState7 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState7)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[7]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState7 <= `CQ WAITBUFFER_READY;
                    else
                        wbState7 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket7)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket7)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket7)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket7)
                    wbState7 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket7)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket7)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket7)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[7] && wbReady)
                    wbState7 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket7)
//                else if(wbReady && conflict_vector[7])
 //                   wbState7 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState7 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[7])
                    wbState7 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag7,wb_bucket7,wb_affineIndex7,wb_N_count7}, valid_J && conflict_J && wbInOnehot[7], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState7 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket7 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket7 == wbOut_bucket_J)
//`assert_clk(wbState7 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket7 |->  conflict_J || wb_bucket7 == wbOut_bucket_J || wb_bucket7 == wbOut_bucket)
`assert_clk(wbState7 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket7 |-> !conflict_J)

assign wbEmpty[8] = wbState8 == WAITBUFFER_IDLE; 
assign wbValid[8] = wbState8 == WAITBUFFER_READY;
assign wbValid_fast[8] = wbState8 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState8 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState8)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[8]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState8 <= `CQ WAITBUFFER_READY;
                    else
                        wbState8 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket8)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket8)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket8)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket8)
                    wbState8 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket8)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket8)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket8)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[8] && wbReady)
                    wbState8 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket8)
//                else if(wbReady && conflict_vector[8])
 //                   wbState8 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState8 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[8])
                    wbState8 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag8,wb_bucket8,wb_affineIndex8,wb_N_count8}, valid_J && conflict_J && wbInOnehot[8], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState8 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket8 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket8 == wbOut_bucket_J)
//`assert_clk(wbState8 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket8 |->  conflict_J || wb_bucket8 == wbOut_bucket_J || wb_bucket8 == wbOut_bucket)
`assert_clk(wbState8 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket8 |-> !conflict_J)

assign wbEmpty[9] = wbState9 == WAITBUFFER_IDLE; 
assign wbValid[9] = wbState9 == WAITBUFFER_READY;
assign wbValid_fast[9] = wbState9 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState9 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState9)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[9]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState9 <= `CQ WAITBUFFER_READY;
                    else
                        wbState9 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket9)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket9)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket9)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket9)
                    wbState9 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket9)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket9)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket9)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[9] && wbReady)
                    wbState9 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket9)
//                else if(wbReady && conflict_vector[9])
 //                   wbState9 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState9 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[9])
                    wbState9 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag9,wb_bucket9,wb_affineIndex9,wb_N_count9}, valid_J && conflict_J && wbInOnehot[9], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState9 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket9 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket9 == wbOut_bucket_J)
//`assert_clk(wbState9 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket9 |->  conflict_J || wb_bucket9 == wbOut_bucket_J || wb_bucket9 == wbOut_bucket)
`assert_clk(wbState9 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket9 |-> !conflict_J)

assign wbEmpty[10] = wbState10 == WAITBUFFER_IDLE; 
assign wbValid[10] = wbState10 == WAITBUFFER_READY;
assign wbValid_fast[10] = wbState10 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState10 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState10)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[10]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState10 <= `CQ WAITBUFFER_READY;
                    else
                        wbState10 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket10)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket10)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket10)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket10)
                    wbState10 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket10)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket10)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket10)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[10] && wbReady)
                    wbState10 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket10)
//                else if(wbReady && conflict_vector[10])
 //                   wbState10 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState10 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[10])
                    wbState10 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag10,wb_bucket10,wb_affineIndex10,wb_N_count10}, valid_J && conflict_J && wbInOnehot[10], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState10 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket10 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket10 == wbOut_bucket_J)
//`assert_clk(wbState10 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket10 |->  conflict_J || wb_bucket10 == wbOut_bucket_J || wb_bucket10 == wbOut_bucket)
`assert_clk(wbState10 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket10 |-> !conflict_J)

assign wbEmpty[11] = wbState11 == WAITBUFFER_IDLE; 
assign wbValid[11] = wbState11 == WAITBUFFER_READY;
assign wbValid_fast[11] = wbState11 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState11 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState11)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[11]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState11 <= `CQ WAITBUFFER_READY;
                    else
                        wbState11 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket11)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket11)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket11)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket11)
                    wbState11 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket11)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket11)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket11)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[11] && wbReady)
                    wbState11 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket11)
//                else if(wbReady && conflict_vector[11])
 //                   wbState11 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState11 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[11])
                    wbState11 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag11,wb_bucket11,wb_affineIndex11,wb_N_count11}, valid_J && conflict_J && wbInOnehot[11], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState11 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket11 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket11 == wbOut_bucket_J)
//`assert_clk(wbState11 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket11 |->  conflict_J || wb_bucket11 == wbOut_bucket_J || wb_bucket11 == wbOut_bucket)
`assert_clk(wbState11 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket11 |-> !conflict_J)

assign wbEmpty[12] = wbState12 == WAITBUFFER_IDLE; 
assign wbValid[12] = wbState12 == WAITBUFFER_READY;
assign wbValid_fast[12] = wbState12 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState12 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState12)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[12]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState12 <= `CQ WAITBUFFER_READY;
                    else
                        wbState12 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket12)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket12)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket12)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket12)
                    wbState12 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket12)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket12)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket12)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[12] && wbReady)
                    wbState12 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket12)
//                else if(wbReady && conflict_vector[12])
 //                   wbState12 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState12 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[12])
                    wbState12 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag12,wb_bucket12,wb_affineIndex12,wb_N_count12}, valid_J && conflict_J && wbInOnehot[12], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState12 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket12 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket12 == wbOut_bucket_J)
//`assert_clk(wbState12 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket12 |->  conflict_J || wb_bucket12 == wbOut_bucket_J || wb_bucket12 == wbOut_bucket)
`assert_clk(wbState12 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket12 |-> !conflict_J)

assign wbEmpty[13] = wbState13 == WAITBUFFER_IDLE; 
assign wbValid[13] = wbState13 == WAITBUFFER_READY;
assign wbValid_fast[13] = wbState13 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState13 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState13)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[13]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState13 <= `CQ WAITBUFFER_READY;
                    else
                        wbState13 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket13)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket13)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket13)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket13)
                    wbState13 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket13)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket13)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket13)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[13] && wbReady)
                    wbState13 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket13)
//                else if(wbReady && conflict_vector[13])
 //                   wbState13 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState13 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[13])
                    wbState13 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag13,wb_bucket13,wb_affineIndex13,wb_N_count13}, valid_J && conflict_J && wbInOnehot[13], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState13 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket13 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket13 == wbOut_bucket_J)
//`assert_clk(wbState13 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket13 |->  conflict_J || wb_bucket13 == wbOut_bucket_J || wb_bucket13 == wbOut_bucket)
`assert_clk(wbState13 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket13 |-> !conflict_J)

assign wbEmpty[14] = wbState14 == WAITBUFFER_IDLE; 
assign wbValid[14] = wbState14 == WAITBUFFER_READY;
assign wbValid_fast[14] = wbState14 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState14 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState14)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[14]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState14 <= `CQ WAITBUFFER_READY;
                    else
                        wbState14 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket14)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket14)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket14)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket14)
                    wbState14 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket14)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket14)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket14)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[14] && wbReady)
                    wbState14 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket14)
//                else if(wbReady && conflict_vector[14])
 //                   wbState14 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState14 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[14])
                    wbState14 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag14,wb_bucket14,wb_affineIndex14,wb_N_count14}, valid_J && conflict_J && wbInOnehot[14], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState14 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket14 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket14 == wbOut_bucket_J)
//`assert_clk(wbState14 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket14 |->  conflict_J || wb_bucket14 == wbOut_bucket_J || wb_bucket14 == wbOut_bucket)
`assert_clk(wbState14 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket14 |-> !conflict_J)

assign wbEmpty[15] = wbState15 == WAITBUFFER_IDLE; 
assign wbValid[15] = wbState15 == WAITBUFFER_READY;
assign wbValid_fast[15] = wbState15 == WAITBUFFER_READY;
    always@(posedge clk or negedge rstN)
        if(!rstN)
            wbState15 <= `CQ WAITBUFFER_IDLE;
        else 
            case(wbState15)
            WAITBUFFER_IDLE:
                if(valid_J && ready_J && conflict_J && wbInOnehot[15]) begin
                    if(valid_S && bucket_S == bucket_J)
                        wbState15 <= `CQ WAITBUFFER_READY;
                    else
                        wbState15 <= `CQ WAITBUFFER_WAIT;
                end
            WAITBUFFER_WAIT:
                if(valid_J && bucket_J == wb_bucket15)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket15)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket15)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(valid_S && bucket_S == wb_bucket15)
                    wbState15 <= `CQ WAITBUFFER_READY;
            WAITBUFFER_READY:
                if(valid_J && bucket_J == wb_bucket15)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_bucket_J == wb_bucket15)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wb_bucket15)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(wbOutOnehot_fast[15] && wbReady)
                    wbState15 <= `CQ WAITBUFFER_OUT;
                //else if(wbReady && wbOut_bucket == wb_bucket15)
//                else if(wbReady && conflict_vector[15])
 //                   wbState15 <= `CQ WAITBUFFER_WAIT;
            WAITBUFFER_OUT:
//                if(valid_J && bucket_J == wbOut_bucket_J)
                if(valid_I && bucket_I == wbOut_bucket_J)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(valid_L && bucket_L == wbOut_bucket_J)
                    wbState15 <= `CQ WAITBUFFER_WAIT;
                else if(wbOut_valid_J && wbOut_ready_J && wbOutOnehot_J[15])
                    wbState15 <= `CQ WAITBUFFER_IDLE;
            endcase

`FlopN({wb_signed_flag15,wb_bucket15,wb_affineIndex15,wb_N_count15}, valid_J && conflict_J && wbInOnehot[15], {signed_flag_J,bucket_J,affineIndex_J,N_count_J})
//`assert_clk(wbState15 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket15 |->  conflict_J || valid_S && bucket_S == bucket_J || valid_S_r && bucket_S_r == bucket_J || wb_bucket15 == wbOut_bucket_J)
//`assert_clk(wbState15 == WAITBUFFER_WAIT  && valid_J && bucket_J == wb_bucket15 |->  conflict_J || wb_bucket15 == wbOut_bucket_J || wb_bucket15 == wbOut_bucket)
`assert_clk(wbState15 == WAITBUFFER_READY && valid_J && bucket_J == wb_bucket15 |-> !conflict_J)

//assert state == WAIT: 
//case 1: conflict;
//case 2: not conflict, but wbOut_bucket_J == wb_bucketj

assign wbNumAdd = valid_J && ready_J && conflict_J;
assign wbNumSub = wbOut_valid_J && !wbOut_conflict_J && wbOut_ready_J && !(valid_J && ready_J && bucket_J == wbOut_bucket_J);
`Flop(wbNum, wbNumAdd != wbNumSub, wbNumAdd ? wbNum + 1 : wbNum - 1)

`assert_clk(wbNum == num16(~wbEmpty))

    so#(
        .DATA(16)
    ) so_wbi(
        .in(wbEmpty),
        .out(wbInOnehot)
    );

    so#(
        .DATA(16)
    ) so_wbv(
        .in(wbValid),
        .out(wbOutOnehot)
    );

    so#(
        .DATA(16)
    ) so_wbv_fast(
        .in(wbValid_fast),
        .out(wbOutOnehot_fast)
    );

//`Flop(mux_priority, 1, wbNum >= 16*3/4)
    assign mux_priority = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//outstanding buffer
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopSC0(otdbValid[0], valid_L && otdbOnehot[0], valid_S && bucket_S == otdb_bucket0)
`FlopN({otdb_bucket0,otdb_affineIndex0,otdb_N_count0},  valid_L && otdbOnehot[0], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[1], valid_L && otdbOnehot[1], valid_S && bucket_S == otdb_bucket1)
`FlopN({otdb_bucket1,otdb_affineIndex1,otdb_N_count1},  valid_L && otdbOnehot[1], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[2], valid_L && otdbOnehot[2], valid_S && bucket_S == otdb_bucket2)
`FlopN({otdb_bucket2,otdb_affineIndex2,otdb_N_count2},  valid_L && otdbOnehot[2], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[3], valid_L && otdbOnehot[3], valid_S && bucket_S == otdb_bucket3)
`FlopN({otdb_bucket3,otdb_affineIndex3,otdb_N_count3},  valid_L && otdbOnehot[3], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[4], valid_L && otdbOnehot[4], valid_S && bucket_S == otdb_bucket4)
`FlopN({otdb_bucket4,otdb_affineIndex4,otdb_N_count4},  valid_L && otdbOnehot[4], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[5], valid_L && otdbOnehot[5], valid_S && bucket_S == otdb_bucket5)
`FlopN({otdb_bucket5,otdb_affineIndex5,otdb_N_count5},  valid_L && otdbOnehot[5], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[6], valid_L && otdbOnehot[6], valid_S && bucket_S == otdb_bucket6)
`FlopN({otdb_bucket6,otdb_affineIndex6,otdb_N_count6},  valid_L && otdbOnehot[6], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[7], valid_L && otdbOnehot[7], valid_S && bucket_S == otdb_bucket7)
`FlopN({otdb_bucket7,otdb_affineIndex7,otdb_N_count7},  valid_L && otdbOnehot[7], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[8], valid_L && otdbOnehot[8], valid_S && bucket_S == otdb_bucket8)
`FlopN({otdb_bucket8,otdb_affineIndex8,otdb_N_count8},  valid_L && otdbOnehot[8], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[9], valid_L && otdbOnehot[9], valid_S && bucket_S == otdb_bucket9)
`FlopN({otdb_bucket9,otdb_affineIndex9,otdb_N_count9},  valid_L && otdbOnehot[9], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[10], valid_L && otdbOnehot[10], valid_S && bucket_S == otdb_bucket10)
`FlopN({otdb_bucket10,otdb_affineIndex10,otdb_N_count10},  valid_L && otdbOnehot[10], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[11], valid_L && otdbOnehot[11], valid_S && bucket_S == otdb_bucket11)
`FlopN({otdb_bucket11,otdb_affineIndex11,otdb_N_count11},  valid_L && otdbOnehot[11], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[12], valid_L && otdbOnehot[12], valid_S && bucket_S == otdb_bucket12)
`FlopN({otdb_bucket12,otdb_affineIndex12,otdb_N_count12},  valid_L && otdbOnehot[12], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[13], valid_L && otdbOnehot[13], valid_S && bucket_S == otdb_bucket13)
`FlopN({otdb_bucket13,otdb_affineIndex13,otdb_N_count13},  valid_L && otdbOnehot[13], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[14], valid_L && otdbOnehot[14], valid_S && bucket_S == otdb_bucket14)
`FlopN({otdb_bucket14,otdb_affineIndex14,otdb_N_count14},  valid_L && otdbOnehot[14], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[15], valid_L && otdbOnehot[15], valid_S && bucket_S == otdb_bucket15)
`FlopN({otdb_bucket15,otdb_affineIndex15,otdb_N_count15},  valid_L && otdbOnehot[15], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[16], valid_L && otdbOnehot[16], valid_S && bucket_S == otdb_bucket16)
`FlopN({otdb_bucket16,otdb_affineIndex16,otdb_N_count16},  valid_L && otdbOnehot[16], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[17], valid_L && otdbOnehot[17], valid_S && bucket_S == otdb_bucket17)
`FlopN({otdb_bucket17,otdb_affineIndex17,otdb_N_count17},  valid_L && otdbOnehot[17], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[18], valid_L && otdbOnehot[18], valid_S && bucket_S == otdb_bucket18)
`FlopN({otdb_bucket18,otdb_affineIndex18,otdb_N_count18},  valid_L && otdbOnehot[18], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[19], valid_L && otdbOnehot[19], valid_S && bucket_S == otdb_bucket19)
`FlopN({otdb_bucket19,otdb_affineIndex19,otdb_N_count19},  valid_L && otdbOnehot[19], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[20], valid_L && otdbOnehot[20], valid_S && bucket_S == otdb_bucket20)
`FlopN({otdb_bucket20,otdb_affineIndex20,otdb_N_count20},  valid_L && otdbOnehot[20], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[21], valid_L && otdbOnehot[21], valid_S && bucket_S == otdb_bucket21)
`FlopN({otdb_bucket21,otdb_affineIndex21,otdb_N_count21},  valid_L && otdbOnehot[21], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[22], valid_L && otdbOnehot[22], valid_S && bucket_S == otdb_bucket22)
`FlopN({otdb_bucket22,otdb_affineIndex22,otdb_N_count22},  valid_L && otdbOnehot[22], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[23], valid_L && otdbOnehot[23], valid_S && bucket_S == otdb_bucket23)
`FlopN({otdb_bucket23,otdb_affineIndex23,otdb_N_count23},  valid_L && otdbOnehot[23], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[24], valid_L && otdbOnehot[24], valid_S && bucket_S == otdb_bucket24)
`FlopN({otdb_bucket24,otdb_affineIndex24,otdb_N_count24},  valid_L && otdbOnehot[24], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[25], valid_L && otdbOnehot[25], valid_S && bucket_S == otdb_bucket25)
`FlopN({otdb_bucket25,otdb_affineIndex25,otdb_N_count25},  valid_L && otdbOnehot[25], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[26], valid_L && otdbOnehot[26], valid_S && bucket_S == otdb_bucket26)
`FlopN({otdb_bucket26,otdb_affineIndex26,otdb_N_count26},  valid_L && otdbOnehot[26], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[27], valid_L && otdbOnehot[27], valid_S && bucket_S == otdb_bucket27)
`FlopN({otdb_bucket27,otdb_affineIndex27,otdb_N_count27},  valid_L && otdbOnehot[27], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[28], valid_L && otdbOnehot[28], valid_S && bucket_S == otdb_bucket28)
`FlopN({otdb_bucket28,otdb_affineIndex28,otdb_N_count28},  valid_L && otdbOnehot[28], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[29], valid_L && otdbOnehot[29], valid_S && bucket_S == otdb_bucket29)
`FlopN({otdb_bucket29,otdb_affineIndex29,otdb_N_count29},  valid_L && otdbOnehot[29], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[30], valid_L && otdbOnehot[30], valid_S && bucket_S == otdb_bucket30)
`FlopN({otdb_bucket30,otdb_affineIndex30,otdb_N_count30},  valid_L && otdbOnehot[30], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[31], valid_L && otdbOnehot[31], valid_S && bucket_S == otdb_bucket31)
`FlopN({otdb_bucket31,otdb_affineIndex31,otdb_N_count31},  valid_L && otdbOnehot[31], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[32], valid_L && otdbOnehot[32], valid_S && bucket_S == otdb_bucket32)
`FlopN({otdb_bucket32,otdb_affineIndex32,otdb_N_count32},  valid_L && otdbOnehot[32], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[33], valid_L && otdbOnehot[33], valid_S && bucket_S == otdb_bucket33)
`FlopN({otdb_bucket33,otdb_affineIndex33,otdb_N_count33},  valid_L && otdbOnehot[33], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[34], valid_L && otdbOnehot[34], valid_S && bucket_S == otdb_bucket34)
`FlopN({otdb_bucket34,otdb_affineIndex34,otdb_N_count34},  valid_L && otdbOnehot[34], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[35], valid_L && otdbOnehot[35], valid_S && bucket_S == otdb_bucket35)
`FlopN({otdb_bucket35,otdb_affineIndex35,otdb_N_count35},  valid_L && otdbOnehot[35], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[36], valid_L && otdbOnehot[36], valid_S && bucket_S == otdb_bucket36)
`FlopN({otdb_bucket36,otdb_affineIndex36,otdb_N_count36},  valid_L && otdbOnehot[36], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[37], valid_L && otdbOnehot[37], valid_S && bucket_S == otdb_bucket37)
`FlopN({otdb_bucket37,otdb_affineIndex37,otdb_N_count37},  valid_L && otdbOnehot[37], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[38], valid_L && otdbOnehot[38], valid_S && bucket_S == otdb_bucket38)
`FlopN({otdb_bucket38,otdb_affineIndex38,otdb_N_count38},  valid_L && otdbOnehot[38], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[39], valid_L && otdbOnehot[39], valid_S && bucket_S == otdb_bucket39)
`FlopN({otdb_bucket39,otdb_affineIndex39,otdb_N_count39},  valid_L && otdbOnehot[39], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[40], valid_L && otdbOnehot[40], valid_S && bucket_S == otdb_bucket40)
`FlopN({otdb_bucket40,otdb_affineIndex40,otdb_N_count40},  valid_L && otdbOnehot[40], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[41], valid_L && otdbOnehot[41], valid_S && bucket_S == otdb_bucket41)
`FlopN({otdb_bucket41,otdb_affineIndex41,otdb_N_count41},  valid_L && otdbOnehot[41], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[42], valid_L && otdbOnehot[42], valid_S && bucket_S == otdb_bucket42)
`FlopN({otdb_bucket42,otdb_affineIndex42,otdb_N_count42},  valid_L && otdbOnehot[42], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[43], valid_L && otdbOnehot[43], valid_S && bucket_S == otdb_bucket43)
`FlopN({otdb_bucket43,otdb_affineIndex43,otdb_N_count43},  valid_L && otdbOnehot[43], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[44], valid_L && otdbOnehot[44], valid_S && bucket_S == otdb_bucket44)
`FlopN({otdb_bucket44,otdb_affineIndex44,otdb_N_count44},  valid_L && otdbOnehot[44], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[45], valid_L && otdbOnehot[45], valid_S && bucket_S == otdb_bucket45)
`FlopN({otdb_bucket45,otdb_affineIndex45,otdb_N_count45},  valid_L && otdbOnehot[45], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[46], valid_L && otdbOnehot[46], valid_S && bucket_S == otdb_bucket46)
`FlopN({otdb_bucket46,otdb_affineIndex46,otdb_N_count46},  valid_L && otdbOnehot[46], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[47], valid_L && otdbOnehot[47], valid_S && bucket_S == otdb_bucket47)
`FlopN({otdb_bucket47,otdb_affineIndex47,otdb_N_count47},  valid_L && otdbOnehot[47], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[48], valid_L && otdbOnehot[48], valid_S && bucket_S == otdb_bucket48)
`FlopN({otdb_bucket48,otdb_affineIndex48,otdb_N_count48},  valid_L && otdbOnehot[48], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[49], valid_L && otdbOnehot[49], valid_S && bucket_S == otdb_bucket49)
`FlopN({otdb_bucket49,otdb_affineIndex49,otdb_N_count49},  valid_L && otdbOnehot[49], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[50], valid_L && otdbOnehot[50], valid_S && bucket_S == otdb_bucket50)
`FlopN({otdb_bucket50,otdb_affineIndex50,otdb_N_count50},  valid_L && otdbOnehot[50], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[51], valid_L && otdbOnehot[51], valid_S && bucket_S == otdb_bucket51)
`FlopN({otdb_bucket51,otdb_affineIndex51,otdb_N_count51},  valid_L && otdbOnehot[51], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[52], valid_L && otdbOnehot[52], valid_S && bucket_S == otdb_bucket52)
`FlopN({otdb_bucket52,otdb_affineIndex52,otdb_N_count52},  valid_L && otdbOnehot[52], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[53], valid_L && otdbOnehot[53], valid_S && bucket_S == otdb_bucket53)
`FlopN({otdb_bucket53,otdb_affineIndex53,otdb_N_count53},  valid_L && otdbOnehot[53], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[54], valid_L && otdbOnehot[54], valid_S && bucket_S == otdb_bucket54)
`FlopN({otdb_bucket54,otdb_affineIndex54,otdb_N_count54},  valid_L && otdbOnehot[54], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[55], valid_L && otdbOnehot[55], valid_S && bucket_S == otdb_bucket55)
`FlopN({otdb_bucket55,otdb_affineIndex55,otdb_N_count55},  valid_L && otdbOnehot[55], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[56], valid_L && otdbOnehot[56], valid_S && bucket_S == otdb_bucket56)
`FlopN({otdb_bucket56,otdb_affineIndex56,otdb_N_count56},  valid_L && otdbOnehot[56], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[57], valid_L && otdbOnehot[57], valid_S && bucket_S == otdb_bucket57)
`FlopN({otdb_bucket57,otdb_affineIndex57,otdb_N_count57},  valid_L && otdbOnehot[57], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[58], valid_L && otdbOnehot[58], valid_S && bucket_S == otdb_bucket58)
`FlopN({otdb_bucket58,otdb_affineIndex58,otdb_N_count58},  valid_L && otdbOnehot[58], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[59], valid_L && otdbOnehot[59], valid_S && bucket_S == otdb_bucket59)
`FlopN({otdb_bucket59,otdb_affineIndex59,otdb_N_count59},  valid_L && otdbOnehot[59], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[60], valid_L && otdbOnehot[60], valid_S && bucket_S == otdb_bucket60)
`FlopN({otdb_bucket60,otdb_affineIndex60,otdb_N_count60},  valid_L && otdbOnehot[60], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[61], valid_L && otdbOnehot[61], valid_S && bucket_S == otdb_bucket61)
`FlopN({otdb_bucket61,otdb_affineIndex61,otdb_N_count61},  valid_L && otdbOnehot[61], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[62], valid_L && otdbOnehot[62], valid_S && bucket_S == otdb_bucket62)
`FlopN({otdb_bucket62,otdb_affineIndex62,otdb_N_count62},  valid_L && otdbOnehot[62], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[63], valid_L && otdbOnehot[63], valid_S && bucket_S == otdb_bucket63)
`FlopN({otdb_bucket63,otdb_affineIndex63,otdb_N_count63},  valid_L && otdbOnehot[63], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[64], valid_L && otdbOnehot[64], valid_S && bucket_S == otdb_bucket64)
`FlopN({otdb_bucket64,otdb_affineIndex64,otdb_N_count64},  valid_L && otdbOnehot[64], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[65], valid_L && otdbOnehot[65], valid_S && bucket_S == otdb_bucket65)
`FlopN({otdb_bucket65,otdb_affineIndex65,otdb_N_count65},  valid_L && otdbOnehot[65], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[66], valid_L && otdbOnehot[66], valid_S && bucket_S == otdb_bucket66)
`FlopN({otdb_bucket66,otdb_affineIndex66,otdb_N_count66},  valid_L && otdbOnehot[66], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[67], valid_L && otdbOnehot[67], valid_S && bucket_S == otdb_bucket67)
`FlopN({otdb_bucket67,otdb_affineIndex67,otdb_N_count67},  valid_L && otdbOnehot[67], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[68], valid_L && otdbOnehot[68], valid_S && bucket_S == otdb_bucket68)
`FlopN({otdb_bucket68,otdb_affineIndex68,otdb_N_count68},  valid_L && otdbOnehot[68], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[69], valid_L && otdbOnehot[69], valid_S && bucket_S == otdb_bucket69)
`FlopN({otdb_bucket69,otdb_affineIndex69,otdb_N_count69},  valid_L && otdbOnehot[69], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[70], valid_L && otdbOnehot[70], valid_S && bucket_S == otdb_bucket70)
`FlopN({otdb_bucket70,otdb_affineIndex70,otdb_N_count70},  valid_L && otdbOnehot[70], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[71], valid_L && otdbOnehot[71], valid_S && bucket_S == otdb_bucket71)
`FlopN({otdb_bucket71,otdb_affineIndex71,otdb_N_count71},  valid_L && otdbOnehot[71], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[72], valid_L && otdbOnehot[72], valid_S && bucket_S == otdb_bucket72)
`FlopN({otdb_bucket72,otdb_affineIndex72,otdb_N_count72},  valid_L && otdbOnehot[72], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[73], valid_L && otdbOnehot[73], valid_S && bucket_S == otdb_bucket73)
`FlopN({otdb_bucket73,otdb_affineIndex73,otdb_N_count73},  valid_L && otdbOnehot[73], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[74], valid_L && otdbOnehot[74], valid_S && bucket_S == otdb_bucket74)
`FlopN({otdb_bucket74,otdb_affineIndex74,otdb_N_count74},  valid_L && otdbOnehot[74], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[75], valid_L && otdbOnehot[75], valid_S && bucket_S == otdb_bucket75)
`FlopN({otdb_bucket75,otdb_affineIndex75,otdb_N_count75},  valid_L && otdbOnehot[75], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[76], valid_L && otdbOnehot[76], valid_S && bucket_S == otdb_bucket76)
`FlopN({otdb_bucket76,otdb_affineIndex76,otdb_N_count76},  valid_L && otdbOnehot[76], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[77], valid_L && otdbOnehot[77], valid_S && bucket_S == otdb_bucket77)
`FlopN({otdb_bucket77,otdb_affineIndex77,otdb_N_count77},  valid_L && otdbOnehot[77], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[78], valid_L && otdbOnehot[78], valid_S && bucket_S == otdb_bucket78)
`FlopN({otdb_bucket78,otdb_affineIndex78,otdb_N_count78},  valid_L && otdbOnehot[78], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[79], valid_L && otdbOnehot[79], valid_S && bucket_S == otdb_bucket79)
`FlopN({otdb_bucket79,otdb_affineIndex79,otdb_N_count79},  valid_L && otdbOnehot[79], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[80], valid_L && otdbOnehot[80], valid_S && bucket_S == otdb_bucket80)
`FlopN({otdb_bucket80,otdb_affineIndex80,otdb_N_count80},  valid_L && otdbOnehot[80], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[81], valid_L && otdbOnehot[81], valid_S && bucket_S == otdb_bucket81)
`FlopN({otdb_bucket81,otdb_affineIndex81,otdb_N_count81},  valid_L && otdbOnehot[81], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[82], valid_L && otdbOnehot[82], valid_S && bucket_S == otdb_bucket82)
`FlopN({otdb_bucket82,otdb_affineIndex82,otdb_N_count82},  valid_L && otdbOnehot[82], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[83], valid_L && otdbOnehot[83], valid_S && bucket_S == otdb_bucket83)
`FlopN({otdb_bucket83,otdb_affineIndex83,otdb_N_count83},  valid_L && otdbOnehot[83], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[84], valid_L && otdbOnehot[84], valid_S && bucket_S == otdb_bucket84)
`FlopN({otdb_bucket84,otdb_affineIndex84,otdb_N_count84},  valid_L && otdbOnehot[84], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[85], valid_L && otdbOnehot[85], valid_S && bucket_S == otdb_bucket85)
`FlopN({otdb_bucket85,otdb_affineIndex85,otdb_N_count85},  valid_L && otdbOnehot[85], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[86], valid_L && otdbOnehot[86], valid_S && bucket_S == otdb_bucket86)
`FlopN({otdb_bucket86,otdb_affineIndex86,otdb_N_count86},  valid_L && otdbOnehot[86], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[87], valid_L && otdbOnehot[87], valid_S && bucket_S == otdb_bucket87)
`FlopN({otdb_bucket87,otdb_affineIndex87,otdb_N_count87},  valid_L && otdbOnehot[87], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[88], valid_L && otdbOnehot[88], valid_S && bucket_S == otdb_bucket88)
`FlopN({otdb_bucket88,otdb_affineIndex88,otdb_N_count88},  valid_L && otdbOnehot[88], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[89], valid_L && otdbOnehot[89], valid_S && bucket_S == otdb_bucket89)
`FlopN({otdb_bucket89,otdb_affineIndex89,otdb_N_count89},  valid_L && otdbOnehot[89], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[90], valid_L && otdbOnehot[90], valid_S && bucket_S == otdb_bucket90)
`FlopN({otdb_bucket90,otdb_affineIndex90,otdb_N_count90},  valid_L && otdbOnehot[90], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[91], valid_L && otdbOnehot[91], valid_S && bucket_S == otdb_bucket91)
`FlopN({otdb_bucket91,otdb_affineIndex91,otdb_N_count91},  valid_L && otdbOnehot[91], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[92], valid_L && otdbOnehot[92], valid_S && bucket_S == otdb_bucket92)
`FlopN({otdb_bucket92,otdb_affineIndex92,otdb_N_count92},  valid_L && otdbOnehot[92], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[93], valid_L && otdbOnehot[93], valid_S && bucket_S == otdb_bucket93)
`FlopN({otdb_bucket93,otdb_affineIndex93,otdb_N_count93},  valid_L && otdbOnehot[93], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[94], valid_L && otdbOnehot[94], valid_S && bucket_S == otdb_bucket94)
`FlopN({otdb_bucket94,otdb_affineIndex94,otdb_N_count94},  valid_L && otdbOnehot[94], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[95], valid_L && otdbOnehot[95], valid_S && bucket_S == otdb_bucket95)
`FlopN({otdb_bucket95,otdb_affineIndex95,otdb_N_count95},  valid_L && otdbOnehot[95], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[96], valid_L && otdbOnehot[96], valid_S && bucket_S == otdb_bucket96)
`FlopN({otdb_bucket96,otdb_affineIndex96,otdb_N_count96},  valid_L && otdbOnehot[96], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[97], valid_L && otdbOnehot[97], valid_S && bucket_S == otdb_bucket97)
`FlopN({otdb_bucket97,otdb_affineIndex97,otdb_N_count97},  valid_L && otdbOnehot[97], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[98], valid_L && otdbOnehot[98], valid_S && bucket_S == otdb_bucket98)
`FlopN({otdb_bucket98,otdb_affineIndex98,otdb_N_count98},  valid_L && otdbOnehot[98], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[99], valid_L && otdbOnehot[99], valid_S && bucket_S == otdb_bucket99)
`FlopN({otdb_bucket99,otdb_affineIndex99,otdb_N_count99},  valid_L && otdbOnehot[99], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[100], valid_L && otdbOnehot[100], valid_S && bucket_S == otdb_bucket100)
`FlopN({otdb_bucket100,otdb_affineIndex100,otdb_N_count100},  valid_L && otdbOnehot[100], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[101], valid_L && otdbOnehot[101], valid_S && bucket_S == otdb_bucket101)
`FlopN({otdb_bucket101,otdb_affineIndex101,otdb_N_count101},  valid_L && otdbOnehot[101], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[102], valid_L && otdbOnehot[102], valid_S && bucket_S == otdb_bucket102)
`FlopN({otdb_bucket102,otdb_affineIndex102,otdb_N_count102},  valid_L && otdbOnehot[102], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[103], valid_L && otdbOnehot[103], valid_S && bucket_S == otdb_bucket103)
`FlopN({otdb_bucket103,otdb_affineIndex103,otdb_N_count103},  valid_L && otdbOnehot[103], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[104], valid_L && otdbOnehot[104], valid_S && bucket_S == otdb_bucket104)
`FlopN({otdb_bucket104,otdb_affineIndex104,otdb_N_count104},  valid_L && otdbOnehot[104], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[105], valid_L && otdbOnehot[105], valid_S && bucket_S == otdb_bucket105)
`FlopN({otdb_bucket105,otdb_affineIndex105,otdb_N_count105},  valid_L && otdbOnehot[105], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[106], valid_L && otdbOnehot[106], valid_S && bucket_S == otdb_bucket106)
`FlopN({otdb_bucket106,otdb_affineIndex106,otdb_N_count106},  valid_L && otdbOnehot[106], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[107], valid_L && otdbOnehot[107], valid_S && bucket_S == otdb_bucket107)
`FlopN({otdb_bucket107,otdb_affineIndex107,otdb_N_count107},  valid_L && otdbOnehot[107], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[108], valid_L && otdbOnehot[108], valid_S && bucket_S == otdb_bucket108)
`FlopN({otdb_bucket108,otdb_affineIndex108,otdb_N_count108},  valid_L && otdbOnehot[108], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[109], valid_L && otdbOnehot[109], valid_S && bucket_S == otdb_bucket109)
`FlopN({otdb_bucket109,otdb_affineIndex109,otdb_N_count109},  valid_L && otdbOnehot[109], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[110], valid_L && otdbOnehot[110], valid_S && bucket_S == otdb_bucket110)
`FlopN({otdb_bucket110,otdb_affineIndex110,otdb_N_count110},  valid_L && otdbOnehot[110], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[111], valid_L && otdbOnehot[111], valid_S && bucket_S == otdb_bucket111)
`FlopN({otdb_bucket111,otdb_affineIndex111,otdb_N_count111},  valid_L && otdbOnehot[111], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[112], valid_L && otdbOnehot[112], valid_S && bucket_S == otdb_bucket112)
`FlopN({otdb_bucket112,otdb_affineIndex112,otdb_N_count112},  valid_L && otdbOnehot[112], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[113], valid_L && otdbOnehot[113], valid_S && bucket_S == otdb_bucket113)
`FlopN({otdb_bucket113,otdb_affineIndex113,otdb_N_count113},  valid_L && otdbOnehot[113], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[114], valid_L && otdbOnehot[114], valid_S && bucket_S == otdb_bucket114)
`FlopN({otdb_bucket114,otdb_affineIndex114,otdb_N_count114},  valid_L && otdbOnehot[114], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[115], valid_L && otdbOnehot[115], valid_S && bucket_S == otdb_bucket115)
`FlopN({otdb_bucket115,otdb_affineIndex115,otdb_N_count115},  valid_L && otdbOnehot[115], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[116], valid_L && otdbOnehot[116], valid_S && bucket_S == otdb_bucket116)
`FlopN({otdb_bucket116,otdb_affineIndex116,otdb_N_count116},  valid_L && otdbOnehot[116], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[117], valid_L && otdbOnehot[117], valid_S && bucket_S == otdb_bucket117)
`FlopN({otdb_bucket117,otdb_affineIndex117,otdb_N_count117},  valid_L && otdbOnehot[117], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[118], valid_L && otdbOnehot[118], valid_S && bucket_S == otdb_bucket118)
`FlopN({otdb_bucket118,otdb_affineIndex118,otdb_N_count118},  valid_L && otdbOnehot[118], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[119], valid_L && otdbOnehot[119], valid_S && bucket_S == otdb_bucket119)
`FlopN({otdb_bucket119,otdb_affineIndex119,otdb_N_count119},  valid_L && otdbOnehot[119], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[120], valid_L && otdbOnehot[120], valid_S && bucket_S == otdb_bucket120)
`FlopN({otdb_bucket120,otdb_affineIndex120,otdb_N_count120},  valid_L && otdbOnehot[120], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[121], valid_L && otdbOnehot[121], valid_S && bucket_S == otdb_bucket121)
`FlopN({otdb_bucket121,otdb_affineIndex121,otdb_N_count121},  valid_L && otdbOnehot[121], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[122], valid_L && otdbOnehot[122], valid_S && bucket_S == otdb_bucket122)
`FlopN({otdb_bucket122,otdb_affineIndex122,otdb_N_count122},  valid_L && otdbOnehot[122], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[123], valid_L && otdbOnehot[123], valid_S && bucket_S == otdb_bucket123)
`FlopN({otdb_bucket123,otdb_affineIndex123,otdb_N_count123},  valid_L && otdbOnehot[123], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[124], valid_L && otdbOnehot[124], valid_S && bucket_S == otdb_bucket124)
`FlopN({otdb_bucket124,otdb_affineIndex124,otdb_N_count124},  valid_L && otdbOnehot[124], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[125], valid_L && otdbOnehot[125], valid_S && bucket_S == otdb_bucket125)
`FlopN({otdb_bucket125,otdb_affineIndex125,otdb_N_count125},  valid_L && otdbOnehot[125], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[126], valid_L && otdbOnehot[126], valid_S && bucket_S == otdb_bucket126)
`FlopN({otdb_bucket126,otdb_affineIndex126,otdb_N_count126},  valid_L && otdbOnehot[126], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[127], valid_L && otdbOnehot[127], valid_S && bucket_S == otdb_bucket127)
`FlopN({otdb_bucket127,otdb_affineIndex127,otdb_N_count127},  valid_L && otdbOnehot[127], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[128], valid_L && otdbOnehot[128], valid_S && bucket_S == otdb_bucket128)
`FlopN({otdb_bucket128,otdb_affineIndex128,otdb_N_count128},  valid_L && otdbOnehot[128], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[129], valid_L && otdbOnehot[129], valid_S && bucket_S == otdb_bucket129)
`FlopN({otdb_bucket129,otdb_affineIndex129,otdb_N_count129},  valid_L && otdbOnehot[129], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[130], valid_L && otdbOnehot[130], valid_S && bucket_S == otdb_bucket130)
`FlopN({otdb_bucket130,otdb_affineIndex130,otdb_N_count130},  valid_L && otdbOnehot[130], {bucket_L,affineIndex_L,N_count_L})
`FlopSC0(otdbValid[131], valid_L && otdbOnehot[131], valid_S && bucket_S == otdb_bucket131)
`FlopN({otdb_bucket131,otdb_affineIndex131,otdb_N_count131},  valid_L && otdbOnehot[131], {bucket_L,affineIndex_L,N_count_L})
`Flop(otdbNum, valid_L != valid_S, valid_S ? otdbNum - 1 : otdbNum + 1)

`assert_clk(otdbNum == num132(otdbValid))

logic [`LOG2(132+1)-1:0] debug_otdbvalid_num;
assign debug_otdbvalid_num = num132(otdbValid);

    sz#(
        .DATA(132)
    ) sz_otdb(
        .in(otdbValid),
        .out(otdbOnehot)
    );
////////////////////////////////////////////////////////////////////////////////////////////////////////
//bucket hyper pipe
//outstanding buffer for bucketControl
//G6->H->I->K->P->S
`Flop(bucket_hyperpipe_valid[  0], 1, valid_L)
`FlopN(bucket_hyperpipe      [  0], 1, bucket_L)

generate 
for(i=0;i<bucket_hp_length-1;i=i+1)begin
`Flop(bucket_hyperpipe_valid[i+1], 1, bucket_hyperpipe_valid[i])
`FlopN(bucket_hyperpipe      [i+1], 1, bucket_hyperpipe      [i])
end
endgenerate

assign real_bucket_S = bucket_hyperpipe[bucket_hp_length-1];

`Flop(valid_S_r, 1, valid_S)
`FlopN(bucket_S_r, 1, bucket_S)

`assert_clk(valid_S |-> real_bucket_S == bucket_S)

//check L with outstanding buffer
generate 
for(i=0;i<bucket_hp_length;i=i+1)begin        
    assign conflictInOtdBuffer_L[i] = bucket_hyperpipe_valid[i] && bucket_L == bucket_hyperpipe[i];
end
endgenerate
`assert_clk(valid_L |-> !(|conflictInOtdBuffer_L))
//

endmodule
