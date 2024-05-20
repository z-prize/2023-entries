`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

module zprize_msm_point_mux import zprize_param::*; import zprize_param::* ; (
    input logic                       pointCreditOk0_0,
    input logic                       pointCreditOk1_0,
    input logic                       pointCreditOk2_0,
    input logic                       pointCreditOk3_0,
    input logic                       pointRamBufferFifo0_rvalid,
    input logic                       pointRamBufferFifo0_rready,
    input logic   [1152-1:0]     pointRamBufferFifo0_rdata,    
    input logic                       pointCreditOk0_1,
    input logic                       pointCreditOk1_1,
    input logic                       pointCreditOk2_1,
    input logic                       pointCreditOk3_1,
    input logic                       pointRamBufferFifo1_rvalid,
    input logic                       pointRamBufferFifo1_rready,
    input logic   [1152-1:0]     pointRamBufferFifo1_rdata,    
    input logic                       pointCreditOk0_2,
    input logic                       pointCreditOk1_2,
    input logic                       pointCreditOk2_2,
    input logic                       pointCreditOk3_2,
    input logic                       pointRamBufferFifo2_rvalid,
    input logic                       pointRamBufferFifo2_rready,
    input logic   [1152-1:0]     pointRamBufferFifo2_rdata,    
    input logic                       pointCreditOk0_3,
    input logic                       pointCreditOk1_3,
    input logic                       pointCreditOk2_3,
    input logic                       pointCreditOk3_3,
    input logic                       pointRamBufferFifo3_rvalid,
    input logic                       pointRamBufferFifo3_rready,
    input logic   [1152-1:0]     pointRamBufferFifo3_rdata,    
    output logic   [`LOG2(4)-1:0] switch,
    output logic                       pointPipeValid,
    output logic   [1152-1:0]     pointPipeData,
    output logic                       switch_ready,
    input logic                       clk,
    input logic                       rstN
);

////////////////////////////////////////////////////////////////////////////////////////////////////////
//mux point from 4 channel into one
////////////////////////////////////////////////////////////////////////////////////////////////////////
    logic                       switch_valid;
////////////////////////////////////////////////////////////////////////////////////////////////////////
//mux point from 4 channel into one
////////////////////////////////////////////////////////////////////////////////////////////////////////
`FlopAll(switch, switch_valid && switch_ready, switch + 1, clk, rstN, 0)    //valid means data_valid, ready means core_CreditOK

assign switch_ready = 
                    pointCreditOk0_2 && 
                    pointCreditOk1_2 && 
                    pointCreditOk2_2 && 
                    pointCreditOk3_2 && 
                    1;

    always@*
        case(switch)
        0: begin
            pointPipeValid = pointRamBufferFifo0_rvalid && pointRamBufferFifo0_rready;
            pointPipeData  = pointRamBufferFifo0_rdata;
            switch_valid   = pointRamBufferFifo0_rvalid;
        end
        1: begin
            pointPipeValid = pointRamBufferFifo1_rvalid && pointRamBufferFifo1_rready;
            pointPipeData  = pointRamBufferFifo1_rdata;
            switch_valid   = pointRamBufferFifo1_rvalid;
        end
        2: begin
            pointPipeValid = pointRamBufferFifo2_rvalid && pointRamBufferFifo2_rready;
            pointPipeData  = pointRamBufferFifo2_rdata;
            switch_valid   = pointRamBufferFifo2_rvalid;
        end
        3: begin
            pointPipeValid = pointRamBufferFifo3_rvalid && pointRamBufferFifo3_rready;
            pointPipeData  = pointRamBufferFifo3_rdata;
            switch_valid   = pointRamBufferFifo3_rvalid;
        end
        default: begin
            pointPipeValid = 'bx;
            pointPipeData  = 'bx;
            switch_valid   = 'bx;
        end
        endcase

endmodule
