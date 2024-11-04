///////////////////////////////////////////////////////////////////////////////////////////////////
// (C) Copyright Ponos Technology 2024
// All Rights Reserved
// *** Ponos Technology Confidential ***
//
// Description: it schedules the write of bucket sets in the proper order
///////////////////////////////////////////////////////////////////////////////////////////////////

`ifndef BUCKET_WR_SCHEDULER_SV
`define BUCKET_WR_SCHEDULER_SV

module bucket_wr_scheduler #(
  parameter unsigned P_NUM_ACCU   = 3,
  parameter unsigned P_DATA_PNT_W = 377,
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // Bucket streams from the accumulator(s)
  input        [P_DATA_PNT_W-1:0] s_data_i [P_NUM_ACCU-1:0],
  input        [  P_NUM_ACCU-1:0] s_valid_i,
  input        [  P_NUM_ACCU-1:0] s_last_i,
  output logic [  P_NUM_ACCU-1:0] s_ready_o,
  // Bucket write streams to HOST[0]
  output logic [P_DATA_PNT_W-1:0] m_data_o,
  output logic                    m_valid_o,
  input  logic                    m_ready_i,

  output logic start_host_wr_o
);

  // FSM state enumeration
  typedef enum logic [1:0] {
    FSM_IDLE = 2'b00,
    FSM_INIT = 2'b01,
    FSM_RUN  = 2'b10
  } fsm_state_e;
  fsm_state_e fsm_state_q, fsm_state_d;

  logic [$clog2(P_NUM_ACCU):0] accu_cnt_d;
  logic [$clog2(P_NUM_ACCU):0] accu_cnt_q;

  logic [P_NUM_ACCU-1:0] accu_done_d;
  logic [P_NUM_ACCU-1:0] accu_done_q;

  logic [3:0] init_cnt_d;
  logic [3:0] init_cnt_q;

  always_comb begin : comb_main_fsm
    // Defaults
    s_ready_o       = '0;
    m_data_o        = '0;
    m_valid_o       = 1'b0;
    start_host_wr_o = 1'b0;
    accu_done_d     = accu_done_q;
    accu_cnt_d      = accu_cnt_q;
    init_cnt_d      = init_cnt_q;
    fsm_state_d     = fsm_state_q;

    unique case (fsm_state_q)
      FSM_IDLE: begin
        accu_done_d = accu_done_q | s_valid_i;
        if (&accu_done_q) begin
          start_host_wr_o = 1'b1;
          init_cnt_d++;
          fsm_state_d = FSM_INIT;
        end
      end
      FSM_INIT: begin
        if (init_cnt_q == '0) begin
          fsm_state_d = FSM_RUN;
        end else begin
          init_cnt_d++;
        end
      end
      FSM_RUN: begin
        if (accu_cnt_q == P_NUM_ACCU) begin
          accu_done_d = '0;
          accu_cnt_d  = '0;
          fsm_state_d = FSM_IDLE;
        end else if (s_valid_i[accu_cnt_q] & m_ready_i) begin
          s_ready_o[accu_cnt_q] = 1'b1;
          m_data_o = s_data_i[accu_cnt_q];
          m_valid_o = s_valid_i[accu_cnt_q];
          if (s_last_i[accu_cnt_q]) begin
            accu_cnt_d++;
          end
        end
      end
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      accu_done_q <= '0;
      accu_cnt_q  <= '0;
      init_cnt_q  <= '0;
      fsm_state_q <= FSM_IDLE;
    end else begin
      accu_done_q <= accu_done_d;
      accu_cnt_q  <= accu_cnt_d;
      init_cnt_q  <= init_cnt_d;
      fsm_state_q <= fsm_state_d;
    end
  end

endmodule
`endif  // BUCKET_WR_SCHEDULER_SV
