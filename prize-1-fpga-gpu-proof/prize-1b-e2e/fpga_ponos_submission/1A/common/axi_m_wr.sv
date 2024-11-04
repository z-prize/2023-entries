

`ifndef AXI_M_WR_SV
`define AXI_M_WR_SV

module axi_m_wr #(
  parameter int C_ADDR_W = 64,
  parameter int C_DATA_W = 32,
  parameter int C_MAX_LENGTH_W = 32,
  parameter int C_BURST_LEN = 256,  // Max AXI burst length for read commands
  parameter int C_ADDR_INCR = 1,  // Address increment in burst(transaction), should be power of 2!
  parameter int C_BR_EN = 0,  // Bit reverse address generation enable
  parameter int C_LOG_NUM_TRANSACTIONS = 12
) (
  // System signals
  input                       clk,
  input                       rst_n,
  // Control interface
  input                       ctrl_start_i,
  input  [      C_ADDR_W-1:0] ctrl_offset_i,
  input  [C_MAX_LENGTH_W-1:0] ctrl_size_i,
  output                      ctrl_done_o,
  // AXI4-Stream interface
  input                       axis_s_tvalid_i,
  input  [      C_DATA_W-1:0] axis_s_tdata_i,
  output                      axis_s_tready_o,
  // AXI4 master interface 
  output [      C_ADDR_W-1:0] axi_m_awaddr_o,
  output [               7:0] axi_m_awlen_o,
  output [               2:0] axi_m_awsize_o,
  output                      axi_m_awvalid_o,
  input                       axi_m_awready_i,
  output [      C_DATA_W-1:0] axi_m_wdata_o,
  output [    C_DATA_W/8-1:0] axi_m_wstrb_o,
  output                      axi_m_wlast_o,
  output                      axi_m_wvalid_o,
  input                       axi_m_wready_i,
  input  [               1:0] axi_m_bresp_i,
  input                       axi_m_bvalid_i,
  output                      axi_m_bready_o
);

  /////////////////////////////////////////////////////////////////////////////
  // Local Parameters
  /////////////////////////////////////////////////////////////////////////////
  localparam int LP_LOG_MAX_W_TO_AW = 8;  // Allow up to 256 outstanding w to aw transactions
  localparam int LP_LOG_BURST_LEN = $clog2(C_BURST_LEN);
  localparam int LP_TRANSACTION_CNTR_W = C_MAX_LENGTH_W - LP_LOG_BURST_LEN;
  localparam int LP_LOG_ADDR_INCR = $clog2(C_ADDR_INCR);
  localparam int LP_LOG_DATA_W = $clog2(C_DATA_W);
  localparam int LP_LOG_TRANSACTION_OFFSET = LP_LOG_BURST_LEN + LP_LOG_ADDR_INCR + LP_LOG_DATA_W - 3;
  localparam int LP_LOG_NUM_TRANSACTIONS = (C_BR_EN) ? C_LOG_NUM_TRANSACTIONS : LP_TRANSACTION_CNTR_W;

  /////////////////////////////////////////////////////////////////////////////
  // Variables
  /////////////////////////////////////////////////////////////////////////////
  // Control logic
  logic [  LP_TRANSACTION_CNTR_W-1:0] num_full_bursts;
  logic                               num_partial_bursts;
  logic                               single_transaction;
  logic [  LP_TRANSACTION_CNTR_W-1:0] num_transactions;
  logic                               has_partial_burst;
  logic [       LP_LOG_BURST_LEN-1:0] final_burst_len;
  logic                               start = 1'b0;
  // AXI Write Data Channel
  logic                               wxfer;  // Unregistered write data transfer
  logic                               wfirst = 1'b1;
  logic                               wfirst_q = 1'b0;
  logic                               wfirst_pulse = 1'b0;
  logic                               wlast_r;
  logic                               load_burst_cntr;
  logic [  LP_TRANSACTION_CNTR_W-1:0] w_transactions_to_go;
  logic                               w_final_transaction;
  logic                               w_almost_final_transaction = 1'b0;
  // AXI Write Address Channel
  logic                               awxfer;
  logic                               aw_idle = 1'b1;
  logic                               aw_done;
  logic                               awvalid_r = 1'b0;
  logic [LP_LOG_NUM_TRANSACTIONS-1:0] awxfer_num;
  logic [LP_LOG_NUM_TRANSACTIONS-1:0] awxfer_num_br;
  logic [               C_ADDR_W-1:0] ctrl_offset_r;
  logic [               C_ADDR_W-1:0] addr;
  logic                               stall_aw;
  logic                               aw_final_transaction;
  // AXI Write Response Channel
  logic                               bxfer;
  logic [  LP_TRANSACTION_CNTR_W-1:0] b_transactions_to_go;
  logic                               b_final_transaction;

  /////////////////////////////////////////////////////////////////////////////
  // Control logic 
  /////////////////////////////////////////////////////////////////////////////
  // Count the number of transfers and assert done when the last axi_m_bvalid_i is received.
  assign num_full_bursts    = ctrl_size_i[LP_LOG_BURST_LEN+:C_MAX_LENGTH_W-LP_LOG_BURST_LEN];
  assign num_partial_bursts = (ctrl_size_i[0+:LP_LOG_BURST_LEN]) ? 1'b1 : 1'b0;

  // Special case if there is only 1 AXI transaction. 
  assign single_transaction = (num_transactions == {LP_TRANSACTION_CNTR_W{1'b0}}) ? 1'b1 : 1'b0;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      num_transactions  <= '0;
      has_partial_burst <= 1'b0;
      final_burst_len   <= '0;
    end else if (ctrl_start_i) begin
      num_transactions  <= (num_partial_bursts == 1'b0) ? num_full_bursts - 1'b1 : num_full_bursts;
      has_partial_burst <= num_partial_bursts;
      final_burst_len   <= ctrl_size_i[0+:LP_LOG_BURST_LEN] - 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    start <= ctrl_start_i;
  end

  assign ctrl_done_o = bxfer & b_final_transaction;

  /////////////////////////////////////////////////////////////////////////////
  // AXI Write Data Channel
  /////////////////////////////////////////////////////////////////////////////
  assign axi_m_wdata_o = axis_s_tdata_i;
  assign axi_m_wvalid_o = axis_s_tvalid_i;
  assign axi_m_wstrb_o = {(C_DATA_W / 8) {1'b1}};
  assign axi_m_wlast_o = wlast_r;
  assign axis_s_tready_o = axi_m_wready_i;

  assign wxfer = axi_m_wvalid_o & axi_m_wready_i;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wfirst <= 1'b1;
    end else begin
      wfirst <= (wxfer) ? wlast_r : wfirst;
    end
  end

  always_ff @(posedge clk) begin
    wfirst_q     <= axi_m_wvalid_o & wfirst;
    wfirst_pulse <= axi_m_wvalid_o & wfirst & ~wfirst_q;
  end

  // Load burst counter with partial burst if on final transaction or if there is only 1 transaction.
  assign load_burst_cntr = (wxfer & wlast_r & w_almost_final_transaction) | (start & single_transaction);

  // Decrement counter that counts transfers in a burst.
  counter_bidir #(
    .C_W   (LP_LOG_BURST_LEN),
    .C_INIT({LP_LOG_BURST_LEN{1'b1}})
  ) i_burst_cntr (
    .clk,
    .rst_n,
    .load_i      (load_burst_cntr),
    .incr_i      (1'b0),
    .decr_i      (wxfer),
    .load_value_i(final_burst_len),
    .count_o     (),
    .is_zero_o   (wlast_r)
  );

  // Decrement counter that counts bursts (transactions).
  counter_bidir #(
    .C_W   (LP_TRANSACTION_CNTR_W),
    .C_INIT({LP_TRANSACTION_CNTR_W{1'b0}})
  ) i_w_trans_cntr (
    .clk,
    .rst_n,
    .load_i      (start),
    .incr_i      (1'b0),
    .decr_i      (wxfer & wlast_r),
    .load_value_i(num_transactions),
    .count_o     (w_transactions_to_go),
    .is_zero_o   (w_final_transaction)
  );

  always_ff @(posedge clk) begin
    w_almost_final_transaction <= (w_transactions_to_go == 'd1) ? 1'b1 : 1'b0;
  end

  /////////////////////////////////////////////////////////////////////////////
  // AXI Write Address Channel
  /////////////////////////////////////////////////////////////////////////////
  // The address channel samples the data channel and send out transactions when 
  // first beat of axi_m_wdata is asserted. This ensures that address requests 
  // are not sent without data on the way.
  assign axi_m_awvalid_o = awvalid_r;
  assign axi_m_awaddr_o = addr;
  assign axi_m_awlen_o = (aw_final_transaction || (start && single_transaction)) ? final_burst_len : C_BURST_LEN - 1;
  assign axi_m_awsize_o = $clog2(C_DATA_W / 8);

  assign awxfer = awvalid_r & axi_m_awready_i;

  // When aw_idle, there are no address write requests to issue.
  assign aw_done = aw_final_transaction & awxfer;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      aw_idle <= 1'b1;
    end else begin
      aw_idle <= (start) ? 1'b0 : (aw_done) ? 1'b1 : aw_idle;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      awvalid_r <= 1'b0;
    end else begin
      awvalid_r <= (!aw_idle && !stall_aw  && !awvalid_r) ? 1'b1 : 
                   (axi_m_awready_i) ? 1'b0 : awvalid_r;
    end
  end

  // Calculate next address after each write address is sent
  assign awxfer_num_br = (C_BR_EN) ? {<<{awxfer_num}} : awxfer_num;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      awxfer_num    <= '0;
      ctrl_offset_r <= '0;
      addr          <= '0;
    end else if (ctrl_start_i) begin
      awxfer_num    <= 'd1;
      ctrl_offset_r <= ctrl_offset_i;
      addr          <= ctrl_offset_i;
    end else if (awxfer) begin
      awxfer_num <= awxfer_num + 1;
      addr       <= ctrl_offset_r + (awxfer_num_br << LP_LOG_TRANSACTION_OFFSET);
    end
  end

  // Counter for outstanding address write requests, increments on each write data burst,
  // decrements on each address write request. Stalls after 256 outstanding 
  // address write requests so that AXI Slave FIFO won't overflow.
  counter_bidir #(
    .C_W   (LP_LOG_MAX_W_TO_AW),
    .C_INIT({LP_LOG_MAX_W_TO_AW{1'b0}})
  ) i_w_to_aw_cntr (
    .clk,
    .rst_n,
    .load_i      (1'b0),
    .incr_i      (wfirst_pulse),
    .decr_i      (awxfer),
    .load_value_i({LP_LOG_MAX_W_TO_AW{1'b0}}),
    .count_o     (),
    .is_zero_o   (stall_aw)
  );

  // Decrement counter that counts address write requests.
  counter_bidir #(
    .C_W   (LP_TRANSACTION_CNTR_W),
    .C_INIT({LP_TRANSACTION_CNTR_W{1'b0}})
  ) i_aw_trans_cntr (
    .clk,
    .rst_n,
    .load_i      (start),
    .incr_i      (1'b0),
    .decr_i      (awxfer),
    .load_value_i(num_transactions),
    .count_o     (),
    .is_zero_o   (aw_final_transaction)
  );

  /////////////////////////////////////////////////////////////////////////////
  // AXI Write Response Channel
  /////////////////////////////////////////////////////////////////////////////
  assign axi_m_bready_o = 1'b1;
  assign bxfer = axi_m_bready_o & axi_m_bvalid_i;

  // Decrement counter that counts write transactions responses.
  counter_bidir #(
    .C_W   (LP_TRANSACTION_CNTR_W),
    .C_INIT({LP_TRANSACTION_CNTR_W{1'b0}})
  ) i_b_trans_cntr (
    .clk,
    .rst_n,
    .load_i      (start),
    .incr_i      (1'b0),
    .decr_i      (bxfer),
    .load_value_i(num_transactions),
    .count_o     (b_transactions_to_go),
    .is_zero_o   (b_final_transaction)
  );

endmodule : axi_m_wr

`endif  // AXI_M_WR_SV
