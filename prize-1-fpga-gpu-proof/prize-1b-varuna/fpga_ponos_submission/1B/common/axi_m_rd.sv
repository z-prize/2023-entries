

`ifndef AXI_M_RD_SV
`define AXI_M_RD_SV

module axi_m_rd #(
  parameter int C_ADDR_W = 64,
  parameter int C_DATA_W = 32,
  parameter int C_SIZE_W = 32,
  parameter int C_BURST_LEN = 256,  // Max AXI burst length for read commands
  parameter int C_MAX_OUTSTANDING = 3,  // Max number of outstanding read request bursts(transactions)
  parameter int C_ADDR_INCR = 1,  // Address increment in burst(transaction), should be power of 2!
  parameter int C_BR_EN = 0,  // Bit reverse address generation enable
  parameter int C_LOG_NUM_TRANSACTIONS = 12
) (
  // System signals
  input                 clk,
  input                 rst_n,
  // Control signals 
  input                 ctrl_start_i,
  input  [C_ADDR_W-1:0] ctrl_offset_i,
  input  [C_SIZE_W-1:0] ctrl_size_i,
  output                ctrl_done_o,
  // AXI4-Stream master interface 
  input                 axis_m_tready_i,
  output                axis_m_tvalid_o,
  output [C_DATA_W-1:0] axis_m_tdata_o,
  output                axis_m_tlast_o,
  output                axis_m_tdone_o,
  // AXI4 master interface                         
  output                axi_m_arvalid_o,
  input                 axi_m_arready_i,
  output [C_ADDR_W-1:0] axi_m_araddr_o,
  output [         7:0] axi_m_arlen_o,
  output [         2:0] axi_m_arsize_o,
  input                 axi_m_rvalid_i,
  output                axi_m_rready_o,
  input  [C_DATA_W-1:0] axi_m_rdata_i,
  input                 axi_m_rlast_i,
  input  [         1:0] axi_m_rresp_i
);

  ///////////////////////////////////////////////////////////////////////////////
  // Local Parameters
  ///////////////////////////////////////////////////////////////////////////////
  localparam int LP_MAX_OUTSTANDING_CNTR_W = $clog2(C_MAX_OUTSTANDING + 1);
  localparam int LP_LOG_BURST_LEN = $clog2(C_BURST_LEN);
  localparam int LP_TRANSACTION_CNTR_W = C_SIZE_W - LP_LOG_BURST_LEN;
  localparam int LP_LOG_ADDR_INCR = $clog2(C_ADDR_INCR);
  localparam int LP_LOG_DATA_W = $clog2(C_DATA_W);
  localparam int LP_LOG_TRANSACTION_OFFSET = LP_LOG_BURST_LEN + LP_LOG_ADDR_INCR + LP_LOG_DATA_W - 3;
  localparam int LP_LOG_NUM_TRANSACTIONS = (C_BR_EN) ? C_LOG_NUM_TRANSACTIONS : LP_TRANSACTION_CNTR_W;

  ///////////////////////////////////////////////////////////////////////////////
  // Variables
  ///////////////////////////////////////////////////////////////////////////////
  // Control logic
  logic [  LP_TRANSACTION_CNTR_W-1:0] num_full_bursts;
  logic                               num_partial_bursts;
  logic                               single_transaction;
  logic [  LP_TRANSACTION_CNTR_W-1:0] num_transactions;
  logic                               has_partial_burst;
  logic [       LP_LOG_BURST_LEN-1:0] final_burst_len;
  logic                               start = 1'b0;
  logic                               done = 1'b0;
  // AXI Read Address Channel
  logic                               arxfer;
  logic                               ar_done;
  logic                               ar_idle = 1'b1;
  logic                               arvalid_r = 1'b0;
  logic [LP_LOG_NUM_TRANSACTIONS-1:0] arxfer_num;
  logic [LP_LOG_NUM_TRANSACTIONS-1:0] arxfer_num_br;
  logic [               C_ADDR_W-1:0] ctrl_offset_r;
  logic [               C_ADDR_W-1:0] addr;
  logic                               stall_ar;
  logic                               ar_final_transaction;
  // AXI Read Data Channel
  logic                               rxfer;
  logic                               decr_r_transaction_cntr;
  logic [  LP_TRANSACTION_CNTR_W-1:0] r_transactions_to_go;
  logic                               r_final_transaction;
  logic                               r_final_transaction_q;

  ///////////////////////////////////////////////////////////////////////////////
  // Control Logic 
  ///////////////////////////////////////////////////////////////////////////////
  // Determine how many full burst to issue and if there are any partial bursts.
  assign num_full_bursts    = ctrl_size_i[LP_LOG_BURST_LEN+:C_SIZE_W-LP_LOG_BURST_LEN];
  assign num_partial_bursts = ctrl_size_i[0+:LP_LOG_BURST_LEN] ? 1'b1 : 1'b0;

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
    done <= (rxfer && axi_m_rlast_i && r_final_transaction) ? 1'b1 : (ctrl_done_o) ? 1'b0 : done;
    r_final_transaction_q <= r_final_transaction;
  end
  assign ctrl_done_o = done;

  assign axis_m_tdone_o = axi_m_rvalid_i & axi_m_rlast_i & r_final_transaction_q;

  ///////////////////////////////////////////////////////////////////////////////
  // AXI Read Address Channel
  ///////////////////////////////////////////////////////////////////////////////
  assign axi_m_arvalid_o = arvalid_r;
  assign axi_m_araddr_o = addr;
  assign axi_m_arlen_o = (ar_final_transaction || (start & single_transaction)) ? final_burst_len : C_BURST_LEN - 1;
  assign axi_m_arsize_o = $clog2(C_DATA_W / 8);

  assign arxfer = arvalid_r & axi_m_arready_i;

  // When ar_idle, there are no address read requests to issue.
  assign ar_done = ar_final_transaction & arxfer;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ar_idle <= 1'b1;
    end else begin
      ar_idle <= (start) ? 1'b0 : (ar_done) ? 1'b1 : ar_idle;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      arvalid_r <= 1'b0;
    end else begin
      arvalid_r <= (!ar_idle && !stall_ar && !arvalid_r) ? 1'b1 :  
                   (axi_m_arready_i) ? 1'b0 : arvalid_r;
    end
  end

  // Calculate next address after each transaction is issued.
  assign arxfer_num_br = (C_BR_EN) ? {<<{arxfer_num}} : arxfer_num;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      arxfer_num    <= '0;
      ctrl_offset_r <= '0;
      addr          <= '0;
    end else if (ctrl_start_i) begin
      arxfer_num    <= 'd1;
      ctrl_offset_r <= ctrl_offset_i;
      addr          <= ctrl_offset_i;
    end else if (arxfer) begin
      arxfer_num <= arxfer_num + 1;
      addr       <= ctrl_offset_r + (arxfer_num_br << LP_LOG_TRANSACTION_OFFSET);
    end
  end

  // Keeps track of the number of outstanding transactions. Stalls 
  // when the value is reached so that the FIFO won't overflow.
  counter_bidir #(
    .C_W   (LP_MAX_OUTSTANDING_CNTR_W),
    .C_INIT(C_MAX_OUTSTANDING[0+:LP_MAX_OUTSTANDING_CNTR_W])
  ) i_ar_to_r_trans_cntr (
    .clk,
    .rst_n,
    .load_i      (1'b0),
    .incr_i      (rxfer & axi_m_rlast_i),
    .decr_i      (arxfer),
    .load_value_i({LP_MAX_OUTSTANDING_CNTR_W{1'b0}}),
    .count_o     (),
    .is_zero_o   (stall_ar)
  );

  // Decrement counter that counts address read requests.
  counter_bidir #(
    .C_W   (LP_TRANSACTION_CNTR_W),
    .C_INIT({LP_TRANSACTION_CNTR_W{1'b0}})
  ) i_ar_trans_cntr (
    .clk,
    .rst_n,
    .load_i      (start),
    .incr_i      (1'b0),
    .decr_i      (arxfer),
    .load_value_i(num_transactions),
    .count_o     (),
    .is_zero_o   (ar_final_transaction)
  );

  ///////////////////////////////////////////////////////////////////////////////
  // AXI Read Data Channel
  ///////////////////////////////////////////////////////////////////////////////
  assign axis_m_tlast_o          = axi_m_rlast_i;
  assign axis_m_tvalid_o         = axi_m_rvalid_i;
  assign axis_m_tdata_o          = axi_m_rdata_i;
  assign axi_m_rready_o          = axis_m_tready_i;
  assign rxfer                   = axi_m_rready_o & axi_m_rvalid_i;
  assign decr_r_transaction_cntr = rxfer & axi_m_rlast_i;

  // Decrement counter after each read data transactions is received.
  counter_bidir #(
    .C_W   (LP_TRANSACTION_CNTR_W),
    .C_INIT({LP_TRANSACTION_CNTR_W{1'b0}})
  ) i_r_trans_cntr (
    .clk,
    .rst_n,
    .load_i      (start),
    .incr_i      (1'b0),
    .decr_i      (decr_r_transaction_cntr),
    .load_value_i(num_transactions),
    .count_o     (r_transactions_to_go),
    .is_zero_o   (r_final_transaction)
  );

endmodule : axi_m_rd

`endif  // AXI_M_RD_SV
