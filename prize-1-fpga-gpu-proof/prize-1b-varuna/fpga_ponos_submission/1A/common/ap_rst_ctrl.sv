

`ifndef AP_RST_CTRL_SV
`define AP_RST_CTRL_SV

module ap_rst_ctrl #(
  parameter int C_RST_DELAY = 20
) (
  // System signals
  input ap_clk,
  input ap_rst_n,

  input ap_start,
  input ap_done,

  output logic ap_srst_n,
  output logic auto_rst_n,
  output logic ap_ready,
  output logic ap_idle,
  output logic ap_start_pulse
);

  (* ASYNC_REG = "TRUE" *)logic areset_n = 1'b0;
  logic sreset_n;
  logic virt_rst = 1'b0;
  logic ap_start_d1;
  logic ap_start_d2;
  logic ap_start_d2_q;

  // Synchronization of asynchronous ap_rst_n
  sync_ff i_sff (
    .clk (ap_clk),
    .ain (ap_rst_n),
    .sout(ap_srst_n)
  );

  // Auto reset of C_RST_DELAY clk duration based on ap_start
  dlyreg #(
    .DATA_W  (1),
    .OUT1_LOC(C_RST_DELAY),
    .REG_SIZE(C_RST_DELAY + C_RST_DELAY)
  ) i_dlyreg (
    .clk    (ap_clk),
    .data_i (ap_start),
    .data1_o(ap_start_d1),
    .data2_o(ap_start_d2)
  );
  always @(posedge ap_clk) begin
    virt_rst   <= ap_start & ~ap_start_d1;
    auto_rst_n <= sreset_n & ~virt_rst;
  end

  // Create pulse when ap_start_d2 transitions to 1
  always @(posedge ap_clk) begin
    begin
      ap_start_d2_q <= ap_start_d2;
    end
  end
  assign ap_start_pulse = ap_start_d2 & ~ap_start_d2_q;

  // ap_idle is asserted when done is asserted, it is de-asserted when ap_start_pulse 
  // is asserted
  always @(posedge ap_clk) begin
    if (!ap_srst_n) begin
      ap_idle <= 1'b1;
    end else begin
      ap_idle <= ap_done ? 1'b1 : ap_start_pulse ? 1'b0 : ap_idle;
    end
  end

  assign ap_ready = ap_done;

endmodule : ap_rst_ctrl

`endif  // AP_RST_CTRL_SV
