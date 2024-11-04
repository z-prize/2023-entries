

`ifndef AP_RST_CTRL_SV
`define AP_RST_CTRL_SV

module ap_rst_ctrl #(
  parameter int C_RST_DELAY = 20
) (
  // System signals
  input ap_clk,
  input ap_rst_n,

  input ap_start,

  output ap_srst_n,
  output auto_rst_n,
  output ap_start_pulse
);

  logic virt_rst = 1'b0;
  logic ap_start_d1;
  logic ap_start_d2;
  logic ap_start_d2_q;
  logic rst_n;
  logic int_auto_rst_n = 1'b1;
  logic int_ap_start_pulse = 1'b0;

  // Synchronization of asynchronous ap_rst_n
  xpm_cdc_sync_rst i_sff (
    .dest_clk (ap_clk),
    .src_rst (ap_rst_n),
    .dest_rst(rst_n)
  );
  (* KEEP = "yes" *)
  shiftreg #(
    .SHIFT  (6),
    .DATA_W (1),
    .USE_SHR("false")
  ) i_shreg_rst_n (
    .clk   (ap_clk),
    .data_i(rst_n),
    .data_o(ap_srst_n)
  );

  // Auto reset of C_RST_DELAY clk duration based on ap_start
  dlyreg_r #(
    .DATA_W  (1),
    .OUT1_LOC(C_RST_DELAY),
    .REG_SIZE(C_RST_DELAY + C_RST_DELAY)
  ) i_dlyreg (
    .clk    (ap_clk),
    .rst_n  (ap_srst_n),
    .data_i (ap_start),
    .data1_o(ap_start_d1),
    .data2_o(ap_start_d2)
  );
  always_ff @(posedge ap_clk) begin
    virt_rst <= ap_start & ~ap_start_d1;
    int_auto_rst_n <= ap_srst_n & ~virt_rst;
  end
  (* KEEP = "yes" *)
  shiftreg #(
    .SHIFT  (6),
    .DATA_W (1),
    .USE_SHR("false")
  ) i_shreg_auto_rst_n (
    .clk   (ap_clk),
    .data_i(int_auto_rst_n),
    .data_o(auto_rst_n)
  );

  // Create pulse when ap_start_d2 transitions to 1
  always_ff @(posedge ap_clk) begin
    begin
      ap_start_d2_q <= ap_start_d2;
      int_ap_start_pulse <= ap_start_d2 & ~ap_start_d2_q;
    end
  end
  (* KEEP = "yes" *)
  shiftreg #(
    .SHIFT  (6),
    .DATA_W (1),
    .USE_SHR("false")
  ) i_shreg_ap_start (
    .clk   (ap_clk),
    .data_i(int_ap_start_pulse),
    .data_o(ap_start_pulse)
  );

endmodule : ap_rst_ctrl

`endif  // AP_RST_CTRL_SV