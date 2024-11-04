

`ifndef PULSE_COMBINE_SV
`define PULSE_COMBINE_SV

module pulse_combine #(
  parameter unsigned P_NUM_PULSES
) (
  input clk,
  input rst_n,

  input  [P_NUM_PULSES-1:0] pulse_vector_i,
  output                    pulse_o
);

  logic [P_NUM_PULSES-1:0] sticky_q;
  logic                    sticky_anded_w;
  logic                    sticky_anded_q;

  // Combine two done signal making then level, anding them and converting to a pulse
  for (genvar i = 0; i < P_NUM_PULSES; i++) begin
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        sticky_q[i] <= 1'b0;
      end else begin
        sticky_q[i] <= (pulse_vector_i[i] | sticky_q[i]) & ~pulse_o;
      end
    end
  end
  assign sticky_anded_w = &sticky_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      sticky_anded_q <= 1'b0;
    end else begin
      sticky_anded_q <= sticky_anded_w;
    end
  end
  assign pulse_o = sticky_anded_w & ~sticky_anded_q;  // edge detect

endmodule

`endif  // PULSE_COMBINE_SV
