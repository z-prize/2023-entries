

`ifndef SYNC_FF_SV
`define SYNC_FF_SV

module sync_ff #(
  parameter type T = logic
) (
  input    clk,
  input  T ain,
  output T sout
);

  T x0, x1;

  always @(posedge clk) begin
    x0   <= ain;
    x1   <= x0;
    sout <= x1;
  end

endmodule

`endif  // SYNC_FF_SV
