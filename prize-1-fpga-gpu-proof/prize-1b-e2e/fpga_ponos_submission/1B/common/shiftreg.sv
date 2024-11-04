

`ifndef SHIFTREG_SV
`define SHIFTREG_SV

module shiftreg #(
  parameter SHIFT   = 0,
  parameter DATA_W  = 64,
  parameter USE_SHR = "true"
) (
  input               clk,
  input  [DATA_W-1:0] data_i,
  output [DATA_W-1:0] data_o
);

  localparam LP_USE_SHREG = (USE_SHR == "true") ? "yes" : "no";
  (* shreg_extract = LP_USE_SHREG *) logic [DATA_W-1:0] shift_array[1:SHIFT];

  always_ff @(posedge clk) begin
    shift_array[1] <= data_i;
  end

  generate
    for (genvar i = 1; i < SHIFT; i = i + 1) begin : DELAY_BLOCK
      always_ff @(posedge clk) begin
        shift_array[i+1] <= shift_array[i];
      end
    end

    if (SHIFT == 0) begin
      assign data_o = data_i;
    end else begin
      assign data_o = shift_array[SHIFT];
    end
  endgenerate

endmodule

`endif  // SHIFTREG_SV
