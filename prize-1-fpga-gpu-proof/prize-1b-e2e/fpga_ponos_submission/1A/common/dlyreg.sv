

`ifndef DLYREG_SV
`define DLYREG_SV

module dlyreg #(
  parameter DATA_W   = 64,
  parameter OUT1_LOC = 5,
  parameter REG_SIZE = 10,
  parameter USE_SHR  = "true"
) (
  input               clk,
  input  [DATA_W-1:0] data_i,
  output [DATA_W-1:0] data1_o,
  output [DATA_W-1:0] data2_o
);

  localparam LP_USE_SHREG = (USE_SHR == "true") ? "yes" : "no";
  (* shreg_extract = LP_USE_SHREG *) logic [DATA_W-1:0] reg_array[1:REG_SIZE];

  always_ff @(posedge clk) begin
    reg_array[1] <= data_i;
  end

  generate
    for (genvar i = 1; i < REG_SIZE; i = i + 1) begin : DELAY_BLOCK
      always_ff @(posedge clk) begin
        reg_array[i+1] <= reg_array[i];
      end
    end

    if (REG_SIZE == 0) begin
      assign data1_o = data_i;
      assign data2_o = data_i;
    end else if (OUT1_LOC > REG_SIZE) begin
      assign data1_o = reg_array[REG_SIZE];
      assign data2_o = reg_array[REG_SIZE];
    end else begin
      assign data1_o = reg_array[OUT1_LOC];
      assign data2_o = reg_array[REG_SIZE];
    end
  endgenerate

endmodule

`endif  // DLYREG_SV
