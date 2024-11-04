

`ifndef BRAM_SV
`define BRAM_SV

module bram #(
  parameter BRAM_DEPTH = 10,
  parameter BRAM_W = 64,
  parameter MEM_MACRO_TYPE = "distributed" // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM
) (
  input                                 clk,
  input                                 wen_i,
  input        [$clog2(BRAM_DEPTH)-1:0] waddr_i,
  input        [            BRAM_W-1:0] data_i,
  input        [$clog2(BRAM_DEPTH)-1:0] raddr_i,
  output logic [            BRAM_W-1:0] data_o
);

  // bram
  (* ram_style = MEM_MACRO_TYPE *) logic [BRAM_W-1:0] blockram[BRAM_DEPTH];

  // write operation
  always_ff @(posedge clk) begin
    if (wen_i) blockram[waddr_i] <= data_i;
  end

  // read operation
  always_ff @(posedge clk) begin
    data_o <= blockram[raddr_i];
  end

endmodule

`endif  // BRAM_SV
