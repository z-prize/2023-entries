

`ifndef COUNTER_BIDIR_SV
`define COUNTER_BIDIR_SV

module counter_bidir #(
  parameter int C_W = 4,
  parameter int C_INC_W = 1,
  parameter [C_W-1:0] C_INIT = '0
) (
  input               clk,
  input               rst_n,
  input               load_i,
  input [C_INC_W-1:0] incr_i,
  input [C_INC_W-1:0] decr_i,
  input [    C_W-1:0] load_value_i,

  output [C_W-1:0] count_o,
  output           is_zero_o
);

  logic [C_W-1:0] count_r = C_INIT;
  logic is_zero_r = (C_INIT == '0);

  assign count_o = count_r;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      count_r <= C_INIT;
    end else begin
      if (load_i) begin
        count_r <= load_value_i;
      end else begin
        count_r <= count_r + incr_i - decr_i;
      end
    end
  end

  assign is_zero_o = is_zero_r;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      is_zero_r <= (C_INIT == '0);
    end else begin
      if (load_i) begin
        is_zero_r <= (load_value_i == '0);
      end else begin
        is_zero_r <= (incr_i > decr_i) ? count_r == '1 :
                     (incr_i < decr_i) ? count_r == 1 : is_zero_r;
      end
    end
  end

endmodule : counter_bidir

`endif  // COUNTER_BIDIR_SV
