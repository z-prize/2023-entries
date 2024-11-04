`default_nettype none

module shift_reg_ca #(
    W=1,
    D=1
)(
    input wire  [1-1:0]                                             clk,
    input wire  [1-1:0]                                             rst,
    input wire  [W-1:0]                                             in0,
    output logic[W-1:0]                                             out0
);

logic [D-1:0][W-1:0] p;
assign out0 = p[0];
always_ff@(posedge clk) begin
    integer i;
    p[D-1] <= in0;
    for (i = 0; i < D-1; i ++)
        p[i] <= p[i+1];
end
endmodule

`default_nettype wire