`default_nettype none
module f_add_sub #(
    W=1,
    Q=0,
    ADD=1,
    RED=1
)(
    input wire  [1-1:0]                                             clk,
    input wire  [1-1:0]                                             rst,
    input wire  [W-1:0]                                             in0,
    input wire  [W-1:0]                                             in1,
    output logic[W-1:0]                                             out0
);

logic [W+1-1:0] c02_s;
logic [W+1-1:0] not_q;

assign not_q = -Q;

generate
    if (ADD) begin
        if (RED == 1) begin

            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c00_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(in0),
                .in1(in1),
                .out0(c02_s[0+:W]),
                .cout0(c02_s[W]),
                .m_i(),
                .m_o()
            );

            piped_adder #(
                .W(W+1),
                .R(1),
                .C(1)
            ) c02_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(c02_s[0+:W+1]),
                .in1(c02_s >= Q ? not_q : 0),
                .out0(out0),
                .cout0(),
                .m_i(),
                .m_o()
            );

            // always_ff@(posedge clk) out0 <= s >= Q ? s-Q : s;
        end else begin
            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c00_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(in0),
                .in1(in1),
                .out0(out0),
                .cout0(),
                .m_i(),
                .m_o()
            );
            // always_ff@(posedge clk) out0 <= s;
        end
    end else begin
        if (RED == 1) begin

            logic [1-1:0] c02_underflow;

            piped_adder #(
                .W(W),
                .R(1),
                .C(1),
                .M(1)
            ) c00_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0(1'b1),
                .in0(in0),
                .in1(~in1),
                .out0(c02_s[0+:W]),
                .cout0(),
                .m_i(in0 < in1),
                .m_o(c02_underflow)
            );

            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c02_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(c02_s[0+:W]),
                .in1(c02_underflow ? Q : 0),
                .out0(out0),
                .cout0(),
                .m_i(),
                .m_o()
            );

            // assign s = in0 - in1;
            // always_ff@(posedge clk) out0 <= in0 < in1 ? s+Q : s;
        end else begin
            // piped_adder #(
            //     .W(W),
            //     .R(1),
            //     .C(0)
            // ) c00_0_inst (
            //     .clk(clk),
            //     .rst(rst),
            //     .cin0('0),
            //     .in0(in0),
            //     .in1(Q),
            //     .out0(c02_s),
            //     .cout0()
            // );
            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c02_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0(1'b1),
                .in0(in0 + Q),
                .in1(~in1),
                .out0(out0),
                .cout0()
            );
            // assign s = (Q + in0) - in1;
            // always_ff@(posedge clk) out0 <= s;
        end
    end
endgenerate

endmodule

`default_nettype wire