module zprize_mul_100 #(
    W=384,W0=W,W1=W,L=3,T=32'h07F7_F999,
    W2=W/2,
    R_I=0,
    CT = T[0 +: 4],
    ST = T >> 4,
    M=32,
    S=0
)(
    input wire                                                      clk,
    input wire                                                      rst,

    input wire [W0-1:0]                                             in0,
    input wire [W1-1:0]                                             in1,
    input wire [M-1:0]                                              m_i,
    output logic [M-1:0]                                            m_o,
    output logic [W0+W1-1:0]                                        out0
);

logic [W0-1:0] in0_r;
logic [W1-1:0] in1_r;
logic [M-1:0] m_i_r;

//////////////////////////////////////////////////////////////////////////////
///////////////////////    RTL BODY        ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
        
        assign in0_r = in0;
        assign in1_r = in1;
        assign m_i_r = m_i;


        localparam WN = W2 + (1 << (L-1));

        logic [2-1:0][W2-1:0] x;
        logic [2-1:0][W2-1:0] y;
        logic [2-1:0][W2-1:0] x_p;
        logic [2-1:0][W2-1:0] y_p;
        logic [W-1:0] z0, z2;
        // logic [W-1:0] z0_p, z2_p;
        logic [WN*2-1:0] z1;
        // logic [WN*2-1:0] z1_p;
        logic [W-1:0] m0, m2;
        logic [WN*2-1:0] m1;
        logic [WN-1:0] a;
        logic [WN-1:0] b;
        logic [3-1:0][M-1:0] m_o_p;

        assign x = in0;
        assign y = in1;

        always@(posedge clk) z0 <= m0;
        always@(posedge clk) z1 <= m1 - m2 - m0;
        always@(posedge clk) z2 <= m2;
        always@(posedge clk) m_o_p[2] <= m_o_p[1];

        always@(posedge clk) out0 <= {z2, {W{1'b0}}} + {z1, {W2{1'b0}}} + z0;
        always@(posedge clk) m_o <= m_o_p[2];

        piped_adder #(
            .W(W2),
            .R(1),
            .C(W2 >= 1280 ? 1 : 0)
        ) ax_50 (
            .clk(clk),
            .rst(rst),
            .cin0('0),
            .in0(x[0]),
            .in1(x[1]),
            .out0(a[0+:W2]),
            .cout0(a[W2]),
            .m_i(),
            .m_o()
        );
        piped_adder #(
            .W(W2),
            .R(1),
            .C(W2 >= 1280 ? 1 : 0),
            .M(M+W+W)
        ) ay_50 (
            .clk(clk),
            .rst(rst),
            .cin0('0),
            .in0(y[0]),
            .in1(y[1]),
            .out0(b[0+:W2]),
            .cout0(b[W2]),
            .m_i({m_i, x, y}),
            .m_o({m_o_p[0], x_p, y_p})
        );

        zprize_mul_50 #(
            .L(L-1),
            .W(W2),
            .T(ST)
        ) u_mul_50_0(
            .clk(clk),
            .rst(rst),
            .in0(x_p[0]),
            .in1(y_p[0]),
            .out0(m0)
        );
        zprize_mul_51 #(
            .L(L-1),
            .W(WN),
            .T(ST),
            .M(M)
        ) u_mul_51_1(
            .clk(clk),
            .rst(rst),
            .in0(a),
            .in1(b),
            .out0(m1),
            .m_i(m_o_p[0]),
            .m_o(m_o_p[1])
        );
        zprize_mul_50 #(
            .L(L-1),
            .W(W2),
            .T(ST),
            .M(M)
        ) u_mul_50_2(
            .clk(clk),
            .rst(rst),
            .in0(x_p[1]),
            .in1(y_p[1]),
            .out0(m2),
            .m_i(),
            .m_o()
        );


endmodule


