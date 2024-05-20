module zprize_mul_49 #(
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


        localparam DEPTH = W0 < 27 ? 1 : W0 <= 26*2 ? 2 : 1+4;


            logic [26+W1-1:0]    m_0;
            logic [26+W1-1:0]    m_0_tmp;
            logic [W0-26+W1-1:0] m_1;
            logic [W0-26+W1-1:0] m_1_tmp;

            zprize_mul_26 #(.W0(26)   , .W1(W1), .T(T)) u_mul_26 (.clk(clk), .rst(rst), .in0(in0[  0 +: 26])   , .in1(in1), .out0(m_0_tmp));
            zprize_mul_23 #(.W0(W0-26), .W1(W1), .T(T)) u_mul_23 (.clk(clk), .rst(rst), .in0(in0[ 26 +: W0-26]), .in1(in1), .out0(m_1_tmp));

            always@(posedge clk) begin
                m_0                <= m_0_tmp;
                m_1                <= m_1_tmp;
                out0[0+:26]        <= m_0[0+:26];
                out0[26+:W0+W1-26] <= m_1 + m_0[26+:W1];
                // out0 <= m_0 + (m_1 << 26);
            end

        //end

        logic [DEPTH-1:0][M-1:0] m_o_p;

        always@(posedge clk) begin
            integer i;
            m_o_p[0] <= m_i;
            for (i = 1; i < DEPTH; i ++)
                m_o_p[i] <= m_o_p[i-1];
        end

        always@(posedge clk) begin
            m_o <= m_o_p[DEPTH-1];
        end

endmodule


