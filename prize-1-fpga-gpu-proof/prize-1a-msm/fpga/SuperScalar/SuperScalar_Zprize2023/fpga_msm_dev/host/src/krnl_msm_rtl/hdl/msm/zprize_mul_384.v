`include "basic.vh"
`include "env.vh"
module zprize_mul_384 #(
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
    `MAX_FANOUT32 output logic [W0+W1-1:0]                                        out0
);

        localparam WN = W2 + (1 << (L-1));
        localparam W_Z1_HALF = W-W2;
        
        logic [W0-1:0] in0_r;
        logic [W1-1:0] in1_r;
        logic [M-1:0]  m_i_r;
        logic          rstN;

        logic [2-1:0][W2-1:0] x;
        logic [2-1:0][W2-1:0] y;
        logic [2-1:0][W2-1:0] x_p;
        logic [2-1:0][W2-1:0] y_p;
        logic [W-1:0] z0_r1, z2_r1;
        logic [WN*2-1:0] z1_r1;
        
        logic [2*W-1:0] z0_full;
        logic [2*W-1:0] z1_m0_full_n;
        logic [2*W-1:0] z1_m1_full;
        logic [2*W-1:0] z1_m2_full_n;
        logic [2*W-1:0] z2_full;

        logic [2*W-1:0] tmp_z_o0;
        logic [2*W-1:0] tmp_z_o1;
        logic [2*W-1:0] tmp_z_o2;

        logic [2*W+2-1:0] tmp_z_s;
        logic [2*W+2-1:0] tmp_z_co;
       
        logic [2*W+2:0] z_s;
        logic [2*W+2:0] z_co;
      
        logic [2*W+2:0] z_s_r1;
        logic [2*W+2:0] z_co_r1;
        logic [2*W+2:0] z_s_r2;
        logic [2*W+2:0] z_co_r2;
        logic [200-1:0]        out0_low_r1;
        logic                 carry_low_r1;
        logic [200-1:0]        out0_low_r2;
        logic [320+200-1:200]  out0_mid_r2;
        logic                 carry_mid_r2;
        logic [200-1:0]        out0_low_r3;
        logic [320+200-1:200]  out0_mid_r3;
        logic [2*W+2:320+200] out0_high_r3;

       
        logic [W-1:0] m0, m2;
        logic [WN*2-1:0] m1;
        logic [WN-1:0] a;
        logic [WN-1:0] b;
        logic [3-1:0][M-1:0] m_o_p;
        logic [M-1:0]     tmp_m_o;
        logic [W0+W1-1:0] tmp_out0;
        logic [W-1:0]     tmp_out0_low_r2;
        logic [W-1:0]     tmp_out0_low_r3;
        logic             carry_r2;
        logic [W-1:0]     tmp_out0_high_r3;
        logic [W0+W1-1:0] out0_golden;

        logic [W*2-1:0]         z2_full_r1;
        logic [WN*2-1+W2:0]     z1_full_r1;
        logic [W-1:0]           z0_full_r1;
        logic [WN*2-1+W2:0]     z1_full_r2;
        logic [W*2-1:0]         z2_full_r2;
//////////////////////////////////////////////////////////////////////////////
///////////////////////    RTL BODY        ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

assign rstN = !rst;

        assign in0_r = in0;
        assign in1_r = in1;
        assign m_i_r = m_i;

        assign x = in0;
        assign y = in1;

        assign z0_full      = {{W{1'b0}},             m0};
        assign z1_m0_full_n = {{W-W2{1'b1}},      ~m0,1'b1,1'b1,{W2-2{1'b0}}};
        assign z1_m1_full   = {{W*2-WN*2-W2{1'b0}},m1,1'b1,1'b1,{W2-2{1'b0}}};
        assign z1_m2_full_n = {{W-W2{1'b1}},      ~m2,1'b1,1'b0,{W2-2{1'b0}}};
        assign z2_full      = {m2,                    {W{1'b0}}};

        csa5_3 #(
        .CSAWIDTH(W*2)
        )
        u_csa5_3_z(
        .i0( z0_full     ),
        .i1( z1_m0_full_n),
        .i2( z1_m1_full  ),
        .i3( z1_m2_full_n),
        .i4( z2_full     ),
        .o0( tmp_z_o0    ),
        .o1( tmp_z_o1    ),
        .o2( tmp_z_o2    )
        );
        csa3_2 #(
        .CSAWIDTH(W*2+2)
        )
        u_csa3_2_z(
        .a ({2'b0,tmp_z_o0} ),
        .b ({1'b0,tmp_z_o1,1'b0} ),
        .c ({tmp_z_o2,2'b0} ),
        .s ( tmp_z_s    ),
        .co( tmp_z_co   )
        );
        
        assign z_s  = {1'b0,tmp_z_s[W*2+2-1:0]};
        assign z_co = {tmp_z_co[W*2+2-1:0],1'b0};
       
        always@(posedge clk) {carry_low_r1,out0_low_r1}  <= z_s[200-1:0] + z_co[200-1:0];
        always@(posedge clk)  z_s_r1                     <= z_s;
        always@(posedge clk) z_co_r1                     <= z_co;

        always@(posedge clk) {carry_mid_r2,out0_mid_r2}  <= z_s_r1[320+200-1:200] + z_co_r1[320+200-1:200]+carry_low_r1;
        always@(posedge clk)  out0_low_r2                <= out0_low_r1;
        always@(posedge clk)  z_s_r2                     <= z_s_r1;
        always@(posedge clk) z_co_r2                     <= z_co_r1;

        always@(posedge clk) out0_high_r3                <= z_s_r2[2*W+2:320+200] + z_co_r2[2*W+2:320+200]+carry_mid_r2;
        always@(posedge clk)  out0_mid_r3                <= out0_mid_r2;
        always@(posedge clk)  out0_low_r3                <= out0_low_r2;

        assign out0                   = {out0_high_r3,out0_mid_r3,out0_low_r3};
        
        //assert
        always@(posedge clk) z0_r1 <= m0;
        always@(posedge clk) z1_r1 <= m1 - m2 - m0;
        always@(posedge clk) z2_r1 <= m2;
        always@(posedge clk) tmp_out0     <= {z2_r1, {W{1'b0}}} + {z1_r1, {W2{1'b0}}} + z0_r1;
        always@(posedge clk) out0_golden  <= tmp_out0;
        `assert_clk(out0_golden === out0 || out0_golden=== W*2'bx)

        always@(posedge clk) m_o_p[2]  <= m_o_p[1];
        always@(posedge clk) tmp_m_o   <= m_o_p[2];
        always@(posedge clk) m_o       <= tmp_m_o ;

        assign a[WN-1:W2+1] = '0;
        assign b[WN-1:W2+1] = '0;

        piped_adder #(
            .W(W2),
            .R(1),
            .C(W2 >= 1280 ? 1 : 0)
        ) ax_192 (
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
        ) ay_192 (
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

        zprize_mul_192 #(
            .L(L-1),
            .W(W2),
            .T(ST)
        ) u_mul_192_0(
            .clk(clk),
            .rst(rst),
            .in0(x_p[0]),
            .in1(y_p[0]),
            .out0(m0)
        );
        zprize_mul_196 #(
            .L(L-1),
            .W(WN),
            .T(ST),
            .M(M)
        ) u_mul_196_1(
            .clk(clk),
            .rst(rst),
            .in0(a),
            .in1(b),
            .out0(m1),
            .m_i(m_o_p[0]),
            .m_o(m_o_p[1])
        );
        zprize_mul_192 #(
            .L(L-1),
            .W(W2),
            .T(ST),
            .M(M)
        ) u_mul_192_2(
            .clk(clk),
            .rst(rst),
            .in0(x_p[1]),
            .in1(y_p[1]),
            .out0(m2),
            .m_i(),
            .m_o()
        );


endmodule


