`include "basic.vh"
`include "env.vh"
(* keep_hierarchy = "yes" *) module zprize_mul_mod #(
    WI=378,WQ=377,WR=384,WM=48*8,L=3,T0=32'h07F7_FBBB,T1=32'h0000_005F,T2=32'h07F7_FBBB,
    M=1,

    // BLS12-377
    Q=384'h1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001,
    QP=384'hbfa5205feec82e3d22f80141806a3cec5b245b86cced7a1335ed1347970debffd1e94577a00000008508bfffffffffff,

    R_I=1,
    R_O=1
)(
    input wire                                                      clk,
    input wire                                                      rst,

    input wire [WI-1:0]                                             in0,
    input wire [WI-1:0]                                             in1,
    input wire [M-1:0]                                              m_i,
    output logic [M-1:0]                                            m_o,
    output logic [WI-1:0]                                           out0
);


logic [WM-1:0] in0_r;
logic [WM-1:0] in1_r;
logic [M-1:0] m_i_r;

logic         rstN;

//////////////////////////////////////////////////////////////////////////////
///////////////////////    RTL BODY        ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

assign rstN = !rst;

`FlopN(in0_r,1,in0)
`FlopN(in1_r,1,in1)
`FlopN(m_i_r,1,m_i)



logic [2-1:0][WM-1:0]           m0_i;
logic [2*WM-1:0]                m0;
logic [3-1:0][1-1:0]            m0_c;
logic [3-1:0][WM*2-WR-1:0]      m0_h;

logic [2-1:0][WM-1:0]           m1_i;
logic [WM*2-1:0]                m1;

logic [2-1:0][WM-1:0]           m2_i;
logic [WM*2-1:0]                m2;

logic [2-1:0][WR-1:0]           a0_i;
logic [WR+1-1:0]                a0;
logic [WR+1-1:0]                a0_p;

logic [2-1:0][WR-1:0]           a1_i;
logic [WR-1:0]                  a1;
`MAX_FANOUT32 logic             a1_carry;

logic [5-1:0][M-1:0]            m_o_p;

assign m0_i[0] = in0_r;
assign m0_i[1] = in1_r;

zprize_mul_384 #(
    .L      (L),
    .W      (WM),
    .T      (T0),
    .M      (M)
) u_mul_384_0 (
    .clk    (clk),
    .rst    (rst),
    .in0    (m0_i[0]),
    .in1    (m0_i[1]),
    .out0   (m0),
    .m_i    (m_i_r),
    .m_o    (m_o_p[0])
);

assign m0_h[0] = m0 >> WR;
assign m0_c[0] = |m0[0+:WR]; // we don't need the lower WR bits, only need to konw if they cause a carry in the add-0
assign m1_i[0] = m0[0 +: WR];
assign m1_i[1] = QP;

zprize_mul_384_shift #(
    .L      (L),
    .W      (WM),
    .T      (T1),
    .M      ($bits({m0_c[0], m0_h[0], m_o_p[0]}))
) u_mul_384_shift_1 (
    .clk    (clk),
    .rst    (rst),
    .in0    (m1_i[0]),
    .in1    (m1_i[1]),
    .out0   (m1),
    .m_i    ({m0_c[0], m0_h[0], m_o_p[0]}),
    .m_o    ({m0_c[1], m0_h[1], m_o_p[1]})
);

assign m2_i[0] = m1[0 +: WR];
assign m2_i[1] = Q;

zprize_mul_384 #(
    .L      (L),
    .W      (WM),
    .T      (T2),
    .M      ($bits({m0_c[1], m0_h[1], m_o_p[1]}))
) u_mul_384_2 (
    .clk    (clk),
    .rst    (rst),
    .in0    (m2_i[0]),
    .in1    (m2_i[1]),
    .out0   (m2),
    .m_i    ({m0_c[1], m0_h[1], m_o_p[1]}),
    .m_o    ({m0_c[2], m0_h[2], m_o_p[2]})
);

assign a0_i[0] = m2 >> WR;
assign a0_i[1] = m0_h[2];

piped_adder #(
    .W      (WR),
    .R      (1),
    .C      (1),
    .M      (M)
) a_384_0 (
    .clk    (clk),
    .rst    (rst),
    .cin0   (m0_c[2]),
    .in0    (a0_i[0]),
    .in1    (a0_i[1]),
    .out0   (a0[0+:WR]),
    .cout0  (a0[WR]),
    .m_i    (m_o_p[2]),
    .m_o    (m_o_p[3])
);

assign a1_i[0] = a0;
//assign a1_i[1] = a0 >= Q ? -Q : '0;
assign a1_i[1] = -Q;

piped_adder #(
    .W      (WR+1),
    .R      (1),
    .C      (1),
    .M      (M)
) a_384_1 (
    .clk    (clk),
    .rst    (rst),
    .cin0   ('0),
    .in0    ({1'b0    ,a1_i[0]}),
    .in1    ({1'b1    ,a1_i[1]}),
    .out0   ({a1_carry,a1}),
    .cout0  (),
    .m_i    (m_o_p[3]),
    .m_o    (m_o_p[4])
);

hyperPipe #(
    .CYCLES (2   ),
    .WIDTH  (WR)
)hyperPipe_a1(
	.din    (a0),
	.dout   (a0_p),
	.clk    (clk   )
);

`FlopN(out0,1,a1_carry?a0_p:a1)
`FlopN(m_o ,1,m_o_p[4])

endmodule

