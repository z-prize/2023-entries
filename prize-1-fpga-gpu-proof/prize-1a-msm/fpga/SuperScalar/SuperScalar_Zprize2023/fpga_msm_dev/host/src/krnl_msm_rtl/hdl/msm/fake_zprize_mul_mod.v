`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

(* keep_hierarchy = "yes" *) module fake_zprize_mul_mod #(
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

logic         rstN;

//////////////////////////////////////////////////////////////////////////////
///////////////////////    RTL BODY        ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

assign rstN = !rst;

logic [WI*2-1:0] temp_ab;
logic [WI*2-1:0] temp_ab_low;
logic [WM*2-1:0] temp_ab_full;
logic [WM*2-1:0] temp_ab_qp_full;
logic [WM*2-1:0] temp_ab_qp_q_full;
logic [WI-1:0]   temp_out;

assign temp_ab            = in0*in1;
assign temp_ab_low        = temp_ab[WM-1:0];
assign temp_ab_full       = {{(2*WM-2*WI){1'b0}},temp_ab};
assign temp_ab_qp_full    = temp_ab_low*QP;
assign temp_ab_qp_q_full  = temp_ab_qp_full[WM-1:0]*Q+temp_ab_full;
//assign temp_out           = temp_ab_qp_q_full[WM+WI-1:WM];
assign temp_out           = (temp_ab_qp_q_full[WM+WI-1:WM]>=Q)?temp_ab_qp_q_full[WM+WI-1:WM]-Q :temp_ab_qp_q_full[WM+WI-1:WM];

	hyperPipe # (
		.CYCLES  (39),
		.WIDTH   (WI+1)
	) fake_hp_out(
		.clk      (clk),
		.din      ({temp_out,m_i}),
		.dout     ({out0,m_o})
	);

endmodule
