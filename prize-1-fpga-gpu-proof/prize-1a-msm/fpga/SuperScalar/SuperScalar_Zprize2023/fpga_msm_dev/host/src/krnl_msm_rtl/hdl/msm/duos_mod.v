
`include "basic.vh"
`include "env.vh"

module duos_mod import zprize_param::*;(

input clk,
input rstN,
input [377:0]in0,//pipeline0
input sel,
input valid,

output valid_out,//pipeline8
output idle,
output reg[377:0]out
);
///////////////////
// if (a+b>p)
//      out=a+b-p
//else
//      out=a+b
//

wire run_P0;
wire [400:0]outAdd_P0;
wire [377:0]outAdd;
wire [378:0]outAdd_Q;
assign outAdd_P0=
        {2'b0,in0[377:0],1'b0};
wire [378:0]subP_out_P;
wire [378:0]subP_out_Q;

//assign subP_out_P = outAdd_P + ~p+1;
//hyperPipe # (
//
//		.CYCLES  ((7+1+1)),
//
//		.WIDTH   (378+379)
//
//	) hp_out(
//
//		.clk      (clk),
//
//		.din      ({outAdd_P,subP_out_P}),
//
//		.dout     ({outAdd_Q,subP_out_Q})
//
//	);

wire [400:0] out0;
wire [400:0] outP0;

//assign out0=outAdd_P+ ~P+1;

addPipe378 //4 pipe
#(
        .WIDTH (379)
)subP_addPipe (
 .dataA_P0(outAdd_P0[378:0]),
 .dataB_P0(Np[378:0]),
 .addOne_P0(1'b0),
 .valid_P0(1'b1),
 .clk    (clk),
 .rstN   (rstN),
 .sel_M (1'b1),
          
 .run_P0(),
 .idle(),
 .outo_P (subP_out_P),
 .valid_P ()
);

hyperPipe # (

		.CYCLES  ((7+1+1)-1+1),

		.WIDTH   (379)

	) hp_out(

		.clk      (clk),

		.din      ({outAdd_P0[378:0]}),

		.dout     ({outAdd_Q})

	);

//assign out=outP0[400]?outAdd_Q[377:0]:outP0[377:0];

hyperPipe # ( .CYCLES  (1),
		.WIDTH   (378*1)
	) hp_out_M1 (
		.clk (clk),
		.din      ({subP_out_P[377:0]}),
		.dout     ({subP_out_Q[377:0]})
);  
//`FlopN (subPi_Q,1 ,subP_out_Q[378])
`KEEP `MAX_FANOUT32 reg subP0_Q;
`FlopN(subP0_Q ,1,subP_out_P[378]);
`FlopN (out[27*0+:27],1 ,subP0_Q? outAdd_Q[27*0+:27]:subP_out_Q[27*0+:27])
`KEEP `MAX_FANOUT32 reg subP1_Q;
`FlopN(subP1_Q ,1,subP_out_P[378]);
`FlopN (out[27*1+:27],1 ,subP1_Q? outAdd_Q[27*1+:27]:subP_out_Q[27*1+:27])
`KEEP `MAX_FANOUT32 reg subP2_Q;
`FlopN(subP2_Q ,1,subP_out_P[378]);
`FlopN (out[27*2+:27],1 ,subP2_Q? outAdd_Q[27*2+:27]:subP_out_Q[27*2+:27])
`KEEP `MAX_FANOUT32 reg subP3_Q;
`FlopN(subP3_Q ,1,subP_out_P[378]);
`FlopN (out[27*3+:27],1 ,subP3_Q? outAdd_Q[27*3+:27]:subP_out_Q[27*3+:27])
`KEEP `MAX_FANOUT32 reg subP4_Q;
`FlopN(subP4_Q ,1,subP_out_P[378]);
`FlopN (out[27*4+:27],1 ,subP4_Q? outAdd_Q[27*4+:27]:subP_out_Q[27*4+:27])
`KEEP `MAX_FANOUT32 reg subP5_Q;
`FlopN(subP5_Q ,1,subP_out_P[378]);
`FlopN (out[27*5+:27],1 ,subP5_Q? outAdd_Q[27*5+:27]:subP_out_Q[27*5+:27])
`KEEP `MAX_FANOUT32 reg subP6_Q;
`FlopN(subP6_Q ,1,subP_out_P[378]);
`FlopN (out[27*6+:27],1 ,subP6_Q? outAdd_Q[27*6+:27]:subP_out_Q[27*6+:27])
`KEEP `MAX_FANOUT32 reg subP7_Q;
`FlopN(subP7_Q ,1,subP_out_P[378]);
`FlopN (out[27*7+:27],1 ,subP7_Q? outAdd_Q[27*7+:27]:subP_out_Q[27*7+:27])
`KEEP `MAX_FANOUT32 reg subP8_Q;
`FlopN(subP8_Q ,1,subP_out_P[378]);
`FlopN (out[27*8+:27],1 ,subP8_Q? outAdd_Q[27*8+:27]:subP_out_Q[27*8+:27])
`KEEP `MAX_FANOUT32 reg subP9_Q;
`FlopN(subP9_Q ,1,subP_out_P[378]);
`FlopN (out[27*9+:27],1 ,subP9_Q? outAdd_Q[27*9+:27]:subP_out_Q[27*9+:27])
`KEEP `MAX_FANOUT32 reg subP10_Q;
`FlopN(subP10_Q ,1,subP_out_P[378]);
`FlopN (out[27*10+:27],1 ,subP10_Q? outAdd_Q[27*10+:27]:subP_out_Q[27*10+:27])
`KEEP `MAX_FANOUT32 reg subP11_Q;
`FlopN(subP11_Q ,1,subP_out_P[378]);
`FlopN (out[27*11+:27],1 ,subP11_Q? outAdd_Q[27*11+:27]:subP_out_Q[27*11+:27])
`KEEP `MAX_FANOUT32 reg subP12_Q;
`FlopN(subP12_Q ,1,subP_out_P[378]);
`FlopN (out[27*12+:27],1 ,subP12_Q? outAdd_Q[27*12+:27]:subP_out_Q[27*12+:27])
`KEEP `MAX_FANOUT32 reg subP13_Q;
`FlopN(subP13_Q ,1,subP_out_P[378]);
`FlopN (out[27*13+:27],1 ,subP13_Q? outAdd_Q[27*13+:27]:subP_out_Q[27*13+:27])

////////////////////////////////////////
reg valid_in0;
`Flop(valid_in0,1,valid)
reg valid_in1;
`Flop(valid_in1,1,valid_in0)
reg valid_in2;
`Flop(valid_in2,1,valid_in1)
reg valid_in3;
`Flop(valid_in3,1,valid_in2)
reg valid_in4;
`Flop(valid_in4,1,valid_in3)
reg valid_in5;
`Flop(valid_in5,1,valid_in4)
reg valid_in6;
`Flop(valid_in6,1,valid_in5)
reg valid_in7;
`Flop(valid_in7,1,valid_in6)
reg valid_in8;
`Flop(valid_in8,1,valid_in7)
reg valid_in9;
`Flop(valid_in9,1,valid_in8)
assign valid_out=valid_in9;
////////////////////////////////////////
assign idle= !(
valid_in0|
valid_in1|
valid_in2|
valid_in3|
valid_in4|
valid_in5|
valid_in6|
valid_in7|
valid_in8|
valid_in9|
valid);

endmodule
