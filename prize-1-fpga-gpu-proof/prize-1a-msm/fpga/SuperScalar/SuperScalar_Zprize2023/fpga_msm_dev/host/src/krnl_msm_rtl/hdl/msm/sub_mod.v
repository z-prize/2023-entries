`include "basic.vh"
`include "env.vh"

module sub_mod import zprize_param::*;(

input clk,
input rstN,
input [377:0]in0,       //a
input [377:0]in1,       //b
input sel,
input valid,

output valid_out,
output sel_out,     //sel_out = a-b<0
output idle,
output reg [377:0]out
);
///////////////////
// if (a-b<0)
//      out=a-b +p
//else
//      out=a-b
//

////////////////////////beat pipe
reg  [27*14-1:0]  in0_P0;//at pipe 0
reg  [27*14-1:0]  in1_P0;//at pipe 0
reg  valid_P0;//at pipe 0
reg  sel_P0;//at pipe 0
logic  valid_P;
`FlopN( in0_P0  ,1,in0  )//at pipe 0
`FlopN( in1_P0  ,1,in1  )
`FlopN( sel_P0  ,1,sel  )
`FlopN( valid_P0,1,valid)
////////////////////////beat pipe
wire run_P0;
wire [378:0]outSub_P;
`REG_LOGIC reg [378:0]outSub_R;
wire [378:0]outSub_Q;
wire idle_addPipe378;

//addPipe378: outSub_P = a - b
//outSub_P[378] = a - b < 0
addPipe378
#(
        .WIDTH (379)
)addPipe378 (
 .dataA_P0({1'b0,in0_P0}),
 .dataB_P0({1'b1,~in1_P0}),
 .addOne_P0(1'b1),
 .valid_P0(valid_P0),
 .clk    (clk),
 .rstN   (rstN),
 .sel_M (sel_P0),
          
 .run_P0(run_P0),
 .outo_P (outSub_P),
 .idle(idle_addPipe378),
 .valid_P (valid_P)
);

wire [378:0]addP_out_P;
wire [378:0]addP_out_Q;

//assign addP_out_P= outSub_P+p;
//wire [378:0]temp_out;
///////////////////////beat a pipe
`FlopN(outSub_R,1,outSub_P)
///////////////////////beat a pipe

//subP_addPipe: outSub_R + (outSub_R[378] ? 0 : 0)
//
addPipe378 //4 pipe
#(
        .WIDTH (379)
)subP_addPipe (
 .dataA_P0(outSub_R),
 .dataB_P0({1'b0,p[377:0]}&{378{outSub_R[378]}}),
 .addOne_P0(1'b0),
 .valid_P0(1'b1),
 .clk    (clk),
 .rstN   (rstN),
 .sel_M (1'b1),
          
 .idle(),
 .run_P0(),
 .outo_P (addP_out_Q),
 .valid_P ()
);

hyperPipe # (

		.CYCLES  ((7+1+1)-1),

		.WIDTH   (379)

	) hp_out(

		.clk      (clk),

		.din      ({outSub_R}),

		.dout     ({outSub_Q})

	);
////////////////////////////////////////
reg valid_in0;
`Flop(valid_in0,1,valid_P)
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
assign idle= idle_addPipe378 & !(
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
valid_P);
////////////////////////////////////////
`FlopN (out ,1 ,addP_out_Q[377:0])
assign sel_out = outSub_Q[378];
endmodule
