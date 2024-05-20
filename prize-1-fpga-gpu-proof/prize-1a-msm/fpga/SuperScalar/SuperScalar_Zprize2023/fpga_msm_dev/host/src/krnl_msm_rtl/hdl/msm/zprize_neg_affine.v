`include "basic.vh"
`include "env.vh"
//for zprize msm

//when 16=15, CORE_NUM=4, so SBITMULTISIZE = 64 for DDR alignment
///define SBITMULTISIZE       (16*4)

module zprize_neg_affine import zprize_param::*;(

input clk,
input rstN,
input valid_in,
input neg_flag_in,
input [384*3-1:0] affine_in,       //{t,y,x}

output valid_out,//pipeline8
output neg_flag_out,
output reg [384*3-1:0] affine_out,
output idle
);

///////////////////
// affine == {t,y,x}
// x  -->  Q-x
// y  -->  y
// t  -->  Q-t
// ///////////////////

wire [384-1:0] x;
wire [384-1:0] y;
wire [384-1:0] t;

wire [377:0] temp_neg_x;
wire [377:0] temp_neg_t;
wire [377:0] neg_x;
wire [377:0] neg_t;

wire valid_P;

wire [384*3-1:0] affine_r;       //{t,y,x}
wire [384-1:0] x_r;
wire [384-1:0] y_r;
wire [384-1:0] t_r;

wire [384-1:0] x_out;
wire [384-1:0] y_out;
wire [384-1:0] t_out;

/////////////////////////////////////////////
// rtl body
/////////////////////////////////////////////
assign x = affine_in[384*0+:384];
assign y = affine_in[384*1+:384];
assign t = affine_in[384*2+:384];

assign temp_neg_x = ~x[377-1:0];
assign temp_neg_t = ~t[377-1:0];

// x
addPipe378
#(
        .WIDTH (377+1)
)Qsubx (
 .dataA_P0(BLS12_377_P_27BIT_B384[377:0]),
 .dataB_P0(temp_neg_x),
 .addOne_P0(1'b1),
 .valid_P0(valid_in),
 .clk    (clk),
 .rstN   (rstN),
 .sel_M (1'b1),
          
 .run_P0(),
 .outo_P (neg_x),
 .idle(),
 .valid_P (valid_P)
);

// t
addPipe378
#(
        .WIDTH (377+1)
)Qsubt (
 .dataA_P0(BLS12_377_P_27BIT_B384[377:0]),
 .dataB_P0(temp_neg_t),
 .addOne_P0(1'b1),
 .valid_P0(valid_in),
 .clk    (clk),
 .rstN   (rstN),
 .sel_M (1'b1),
          
 .run_P0(),
 .outo_P (neg_t),
 .idle(),
 .valid_P ()
);

////////////////////////////////////////
////////////////////////////////////////
// out
//
hyperpipe # (
	.CYCLES  ((7+1+1)-1),
	.WIDTH   (1152)
) hp_point(
    .enable   (1'b1),
	.clk      (clk),
	.din      (affine_in),
	.dout     (affine_r)
);

assign x_r = affine_r[384*0+:384];
assign y_r = affine_r[384*1+:384];
assign t_r = affine_r[384*2+:384];

assign x_out = neg_flag_out ? {7'b0,neg_x[377-1:0]} : x_r;
assign y_out = y_r;
assign t_out = neg_flag_out ? {7'b0,neg_t[377-1:0]} : t_r;

assign affine_out = {t_out, y_out, x_out};

validpipe #(
    .CYCLES ((7+1+1)-1),
    .WIDTH  (1)
)validpipe_io(
  .din    (valid_in),
  .dout   (valid_out),
  .enable (1'b1),
  .idle   (idle),
  .clk    (clk),
  .rstN   (rstN) 
);

validpipe #(
    .CYCLES ((7+1+1)-1),
    .WIDTH  (1)
)validpipe_neg_flag(
  .din    (neg_flag_in),
  .dout   (neg_flag_out),
  .enable (1'b1),
  .idle   (),
  .clk    (clk),
  .rstN   (rstN) 
);

`assert_clk(valid_P==valid_out)

endmodule
