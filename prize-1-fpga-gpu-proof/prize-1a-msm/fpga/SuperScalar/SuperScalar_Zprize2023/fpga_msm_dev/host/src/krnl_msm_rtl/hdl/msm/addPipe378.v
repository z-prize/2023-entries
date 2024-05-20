`include "basic.vh"
`include "env.vh"

    module addPipe378 
#(
        parameter WIDTH =757
) 
(
input [(WIDTH-1):0]dataA_P0,
input [(WIDTH-1):0]dataB_P0,
input addOne_P0,
input valid_P0,
input clk,
input rstN,
input sel_M,

output run_P0,
output [(WIDTH-1):0]outo_P,
output  idle,
output valid_P
);
localparam WIDTH_SPLIT =`DIV_UP((WIDTH),(7+1+1));
localparam WIDTH_ACTUALY = (7+1+1)*WIDTH_SPLIT;
wire [WIDTH_ACTUALY-1:0]data0_P0;
wire [WIDTH_ACTUALY-1:0]data1_P0;
assign data0_P0={{WIDTH_ACTUALY-WIDTH{1'b0}},dataA_P0};
assign data1_P0={{WIDTH_ACTUALY-WIDTH{1'b0}},dataB_P0};
////////////////control signal/////////////// ////////////////control signal///////////////
 (* max_fanout=400 *)reg valid_P1 ; wire run_P1;
assign run_P0=!(valid_P1 & !run_P1);
////////////////////////////RUN_P1
 (* max_fanout=400 *)reg valid_P2 ; wire run_P2;
assign run_P1=!(valid_P2 & !run_P2);
 `Flop (valid_P1,1 ,run_P0 ? valid_P0 : !run_P1 & valid_P1)
////////////////////////////RUN_P2
 (* max_fanout=400 *)reg valid_P3 ; wire run_P3;
assign run_P2=!(valid_P3 & !run_P3);
 `Flop (valid_P2,1 ,run_P1 ? valid_P1 : !run_P2 & valid_P2)
////////////////////////////RUN_P3
 (* max_fanout=400 *)reg valid_P4 ; wire run_P4;
assign run_P3=!(valid_P4 & !run_P4);
 `Flop (valid_P3,1 ,run_P2 ? valid_P2 : !run_P3 & valid_P3)
////////////////////////////RUN_P4
 (* max_fanout=400 *)reg valid_P5 ; wire run_P5;
assign run_P4=!(valid_P5 & !run_P5);
 `Flop (valid_P4,1 ,run_P3 ? valid_P3 : !run_P4 & valid_P4)
////////////////////////////RUN_P5
 (* max_fanout=400 *)reg valid_P6 ; wire run_P6;
assign run_P5=!(valid_P6 & !run_P6);
 `Flop (valid_P5,1 ,run_P4 ? valid_P4 : !run_P5 & valid_P5)
////////////////////////////RUN_P6
 (* max_fanout=400 *)reg valid_P7 ; wire run_P7;
assign run_P6=!(valid_P7 & !run_P7);
 `Flop (valid_P6,1 ,run_P5 ? valid_P5 : !run_P6 & valid_P6)
////////////////////////////RUN_P7
 (* max_fanout=400 *)reg valid_P8 ; wire run_P8;
assign run_P7=!(valid_P8 & !run_P8);
 `Flop (valid_P7,1 ,run_P6 ? valid_P6 : !run_P7 & valid_P7)
////////////////////////////RUN_P8
assign run_P8 = sel_M;
 `Flop (valid_P8,1 , valid_P7 )
////////////////control signal/////////////// ////////////////control signal///////////////
//////////////////////P0
////////////////////////
wire co_P0;
wire [WIDTH_SPLIT-1:0] outo_P0;
assign {co_P0,outo_P0[WIDTH_SPLIT-1:0]} = data0_P0[WIDTH_SPLIT-1:0] + data1_P0[WIDTH_SPLIT-1:0]+addOne_P0;
//
////////////////////////P1
reg cin_P1;
wire co_P1;
//reg [2*((WIDTH-1)/(7+1+1)):0]outo_P1;
`REG_LOGIC logic [2*WIDTH_SPLIT-1:0]outo_P1;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*1]data0_P1;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*1]data1_P1;

`FlopN(data0_P1[(WIDTH_ACTUALY-1):WIDTH_SPLIT*1] ,1 ,data0_P0[(WIDTH_ACTUALY-1):WIDTH_SPLIT*1])
`FlopN(data1_P1[(WIDTH_ACTUALY-1):WIDTH_SPLIT*1] ,1 ,data1_P0[(WIDTH_ACTUALY-1):WIDTH_SPLIT*1])
`FlopN(cin_P1 ,1 ,co_P0)
assign {co_P1,outo_P1[(1*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P1[(1*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P1[(1*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P1;
//`FlopN(outo_P1[(1*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P0[(1*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P0[(1*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P0)
`FlopN(outo_P1[1*WIDTH_SPLIT-1:0] ,1 ,outo_P0[1*WIDTH_SPLIT-1:0])
////////////////////////P2
reg cin_P2;
wire co_P2;
//reg [3*((WIDTH-1)/(7+1+1)):0]outo_P2;
`REG_LOGIC logic [3*WIDTH_SPLIT-1:0]outo_P2;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*2]data0_P2;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*2]data1_P2;

`FlopN(data0_P2[(WIDTH_ACTUALY-1):WIDTH_SPLIT*2] ,1 ,data0_P1[(WIDTH_ACTUALY-1):WIDTH_SPLIT*2])
`FlopN(data1_P2[(WIDTH_ACTUALY-1):WIDTH_SPLIT*2] ,1 ,data1_P1[(WIDTH_ACTUALY-1):WIDTH_SPLIT*2])
`FlopN(cin_P2 ,1 ,co_P1)
assign {co_P2,outo_P2[(2*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P2[(2*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P2[(2*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P2;
//`FlopN(outo_P2[(2*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P1[(2*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P1[(2*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P1)
`FlopN(outo_P2[2*WIDTH_SPLIT-1:0] ,1 ,outo_P1[2*WIDTH_SPLIT-1:0])
////////////////////////P3
reg cin_P3;
wire co_P3;
//reg [4*((WIDTH-1)/(7+1+1)):0]outo_P3;
`REG_LOGIC logic [4*WIDTH_SPLIT-1:0]outo_P3;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*3]data0_P3;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*3]data1_P3;

`FlopN(data0_P3[(WIDTH_ACTUALY-1):WIDTH_SPLIT*3] ,1 ,data0_P2[(WIDTH_ACTUALY-1):WIDTH_SPLIT*3])
`FlopN(data1_P3[(WIDTH_ACTUALY-1):WIDTH_SPLIT*3] ,1 ,data1_P2[(WIDTH_ACTUALY-1):WIDTH_SPLIT*3])
`FlopN(cin_P3 ,1 ,co_P2)
assign {co_P3,outo_P3[(3*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P3[(3*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P3[(3*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P3;
//`FlopN(outo_P3[(3*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P2[(3*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P2[(3*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P2)
`FlopN(outo_P3[3*WIDTH_SPLIT-1:0] ,1 ,outo_P2[3*WIDTH_SPLIT-1:0])
////////////////////////P4
reg cin_P4;
wire co_P4;
//reg [5*((WIDTH-1)/(7+1+1)):0]outo_P4;
`REG_LOGIC logic [5*WIDTH_SPLIT-1:0]outo_P4;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*4]data0_P4;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*4]data1_P4;

`FlopN(data0_P4[(WIDTH_ACTUALY-1):WIDTH_SPLIT*4] ,1 ,data0_P3[(WIDTH_ACTUALY-1):WIDTH_SPLIT*4])
`FlopN(data1_P4[(WIDTH_ACTUALY-1):WIDTH_SPLIT*4] ,1 ,data1_P3[(WIDTH_ACTUALY-1):WIDTH_SPLIT*4])
`FlopN(cin_P4 ,1 ,co_P3)
assign {co_P4,outo_P4[(4*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P4[(4*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P4[(4*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P4;
//`FlopN(outo_P4[(4*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P3[(4*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P3[(4*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P3)
`FlopN(outo_P4[4*WIDTH_SPLIT-1:0] ,1 ,outo_P3[4*WIDTH_SPLIT-1:0])
////////////////////////P5
reg cin_P5;
wire co_P5;
//reg [6*((WIDTH-1)/(7+1+1)):0]outo_P5;
`REG_LOGIC logic [6*WIDTH_SPLIT-1:0]outo_P5;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*5]data0_P5;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*5]data1_P5;

`FlopN(data0_P5[(WIDTH_ACTUALY-1):WIDTH_SPLIT*5] ,1 ,data0_P4[(WIDTH_ACTUALY-1):WIDTH_SPLIT*5])
`FlopN(data1_P5[(WIDTH_ACTUALY-1):WIDTH_SPLIT*5] ,1 ,data1_P4[(WIDTH_ACTUALY-1):WIDTH_SPLIT*5])
`FlopN(cin_P5 ,1 ,co_P4)
assign {co_P5,outo_P5[(5*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P5[(5*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P5[(5*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P5;
//`FlopN(outo_P5[(5*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P4[(5*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P4[(5*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P4)
`FlopN(outo_P5[5*WIDTH_SPLIT-1:0] ,1 ,outo_P4[5*WIDTH_SPLIT-1:0])
////////////////////////P6
reg cin_P6;
wire co_P6;
//reg [7*((WIDTH-1)/(7+1+1)):0]outo_P6;
`REG_LOGIC logic [7*WIDTH_SPLIT-1:0]outo_P6;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*6]data0_P6;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*6]data1_P6;

`FlopN(data0_P6[(WIDTH_ACTUALY-1):WIDTH_SPLIT*6] ,1 ,data0_P5[(WIDTH_ACTUALY-1):WIDTH_SPLIT*6])
`FlopN(data1_P6[(WIDTH_ACTUALY-1):WIDTH_SPLIT*6] ,1 ,data1_P5[(WIDTH_ACTUALY-1):WIDTH_SPLIT*6])
`FlopN(cin_P6 ,1 ,co_P5)
assign {co_P6,outo_P6[(6*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P6[(6*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P6[(6*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P6;
//`FlopN(outo_P6[(6*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P5[(6*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P5[(6*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P5)
`FlopN(outo_P6[6*WIDTH_SPLIT-1:0] ,1 ,outo_P5[6*WIDTH_SPLIT-1:0])
////////////////////////P7
reg cin_P7;
wire co_P7;
//reg [8*((WIDTH-1)/(7+1+1)):0]outo_P7;
`REG_LOGIC logic [8*WIDTH_SPLIT-1:0]outo_P7;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*7]data0_P7;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*7]data1_P7;

`FlopN(data0_P7[(WIDTH_ACTUALY-1):WIDTH_SPLIT*7] ,1 ,data0_P6[(WIDTH_ACTUALY-1):WIDTH_SPLIT*7])
`FlopN(data1_P7[(WIDTH_ACTUALY-1):WIDTH_SPLIT*7] ,1 ,data1_P6[(WIDTH_ACTUALY-1):WIDTH_SPLIT*7])
`FlopN(cin_P7 ,1 ,co_P6)
assign {co_P7,outo_P7[(7*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P7[(7*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P7[(7*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P7;
//`FlopN(outo_P7[(7*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P6[(7*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P6[(7*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P6)
`FlopN(outo_P7[7*WIDTH_SPLIT-1:0] ,1 ,outo_P6[7*WIDTH_SPLIT-1:0])
////////////////////////P8
reg cin_P8;
wire co_P8;
//reg [9*((WIDTH-1)/(7+1+1)):0]outo_P8;
`REG_LOGIC logic [9*WIDTH_SPLIT-1:0]outo_P8;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*8]data0_P8;
reg [(WIDTH_ACTUALY-1):WIDTH_SPLIT*8]data1_P8;

`FlopN(data0_P8[(WIDTH_ACTUALY-1):WIDTH_SPLIT*8] ,1 ,data0_P7[(WIDTH_ACTUALY-1):WIDTH_SPLIT*8])
`FlopN(data1_P8[(WIDTH_ACTUALY-1):WIDTH_SPLIT*8] ,1 ,data1_P7[(WIDTH_ACTUALY-1):WIDTH_SPLIT*8])
`FlopN(cin_P8 ,1 ,co_P7)
assign {co_P8,outo_P8[(8*WIDTH_SPLIT)+:WIDTH_SPLIT]} = data0_P8[(8*WIDTH_SPLIT)+:WIDTH_SPLIT] + data1_P8[(8*WIDTH_SPLIT)+:WIDTH_SPLIT] +cin_P8;
//`FlopN(outo_P8[(8*WIDTH_SPLIT)+:WIDTH_SPLIT] ,1 ,dataA_P7[(8*WIDTH_SPLIT)+:WIDTH_SPLIT] + dataB_P7[(8*WIDTH_SPLIT-1)+:WIDTH_SPLIT] +c_P7)
`FlopN(outo_P8[8*WIDTH_SPLIT-1:0] ,1 ,outo_P7[8*WIDTH_SPLIT-1:0])
//assign outo_P8 = {1'b0,final_compressed_s_P0} + {final_compressed_c_P0,1'b0};
assign outo_P = outo_P8[(WIDTH-1):0];
assign valid_P =valid_P8;
assign idle=!( 
valid_P0|
valid_P1|
valid_P2|
valid_P3|
valid_P4|
valid_P5|
valid_P6|
valid_P7|
valid_P8|
0);

endmodule 
