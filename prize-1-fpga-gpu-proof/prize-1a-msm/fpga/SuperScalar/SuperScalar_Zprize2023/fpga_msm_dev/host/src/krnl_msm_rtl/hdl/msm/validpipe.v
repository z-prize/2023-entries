// Quartus Prime Verilog Template
//
// valid-Pipelining Module

(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION off" *) 
module validpipe 
#(
    parameter CYCLES = 1,
    parameter WIDTH = 1
) 
(
	input [WIDTH-1:0]  din,
	output [WIDTH-1:0] dout,
    input  enable,
    output idle,            //only useful for WIDTH == 1, and R_data store valid signal
	input clk,
    input rstN 
);

	generate if (CYCLES==0) begin : GEN_COMB_INPUT
		assign dout = din;
        assign idle = 1;
	end 
	else begin : GEN_REG_INPUT  
		integer i;
		reg [WIDTH-1:0]            R_data [CYCLES-1:0];
        logic [CYCLES-1:0]          R_data0_d;
        always@* 
            for(i = 0; i < CYCLES; i = i + 1) 
                R_data0_d[i] = R_data[i][0];

        assign idle = !(|R_data0_d);
        
		always @ (posedge clk or negedge rstN) 
            if(!rstN) begin
			    for(i = 0; i < CYCLES; i = i + 1) 
                	R_data[i] <= 0;
            end
            else if(enable)begin
			    R_data[0] <= din;      
			    for(i = 1; i < CYCLES; i = i + 1) 
                	R_data[i] <= R_data[i-1];
            end
		assign dout = R_data[CYCLES-1];
	end
	endgenerate  



endmodule
