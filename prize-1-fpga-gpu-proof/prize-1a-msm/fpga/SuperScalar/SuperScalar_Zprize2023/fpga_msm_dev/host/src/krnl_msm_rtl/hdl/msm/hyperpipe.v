module hyperpipe 
#(
    parameter CYCLES = 1,
    parameter WIDTH = 1 
) 
(
	input clk,
    input enable,
	input [WIDTH-1:0] din,
	output [WIDTH-1:0] dout
);

    localparam CYCLES_ = CYCLES == 0 ? 1 : CYCLES;      //fix when CYCLES == 0

		integer i;
		reg [WIDTH-1:0] R_data [CYCLES_-1:0];
        
		always @ (posedge clk) 
            if(enable) begin   
		    	R_data[0] <= din;      
		    	for(i = 1; i < CYCLES; i = i + 1) 
                	R_data[i] <= R_data[i-1];
		    end
		assign dout = CYCLES == 0 ? din : R_data[CYCLES_-1];

endmodule
