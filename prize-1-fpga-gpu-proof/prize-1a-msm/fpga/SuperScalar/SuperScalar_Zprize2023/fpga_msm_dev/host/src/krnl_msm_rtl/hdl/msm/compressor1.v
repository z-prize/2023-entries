 `include "basic.vh"
`include "env.vh"

module compressor1
(
        input [2:0]x,
        output reg [1:0] compressor
);
always @(*)
       case (x)                                                                                         
           3'b000 : compressor=2'b00;                                                                
           3'b001 : compressor=2'b01;                                                                
           3'b010 : compressor=2'b01;                                                                
           3'b011 : compressor=2'b10;                                                                
           3'b100 : compressor=2'b01;                                                                
           3'b101 : compressor=2'b10;                                                                
           3'b110 : compressor=2'b10;                                                                
           3'b111 : compressor=2'b11;                                                                
   endcase                                                                                              
endmodule
