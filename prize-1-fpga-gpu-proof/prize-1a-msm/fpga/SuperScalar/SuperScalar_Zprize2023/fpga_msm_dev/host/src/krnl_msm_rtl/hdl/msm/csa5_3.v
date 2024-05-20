 `include "basic.vh"
`include "env.vh"

module csa5_3 
#(
        parameter CSAWIDTH = 27
)
(                                                                                              
  input [CSAWIDTH -1:0] i0,                                                                                              
  input [CSAWIDTH -1:0] i1,                                                                                              
  input [CSAWIDTH -1:0] i2,                                                                                              
  input [CSAWIDTH -1:0] i3,                                                                                              
  input [CSAWIDTH -1:0] i4,                                                                                              
  output[CSAWIDTH -1:0] o0,                                                                                             
  output[CSAWIDTH -1:0] o1,                                                                                             
  output[CSAWIDTH -1:0] o2                                                                                              
  );                                                                                                           
                                                                                                               

 wire [2:0]temp[CSAWIDTH];                                                                                         
genvar i;
generate 
   for (i=0;i<CSAWIDTH;i=i+1)begin:inst
compressor compressor (
        .x({i4[i],i3[i],i2[i],i1[i],i0[i]}),
        .compressor(temp[i])
);
   assign o0[i]=temp[i][0];                                                                            
   assign o1[i]=temp[i][1];                                                                            
   assign o2[i]=temp[i][2];                                                                            
   end
endgenerate
                                                                                                               
//////////////////////////////////                                                                                                               
                                                                                                               
  endmodule                                                                                                    

module compressor                                                                                                               
(
        input [4:0]x,
        output reg [2:0] compressor
);
always @(*)
       case (x)                                                                                         
           5'b00000 : compressor=3'b000;                                                                
           5'b00001 : compressor=3'b001;                                                                
           5'b00010 : compressor=3'b001;                                                                
           5'b00011 : compressor=3'b010;                                                                
           5'b00100 : compressor=3'b001;                                                                
           5'b00101 : compressor=3'b010;                                                                
           5'b00110 : compressor=3'b010;                                                                
           5'b00111 : compressor=3'b011;                                                                
           5'b01000 : compressor=3'b001;                                                                
           5'b01001 : compressor=3'b010;                                                                
           5'b01010 : compressor=3'b010;                                                                
           5'b01011 : compressor=3'b011;                                                                
           5'b01100 : compressor=3'b010;                                                                
           5'b01101 : compressor=3'b011;                                                                
           5'b01110 : compressor=3'b011;                                                                
           5'b01111 : compressor=3'b100;                                                                
           5'b10000 : compressor=3'b001;                                                                
           5'b10001 : compressor=3'b010;                                                                
           5'b10010 : compressor=3'b010;                                                                
           5'b10011 : compressor=3'b011;                                                                
           5'b10100 : compressor=3'b010;                                                                
           5'b10101 : compressor=3'b011;                                                                
           5'b10110 : compressor=3'b011;                                                                
           5'b10111 : compressor=3'b100;                                                                
           5'b11000 : compressor=3'b010;                                                                
           5'b11001 : compressor=3'b011;                                                                
           5'b11010 : compressor=3'b011;                                                                
           5'b11011 : compressor=3'b100;                                                                
           5'b11100 : compressor=3'b011;                                                                
           5'b11101 : compressor=3'b100;                                                                
           5'b11110 : compressor=3'b100;                                                                
           5'b11111 : compressor=3'b101;                                                                
           5'bxxxxx : compressor=3'bxxx;                                                                
   endcase                                                                                              
endmodule
