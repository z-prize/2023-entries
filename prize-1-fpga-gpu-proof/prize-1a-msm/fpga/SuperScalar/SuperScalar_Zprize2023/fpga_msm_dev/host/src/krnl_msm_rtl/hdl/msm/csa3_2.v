 `include "basic.vh"
`include "env.vh"

module csa3_2
#(
        parameter CSAWIDTH = 27
)
(                                                                                              
  input [CSAWIDTH -1:0] a,                                                                                              
  input [CSAWIDTH -1:0] b,                                                                                              
  input [CSAWIDTH -1:0] c,                                                                                              
  output[CSAWIDTH -1:0] s,                                                                                             
  output[CSAWIDTH -1:0] co
  );                                                                                                           
                                                                                                               

 wire [1:0]temp[CSAWIDTH];                                                                                         
genvar i;
generate 
   for (i=0;i<CSAWIDTH;i=i+1)begin:inst
compressor1 compressor (
        .x({c[i],b[i],a[i]}),
        .compressor(temp[i])
);
   assign s[i]=temp[i][0];                                                                            
   assign co[i]=temp[i][1];                                                                            
   end
endgenerate
                                                                                                               
                                                                                                               
endmodule                                                                                                    
