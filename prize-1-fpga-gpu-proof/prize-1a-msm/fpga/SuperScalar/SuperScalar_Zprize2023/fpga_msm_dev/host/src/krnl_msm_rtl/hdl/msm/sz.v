`include "basic.vh"
`include "env.vh"

module sz # (
    parameter                       DATA = 4                                      
    )
    (
    input       [DATA-1:0]          in,
    output      [DATA-1:0]          out
    );
    
    logic       [DATA-1:0]          inv;
    assign inv                      = ~in;

    genvar                          i;
    assign out[0]              = inv[0];
    generate 
        for(i=1;i<DATA;i=i+1)
        begin:a
        assign out[i]     = inv[i] && !(|inv[i-1:0]);
        end
    endgenerate

endmodule

