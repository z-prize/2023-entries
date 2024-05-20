`include "basic.vh"
`include "env.vh"

module so # (
    parameter                       DATA = 4                                      
    )
    (
    input       [DATA-1:0]          in,
    output      [DATA-1:0]          out
    );

    genvar                          i;
    assign out[0]              = in[0];
    generate 
        for(i=1;i<DATA;i=i+1)
        begin:a
        assign out[i]     = in[i] && !(|in[i-1:0]);
        end
    endgenerate

endmodule
