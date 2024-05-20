`include "basic.vh"
`include "env.vh"

//one cycle pipeline with no enable
module onepipe #(
    parameter WIDTH = 1
)(
//input        
    input                    valid_i,
    input        [WIDTH-1:0] data_i,
//output         
    (* SRL_STYLE = "register" *)output reg               valid_o,
    (* SRL_STYLE = "register" *)output reg   [WIDTH-1:0] data_o,
//Global
    input               clk,
    input               rstN
);
`Flop(valid_o, 1, valid_i)
`FlopN(data_o, 1, data_i)

endmodule
