`include "basic.vh"
`include "env.vh"
/*
backup pipe design
the pipe stage has addtional bpipe register to hold input, so that ready_pipe = !valid_bpipe
this design break the ready chain from ready_output to ready_input, for optimize timing

    input
     |
     |
     |__________________
     |    ________     |
     |    |      |     |      
    _V____V__    |     |      
    \_______/    |     |      
        |        |     |      
        V        |     V      
       pipe      |   bpipe  
        |        |______|    
        |        
        V
      output
*/


module backupPipe #(
    parameter WIDTH = 1
)(
//input        
    input                    valid_i,
    output                   ready_i,
    input        [WIDTH-1:0] data_i,
//output         
    output reg               valid_o,
    input                    ready_o,
    output reg   [WIDTH-1:0] data_o,
//Global
    input               clk,
    input               rstN
);
////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////

    `MAX_FANOUT32 reg                 valid_bpipe;
    reg     [WIDTH-1:0] data_bpipe;
    wire                ready_bpipe;

////////////////////////////////////////////////////////////////////////////////////////////////////////
//Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////

    assign ready_i      = !valid_bpipe;

    assign ready_bpipe  = valid_o ? ready_o : 1;


`FlopSC0(valid_o, valid_i && ready_i && !(valid_o && !ready_o) || valid_bpipe && ready_bpipe, valid_o && ready_o)
`FlopN(data_o   , valid_i && ready_i && !(valid_o && !ready_o) || valid_bpipe && ready_bpipe, valid_bpipe ? data_bpipe : data_i)

`FlopSC0(valid_bpipe, valid_i && ready_i && valid_o && !ready_o, valid_bpipe && ready_bpipe)
`FlopN(data_bpipe   , valid_i && ready_i && valid_o && !ready_o, data_i)



endmodule
