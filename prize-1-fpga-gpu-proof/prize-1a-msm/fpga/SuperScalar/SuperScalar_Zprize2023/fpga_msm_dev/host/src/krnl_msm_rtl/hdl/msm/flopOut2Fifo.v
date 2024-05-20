//**********************************************************************************************************************
//sync fifo
//the head of fifo is flop out twice, for timing
//other item is array, for power
//at least depth 4
//latency from wDat to rDat is at least 3 cycle
//**********************************************************************************************************************
//**********************************************************************************************************************
`include	"basic.vh"
`include	"env.vh"

module flopOut2Fifo
    #(
	parameter	RAM_STYLE           = "block",
                DATA				= 1,				///<p>	data width
				DEPTH				= 1			        ///<p>	fifo depth, include u1 and rDat, must >= 2
    )
    (
	///
	input							wDatValid,			///<i>	write-data valid
	output							wDatReady,			///<o>	ready to accept write-data
    output                          wDatAlmostFull,     //has only one empty item
	input		[DATA	-1:0]		wDat,				///<i>	write-data
    ///
	output    					rDatValid,			///<o>	read-data valid
	input							rDatReady,			///<i>	read-data ready to be accepted
	output    [DATA	-1:0]		rDat,				///<o>	read-data
    //
    output reg  [`LOG2(DEPTH+1)-1:0]num,                //the current num of items in fifo
    input                           clear,
    input                           clk,
    input                           rstN
    );

`ifdef FPGA_ALTERA
(* dont_replicate *) reg					rDatValid_r;
(* dont_replicate *) reg[DATA	-1:0]		rDat_r;
`else
(* dont_touch="true" *)	reg					rDatValid_r;
(* dont_touch="true" *)	reg[DATA	-1:0]		rDat_r;
`endif
    wire                            t1Valid;
    wire                            t1Ready;
    wire        [DATA   -1:0]       t1Dat;
    
    reg                             t2Valid;
    wire                            t2Ready;
    reg         [DATA   -1:0]       t2Dat;

    
    wire                            wack;
    wire                            rack;
    assign wack = wDatValid && wDatReady;
    assign rack = rDatValid && rDatReady;
    `Flop(num,  clear || (wack != rack), clear ? 0 : wack ? num + 1 : num - 1)


    
Fifo   
      #(
       .RAM_STYLE   (RAM_STYLE),
  	   .DATA	    (DATA    ),
	   .DEPTH	    (DEPTH-1   )		
        )u1 
        (
        .wDatValid(wDatValid    ),		
        .wDatReady(wDatReady    ),		
        .wDatAlmostFull(wDatAlmostFull),
        .wDat     (wDat     ),		

        .rDatValid(t1Valid    ),		
        .rDatReady(t1Ready    ),	
        .rDat     (t1Dat     ),			

        .num        (          ),
        .clk     (clk   ),
        .rstN    (rstN  ),
        .clear   (clear    )
        );

`FlopAll(t2Valid, clear || t1Valid && t1Ready || t2Valid && t2Ready, clear ? 0 : t1Valid && t1Ready, clk, rstN, 0) 
`FlopN(t2Dat,    t1Valid && t1Ready, t1Dat)
    assign t1Ready = t2Valid ? t2Ready : 1;
        
`FlopAll(rDatValid_r, clear || t2Valid && t2Ready || rDatValid_r && rDatReady, clear ? 0 : t2Valid && t2Ready, clk, rstN, 0) 
`FlopN(rDat_r,    t2Valid && t2Ready, t2Dat)

assign rDatValid = rDatValid_r;
assign rDat      = rDat_r;

    assign t2Ready = rDatValid ? rDatReady : 1;
        


endmodule
