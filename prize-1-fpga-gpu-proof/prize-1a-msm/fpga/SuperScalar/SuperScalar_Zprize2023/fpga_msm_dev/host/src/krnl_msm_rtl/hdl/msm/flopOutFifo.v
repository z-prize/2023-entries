//**********************************************************************************************************************
//sync fifo
//the head of fifo is flop out, for timing
//other item is array, for power
//at least depth 4
//latency from wDat to rDat is at least 2 cycle
//**********************************************************************************************************************
//**********************************************************************************************************************
`include	"basic.vh"
`include	"env.vh"

module flopOutFifo
    #(
	parameter	RAM_STYLE			= "block",
	parameter	DATA				= 1,				///<p>	data width
				DEPTH				= 1			        ///<p>	fifo depth, include u1 and rDat, must >= 2
    )
    (
	input							rDatReady,			///<i>	read-data ready to be accepted
	output reg					rDatValid,			///<o>	read-data valid
	output reg[DATA	-1:0]		rDat,				///<o>	read-data
	///
	output							wDatReady,			///<o>	ready to accept write-data
	input							wDatValid,			///<i>	write-data valid
	input		[DATA	-1:0]		wDat,				///<i>	write-data
    //
    output reg  [`LOG2(DEPTH+1)-1:0]num,                //the current num of items in fifo
    input                           clear,
    input                           clk,
    input                           rstN
    );

    wire                            tmpValid;
    wire                            tmpReady;
    wire        [DATA   -1:0]       tmpDat;
    
    wire                            wack;
    wire                            rack;
    assign wack = wDatValid && wDatReady;
    assign rack = rDatValid && rDatReady;
    `Flop(num,  clear || (wack != rack), clear ? 0 : wack ? num + 1 : num - 1)
    
Fifo   
      #(
  	   .RAM_STYLE   (RAM_STYLE),
  	   .DATA	    (DATA     ),
	   .DEPTH	    (DEPTH-1  )		
        )u1 
        (
        .rDatReady(tmpReady    ),	
        .rDatValid(tmpValid    ),		
        .rDat     (tmpDat     ),			

        .wDatReady(wDatReady    ),		
        .wDatAlmostFull(),
        .wDatValid(wDatValid    ),		
        .wDat     (wDat     ),		

        .num        (   ),
        .clk     (clk   ),
        .rstN    (rstN  ),
        .clear   (clear    )
        );

`FlopAll(rDatValid, clear || tmpValid && tmpReady || rDatValid && rDatReady, clear ? 0 : tmpValid && tmpReady, clk, rstN, 0) 
`FlopN(rDat,    tmpValid && tmpReady, tmpDat)

    assign tmpReady = rDatValid ? rDatReady : 1;
        


endmodule
