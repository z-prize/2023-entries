//**********************************************************************************************************************
//normal fifo
//**********************************************************************************************************************
//**********************************************************************************************************************
`include	"basic.vh"
`include	"env.vh"

module Fifo
    #(
	parameter	RAM_STYLE           = "block",
                DATA				= 1,				///<p>	data width
				DEPTH				= 1,			    ///<p>	sync-fifo depth
                //RDYFWD              = 0,                ///<p>  to forward ready from read to write
                ADDRWIDTH           = `LOG2(DEPTH)
    )
    (
	///
	input							wDatValid,			///<i>	write-data valid
	output reg  					wDatReady,			///<o>	ready to accept write-data
    output reg                      wDatAlmostFull,     //equal to num == DEPTH-1
	input		[DATA	-1:0]		wDat,				///<i>	write-data
	///
	output							rDatValid,			///<o>	read-data valid
	input							rDatReady,			///<i>	read-data ready to be accepted
	output		[DATA	-1:0]		rDat,				///<o>	read-data
	///
    output reg  [`LOG2(DEPTH+1)-1:0]num,                //the current num of items in fifo
	input							clk,
	input							rstN,
    input                           clear            
    );

    
`ifdef FPGA_ALTERA
(* ramstyle = RAM_STYLE *)	reg			[DATA	-1:0]		arr[DEPTH-1:0];		//*	[flop]
`else
(* ram_style= RAM_STYLE *)	reg			[DATA	-1:0]		arr[DEPTH-1:0];		//*	[flop]
`endif
	reg 		[ADDRWIDTH -1:0]    rp;
	reg 		[ADDRWIDTH -1:0]	wp;
    reg                             rp_wrap;
    reg                             wp_wrap;
    integer                         i;


    wire                            wack;
    wire                            rack;
    assign wack = wDatValid && wDatReady;
    assign rack = rDatValid && rDatReady;

    `Flop(num,  clear || (wack != rack), clear ? 0 : wack ? num + 1 : num - 1)

    `Flop(wDatAlmostFull, 1, clear ? 0 : num == DEPTH && !wack && rack || num == DEPTH-1 && wack == rack || num == DEPTH-2 && wack && !rack)


    always@(posedge clk)
        if(wDatValid && wDatReady)
            arr[wp]      <= `CQ wDat;

	assign							rDat					= arr[rp];
    
`FlopAll(wp,      clear || wDatValid && wDatReady,                  clear ? 0 : wp == DEPTH-1 ? 0 : wp + 1, clk, rstN, 0)
`FlopAll(wp_wrap, clear || wDatValid && wDatReady && wp == DEPTH-1, clear ? 0 : !wp_wrap,                   clk, rstN, 0)

`FlopAll(rp,      clear || rDatValid && rDatReady,                  clear ? 0 : rp == DEPTH-1 ? 0 : rp + 1, clk, rstN, 0)
`FlopAll(rp_wrap, clear || rDatValid && rDatReady && rp == DEPTH-1, clear ? 0 : !rp_wrap,                   clk, rstN, 0)


    //assign wDatReady        = RDYFWD && rDatReady ||
    //                            !(wp == rp && wp_wrap != rp_wrap);
`FlopAll(wDatReady, 1, !(num == DEPTH && !rack || num == DEPTH-1 && wack && !rack), clk, rstN, 1)

`assert_clk(wDatReady == !(wp == rp && wp_wrap != rp_wrap))

    assign rDatValid        = 
                                !(wp == rp && wp_wrap == rp_wrap);

endmodule
