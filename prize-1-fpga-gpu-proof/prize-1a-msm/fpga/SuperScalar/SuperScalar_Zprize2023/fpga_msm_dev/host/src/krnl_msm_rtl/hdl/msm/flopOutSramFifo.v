//**********************************************************************************************************************
//sync fifo use sram
//actually the max num is DEPTH + 2
//flop out
//writeValid to readValid latency is at least 3 cycle
//**********************************************************************************************************************
//**********************************************************************************************************************
`include	"basic.vh"
`include	"env.vh"

module flopOutSramFifo
    #(
	parameter	DATA				= 1,				///<p>	data width
				DEPTH				= 1 			    ///<p>	sync-fifo depth
    )
    (
	output                          rDatValid,			///<o>	read-data valid
	input							rDatReady,			///<i>	read-data ready to be accepted
	output      [DATA	-1:0]		rDat,				///<o>	read-data
	///
	input							wDatValid,			///<i>	write-data valid
	`MAX_FANOUT32 output reg                      wDatReady,			///<o>	ready to accept write-data
	input		[DATA	-1:0]		wDat,				///<i>	write-data
	///
    output reg  [`LOG2(DEPTH+2+1)-1:0]num,                //the current num of items in fifo
    input                           clear,   
	input							clk,
	input							rstN
    );

    localparam ADDRWIDTH           = `LOG2(DEPTH);    
    //there 4 stage pipeline
    //1. w   write to sram
    //2. r1  setup read sram
    //3. r2  data out from sram
    //4. r3  flop out read data
    
	reg 		[ADDRWIDTH -1:0]	wp;
    reg                             wp_wrap;

    //use internal
	reg 		[ADDRWIDTH -1:0]    rp;                
    reg                             rp_wrap;           

    wire                            wack;
    wire                            rack;
    wire                            rack_r1;
    reg  [`LOG2(DEPTH+1)-1:0]       num_r1;

	wire							r1DatValid;			
	wire  						    r1DatReady;			

	reg 							r2DatValid;			
	wire  						    r2DatReady;			
    //sram's Q always hold data until new read

	reg 							r3DatValid;			
	wire  						    r3DatReady;			
    reg         [DATA   -1:0]       r3Dat;

    
    wire        [DATA   -1:0]       sramQ;

`Flop(wp,      clear || wDatValid && wDatReady,                  clear ? 0 : wp == DEPTH-1 ? 0 : wp + 1)
`Flop(wp_wrap, clear || wDatValid && wDatReady && wp == DEPTH-1, clear ? 0 : !wp_wrap                 )

`Flop(rp,      clear || r1DatValid && r1DatReady,                  clear ? 0 : rp == DEPTH-1 ? 0 : rp + 1)
`Flop(rp_wrap, clear || r1DatValid && r1DatReady && rp == DEPTH-1, clear ? 0 : !rp_wrap                 )


    assign rack_r1 = r1DatValid && r1DatReady;
    `Flop(num_r1,  clear || (wack != rack_r1), clear ? 0 : wack ? num_r1 + 1 : num_r1 - 1)

    wire   wDatReady_golden;
    assign wDatReady_golden        = !(wp == rp && wp_wrap != rp_wrap);
`FlopAll(wDatReady, 1, num_r1 == DEPTH && rack_r1 || num_r1 == DEPTH-1 && !(wack && !rack_r1) || num_r1 < DEPTH-1, clk, rstN, 1)
`assert_clk(wDatReady == wDatReady_golden)
`assert_clk(wDatReady_golden == (num_r1 < DEPTH))

    assign r1DatValid        = !(wp == rp && wp_wrap == rp_wrap);

    assign r1DatReady        = r2DatValid ? r2DatReady : 1;

    assign r2DatReady        = r3DatValid ? r3DatReady : 1;

`Flop(r2DatValid,   clear || (r1DatValid && r1DatReady) != (r2DatValid && r2DatReady),    clear ? 0 : (r1DatValid && r1DatReady) ? 1 : 0)

`Flop(r3DatValid,   clear || (r2DatValid && r2DatReady) != (r3DatValid && r3DatReady),    clear ? 0 : (r2DatValid && r2DatReady) ? 1 : 0)
`FlopN(r3Dat,        r2DatValid && r2DatReady, sramQ)

    assign rDatValid        = r3DatValid;
    assign rDat             = r3Dat;
    assign r3DatReady       = rDatReady;

    assign wack = wDatValid && wDatReady;
    assign rack = rDatValid && rDatReady;
    `Flop(num,  clear || (wack != rack), clear ? 0 : wack ? num + 1 : num - 1)


`ifdef FPGA_ALTERA_RAM
/*
a5_ram_2p_nomask #(
.NWORDS(DEPTH),
.WORDSZ(DATA)
)
ram(
	.clock        (clk),
	.data         (wDat),
	.rdaddress    (rp),
	.rden         (r1DatValid && r1DatReady),
	.wraddress    (wp),
	.wren         (wDatValid && wDatReady),
	.q            (sramQ)
);
*/
    //only below code can be infer to RAM
    localparam NWORDS = DEPTH;
    localparam WORDSZ = DATA;
    localparam ADDRSZ = `LOG2(NWORDS);
    wire               we   ;
    wire               re   ;
    wire  [ADDRSZ-1:0] waddr    ;
    wire  [ADDRSZ-1:0] raddr    ;
    wire  [WORDSZ-1:0] wdata    ;
    reg [WORDSZ-1:0] rdata  ;
    assign we = wDatValid && wDatReady;
    assign waddr = wp;
    assign wdata = wDat;
    assign re = r1DatValid && r1DatReady;
    assign raddr = rp;
    assign sramQ = rdata;
    
`ifdef FPGA_ALTERA_A2
    (* ramstyle= "M9K" *)  reg [WORDSZ-1:0] mem[NWORDS-1:0];      
`else
    `ifdef FPGA_ALTERA_S10
    (* ramstyle= "M20K" *)  reg [WORDSZ-1:0] mem[NWORDS-1:0];      
    `elsif FPGA_ALTERA_AGILEX
    (* ramstyle= "M20K" *)  reg [WORDSZ-1:0] mem[NWORDS-1:0];      
    `else
    (* ramstyle= "M10K" *)  reg [WORDSZ-1:0] mem[NWORDS-1:0];      
    `endif
`endif

    always@(posedge clk) if(we) mem[waddr] <= wdata;
    always@(posedge clk) if(re) rdata <= mem[raddr];
`else
ram_1r1w #(
     .NWORDS (DEPTH      ),
     .WORDSZ (DATA      ),
     .ADDRSZ (`LOG2(DEPTH)),
     .WESZ   (1      ),
     .FPGA_RAM_STYLE("block"),
     .FLOPOUT(0),
     .HOLD_DATA_OUT(0)
) 
    ram(
    .wclk   (clk),
    .rclk   (clk),
    .we     (wDatValid && wDatReady),
    .re     (r1DatValid && r1DatReady),
    .wem    (1'b1),       //the same as we
    .waddr  (wp),
    .raddr  (rp),
    .wdata  (wDat),
    .rdata  (sramQ)
);
`endif



endmodule
