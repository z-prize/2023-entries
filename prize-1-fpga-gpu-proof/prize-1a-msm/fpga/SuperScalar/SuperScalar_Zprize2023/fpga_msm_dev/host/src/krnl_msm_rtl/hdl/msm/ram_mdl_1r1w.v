module ram_mdl_1r1w #(
 parameter          NWORDS = 1024
,parameter          WORDSZ = 32
,parameter          ADDRSZ = 10
,parameter          WESZ = 4
,parameter          RAM_STYLE = "block"
) (
 input              wclk
,input              rclk
,input              we
,input              re
,input   [WESZ-1:0] wem
,input [ADDRSZ-1:0] waddr
,input [ADDRSZ-1:0] raddr
,input [WORDSZ-1:0] wdata
,output reg [WORDSZ-1:0] rdata
);

initial
begin
  $display("warning: using memory model in module %M");
  if(((WORDSZ/WESZ)*WESZ)!=WORDSZ)
  begin
    $display("ram parameter just a warning! WORDSZ is not a multiple of WESZ in module %M");
    $display("WORDSZ = %d; WESZ = %d",WORDSZ,WESZ);
  end
  if(NWORDS>(2**ADDRSZ))
  begin
    $display("ram parameter error! NWORDS is bigger than 2**ADDRSZ in module %M");
    $display("NWORDS = %d; ADDRSZ = %d",NWORDS,ADDRSZ);
    $finish;
  end
end

localparam SLCSZ = WORDSZ/WESZ;
`ifdef FPGA_ALTERA
    `ifdef FPGA_ALTERA_A5
initial begin
  $display("DEBUG: FPGA_ALTERA & FPGA_ALTERA_A5, want to M10K!!!!");
end
    (* ramstyle= "M10K" *)  reg [WORDSZ-1:0] mem[0:NWORDS-1];      //Altera always use block memory
    `elsif FPGA_ALTERA_A2
    (* ramstyle= "M9K" *)   reg [WORDSZ-1:0] mem[0:NWORDS-1];   //Altera always use block memory
    `elsif FPGA_ALTERA_S10
    (* ramstyle= "M20K" *)  reg [WORDSZ-1:0] mem[0:NWORDS-1];    //Altera always use block memory
    `elsif FPGA_ALTERA_AGILEX
    (* ramstyle= "M20K" *)  reg [WORDSZ-1:0] mem[0:NWORDS-1];    //Altera always use block memory
    `else
    (* ramstyle= "M20K" *)  reg [WORDSZ-1:0] mem[0:NWORDS-1];    //Altera always use block memory
    `endif
`else//FPGA_ALTERA
    //Xilinx
    // reg [WORDSZ-1:0] mem[0:NWORDS-1];
    (* ram_style= RAM_STYLE *) reg [WORDSZ-1:0] mem[0:NWORDS-1];
`endif//FPGA_ALTERA

    genvar slc_size;
    //genvar slc_size = WESZ == 1          ? SLCSZ : 
    //                  slc_num < WESZ - 1 ? SLCSZ : 
    //                                       WORDSZ - SLCSZ * (WESZ-1);
genvar slc_num;
generate
    for(slc_num=0;slc_num<WESZ;slc_num=slc_num+1) begin:slc_U
    //handle when WORDSZ is not a multiple of WESZ
    always@(posedge wclk) if(we&&wem[slc_num]) mem[waddr][slc_num*SLCSZ+:WESZ == 1          ? SLCSZ :slc_num < WESZ - 1 ? SLCSZ :WORDSZ - SLCSZ * (WESZ-1)] <= wdata[slc_num*SLCSZ+:WESZ == 1          ? SLCSZ :slc_num < WESZ - 1 ? SLCSZ :WORDSZ - SLCSZ * (WESZ-1)];
    end
endgenerate

    always@(posedge rclk) if(re) rdata <= mem[raddr];

/*
localparam SLCSZ = WORDSZ/WESZ;
genvar slc_num;
generate
  for(slc_num=0;slc_num<WESZ;slc_num=slc_num+1)
  begin:mem
`ifdef FPGA_ALTERA
    `ifdef FPGA_ALTERA_A5
initial
begin
  $display("DEBUG: FPGA_ALTERA & FPGA_ALTERA_A5, want to M10K!!!!");
end
    (* ramstyle= "M10K" *)  reg [SLCSZ-1:0] mem[0:NWORDS-1];      //Altera always use block memory
	 `endif
    `ifdef FPGA_ALTERA_A2
    (* ramstyle= "M9K" *)   reg [SLCSZ-1:0] mem[0:NWORDS-1];   //Altera always use block memory
	 `endif
    `ifdef FPGA_ALTERA_S10
    (* ramstyle= "M20K" *)  reg [SLCSZ-1:0] mem[0:NWORDS-1];    //Altera always use block memory
    `endif
`else
    (* ram_style= RAM_STYLE *) reg [SLCSZ-1:0] mem[0:NWORDS-1];
`endif
    always@(posedge wclk) if(we&&wem[slc_num]) mem[waddr] <= wdata[slc_num*SLCSZ+SLCSZ-1:slc_num*SLCSZ];
    always@(posedge rclk) if(re) rdata[slc_num*SLCSZ+SLCSZ-1:slc_num*SLCSZ] <= mem[raddr];
  end
endgenerate
*/

endmodule
