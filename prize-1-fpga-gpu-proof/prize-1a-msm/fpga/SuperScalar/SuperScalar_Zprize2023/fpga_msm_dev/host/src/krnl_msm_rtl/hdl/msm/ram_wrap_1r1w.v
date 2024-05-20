module ram_wrap_1r1w#(
 parameter          NWORDS = 1024
,parameter          WORDSZ = 32
,parameter          ADDRSZ = 10
,parameter          WESZ = 4
,parameter          FPGA_RAM_STYLE = "block"
,parameter          FLOPOUT = 0         //if 1, flop out ram data, the latency is add one cycle. otherwise 0
,parameter          HOLD_DATA_OUT = 1   //if 1, hold the ram data out when no read. otherwise 0. only valid when FLOPOUT == 0
) (
 input              wclk
,input              rclk
,input              we
,input              re
,input   [WESZ-1:0] wem
,input [ADDRSZ-1:0] waddr
,input [ADDRSZ-1:0] raddr
,input [WORDSZ-1:0] wdata
,output [WORDSZ-1:0] rdata
);

wire [WORDSZ-1:0] rdata_read;
reg  [WORDSZ-1:0] rdata_hold;
reg               mem_read_dly;

always@(posedge rclk) begin mem_read_dly <= re; end
always@(posedge rclk) begin if(mem_read_dly) rdata_hold <= rdata_read; end
assign rdata = FLOPOUT ? rdata_hold : HOLD_DATA_OUT ? (mem_read_dly ? rdata_read : rdata_hold) : rdata_read;

  ram_mdl_1r1w #(.NWORDS(NWORDS),.WORDSZ(WORDSZ),.ADDRSZ(ADDRSZ),.WESZ(WESZ),.RAM_STYLE(FPGA_RAM_STYLE)) 
  mem(.wclk(wclk),.rclk(rclk),.we(we),.re(re),.wem(wem),.waddr(waddr),.raddr(raddr),.wdata(wdata),.rdata(rdata_read));

endmodule
