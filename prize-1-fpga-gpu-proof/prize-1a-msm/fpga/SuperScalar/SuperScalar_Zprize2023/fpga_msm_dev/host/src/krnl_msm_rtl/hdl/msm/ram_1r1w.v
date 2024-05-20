`include "basic.vh"
`include "env.vh"

module ram_1r1w#(
 parameter          NWORDS = 1024
,parameter          WORDSZ = 32
,parameter          ADDRSZ = `LOG2(NWORDS)
,parameter          WESZ = 4
,parameter          FPGA_RAM_STYLE = "block"
,parameter          FLOPOUT = 0         //if 1, flop out ram data, the latency is add one cycle. otherwise 0
,parameter          HOLD_DATA_OUT = 0   //if 1, hold the ram data out when no read. otherwise 0
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

ram_wrap_1r1w #(
     .HOLD_DATA_OUT(HOLD_DATA_OUT),
     .FPGA_RAM_STYLE(FPGA_RAM_STYLE),
     .FLOPOUT(FLOPOUT),
     .NWORDS (NWORDS),
     .WORDSZ (WORDSZ),
     .ADDRSZ (ADDRSZ),
     .WESZ   (WESZ      )
) 
    rami(
    .wclk   (wclk),
    .rclk   (rclk),
    .we     (we),
    .re     (re),
    .wem    (wem),
    .waddr  (waddr),
    .raddr  (raddr),
    .wdata  (wdata),
    .rdata  (rdata)
);

endmodule
