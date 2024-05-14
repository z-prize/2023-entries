

`ifndef KRNL_MSM_381_PKG_SV
`define KRNL_MSM_381_PKG_SV

package krnl_msm_381_pkg;

  localparam unsigned P_DATA_PNT_W = 381;  // Width of data point
  localparam unsigned P_NUM_ACCU = 2;
  localparam unsigned P_FUL_SCLR_W = 256;  // Width of original scalar 
  localparam unsigned P_RED_SCLR_W = 13;  // Width of reduced signed scalar i.e. C
  localparam unsigned P_TOTAL_WIN = (P_FUL_SCLR_W + P_RED_SCLR_W - 1) / P_RED_SCLR_W;
  localparam unsigned P_NUM_WIN = (P_TOTAL_WIN + P_NUM_ACCU - 1)/ P_NUM_ACCU;  // 7=BLS12-377, 10=BLS12-381
  localparam unsigned P_NUM_AFF_COORD = 2;
  localparam unsigned Q  = 'h1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB;
  localparam unsigned QP = 'hCEB06106FEAAFC9468B316FEE268CF5819ECCA0E8EB2DB4C16EF2EF0C8E30B48286ADB92D9D113E889F3FFFCFFFCFFFD;

endpackage

`endif  // KRNL_MSM_381_PKG_SV