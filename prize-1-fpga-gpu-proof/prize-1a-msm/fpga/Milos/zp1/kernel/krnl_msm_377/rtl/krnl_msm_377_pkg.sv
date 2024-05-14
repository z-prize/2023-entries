
`ifndef KRNL_MSM_377_PKG_SV
`define KRNL_MSM_377_PKG_SV

package krnl_msm_377_pkg;

  localparam unsigned P_DATA_PNT_W = 377;  // Width of data point
  localparam unsigned P_NUM_ACCU = 3;
  localparam unsigned P_FUL_SCLR_W = 256;  // Width of original scalar 
  localparam unsigned P_RED_SCLR_W = 13;  // Width of reduced signed scalar i.e. C
  localparam unsigned P_TOTAL_WIN = (P_FUL_SCLR_W + P_RED_SCLR_W - 1) / P_RED_SCLR_W;
  localparam unsigned P_NUM_WIN = (P_TOTAL_WIN + P_NUM_ACCU - 1)/ P_NUM_ACCU;  // 7=BLS12-377, 10=BLS12-381
  localparam unsigned P_NUM_AFF_COORD = 3;
  localparam unsigned Q = 'h1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001;

endpackage

`endif  // KRNL_MSM_377_PKG_SV