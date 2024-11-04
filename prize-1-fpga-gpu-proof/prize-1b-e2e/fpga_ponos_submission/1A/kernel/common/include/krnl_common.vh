  parameter integer C_AXI_S_CTRL_DATA_W = 32,
  parameter integer C_AXI_S_CTRL_ADDR_W = 8,
  parameter integer C_AXI_M_ID_W = 1,
  parameter integer C_AXI_M_ADDR_W = 64,
  parameter integer C_AXI_M_DATA_W = 512,

  parameter integer C_AXI_BURST_LEN = (C_AXI_M_DATA_W < 128) ? 256 : 4096*8 / C_AXI_M_DATA_W, // For small AXI data widths limit BURST_LEN to 256, otherwise keep the burst size 4096B irrespective of data width!
  parameter integer C_RD_MAX_OUTSTANDING = 7,

  parameter integer C_AXI_WR_BURST_LEN = 64,
  parameter integer C_AXI_RD_BURST_LEN = 64,

  parameter integer C_RST_DELAY = 30,

  parameter integer P_FUL_SCLR_W = 256,  // Width of original scalar 
  parameter integer P_RED_SCLR_W = 13,   // Width of reduced signed scalar i.e. C
  parameter integer P_TOTAL_WIN = (P_FUL_SCLR_W + P_RED_SCLR_W - 1) / P_RED_SCLR_W,
  parameter integer P_NUM_PROJ_COORD = 4,

  parameter integer P_INIT_ACCUM_MODE_IDX  = 0,
  parameter integer P_JUST_ACCUM_MODE_IDX  = 1,
  parameter integer P_ACCUM_WRITE_MODE_IDX = 2,

  parameter P_MUL_TYPE     = "montgomery", 
  parameter P_USE_DUMMY_CA = "false"     // switches to a very simplified curve-adder model for test purposes

