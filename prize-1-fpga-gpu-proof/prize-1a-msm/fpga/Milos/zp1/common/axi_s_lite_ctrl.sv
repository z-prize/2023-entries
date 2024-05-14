

`ifndef AXI_S_LITE_CTRL_SV
`define AXI_S_LITE_CTRL_SV

module axi_s_lite_ctrl #(
  parameter C_NUM_AXI_M_PORTS = 4,  // Maximum value is 4
  parameter C_S_AXI_ADDR_W = 8,
  parameter C_S_AXI_DATA_W = 32,
  parameter C_NUM_EXTRA_REGS = 1
) (
  // AXI4 Lite Slave signals
  input clk,
  input rst_n,
  input clk_en,

  output ap_start,
  input  ap_done,
  input  ap_ready,
  input  ap_idle,
  output interrupt,

  input  [  C_S_AXI_ADDR_W-1:0] axi_s_awaddr_i,
  input                         axi_s_awvalid_i,
  output                        axi_s_awready_o,
  input  [  C_S_AXI_DATA_W-1:0] axi_s_wdata_i,
  input  [C_S_AXI_DATA_W/8-1:0] axi_s_wstrb_i,
  input                         axi_s_wvalid_i,
  output                        axi_s_wready_o,
  input  [  C_S_AXI_ADDR_W-1:0] axi_s_araddr_i,
  input                         axi_s_avalid_i,
  output                        axi_s_arready_o,
  output [  C_S_AXI_DATA_W-1:0] axi_s_rdata_o,
  output [                 1:0] axi_s_rresp_o,
  output                        axi_s_rvalid_o,
  input                         axi_s_rready_i,
  output [                 1:0] axi_s_bresp_o,
  output                        axi_s_bvalid_o,
  input                         axi_s_bready_i,

  // User signals
  // Memory pointer and scalar registers parameterized 
  output [2*C_S_AXI_DATA_W-1:0] src_addr_array_o[C_NUM_AXI_M_PORTS-1:0],
  output [2*C_S_AXI_DATA_W-1:0] dst_addr_array_o[C_NUM_AXI_M_PORTS-1:0],
  output [  C_S_AXI_DATA_W-1:0] src_size_array_o[C_NUM_AXI_M_PORTS-1:0],
  output [  C_S_AXI_DATA_W-1:0] dst_size_array_o[C_NUM_AXI_M_PORTS-1:0],
  // Extra registers
  output [  C_S_AXI_DATA_W-1:0] reg_array_o     [ C_NUM_EXTRA_REGS-1:0]
  // Status and Debug registers - These registers are read only registers from the host CPU.
);
  //------------------------Address Info---------------------------------------------------
  // (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
  // +--------+---------------------------------------------------------------------------+
  // | OFFSET | DESCRIPTION                                                               |
  // +--------+---------------------------------------------------------------------------+
  // | 0x00   | Control signals                                                           |
  // |        | bit 0  - ap_start (Read/Write/COH)                                        |
  // |        | bit 1  - ap_done (Read/COR)                                               |
  // |        | bit 2  - ap_idle (Read)                                                   |
  // |        | bit 3  - ap_ready (Read)                                                  |
  // |        | bit 7  - auto_restart (Read/Write)                                        |
  // |        | others - reserved                                                         |
  // +--------+---------------------------------------------------------------------------+
  // | 0x04   | Global Interrupt Enable Register                                          |
  // |        | bit 0  - Global Interrupt Enable (Read/Write)                             |
  // |        | others - reserved                                                         |
  // +--------+---------------------------------------------------------------------------+
  // | 0x08   | ip Interrupt Enable Register (Read/Write)                                 |
  // |        | bit 0  - Channel 0 (ap_done)                                              |
  // |        | bit 1  - Channel 1 (ap_ready)                                             |
  // |        | others - reserved                                                         |
  // +--------+---------------------------------------------------------------------------+
  // | 0x0c   | ip Interrupt Status Register (Read/TOW)                                   |
  // |        | bit 0  - Channel 0 (ap_done)                                              |
  // |        | bit 1  - Channel 1 (ap_ready)                                             |
  // |        | others - reserved                                                         |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | Base   |      |       |        |                                                   |
  // | offset |      |       |        |                                                   |
  // |  port  | Port | Port  |  port  |                                                   | 
  // |   1    |   2  |  3    |   4    |                                                   |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | 0x10   | 0x28 | 0x40  | 0x58   | Data signal of src_addr_array_o LSB               |
  // |        |      |       |        | bit 31~0 -    src_addr_array_o[31:0] (Read/Write) |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | 0x14   | 0x2C | 0x44  | 0x5C   | Data signal of src_addr_array_o MSB               |
  // |        |      |       |        | bit 31~0 -    src_addr_array_o[63:32] (Read/Write)|
  // +--------+------+-------+--------+---------------------------------------------------+
  // | 0x18   | 0x30 | 0x48  | 0x60   | Data signal of dst_addr_array_o LSB               |
  // |        |      |       |        |  bit 31~0 - dst_addr_array_o[31:0] (Read/Write)   |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | 0x1c   | 0x34 | 0x4C  | 0x64   |  Data signal of dst_addr_array_o MSB              |
  // |        |      |       |        |  bit 31~0 - dst_addr_array_o[63:32] (Read/Write)  |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | 0x20   | 0x38 | 0x50  | 0x68   | Data signal of src_size_array_o                  |
  // |        |      |       |        | bit 31~0 - src_size_array_o[31:0] (Read/Write)   |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | 0x24   | 0x3C | 0x54  | 0x6C   | Data signal of dst_size_array_o                  |
  // |        |      |       |        | bit 31~0 - dst_size_array_o[31:0] (Read/Write)   |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | Registers accessible from AXI and RTL                                              |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | Base   |                                                                           |
  // | offset |                                                                           |
  // |  Reg   |                                                                           | 
  // |   1    |                                                                           |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | 0x70   | Data signal of reg_array_o                                                    |
  // |        | bit 31~0 - reg_array_o[31:0] (Read/Write)                                     |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | Read-only registers - Status and Debug values of the hw design.                    |
  // +--------+------+-------+--------+---------------------------------------------------+
  // | Base   |      |       |        |       |       |       |       | 
  // | offset |      |       |        |       |       |       |       |
  // | Reg 1  | Reg 2| Reg 3 | Reg 4  | Reg 5 | Reg 6 | Reg 7 | Reg 8 | 
  // +--------+------+-------+--------+-------+-------------------------------------------+
  // |  0xE0  | 0xE4 |  0xE8 |  0xEC  |  0xF0 |  0xF4 | 0xF8  |  0xFC |
  // +--------+------+-------+--------+-------+-------------------------------------------+

  //------------------------Address Parameter----------------------
  localparam C_ADDR_AP_CTRL = 8'h00;
  localparam C_ADDR_GIE = 8'h04;
  localparam C_ADDR_IER = 8'h08;
  localparam C_ADDR_ISR = 8'h0c;

  localparam C_BASE_OFFSET_SRC_ADDR_LSB = 8'h10;
  localparam C_BASE_OFFSET_SRC_ADDR_MSB = 8'h14;
  localparam C_BASE_OFFSET_DST_ADDR_LSB = 8'h18;
  localparam C_BASE_OFFSET_DST_ADDR_MSB = 8'h1c;
  localparam C_BASE_OFFSET_LENGTH_RD = 8'h20;
  localparam C_BASE_OFFSET_LENGTH_WR = 8'h24;

  localparam C_REGS_OFFSET_JUMPS = 8'h18;

  localparam C_BASE_RTL_REGS = 8'h70;
  //------------------------Local signal-------------------
  // Write state enumeration
  typedef enum logic [1:0] {
    WRIDLE = 2'd0,
    WRDATA = 2'd1,
    WRRESP = 2'd2
  } wr_state_e;
  wr_state_e wstate_d, wstate_q = WRIDLE;
  logic [C_S_AXI_ADDR_W-1:0] waddr_d, waddr_q;
  logic [C_S_AXI_DATA_W-1:0] wmask;
  logic                      aw_hs;
  logic                      w_hs;
  // Read state enumeration
  typedef enum logic [1:0] {
    RDIDLE = 2'd0,
    RDDATA = 2'd1
  } rd_state_e;
  rd_state_e rstate_d, rstate_q = RDIDLE;
  logic [  C_S_AXI_DATA_W-1:0] rdata;
  logic                        ar_hs;
  logic [  C_S_AXI_ADDR_W-1:0] raddr;
  // internal registers
  logic                        int_ap_idle;
  logic                        int_ap_ready;
  logic                        int_ap_done = 1'b0;
  logic                        int_ap_start = 1'b0;
  logic                        int_auto_restart = 1'b0;
  logic                        int_gie = 2'b0;
  logic [                 1:0] int_ier = 2'b0;
  logic [                 1:0] int_isr = 2'b0;
  logic [2*C_S_AXI_DATA_W-1:0] int_src_addr_array      [C_NUM_AXI_M_PORTS-1:0];
  logic [2*C_S_AXI_DATA_W-1:0] int_dst_addr_array      [C_NUM_AXI_M_PORTS-1:0];
  logic [  C_S_AXI_DATA_W-1:0] int_length_rd_array     [C_NUM_AXI_M_PORTS-1:0];
  logic [  C_S_AXI_DATA_W-1:0] int_length_wr_array     [C_NUM_AXI_M_PORTS-1:0];
  // Extra general purpose registers
  logic [  C_S_AXI_DATA_W-1:0] int_reg_array           [ C_NUM_EXTRA_REGS-1:0];

  //------------------------Instantiation------------------

  //------------------------AXI write fsm------------------
  assign axi_s_awready_o = rst_n & (wstate_q == WRIDLE);
  assign axi_s_wready_o  = (wstate_q == WRDATA);
  assign axi_s_bresp_o   = 2'b00;  // OKAY
  assign axi_s_bvalid_o  = (wstate_q == WRRESP);

  always_comb begin
    wmask = {
      {8{axi_s_wstrb_i[3]}}, {8{axi_s_wstrb_i[2]}}, {8{axi_s_wstrb_i[1]}}, {8{axi_s_wstrb_i[0]}}
    };
    aw_hs = axi_s_awvalid_i & axi_s_awready_o;
    w_hs = axi_s_wvalid_i & axi_s_wready_o;

    if (aw_hs) begin
      waddr_d = axi_s_awaddr_i[C_S_AXI_ADDR_W-1:0];
    end else begin
      waddr_d = waddr_q;
    end

    case (wstate_q)
      WRIDLE:
      if (axi_s_awvalid_i) begin
        wstate_d = WRDATA;
      end else begin
        wstate_d = WRIDLE;
      end
      WRDATA:
      if (axi_s_wvalid_i) begin
        wstate_d = WRRESP;
      end else begin
        wstate_d = WRDATA;
      end
      WRRESP:
      if (axi_s_bready_i) begin
        wstate_d = WRIDLE;
      end else begin
        wstate_d = WRRESP;
      end
      default: wstate_d = WRIDLE;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      waddr_q  <= '0;
      wstate_q <= WRIDLE;
    end else if (clk_en) begin
      waddr_q  <= waddr_d;
      wstate_q <= wstate_d;
    end
  end

  //------------------------AXI read fsm-------------------
  assign axi_s_arready_o = rst_n && (rstate_q == RDIDLE);
  assign axi_s_rdata_o   = rdata;
  assign axi_s_rresp_o   = 2'b00;  // OKAY
  assign axi_s_rvalid_o  = (rstate_q == RDDATA);

  always_comb begin
    ar_hs = axi_s_avalid_i & axi_s_arready_o;
    raddr = axi_s_araddr_i[C_S_AXI_ADDR_W-1:0];
    case (rstate_q)
      RDIDLE:
      if (axi_s_avalid_i) begin
        rstate_d = RDDATA;
      end else begin
        rstate_d = RDIDLE;
      end
      RDDATA:
      if (axi_s_rready_i & axi_s_rvalid_o) begin
        rstate_d = RDIDLE;
      end else begin
        rstate_d = RDDATA;
      end
      default: rstate_d = RDIDLE;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rstate_q <= RDIDLE;
    end else if (clk_en) begin
      rstate_q <= rstate_d;
    end
  end

  integer i = 0;
  integer j = 0;
  // rdata
  always_ff @(posedge clk) begin
    if (clk_en) begin
      if (ar_hs) begin
        rdata <= 1'b0;
        if (raddr < C_BASE_OFFSET_SRC_ADDR_LSB) begin
          // address offsets below the offset "C_BASE_OFFSET_SRC_ADDR_LSB" are hardened and fixed since the XRT uses them not the user.
          case (raddr)
            C_ADDR_AP_CTRL: begin
              rdata[0] <= int_ap_start;
              rdata[1] <= int_ap_done;
              rdata[2] <= int_ap_idle;
              rdata[3] <= int_ap_ready;
              rdata[7] <= int_auto_restart;
            end
            C_ADDR_GIE: begin
              rdata <= int_gie;
            end
            C_ADDR_IER: begin
              rdata <= int_ier;
            end
            C_ADDR_ISR: begin
              rdata <= int_isr;
            end
            default: ;
          endcase
        end else begin
          // user registers:
          // parameterized generation of the set of registers (three registers: source, destination and length) for each memory ports
          for (i = 0; i < C_NUM_AXI_M_PORTS; i++) begin : gen_loop_regs
            case (raddr)
              C_BASE_OFFSET_SRC_ADDR_LSB + (i * C_REGS_OFFSET_JUMPS): begin
                rdata <= int_src_addr_array[i][31:0];
              end
              C_BASE_OFFSET_SRC_ADDR_MSB + (i * C_REGS_OFFSET_JUMPS): begin
                rdata <= int_src_addr_array[i][63:32];
              end
              C_BASE_OFFSET_DST_ADDR_LSB + (i * C_REGS_OFFSET_JUMPS): begin
                rdata <= int_dst_addr_array[i][31:0];
              end
              C_BASE_OFFSET_DST_ADDR_MSB + (i * C_REGS_OFFSET_JUMPS): begin
                rdata <= int_dst_addr_array[i][63:32];
              end
              C_BASE_OFFSET_LENGTH_RD + (i * C_REGS_OFFSET_JUMPS): begin
                rdata <= int_length_rd_array[i][31:0];
              end
              C_BASE_OFFSET_LENGTH_WR + (i * C_REGS_OFFSET_JUMPS): begin
                rdata <= int_length_wr_array[i][31:0];
              end
              default: ;
            endcase
          end
          // Extra user registers
          for (j = 0; j < C_NUM_EXTRA_REGS; j++) begin : gen_loop_rtl_regs
            case (raddr)
              C_BASE_RTL_REGS + (j * 4): begin
                rdata <= int_reg_array[j][31:0];
              end
              default: ;
            endcase
          end
        end
      end
    end
  end

  //------------------------Register logic-----------------
  assign interrupt    = int_gie & (|int_isr);
  assign ap_start     = int_ap_start;
  assign int_ap_idle  = ap_idle;
  assign int_ap_ready = ap_ready;

  generate
    for (genvar i = 0; i < C_NUM_AXI_M_PORTS; i = i + 1) begin
      assign src_addr_array_o[i] = int_src_addr_array[i];
      assign dst_addr_array_o[i] = int_dst_addr_array[i];
      assign src_size_array_o[i] = int_length_rd_array[i];
      assign dst_size_array_o[i] = int_length_wr_array[i];
    end
    // connecting internal registers towards extra registers
    for (genvar j = 0; j < C_NUM_EXTRA_REGS; j = j + 1) begin
      assign reg_array_o[j] = int_reg_array[j];
    end
  endgenerate

  // int_ap_start
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      int_ap_start <= 1'b0;
    end else if (clk_en) begin
      if (w_hs && waddr_q == C_ADDR_AP_CTRL && axi_s_wstrb_i[0] && axi_s_wdata_i[0]) begin
        int_ap_start <= 1'b1;
      end else if (int_ap_ready) begin
        int_ap_start <= int_auto_restart;  // clear on handshake/auto restart
      end
    end
  end

  // int_ap_done
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      int_ap_done <= 1'b0;
    end else if (clk_en) begin
      if (ap_done) begin
        int_ap_done <= 1'b1;
      end else if (ar_hs && raddr == C_ADDR_AP_CTRL) begin
        int_ap_done <= 1'b0;  // clear on read
      end
    end
  end

  // int_auto_restart
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      int_auto_restart <= 1'b0;
    end else if (clk_en) begin
      if (w_hs && waddr_q == C_ADDR_AP_CTRL && axi_s_wstrb_i[0]) begin
        int_auto_restart <= axi_s_wdata_i[7];
      end
    end
  end

  // int_gie
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      int_gie <= 1'b0;
    end else if (clk_en) begin
      if (w_hs && waddr_q == C_ADDR_GIE && axi_s_wstrb_i[0]) begin
        int_gie <= axi_s_wdata_i[0];
      end
    end
  end

  // int_ier
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      int_ier <= 1'b0;
    end else if (clk_en) begin
      if (w_hs && waddr_q == C_ADDR_IER && axi_s_wstrb_i[0]) begin
        int_ier <= axi_s_wdata_i[1:0];
      end
    end
  end

  // int_isr[0]
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      int_isr[0] <= 1'b0;
    end else if (clk_en) begin
      if (int_ier[0] & ap_done) begin
        int_isr[0] <= 1'b1;
      end else if (w_hs && waddr_q == C_ADDR_ISR && axi_s_wstrb_i[0]) begin
        int_isr[0] <= int_isr[0] ^ axi_s_wdata_i[0];  // toggle on write
      end
    end
  end

  // int_isr[1]
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      int_isr[1] <= 1'b0;
    end else if (clk_en) begin
      if (int_ier[1] & ap_ready) begin
        int_isr[1] <= 1'b1;
      end else if (w_hs && waddr_q == C_ADDR_ISR && axi_s_wstrb_i[0]) begin
        int_isr[1] <= int_isr[1] ^ axi_s_wdata_i[1];  // toggle on write
      end
    end
  end

  generate
    for (genvar i = 0; i < C_NUM_AXI_M_PORTS; i++) begin
      // int_src_addr_array[i][31:0]
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          int_src_addr_array[i][31:0] <= 0;
        end else if (clk_en) begin
          if (w_hs && waddr_q == C_BASE_OFFSET_SRC_ADDR_LSB + (i * C_REGS_OFFSET_JUMPS)) begin
            int_src_addr_array[i][31:0] <= (axi_s_wdata_i[31:0] & wmask) | (int_src_addr_array[i][31:0] & ~wmask);
          end
        end
      end

      // int_src_addr_array[i][63:32]
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          int_src_addr_array[i][63:32] <= 0;
        end else if (clk_en) begin
          if (w_hs && waddr_q == C_BASE_OFFSET_SRC_ADDR_MSB + (i * C_REGS_OFFSET_JUMPS)) begin
            int_src_addr_array[i][63:32] <= (axi_s_wdata_i[31:0] & wmask) | (int_src_addr_array[i][63:32] & ~wmask);
          end
        end
      end

      // int_dst_addr_array[i][31:0]
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          int_dst_addr_array[i][31:0] <= 0;
        end else if (clk_en) begin
          if (w_hs && waddr_q == C_BASE_OFFSET_DST_ADDR_LSB + (i * C_REGS_OFFSET_JUMPS)) begin
            int_dst_addr_array[i][31:0] <= (axi_s_wdata_i[31:0] & wmask) | (int_dst_addr_array[i][31:0] & ~wmask);
          end
        end
      end

      // int_dst_addr_array[i][63:32]
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          int_dst_addr_array[i][63:32] <= 0;
        end else if (clk_en) begin
          if (w_hs && waddr_q == C_BASE_OFFSET_DST_ADDR_MSB + (i * C_REGS_OFFSET_JUMPS)) begin
            int_dst_addr_array[i][63:32] <= (axi_s_wdata_i[31:0] & wmask) | (int_dst_addr_array[i][63:32] & ~wmask);
          end
        end
      end

      // int_length_rd_array[i][31:0]
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          int_length_rd_array[i][31:0] <= 0;
        end else if (clk_en) begin
          if (w_hs && waddr_q == C_BASE_OFFSET_LENGTH_RD + (i * C_REGS_OFFSET_JUMPS)) begin
            int_length_rd_array[i][31:0] <= (axi_s_wdata_i[31:0] & wmask) | (int_length_rd_array[i][31:0] & ~wmask);
          end
        end
      end

      // int_length_wr_array[i][31:0]
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          int_length_wr_array[i][31:0] <= 0;
        end else if (clk_en) begin
          if (w_hs && waddr_q == C_BASE_OFFSET_LENGTH_WR + (i * C_REGS_OFFSET_JUMPS)) begin
            int_length_wr_array[i][31:0] <= (axi_s_wdata_i[31:0] & wmask) | (int_length_wr_array[i][31:0] & ~wmask);
          end
        end
      end
    end

    for (genvar j = 0; j < C_NUM_EXTRA_REGS; j++) begin
      // int_reg_array[i][31:0]
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          int_reg_array[j][31:0] <= 0;
        end else if (clk_en) begin
          if (w_hs && waddr_q == C_BASE_RTL_REGS + (j * 4)) begin
            int_reg_array[j][31:0] <= (axi_s_wdata_i[31:0] & wmask) | (int_reg_array[j][31:0] & ~wmask);
          end
        end
      end
    end
  endgenerate
  //------------------------Memory logic-------------------

endmodule

`endif  // AXI_S_LITE_CTRL_SV
