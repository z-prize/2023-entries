

`ifndef ACC_CTRL_SV
`define ACC_CTRL_SV

module acc_ctrl #(
  parameter int P_DATA_PNT_W = 377,  // Data point width, 377: BLS12-377, 381: BLS12-381
  parameter int P_NUM_WIN = 7,  // Number of bucket sets: 7 or 10
  parameter int P_BM_OUTPUT_STAGE_NUM = 9,  // Number of clk cycles delay in BM: addr -> bucket_val
  parameter int P_NUM_AFF_COORD = 3,
  `include "krnl_common.vh"
) (
  input clk,
  input rst_n,

  // Control interface: ap_start/ap_done from XRT and mode from axi4-lite control reg
  input              ctrl_ap_start_i,
  input        [2:0] ctrl_mode_i,
  output logic       ctrl_ap_done_o,

  // Signal from merger indicating all scalar/points have been processed i.e. merger table completely empty
  input mrgr_merged_last_i,

  // Interface with the scheduler
  input        [P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] sch_point_i,
  input        [                P_RED_SCLR_W-2:0] sch_bucket_addr_i,
  input        [           $clog2(P_NUM_WIN)-1:0] sch_bucket_set_addr_i,
  input                                           sch_add_sub_i,
  input                                           sch_valid_i,
  output logic                                    sch_init_done_o,

  // Interface with bucket memory
  output logic [                 P_RED_SCLR_W-2:0] bm_bucket_addr_o,
  output logic [            $clog2(P_NUM_WIN)-1:0] bm_bucket_set_addr_o,
  output logic                                     bm_clear_bucket_o,
  input        [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] bm_bucket_val_i,

  // Interface towards curve adder
  output [P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] ca_point_o,
  output [                P_RED_SCLR_W-2:0] ca_bucket_addr_o,
  output [           $clog2(P_NUM_WIN)-1:0] ca_bucket_set_addr_o,
  output                                    ca_add_sub_o,
  output                                    ca_valid_o,

  // AXI-S interface towards bucket writer
  output [P_DATA_PNT_W-1:0] bw_bucket_val_o,
  output                    bw_valid_o,
  output                    bw_last_o,
  input                     bw_ready_i
);

  // Nr of clock cycles before we can be sure that last accumulation is written into BM
  localparam unsigned LP_BUCKET_WR_DLY_CNT_MAX = 119;
  localparam unsigned LP_BUCKET_WR_DLY_CNT_W = $clog2(LP_BUCKET_WR_DLY_CNT_MAX);


  // Address of last bucket in last bucket set.
  localparam unsigned LP_CUR_BUCKET_ADDR_W = $clog2(P_NUM_WIN) + (P_RED_SCLR_W - 1);
  localparam unsigned LP_CUR_BUCKET_ADDR_MAX = ((P_NUM_WIN - 1) << (P_RED_SCLR_W - 1)) + 2**(P_RED_SCLR_W -1) - 1;

  // verilog_format: off
  typedef enum logic [2:0] {
    ST_MODE_DETECT        = 3'd0, // Wakeup state to determine next based on mode
    ST_INIT               = 3'd1, // State to clear bucket memory with identity val
    ST_ACCUM              = 3'd2, // Normal accumulation state
    ST_AWAIT_BM_WRT_COMPL = 3'd3, // State to wait for all remaining CA data to be written in bucket memory
    ST_WRITE_DDR_CLR_BMEM = 3'd4, // State for writing bucket memory contents to DDR
    ST_DONE               = 3'd5
  } acc_ctrl_st;
  acc_ctrl_st                                     cur_state;
  // verilog_format: on

  logic [         LP_CUR_BUCKET_ADDR_W-1:0] cur_bucket_addr;
  logic [         LP_CUR_BUCKET_ADDR_W-1:0] cur_bucket_addr_dld;

  logic [       LP_BUCKET_WR_DLY_CNT_W-1:0] bucket_write_dly_cntr;

  logic [ P_NUM_AFF_COORD*P_DATA_PNT_W-1:0] ca_point;
  logic                                     ca_add_sub;
  logic                                     ca_valid;

  logic                                     axis_ready_int;

  logic [                              1:0] proj_coord_cntr;

  logic                                     axis_valid_int_dld;
  logic [                              1:0] proj_coord_cntr_dld;

  logic                                     axis_valid_int_reg;
  logic [                              1:0] proj_coord_cntr_reg;
  logic [P_NUM_PROJ_COORD*P_DATA_PNT_W-1:0] bm_bucket_val_reg;
  logic [                 P_DATA_PNT_W-1:0] fifo_in_data;

  logic [                 P_DATA_PNT_W-1:0] cdc_fifo_out_data;

  logic                                     axis_last;
  logic                                     axis_last_dld;
  logic                                     axis_last_reg;

  // Central statemachine performing control task
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cur_state <= ST_MODE_DETECT;
      cur_bucket_addr <= 0;
      proj_coord_cntr <= P_NUM_PROJ_COORD - 1;
      bucket_write_dly_cntr <= 0;
      sch_init_done_o <= 1'b0;
      ctrl_ap_done_o <= 1'b0;
      bm_clear_bucket_o <= 1'b0;
      ca_point <= 0;
      ca_add_sub <= 1'b0;
      ca_valid <= 1'b0;
    end else begin
      case (cur_state)
        ST_MODE_DETECT: begin
          if (ctrl_ap_start_i) begin
            proj_coord_cntr <= P_NUM_PROJ_COORD - 1;
            if (ctrl_mode_i[P_INIT_ACCUM_MODE_IDX]) begin
              cur_bucket_addr <= 0;
              bm_clear_bucket_o <= 1'b1;
              cur_state <= ST_INIT;
            end else if (ctrl_mode_i[P_JUST_ACCUM_MODE_IDX] | ctrl_mode_i[P_ACCUM_WRITE_MODE_IDX]) begin
              sch_init_done_o <= 1'b1;
              cur_state <= ST_ACCUM;
            end
          end
        end
        ST_INIT: begin
          if (cur_bucket_addr < LP_CUR_BUCKET_ADDR_MAX) begin
            cur_bucket_addr <= cur_bucket_addr + 1;
            sch_init_done_o <= 1'b0;
          end else begin
            cur_bucket_addr <= 0;
            bm_clear_bucket_o <= 1'b0;
            ctrl_ap_done_o <= 1'b0;
            cur_state <= ST_ACCUM;
            sch_init_done_o <= 1'b1;
          end
        end
        ST_ACCUM: begin
          ca_point <= sch_point_i;
          cur_bucket_addr <= {sch_bucket_set_addr_i, sch_bucket_addr_i};
          ca_add_sub <= sch_add_sub_i;
          ca_valid <= sch_valid_i;
          if (mrgr_merged_last_i) begin
            if (ctrl_mode_i[P_ACCUM_WRITE_MODE_IDX]) begin
              // Merger indicates all is done:go to await state for completion of write to BM
              cur_state <= ST_AWAIT_BM_WRT_COMPL;
              ctrl_ap_done_o <= 1'b0;
            end else begin
              cur_state <= ST_DONE;
              ctrl_ap_done_o <= 1'b1;
            end
          end
        end
        ST_AWAIT_BM_WRT_COMPL: begin
          if (bucket_write_dly_cntr < LP_BUCKET_WR_DLY_CNT_MAX) begin
            bucket_write_dly_cntr <= bucket_write_dly_cntr + 1;
          end else begin
            // Buckets now properly written: go to done in when doing just accum, otherwise write to DDR first
            bucket_write_dly_cntr <= 0;
            bm_clear_bucket_o <= 1'b1;
            cur_bucket_addr <= 0;
            proj_coord_cntr <= 0;
            cur_state <= ST_WRITE_DDR_CLR_BMEM;
          end
        end
        ST_WRITE_DDR_CLR_BMEM: begin
          if (axis_ready_int) begin
            if (proj_coord_cntr < P_NUM_PROJ_COORD - 1) begin
              proj_coord_cntr <= proj_coord_cntr + 1;
            end else begin
              proj_coord_cntr <= 0;
              if (cur_bucket_addr < LP_CUR_BUCKET_ADDR_MAX) begin
                cur_bucket_addr <= cur_bucket_addr + 1;
              end else begin
                cur_bucket_addr <= 0;
                bm_clear_bucket_o <= 1'b0;
                cur_state <= ST_DONE;
              end
            end
          end
        end
        ST_DONE: begin
          ctrl_ap_done_o <= 1'b0;
          sch_init_done_o <= 1'b0;
          cur_state <= ST_MODE_DETECT;
        end
      endcase
    end
  end

  assign {bm_bucket_set_addr_o, bm_bucket_addr_o} = cur_bucket_addr;

  assign axis_last = (cur_bucket_addr == LP_CUR_BUCKET_ADDR_MAX) && (proj_coord_cntr == P_NUM_PROJ_COORD - 1);

  // Delay point to align with output of bucket memory
  shiftreg #(
    .SHIFT (P_BM_OUTPUT_STAGE_NUM),
    .DATA_W(P_NUM_AFF_COORD * P_DATA_PNT_W)
  ) i_shiftreg_point (
    .clk,
    .data_i(ca_point),
    .data_o(ca_point_o)
  );

  // Delay cur bucket_addr to align with output of bucket memory
  shiftreg #(
    .SHIFT (P_BM_OUTPUT_STAGE_NUM),
    .DATA_W(LP_CUR_BUCKET_ADDR_W)
  ) i_shiftreg_bucket_addr (
    .clk,
    .data_i(cur_bucket_addr),
    .data_o(cur_bucket_addr_dld)
  );

  assign {ca_bucket_set_addr_o, ca_bucket_addr_o} = cur_bucket_addr_dld;

  shiftreg #(
    .SHIFT (P_BM_OUTPUT_STAGE_NUM),
    .DATA_W(1)
  ) i_shiftreg_add_sub (
    .clk,
    .data_i(ca_add_sub),
    .data_o(ca_add_sub_o)
  );

  shiftreg #(
    .SHIFT (P_BM_OUTPUT_STAGE_NUM),
    .DATA_W(1)
  ) i_shiftreg_ca_valid (
    .clk,
    .data_i(ca_valid),
    .data_o(ca_valid_o)
  );

  shiftreg #(
    .SHIFT (P_BM_OUTPUT_STAGE_NUM),
    .DATA_W(1)
  ) i_shiftreg_axis_valid_int (
    .clk,
    .data_i(axis_ready_int & (cur_state == ST_WRITE_DDR_CLR_BMEM)),
    .data_o(axis_valid_int_dld)
  );

  shiftreg #(
    .SHIFT (P_BM_OUTPUT_STAGE_NUM),
    .DATA_W(1)
  ) i_shiftreg_axis_last (
    .clk,
    .data_i(axis_last),
    .data_o(axis_last_dld)
  );

  shiftreg #(
    .SHIFT (P_BM_OUTPUT_STAGE_NUM),
    .DATA_W(2)
  ) i_shiftreg_proj_coord_cntr (
    .clk,
    .data_i(proj_coord_cntr),
    .data_o(proj_coord_cntr_dld)
  );

  // Reg to remember full bucket value for 4 cycles
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      axis_valid_int_reg  <= 1'b0;
      proj_coord_cntr_reg <= 0;
    end else begin
      axis_valid_int_reg <= axis_valid_int_dld;
      axis_last_reg <= axis_last_dld;
      proj_coord_cntr_reg <= proj_coord_cntr_dld;
      if ((proj_coord_cntr_dld == 0) & (proj_coord_cntr_reg == P_NUM_PROJ_COORD - 1)) begin
        // Store on transtion of proj_coord_cntr Max -> 0 since val from bm in valid for only one cycle (also cleared)
        bm_bucket_val_reg <= bm_bucket_val_i;
      end
    end
  end

  // Select coordinate based on serialization counter
  always_comb begin
    case (proj_coord_cntr_reg)
      0: fifo_in_data = bm_bucket_val_reg[0*P_DATA_PNT_W+:P_DATA_PNT_W];
      1: fifo_in_data = bm_bucket_val_reg[1*P_DATA_PNT_W+:P_DATA_PNT_W];
      2: fifo_in_data = bm_bucket_val_reg[2*P_DATA_PNT_W+:P_DATA_PNT_W];
      default: fifo_in_data = bm_bucket_val_reg[3*P_DATA_PNT_W+:P_DATA_PNT_W];
    endcase
  end

  assign axis_ready_int = bw_ready_i | ~bw_valid_o;

  // Backpressure fifo
  fifo_axis #(
    .FIFO_DEPTH    (P_BM_OUTPUT_STAGE_NUM * 3),
    .FIFO_W        (P_DATA_PNT_W),
    .MEM_MACRO_TYPE("block"),
    .HAS_LAST      ("true")
  ) i_fifo_axis_bckpr (
    .clk,
    .rst_n,

    .axis_data_i (fifo_in_data),
    .axis_valid_i(axis_valid_int_reg),
    .axis_last_i (axis_last_reg),
    .axis_ready_o(),

    .axis_data_o (bw_bucket_val_o),
    .axis_valid_o(bw_valid_o),
    .axis_last_o (bw_last_o),
    .axis_ready_i(bw_ready_i)
  );

endmodule

`endif  // ACC_CTRL_SV
