

`ifndef FIFO_SYNC_SV
`define FIFO_SYNC_SV

module fifo_sync #(
  parameter FIFO_DEPTH = 10,
  parameter FIFO_W = 64,
  parameter MEM_MACRO_TYPE = "distributed",  // "registers", "distributed"-LUTRAM, "block"-BRAM, "ultra"-URAM
  parameter bit EXTRA_OUTPUT_STAGE = 0,
  parameter [$clog2(FIFO_DEPTH)-1:0] ALMOST_EMPTY_THR = 1,
  parameter [$clog2(FIFO_DEPTH)-1:0] ALMOST_FULL_THR = 1
) (
  input clk,
  input rst_n,

  // synthesis translate_off
  output logic [$clog2(FIFO_DEPTH):0] fill_level_sim,
  output logic [$clog2(FIFO_DEPTH):0] fill_level_max,
  // synthesis translate_on

  input                     wr_req_i,
  input        [FIFO_W-1:0] wr_data_i,
  output logic              full_o,
  output logic              almost_full_o,

  input                     rd_req_i,
  output logic [FIFO_W-1:0] rd_data_o,
  output logic              empty_o,
  output logic              almost_empty_o
);

  localparam PTR_W = $clog2(FIFO_DEPTH);
  localparam [PTR_W-1:0] PTR_MAX_VAL = FIFO_DEPTH - 1;
  localparam [PTR_W:0] PTR_DEPTH_DIFF = {1'b1, {PTR_W{1'b0}}} - FIFO_DEPTH[PTR_W:0];
  localparam [PTR_W:0] ALMOST_FULL_CLRVAL = FIFO_DEPTH - ALMOST_FULL_THR;
  localparam [PTR_W:0] ALMOST_FULL_SETVAL = FIFO_DEPTH - ALMOST_FULL_THR - 1;
  localparam [PTR_W:0] ALMOST_EMPTY_CLRVAL = ALMOST_EMPTY_THR;
  localparam [PTR_W:0] ALMOST_EMPTY_SETVAL = ALMOST_EMPTY_THR + 1;

  // Internal registers and wires
  logic [PTR_W:0] wr_ptr_nxt;
  logic [PTR_W:0] wr_ptr;
  logic [PTR_W:0] rd_ptr_nxt;
  logic [PTR_W:0] rd_ptr;
  logic           ptr_msb_diff;
  logic [PTR_W:0] words_num;
  logic           flags_update;
  logic           wr_full_val;
  logic           wr_full;
  logic           almost_full_set;
  logic           almost_full_clr;
  logic           almost_full;
  logic           rd_empty;
  logic           almost_empty_set;
  logic           almost_empty_clr;
  logic           almost_empty;

  // FIFO write pointer
  always_comb begin
    if (wr_req_i & !wr_full) begin
      if (wr_ptr[PTR_W-1:0] == PTR_MAX_VAL) begin
        wr_ptr_nxt = {!wr_ptr[PTR_W], {PTR_W{1'b0}}};
      end else begin
        wr_ptr_nxt = wr_ptr + {{PTR_W{1'b0}}, 1'b1};
      end
    end else begin
      wr_ptr_nxt = wr_ptr;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_ptr <= '0;
    end else begin
      wr_ptr <= wr_ptr_nxt;
    end
  end

  // FIFO read pointer
  always_comb begin
    if (rd_req_i & !rd_empty) begin
      if (rd_ptr[PTR_W-1:0] == PTR_MAX_VAL) begin
        rd_ptr_nxt = {!rd_ptr[PTR_W], {PTR_W{1'b0}}};
      end else begin
        rd_ptr_nxt = rd_ptr + {{PTR_W{1'b0}}, 1'b1};
      end
    end else begin
      rd_ptr_nxt = rd_ptr;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_ptr <= '0;
    end else begin
      rd_ptr <= rd_ptr_nxt;
    end
  end

  // FIFO full logic
  assign flags_update = wr_req_i | rd_req_i;
  assign wr_full_val  = wr_ptr_nxt == {!rd_ptr_nxt[PTR_W], rd_ptr_nxt[PTR_W-1:0]};

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_full <= 1'b0;
    end else if (flags_update) begin
      wr_full <= wr_full_val;
    end
  end


  assign ptr_msb_diff = wr_ptr[PTR_W] ^ rd_ptr[PTR_W];
  assign words_num    = {ptr_msb_diff, wr_ptr[PTR_W-1:0]} - {1'b0, rd_ptr[PTR_W-1:0]} -
                        (ptr_msb_diff? PTR_DEPTH_DIFF : {(PTR_W+1){1'b0}});

  assign almost_full_clr = (words_num == ALMOST_FULL_CLRVAL) & rd_req_i & !wr_req_i;
  assign almost_full_set = (words_num == ALMOST_FULL_SETVAL) & !rd_req_i & wr_req_i;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      almost_full <= 1'b0;
    end else if (almost_full_clr) begin
      almost_full <= 1'b0;
    end else if (almost_full_set) begin
      almost_full <= 1'b1;
    end
  end

  // FIFO empty logic
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rd_empty <= 1'b1;
    end else if (flags_update) begin
      rd_empty <= rd_ptr_nxt == wr_ptr_nxt;
    end
  end

  assign almost_empty_set = (words_num == ALMOST_EMPTY_SETVAL) & rd_req_i & !wr_req_i;
  assign almost_empty_clr = (words_num == ALMOST_EMPTY_CLRVAL) & !rd_req_i & wr_req_i;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      almost_empty <= 1'b1;
    end else if (almost_empty_set) begin
      almost_empty <= 1'b1;
    end else if (almost_empty_clr) begin
      almost_empty <= 1'b0;
    end
  end

  // Output FIFO flags
  assign empty_o        = rd_empty;
  assign almost_empty_o = almost_empty;

  assign full_o         = wr_full;
  assign almost_full_o  = almost_full;

  // FIFO
  (* ram_style = MEM_MACRO_TYPE *) logic [FIFO_W-1:0] fifo_reg[FIFO_DEPTH];

  always_ff @(posedge clk) begin
    if (wr_req_i & !wr_full) begin
      fifo_reg[wr_ptr[PTR_W-1:0]] <= wr_data_i;
    end
  end

  generate
    if ((MEM_MACRO_TYPE == "block") || (MEM_MACRO_TYPE == "ultra") || (EXTRA_OUTPUT_STAGE)) begin
      always_ff @(posedge clk)  // block and ultra ram require this extra stage
        rd_data_o <= fifo_reg[rd_ptr[PTR_W-1:0]];
    end else begin
      assign rd_data_o = fifo_reg[rd_ptr[PTR_W-1:0]];
    end  //if
  endgenerate

  // synthesis translate_off
  assign fill_level_sim = wr_ptr - rd_ptr;
  always @(posedge clk)
    if (rst_n == 0) fill_level_max = 0;
    else if (fill_level_max < fill_level_sim) fill_level_max = fill_level_sim;
  // synthesis translate_on

  // Assertions
  // synthesis translate_off
`ifdef ASSERTIONS_ON
  bit           assert_disable;
  bit [PTR_W:0] cnt;

  initial begin
    assert_disable = 1'b1;
    @(posedge clk);
    @(negedge rst_n);
    assert_disable = 1'b0;
  end

  always @(posedge clk)
    if (!rst_n) cnt = '0;
    else cnt = cnt - {'0, rd_req_i & !rd_empty} + {'0, wr_req_i & !wr_full};

  property p_bad_access(inc, flag);
    @(posedge clk) disable iff (!rst_n | assert_disable) inc |-> ~flag;
  endproperty : p_bad_access

  property p_almost_full(set_clr, threshold);
    @(posedge clk)
      disable iff (!rst_n | assert_disable)
                  (set_clr ? (cnt >= (FIFO_DEPTH - threshold)) : (cnt <  (FIFO_DEPTH - threshold))) 
                  |-> (set_clr ? almost_full : !almost_full);
  endproperty : p_almost_full

  property p_almost_empty(set_clr, threshold);
    @(posedge clk) 
      disable iff (!rst_n | assert_disable)
                  (set_clr ? (cnt <= threshold) : (cnt >  threshold))
                   |-> (set_clr ? almost_empty : !almost_empty);
  endproperty : p_almost_empty

  property p_empty;
    @(posedge clk) disable iff (!rst_n | assert_disable) (rd_empty ^ (cnt != '0));
  endproperty : p_empty

  property p_full;
    @(posedge clk) disable iff (!rst_n | assert_disable) (wr_full ^ (cnt != FIFO_DEPTH));
  endproperty

  property p_words_num;
    @(posedge clk) disable iff (!rst_n | assert_disable) (words_num === cnt);
  endproperty

  // Checks if attempt to write into full FIFO happened
  check_wr_to_full :
  assert property (p_bad_access(wr_req_i, full_o))
  else $error("Write access attempted to full FIFO");

  // Checks if attempt to read from empty FIFO happened
  check_rd_from_empty :
  assert property (p_bad_access(rd_req_i, empty_o))
  else $error("Read access attempted from empty FIFO");

  // Checks whether almost_full flag goes to 1 at the right moment
  check_almost_full_set :
  assert property (p_almost_full(1, ALMOST_FULL_THR))
  else $error("almost_full flag is incorrectly set to 1");

  // Checks whether almost_full flad goes to 0 at the right moment
  check_almost_full_clr :
  assert property (p_almost_full(0, ALMOST_FULL_THR))
  else $error("almost_full flag is incorrectly set to 0");

  // Checks whether almost_empty flag goes to 1 at the right moment
  check_almost_empty_set :
  assert property (p_almost_empty(1, ALMOST_EMPTY_THR))
  else $error("almost_empty flag is incorrectly set to 1");

  // Checks whether almost_empty flad goes to 0 at the right moment
  check_almost_empty_clr :
  assert property (p_almost_empty(0, ALMOST_EMPTY_THR))
  else $error("almost_empty flag is incorrectly set to 0");

  // Check if empty flag works correctly
  check_empty :
  assert property (p_empty)
  else $error("empty flag has inccorrect value");

  // Check if full flag operates correctly
  check_full :
  assert property (p_full)
  else $error("full flag has incorrect value");

  // Check if words_num works correctly
  check_words_num :
  assert property (p_words_num)
  else $error("Incorrect value in words_num counter");

`endif
  // synthesis translate_on

endmodule

`endif  // FIFO_SYNC_SV
