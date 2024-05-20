// ZPRIZE
`define FPGA

//Async reset Flop
`define FlopAll(Q, we, D, clk, rstN, rstValue)          \
    always@(posedge clk or negedge rstN)                \
        if(~rstN)                                       \
            Q   <= `CQ rstValue;                        \
        else if(we)                                     \
            Q   <= `CQ D;                   


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//Async reset Flop
//////////////////////////////////////////////////////////////////////////////////////////////////////////

`define Flop(Q, we, D)                                  \
    `FlopAll(Q, we, D, clk, rstN, 0)             

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//No reset Flop
//////////////////////////////////////////////////////////////////////////////////////////////////////////
`define FlopNAll(Q, we, D, clk)          \
    always@(posedge clk)                \
        if(we)                                     \
            Q   <= `CQ D;

`define FlopN(Q, we, D)                                  \
    `FlopNAll(Q, we, D, clk)


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//Flop with set/clear
//set priority first
//default is 0 or 1
//////////////////////////////////////////////////////////////////////////////////////////////////////////
`define FlopSC0(Q, set, clear) `FlopAll(Q, (set) || (clear), set, clk, rstN, 0)
`define FlopSC1(Q, set, clear) `FlopAll(Q, (set) || (clear), set, clk, rstN, 1)

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//Other
//////////////////////////////////////////////////////////////////////////////////////////////////////////


`define MAX_FANOUT32 (* max_fanout = "32" *)
`define MAX_FANOUT16 (* max_fanout = "16" *)
`define REG_LOGIC    
`define KEEP_HIE    (* keep_hierarchy="yes" *)    
`define KEEP        (* keep="true" *)



`ifdef ENABLE_ASSERT
`define assert_clk(arg)                                 \
    assert property (@(posedge clk) disable iff (rstN!==1'b1) arg) \
    else #5 $fatal;
`else
`define assert_clk(arg)
`endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//zprize mul_montgomery --> zprize_mul_mod
//////////////////////////////////////////////////////////////////////////////////////////////////////////
`ifdef ZPRIZE_FAKE_MUL_MOD
`define MULMOD(pipe,result,in_0,in_1) \
    logic       tmpmul_``result;    \
fake_zprize_mul_mod #(\
        .T0(32'h07F7_F999), \
        .T1(32'h0000_005F), \
        .T2(32'h07F7_F999)) \
mul_mod_``result(\
    .m_i  (valid_``pipe), \
    .in0({1'b0,in_0}), \
    .in1({1'b0,in_1}), \
    .out0({tmpmul_``result,result}),   \
    .m_o(),   \
    .clk,   \
    .rst(!rstN)  \
);
`else
`define MULMOD(pipe,result,in_0,in_1) \
    logic       tmpmul_``result;    \
zprize_mul_mod #(\
        .T0(32'h07F7_F999), \
        .T1(32'h0000_005F), \
        .T2(32'h07F7_F999)) \
mul_mod_``result(\
    .m_i  (valid_``pipe), \
    .in0({1'b0,in_0}), \
    .in1({1'b0,in_1}), \
    .out0({tmpmul_``result,result}),   \
    .m_o(),   \
    .clk,   \
    .rst(!rstN)  \
);
`endif

`define ADDMOD(pipe,result,in_0,in_1)   \
    logic       tmpadd_``result;    \
add_mod add_mod_``result(\
    .valid  (valid_``pipe), \
    .in0({1'b0,in_0}), \
    .in1({1'b0,in_1}), \
    .out({tmpadd_``result,result}),   \
    .idle(),    \
    .valid_out(),   \
    .sel_out(),\
    .clk,   \
    .rstN,  \
    .sel(1'b1)    \
);

`define SUBMOD(pipe,result,in_0,in_1)   \
    logic       tmpsub_``result;    \
sub_mod sub_mod_``result(\
    .valid  (valid_``pipe), \
    .in0({1'b0,in_0}), \
    .in1({1'b0,in_1}), \
    .out({tmpsub_``result,result}),   \
    .idle(),    \
    .valid_out(),   \
    .sel_out(),\
    .clk,   \
    .rstN,  \
    .sel(1'b1)    \
);
`define DUOMOD(pipe,result,in_0)    \
    logic       tmpduo_``result;    \
duos_mod duos_mod_``result(\
    .valid  (valid_``pipe), \
    .in0({1'b0,in_0}), \
    .out({tmpdup_``result,result}),   \
    .valid_out(),   \
    .idle(),        \
    .clk,   \
    .rstN,  \
    .sel(1'b1)    \
);

`define PIPE(latency,width,out,in) hyperPipe #(\
		.CYCLES  (latency),\
		.WIDTH   (width)) \
        hp``out(\
		.din     (in),\
		.dout     (out),    \
		.clk     (clk)  \
);

