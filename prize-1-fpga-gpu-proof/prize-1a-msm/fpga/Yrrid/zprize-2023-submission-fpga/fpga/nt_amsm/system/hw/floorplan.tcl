create_pblock k_main
create_pblock k_inverter
create_pblock k_proc_0
create_pblock k_proc_1
create_pblock k_proc_2
create_pblock k_ec_adder_0
create_pblock k_ec_adder_1
create_pblock k_ec_adder_2

set_property parent pblock_dynamic_SLR1 [get_pblocks k_main]
set_property parent pblock_dynamic_SLR1 [get_pblocks k_inverter]
set_property parent pblock_dynamic_SLR0 [get_pblocks k_proc_0]
set_property parent pblock_dynamic_SLR2 [get_pblocks k_proc_1]
set_property parent pblock_dynamic_SLR3 [get_pblocks k_proc_2]
set_property parent pblock_dynamic_SLR0 [get_pblocks k_ec_adder_0]
set_property parent pblock_dynamic_SLR2 [get_pblocks k_ec_adder_1]
set_property parent pblock_dynamic_SLR3 [get_pblocks k_ec_adder_2]

resize_pblock k_main -add CLOCKREGION_X3Y4:CLOCKREGION_X3Y7
resize_pblock k_main -remove {SLICE_X115Y240:SLICE_X116Y479}
resize_pblock k_inverter -add CLOCKREGION_X0Y4:CLOCKREGION_X2Y7

resize_pblock k_proc_0 -add {CLOCKREGION_X5Y0:CLOCKREGION_X7Y3}
resize_pblock k_proc_0 -remove {SLICE_X146Y230:SLICE_X205Y239 LAGUNA_X20Y220:LAGUNA_X27Y239}
resize_pblock k_ec_adder_0 -add {CLOCKREGION_X0Y0:CLOCKREGION_X4Y3}
resize_pblock k_ec_adder_0 -remove {SLICE_X146Y230:SLICE_X205Y239 LAGUNA_X20Y220:LAGUNA_X27Y239}

resize_pblock k_proc_1 -add {CLOCKREGION_X5Y8:CLOCKREGION_X7Y11}
resize_pblock k_proc_1 -remove {SLICE_X120Y480:SLICE_X202Y491 LAGUNA_X16Y480:LAGUNA_X27Y503}
resize_pblock k_ec_adder_1 -add {CLOCKREGION_X0Y8:CLOCKREGION_X4Y11}
resize_pblock k_ec_adder_1 -remove {SLICE_X120Y480:SLICE_X202Y491 LAGUNA_X16Y480:LAGUNA_X27Y503}

resize_pblock k_proc_2 -add {CLOCKREGION_X5Y12:CLOCKREGION_X7Y15}
resize_pblock k_ec_adder_2 -add {CLOCKREGION_X0Y12:CLOCKREGION_X4Y15}

add_cells_to_pblock [get_pblocks k_main] [get_cells level0_i/level1/level1_i/ulp/k_main]
add_cells_to_pblock [get_pblocks k_inverter] [get_cells level0_i/level1/level1_i/ulp/k_inverter]
add_cells_to_pblock [get_pblocks k_proc_0] [get_cells level0_i/level1/level1_i/ulp/k_proc_0/inst/k_proc_inst_U0]
add_cells_to_pblock [get_pblocks k_proc_1] [get_cells level0_i/level1/level1_i/ulp/k_proc_1/inst/k_proc_inst_U0]
add_cells_to_pblock [get_pblocks k_proc_2] [get_cells level0_i/level1/level1_i/ulp/k_proc_2/inst/k_proc_inst_U0]
add_cells_to_pblock [get_pblocks k_proc_0] [get_cells level0_i/level1/level1_i/ulp/k_proc_0/inst/k_d_mult_inst_U0]
add_cells_to_pblock [get_pblocks k_proc_1] [get_cells level0_i/level1/level1_i/ulp/k_proc_1/inst/k_d_mult_inst_U0]
add_cells_to_pblock [get_pblocks k_proc_2] [get_cells level0_i/level1/level1_i/ulp/k_proc_2/inst/k_d_mult_inst_U0]
add_cells_to_pblock [get_pblocks k_proc_0] [get_cells level0_i/level1/level1_i/ulp/k_proc_0/inst/k_ec_adder_inst_U0/stage_0_mult_U0]
add_cells_to_pblock [get_pblocks k_proc_1] [get_cells level0_i/level1/level1_i/ulp/k_proc_1/inst/k_ec_adder_inst_U0/stage_0_mult_U0]
add_cells_to_pblock [get_pblocks k_proc_2] [get_cells level0_i/level1/level1_i/ulp/k_proc_2/inst/k_ec_adder_inst_U0/stage_0_mult_U0]
add_cells_to_pblock [get_pblocks k_ec_adder_0] [get_cells level0_i/level1/level1_i/ulp/k_proc_0/inst/k_ec_adder_inst_U0/stage_1_U0]
add_cells_to_pblock [get_pblocks k_ec_adder_1] [get_cells level0_i/level1/level1_i/ulp/k_proc_1/inst/k_ec_adder_inst_U0/stage_1_U0]
add_cells_to_pblock [get_pblocks k_ec_adder_2] [get_cells level0_i/level1/level1_i/ulp/k_proc_2/inst/k_ec_adder_inst_U0/stage_1_U0]
