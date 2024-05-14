# -- [main to proc pipe] ----------------------------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_main_to_proc_0_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_main_to_proc_0_pipe]
delete_bd_objs [get_bd_intf_nets k_main_to_proc_0]
connect_bd_intf_net [get_bd_intf_pins k_main_to_proc_0_pipe/S_AXIS] [get_bd_intf_pins k_main/to_proc_0]
connect_bd_intf_net [get_bd_intf_pins k_main_to_proc_0_pipe/M_AXIS] [get_bd_intf_pins k_proc_0/from_main]
connect_bd_net [get_bd_pins k_main_to_proc_0_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_main_to_proc_0_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_main_to_proc_1_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_main_to_proc_1_pipe]
delete_bd_objs [get_bd_intf_nets k_main_to_proc_1]
connect_bd_intf_net [get_bd_intf_pins k_main_to_proc_1_pipe/S_AXIS] [get_bd_intf_pins k_main/to_proc_1]
connect_bd_intf_net [get_bd_intf_pins k_main_to_proc_1_pipe/M_AXIS] [get_bd_intf_pins k_proc_1/from_main]
connect_bd_net [get_bd_pins k_main_to_proc_1_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_main_to_proc_1_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_main_to_proc_2_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {2} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_MIDDLE {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_main_to_proc_2_pipe]
delete_bd_objs [get_bd_intf_nets k_main_to_proc_2]
connect_bd_intf_net [get_bd_intf_pins k_main_to_proc_2_pipe/S_AXIS] [get_bd_intf_pins k_main/to_proc_2]
connect_bd_intf_net [get_bd_intf_pins k_main_to_proc_2_pipe/M_AXIS] [get_bd_intf_pins k_proc_2/from_main]
connect_bd_net [get_bd_pins k_main_to_proc_2_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_main_to_proc_2_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]
# ---------------------------------------------------------------------------------------------------------------------

# -- [proc to main status pipe] ---------------------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_0_to_main_status_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_0_to_main_status_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_0_to_main_status]
connect_bd_intf_net [get_bd_intf_pins k_proc_0_to_main_status_pipe/S_AXIS] [get_bd_intf_pins k_proc_0/to_main_status]
connect_bd_intf_net [get_bd_intf_pins k_proc_0_to_main_status_pipe/M_AXIS] [get_bd_intf_pins k_main/from_proc_0_status]
connect_bd_net [get_bd_pins k_proc_0_to_main_status_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_proc_0_to_main_status_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_1_to_main_status_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_1_to_main_status_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_1_to_main_status]
connect_bd_intf_net [get_bd_intf_pins k_proc_1_to_main_status_pipe/S_AXIS] [get_bd_intf_pins k_proc_1/to_main_status]
connect_bd_intf_net [get_bd_intf_pins k_proc_1_to_main_status_pipe/M_AXIS] [get_bd_intf_pins k_main/from_proc_1_status]
connect_bd_net [get_bd_pins k_proc_1_to_main_status_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_proc_1_to_main_status_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_2_to_main_status_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {2} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_MIDDLE {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_2_to_main_status_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_2_to_main_status]
connect_bd_intf_net [get_bd_intf_pins k_proc_2_to_main_status_pipe/S_AXIS] [get_bd_intf_pins k_proc_2/to_main_status]
connect_bd_intf_net [get_bd_intf_pins k_proc_2_to_main_status_pipe/M_AXIS] [get_bd_intf_pins k_main/from_proc_2_status]
connect_bd_net [get_bd_pins k_proc_2_to_main_status_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_proc_2_to_main_status_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]
# ---------------------------------------------------------------------------------------------------------------------

# -- [proc to main data pipe] -----------------------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_0_to_main_data_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_0_to_main_data_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_0_to_main_data]
connect_bd_intf_net [get_bd_intf_pins k_proc_0_to_main_data_pipe/S_AXIS] [get_bd_intf_pins k_proc_0/to_main_data]
connect_bd_intf_net [get_bd_intf_pins k_proc_0_to_main_data_pipe/M_AXIS] [get_bd_intf_pins k_main/from_proc_0_data]
connect_bd_net [get_bd_pins k_proc_0_to_main_data_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_proc_0_to_main_data_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_1_to_main_data_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_1_to_main_data_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_1_to_main_data]
connect_bd_intf_net [get_bd_intf_pins k_proc_1_to_main_data_pipe/S_AXIS] [get_bd_intf_pins k_proc_1/to_main_data]
connect_bd_intf_net [get_bd_intf_pins k_proc_1_to_main_data_pipe/M_AXIS] [get_bd_intf_pins k_main/from_proc_1_data]
connect_bd_net [get_bd_pins k_proc_1_to_main_data_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_proc_1_to_main_data_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_2_to_main_data_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {2} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_MIDDLE {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_2_to_main_data_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_2_to_main_data]
connect_bd_intf_net [get_bd_intf_pins k_proc_2_to_main_data_pipe/S_AXIS] [get_bd_intf_pins k_proc_2/to_main_data]
connect_bd_intf_net [get_bd_intf_pins k_proc_2_to_main_data_pipe/M_AXIS] [get_bd_intf_pins k_main/from_proc_2_data]
connect_bd_net [get_bd_pins k_proc_2_to_main_data_pipe/aclk] [get_bd_pins k_main/ap_clk]
connect_bd_net [get_bd_pins k_proc_2_to_main_data_pipe/aresetn] [get_bd_pins k_main/ap_rst_n]
# ---------------------------------------------------------------------------------------------------------------------

# -- [k_proc to k_inverter pipe] ---------------------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_0_to_k_inverter_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_0_to_k_inverter_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_0_to_inverter]
connect_bd_intf_net [get_bd_intf_pins k_proc_0_to_k_inverter_pipe/S_AXIS] [get_bd_intf_pins k_proc_0/to_inverter]
connect_bd_intf_net [get_bd_intf_pins k_proc_0_to_k_inverter_pipe/M_AXIS] [get_bd_intf_pins k_inverter/from_proc_0]
connect_bd_net [get_bd_pins k_proc_0_to_k_inverter_pipe/aclk] [get_bd_pins k_inverter/ap_clk]
connect_bd_net [get_bd_pins k_proc_0_to_k_inverter_pipe/aresetn] [get_bd_pins k_inverter/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_1_to_k_inverter_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_1_to_k_inverter_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_1_to_inverter]
connect_bd_intf_net [get_bd_intf_pins k_proc_1_to_k_inverter_pipe/S_AXIS] [get_bd_intf_pins k_proc_1/to_inverter]
connect_bd_intf_net [get_bd_intf_pins k_proc_1_to_k_inverter_pipe/M_AXIS] [get_bd_intf_pins k_inverter/from_proc_1]
connect_bd_net [get_bd_pins k_proc_1_to_k_inverter_pipe/aclk] [get_bd_pins k_inverter/ap_clk]
connect_bd_net [get_bd_pins k_proc_1_to_k_inverter_pipe/aresetn] [get_bd_pins k_inverter/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_proc_2_to_k_inverter_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {2} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_MIDDLE {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_proc_2_to_k_inverter_pipe]
delete_bd_objs [get_bd_intf_nets k_proc_2_to_inverter]
connect_bd_intf_net [get_bd_intf_pins k_proc_2_to_k_inverter_pipe/S_AXIS] [get_bd_intf_pins k_proc_2/to_inverter]
connect_bd_intf_net [get_bd_intf_pins k_proc_2_to_k_inverter_pipe/M_AXIS] [get_bd_intf_pins k_inverter/from_proc_2]
connect_bd_net [get_bd_pins k_proc_2_to_k_inverter_pipe/aclk] [get_bd_pins k_inverter/ap_clk]
connect_bd_net [get_bd_pins k_proc_2_to_k_inverter_pipe/aresetn] [get_bd_pins k_inverter/ap_rst_n]
# ---------------------------------------------------------------------------------------------------------------------

# -- [inverter to proc pipe] --------------------------------------------------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_inverter_to_k_proc_0_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_inverter_to_k_proc_0_pipe]
delete_bd_objs [get_bd_intf_nets k_inverter_to_proc_0]
connect_bd_intf_net [get_bd_intf_pins k_inverter_to_k_proc_0_pipe/S_AXIS] [get_bd_intf_pins k_inverter/to_proc_0]
connect_bd_intf_net [get_bd_intf_pins k_inverter_to_k_proc_0_pipe/M_AXIS] [get_bd_intf_pins k_proc_0/from_inverter]
connect_bd_net [get_bd_pins k_inverter_to_k_proc_0_pipe/aclk] [get_bd_pins k_inverter/ap_clk]
connect_bd_net [get_bd_pins k_inverter_to_k_proc_0_pipe/aresetn] [get_bd_pins k_inverter/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_inverter_to_k_proc_1_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {1} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_inverter_to_k_proc_1_pipe]
delete_bd_objs [get_bd_intf_nets k_inverter_to_proc_1]
connect_bd_intf_net [get_bd_intf_pins k_inverter_to_k_proc_1_pipe/S_AXIS] [get_bd_intf_pins k_inverter/to_proc_1]
connect_bd_intf_net [get_bd_intf_pins k_inverter_to_k_proc_1_pipe/M_AXIS] [get_bd_intf_pins k_proc_1/from_inverter]
connect_bd_net [get_bd_pins k_inverter_to_k_proc_1_pipe/aclk] [get_bd_pins k_inverter/ap_clk]
connect_bd_net [get_bd_pins k_inverter_to_k_proc_1_pipe/aresetn] [get_bd_pins k_inverter/ap_rst_n]

create_bd_cell -type ip -vlnv xilinx.com:ip:axis_register_slice:1.1 k_inverter_to_k_proc_2_pipe
set_property -dict [list \
  CONFIG.NUM_SLR_CROSSINGS {2} \
  CONFIG.PIPELINES_MASTER {2} \
  CONFIG.PIPELINES_MIDDLE {2} \
  CONFIG.PIPELINES_SLAVE {2} \
  CONFIG.REG_CONFIG {15} \
] [get_bd_cells k_inverter_to_k_proc_2_pipe]
delete_bd_objs [get_bd_intf_nets k_inverter_to_proc_2]
connect_bd_intf_net [get_bd_intf_pins k_inverter_to_k_proc_2_pipe/S_AXIS] [get_bd_intf_pins k_inverter/to_proc_2]
connect_bd_intf_net [get_bd_intf_pins k_inverter_to_k_proc_2_pipe/M_AXIS] [get_bd_intf_pins k_proc_2/from_inverter]
connect_bd_net [get_bd_pins k_inverter_to_k_proc_2_pipe/aclk] [get_bd_pins k_inverter/ap_clk]
connect_bd_net [get_bd_pins k_inverter_to_k_proc_2_pipe/aresetn] [get_bd_pins k_inverter/ap_rst_n]
# ---------------------------------------------------------------------------------------------------------------------
