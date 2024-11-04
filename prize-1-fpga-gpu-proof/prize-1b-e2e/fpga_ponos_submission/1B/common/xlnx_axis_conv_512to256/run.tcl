set ipName xlnx_axis_conv_512to256

create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name $ipName

set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {64} CONFIG.M_TDATA_NUM_BYTES {32} CONFIG.HAS_TLAST {1}] [get_ips $ipName]

