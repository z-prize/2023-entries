set ipName xlnx_axis_conv_384to768

create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name $ipName

set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {48} CONFIG.M_TDATA_NUM_BYTES {96} CONFIG.HAS_TLAST {0}] [get_ips $ipName]

