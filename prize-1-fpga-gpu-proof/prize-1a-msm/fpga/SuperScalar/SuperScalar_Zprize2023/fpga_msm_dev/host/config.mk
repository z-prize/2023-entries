VIVADO := $(XILINX_VIVADO)/bin/vivado
# shen
$(TEMP_DIR)/krnl_msm_rtl.xo: src/krnl_msm_rtl/package_kernel.tcl gen_xo.tcl src/krnl_msm_rtl/hdl/*.sv src/krnl_msm_rtl/hdl/*.v
	mkdir -p $(TEMP_DIR)
	$(VIVADO) -mode batch -source src/krnl_msm_rtl/gen_xo.tcl -tclargs $(TEMP_DIR)/krnl_msm_rtl.xo $(TARGET) $(PLATFORM) $(XSA) $(ZPRIZE_MUL_MOD)


#$(TEMP_DIR)/krnl_vadd_rtl.xo: src/krnl_vadd_rtl/package_kernel.tcl gen_xo.tcl src/krnl_vadd_rtl/hdl/*.sv src/krnl_vadd_rtl/hdl/*.v 
#	mkdir -p $(TEMP_DIR)
#	$(VIVADO) -mode batch -source src/krnl_vadd_rtl/gen_xo.tcl -tclargs $(TEMP_DIR)/krnl_vadd_rtl.xo $(TARGET) $(PLATFORM) $(XSA)



# VIVADO := $(XILINX_VIVADO)/bin/vivado
# $(TEMP_DIR)/krnl_mm2s.xo: src/krnl_mm2s/*.cpp
# 	mkdir -p $(TEMP_DIR)
# 	v++ $(VPP_FLAGS) --config src/krnl_mm2s/kernel.cfg  -c -k krnl_mm2s -I'$(<D)' -o'$@' '$<'

# $(TEMP_DIR)/krnl_msm_pippenger.xo: package_kernel.tcl src/krnl_msm_pippenger/kernel_ports.tcl gen_xo.tcl 
# 	mkdir -p $(TEMP_DIR)
# 	$(VIVADO) -mode batch -source gen_xo.tcl -tclargs $(TEMP_DIR)/krnl_msm_pippenger.xo $(TARGET) $(PLATFORM) $(XSA) krnl_msm_pippenger

# $(TEMP_DIR)/krnl_msm_rtl.xo: package_kernel.tcl src/krnl_msm/kernel_ports.tcl gen_xo.tcl 
# 	mkdir -p $(TEMP_DIR)
# 	$(VIVADO) -mode batch -source gen_xo.tcl -tclargs $(TEMP_DIR)/krnl_msm.xo $(TARGET) $(PLATFORM) $(XSA) krnl_msm

# $(TEMP_DIR)/krnl_s2mm.xo: src/krnl_s2mm/*.cpp
# 	mkdir -p $(TEMP_DIR)
# 	v++ $(VPP_FLAGS) --config src/krnl_s2mm/kernel.cfg  -c -k krnl_s2mm -I'$(<D)' -o'$@' '$<'
