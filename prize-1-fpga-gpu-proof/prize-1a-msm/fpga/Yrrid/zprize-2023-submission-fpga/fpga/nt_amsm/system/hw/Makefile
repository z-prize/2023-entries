PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1
CURVE?=bls12_381

csynth_k_inverter:
	KERNEL=k_$(CURVE)_inverter make -C ../../kernels/k_inverter csynth

csynth_k_main:
	make -C ../../kernels/k_main csynth

csynth_k_unified_proc_0:
	KERNEL=k_$(CURVE)_unified_proc_0 make -C ../../kernels/k_unified_proc csynth

csynth_k_unified_proc_1:
	KERNEL=k_$(CURVE)_unified_proc_1 make -C ../../kernels/k_unified_proc csynth

csynth_k_unified_proc_2:
	KERNEL=k_$(CURVE)_unified_proc_2 make -C ../../kernels/k_unified_proc csynth

kernels: csynth_k_inverter csynth_k_main csynth_k_unified_proc_0 csynth_k_unified_proc_1 csynth_k_unified_proc_2

link_hw:
	mkdir -p work/$(CURVE)
	cd work/$(CURVE); \
	v++ -l -t hw --platform $(PLATFORM) \
		--save-temps \
		-R2 \
		-O3 \
		--config ../../$(CURVE).cfg \
		../../../../kernels/k_main/work/k_main/k_main.xo \
		../../../../kernels/k_unified_proc/work/k_$(CURVE)_unified_proc_0/k_$(CURVE)_unified_proc_0.xo \
		../../../../kernels/k_unified_proc/work/k_$(CURVE)_unified_proc_1/k_$(CURVE)_unified_proc_1.xo \
		../../../../kernels/k_unified_proc/work/k_$(CURVE)_unified_proc_2/k_$(CURVE)_unified_proc_2.xo \
		../../../../kernels/k_inverter/work/k_$(CURVE)_inverter/k_$(CURVE)_inverter.xo \
		-o nt_amsm_$(CURVE).xclbin

link_hw_emu:
	mkdir -p work/$(CURVE)_hw_emu
	cd work/$(CURVE)_hw_emu; \
	v++ -l -t hw_emu --platform $(PLATFORM) \
		-g \
		--save-temps \
		--config ../../$(CURVE)_hw_emu.cfg \
		../../../../kernels/k_main/work/k_main/k_main.xo \
		../../../../kernels/k_unified_proc/work/k_$(CURVE)_unified_proc_0/k_$(CURVE)_unified_proc_0.xo \
		../../../../kernels/k_unified_proc/work/k_$(CURVE)_unified_proc_1/k_$(CURVE)_unified_proc_1.xo \
		../../../../kernels/k_unified_proc/work/k_$(CURVE)_unified_proc_2/k_$(CURVE)_unified_proc_2.xo \
		../../../../kernels/k_inverter/work/k_$(CURVE)_inverter/k_$(CURVE)_inverter.xo \
		-o nt_amsm_$(CURVE)_hw_emu.xclbin

clean:
	rm -rf ./work

clean_all: clean
	make -C ../../kernels/k_inverter clean
	make -C ../../kernels/k_main clean
	make -C ../../kernels/k_unified_proc clean
