KERNEL=k_main

csim:
	mkdir -p work/$(KERNEL)
	cd work/$(KERNEL); vitis-run --csim --config ../../$(KERNEL).cfg --work_dir .

csynth:
	mkdir -p work/$(KERNEL)
	cd work/$(KERNEL); v++ -c --mode hls --config ../../$(KERNEL).cfg --work_dir .

cosim:
	cd work/$(KERNEL); vitis-run --cosim --config ../../$(KERNEL).cfg --work_dir .

impl:
	cd work/$(KERNEL); vitis-run --impl --config ../../$(KERNEL).cfg --work_dir .

clean:
	rm -rf ./work
