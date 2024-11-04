.PHONY: all hw sw clean test

all: hw sw

hw: addmixed.xclbin

addmixed.xo: addmixed.cpp
	v++ --kernel addMixedKernel --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -t hw -s -g -c -o $@ $<

addmixed.xclbin: addmixed.xo
	v++ --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -t hw -s -g --config connectivity.ini -l --jobs $(shell nproc) -o $@ $<


sw: build/host

build/host: host.cpp
	mkdir -p build
	cd build && cmake .. && cmake --build .

clean:
	rm -rf build *.xclbin* *.xo* *.log *.ltx _x

test: all
	./build/host ./mmult.xclbin
