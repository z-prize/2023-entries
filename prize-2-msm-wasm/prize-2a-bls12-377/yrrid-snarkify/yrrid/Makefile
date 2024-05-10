CC := clang-17
CFLAGS := --target=wasm32 -matomics -msimd128 -mrelaxed-simd -O3 -nostdlib -DWASM
LD_FLAGS := --no-entry --export=fieldMultiples --export=pointsAddress --export=scalarsAddress \
	    --export=resultAddress --export=initializeCounters --export=computeMSM  \
	    --export=__stack_pointer --export=__heap_base \
	    --lto-O3 --import-memory --shared-memory --initial-memory=170393600 --max-memory=170393600 \
            --features=bulk-memory,atomics,mutable-globals,sign-ext,simd128,relaxed-simd
TARGET := msm.wasm
SRC := MSM.c

all: clean msm.wasm

clean:
	rm -f msm.o msm.wasm x86simd

msm.o: MSM.c
	clang-17 $(CFLAGS) -c MSM.c -o msm.o

msm.wasm: msm.o
	wasm-ld $(LD_FLAGS) msm.o -o msm.wasm	

x86simd:
	gcc -mfma -fopenmp -O3 -DX86 MSM.c -o x86simd
