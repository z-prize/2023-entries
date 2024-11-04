#!/bin/bash
nvcc -DCUDA -arch=native -Xcompiler -O3 -std=c++17 -I../../../matter_gpu_msm/ poseidon_hash.cu  -c -o poseidon_hash.o &
nvcc -DCUDA -arch=native -Xcompiler -O3 -std=c++17 -I../../../matter_gpu_msm/ poly_functions.cu -c -o poly_functions.o &
wait
