#!/bin/bash

nvcc -arch=native -Xcompiler -O3 -std=c++17 MSMBatch.cu -c -o msm_yrrid.o
