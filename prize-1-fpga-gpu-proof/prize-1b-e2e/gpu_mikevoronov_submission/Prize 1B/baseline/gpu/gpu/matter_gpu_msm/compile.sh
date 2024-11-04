#!/bin/bash

nvcc -arch=native -Xcompiler -O3 -std=c++17 multiexp_bucket.cu -c -o msm_matter.o
