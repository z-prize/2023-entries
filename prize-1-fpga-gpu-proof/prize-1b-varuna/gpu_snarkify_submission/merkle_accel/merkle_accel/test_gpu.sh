#!/bin/bash

rm -rf build
cmake -S . -B build -DUSE_MERKLE_ACCEL_GPU=on && cmake --build build --target test_gpu && cd build && ./test_gpu
