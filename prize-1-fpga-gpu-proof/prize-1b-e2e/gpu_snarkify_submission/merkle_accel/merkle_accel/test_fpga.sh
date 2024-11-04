#!/bin/bash

rm -rf build
cmake -S . -B build -DUSE_MERKLE_ACCEL_FPGA=on && cmake --build build --target test_fpga && cd build && ./test_fpga
