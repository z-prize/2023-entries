#!/bin/bash

rm -rf build
cmake -S . -B build && cmake --build build --target test_cpu && cd build && ./test_cpu
