#!/bin/bash

rm -rf build
cmake -S . -B build && cmake --build build --target test_stub && cd build && ./test_stub