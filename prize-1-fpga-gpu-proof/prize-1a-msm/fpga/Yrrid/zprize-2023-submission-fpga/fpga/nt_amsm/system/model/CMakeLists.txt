cmake_minimum_required(VERSION 3.10)

project(model)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
add_definitions(-D__BLS12_377__)

include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(../../common/include/)
include_directories(../../kernels/k_d_mult/include/)
include_directories(../../kernels/k_ec_adder/include/)
include_directories(../../kernels/k_inverter/include/)
include_directories(../../kernels/k_main/include/)
include_directories(../../kernels/k_proc/include/)
include_directories(../../kernels/k_unified_proc/include/)
include_directories(../../host/include/)

add_executable(
        model
        ./test/test.cpp
        ./test/test_vectors.h
        ../../kernels/k_unified_proc/src/k_unified_proc.cpp
        ../../kernels/k_main/src/k_main.cpp
)
