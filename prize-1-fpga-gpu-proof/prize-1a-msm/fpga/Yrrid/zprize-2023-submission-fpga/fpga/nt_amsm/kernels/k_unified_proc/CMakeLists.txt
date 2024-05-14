cmake_minimum_required(VERSION 3.10)

project(k_unified_proc)

include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(../../common/include/)
include_directories(./include/)
include_directories(../k_proc/include/)
include_directories(../k_d_mult/include/)
include_directories(../k_ec_adder/include/)

add_executable(
        k_unified_proc_test
        ./src/k_unified_proc.cpp
        ./test/test.cpp
        include/k_unified_proc.h
)

target_link_libraries(k_unified_proc_test gmp fmt)
