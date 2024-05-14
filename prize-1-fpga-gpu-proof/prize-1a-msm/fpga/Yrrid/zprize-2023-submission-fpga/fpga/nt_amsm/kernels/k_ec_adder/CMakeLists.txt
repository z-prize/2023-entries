cmake_minimum_required(VERSION 3.10)

project(k_ec_adder)

include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(../../common/include/)
include_directories(./include/)

add_executable(
        k_ec_adder_test
        src/k_ec_adder.cpp
        test/test.cpp
        test/test_vectors.h
)

target_link_libraries(k_ec_adder_test gmp fmt)
