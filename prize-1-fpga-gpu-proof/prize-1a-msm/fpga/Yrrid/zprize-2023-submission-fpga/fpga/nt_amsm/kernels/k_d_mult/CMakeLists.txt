cmake_minimum_required(VERSION 3.10)

project(k_d_mult)

include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(../../common/include/)
include_directories(./include/)

add_executable(
        k_d_mult_test
        ./test/test_vectors.h
        ./test/test.cpp
        src/k_d_mult.cpp
)

target_link_libraries(k_d_mult_test gmp)
