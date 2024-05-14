cmake_minimum_required(VERSION 3.10)

project(k_main)

include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(../../common/include/)
include_directories(./include/)

add_executable(
        k_main
        ./src/k_main.cpp
)
