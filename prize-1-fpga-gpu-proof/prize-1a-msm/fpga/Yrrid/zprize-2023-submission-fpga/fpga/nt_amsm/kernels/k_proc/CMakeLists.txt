cmake_minimum_required(VERSION 3.10)

project(k_proc)

include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(../../common/include/)
include_directories(../../common/hw/)
include_directories(./include/)

add_executable(
        k_proc_test
        ./src/k_proc.cpp
        ./test/test.cpp
)

target_link_libraries(k_proc_test gmp fmt)
