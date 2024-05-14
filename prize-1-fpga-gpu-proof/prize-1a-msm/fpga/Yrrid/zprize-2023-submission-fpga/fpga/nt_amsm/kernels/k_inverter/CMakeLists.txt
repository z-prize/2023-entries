cmake_minimum_required(VERSION 3.10)

project(k_inverter)

include_directories(/tools/Xilinx/Vitis_HLS/2023.2/include/)
include_directories(../../common/include/)
include_directories(./include/)

add_executable(
        k_inverter_test
        src/k_inverter.cpp
        test/test.cpp
)

target_link_libraries(k_inverter_test gmp fmt)
