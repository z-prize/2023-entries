cmake_minimum_required(VERSION 3.16)
project(merkle_accel_fpga)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

include_directories(../common/)

add_library(
        merkle_accel_fpga
        STATIC
        src/merkle_accel.cpp
)
