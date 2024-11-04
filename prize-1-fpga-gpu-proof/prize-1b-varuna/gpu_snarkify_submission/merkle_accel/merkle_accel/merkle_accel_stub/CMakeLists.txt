cmake_minimum_required(VERSION 3.16)
project(merkle_accel_stub)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fopenmp")

include_directories(../common/)

add_library(
        merkle_accel_stub
        STATIC
        src/merkle_accel.cpp
)
