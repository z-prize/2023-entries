cmake_minimum_required(VERSION 3.16)

project(merkle_accel_gpu LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fopenmp")

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

include_directories(../common/)

add_library(
        merkle_accel_gpu
        STATIC
        src/merkle_accel.cu
)
