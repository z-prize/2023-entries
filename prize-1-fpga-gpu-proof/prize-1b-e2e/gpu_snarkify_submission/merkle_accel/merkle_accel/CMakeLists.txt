cmake_minimum_required(VERSION 3.16)
project(merkle_accel)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

option(USE_MERKLE_ACCEL_GPU "" OFF)
option(USE_MERKLE_ACCEL_FPGA "" OFF)

add_subdirectory(merkle_accel_cpu)
add_subdirectory(merkle_accel_stub)

include_directories(common)

add_executable(
        test_cpu
        test/test.cpp
)
add_executable(
        test_stub
        test/test.cpp
)

target_link_libraries(test_cpu merkle_accel_cpu)
target_link_libraries(test_stub merkle_accel_stub)

if (USE_MERKLE_ACCEL_GPU)
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(merkle_accel_gpu)
    add_executable(
            test_gpu
            test/test.cpp
    )
    target_link_libraries(test_gpu merkle_accel_gpu CUDA::cudart)
endif ()

if (USE_MERKLE_ACCEL_FPGA)
    add_subdirectory(merkle_accel_fpga)
    add_executable(
            test_fpga
            test/test.cpp
    )
    target_link_libraries(test_fpga merkle_accel_fpga)
endif ()
