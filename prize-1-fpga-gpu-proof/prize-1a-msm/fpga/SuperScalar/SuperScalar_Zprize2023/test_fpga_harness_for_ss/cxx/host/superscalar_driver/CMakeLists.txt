cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp")

get_filename_component(CUR_REAL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ REALPATH)
set(COMMON_DIR ./common)

project(driver_ss)

include_directories(/opt/xilinx/xrt/include)
include_directories(common/includes/xcl2)
include_directories(libs/algo_ss)
include_directories(libs/algo_ss/common)
include_directories(libs/algo_ss/common/math)
include_directories(libs/algo_ss/curve)
include_directories(libs/algo_ss/curve/bls12_377)
include_directories(libs/algo_ss/field)
include_directories(libs/algo_ss/msm)

set(HOST_SRCS 
    common/includes/xcl2/xcl2.cpp
    host.cpp
    libs/algo_ss/curve/curve.cpp
    libs/algo_ss/curve/twisted_edwards_extended/edwards.cpp
    libs/algo_ss/msm/msm.cpp
)

# add_executable(driver_ss ${HOST_SRCS} ${COMMON_DIR}/includes/xcl2/xcl2.cpp)
add_library(driver_ss SHARED ${HOST_SRCS} ${COMMON_DIR}/includes/xcl2/xcl2.cpp)

target_include_directories(driver_ss
	PUBLIC $ENV{XILINX_XRT}/include
	PUBLIC $ENV{XILINX_VIVADO}/include
	PUBLIC ${COMMON_DIR}/includes/xcl2)

target_link_directories(driver_ss PUBLIC /opt/xilinx/xrt/lib)
target_link_libraries(driver_ss PUBLIC OpenCL rt uuid xrt_coreutil xilinxopencl) # stdc++ 

message(STATUS "Finished driver_ss/")