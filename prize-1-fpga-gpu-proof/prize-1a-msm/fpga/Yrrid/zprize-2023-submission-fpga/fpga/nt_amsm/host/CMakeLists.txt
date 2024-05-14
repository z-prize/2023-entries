cmake_minimum_required(VERSION 3.10)

project(nt_amsm)

include_directories(/opt/xilinx/xrt/include/)
include_directories(./include/)

link_directories(/opt/xilinx/xrt/lib)

add_executable(
        main
        ./src/main.cpp
)

target_link_libraries(main xrt_core xrt_coreutil uuid)
