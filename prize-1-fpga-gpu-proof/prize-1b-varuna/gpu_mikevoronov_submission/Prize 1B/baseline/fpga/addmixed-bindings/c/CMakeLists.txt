cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmakeModules")

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)
find_package(XRT REQUIRED)

if(NOT XRT_FOUND)
  MESSAGE(FATAL_ERROR "XRT not found, please source XRT setup script before building")
endif()

add_executable(host
  host.cpp)

target_include_directories(host PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}
${XRT_INCLUDE_DIRS}
${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(host PRIVATE
${XRT_LIBS}
${OpenCL_LIBRARIES}
pthread
uuid
${CMAKE_DL_LIBS}
)
