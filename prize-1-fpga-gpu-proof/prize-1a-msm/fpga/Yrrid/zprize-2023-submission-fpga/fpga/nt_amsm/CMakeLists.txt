cmake_minimum_required(VERSION 3.10)

project(nt_amsm)

add_subdirectory(kernels/k_ll_mod_mult)

add_subdirectory(kernels/k_d_mult/)
add_subdirectory(kernels/k_ec_adder/)
add_subdirectory(kernels/k_inverter/)
add_subdirectory(kernels/k_main/)
add_subdirectory(kernels/k_proc/)
add_subdirectory(kernels/k_unified_proc/)
add_subdirectory(system/model/)
add_subdirectory(host/)
