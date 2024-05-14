#!/bin/bash

source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export XCL_EMULATION_MODE=hw_emu
export EMCONFIG_PATH=$(pwd)
export XRT_INI_PATH=$(pwd)/xrt.ini
