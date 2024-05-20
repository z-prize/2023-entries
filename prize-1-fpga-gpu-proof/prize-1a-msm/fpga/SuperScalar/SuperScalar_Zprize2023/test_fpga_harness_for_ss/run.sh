#!/bin/sh
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /tools/Xilinx/Vivado/2023.2/settings64.sh

#export XCL_EMULATION_MODE=sw_emu
#export XCL_EMULATION_MODE=hw

#CMAKE=/usr/bin/cmake XCLBIN=./cxx/host/superscalar_driver/bin/msm.xclbin TEST_NPOW=13 cargo test --release -- --nocapture
#CMAKE=/usr/bin/cmake XCLBIN=../fpga_msm_dev/host/src/krnl_msm_rtl/prebuild_xclbin/zprize_msm_100M_0318.xclbin TEST_NUM_ROUNDS=10 TEST_NPOW=24  cargo test --release -- --nocapture
CMAKE=/usr/bin/cmake XCLBIN=../fpga_msm_dev/host/src/krnl_msm_rtl/prebuild_xclbin/zprize_msm_200M_0318.xclbin TEST_NUM_ROUNDS=10 TEST_NPOW=24 cargo test --release -- --nocapture
#CMAKE=/usr/bin/cmake XCLBIN=../fpga_msm_dev/host/src/krnl_msm_rtl/prebuild_xclbin/zprize_msm_280M_0317_goodTiming.xclbin TEST_NUM_ROUNDS=10 TEST_NPOW=24 cargo test --release -- --nocapture
#CMAKE=/usr/bin/cmake XCLBIN=../fpga_msm_dev/host/src/krnl_msm_rtl/prebuild_xclbin/zprize_msm_200M_0318.xclbin TEST_NUM_ROUNDS=10 TEST_NPOW=24 TEST_LOAD_DATA_FROM=data/k24-01 cargo test --release -- --nocapture
#CMAKE=/usr/bin/cmake XCLBIN=../fpga_msm_dev/host/src/krnl_msm_rtl/prebuild_xclbin/zprize_msm_280M_0317_goodTiming.xclbin TEST_NUM_ROUNDS=10 TEST_NPOW=24 TEST_LOAD_DATA_FROM=data/k24-01 cargo test --release -- --nocapture




