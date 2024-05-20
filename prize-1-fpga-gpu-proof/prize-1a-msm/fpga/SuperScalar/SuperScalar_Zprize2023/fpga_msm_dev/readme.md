## fpga_msm_dev

- We have implemented a BLS12-377 Multi-Scalar Multiplication (MSM) design running on the U250 FPGA accelerator card.
- By default, there is no need to modify or run any files in this directory, as FPGA synthesis typically requires a substantial amount of time. We have precompiled several xclbin files for convenience and placed them in `host/src/krnl_msm_rtl/prebuild_xclbin`.
- If you wish to synthesize an xclbin file from source code, you can directly navigate to the `host` folder and execute `run_hw.sh`. Provided that the Vitis environment is correctly configured, this will generate a Vivado project. To locate the resulting xclbin file, use the command `find -name "*.xclbin"` within the `host` directory. Please note that synthesizing from source using our development environment (as detailed in the readme.md file in this repository) may take approximately 10 hours. The end result will be three compute cores running at a frequency of 280MHz.
- Cleaning temporary compilation files: Run `make cleanall` in the `host` directory.

Directory Structure:

- `host`: Contains most of our project files
  - Script for compiling from source code: `host/run_hw.sh`
  - RTL (Verilog HDL): Files in `host/src/krnl_msm_rtl/hdl` and within the `host/src/krnl_msm_rtl/hdl/msm` subdirectory
  - Pre-compiled FPGA binary files (xclbin): Stored in `host/src/krnl_msm_rtl/prebuild_xclbin/`
- `doc`: This folder houses our hardware design documentation.
