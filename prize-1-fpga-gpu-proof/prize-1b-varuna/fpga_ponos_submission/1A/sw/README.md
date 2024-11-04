# Bench

To run build and run the bench, run `cargo bench`.

# Our additional test runner

Our own helper tool `ponos-msm-runner` can be built using `cargo build --workspace`.

For a quick test, run `ponos-msm-runner -c 377` and `ponos-msm-runner -c 381`.
The tool provides help describing optional arguments.

# Location of XCLBIN files

In `bench.rs`, our engine is initialized with the following line:

`let engine = ponos_msm_fpga::create_engine(1 << LOG_SIZE, "")`

The directory in which xclbin files are contained can be specified, such as:

`let engine = ponos_msm_fpga::create_engine(1 << LOG_SIZE, "xclbins/")`
