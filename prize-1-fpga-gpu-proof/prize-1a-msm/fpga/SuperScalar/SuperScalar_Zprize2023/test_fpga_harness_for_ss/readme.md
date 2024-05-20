# MSM FPGA Test Harness By SuperScalar

This is modified based on the Zprize 2023 test harness to test our FPGA MSM design.
It supports multiple rounds, with 4 batches of MSM computation in each round.

## Feature
- support 377-bit curves.
- utilize Twisted Edwards Extended curve for acceleration and performing precomputation.
- based on the Pippenger algorithm with 16-bit grouping.
- use signed scalar method.
- use multi-threading in software.

## Driver
We develop a new driver named superscalar_driver in `./cxx/host/superscalar_driver`.

The code structure of the driver:
- host.cpp: It is the host code, responsible for overall control, scheduling, resource allocation, invoking the FPGA hardware, and preprocessing scalar data, etc.
- common: xcl2 code
- libs/algo_ss: Our software algorithm library includes operations such as curve point transformations, point addition, field operations, and some computations for MSM post-processing.

The function structure of the driver:
- driver->init: Converts points, performs precomputation, and divides all points into four channels.
- driver->load_xclbin: Loads the xclbin, creates context, constructs OpenCL buffers, and transfers point data from the host to the FPGA device.
- driver->run: Preprocesses scalars through preprocessing threads using the signed bucket method.

If you want to print more logs, you can modify the macro `DEBUG_INFO` and `PROFILING_INFO` in `./cxx/host/superscalar_driver/host.cpp`.

## Points Representation
- Rust generates points in the affine coordinate system of the Short Weierstrass Jacobian curve.
- The driver needs to convert points passed from Rust into points in the affine coordinate system of the Twisted Edwards Extended curve for the FPGA.
- The hardware returns results points in the projective coordinate system of the Twisted Edwards Extended curve.
- Finally, the driver converts the points into the projective coordinate system of the Short Weierstrass Jacobian curve and returns them to Rust.

## Interface
In Rust, the `msm_init_for_ss` and `msm_mult_for_ss` functions are accessed through the FFI interface to communicate with the driver.

- msm_init_for_ss:
  It creates the superscalar driver, acquires resources, and allocates software memory.
- msm_mult_for_ss:
  It calls the driver: It drives the hardware to start its work, and once all four batches of MSM are executed, it returns the final data results to Rust.

## Generating Test Data
_Note: If you have already previously generated test data, you can skip this step._

Run with freshly generated points and save the testdata and save the testdata
to a directory. Note that this overwrites anything you have in 'path/to/write/to'
without warning!

```bash
# The --nocapture flag will cause println! in rust to be displayed to the user
CMAKE=cmake XCLBIN=<file> TEST_NPOW=10 TEST_WRITE_DATA_TO=path/to/write/to cargo test --release -- --nocapture
```

For `TEST_NPOW=10`, this should run in a few seconds. For `TEST_NPOW=24`, it can take
less than an hour to generate all the points.

## Running Tests

To load the points from disk, use the `TEST_LOAD_DATA_FROM` env-var. Note that
you don't have to specify `TEST_NPOW` when specifying `TEST_LOAD_DATA_FROM`.
(If you do, it will just be ignored)

```bash
# The --nocapture flag will cause println! in rust to be displayed to the user
CMAKE=cmake XCLBIN=<file> TEST_LOAD_DATA_FROM=path/to/load/from cargo test --release -- --nocapture
```

We also provide scripts. You can directly run like this: 
```bash
  ./run.sh
```
