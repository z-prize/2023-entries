# zprize-2023-submission-fpga

## Running the bench
Now we need to initialize the card and program the shell partition.
This only needs to be done once per reboot of the server.
Locate the BDF of the U250 card using lspci:

```bash
lspci | grep Xilinx
```

You should see at least two entries appear, for example:
```
01:00.0 Processing accelerators: Xilinx Corporation Device 5004
01:00.1 Processing accelerators: Xilinx Corporation Alveo U250
```
In this case, the `01:00.0` function is the management interface, and `01:00.1`
is the user interface.

We need to export the string related to the management interface to
the shell and run `setup_hw.sh`.

```bash
export U250_DEVICE=01:00.0
./setup_hw.sh
```

Now we download the pre-built FPGA binaries from the v1-zprize-2023-submission release.
Due to permissions on this repo, you may need to download them manually.
https://github.com/yrrid/zprize-2023-submission-fpga/releases/tag/v1-zprize-2023-submission
```bash
mkdir xclbin
wget https://github.com/yrrid/zprize-2023-submission-fpga/releases/download/v1-zprize-2023-submission/nt_amsm_bls12_377.xclbin -P xclbin
wget https://github.com/yrrid/zprize-2023-submission-fpga/releases/download/v1-zprize-2023-submission/nt_amsm_bls12_381.xclbin -P xclbin
```
The host code will look for these binaries in the xclbin directory located at the same file hierarchy as the Cargo.toml.

Then simply run the bench.
```bash
cargo bench --features fpga
```

To check correctness run:
```bash
cargo run --release --features fpga --bin zprize-2023-submission-fpga --  -c 377 -d 1048576 -s 0
cargo run --release --features fpga --bin zprize-2023-submission-fpga --  -c 381 -d 1048576 -s 0
```

## Instructions for build
### Warning could take > 8 hours depending on your machine
To build the FPGA binary from source, make sure you have the following installed:
- AMD Vitis 2023.2
- XRT 2023.2
- Alveo U250 Development Target Platform

You can find these at https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted.

Now we can build:
```bash
cd fpga/nt_amsm/system/hw/
source /tools/Xilinx/Vivado/2023.2/settings64.sh
```

For bls12_377
```
CURVE=bls12_377 make -j8 kernels
CURVE=bls12_377 make link_hw
```

For bls12_381
```
CURVE=bls12_381 make -j8 kernels
CURVE=bls12_381 make link_hw
```

The output binaries will be generated in `./work` directory.

## Brief Architecture Overview
For our FPGA submission we implemented a batched affine approach, with 13-bit signed windows for both bls12_377 and bls12_381 curves.
This design has several advantages:
- Allowed us to fit 3 ec-adders on the chip.
- Reduced memory requirements due to affine points representation.
- Allowed us to target both curves with one code base.

More detailed documentation will be supplied soon.

In our testing on the supplied VM we obtain the following batch-4 latencies for 2^24 point MSMs:
- bls12_377: 2.0406s
- bls12_381: 2.0455s

```
Zprize 23, prize 1/bench 4 batches of 16777216 msm over BLS12-377 curve; splitted form
                        time:   [2.0402 s 2.0406 s 2.0410 s]

Zprize 23, prize 1/bench 4 batches of 16777216 msm over BLS12-381 curve; splitted form
                        time:   [2.0451 s 2.0455 s 2.0459 s]
```
