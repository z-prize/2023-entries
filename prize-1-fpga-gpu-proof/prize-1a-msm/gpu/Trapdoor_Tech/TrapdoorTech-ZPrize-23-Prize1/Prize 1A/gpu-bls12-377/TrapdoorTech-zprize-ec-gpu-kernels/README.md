# Elliptic Curve GPU Kernel

## Usage

specify generating cuda kernel:

``` bash
# for cuda kernel
$ cargo run  --features "bls12_377, cuda" --release

```

## Running a full TPS test

1. generate cuda kernel for the test, using above commands
2. prepare all the files needed and copy them to XXXX folder: `gpu-kernel.bin`

