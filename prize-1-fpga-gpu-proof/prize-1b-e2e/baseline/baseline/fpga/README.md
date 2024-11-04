# 1. Introduction

This baseline repo is an optimization of FPGA version based on cysic-labs • [ZPrize-23-Prize1](https://github.com/cysic-labs/ZPrize-23-Prize1).

You will learn how to use HLS to match the C function into FPGA, and integrated it into Rust code.

Hope you will enjoy.

# 2. Init env

source relevant shell , to use the xlinx lib. 

When you open a new terminal, you need to rerun the init command. Thus, it would be better to write this command in .bashrc

```bash
source /home/cysic/Xilinx/Vitis/2023.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
```

# 3. Demo

this chapter will introduce how to use FPGA by using HLS.

## 3.1 Clone the demo

```bash
git clone git@github.com:cysic-labs/alveo-demo.git
cd alveo-demo
```

## 3.2 Run the demo

the demo provide a matrix calculation sample, by using HLS

```bash
#1. run the hardware synthesis, it will take around two hours, so be patient
make hw

#2. run the host code compile
make sw

#3. run the test
./build/host ./mmult.xclbin

#tips: you can also run make test, to do all the thing above
```

# 4. HLS Coding

## 4.1 Benchmark

you need to find the place in the code where can be accelerated.

## 4.2 Recode by using C

When you find the code can be accelerated, recode by C.

## 4.3 Synthesis C code into FPGA

if your c file is “sample.cpp”

you can change the DEMO Makefile like 

```makefile
hw: sample.xclbin

mmult.xo: sample.cpp
	v++ --kernel mmult --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -t hw -s -g -c -o $@ $<

mmult.xclbin: sample.xo
	v++ --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 -t hw -s -g --config connectivity.ini -l --jobs $(shell nproc) -o $@ $<
```

and run “make hw ” to synthesis

```makefile
make hw
```

tips: make sure your c function is right, otherwise you will waste your time to get a wrong FPGA Binary.

## 4.4 Package the FPGA Binary

you can reference the [“mmult_fpga“ function](https://github.com/cysic-labs/alveo-demo/blob/066ba65d3a327cce62dffdcbb3b4122349e29043/host.cpp#L73)  in host.cpp, what you need to do is to change the interface according to your design.

# 5. Integrate your FPGA design

when you finish the HDL design, and verify the function successfully, you are supposed to integrate your design into rust code.

## 5.1 Bindgen your design

1. install bindgen tool

```bash
 cargo install bindgen-cli
```

2. use bindgen tool to create binding.rs

```bash
bindgen xxx.h -o binding.rs
```

3. create build.rs

the follow code is an example, you need to change the c file name and path.

```rust
fn main() {
    pkg_config::Config::new();
    let src = ["c/addmixed_rust.cpp"];
    let mut builder = cc::Build::new();
    let build = builder
        .files(src.iter())
        .cpp(true)
        .flag("-I/home/cysic/Xilinx/Vitis_HLS/2023.1/include")
        .flag("-I/opt/xilinx/xrt/include")
        .flag("-I/opt/xilinx/xrt/include/CL")
        .flag("-std=gnu++14")
        .flag("-lOpenCL ")
        .flag("-lpthread")
        .flag("-luuid")
        .flag("-ldl")
        .flag("-Wl,-rpath,/opt/xilinx/xrt/lib");

    build.compile("addmixed_rust_cpp");
}
```

4. test cargo compile

if build successfully,  c file works.

```rust
cargo build
```

## 5.2 Add your design

Add the c lib created by bindgen tool  into cargo.toml, in this way, you can replace the old function with the new FPGA function.

# 6. Run test

You can run the test when you finish the code.

```rust
cd merkle-tree
cargo run --release
```

tips: if you want to do a quick test, you can change [the height size](https://github.com/cysic-labs/ZPrize-23-baseline-op/blob/3654f69128c4e57122839dd04b2c1f5a31064e79/merkle-tree/src/lib.rs#L20) into a small one, like 7.