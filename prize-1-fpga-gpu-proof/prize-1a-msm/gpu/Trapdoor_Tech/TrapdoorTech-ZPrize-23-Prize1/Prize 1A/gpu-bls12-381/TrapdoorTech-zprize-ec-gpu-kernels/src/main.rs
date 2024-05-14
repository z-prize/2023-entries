mod opencl;
// use opencl::*;

mod cuda;
// use cuda::*;

mod error;
use error::*;

mod sources;
// use sources::*;

mod structs;
use structs::*;

mod utils;
// use utils::*;

mod tools;
// use tools::*;

// use clap::{App, Arg};
use std::env;

// use crypto_cuda::{CudaDevice, CudaContext, CudaModule, CudaParams};
use crypto_cuda::CudaDevice;

pub use ocl::{Device as OpenclDevice, Platform as OpenclPlatform};

fn generate_opencl_kernel() {
    println!("generating OpenCL kernel...");
    opencl::write_kernel::<Fr, Fq>("gpu-kernel-381.bin").unwrap();
}

fn generate_cuda_kernel(curve: &str) {
    println!("generating Cuda kernel...");
    cuda::write_kernel::<Fr, Fq>("gpu-kernel-381.bin", curve).unwrap();
}

fn feature_check() {
    let curves = ["bls12_377", "bls12_381"];
    let curves_as_features: Vec<String> = (0..curves.len())
        .map(|i| format!("CARGO_FEATURE_{}", curves[i].to_uppercase()))
        .collect();

    let mut curve_counter = 0;
    for curve_feature in curves_as_features.iter() {
        println!("curve:{}", curve_feature);
        curve_counter += env::var(&curve_feature).is_ok() as i32;
    }

    for (key, v2) in env::vars() {
        println!("v :{}, v2:{}", key, v2);
    }

    match curve_counter {
        0 => panic!("Can't run without a curve being specified, please select one with --features=<curve>. Available options are\n{:#?}\n", curves),
        2.. => panic!("Multiple curves are not supported, please select only one."),
        _ => (),
    };
}

fn main() {
    let _ = env_logger::try_init();

    let mut curve = "";
    if cfg!(feature = "bls12_377") {
        curve = "-DBLS12_377";
    } else if cfg!(feature = "bls12_381") {
        curve = "-DBLS12_381";
    }

    let mut kernel_type_opencl = false;
    let mut kernel_type_cuda = false;

    if cfg!(feature = "cuda") {
        kernel_type_cuda = true;
    } else if cfg!(feature = "opnecl") {
        kernel_type_opencl = true;
    }

    println!("curve:{}, kernel_type_cuda:{}", curve, kernel_type_cuda);

    if kernel_type_opencl {
        generate_opencl_kernel();
    }

    if kernel_type_cuda {
        generate_cuda_kernel(curve);
    }

    let curves = ["bls12_377", "bls12_381"];
    let kernels = ["cuda", "opencl"];
    if !kernel_type_opencl && !kernel_type_cuda || curve.is_empty() {
        println!("Error: must specify kernel type \n{:#?} or please select one with --features=<curve>. Available options are\n{:#?}\n",kernels, curves);
        println!("example: cargo run  --features \"bls12_381, cuda\"");
    }
}
