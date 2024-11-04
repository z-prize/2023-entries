/***

Copyright (c) 2024, Ingonyama, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Tony Wu

***/

use std::env;
use std::path::PathBuf;

fn main() {
    if cfg!(feature = "cpu") {
        let dst = cmake::Config::new("merkle_accel")
            .build_target("merkle_accel_cpu")
            .build()
            .join("build/merkle_accel_cpu");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=static=merkle_accel_cpu");
        println!("cargo:rustc-link-search=native={}", dst.display());
    }

    if cfg!(feature = "stub") {
        let dst = cmake::Config::new("merkle_accel")
            .build_target("merkle_accel_stub")
            .build()
            .join("build/merkle_accel_stub");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=static=merkle_accel_stub");
        println!("cargo:rustc-link-search=native={}", dst.display());
    }

    if cfg!(feature = "gpu") {
        let dst = cmake::Config::new("merkle_accel")
            .build_target("merkle_accel_gpu")
            .define("USE_MERKLE_ACCEL_GPU", "ON")
            .build()
            .join("build/merkle_accel_gpu");
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=static=merkle_accel_gpu");
        println!("cargo:rustc-link-search=native={}", dst.display());
    }

    if cfg!(feature = "fpga") {
        let dst = cmake::Config::new("merkle_accel")
            .build_target("merkle_accel_fpga")
            .define("USE_MERKLE_ACCEL_FPGA", "ON")
            .build()
            .join("build/merkle_accel_fpga");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-link-lib=static=merkle_accel_fpga");
        println!("cargo:rustc-link-search=native={}", dst.display());
    }
}
