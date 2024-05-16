use std::fmt::format;
use std::path::PathBuf;

fn main() {
    if cfg!(feature = "cpu") {
        println!("cargo:rerun-if-changed=yrrid-cpu-msm/Main.cpp");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .opt_level(3)
            .file("yrrid-cpu-msm/Main.cpp")
            .compile("cpu-msm");
    }

    if cfg!(feature = "gpu") {
        cc::Build::new()
            .cpp(true)
            .cuda(true)
            .flag("-arch=sm_80")
            .std("c++17")
            .file("yrrid-gpu-msm/MSMBatch.cu")
            .opt_level(3)
            .compile("gpu-msm");
    }
}
