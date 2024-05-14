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

    if cfg!(feature = "fpga") {
        println!("cargo:rerun-if-changed=yrrid-fpga-msm/Main.cpp");
        let mut bls12_381_xclbin_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut bls12_377_xclbin_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        bls12_381_xclbin_path.push("xclbin/nt_amsm_bls12_381.xclbin");
        bls12_377_xclbin_path.push("xclbin/nt_amsm_bls12_377.xclbin");
        let bls12_381_xclbin_str = format!("\"{}\"", bls12_381_xclbin_path.display());
        let bls12_377_xclbin_str = format!("\"{}\"", bls12_377_xclbin_path.display());

        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .flag("-fopenmp")
            .file("yrrid-fpga-msm/Main.cpp")
            .define("__BLS12_381_XCLBIN_PATH__", &*bls12_381_xclbin_str)
            .define("__BLS12_377_XCLBIN_PATH__", &*bls12_377_xclbin_str)
            .include("/opt/xilinx/xrt/include/")
            .opt_level(3)
            .compile("fpga-msm");

        println!("cargo:rustc-link-search=native=/opt/xilinx/xrt/lib/");
        println!("cargo:rustc-link-lib=xrt_coreutil");
        println!("cargo:rustc-link-lib=uuid");
        println!("cargo:rustc-link-lib=gomp");
        println!("cargo:rustc-env=LD_LIBRARY_PATH=/opt/xilinx/xrt/lib/");
    }
}
