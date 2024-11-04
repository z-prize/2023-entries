extern crate pkg_config;
extern crate cc;
fn main() {
    println!("cargo:rustc-link-lib=OpenCL");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=uuid");
    println!("cargo:rustc-link-lib=dl");

    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/xilinx/xrt/lib");
    println!("cargo:rustc-link-arg=/opt/xilinx/xrt/lib/libxilinxopencl.so");
    //println!("cargo:rustc-link-search=/home/cysic/zPrize/binding/addmixed-bindings/target");

    pkg_config::Config::new();
    let src = ["c/addmixed.cpp"];
    let mut builder = cc::Build::new();
    let build = builder
        .files(src.iter())
        .include("c")
        .include("/home/cysic/Xilinx/Vitis_HLS/2023.1/include")
        .include("/opt/xilinx/xrt/include")
        .include("/opt/xilinx/xrt/include/CL")
        .cpp(true)
        .flag("-Wno-unused-parameter")
        .flag("-Wall")
        .flag("-c")
        .flag("-g")
        .flag("-std=gnu++14")
        .flag("-o")
        .define("USE_ZLIB", None);

    build.compile("addmixed_cpp");
}
