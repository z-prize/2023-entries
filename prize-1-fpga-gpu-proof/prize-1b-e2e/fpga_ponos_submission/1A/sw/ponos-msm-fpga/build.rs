extern crate cxx_build;

fn main() {
    cxx_build::bridges(["src/engine.rs"])
        .file("src/engine/engine12377.cpp")
        .file("src/engine/engine12381.cpp")
        .file("src/engine/math.cpp")
        .cpp(true)
        .std("c++2a")
        .flag("-fopenmp")
        .define("NDEBUG", None)
        .include("/opt/xilinx/xrt/include")
        .compile("engine");

    println!("cargo:rerun-if-changed=include/engine.hpp");
    println!("cargo:rerun-if-changed=src/engine/engine12377.rs");
    println!("cargo:rerun-if-changed=src/engine/engine12377.cpp");
    println!("cargo:rerun-if-changed=src/engine/engine12381.cpp");
    println!("cargo:rerun-if-changed=src/engine/fpga.hpp");
    println!("cargo:rerun-if-changed=src/engine/math.cpp");
    println!("cargo:rerun-if-changed=src/engine/math.hpp");
    println!("cargo:rerun-if-changed=src/engine/utils.hpp");

    println!("cargo:rustc-link-lib=dylib=gomp");

    println!("cargo:rustc-link-lib=dylib=gmp");

    println!("cargo:rustc-link-search=native=/opt/xilinx/xrt/lib");
    println!("cargo:rustc-link-lib=dylib=xrt_core");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/xilinx/xrt/lib")
}
