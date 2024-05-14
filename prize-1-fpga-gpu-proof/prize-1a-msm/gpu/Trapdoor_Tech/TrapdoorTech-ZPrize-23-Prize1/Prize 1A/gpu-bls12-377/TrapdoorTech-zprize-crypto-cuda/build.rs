fn main() {
    /* Link CUDA Driver API (libcuda.so) */

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=cuda");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    cc::Build::new()
        .cuda(true)
        .define("THRUST_IGNORE_CUB_VERSION_CHECK", None)
        .flag("-arch=sm_86")
        .file("src/cub/cub.cu")
        .compile("cub");
    println!("cargo:rerun-if-changed=src/cub.cu");
}
