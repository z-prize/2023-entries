fn main() {
    /* Link CUDA Driver API (libcuda.so) */

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=cuda");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    cc::Build::new()
        .cuda(true)
        .define("THRUST_IGNORE_CUB_VERSION_CHECK", None)
        .flag("-arch=native")
        .flag("-Xcompiler")
        .flag("-O3")
        .flag("-I../matter_gpu_msm/")
        .flag("-Isrc/cub/")
        .flag("-std=c++17")
        .file("src/cub/cub.cu")
        .file("src/cub/generate_prove.cu")        
        .file("src/cub/poseidon_hash.cu")
        .object("src/cub/poly_functions.o")
        .object("../yarrid-gpu-msm/msm_yrrid.o")
        .object("../matter_gpu_msm/msm_matter.o")
        .compile("cub");
    println!("cargo:rerun-if-changed=src/cub.cu");
    println!("cargo:rerun-if-changed=src/generate_prove.cu");
    
    //.object("src/cub/poseidon_hash.o")
}
