# Implemetation

1. This code is just bench and test code for our GPU MSM solutions.
2. We provide 2 packages one for BLS12-377 and the other for BLS12-381 just like ark. The source code is on the following:\
[BLS12-377](./gpu-bls12-377/)\
[BLS12-381](./gpu-bls12-381/)
3. We also provide related bench and test code in the package, and provide detailed README to help build kernel, run test seperately. You can check the details in the following documents as you wish. **NOTE:** we provide precompiled kernel in the repository, if you want to build the kernel by yourself, you can also follow the steps in the document.\
[BLS12-377-README](./gpu-bls12-377/README.md)\
[BLS12-381-README](./gpu-bls12-381/README.md)
4. Our code is based on Matter Labs submission for the ZPRIZE 2022. We integrate some parts of the code into ours and do some optimizations. Thanks to the Matter Labs team.
5. Our implementations and optimizations are:
    * Enable BLS12-381
    * Replace CUDA runtime API with driver API. This aims to integrate with our code.
    * Allocate some host buffers with pinned memory. This aims to realize cuda asynchronous memory transfer. 
    * Develop a comprehensive and complete solution to handle the top window if the scalar bits are not divisible by the window bits.
    * Implement Twisted Edwards on BLS12-377.
    * We also provide correctness check with test_code, and the command is unchanged.

# Run benchmark and correctness

1. Make sure the two kernels exist
    * gpu-kernel-377.bin
    * gpu-kernel-381.bin
2. Exporting your nvcc path, for example:
    * export PATH=$PATH:/usr/local/cuda/bin/
3. Run benchmark
    ***
    cargo bench
    ***
4. Run correctness check
    ***
    cargo run --bin zprize-msm-samples --release --  -c 381 -d 24 -s 0\
    cargo run --bin zprize-msm-samples --release --  -c 377 -d 24 -s 0
    ***

# Performance
Our building environment is:
1. CUDA 12.1 and NVCC 12.1, V12.1.105
2. rustc: 1.74.0 (79e9716c9 2023-11-13)

Our test environment is:
1. Hardware: 8 cores AMD EPYC 7742 and an RTX 5000 Ada generation GPU
2. OS: Ubuntu 22.04.3 LTS
3. CUDA Version: 12.4

And we test MSM with size 2^24, the performance is :

|bls12-377|bls12-381|
|---|---|
|148.7ms | 158.8ms|

