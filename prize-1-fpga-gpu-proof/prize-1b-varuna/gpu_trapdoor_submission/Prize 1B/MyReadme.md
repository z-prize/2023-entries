### 1. Performance Optimizations

#### PLONK proving part

- Ignore lookup table proving part due to no lookup table is used by merkle circuit
- Product argument, Quotient calculation, Permutation/Poly calculation, MSM and FFT by GPU
- Opt buffer management between different components
- Pre-calcuation about constant data
- Share the same permutation data between multiple provers 



#### Synthesize Part

- Use multiple threads to generate and variables and gates information



### 2. How to Run? 

Building enviornments:

```
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```



Go to the folder "Prize 1B" and run: 

```
export PATH=/usr/local/cuda/bin/:$PATH   //set nvcc path
cargo bench
```



### 3. How to compile GPU kernel? 

The kernel is built and included in the source code by default. That's to say, to run the bench, you don't need to build the kernel by yourself.

If you want to build the kernel from source code,  please follows the steps: 

```
export PATH=/usr/local/cuda/bin/:$PATH   //set nvcc path
cd ./gpu/TrapdoorTech-zprize-ec-gpu-kernels
cargo run -- --cuda
```

You can find the kernel file with "gpu-kernel.bin" in the folder after compiling.



### 4. Results

Machine info:  AMD EPYC 75F3 32-Core Processor (128 threads) + 512G memory + NVIDIA RTX 6000



```
The total prove generation time is 11.916613924s
Aromatized cost for each proof is 2.979153993s
```



