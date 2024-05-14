# Z-Prize MSM on the GPU 

## Introduction

  TODO 

## Full Run Performance

  TODO 

## Building and running the submission

Install [CUDA 11.7](https://developer.nvidia.com/cuda-downloads). Install rust, for example, `rustup install stable`.
Next clone the repository, and run the benchmark.

```
git clone https://github.com/storswiftlabs/ZPrize-23-Prize1A.git
cd ZPrize-23-Prize1A
cargo bench
```

To run the correctness test, use the following command. Note, it can take several hours to generate the input points for a full 2^24 run.

```
cargo test --release
```

## GPU requiments

We adopt the same GPU requirements as the ZPrize competition.  The software has been tuned to run a batch a batch of 4 x 2^24 MSMs on the 
target GPU, an NVIDIA A40. The solution requires Compute Capability 8.0 (Ampere) and roughly 46 GB (46 x 2^30 bytes) of memory.

## Optimizations in our solution
  TODO 

## Questions

  TODO 
