# WASM MSM Z-prize challenge

## Introduction

The following is Yrrid Software and [Snarkify](https://snarkify.io/) (joint team) Twisted Edwards WASM submission to the 2023 Z-Prize.

## The Goal

Improve the performance of WASM based MSM on a Twisted Edwards curve based on the BLS12377 scalar field.

We have tested our solution on a MacBook Pro M2 laptop, with 10 cores.  Our solution uses multiple cores and
SIMD, as specified by the competition.  We do not use the GPU for acceleration.  Performance seems
to vary somewhat from run to run, with the first run being the slowest, probably due to JIT compilation
overhead.
 
Ignoring the first run and averaging over a few runs, we oberve the following performance numbers on the MacBook Pro M2
(10 cores):

|Input Vector Length | Our WASM Submission (milliseconds) | 
| --- | --- |
| 2^16 | 81 |
| 2^17 | 117 |
| 2^18 | 203 | 
| 2^19 | 354 | 
| 2^20 | 660 | 

## Submission Authors

This solution was developed by Niall Emmart, Sougata Bhattacharya, and Antony Suresh.  

## Getting Started

Ensure you have:

- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-st
orage)  installed
- [Node.js](https://nodejs.org) 16 or later installed
- [Yarn](https://yarnpkg.com) v1 or v2 installed

Then run the following:

### 1) Clone the repositories

```bash
git clone https://github.com/yrrid/submission-wasm-twisted-edwards
git clone https://github.com/demox-labs/webgpu-msm
```

### 2) Copy the data

```bash
cp -r webgpu-msm/public/test-data submission-wasm-twisted-edwards/public
```

### 3) Install dependencies & start server

```bash
cd submission-wasm-twisted-edwards
yarn
yarn start
```

### 4) Chrome Web Page

Open Chrome and browse to [http://localhost:4040/](http://localhost:4040/)

## Optimizations in our Solution

In this section, we give a high level overview of the optimizations we have used to accelerate the MSM computation:

-  Our implementation was written completely in C and cross compiled to WASM.
-  We used Pippenger's bucket algorithm, supporting window lengths of 14 and 16 bits.
-  We slice the scalars, and use a parallel bin sort algorithm to build a list of points for each Pippenger bucket.
-  We use signed digits to minimize the number of buckets in the system.
-  We use parallel algorithms (run across multiple cores) for sorting, bucket accumulation, and bucket reduction.
-  We haven't used any libraries (such as pthreads) to distribute work across threads, instead we have rolled our
   own, using atomic counters and a simple barrier mechanism.  For each major step in the computation, we break that
   step into small units of work, which are numbered 0 through n-1.  Each thread checks out a unit of work (using an
   atomic counter) and processes that unit.  Threads repeat that process until all n units of work are complete and
   then all threads check in to a barrier before moving on to the next major step in the computation.  This work scheduling
   is very fast and efficient on the architectures we tested.
-  The FF and EC routines have been carefully optimized:
   - This Twisted Edwards curve uses a 253-bit finite field.  We implement this using a sequence of 5x 51-bit limbs,
     where each limbs is stored as a 51-bit integer in an double precision FP64 value.   We use FMA hardware to compute
     low and high products.  The basic approach is described in the paper ["Faster Modular Exponentiation using Double 
     Precision Floating Point Arithmetic on the GPU](http://www.acsel-lab.com/arithmetic/arith25/pdf/17.pdf), 
     *2018 IEEE 25th Symposium on Computer Arithmetic (ARITH)* by Emmart, Zheng and Weems.
   - We use a [non-unified 7-product Twisted Edwards EC mixed addition](https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-madd-2008-hwcd-4)
     for bucket accumulation and a [non-unified 8-product Twisted Edwards EC addition](https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4)
     for bucket reduction.  
   - We use WASM 128-bit SIMD instructions to process two curve operations simulateously in each thread.
   - We just use the grade-school O(N^2) multiplication algorithm and word-by-word O(N^2) Montgomery
     reduction.
   - On a MacBook Pro M2 (10 core), our solution delivers about 28M EC mixed additions per second, or
     about 196M finite field multiplication ops/sec.

## Fused-Multiply-And-Add Hardware

- Our implementation takes advantage of FP64 Fused-Multiply-and-Add (FMA) instructions that are available on most modern processors
  (for example, Intel, AMD, 64-bit ARM processors, etc).  It would be possible to build a fall-back path in the code that uses smaller
  limbs sizes in each FP64 value.  However, none of the machines we tested on required a fall-back solution, so we didn't implement one. 
  Our best guess though is that a fall-back implementation would be about 10-25% slower than this implementation.

## Building Binaries from Source

- The wasm binary was built on Ubuntu 22.04 using clang-17.
- Clone the GIT repo, cd to the yrrid directory
- Run `make msm.wasm` 
- During development, we used GCC and x86 SIMD.  See README for details.

## Footnotes

We have tried to limit our changes to the src/submission folder.  But we have had make a few edits
to the tooling files.  In particular, we have updated webpack.dev.config.cjs.  We have deleted 
webpack.prod.config.cjs to avoid any confusion, and we have updated `src/ui/AllBenchmarks.tsx`
to use the `bufferPoints` and `bufferScalars` compute_msm interface.

## Questions

For questions about this submission, please contact `nemmart at yrrid.com`.
