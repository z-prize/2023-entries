# ZPrize 2023 Prize 2

Authors: Tal Derei and Koh Wei Jie


## Documentation for our 2023 ZPrize Submission

By [Tal Derei](https://github.com/TalDerei/) and [Koh Wei Jie](https://github.com/weijiekoh/)

This document (https://hackmd.io/HNH0DcSqSka4hAaIfJNHEA) describes our implementation of in-browser multi-scalar multiplication (MSM) for [Prize #2: “Beat the Best” (WASM)](https://github.com/demox-labs/webgpu-msm) of the [2023 ZPrize competition](https://www.zprize.io/). Our codebase uses [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) extensively. WebGPU is a relatively new browser API which allows developers to leverage users' GPUs to accelerate heavy computational tasks via parallelism. We hope that our work can contribute to client-side proving libraries such as [webgpu-crypto](https://github.com/demox-labs/webgpu-crypto), so as to speed up client-side proof generation particularly for privacy-centric use cases.

Our work is heavily inspired by the <u>cuZK</u> paper ([Lu et al, 2022](https://eprint.iacr.org/2022/1321.pdf)), as well as techniques employed by competitors in the 2022 iteration of ZPrize. We also explored other techniques that no other prior teams had employed, and chose only those which performed the most efficiently. We strove to cite all third-party sources and will update this documentation to credit any that we missed.

## Quick Start

Ensure you have:

- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)  installed
- [Node.js](https://nodejs.org) 16 or later installed
- [Yarn](https://yarnpkg.com) v1 or v2 installed

Then run the following:

### 1) Clone the repository

```bash
git clone https://github.com/demox-labs/webgpu-msm && cd webgpu-msm
```

### 2) Install dependencies

```bash
yarn
```

### 3) Development

Run a local server on localhost:4040.

```bash
yarn start
```
Note -- running webgpu functions will only work on [browsers compatible](https://caniuse.com/webgpu) with webgpu.

## Performance

Please refer to [this spreadsheet](https://docs.google.com/spreadsheets/d/1JR8Rzern0DkXc8oHZWcGlPjxSCfP4n4-ZDrpYihgo8M/edit#gid=0) for benchmarks which we recorded on our own devices: **Apple M1 MacBook (2020), Apple M3 Pro MacBook (2023), Nvidia RTX A1000 GPU**.

## The elliptic curve

We completed submissions for both the **Twisted Edwards BLS12** and **Short Weistrass** **BLS12-377** elliptic curves.

### BLS12-377 curve

The BLS12-377 elliptic curve originates in the Zexe paper ([BCGMMW20](https://eprint.iacr.org/2018/962)). It has a 253-bit scalar field and 377-bit base field, whose moduli can be found on [this page](https://developer.aleo.org/advanced/the_aleo_curves/bls12-377/). Its G1 curve equation in Weierstrass form is $y^2 = x^3 + 1$.

The <u>**base**</u> field of the [BLS12-377](https://developer.aleo.org/advanced/the_aleo_curves/bls12-377/) curve is:

```
258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
```

and its <u>**scalar**</u> field is:

```
8444461749428370424248824938781546531375899335154063827935233455917409239041
```

### Twisted Edwards BLS12 curve

For our implementation using the [Twisted Edwards BLS12](https://developer.aleo.org/advanced/the_aleo_curves/overview) extended Twisted Edwards curve, the modulus of the scalar field is `8444461749428370424248824938781546531375899335154063827935233455917409239041`. Please note that although this is the base field of the <u>Edwards BLS12 curve</u>, the test harness code uses it as the scalar field order, and our submission produces results that match the provided test cases and reference implementations. The test harness provides curve points in the Extended Twisted Edwards (<u>ETE</u>) coordinate system. 

## Approach

<p align="center">
  <img src="https://hackmd.io/_uploads/BJI_KheCa.png" width="50%" />
</p>


The objective of the MSM procedure is to compute the linear combination of an array of $n$ elliptic curve points and scalars:

$$Q = \sum^n_{i=1}{k_iP_i}$$ where $n$ falls between $2^{16}$ and $2^{20}$, inclusive.

Our approach is very similar to the Pippenger method, and we assume that the reader is familiar with how it works. To recap, the Pippenger method has four main stages:

1. **Scalar decomposition**, where each scalar is split into $s$-bit windows.
2. **Bucket accumulation**, where points which correspond to the same scalar chunk are placed into a so-called bucket, and all the points in each bucket are independently aggregated.
3. **Bucket reduction**, where each bucket is multiplied by its bucket index, and the buckets are summed. 
4. **Result aggregation**, where each bucket sum is multiplied by $2^s$ raised to the appropriate power (using <u>Horner's method</u>). 

The main difference between our method and the Pippenger method is that we use sparse-matrix algorithms including sparse matrix transposition, a sparse matrix vector product (SMVP) algorithm, and a parallel running-sum bucket reduction algorithm. 

There are several stages to our procedure, the vast majority of which runs inside the GPU. This is to avoid relatively costly data transfers in and out of the GPU device. The only time data is written to the GPU is when it is absolutely necessary:

- At the start of the process, where the points and scalars are written to GPU buffers.
- Near the end of the process, to extract a relatively low number of reduced bucket sums so the final result can be computed in the CPU.

### The WebGPU programming model

GPUs execute code in a fundamentally parallel fashion. There are three main concepts to understand before reading the rest of this document: shaders, workgroups, and buffers.

A ***GPU shader*** is responsible for each stage of our process. A shader is basically code that runs in the GPU, and the way that it operates on data in parallel is orchestrated via the WebGPU API. Shaders are defined in the [WebGPU Shader Language (WGSL)](https://www.w3.org/TR/WGSL/) and at the time of writing, it is the only available language. The browser maps well to native APIs and compiles, via the dawn compiler, the WGSL (WebGPU shader language) to whatever intermediate target the underlying system expects: HLSL for DirectX12, MSL for metal, SPIR-V for Vulkan.

Besides WGSL, there is no other way to write GPU instructions in WebGPU, unlike other platforms.

To control parallelism, the programmer specifies the workgroup size and the number of workgroups to dispatch per shader invocation. Essentially, the number of threads is the product of the workgroup size and the number of workgroups dispatched. Each thread executes an instance of a shader, and the shader logic determines which pieces of data are read, manipulated, and stored in the allocated storage buffers. The workgroup size is <u>static</u>, but the number of workgroups is <u>dynamic</u>.

Data is stored in *buffers*, which can be read by and written to by the CPU. Buffers can also be read by and written to by different invocations of the same shader or different shaders. During execution, buffers can be accessed by any thread in any workgroup, and since WebGPU provides no guarantees on order of access, race conditions are possible and must be avoided using atomic operations and barriers.

It is also important to note that WebGPU terminology differs somewhat from that of other GPU programming frameworks and platforms, such as CUDA or Metal, which use terms like *warps* or *kernels* to refer to workgroups and shaders. WebGPU also differs in its capabilities; there are certain types of operations which can be done in CUDA but not in WebGPU, such as dynamic workgroup sizes and supporting subgroup functionality.

Readers may refer to [*WebGPU — All of the cores, none of the canvas*](https://surma.dev/things/webgpu/) for a more in-depth exploration of WebGPU concepts.

### Scalar decomposition and point coordinate conversion

First, we decompose the scalars into $s$-bit chunks (also known as windows) and use the [signed bucket index method](https://hackmd.io/@drouyang/signed-bucket-index) (which various ZPrize 2022 contestants also employed) to halve the number of buckets.

Furthermore, we store the point coordinates in $13$-bit limbs and in [Montgomery form](https://en.wikipedia.org/wiki/Montgomery_modular_multiplication) so as to speed up field multiplications. This technique is described by Gregor Mitscha-Baude in his [ZPrize 2022 submission](https://github.com/z-prize/2022-entries/tree/main/open-division/prize4-msm-wasm/mitschabaude#13-x-30-bit-multiplication), and we elaborate on how we adapted it to the WebGPU context in a section below.

We perform this step in parallel using the `convert_point_coords_and_decompose_scalars.template.wgsl` shader.

The input storage buffers are:

- `coords: array<u32>`
- `scalars: array<u32>`

The output storage buffers are:
- `point_x: array<BigInt>`
- `point_y: array<BigInt>`
- `chunks: array<u32>`

The shader performs the following steps:

1. Convert a set of point coordinates into $13$-bit multiprecision big integers (`BigInt` structs, each of which consists of $20$ limbs), multiply each `BigInt` by the **Montgomery** radix `r` using big-integer multiplication and **Barrett reduction**, and stores the results in the output buffers.
2. Decompose a scalar into signed scalar indices. Each signed index is represented by a `u32` value, which is negative if it is less than $2^{s-1}$, and store the results in an output buffer. (More information about the signed bucket index technique will be presented below.)

### Sparse matrix transposition

The next step is to compute the indices of the points which share the same scalar chunks, so that the points may be accumulated into buckets in parallel. Our implementation modifies the serial transpose algorithm adapted from [Wang et al, 2016, "Parallel Transposition of Sparse Data Structures"](https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf).

This step is from the cuZK paper, where it is dubbed sparse matrix transposition. Our implementation is slightly different, however. While the original cuZK approach stores the points in <u>ELL</u> sparse matrix form before converting them into CSR (compressed sparse row) sparse matrix form, we skip the ELL step altogether, and directly generate the CSR matrices. Specifically, our transposition takes the scalar chunks per subtask as the `all_csr_col_idx` array, step assumes that the `row_ptr` array is simply:

$$[0, r, 2r, \dots, m]$$

where $m = 2^s$ and $r = \frac{n}{2^s}$. If $s = 16$ and $n = 2^{16}$, then `row_ptr` is `[0, 65536]`.

Following the transposition algorithm, we generate the `all_csc_col_ptr` and `all_csc_val_idxs` output arrays.

#### Example

Let the scalar chunks for one subtask be: `[2, 3, 3, 5]` and the points be: `[P0, P1, P2, P3]`.

The expected bucket sum for this subtask should be: `2P0 + 3(P1 + P2) + 5P3`

We treat the scalar chunks as the `col_idx` array of a CSR matrix. The `row_ptr` array is set to: `[0, 4]`

In <u>dense</u> form, the matrix looks like:

```
x  x  P0  P1,P2  x  P3  x  x
```

Note that we have *P1* and *P2* in the same cell, which is technically not in line with how sparse matrices are constructed, but doing so does not affect the correctness of our result.

The result of transposition should yield a matrix whose dense form looks like:

```
x
x
P0
P1,P2
x
P3
x
x
```

In **CSR** form:

`row_ptr`: `[0, 0, 0, 1, 3, 3, 4, 4, 4]`
`val_idxs`: `[0, 1, 2, 3]`

#### Non-Uniform Memory Access 

It's worth noting that we index and access memory across memory addresses in a *non-uniform* manner. A more optimized sequential memory access pattern scheme can be potentially implemented with access to a sorting algorithm. 

#### Atomic Operations
Our implementation processes each CSR matrix subtask serially, and we found that using atomic operations in the transposition was *faster* than not using them. Although the webgpu compiler is generally treated as a black box of sorts, one [explanation](https://www.sciencedirect.com/topics/computer-science/atomic-operation) is: "**atomic operations are very efficient because they are implemented by low-level hardware and/or operating system instructions specially designed to support thread synchronizations**". 

#### Parallel Transposition
We considered implementing a parallel transposition scheme [(algorithm 3)](https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf), but it boasts a steep parallelization-memory tradeoff curve. For the `inter` array alone, it would require: `(256 threads  + 1) * 2^16 columns * 16 subtasks * 4 bytes` = **$1$ GB** in-browser memory for $2^{16}$ inputs, and a staggering **$17$ GB** in-browser memory for $2^{20}$ points. There are inherent memory requirements that scale with the number of threads and the size.

### Sparse matrix vector product and scalar multiplication (SMVP)

The transposition step allows us to perform bucket accumulation in parallel. Observe that the original scalar chunk values have been *positionally encoded* in the indices of elements of `row_ptr` whose consecutive element differs. For instance, `P0` is at index `2`, which is its corresponding scalar chunk.

Crucially, there are no data dependencies across iterations, so each thread can handle a pair of `row_ptr` elements and look up the required point data from a storage buffer.

For instance, when we iterate over `row_ptr` two elements at a time and encounter `row_ptr[2] = 0` and `row_ptr[3] = 1`, the loop index should be `2`, and when we look up the point index in `val_idxes[0]`, we know to add `P0` to bucket `2`.


```
i = 0; start = 0, end = 0
i = 1; start = 0, end = 0
i = 2; start = 0, end = 1
    Add points[val_idxs[0]] to bucket 2
i = 3; start = 1, end = 3
    Add points[val_idxs[1]] to bucket 3
    Add points[val_idxs[2]] to bucket 3
i = 4; start = 3, end = 3
i = 5; start = 3, end = 4
    Add points[val_idxs[3]] to bucket 5
i = 6; start = 4, end = 4
i = 7; start = 4, end = 4
i = 8; start = 4, end = 4
```

Note that our implementation uses signed bucket indices, so the actual procedure differs, but the iteration over `row_ptr` works in the same way.

### Bucket reduction

Now that we have the aggregated buckets per subtask, we need to compute the bucket reduction:

$$\sum_{2^{s - 1}}^{l=1}l{B}_l$$

We follow the `pBucketPointsReduction` algorithm from the cuZK paper ([Algorithm 4](https://eprint.iacr.org/2022/1321)) that splits the buckets into multiple threads, computes a running sum of the points per thread, and multiplies each running sum by a required amount. 

Here is an illustration of how it works for an example list of 8 buckets:

```js
B7 +
B7 + B6 +
B7 + B6 + B5 + 
B7 + B6 + B5 + B4 +
B7 + B6 + B5 + B4 + B3 +
B7 + B6 + B5 + B4 + B3 + B2 +
B7 + B6 + B5 + B4 + B3 + B2 + B1 +
B7 + B6 + B5 + B4 + B3 + B2 + B1 + B0 =
/*
t0      | t1      | t2      | t3
*/
// Thread 0 computes:
B7 +
B7 + B6 +
(B7 + B6) * 6
// Thread 1 computes:
B5 +
B5 + B4 +
(B5 + B4) * 4
// Thread 2 computes:
B3 +
B3 + B2 +
(B3 + B2) * 2
// Thread 3 computes:
B1 +
B1 + B0
```

Our implementation of `pBucketPointsReduction` had to be split into 2 GPU shaders because a single shader that contained multiple elliptic curve addition calls, as well as a call to a elliptic curve scalar multiplication function, failed to run on Apple **M1, M2, and M3** machines. When we split it into 2 shaders, however, it worked on these machines.

The output points are then read from the GPU and summed in the CPU. This is extremely fast (double-digit milliseconds) as the number of points to sum, ie. the number of subtasks multiplied by the number of threads, is relatively small.

### Result aggregation

The final stage of our MSM procedure is to use Horner's method. As described in page 8 of the cuZK paper, the final result $Q$ is computed as such:

$$Q = \sum_{j=1}^{\lceil\frac{\lambda}{s}\rceil} 2^{(j-1)s} G_j$$

To compute the result, we read each subtask sum from GPU memory, and perform the computation in the CPU. Our benchmarks show that doing this in the CPU is faster than in the GPU because the number of points is low (only $\lceil\frac{\lambda}{s}\rceil$ points), and the time taken for WebGPU or the GPU to compile a shader that performs this computation is far longer than the time taken to read the points and calculate the result. As described below, this is because compilation times for shaders that contain the `add_points` WGSL function are relatively high due to the complexity of the point addition function, which in turn calls the Montgomery multiplication function many times.

## Implementation differences

Our implementations for the **Edwards BLS12** and **BLS12-377** elliptic curves were structurally identical, and only differed in the following aspects:

1. Number of limbs per point coordinate

    - Since the Edwards BLS12 base field order set by the competition was 253 bits, we used **20 13-bit** limbs for each point coordinate. The BLS12-377 implementation, however, used **30 13-bit** limbs as its base field order is 377 bits (and the Montgomery multiplication algorithm requires the product of the limb size and number of limbs to be greater than the modulo's bit-length). 

2. The curve addition and doubling algorithms

    - For the Edwards BLS12 implementation, we implemented [extended twisted Edwards point addition and doubling](https://eprint.iacr.org/2008/522.pdf) algorithms. For BLS12-377, we used [projective algorithms](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) instead.

## Memory use

Our implementation accepts input data as `Buffer` objects. These `Buffer` objects are directly written to GPU memory (with minimal processing only in the BLS12-377 implementation so as not to exceed WebGPU's default buffer binding size limit).

### Edwards BLS12

#### Inputs from the CPU

The test harness provides the scalars as a `Buffer` of $32n$ bytes, and the points as a `Buffer` of $64n$ bytes, where *n* is the input size.

The X and Y point coordinates are laid out in alternating fashion, e.g. $[x[0][0], ..., x[0][31], y[0][0], ..., y[0][31], x[1][0], ...]$ in little endian form.

#### Scalar decomposition and point coordinate conversion

The decomposed scalar chunks are stored in a storage buffer of $4n\cdot\frac{256}{s}$ bytes, which the GPU reads as $n\cdot\frac{256}{s}$ 32-bit unsigned integers. 

The X and Y point coordinates are stored in storage buffers as $u32$ arrays. Each coordinate storage buffer contains `20 * n` $13$-bit values. They are each stored in a storage buffer of $4 \cdot 20n$ bytes, which the GPU reads as $20n$ 32-bit unsigned integers. 


You may notice that the output storage buffers are larger than the input buffers for storing the resulting points after executing the shader. The shader initially reads in `8` $32$-bit limbs for each coordinate, but internally converts them  to `20` 13-bit limbs, each stored in a $u32$.

#### CSR matrix transposition

The transposition step creates two new storage buffers:

1. `all_csc_col_ptr_sb`, which contains $4 \cdot (\frac{256}{s} \cdot 2^{c} + 1)$ bytes.
2. `all_csc_val_idxs_sb`, which has the same size as the storage buffer for the scalar chunks from the previous stage.

#### SMVP

Four storage buffers are used for all SMVP runs, one per point coordinate. The size of each storage buffer is $4\cdot\frac{2^s}{2} \cdot 20 \cdot \frac{256}{s}$ bytes.

#### Bucket points reduction

Four additional storage buffers are created for the two stages of the `pBucketPointsReduction` shader, one per coordinate. The size of each of these storage buffers is $4\cdot 256 \cdot 20 \cdot \frac{256}{s}$ bytes, assuming that we parallelise this stage by $256$ threads.

### BLS12-377

#### Inputs from the CPU

The test harness provides the scalars as a `Buffer` of $48n$ bytes, and the points as a `Buffer` of $96n$ bytes.

The X and Y coordinates are laid out in alternating fashion, e.g. $[x[0][0], ..., x[0][48], y[0], ..., y[0][48], x[1][0], ...]$.

#### Scalar decomposition and point coordinate conversion

The decomposed scalar chunks are stored in a storage buffer of $4n\cdot\frac{256}{s}$ bytes, which the GPU reads as $n\cdot\frac{256}{s}$ $32$-bit unsigned integers. 

The X and Y coordinates are stored in input storage buffers as $u32$ arrays. Each coordinate storage buffer contains `30n` 13-bit values. They are each stored in a storage buffer of $4 \cdot 30n$ bytes, which the GPU reads as $30n$ $32-bit$ unsigned integers.

#### CSR matrix transposition

*Same* as the CSR matrix transposition step from the Edwards BLS12 implementation.

#### SMVP

*Same* as the SMVP step from the Edwards BLS12 implementation.

#### Bucket points reduction

*Same* as the bucket points reduction step from the Edwards BLS12 implementation.

## Thread count
The workgroup size, x-dimension workgroups, y-dimension workgroups, and z-dimension groups are structured to minimize the number of shader invocations. The total number of threads launched for each phase is:

```
total threads = workgroup size * x workgroups * y workgroups * z workgroups
```

**Scalar decomposition and point coordinate conversion**
$n$ threads total

**CSR matrix transposition**
$num \ subtasks$ threads

**SMVP**
$n / 2$ * $num \ subtasks$ threads

**Bucket points reduction**
$256$ * $num \ subtasks$ threads

## Optimization techniques

### Montgomery multiplication

We implemented Montgomery multiplication for point coordinates to obtain a significant speedup in shader runtime. We adapted Gregor Mitscha-Baude's approach as described in [this writeup](https://github.com/mitschabaude/montgomery#13-x-30-bit-multiplication), specifically by benchmarking the performance of Montgomery multiplication using limb sizes that ranged from $12$ to $16$, inclusive.

In WASM, the max unsigned integer type is $64$-bits, so max limb size is $32$-bits. In the GPU context using WebGPU, the max unsigned integer type is $32$-bits, so max limb-size is $16$-bits wide. We therefore have to recalculate Mistcha’s calculations to determine the optimal limb size in WebGPU.

Modular multiplications are inherently slow operations because of the large bit-width of the elements, and **X*Y mod P** requires a division (~ $100$ times more expensive than a multiplication) to know how many times P has to be subtracted from the product to be contained in the bounds of the prime field. [Montgomery Multiplication](https://hackmd.io/lF8qDQZnR0quXhXJ1TkC5Q) makes modular multiplications more efficient by transforming the problem into Montgomery Form. 

The motivation at reducing the montgomery multiplication execution time is:

**1.** MSMs comprises about $70-80$% of the proof generation time, and (2) Field multiplications are the clear bottleneck in the computation, consuming ~ $60-80$% of the total MSM runtime, (3) carry operation comprises ~ $50$% of a single multiplication. 

Since WGSL shaders are limited to `u32` integers, and have no $64$-bit data types to support limb sizes above $16$ and up to $32$, we had to redo Mitscha-Baude's calculations to determine the number of loop iterations that can be done without a carry (also referred to as carry-free terms, or $k$). Our results, labeled with the same notation that Mitscha-Baude used, are as such:

| Limb size | Number of limbs | $N$ | Max terms to be added | $k$ | `nSafe` |
|-|-|-|-|-|-|
| 12 | 22 | 264 | 44 | 257 | 128 |
| 13 | 20 | 260 | 40 | 65 | 32 |
| 14 | 19 | 266 | 38 | 17 | 8 |
| 15 | 17 | 255 | 34 | 5 | 2 |
| 16 | 16 | 256 | 32 | 2 | 1 |

The so-called sweet spot for limb sizes if one is limited to `u32` integer types is that which has `nSafe` less than the maximum number terms per iteration, and whose algorithm is simplest. We found that $W = 13$ is the ideal limb size for minimizing the number of carry operations in a montgomery multiplication for GPU targets.

For limb sizes $12$ to $15$, the Montgomery multiplication algorithm we use is the same as the one that Mitscha-Baude describes in the abovementioned writeup (and contains checks on the loop counter modulo `nSafe`). For limb sizes $12$ and $13$, unused conditional branches are omitted (the "optmised" algorithm), while the algorithm for limb sizes $14$ and $15$, include the conditionals as they are necessary (the "modified" algorithm). For the sake of completeness, we also implemented a benchmark for Montgomery multiplication for limb sizes of 16 using the Coarsely Integrated Operand Scanning (**CIOS**) method.

Our benchmarks (in `src/submission/mont_mul_benchmarks.ts`) performed a series of Montgomery multiplications ((ar)^{65536} * (br)) involving two field elements in Montgomery form and a high cost parameter meant to magnify the shader runtime so that the average difference in performance across runs would be obvious.

The results from a Linux laptop with a **Nvidia RTX A1000 GPU**:

| Limb size | Algorithm | Average time taken (ms) |
|-|-|-|
| $12$ | optimized | $53$ |
| $13$ | optimized | $47$ |
| $14$ | modified | $43$ |
| $15$ | modified | $35$ |
| $16$ | CIOS | $64$ |

The results from an **Apple M3 MacBook (2023)**:

| Limb size | Algorithm | Average time taken (ms) |
|-|-|-|
| $12$ | optimized | $101$ |
| $13$ | optimized | $88$ |
| $14$ | modified | $104$ |
| $15$ | modified | $104$ |
| $16$ | CIOS | $366$ |

The results from an **Apple M1 MacBook (2020)**:

| Limb size | Algorithm | Average time taken (ms) |
|-|-|-|
| $12$ | optimized | $160$ |
| $13$ | optimized | $129$ |
| $14$ | modified | $146$ |
| $15$ | modified | $154$ |
| $16$ | CIOS | $645$ |

The source code for these benchmarks is in `src/submission/miscellaneous/mont_mul_benchmarks.ts`.

### Barrett reduction

In order to convert point coordinates to Montgomery form, we have to multiply each coordinate with the Montgomery radix $r$, modulo the field order. To achieve this, we explored two approaches:

- ["Vanilla" Barrett reduction](https://github.com/sampritipanda/msm-webgpu/blob/main/field.wgsl#L92)
- [Barrett-Domb reduction](https://github.com/ingonyama-zk/modular_multiplication)

Our implementation of <u>vanilla Barrett reduction</u> is based on code written by [Sampriti Panda, et al](https://github.com/sampritipanda/msm-webgpu/blob/main/field.wgsl#L92) and we compute the constants specific to the algorithm described by [Project Nayuki](https://www.nayuki.io/page/barrett-reduction-algorithm). Our WGSL and Typescript implementations of Barrett-Domb reduction can be found here at `src/submission/wgsl/barrett_domb.template.wgsl`, and are directly based on the reference implementation by [Ingonyama](https://github.com/ingonyama-zk/modular_multiplication).

In our benchmarks, we found that vanilla Barrett reduction was faster than Barrett-Domb reduction but we have not yet determined why this is the case.

The source code for the <u>Barrett-Domb benchmarks</u> is in `src/submission/miscellaneous/barrett_domb.ts` and the source code for the "vanilla" Barrett benchmarks is in `src/submission/miscellaneous/barrett_mul_benchmarks.ts`

### Signed bucket indices

This technique was used by previous ZPrize contestants and described in [drouyang.eth's writeup here](https://hackmd.io/@drouyang/signed-bucket-index). To adapt it to our implementation, we had to modify the scalar decomposition and bucket aggregation stages of our process.

In the scalar decomposition stage, our shader converts the scalar chunks into signed scalar indices (see `decompose_scalars_signed()` in `src/submission/utils.ts`) to see the algorithm in Typescript `src/submission/wgsl/convert_point_coords_and_decompose_scalars.template.wgsl` for the same algorithm in WGSL. Unlike the original algorithm, however, we add $2^{s-1}$ to each signed index so that they can be stored as unsigned integers while retaining the information about their sign. We do so as the transposition algorithm will not work correctly with negative `all_csr_col_idx` values.

Furthermore, we do not need to handle the case where the final carry equals 1 (as described in drouyang's writeup), as the highest word of the field modulus (`0x12ab`) is smaller than $2^{16-1}$.

In the bucket aggregation (SMVP) stage, which involves a separate shader, we subtract $2^{s-1}$ from each signed index, and negate the sum of points to be added to the corresponding bucket if the index is negative, before adding it to the bucket.

#### Example

The signed bucket indices method is relevant to the Pippenger algorithm. It reduces the number of buckets by half by decomposing each scalar into **K** $c-bit$ chunks in the range:
$-2^{c-1}, ..., -1, 1, 2^{c-1} - 1$

Each signed integer has **c** bits, and we ignore bucket 0 since $0 \cdot P = 0$


For example, if we have 2 scalars: `[255, 301]`, and the window size is 5, and the number of words per scalar is 2, the scalar decomposition will be:

```
[[15, 29], [24, 25]]
```

To reconstruct the original scalars, we first subtract $2^{5 -1}$ from each scalar:

```
[[-1, 13], [8, 9]]
```

We then treat each $i$ th element as the coefficient of a polynomial $f(x)$ and evaluate it at $2^5$:

$-1x^0 + 8x = -1 + 8(32) = 255$
$13x^0 + 9x = 13 + 9(32) = 301$

#### Which steps need to be modified

1. **Scalar decomposition**: convert scalars into signed index form, and then add `2 ** (c-1)` so that they are all nonzero, but retain the information about their original sign and value. We can call this "*shifted signed index form*". e.g. when `c = 5`, the signed index `-1` should be `15`.
2. **Bucket point reduction**: during this step, subtract `2 ** (c-1)` from each index and perform point negation depending on the sign of the result.

### Elliptic curve point addition and doubling

We explored two ways to perform scalar multiplication: the double-and-add method and Booth encoding.

The double-and-add method is the most straightforward, and is also what the test harness uses in [`Curve.ts`](https://github.com/demox-labs/webgpu-msm/blob/main/src/reference/webgpu/wgsl/Curve.ts#L161).

We discovered the Booth encoding method from discussions in the [privacy-scaling-explorations/halo2curves](https://github.com/privacy-scaling-explorations/halo2curves/pull/106) repository. Booth encoding converts a scalar to a signed encoding which reduces the number of point additions required in a scalar multiplication algorithm, assuming that negation is computationally trivial. For instance, the following scalar `7` in binary is `0111`, so `7P = P + 2P + 4P` (3 doublings and 3 additions). With Booth encoding, however, we convert `0111` to `100 -1`, and get `7P = 8P -1P` (3 doublings, 1 addition, and 1 cheap negation).

Our implementation of the Booth encoding method, however, performed slightly slower on random scalars than simple double-and-add, although it did perform faster than double-and-add on scalars with high Hamming weight, as expected.

Since the competition rules state that random scalars will be used, we opted to use the double-and-add method .

### Memory management

We ran into a problem where the [maximum storage buffer binding size limit](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits) in WebGPU, which is 134,217,728 bytes (****~134 MB****) by default, was insufficient for storing the coordinates of $2^{20}$ affine points in a single storage buffer.

```
* X coordinate = 20 limbs * 4 bytes per limb = 80 bytes (320-bits) 
* Y coordinate = 20 limbs * 4 bytes per limb = 80 bytes (320-bits) 

* Bytes needed for 2^20 X and Y coordinates:

160 bytes per pair * 2^20 points = 167,772,160 bytes (~167 MB)
```

The solution is to store each X and Y coordinates in a separate buffer:

```
Bytes needed for 2^20 X coordinates:

80 * 2 ** 20 = 83,886,080 (~84 MB) < 134,217,728 bytes (~134 MB)
```

This further highlights the wasted storage being consumed to store these coordinates! WGSL shader's don't currently support $u16$ types, instead each limb comprising the coordinate is stored as a $u32$, consuming *80 bytes / coordinate*. If WGSL introduces $u16$, then each coordinate will only consume `20 limbs * 2 bytes (16-bits) per limbs = 40 bytes (320-bits)`, which is closer to the underlying base field width of $253$-bits. 


Another limitation of the WebGPU API is that there is a default limit of $8$ storage buffers per shader (`maxStorageBuffersPerShaderStage`) in *Chrome v115*, which is what the competition is based on. We had to carefully design our end-to-end process to ensure that we did not exceed this limit. **Note:** *Chrome v123* increases the limit to supporting $10$ storage buffers per shader. 

1. In the `convert_point_coords_and_decompose_scalars` shader, we only handled the X and Y point coordinates, as well as the scalars, rather than all four **X, Y, T, and Z** coordinates and the scalars. Since the computation of the T and Z coordinates is trivial, we made that the task of the SMVP shader instead. This allowed us to have $6$ storage buffers instead of $10$.
2. In the `bucket_points_reduction` shader, all $8$ storage buffers contain the input and output point coordinates, so we use a uniform buffer to pass in runtime parameters. Using a uniform buffer instead of hardcoding the parameters in the shader code has the additional benefit of avoiding costly shader recompilation.

Do note that for our implementation using the BLS12-377 curve, we use projective coordinates, so we only use 3: the X, Y, and Z coordinates.

### Workgroup and thread allocation

Another factor in runtime performance is the number of threads that are dispatched to execute a shader. Dispatching more workgroups than necessary, however, incurs some overhead.
A previous version of our work used multiple calls to a bucket reduction shader that uses a tree-summation method to sum the buckets per subtask (though we subsequently replaced that shader with a parallel running-sum method). We found that our implementation was dispatching far too many threads per bucket reduction shader, and optimizing our code to dispatch only the necessary number of threads saved us hundreds of milliseconds to multiple seconds of runtime, depending on the input size.

This phenomenon, however, is not consistent across platforms. We noticed that reducing the number of X, Y, and Z workgroups dispatched by maximizing the workgroup size per dispatch led to a slight increase in performance ($200 - 300$ ms for the entire MSM) on a Linux machine with an Nvidia GPU, but there was no discernible performance difference on an Apple M1 Macbook.

## Techniques we explored but did not use

### Loop unrolling in WGSL shaders

We [unrolled the loops](https://github.com/TalDerei/webgpu-msm/pull/136) in big integer and Montgomery multiplication shaders, but this caused worse performance. As such, we did not use this technique.

### Workload balancing

Our implementation performs best on uniformly distributed scalars as this results in an even distribution of computational load per thread during the bucket aggregation stage. If the scalars are non uniformly distributed, some scalar chunks will appear more frequently than others, causing some threads to have to add more points to their respective buckets. Since the runtime of the bucket aggregation stage is determined by the slowest thread, the benefits of parallelism will be lost on such inputs.

We did not implement any means to solve this issue as the prize architects stated that the inputs would be uniformly distributed. If our work is to be integrated with a ZK prover for a protocol such as Groth16, however, we must expect non uniformly distributed scalars, and modifications must be made to evenly distribute the work across threads. Our work differs from the cuZK paper in this regard, as the latter describes extra steps (namely, the construction of ELL sparse matrices before converting them to CSR sparse matrices) to address this load imbalance. Ultimately, skipping this step allows our implementation to be more performant than one which is truer to the cuZK approach, but only for uniformly distributed scalars.

## Ideas and improvements

We expect the WebGPU API to evolve over the coming years, and we hope to see its improvements benefit client-side cryptography. Areas which have not yet been explored include:

- Multi-device proof generation, where multiple WebGPU instances across separate devices work together via peer-to-peer networking to share a workload.
- New low-level features, such as storing big integer limbs in [16-bit floating-point](https://www.w3.org/TR/webgpu/#shader-f16) variables.

We also hope to see more lower-level abstractions over the features of the underlying GPU device, such as the ability to set workgroup sizes dynamically (which is possible in CUDA), or lower-level language features such as a kind of assembly language for WGSL.

Our implementation of multi-scalar multiplication can be improved by:

- Supporting non uniformly distributed scalars, as the cuZK paper does.
- Supporting other curves.
- Exploring the use of the GLV endomorphism to speed up curve arithmetic.
- Refactoring the implementation into a library that works in both the command line (such as via [wgpu](https://github.com/gfx-rs/wgpu)) and in the browser.

Additionally, we explored the use of **floating-point** arithmetic in our implementation, but our error bounds weren't quite correct.

## Conclusion

We are grateful to the authors of the [cuZK](https://eprint.iacr.org/2022/1321) paper, last year's ZPrize participants, and other authors whose work we referenced. We hope that the outcome of this year's ZPrize can continue to drive innovation forward in the industry.

## Questions
For questions regarding this submission, please contact tal@penumbralabs.xyz or contact@kohweijie.com. 
