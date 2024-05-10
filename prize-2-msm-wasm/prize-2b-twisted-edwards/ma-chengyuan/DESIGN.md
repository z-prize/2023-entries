# Design Doc

Author: Chengyuan Ma (MaCY404 at mit.edu)

This submission is done as part of my UROP at MIT's Computation Structures Group.

## Variants

This submission has three variants:

- The GPU-only variant (`http://localhost:4040/?cpuWorkRatio=0`)
  - For machines with very powerful GPU and relatively weak CPU
- The co-computation variant (20% CPU + 80% GPU) (`http://localhost:4040/?cpuWorkRatio=0.2`)
  - For machines with relatively balanced GPU and CPU power
- The CPU-only variant (`http://localhost:4040/?cpuWorkRatio=1`)
  - For machines with weak or no GPU (such as Intel's integrated GPU)

I kindly request that the prize commitee benchmark all three variants, if
possible. If not possible, the co-computation variant takes priority over the
GPU-only variant, which takes priority over the CPU-only variant.

## Features

Some features that aren't directly related to performance but is good for
practical applications:

- Low VRAM consumption: uses less than 128MB of VRAM.
- Flexible window size: this submission uses Pippenger algorithm and supports a
  window size of 8, 9, 10, 11, 12, 13, 14, 15, 16, and 20 bits. The submission
  chooses a sane default, but it can be overriden with `windowSize` search
  parameter in the URL. More window sizes can be supported easily.
- CPU-GPU co-computation: A configurable portion of the input can be split to
  run on the GPU. This is controlled by the `cpuWorkRatio` query parameter. It can
  be any value from 0 (GPU-only) to 1 (CPU-only).
- Supports both input format. `Uint32Points` is preferred.

## Design

The submission follows the standard Pippenger's algorithm.

### Splitting input scalars by windows

This part is done in Rust. See `split` in `msm-wasm` and
`define_msm_scalar_splitter!` in `wasm-macro`.
The latter is a procedural macro that generates the splitting code for
any given window size with minimum overhead. E.g.,

```rust
// Split 256-bit scalar in [u32; 8] into 20-bit scalars, each stored as u32s
define_msm_scalar_splitter! { Split20: [u32; 8] -> [20u32] }
// Expands to:
pub(crate) struct Split20;

impl SplitImpl for Split20 {
    const WINDOW_SIZE: usize = (20u8) as usize;
    const N_WINDOWS: usize = 13usize;
    type Output = u32;
    fn split(input: &[u32; 8]) -> [u32; 13usize] {
        [
            (input[0usize] >> 16usize) as u32,
            (input[1usize] >> 28usize) as u32 | (((input[0usize] & 0xffffu32) as u32) << 4usize),
            ((input[1usize] >> 8usize) & 0xfffffu32) as u32,
            (input[2usize] >> 20usize) as u32 | (((input[1usize] & 0xffu32) as u32) << 12usize),
            (input[2usize] & 0xfffffu32) as u32,
            (input[3usize] >> 12usize) as u32,
            (input[4usize] >> 24usize) as u32 | (((input[3usize] & 0xfffu32) as u32) << 8usize),
            ((input[4usize] >> 4usize) & 0xfffffu32) as u32,
            (input[5usize] >> 16usize) as u32 | (((input[4usize] & 0xfu32) as u32) << 16usize),
            (input[6usize] >> 28usize) as u32 | (((input[5usize] & 0xffffu32) as u32) << 4usize),
            ((input[6usize] >> 8usize) & 0xfffffu32) as u32,
            (input[7usize] >> 20usize) as u32 | (((input[6usize] & 0xffu32) as u32) << 12usize),
            (input[7usize] & 0xfffffu32) as u32,
        ]
    }
}
```

### Bucketing (intra-bucket reduction)

This part is done in JavaScript + WebGPU. In the code this is referred to as
"intra-bucket reduction."

The reduction kernel (see `entry_padd_idx.wgsl`) takes four arrays:

- `input`, an array of input points (in projective coordinates),
- `output`, an array of output points (same),
- `bucket`, the final bucket array,
- `indices`: an array which has up to three indices `(i, j, k)` for each thread. Depending
  on the flags set, each thread can execute one of the following operations:

  - `output[k] = input[i] + input[j]`;
  - `bucket[k] = input[i] + input[j]`;
  - `bucket[k] += input[i]`;

The `input` array is initialized to be the array of base points. If scheduled
correctly, each invocation of the GPU kernel should leave half of the points
in the output array. Then we swap the output and input and invoke
the kernel again, and so on, until all points has been accumulated to their bucket.
In total, we run O(log N) rounds of invocations to finish the bucketing
process.

JavaScript, instead of Rust, is used to compute the indices array for each
invocation and interact with the WebGPU API, for the following reasons:

- There is a considerable overhead of using WebGPU API in Rust
  (at least through `wgpu`), mostly due to the need to copy data from WASM
  heap to JS heap. This offsets Rust's performance advantage.
- The indices for the next invocation are computed while the current
  invocation is inflight. The GPU is kept busy anyways, so the performance on
  CPU is less important.

The WGSL code for EC and finite field operations is based on the starter code. I
manually performed loop unrolling, function inlining, and copy elision, which
seem to be quite effective.

For inputs with a large number of base points, bucketing is done in batches to
keep the VRAM usage under 128MB. Despite ZPrize's allowance for more VRAM, batching remains advantageous because copying data from JS
heap to GPU memory turns out to be very expensive. If everything is done in one pass, we
must transfer the entire input data to the GPU first -- and we won't be able to do
anything else in that time. If we batch, we can copy the input for the next batch to a staging buffer while the previous batch is in flight. Then, when we start the next batch, we simply need
to initiate a GPU-side copy which is much much faster than copying from JS heap.
In this case, the time for JS-GPU transfer is effectively covered by the running time of the
previous batches and does not add to the total running time. The only uncovered part is the transfer time of the first batch. Hence, as long as the batch is large enough to keep the whole GPU busy,
the batch size should be as small as possible.

### Computing and aggregating bucket MSMs

This part is done in Rust and parallelized with Rayon. Nothing special here.

## Attempted optimizations that did not quite work

- Doing point addition under Montgomery forms turns out to be slower with
  WebGPU. In particular, regular multiplication of two 256-bit integers cannot be
  efficiently implemented with WebGPU because
  - WebGPU does not support 64-bit integers
  - Even though WebGPU supports 32-bit integers, it does not support extracting the top 32-bits of
    the product of two u32s.
  - WebGPU does not yet support subgroup (warp-level) primitives.
- Doing the last step in WebGPU too: seems to be slower than CPU.

# Building

To build the WASM part of the submission,

```bash
cd ./src/submission/msm-wasm
wasm-pack build --target web
```

A nightly Rust compiler is needed. I used `nightly-2023-12-28`. Other nightly
versions may work but is not tested.

After building the WASM part, the rest of the submission can be built
with `npx webpack --config webpack.prod.config.cjs`, or served
with `yarn start`, as usual.

If you have any trouble building,

- Try unzipping `pkg.zip` under `src/submission/msm-wasm` to
  `src/submission/msm-wasm/pkg`. `pkg.zip` contains the pre-built WASM binary.
- Try https://danglingpointer.fun:4041/ which deploys a full prod build of this submission. Add query parameters (e.g., `?cpuWorkRatio=0`) as usual.
