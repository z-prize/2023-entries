# webgpu-msm

- [Quick Start](#quick-start)
- [Prize Desription](#prize-description)
- [Judging](#judging)
- [Prize Allocation](#prize-allocation)
- [Submission Instruction](#submission-instruction)
- [FAQs](#faqs)
- [Further Questions](#further-questions)
- [Trouble Shooting](#trouble-shooting)
- [Reference](#reference)

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

## Prize Description

### Summary

Multi-Scalar multiplication (MSM) operations are critical for building proofs and zk computations. They are also some of the slowest operations in these processes, especially running on consumer-grade hardware. This z-prize seeks to optimize for consumer-grade devices running msms in a browser environment.

### Optimization Objective

Achieve the fastest MSM run in a browser over a range of input vector lengths and a range of consumer hardware.

### Constraints

The implementation must provide the following interface in JavaScript: `compute_msm = async (baseAffinePoints: BigIntPoint[] | U32ArrayPoint[], scalars: bigint[] | Uint32Array[]): Promise<{x: bigint, y: bigint}>`
    
1.  The function name is `compute_msm`
    
2.  There are two input vectors: baseAffinePoints for a vector of elliptic curve affine points and scalars for a vector of finite field elements. There are two options for how the inputs are ingested, using `bigint`s or `Uint32Array`s:

    a.
    ```
    BigIntPoint: {
      x: bigint,
      y: bigint,
      t: bigint,
      z: bigint
    }
    scalar: bigint
    ```
    b.
    ```
    U32ArrayPoint: {
      x: Uint32Array,
      y: Uint32Array,
      t: Uint32Array,
      z: Uint32Array
    }
    scalar: Uint32Array
    ```

    Note -- the inputs are affine points, so the `z` property will be `1n` or `[0, 0, 0, 0, 0, 0, 0, 1]`. The `t` property is the field multiplication of `x` by `y`.
    
3.  The output is a single elliptic curve affine point: `{ x: bigint, y: bigint }`.
    
-   The submission should produce correct outputs on input vectors with length up to 2^20. The evaluation will be using input randomly sampled from size 2^16 ~ 2^20.
    
-   The submissions will be evaluated in a browser context on a variety of consumer-grade hardware.
    
-   All submissions must include documentation (in English) sufficient to understand the approach being taken.

## Judging

Submissions will be analyzed for both correctness and performance. All submission code must be open-sourced at the time of submission. Code and documentation must be dual-licensed under both the MIT and Apache-2.0 licenses.

### Correctness

We have provided test case data and answers to sanity-check your msm function against, as well as multiple msm implementations to check against.

The final correctness of the submission will be tested using randomly sampled test inputs/outputs that are not disclosed to the competitors during the competition in addition to the test input/output distributed to the competitor. All these test cases will be generated using Aleo's reference wasm implementation. Submissions failed in any test cases will be judged as incorrect and lose the opportunity to win the prize.

### Performance

To evaluate the performance of each submission, the prize committee will sample a large number of input vectors at random, in terms of both the input vector length and the point values. The input vector length will be sampled between 2^16 and 2^20. Then, we will compute the multi-scalar multiplication using the submission. The submitted score will be the relative speedup from the provided best webgpu msm implementation, averaged across many trials.

In addition, all submissions will be manually reviewed by the prize committee.

### Hardware & Benchmarks

The baseline will be the Demox Labs WebGPU MSM implementation over BLS12-377 G1. This baseline is originally implemented in typescript and wgsl. Submissions must beat this baseline by at least 10% in order to be eligible for the prize.

## Prize Allocation

Prizes will be given out in good faith and in the sole discretion of the prize committee.

### Timeline
* August 1 - 2023 Registration Opens
* September 15 - Competition Begins
* February - Submission Deadline
* March - Winners Announced

## Submission Instruction

Please include your implementation under the `submission` folder. The `compute_msm` function in `submission/submission.ts` will be run when benchmarking your submission.

## FAQs

### What is the focus of ZPrize Track 2: Beat the Best (Web)?

The goal of this track is to enhance the efficiency of client-side proving for the Aleo proof system using various web technologies.

### Can I utilize WebGPU and multithreading (web workers) for optimization?

Yes, you are allowed to use WebGPU and multithreading (web workers) to enhance performance.

### What elliptic curve will be used in this competition?

The competition will use BLS12-377 as the elliptic curve.

### Is there a preferred web technology for this track? Are there any restrictions on the technologies we can use?

The competition encourages the use of widely supported web technologies. There are no strict restrictions on the technologies you can use, but the focus is on ensuring client proving speed on different devices.

### Will participants have access to evaluation platforms for benchmarking? Can all WASM features be utilized?

Access to evaluation platforms will not be provided. The benchmarking aims to test solutions on various commodity devices and seeks general browser environment improvements. Most WASM features that work within Chrome v115 are allowed, including multithreading through web workers.

### Is WebGPU included in this prize category?

Yes! WebGPU may be used for any amount of the submission implementation.

### Which WebAssembly (Wasm) features are supported for this track?

You can use all of the WASM features that work within chrome v115. Multithreading is allowed for instance but browsers don't have threads so you have to use web workers to accomplish this using something like: https://github.com/GoogleChromeLabs/wasm-bindgen-rayon.

## Further Questions

If there are any questions about this prize:
* Ask in the zprize discord: https://discord.gg/DKqrz6F42D.
* Consult the zprize website: https://www.zprize.io/

## Trouble Shooting

Common issues:
* If you are unable to run the webgpu benchmarks, ensure you are using a webgpu-compatible browser.
* If you are not able to load the test case data, be sure you have installed git LFS. You can either reclone the repo after installing git LFS or run `git lfs fetch && git lfs pull`.
* If you run into general npm package errors, make sure you have nodejs v16 or later installed.
* If you are using webgpu functions and getting all 0s as output, you may have hit an out of memory error in the gpu. Reduce your input size or consider breaking your computions into smaller steps.
* If you are running on a windows machine with multiple graphics cards (ie. integrated chip + dedicated card) you can force Chrome to use one of them via windows settings.
Go to Settings -> Graphics Settings -> Custom options for apps -> Select / add Chrome -> Options -> Select graphics preference

## Reference

[1] Scalar-multiplication algorithms. [https://cryptojedi.org/peter/data/eccss-20130911b.pdf](https://cryptojedi.org/peter/data/eccss-20130911b.pdf)

[2] compute_msm functions in Typescript/WebGPU/WASM [https://github.com/demox-labs/webgpu-msm/blob/main/src/reference/reference.ts](https://github.com/demox-labs/webgpu-msm/blob/main/src/reference/reference.ts)j

[3] wgsl reference [https://www.w3.org/TR/WGSL/](https://www.w3.org/TR/WGSL/)

[4] aleo wasm reference [https://github.com/demox-labs/aleo/blob/gpu-expose/wasm/src/account/address.rs#L167](https://github.com/demox-labs/aleo/blob/gpu-expose/wasm/src/account/address.rs#L167)
