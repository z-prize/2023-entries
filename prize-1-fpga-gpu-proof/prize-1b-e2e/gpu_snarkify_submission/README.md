# ZPrize 2023 - Prize 1b submission

## Introduction

The following has been a team effort by Niall Emmart, project architect and GPU engineer representing [Snarkify](https://snarkify.io/), 
Tony Wu, FPGA engineer representing [Ingonyama](https://ingonyama.com/), and Fabrizio Muraca, Rust engineer representing himself.

Our original goal was to submit both a GPU implementation and an FPGA implementation.  But despite our best efforts, the FPGA
implementation didn't quite make it in time for the Oct 7th deadline.  We do intend to release the complete FPGA implementation
in the next few weeks.

## Performance

The GPU implementation runs a batch of 4 proofs in 2.16 seconds, or about 540 milliseconds per proof on an A6000 Ada card.

The baseline Rust code running on the host CPU takes around 490 seconds per proof.  Thus our GPU implementation is about 900x
faster than the baseline host implementation.

## Approach

Our approach consisted of three phases.   Phase 1 - optimize the Rust proof system, eliminate unused proof system features, such as lookups,
and any associated polynomials and computations.  Phase 2 - build a ground up C implementation of the proof system, from 
witness generation through the 5 proving rounds.  The only part of the computation that remains in RUST is the transcript object.
Phase 3 - implement the accelerators on the GPU and FPGA.

| Person | Tasks |
|--|--|
| Fabrizio Muraca | Optimized the Rust proof system, eliminating unused features and related computations |
| Niall Emmart | GPU implementation, host side C library, initial C proof system development |
| Tony Wu | FPGA implementation, rust integeration, analysis and data collection for initial C proof system |  

## Compiling and Running

Clone this git repository `git clone http://github.com/snarkify/zprize-2023-prize1b`.
Ensure you have cmake installed, `sudo apt-get install cmake`, and version 12.5 of the CUDA development toolkit.

```
export PATH=$PATH:/usr/local/cuda/bin
cd submission
cargo bench zprize_bench --features gpu
```

## GPU Micro-Benchmarks

We have also included a tool to run individual micro-benchmarks for important mathematical operations.

```
cd GPU-Micro-Benchmarks
nvcc -arch=sm_80 MicroBenchmarks.cu -o mb
./mb
```

Micro-benchmark performance of Snarkify's GPU implementations.  All times in milliseconds.
Please note, these times are very fast, especially compared to other publicly available GPU
libraries.

| operation (BLS12381) | size 2^22 | size 2^25 |
|--|--|--|
|msm|28.40||
|ntt|1.13|13.35|
|intt|1.23||
|coset-ntt||14.45|
|polynomial eval|0.16|1.18|
|polynomial quotient|0.69||
|batch invert|0.77||

## Questions & Comments

For technical questions about this submission, please feel free to email us at:
 `nemmart at snarkify.io`, `tony at ingonyama.com`, and `fabrizio_m at hotmail.com`.

