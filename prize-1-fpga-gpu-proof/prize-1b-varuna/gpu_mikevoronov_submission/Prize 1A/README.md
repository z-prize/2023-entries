# Prize 1a: Accelerating Multiscalar Multiplication

This spec is based on ZPrize'22 MSM track (https://assets-global.website-files.com/625a083eef681031e135cc99/6314b255fac8ee1e63c4bde3_gpu-fpga-msm.pdf). 

## Summary
Multiscalar multiplication (MSM) operations are an essential building block for ZK proof generation. This prize will focus on computing the fastest MSM using either GPUs or FPGAs.

## Optimization Objective
Compute the fixed-base point MSM for $2^{24}$ randomly sampled scalars from the BLS 12-377 and BLS12-381 scalar field with lowest latency.  More specifically, given a fixed set of elliptic curve points from the BLS12-377 (or BLS12-381) curve $(P_1, P_2,..., P_n)$ and a randomly sampled input vector of finite field elements from the scalar field $(k_1, k_2,..., k_n)$, calculate the elliptic curve point $Q = \sum_{i = 1}^n k_i * P_i$, where $n=2^{24}$.

### Constraints
- Sufficient documentation must be provided along with the implementation.
- Only a single GPU/FPGA may be used; the problem may not be parallelized across multiple hardware instances.
- Submissions may be written in any language. The provided test harness, however, will be in Rust. So competitors submitting solutions using other languages will be required to create their own Rust bindings.

The test harness repo: https://github.com/cysic-labs/ZPrize-23-Prize1, consists of test sample code.

### Timeline
- October 15 - Competition begins.
- December 1 - Mid-competition IPR.
- March 15 - Deadline for submission.
- End of March - Winners announced.

## Judging
Submissions will be checked for correctness and ranked by performance. In addition, documentation (in English) must be provided along with the implementation. The documentation can be written in-line or as a separate document. It should be thorough and explanatory enough to provide an understanding of the techniques used in the submitted implementation without requiring an associated verbal explanation.

### Correctness
The final correctness of the submission will be tested using randomly sampled test inputs/outputs that are not disclosed to the competitors during the competition. Submissions that fail any test cases will be judged as incorrect and lose the opportunity to win the prize.

### Performance
Given input vectors consisting of $2^{24}$ fixed elliptic curve points (bases), participants will compute a scalar multiplication of those bases with a set of four vectors of scalar elements from the associated BLS12-377 (or BLS12-381) G1 field in succession. The measurement of two curves will be in a sequential order. Competitors will be provided with test sample generator to use while building the solution (of course, competitors can, and are encouraged to, use other vectors during the design and build process). 

For scoring, solutions will be run using four randomly selected test vectors for BLS12-377 and then BLS12-381 curve as input across ten trials in total. The winning submission will be the one with the lowest average latency across all ten trials and the two curves. More specifically, let $\ell$ be the lowest latency of the 10 executions of MSM on BLS12-377 and $m$ be the counterpart on BLS12-377, then the score is calculated as $\frac{1}{2} (\ell + m)$. If the participant only submits only one solution, then a penalty factor (1.5) will be multiplied on the latency of the submitted curve. For instance, if a team submits solution only for BLS12-377 and the evaluated latency is $x$, then the score of this team is $1.5x$. Copy time for the bases to the device will not be counted (fixed base). The copy time starts when the set of random scalars are generated and finishes when the final results are transmitted to the host CPU.

## Hardware & Benchmarks
Competitors will be given access to one of the following:
- A dedicated instance of baseline image consisting of 8 cores of AMD EPYC 7742 and an RTX 5000 Ada generation GPU.
- A dedicated instance of baseline image consisting of 8 cores of AMD EPYC 7742 and a U250 FPGA card. 

## Prize Allocation
The total reward of this prize is 500k Aleo credits. The prize will be divided into FPGA and GPU tracks, where the prize for each track is proportional to the number of eligible submissions. The allocation principle is "winners take all". 
For instance, in a competition with X GPU teams and Y FPGA teams , the prize that the winner of GPU track can get is (X/(X + Y) * 500k).

Prizes will be awarded in good faith and at the sole discretion of the prize committee members.

## Notes
All submissions must be open-sourced at the time of submission. Code and documentation must be dual-licensed under both the MIT and Apache-2.0 licenses.

## Questions
If you have any questions, please contact Leo Fan at Cysic: leo@cysic.xyz or use Discord.