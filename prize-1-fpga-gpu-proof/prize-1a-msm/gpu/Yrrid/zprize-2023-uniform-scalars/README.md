# Uniform Random Scalars

## Introduction

The following is an update to Yrrid Software's and [Snarkify](https://snarkify.io/) (joint team) submission to the 2023 
Z-Prize MSM on the GPU competition.

We start with our latter submission to the ZPrize (it was submitted a few days after the competition closed) which was
the fastest GPU MSM submission and here we update the code to support our uniformly distributed scalars idea.

## ZPrize vs Real World Proof Systems

The ZPrize competitions have been using uniformly random IID generated scalars.  This means that the count of points in each Pippenger buckets are 
fairly evenly distributed -- technically, the number of points in each bucket is sampled from something close to a normal distribution.   Now 
it turns out, with a real word proof system the scalars aren't uniformly distributed.  Small scalars, such as 1, 2, 4, 8, -1, -2, etc, occur 
far more frequently than others.  This means that some Pippenger buckets can have thousands of points, while most only have a few dozen.  

There are some known solutions for this problem, for example assigning multiple threads to larger buckets.  But these approaches are complex
to implement and if not implemented with great care, there is the potential for timing attacks that could compromise the proof system security.

We have discovered a solution, which we believe is simple, elegant, and efficient.

## Our Solution

We are given n fixed points, $P_1, P_2 ... P_n$ and $n$ variable scalars, $s_1, s_2, ... s_n$, and we need to compute:

   $$\mbox{MSM}(s, P) = \sum_{i\in 1 .. n}{s_i * P_i}$$

One time setup step: pregenerate $n$ IID uniform random scalars, $r_1, r_2, r_3, ... r_n$ and precompute a single 
point:  

   $$P_r = \mbox{MSM}(r, P) = \sum_{i\in 1 .. n}{r_i * P_i}$$

To compute an MSM(s, P), we now compute:

   $$\mbox{MSM}(s, P) = \mbox{MSM}(s+r, P) - P_r = \Big(\sum_{i\in 1 .. n}{(s_i + r_i) * P_i}\Big) - P_r$$
     
Note since $r_i$ was IID and uniformly distributed, $s_i + r_i$ will be IID and uniformly distributed, and therefore, we have solved the
bucket problem.

## Overhead Analysis

We have the one time costs of randomly generating scalars and a single $n$ point MSM.

Then on each run we have the overhead of $n$ scalar additions, and 1 EC point subtraction.  

## Run Time Results

We have implemented the uniform scalars idea in this repo, and comparing the MSM run-times using the randoms vs a control run where
they are not used, are almost identical.  The overhead is much less than the run to run variation.  But to put a number on it, the
random scalars overhead is less than 1% of the total MSM run time.

## Questions

For technical questions, please feel free to email me at `nemmart at yrrid.com`.

-----------------------------------------------------------------------------------------


# Z-Prize MSM on the GPU Submission

##Performance

Our run times are 357 ms (bls12377) and 436 ms (bls12381)

## Competition Objective

The objective for this competition is to compute a batch of 4 MSMs of size 2^24 on an RTX A5000 Ada Generation GPU card, and support 
two curves, BLS-12377-G1 and BLS-12381-G1.  The process is as follows.  First a set of 2^24 base points are constructed on the
CPU.  These are copied over to the GPU, and the GPU implementation has the opportunity to perform limited pre-computations.
Next 4 sets of 2^24 scalars are generated (on the CPU) and the timer is started.  The scalars must be copied over to the GPU
and the GPU implementation must compute 4 MSM values, one for each set against the fixed base points.  The timer is stopped 
once the GPU returns the 4 MSM results.

## Thanks

A heartfelt thank you to Robert Remen from Matter Labs and John Bartlett from Snarkify for their assistance in performancing
testing some of the components this submission.

## Building and running the submission

To run this code, install rust (for example `rustup install stable`).  Clone this repository, and in a shell, run the
following commands:
```
cd zprize-2023-submission-gpu
cargo bench --features gpu
```

## Approach

On the GPU, an MSM is typically done using Pippenger's algorithm in 3 steps.  First is a *planning* step, where the scalars
are sliced up and the data is sorted, such that there is a list of points to be added to each bucket.  Second is the
*accumulation* phase, where we add the points in each bucket's list to compute the bucket value.  The third step is the
*reduction* phase, where we add the buckets together to compute window sums using the sum of sums algorithm.

My original plan for this competition was to build a better sorting implementation, and whole suite of accumulation implementations
using different point representations and then submit the ones that turned out to be the fastest.  The plan was to support Twisted 
Edwards, batched affine, and xyzz on BLS12377, and batched affine and xyzz on BLS381.  Yrrid entered four ZPrize competitions this year,
and unfortunately due to time constraints, had to cut some ideas.  We ended up building the improved  sorter, Twisted Edwards, and XYZZ, 
but batched affine didn't make the cut.  Batched affine requires a low latency field inversion routine.  We have made some progress, 
with an implementation that runs 32 instances across a warp in about 40K cycles.  But it is still a work in progress.

## Point Representations and Run Times

| Curve | Point Representation / implementation | MSM Run Time (MS) |
|---|---|---|
| bls12377 | Extended Jacobian (XYZZ + XY) / ML | 426 | 
| bls12377 | Extended Jacobian (XYZZ + XY) / Yrrid| 420 | 
| bls12377 | Twisted Edwards (XYTZ + XYT) | 373 |
| bls12377 | Twisted Edwards (XYTZ + XY) | 356 | 
| bls12381 | Extended Jacobian (XYZZ + XY) / Yrrid| 441 | 
| bls12381 | Extended Jacobian (XYZZ + XY) / ML | 436 | 

Discussion.   We have a framework where it is fairly easy to add curve implementations, so along with the Yrrid
curves, we incorporated Matter Labs XYZZ implementations from last year's ZPrize.  These are tagged as ML in the 
table above.  For BLS12377, Yrrid's XYZZ seems marginally faster.  For BLS12381, Matter Labs' XYZZ is marginally faster.
For Twisted Edwards, we have two implementations.   XYTZ + XYT and XYTZ + XY.  The former requires loading of three 
field values (X, Y, and T) per 7-modmul EC add, whereas the latter loads two fields (X, Y) and uses a non-unified
8-modmul adder.  Ada generation GPU cards, have a lot of compute and a big L2 cache, but make compromises in the 
memory bandwidth as compared to previous generations.  As a result of the limited memory bandwidth, we see a good
jump in performance going from XYTZ + XYT to XYTZ + XY, even though the latter requires more modmul operations.

## Optimizations in our solution

In this section, we give a high level overview of the optimizations we have used to accelerate the computation:

-  Pippenger bucket algorithm with a 23-bit windows for BLS12377 and 22-bit windows for BLS12381.
-  Scalar preprocessing.  We process the scalars to ensure that buckets are as uniformly distributed as possible.
   This is described in detail in a section below.
-  We take advantage of the signed digits endormorphism to reduce the number of buckets required for the MSM computation.
   We do not currently use the cube root endomorphism.
-  Use very fast custom sorting algorithms to generate lists of points for each bucket.   
-  The buckets are then sorted, based on the number of points in each bucket.   The buckets with the most points are run 
   first.  This allows the GPU warps to run convergent workloads and minimizes the tail effect.
-  For each point Pi, we precompute 11 (BLS12377) or 12 (BLS12381) multiples of the point.  This allows us to
   use only a single window during accumulation and reduction.  This greatly reduces the number of buckets required and 
   speeds up the reduction phase.
-  Overlap the copying (host -> device) of the scalar data for the next MSM with computation of the current MSM.
-  Implemented new host code for the final summation in the reduction phase.  It's much faster and supports both curves.
-  The FF and EC routines have been carefully optimized:  
   - Based on Montgomery multiplication 
   - Use the even/odd alignment algorithms that maximize modmul perf on the GPU
   - Minimize correction steps in the FF operations
   - Try multiple point representations and pick the fastest
   - Use fast squaring
   - The Matter Labs XYZZ implementation takes advantage of the following optimization:  `REDC(a*b) + REDC(c*d) = REDC(a*b + c*d)`

## Scalar Preprocessing

Our goal with the preprocessing is to insure the distribution of points to buckets is as uniform as possible.  For BLS12377,
the scalar field size, _b_, is 253, and _c_, the number of bits in each window is 23.  For BLS1281, we have _b_ is 255 and
_c_ is 22.   Let N be the scalar field prime.  Let si and Pi be the ith scalar and point.   For BLS12377, we have the scenario 
where _c_ exactly divides _b_.  Here we check if the most significant bit of si is set, if so, we replace si and Pi with N-si and -Pi.
Then we proceed normally.  For BLS12381, we have ceil(b/c)=12 windows.  12*22 is 264.  Here we run into a problem -- our scalar is
just 255 bits in length, so approximately 99% of the buckets in the top window are empty, and the remaining 1% have thousands of points each.
One possible solution is to pick a uniform random whole number _u_ in the range of [0, 281], and then replace the scalar si with si+u*N.
Note, adding a multiple of N to a scalar does not change the MSM result.  In our solution, rather than generating random numbers, we 
simply choose `u=i % 282`.   282 happens to be the largest multiple of N less than 2^263.  The result is not a perfectly uniform distribution,
but in practice, it's close enough.

## Twisted Edwards and Supporting Files

In Hardcaml's 2022 ZPrize submission, they used a Twisted Edwards version of BLS12377G1 for their FPGA solution.  They noted there are
5 points that satisfy the curve equation that can not be converted from Short Weierstrass form to Twisted Edwards form.  But they left
open the question of whether any of the 5 points are in the set of BLS12377G1 point.  We have provided a definitive proof that the five
points are not part of the BLS12377G1 curve, and therefore, we can use the Twisted Edwards version of BLS12377G1 without having to worry 
about any special cases.  Please see the `yrrid-supporting-files/TwistedEdwards` directory.

Our codes also use a number of bounds assertions in order to minimize the correction steps.  The directory `yrrid-supporting-files/AssertionCheckers`
contains some Java code for checking the assertions.

## Questions

For technical questions about this submission, please contact `nemmart at yrrid.com`.
