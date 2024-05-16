Please read [this doc](./Documentation%20%20for%20Zprize3%20%20(1).pdf) for the solution of the prize3.

For the steps to run this project, please read our [writeup](writeup.md) :\)

[How to get the time taken for the proof generation of keccak and ecdsa ](./Time%20taken%20on%20each%20part.pdf)

# Zprize: High Throughput Signature Verification

How fast do you think you can verify our ECDSA signatures on Aleo?

This year's Zprize is winner-take-all. Join and compete for $500,000.

The competition will start on September 15th, 2023, and will end in February, 2024.

Read on the learn more about the _High Throughput Signature Verification_ prize.

## What is ECDSA?

[ECDSA](https://en.wikipedia.org/wiki/Elliptic_Curve_Digital_Signature_Algorithm) is the most widely used digital signature scheme as of today. It is based on the now depreciated [DSA](https://en.wikipedia.org/wiki/Digital_Signature_Algorithm) standard, which was already based on Schnorr signatures at the time (but different enough to avoid [patents](https://en.wikipedia.org/wiki/Digital_Signature_Algorithm#History)).

ECDSA is defined on elliptic curves, must often with the [P-256](https://credelius.com/credelius/?p=97) curve, but in the cryptocurrency world the [secp256k1](https://en.bitcoin.it/wiki/Secp256k1) curve is most often found instead.

A signature scheme is usually always accompanied with a hash function. ECDSA is commonly used with the SHA-256 hash function, but other hash functions can be used as well. For example, Ethereum uses the [keccak256](https://medium.com/0xcode/hashing-functions-in-solidity-using-keccak256-70779ea55bb0) hash function (which is essentially SHA-3 with custom parameters).

## Why ECDSA signature verification?

Verifying ECDSA signatures allows bridging on-chain applications with either off-chain applications or other web3 platforms that make use of the signature standard (for example, Ethereum, Celestia, Bitcoin, etc.)

In addition, many of the problems that have to be optimized in order to make ECDSA signature verification more efficient in SNARKs are found in other useful use cases as well (see next section).

## Why is it hard?

Verifying ECDSA signature(s) inside a SNARK is considered _problematic_, as it involves:

- **bitwise operations** (XOR, ROT, etc.) for the hash function which is not SNARK-friendly
- **non-native arithmetic** (NNA) for the elliptic curve operations that are not native to the field used by the circuit

On top of this, you need to implement elliptic curve operations (like scalar multiplication) and the ECDSA verifier algorithm itself. Due to this, every seemingly-benign operation will tend to blow up, causing the ECDSA verifier circuit to be very large.

## What is the prize about?

Different projects and proof systems use different techniques, and techniques used in one project don't often easily translate to other projects as they involve a lot of ad-hoc manual work.

For this reason, it is a good idea to narrow the context of this competition to make it useful for a specific proof system and framework.

For this prize, we'll specifically target the Aleo proof system [Varuna](https://drive.google.com/file/d/1W9vsn5xT1vUmJbzO8VXoNS4W1wGWLDHN/view?usp=sharing) based on the [Marlin proof system](https://eprint.iacr.org/2019/1047) ([see here for a nice explainer](https://github.com/ingonyama-zk/papers/blob/main/Marlin_and_me.pdf)).

## What am I allowed to do?

The proof system of Aleo, Varuna, is based on the R1CS arithmetization. As such, your first leverage is to optimize the R1CS circuit itself. You can do this in two ways: by using the high-level DSL [Leo](https://developer.aleo.org/leo/), [Aleo instructions](https://developer.aleo.org/aleo), or by writing R1CS directly in Rust using Aleo's embedded DSL (more on that later). Writing R1CS directly is usually preferred when one is trying to reduce the number of constraints as much as they can. But low-level R1CS gadgets can be exposed to Leo through [Leo operators](https://developer.aleo.org/leo/operators/).

Your second line of attack is to blindly optimize the prover's implementation. We do not expect the most impressive changes to come from there, and as such we will not place a lot of emphasis on this part of the solution. Furthermore, we won't allow GPU and more generally approaches making use of hardware acceleration.

Finally, we're also doing something special: we'll allow protocol-level modifications on Varuna, as long as they're contained and realistic. That is, they should not dramatically increase the prover time and memory usage for current programs that people are writing. The proof sizes should not drastically increase as well. In general, we expect people to take advantage of lookup tables, which are specified in the latest [Varuna specification](https://drive.google.com/file/d/1W9vsn5xT1vUmJbzO8VXoNS4W1wGWLDHN/view?usp=sharing).

> Note: another way to tackle this problem could be to use a different proof system, as long as that other proof system is secure, and that a proof for that different proof system can be safely wrapped in a Varuna proof (that is, you can write a verifier circuit with the Varuna stack).

## How should I start?

First, you should take a look at the `zprize_2023` branch of the [SnarkVM](https://github.com/AleoHQ/snarkVM/tree/zprize_2023/) repository.

If you want to write in Aleo instructions, the `vm/` folder contains [various tests which show how to write and prove programs](https://github.com/AleoHQ/snarkVM/tree/zprize_2023/synthesizer/tests).

If you want to write R1CS or lookups directly in Aleo's embedded DSL, you can take a look at the tests in the [`circuit/`](https://github.com/AleoHQ/snarkVM/blob/zprize_2023/circuit) folder. The subfolders contain gadgets showing how to create R1CS or lookup constraints.

A naive solution should be the start of any solution :) you can look at [xJsnark](https://akosba.github.io/papers/xjsnark.pdf) and [Gnark's pure R1CS optimizations for non-native arithmetic](https://www.youtube.com/watch?v=05JemsgfEX4&list=PLj80z0cJm8QHm_9BdZ1BqcGbgE-BEn-3Y). These solutions could be implemented using Aleo's embedded DSL.

To get a sense of how costs scale asymptotically, participants can either run a benchmark or look at the [Varuna specification](https://drive.google.com/file/d/1W9vsn5xT1vUmJbzO8VXoNS4W1wGWLDHN/view?usp=sharing).

Don't hesitate to check prior work in lookups (e.g. the XOR table optimization in [this paper](https://github.com/ingonyama-zk/papers/blob/main/lookups.pdf)).

## Security first

Playing with constraint optimizations, or protocol-level changes can be dangerous. You should always make sure that your changes are sound and secure.

As such, security will be the first metric we'll use to judge results. As it will be hard for us to review different solutions that explore different (potentially dangerous) techniques, we will place a lot of emphasis in clarity, readability, and auditability of the code and the associated documentation.

In other words, if your solution looks like [this PR](https://github.com/0xPolygonMiden/miden-vm/pull/123), this is not going to work for us :)

## What are the other metrics for the prize?

The competition will start in a free-form format, but later in September we will release a set of benchmarks that will be used to judge the different solutions.

We're expecting these to be Rust benchmarks that you'll be able to run locally with your code to measure how long it takes to verify our random ECDSA signatures. Note that messages should be hashed in-circuit, not outside.

As such, you can expect benchmarks that will attempt to verify proofs with the following public inputs: ECDSA signatures, public keys, and messages (all unrelated). The ECDSA configuration will use secp256k1 and keccak256. 

Competitors will be able to split these public inputs over several proofs if needed (for example, if you need 4 proofs to verify our 3 signatures, you can do that).

## Licensing

All submission code must be open-sourced at the time of submission. Code and documentation must be dual-licensed under both the MIT and Apache-2.0 licenses

## Questions

If there are any questions about this prize, please contact [david@zksecurity.xyz](mailto:david@zksecurity.xyz&subject=zprize) or ask your question on [Discord](https://discord.com/channels/954403678079578173/1131190201205665862).
