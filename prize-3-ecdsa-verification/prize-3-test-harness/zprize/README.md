# Benchmarks

You can run the benchmark with the following command:

```
cargo bench
```

The rationale behind the benchmarks are the following: we're benching three levels of signature verifications:

1. signatures on messages of 100 bytes
2. signatures on messages of 1,000 bytes
3. signatures on messages of 50,000 bytes

For each level, we have 50 signatures to verify. The 50 signatures have 50 distinct public keys and messages.

Benching should include the computation of the assignment, as well as the computation of the proof. If you want to modify the benchmarks in a reasonable way (which preserve the intent) then do it. For example, verifying is currently done in the bench loop, but feel free to take it out as we  only care about measuring proving-time.

A few things to note about the current circuit:

1. It doesn't do anything at the moment. It's up to you to make it do something (like, verify ECDSA signatures).
2. It uses some naive and truly inefficient encoding, where inputs are encoded as bytestrings, and 1 byte = 1 field element.
3. It only attempts to verify one signature per proof. There's nothing that prevents you to try to produce less than 50 proofs to verify the 50 signatures.

In addition, expect that we will run the benchmarks on a consumer laptop (Macbook pro M2) and similar machines.
