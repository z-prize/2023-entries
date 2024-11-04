/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__global__ void fp_array_to_montgomery_kernel(void* to, void* from, uint32_t count) {
  uint32_t          tid=blockDim.x*blockIdx.x+threadIdx.x;
  BLS12381G1::Field field;
 
  if(tid>=count)
    return;

  BLS12381G1::load(field, fieldOffset(from, tid));
  BLS12381G1::toMontgomery(field, field);
  BLS12381G1::reduceFully(field, field);
  BLS12381G1::store(fieldOffset(to, tid), field);
}

__global__ void fp_array_from_montgomery_kernel(void* to, void* from, uint32_t count) {
  uint32_t          tid=blockDim.x*blockIdx.x+threadIdx.x;
  BLS12381G1::Field field;

  if(tid>=count)
    return;

  BLS12381G1::load(field, fieldOffset(from, tid));
  BLS12381G1::fromMontgomery(field, field);
  BLS12381G1::reduceFully(field, field);
  BLS12381G1::store(fieldOffset(to, tid), field);
}

__global__ void array_to_montgomery_kernel(bls12381_fr* to, bls12381_fr* from, uint32_t count) {
  uint32_t tid=blockDim.x*blockIdx.x+threadIdx.x;

  if(tid>=count)
    return;

  to[tid]=bls12381_fr::to_montgomery(from[tid]);
}

__global__ void array_from_montgomery_kernel(bls12381_fr* to, bls12381_fr* from, uint32_t count) {
  uint32_t tid=blockDim.x*blockIdx.x+threadIdx.x;

  if(tid>=count)
    return;

  to[tid]=bls12381_fr::from_montgomery(from[tid]);
}

