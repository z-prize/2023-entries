/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

typedef enum {
  reduce_add,
  reduce_multiply,
  prefix_scan_add,
  prefix_scan_multiply,
  suffix_scan_add,
  suffix_scan_multiply
} operation_t;

template<operation_t op>
__device__ __forceinline__ bls12381_fr identity() {
  if constexpr(op==reduce_add || op==prefix_scan_add || op==suffix_scan_add)
    return bls12381_fr::zero();
  else
    return bls12381_fr::r();
}

template<operation_t op>
__device__ bls12381_fr binary_op(bls12381_fr left, bls12381_fr right) {
  if constexpr(op==reduce_add || op==prefix_scan_add || op==suffix_scan_add)
    return left + right;
  else
    return left * right;
}

template<operation_t op>
__device__ bls12381_fr warp_reduce(bls12381_fr current, bls12381_fr* shared_exchange) {
  bls12381_fr temp;

  for(int32_t i=1;i<32;i=i+i) {
    shared_exchange[threadIdx.x]=current;
    temp=shared_exchange[threadIdx.x^i];
    current=binary_op<op>(current, temp);
  }
  return current;
}

template<operation_t op>
__device__ bls12381_fr block_reduce(bls12381_fr current, bls12381_fr* shared_exchange) {
  int32_t          warp_thread=threadIdx.x & 0x1F, warp=threadIdx.x>>5, warps=blockDim.x>>5;
  bls12381_fr temp;

  current=warp_reduce<op>(current, shared_exchange);

  __syncthreads();

  if(warp_thread==0)
    shared_exchange[warp]=current;

  __syncthreads();

  if(warp==0) {
    temp=identity<op>();
    if(threadIdx.x<warps)
      temp=shared_exchange[threadIdx.x];
    temp=warp_reduce<op>(temp, shared_exchange);
    if(threadIdx.x==0)
      shared_exchange[0]=temp;
  }

  __syncthreads();
 
  return shared_exchange[0];
}

template<operation_t op>
__device__ bls12381_fr warp_scan(const bls12381_fr& current, bls12381_fr* shared_exchange) {
  int32_t          warp_thread=threadIdx.x & 0x1F;
  bls12381_fr local, temp;

  local=current;
  temp=identity<op>();
  if constexpr(op==prefix_scan_add || op==prefix_scan_multiply) {
    for(int32_t i=16;i>0;i=i>>1) {
      shared_exchange[threadIdx.x]=local;
      if(warp_thread-i>=0)
        temp=shared_exchange[threadIdx.x - i];
      local=binary_op<op>(local, temp);
    }
  }
  else {
    for(int32_t i=16;i>0;i=i>>1) {
      shared_exchange[threadIdx.x]=local;
      if(warp_thread+i<32)
        temp=shared_exchange[threadIdx.x + i];
      local=binary_op<op>(local, temp);
    }
  }
  return local;
}

template<operation_t op>
__device__ bls12381_fr block_scan(const bls12381_fr& current, bls12381_fr* shared_exchange) {
  int32_t          warp_thread=threadIdx.x & 0x1F, warp=threadIdx.x>>5;
  bls12381_fr local, temp;

  local=warp_scan<op>(current, shared_exchange);

  __syncthreads();

  if constexpr(op==prefix_scan_add || op==prefix_scan_multiply) {
    if(warp_thread==31)
      shared_exchange[warp]=local;
  }
  else {
    if(warp_thread==0)
       shared_exchange[warp]=local;
  }

  __syncthreads();

  if(warp==0) {
    temp=shared_exchange[warp_thread];
    temp=warp_scan<op>(temp, shared_exchange);
    shared_exchange[warp_thread]=temp;
  }

  __syncthreads();

  temp=identity<op>();
  if constexpr(op==prefix_scan_add || op==prefix_scan_multiply) {
    if(warp>0)
      temp=shared_exchange[warp-1];
  }
  else {
    if(warp<31)
      temp=shared_exchange[warp+1];
  }

  __syncthreads();

  return binary_op<op>(local, temp);
}

template<operation_t op>
__global__ void reduce_1024_kernel(bls12381_fr* reduced, bls12381_fr* from, uint32_t count) {
  bls12381_fr current;

  __shared__ bls12381_fr shared_exchange[1024];

  // launch blocks with 1024 threads

  current=identity<op>();
  if(blockIdx.x*1024 + threadIdx.x<count)
    current=from[blockIdx.x*1024 + threadIdx.x];

  current=block_reduce<op>(current, shared_exchange);
  if(threadIdx.x==0)
    reduced[blockIdx.x]=current;
}

template<operation_t op>
__global__ void reduce_kernel(bls12381_fr* reduced, bls12381_fr* from, uint32_t count) {
  uint32_t                globalTID=blockIdx.x*blockDim.x + threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr             current;
  __shared__ bls12381_fr shared_exchange[1024];

  current=identity<op>();
  if(globalTID<count)
    current=from[globalTID];

  for(int idx=globalTID+globalStride;idx<count;idx+=globalStride) 
    current=binary_op<op>(current, from[idx]);

  current=block_reduce<op>(current, shared_exchange);
  if(threadIdx.x==0)
    reduced[blockIdx.x]=current;
}

template<operation_t op>
__global__ void scan_kernel(bls12381_fr* to, bls12381_fr* reduced, bls12381_fr* from, uint32_t count) {
  bls12381_fr            ps, current;
  uint32_t               reduced_count=(count+1023)>>10;
  uint32_t               globalTID=blockIdx.x*1024 + threadIdx.x;

  __shared__ bls12381_fr shared_exchange[1024];

  ps=identity<op>();
  if constexpr(op==prefix_scan_add || op==prefix_scan_multiply) {
    if(blockIdx.x>0)
      ps=reduced[blockIdx.x-1];
  }
  else {
    if(blockIdx.x+1<reduced_count)
      ps=reduced[blockIdx.x+1];
  }

  current=identity<op>();
  if(globalTID<count)
    current=from[globalTID];

  if(count<=32) 
    current=warp_scan<op>(current, shared_exchange);
  else
    current=block_scan<op>(current, shared_exchange);

  current=binary_op<op>(ps, current);
  if(globalTID<count)
    to[globalTID]=current;
}

