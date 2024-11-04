/***

Copyright (c) 2024, Snarkify, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__global__ void array_zero_kernel(bls12381_fr* to, uint32_t count) {
  int32_t      idx1=blockIdx.x*blockDim.x + threadIdx.x, idx2=idx1 + blockDim.x*gridDim.x;
  uint4*       to_ptr=(uint4*)to;

  // copy 2 uint4 values per bls12381 value

  if(idx1<count*2)
    to_ptr[idx1]=make_uint4(0, 0, 0, 0);
  if(idx2<count*2)
    to_ptr[idx2]=make_uint4(0, 0, 0, 0);
}

__global__ void array_set_l1_kernel(bls12381_fr* to, uint32_t count) {
  int32_t      idx1=blockIdx.x*blockDim.x + threadIdx.x, idx2=idx1 + blockDim.x*gridDim.x;
  uint4*       to_ptr=(uint4*)to;
  uint4        value;

  // to_montgomery(2^-22 mod P)

  if((threadIdx.x & 0x01)==0)
    value=make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000000);
  else
    value=make_uint4(0x00000000, 0x00000000, 0x00000000, 0x00000400);

  if(idx1<count*2)
    to_ptr[idx1]=value;
  if(idx2<count*2)
    to_ptr[idx2]=value;
}

__global__ void array_copy_kernel(bls12381_fr* to, bls12381_fr* from, uint32_t count) {
  int32_t idx1=blockIdx.x*blockDim.x + threadIdx.x, idx2=idx1 + blockDim.x*gridDim.x;
  uint4*  to_ptr=(uint4*)to;
  uint4*  from_ptr=(uint4*)from;

  // copy 2 uint4 values per bls12381 value

  if(idx1<count*2)
    to_ptr[idx1]=from_ptr[idx1];
  if(idx2<count*2)
    to_ptr[idx2]=from_ptr[idx2];
}

__global__ void array_power_kernel(bls12381_fr* to, bls12381_fr* from, bls12381_fr* pow_table, uint32_t count) {
  int32_t      globalTID=blockIdx.x*blockDim.x + threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr  current, advance;

  current=power(pow_table, globalTID, count);
  advance=power(pow_table, globalStride, count);

  for(int32_t idx=globalTID;idx<count;idx+=globalStride) {
    to[idx]=from[idx] * current;
    current=current * advance;
  }
}

__global__ void array_batch_invert_kernel(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, uint32_t count) {
  int32_t      globalTID=blockIdx.x*blockDim.x + threadIdx.x, globalStride=blockDim.x*gridDim.x;
  int32_t      idx;
  bls12381_fr  products=bls12381_fr::r(), inverse, load;

  idx=globalTID;
  while(idx<count) {
    temp[idx]=products;
    products=products*from[idx];
    idx+=globalStride;
  }
  products=products.inv();
  while(true) {
    idx-=globalStride;
    load=from[idx];                  // handle the case where to==from
    to[idx]=products*temp[idx];
    products=products*load;
    if(idx<globalStride)
      break;
  }
}

__global__ void polynomial_evaluate_kernel(bls12381_fr* to, bls12381_fr* from, bls12381_fr* pow_table, uint32_t count) {
  uint32_t               globalTID=blockIdx.x*blockDim.x + threadIdx.x, globalStride=blockDim.x*gridDim.x;
  bls12381_fr            sum, current, advance;
  __shared__ bls12381_fr shared_exchange[1024];

  current=power(pow_table, globalTID, count);
  advance=power(pow_table, globalStride, count);

  sum=bls12381_fr::zero();
  for(uint32_t i=globalTID;i<count;i+=globalStride) {
    sum=sum + from[i]*current;
    current=current*advance;
  }
  sum=block_reduce<reduce_add>(sum, shared_exchange);
  if(threadIdx.x==0)
    to[blockIdx.x]=sum;
}

__global__ void polynomial_divide_reduce_kernel(bls12381_fr* temp, bls12381_fr* from, bls12381_fr* pow_table, uint32_t count) {
  uint32_t               globalTID=blockIdx.x*blockDim.x + threadIdx.x;
  bls12381_fr            current;
  __shared__ bls12381_fr shared_exchange[1024];

  current=bls12381_fr::zero();
  if(globalTID<count)
    current=from[globalTID]*power(pow_table, globalTID, count);
  current=block_reduce<reduce_add>(current, shared_exchange);

  if(threadIdx.x==0)
    temp[blockIdx.x]=current;
}

__global__ void polynomial_divide_kernel(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, bls12381_fr* pow_table, bls12381_fr* ipow_table, uint32_t count) {
  uint32_t               globalTID=blockIdx.x*blockDim.x + threadIdx.x;
  bls12381_fr            suffix, current;
  __shared__ bls12381_fr shared_exchange[1024];

  current=bls12381_fr::zero();
  if(globalTID<count)
    current=from[globalTID]*power(pow_table, globalTID, count);

  suffix=block_scan<suffix_scan_add>(current, shared_exchange);
  if(blockIdx.x!=gridDim.x-1)
    suffix=suffix + temp[blockIdx.x+1];

  if(globalTID<count)
    to[globalTID]=(suffix-current)*power(ipow_table, globalTID+1, count);  
}

__global__ void polynomial_divide_from_montgomery_kernel(bls12381_fr* to, bls12381_fr* from, bls12381_fr* temp, bls12381_fr* pow_table, bls12381_fr* ipow_table, uint32_t count) {
  uint32_t               globalTID=blockIdx.x*blockDim.x + threadIdx.x;
  bls12381_fr            suffix, current;
  __shared__ bls12381_fr shared_exchange[1024];

  current=bls12381_fr::zero();
  if(globalTID<count)
    current=from[globalTID]*power(pow_table, globalTID, count);

  suffix=block_scan<suffix_scan_add>(current, shared_exchange);
  if(blockIdx.x!=gridDim.x-1)
    suffix=suffix + temp[blockIdx.x+1];

  if(globalTID<count)
    to[globalTID]=bls12381_fr::from_montgomery((suffix-current)*power(ipow_table, globalTID+1, count));  
}

