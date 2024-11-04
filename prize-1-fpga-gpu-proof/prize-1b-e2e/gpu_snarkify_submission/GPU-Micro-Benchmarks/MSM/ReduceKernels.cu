/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

template<class Curve, uint32_t bucketBits>
__launch_bounds__(256,1)
__global__ void reduceExtendedJacobianBuckets(void* reduceMemory, void* bucketMemory) {
  const uint32_t BUCKET_COUNT=1<<bucketBits;  

  uint32_t globalTID=blockIdx.x*blockDim.x+threadIdx.x, warpThread=threadIdx.x & 0x1F;
  uint32_t bucketsPerThread=(BUCKET_COUNT+gridDim.x*blockDim.x-1)/(gridDim.x*blockDim.x);
  uint32_t loadIndex, storeIndex, scale;
  bool     infinity, bit;

  typename Curve::PointXYZZ sum, sumOfSums, pt;

  Curve::initializeXYZZ(sum);
  Curve::initializeXYZZ(sumOfSums);
  Curve::initializeXYZZ(pt);

  Curve::initializeShared();

  for(int32_t i=bucketsPerThread-1;i>=0;i--) {
    loadIndex=globalTID*bucketsPerThread + i;
    if(loadIndex<BUCKET_COUNT) {
      Curve::load(pt.x, fieldOffset(bucketMemory, loadIndex*FIELDS_XYZZ + FIELD_X));
      Curve::load(pt.y, fieldOffset(bucketMemory, loadIndex*FIELDS_XYZZ + FIELD_Y));
      Curve::load(pt.zz, fieldOffset(bucketMemory, loadIndex*FIELDS_XYZZ + FIELD_ZZ));
      Curve::load(pt.zzz, fieldOffset(bucketMemory, loadIndex*FIELDS_XYZZ + FIELD_ZZZ));
      Curve::accumulateXYZZ(sum, pt);
    }
    Curve::accumulateXYZZ(sumOfSums, sum);
  }

  scale=globalTID * bucketsPerThread; 
  
  // compute sum=scale*sum
  Curve::setXYZZ(pt, sum);
  infinity=true;
  for(int32_t i=31;i>=0;i--) {
    if(!infinity) 
      Curve::doubleXYZZ(sum, sum);
    bit=(scale & (1<<i))!=0;
    if(infinity && bit) 
      infinity=false;
    else if(!infinity && bit)
      Curve::accumulateXYZZ(sum, pt);

    __syncwarp(0xFFFFFFFF);
  }

  if(infinity)
    Curve::setXYZZ(sum, sumOfSums);
  else
    Curve::accumulateXYZZ(sum, sumOfSums);

  // sum across the warp
  #pragma unroll 1
  for(int32_t i=1;i<32;i=i+i) {
    Curve::warpShuffleXYZZ(pt, sum, threadIdx.x ^ i);
    Curve::accumulateXYZZ(sum, pt);
  }

  // since all threads in the warp have the same sum, we can use a trick!
  storeIndex=globalTID>>5;
  if(warpThread==0) {
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYZZ + FIELD_X), sum.x);
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYZZ + FIELD_Y), sum.y);
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYZZ + FIELD_ZZ), sum.zz);
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYZZ + FIELD_ZZZ), sum.zzz);
  }
}

template<class Curve, uint32_t bucketBits>
__launch_bounds__(256,1)
__global__ void reduceTwistedEdwardsBuckets(void* reduceMemory, void* bucketMemory) {
  const uint32_t BUCKET_COUNT=1<<bucketBits;  

  uint32_t globalTID=blockIdx.x*blockDim.x+threadIdx.x, warpThread=threadIdx.x & 0x1F;
  uint32_t bucketsPerThread=(BUCKET_COUNT+gridDim.x*blockDim.x-1)/(gridDim.x*blockDim.x);
  uint32_t loadIndex, storeIndex, scale;
  bool     infinity, bit;

  typename Curve::PointXYTZ sum, sumOfSums, pt;

  Curve::initializeXYTZ(sum);
  Curve::initializeXYTZ(sumOfSums);
  Curve::initializeXYTZ(pt);

  for(int32_t i=bucketsPerThread-1;i>=0;i--) {
    loadIndex=globalTID*bucketsPerThread + i;
    if(loadIndex<BUCKET_COUNT) {
      Curve::load(pt.x, fieldOffset(bucketMemory, loadIndex*FIELDS_XYTZ + FIELD_X));
      Curve::load(pt.y, fieldOffset(bucketMemory, loadIndex*FIELDS_XYTZ + FIELD_Y));
      Curve::load(pt.t, fieldOffset(bucketMemory, loadIndex*FIELDS_XYTZ + FIELD_T));
      Curve::load(pt.z, fieldOffset(bucketMemory, loadIndex*FIELDS_XYTZ + FIELD_Z));
      Curve::accumulateXYTZ(sum, pt);
    }
    Curve::accumulateXYTZ(sumOfSums, sum);
  }

  scale=globalTID * bucketsPerThread; 
  
  // compute sum=scale*sum
  Curve::setXYTZ(pt, sum);
  infinity=true;
  for(int32_t i=31;i>=0;i--) {
    if(!infinity) 
      Curve::accumulateXYTZ(sum, sum);
    bit=(scale & (1<<i))!=0;
    if(infinity && bit) 
      infinity=false;
    else if(!infinity && bit)
      Curve::accumulateXYTZ(sum, pt);

    __syncwarp(0xFFFFFFFF);
  }

  if(infinity)
    Curve::setXYTZ(sum, sumOfSums);
  else
    Curve::accumulateXYTZ(sum, sumOfSums);

  // sum across the warp
  #pragma unroll 1
  for(int32_t i=1;i<32;i=i+i) {
    Curve::warpShuffleXYTZ(pt, sum, threadIdx.x ^ i);
    Curve::accumulateXYTZ(sum, pt);
  }

  // since all threads in the warp have the same sum, we can use a trick!
  storeIndex=globalTID>>5;
  if(warpThread==0) {
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYTZ + FIELD_X), sum.x);
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYTZ + FIELD_Y), sum.y);
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYTZ + FIELD_T), sum.t);
    Curve::store(fieldOffset(reduceMemory, storeIndex*FIELDS_XYTZ + FIELD_Z), sum.z);
  }
}

template<class Curve>
__launch_bounds__(256,1)
__global__ void sumExtendedJacobian(void* result, void* uniformResult, void* reduceMemory, uint32_t pointCount) {
  uint32_t                  warpThread=threadIdx.x & 0x1F, warp=threadIdx.x>>5;
  uint32_t                  sharedBase=1536;  // leave space for tables, total is 1536 + 192*8 = 3K
  typename Curve::PointXYZZ sum, pt;

  Curve::initializeShared();

  Curve::initializeXYZZ(sum);
  if(uniformResult!=NULL && threadIdx.x==0) {
    Curve::load(sum.x, fieldOffset(uniformResult, FIELD_X));
    Curve::load(sum.y, fieldOffset(uniformResult, FIELD_Y));
    Curve::load(sum.zz, fieldOffset(uniformResult, FIELD_ZZ));
    Curve::load(sum.zzz, fieldOffset(uniformResult, FIELD_ZZZ));
    Curve::negateXYZZ(sum);
  }

  __syncthreads();

  for(int32_t idx=threadIdx.x;idx<pointCount;idx+=blockDim.x) {
    Curve::load(pt.x, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_X));
    Curve::load(pt.y, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_Y));
    Curve::load(pt.zz, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_ZZ));
    Curve::load(pt.zzz, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_ZZZ));
    Curve::accumulateXYZZ(sum, pt);
  }

  __syncthreads();

  // sum across the warp
  #pragma unroll 1
  for(int32_t i=1;i<32;i=i+i) {
    Curve::warpShuffleXYZZ(pt, sum, threadIdx.x ^ i);
    Curve::accumulateXYZZ(sum, pt);
  }

  if(warpThread==0) {
    Curve::storeShared(sharedBase + warp*48 + 0, sum.x);
    Curve::storeShared(sharedBase + warp*48 + 384, sum.y);
    Curve::storeShared(sharedBase + warp*48 + 768, sum.zz);
    Curve::storeShared(sharedBase + warp*48 + 1152, sum.zzz);
  }

  __syncthreads();

  if(warp==0) {
    Curve::loadShared(sum.x, sharedBase + (warpThread & 0x07)*48 + 0);
    Curve::loadShared(sum.y, sharedBase + (warpThread & 0x07)*48 + 384);
    Curve::loadShared(sum.zz, sharedBase + (warpThread & 0x07)*48 + 768);
    Curve::loadShared(sum.zzz, sharedBase + (warpThread & 0x07)*48 + 1152);

    #pragma unroll 1
    for(int32_t i=1;i<8;i=i+i) {
      Curve::warpShuffleXYZZ(pt, sum, threadIdx.x ^ i);
      Curve::accumulateXYZZ(sum, pt);
    }
  }

  if(threadIdx.x==0) {
    Curve::store(fieldOffset(result, FIELD_X), sum.x);
    Curve::store(fieldOffset(result, FIELD_Y), sum.y);
    Curve::store(fieldOffset(result, FIELD_ZZ), sum.zz);
    Curve::store(fieldOffset(result, FIELD_ZZZ), sum.zzz);
  }    
}

template<class Curve>
__launch_bounds__(256,1)
__global__ void sumAndNormalizeExtendedJacobian(void* result, void* uniformResult, void* reduceMemory, uint32_t pointCount) {
  uint32_t                  warpThread=threadIdx.x & 0x1F, warp=threadIdx.x>>5;
  uint32_t                  sharedBase=1536;  // leave space for tables, total is 1536 + 192*8 = 3K
  typename Curve::PointXYZZ sum, pt;
  typename Curve::PointXY   res;

  // launch a CTA for each MSM in the batch

  Curve::initializeShared();

  Curve::initializeXYZZ(sum);
  if(uniformResult!=NULL && threadIdx.x==0) {
    Curve::load(sum.x, fieldOffset(uniformResult, FIELD_X));
    Curve::load(sum.y, fieldOffset(uniformResult, FIELD_Y));
    Curve::load(sum.zz, fieldOffset(uniformResult, FIELD_ZZ));
    Curve::load(sum.zzz, fieldOffset(uniformResult, FIELD_ZZZ));
    Curve::negateXYZZ(sum);
  }

  __syncthreads();

  reduceMemory=fieldOffset(reduceMemory, blockIdx.x*pointCount*FIELDS_XYZZ);
  result=fieldOffset(result, blockIdx.x*FIELDS_XY);

  for(int32_t idx=threadIdx.x;idx<pointCount;idx+=blockDim.x) {
    Curve::load(pt.x, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_X));
    Curve::load(pt.y, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_Y));
    Curve::load(pt.zz, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_ZZ));
    Curve::load(pt.zzz, fieldOffset(reduceMemory, idx*FIELDS_XYZZ + FIELD_ZZZ));
    Curve::accumulateXYZZ(sum, pt);
  }

  __syncthreads();

  // sum across the warp
  #pragma unroll 1
  for(int32_t i=1;i<32;i=i+i) {
    Curve::warpShuffleXYZZ(pt, sum, threadIdx.x ^ i);
    Curve::accumulateXYZZ(sum, pt);
  }

  if(warpThread==0) {
    Curve::storeShared(sharedBase + warp*48 + 0, sum.x);
    Curve::storeShared(sharedBase + warp*48 + 384, sum.y);
    Curve::storeShared(sharedBase + warp*48 + 768, sum.zz);
    Curve::storeShared(sharedBase + warp*48 + 1152, sum.zzz);
  }

  __syncthreads();

  if(warp==0) {
    Curve::loadShared(sum.x, sharedBase + (warpThread & 0x07)*48 + 0);
    Curve::loadShared(sum.y, sharedBase + (warpThread & 0x07)*48 + 384);
    Curve::loadShared(sum.zz, sharedBase + (warpThread & 0x07)*48 + 768);
    Curve::loadShared(sum.zzz, sharedBase + (warpThread & 0x07)*48 + 1152);

    #pragma unroll 1
    for(int32_t i=1;i<8;i=i+i) {
      Curve::warpShuffleXYZZ(pt, sum, threadIdx.x ^ i);
      Curve::accumulateXYZZ(sum, pt);
    }

    Curve::normalize(res, sum);

    Curve::fromMontgomery(res.x, res.x);
    Curve::fromMontgomery(res.y, res.y);

    Curve::reduceFully(res.x, res.x);
    Curve::reduceFully(res.y, res.y);

    if(threadIdx.x==0) {
      Curve::store(fieldOffset(result, FIELD_X), res.x);
      Curve::store(fieldOffset(result, FIELD_Y), res.y);
    }
  }
}

