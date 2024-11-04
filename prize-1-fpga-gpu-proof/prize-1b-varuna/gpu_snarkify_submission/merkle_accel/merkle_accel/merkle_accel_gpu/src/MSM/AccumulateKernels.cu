/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

__device__ __forceinline__ uint32_t div6(uint32_t x) {
  const uint32_t i=(0xFFFFFFFF/6)+1;

  // valid for small x
  return __umulhi(i, x);
}

__device__ __forceinline__ uint32_t div9(uint32_t x) {
  const uint32_t i=(0xFFFFFFFF/9)+1;

  // valid for small x
  return __umulhi(i, x);
}
   
__device__ __forceinline__ uint32_t div12(uint32_t x) {
  const uint32_t i=(0xFFFFFFFF/12)+1;

  // valid for small x
  return __umulhi(i, x);
}

template<class Curve, uint32_t launchBounds>
__launch_bounds__(128, launchBounds)
__global__ void accumulateTwistedEdwardsXYBuckets(void* bucketsMemory, PlanningLayout layout, void* pointsMemory, bool preloaded) {
  // shared memory layout, total size is 32416 bytes (approx 31.7K)
  const uint32_t fieldMultiples=0;
  const uint32_t countsAndOffsetsTable=fieldMultiples + 1536;
  const uint32_t bucketIndexesTable=countsAndOffsetsTable + 128*8;
  const uint32_t pointIndexesTable=bucketIndexesTable + 128*4;
  const uint32_t fieldStorage=pointIndexesTable + 133*8*4;

  uint32_t                  warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  typename Curve::PointXYTZ acc;
  typename Curve::PointXY   pt;
  uint32_t                  nextBucket, bucketIndex, bucketCount, bucketOffset, pointIndex, fromThread, offset, pointsLoaded;
  bool                      pointLoaded, first, negate=false;
  uint2                     countAndOffset;
  uint4*                    buckets=(uint4*)bucketsMemory;
  uint4*                    points=(uint4*)pointsMemory;

  Curve::initializeShared();

  nextBucket=blockIdx.x*128 + threadIdx.x;
  
  bucketIndex=layout.sortedBucketIndexes[nextBucket];
  bucketCount=layout.sortedBucketCounts[nextBucket];
  bucketOffset=layout.sortedBucketOffsets[nextBucket];

  store_shared_u32(bucketIndexesTable + threadIdx.x*4, bucketIndex);
  store_shared_u2(countsAndOffsetsTable + threadIdx.x*8, make_uint2(bucketCount, bucketOffset));

  if(preloaded) {
    Curve::load(acc.x, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_X));
    Curve::load(acc.y, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Y));
    Curve::load(acc.t, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_T));
    Curve::load(acc.z, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Z));
    first=false;
  }
  else {
    Curve::initializeXYTZ(acc);    
    first=true;
  }

  pointsLoaded=0;
  pointLoaded=false;
  while(__any_sync(0xFFFFFFFF, bucketCount>0)) {
    if(pointLoaded) {
      shared_async_copy_wait();
      __syncwarp(0xFFFFFFFF);
      Curve::loadShared(pt.x, fieldStorage + threadIdx.x*6*16 + 0);
      Curve::loadShared(pt.y, fieldStorage + threadIdx.x*6*16 + 48);
      if(negate)
        Curve::negateXYT(pt);
    }
    if(!pointLoaded || __any_sync(0xFFFFFFFF, bucketCount>1)) {
      if((pointsLoaded & 0x07)==0) {
        #pragma unroll 1
        for(uint32_t i=warpThread>>3;i<32;i+=4) {
          fromThread=warp*32 + i;
          offset=threadIdx.x & 0x07;
          countAndOffset=load_shared_u2(countsAndOffsetsTable + fromThread*8);
          pointIndex=0xFFFFFFFF;
          if(pointsLoaded + offset<countAndOffset.x) 
            pointIndex=layout.pointIndexes[countAndOffset.y + pointsLoaded + offset];
          store_shared_u32(pointIndexesTable + fromThread*4 + offset*133*4, pointIndex);
        }
      }
      negate=(load_shared_u32(pointIndexesTable + threadIdx.x*4 + (pointsLoaded & 0x07)*133*4) & 0x80000000)!=0;
      #pragma unroll 1
      for(int i=warpThread;i<32*6;i+=32) {
        fromThread=warp*32 + div6(i);
        offset=i - div6(i)*6;
        pointIndex=load_shared_u32(pointIndexesTable + fromThread*4 + (pointsLoaded & 0x07)*133*4);
        if(pointIndex!=0xFFFFFFFF) 
          shared_async_copy_u4(fieldStorage + fromThread*6*16 + offset*16, &points[(pointIndex & 0x7FFFFFFF)*6 + offset]);
        shared_async_copy_commit();
      }
      pointsLoaded++;
    }
    if(pointLoaded && bucketCount>0) {
      if(first) {
        Curve::setField(acc.x, pt.x);
        Curve::setField(acc.y, pt.y);
        Curve::modmul(acc.t, pt.x, pt.y);
        first=false;
      }
      else 
        Curve::accumulateXY(acc, pt);
      bucketCount--;
    }
    pointLoaded=true;
  }

  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_X), acc.x);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Y), acc.y);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_T), acc.t);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Z), acc.z);
}

template<class Curve, uint32_t launchBounds>
__launch_bounds__(128, launchBounds)
__global__ void accumulateTwistedEdwardsXYTBuckets(void* bucketsMemory, PlanningLayout layout, void* pointsMemory, bool preloaded) {
  // shared memory layout, total size is 32416 bytes (approx 31.7K)
  const uint32_t countsAndOffsetsTable=0;
  const uint32_t bucketIndexesTable=countsAndOffsetsTable + 128*8;
  const uint32_t pointIndexesTable=bucketIndexesTable + 128*4;
  const uint32_t fieldStorage=pointIndexesTable + 133*8*4;

  uint32_t                  warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  typename Curve::PointXYTZ acc;
  typename Curve::PointXYT  pt;
  uint32_t                  nextBucket, bucketIndex, bucketCount, bucketOffset, pointIndex, fromThread, offset, pointsLoaded;
  bool                      pointLoaded, negate=false, first;
  uint2                     countAndOffset;
  uint4*                    buckets=(uint4*)bucketsMemory;
  uint4*                    points=(uint4*)pointsMemory;

  nextBucket=blockIdx.x*128 + threadIdx.x;
  
  bucketIndex=layout.sortedBucketIndexes[nextBucket];
  bucketCount=layout.sortedBucketCounts[nextBucket];
  bucketOffset=layout.sortedBucketOffsets[nextBucket];

  store_shared_u32(bucketIndexesTable + threadIdx.x*4, bucketIndex);
  store_shared_u2(countsAndOffsetsTable + threadIdx.x*8, make_uint2(bucketCount, bucketOffset));

  if(preloaded) {
    Curve::load(acc.x, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_X));
    Curve::load(acc.y, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Y));
    Curve::load(acc.t, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_T));
    Curve::load(acc.z, fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Z));
    first=false;
  }
  else {
    Curve::initializeXYTZ(acc);
    first=true;
  }

  pointsLoaded=0;
  pointLoaded=false;
  while(__any_sync(0xFFFFFFFF, bucketCount>0)) {
    if(pointLoaded) {
      shared_async_copy_wait();
      __syncwarp(0xFFFFFFFF);
      Curve::loadShared(pt.x, fieldStorage + threadIdx.x*13*16 + 0);
      Curve::loadShared(pt.y, fieldStorage + threadIdx.x*13*16 + 48);
      Curve::loadShared(pt.t, fieldStorage + threadIdx.x*13*16 + 96);
      if(negate)
        Curve::negateXYT(pt);
    }
    if(!pointLoaded || __any_sync(0xFFFFFFFF, bucketCount>1)) {
      if((pointsLoaded & 0x07)==0) {
        #pragma unroll 1
        for(uint32_t i=warpThread>>3;i<32;i+=4) {
          fromThread=warp*32 + i;
          offset=threadIdx.x & 0x07;
          countAndOffset=load_shared_u2(countsAndOffsetsTable + fromThread*8);
          pointIndex=0xFFFFFFFF;
          if(pointsLoaded + offset<countAndOffset.x) 
            pointIndex=layout.pointIndexes[countAndOffset.y + pointsLoaded + offset];
          store_shared_u32(pointIndexesTable + fromThread*4 + offset*133*4, pointIndex);
        }
      }
      negate=(load_shared_u32(pointIndexesTable + threadIdx.x*4 + (pointsLoaded & 0x07)*133*4) & 0x80000000)!=0;
      #pragma unroll 1
      for(int i=warpThread;i<32*9;i+=32) {
        fromThread=warp*32 + div9(i);
        offset=i - div9(i)*9;
        pointIndex=load_shared_u32(pointIndexesTable + fromThread*4 + (pointsLoaded & 0x07)*133*4);
        if(pointIndex!=0xFFFFFFFF) 
          shared_async_copy_u4(fieldStorage + fromThread*13*16 + offset*16, &points[(pointIndex & 0x7FFFFFFF)*9 + offset]);
        shared_async_copy_commit();
      }
      pointsLoaded++;
    }
    if(pointLoaded && bucketCount>0) {
      if(first) {
        Curve::setField(acc.x, pt.x);
        Curve::setField(acc.y, pt.y);
        Curve::modmul(acc.t, pt.x, pt.y);
        first=false;
      }
      else
        Curve::accumulateXYT(acc, pt);
      bucketCount--;
    }
    pointLoaded=true;
  }

  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_X), acc.x);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Y), acc.y);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_T), acc.t);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYTZ + FIELD_Z), acc.z);
}

template<class Curve, uint32_t launchBounds>
__launch_bounds__(128, launchBounds)
__global__ void accumulateExtendedJacobianBuckets(void* bucketsMemory, PlanningLayout layout, void* pointsMemory, bool preloaded) {
  // shared memory layout, total size is 19616 bytes (approx 20)
  const uint32_t fieldMultiplesTable=0;                                    //  1536
  const uint32_t countsAndOffsetsTable=fieldMultiplesTable + 1536;         //  1024
  const uint32_t bucketIndexesTable=countsAndOffsetsTable + 128*8;         //   512
  const uint32_t pointIndexesTable=bucketIndexesTable + 128*4;             //  4256
  const uint32_t fieldStorage=pointIndexesTable + 133*8*4;                 // 12288

  uint32_t                  warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  typename Curve::PointXYZZ acc;
  typename Curve::PointXY   pt;
  uint32_t                  nextBucket, bucketIndex, bucketCount, bucketOffset, pointIndex, fromThread, offset, pointsLoaded;
  bool                      pointLoaded, negate=false, first;
  uint2                     countAndOffset;
  uint4*                    buckets=(uint4*)bucketsMemory;
  uint4*                    points=(uint4*)pointsMemory;

  Curve::initializeShared();

  nextBucket=blockIdx.x*128 + threadIdx.x;
  
  bucketIndex=layout.sortedBucketIndexes[nextBucket];
  bucketCount=layout.sortedBucketCounts[nextBucket];
  bucketOffset=layout.sortedBucketOffsets[nextBucket];

  store_shared_u32(bucketIndexesTable + threadIdx.x*4, bucketIndex);
  store_shared_u2(countsAndOffsetsTable + threadIdx.x*8, make_uint2(bucketCount, bucketOffset));

  if(preloaded) {
    Curve::load(acc.x, fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_X));
    Curve::load(acc.y, fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_Y));
    Curve::load(acc.zz, fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_ZZ));
    Curve::load(acc.zzz, fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_ZZZ));
    first=false;
  }
  else {
    Curve::initializeXYZZ(acc);
    first=true;
  }

  pointsLoaded=0;
  pointLoaded=false;
  while(__any_sync(0xFFFFFFFF, bucketCount>0)) {
    if(pointLoaded) {
      shared_async_copy_wait();
      __syncwarp(0xFFFFFFFF);
      Curve::loadShared(pt.x, fieldStorage + threadIdx.x*6*16 + 0);
      Curve::loadShared(pt.y, fieldStorage + threadIdx.x*6*16 + 48);
      if(negate)
        Curve::negateXY(pt);
    }
    if(!pointLoaded || __any_sync(0xFFFFFFFF, bucketCount>1)) {
      if((pointsLoaded & 0x07)==0) {
        #pragma unroll 1
        for(uint32_t i=warpThread>>3;i<32;i+=4) {
          fromThread=warp*32 + i;
          offset=threadIdx.x & 0x07;
          countAndOffset=load_shared_u2(countsAndOffsetsTable + fromThread*8);
          pointIndex=0xFFFFFFFF;
          if(pointsLoaded + offset<countAndOffset.x) 
            pointIndex=layout.pointIndexes[countAndOffset.y + pointsLoaded + offset];
          store_shared_u32(pointIndexesTable + fromThread*4 + offset*133*4, pointIndex);
        }
      }
      negate=(load_shared_u32(pointIndexesTable + threadIdx.x*4 + (pointsLoaded & 0x07)*133*4) & 0x80000000)!=0;
      #pragma unroll 1
      for(int i=warpThread;i<32*6;i+=32) {
        fromThread=warp*32 + div6(i);
        offset=i - div6(i)*6;
        pointIndex=load_shared_u32(pointIndexesTable + fromThread*4 + (pointsLoaded & 0x07)*133*4);
        if(pointIndex!=0xFFFFFFFF) 
          shared_async_copy_u4(fieldStorage + fromThread*6*16 + offset*16, &points[(pointIndex & 0x7FFFFFFF)*6 + offset]);
        shared_async_copy_commit();
      }
      pointsLoaded++;
    }
    if(pointLoaded && bucketCount>0) {
      if(first) {
        Curve::setField(acc.x, pt.x);
        Curve::setField(acc.y, pt.y);
        Curve::setR(acc.zz);
        Curve::setR(acc.zzz);
        first=false;
      }
      else 
        Curve::accumulateXY(acc, pt);
      bucketCount--;
    }
    pointLoaded=true;
  }

  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_X), acc.x);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_Y), acc.y);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_ZZ), acc.zz);
  Curve::store(fieldOffset(buckets, bucketIndex*FIELDS_XYZZ + FIELD_ZZZ), acc.zzz);
}

template<class Curve, uint32_t launchBounds>
__launch_bounds__(128, launchBounds)
__global__ void accumulateExtendedJacobianMLBuckets(void* bucketsMemory, PlanningLayout layout, void* pointsMemory, bool preloaded) {
  // shared memory layout, total size is 19616 bytes (approx 20)
  const uint32_t fieldMultiplesTable=0;                                    //  1536
  const uint32_t countsAndOffsetsTable=fieldMultiplesTable + 1536;         //  1024
  const uint32_t bucketIndexesTable=countsAndOffsetsTable + 128*8;         //   512
  const uint32_t pointIndexesTable=bucketIndexesTable + 128*4;             //  4256
  const uint32_t fieldStorage=pointIndexesTable + 133*8*4;                 // 12288

  typedef typename Curve::MLMSM msm;

  constexpr typename msm::field fd = msm::field();

  uint32_t                   warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  typename Curve::PointXY    point;
  typename msm::point_xyzz   acc;
  typename msm::point_affine mlPoint;
  uint32_t                   nextBucket, bucketIndex, bucketCount, bucketOffset, pointIndex, fromThread, offset, pointsLoaded;
  bool                       pointLoaded, negate=false, first;
  uint2                      countAndOffset;
  uint4*                     points=(uint4*)pointsMemory;

  // USING MATTER LABS CODE

  nextBucket=blockIdx.x*128 + threadIdx.x;
  
  bucketIndex=layout.sortedBucketIndexes[nextBucket];
  bucketCount=layout.sortedBucketCounts[nextBucket];
  bucketOffset=layout.sortedBucketOffsets[nextBucket];

  store_shared_u32(bucketIndexesTable + threadIdx.x*4, bucketIndex);
  store_shared_u2(countsAndOffsetsTable + threadIdx.x*8, make_uint2(bucketCount, bucketOffset));

  if(preloaded) {
    acc = memory::load<typename msm::point_xyzz, memory::ld_modifier::cs>((typename msm::point_xyzz *)bucketsMemory + bucketIndex);
    first=false;
  }
  else {
    acc = msm::point_xyzz::point_at_infinity(fd);
    first=true;
  }

  pointsLoaded=0;
  pointLoaded=false;
  while(__any_sync(0xFFFFFFFF, bucketCount>0)) {
    if(pointLoaded) {
      shared_async_copy_wait();
      __syncwarp(0xFFFFFFFF);
      Curve::loadShared(point.x, fieldStorage + threadIdx.x*6*16 + 0);
      Curve::loadShared(point.y, fieldStorage + threadIdx.x*6*16 + 48);
      mlPoint = reinterpret_cast<typename msm::point_affine &>(point);
      if(negate)
        mlPoint = msm::point_affine::neg(mlPoint, fd);
    }
    if(!pointLoaded || __any_sync(0xFFFFFFFF, bucketCount>1)) {
      if((pointsLoaded & 0x07)==0) {
        #pragma unroll 1
        for(uint32_t i=warpThread>>3;i<32;i+=4) {
          fromThread=warp*32 + i;
          offset=threadIdx.x & 0x07;
          countAndOffset=load_shared_u2(countsAndOffsetsTable + fromThread*8);
          pointIndex=0xFFFFFFFF;
          if(pointsLoaded + offset<countAndOffset.x) 
            pointIndex=layout.pointIndexes[countAndOffset.y + pointsLoaded + offset];
          store_shared_u32(pointIndexesTable + fromThread*4 + offset*133*4, pointIndex);
        }
      }
      negate=(load_shared_u32(pointIndexesTable + threadIdx.x*4 + (pointsLoaded & 0x07)*133*4) & 0x80000000)!=0;
      #pragma unroll 1
      for(int i=warpThread;i<32*6;i+=32) {
        fromThread=warp*32 + div6(i);
        offset=i - div6(i)*6;
        pointIndex=load_shared_u32(pointIndexesTable + fromThread*4 + (pointsLoaded & 0x07)*133*4);
        if(pointIndex!=0xFFFFFFFF) 
          shared_async_copy_u4(fieldStorage + fromThread*6*16 + offset*16, &points[(pointIndex & 0x7FFFFFFF)*6 + offset]);
        shared_async_copy_commit();
      }
      pointsLoaded++;
    }
    if(pointLoaded && bucketCount>0) {
      if(first) {
        acc = msm::point_affine::to_xyzz(mlPoint, fd);
        first=false;
      }
      else 
        acc = msm::curve::add(acc, mlPoint, fd);
      bucketCount--;
    }
    pointLoaded=true;
  }

  memory::store<typename msm::point_xyzz, memory::st_modifier::cs>((typename msm::point_xyzz *)bucketsMemory + bucketIndex, acc);
}
