/***

Copyright (c) 2022, Yrrid Software, Inc, and Matter Labs, LLC.
All rights reserved.

Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Robert Remen
            Michael Carilli

***/

__device__ __constant__ uint32_t divisorApprox[24]={
  0x00000000, 0xFFFFFFFF, 0x7FFFFFFF, 0x55555555, 0x3FFFFFFF, 0x33333333,
  0x2AAAAAAA, 0x24924924, 0x1FFFFFFF, 0x1C71C71C, 0x19999999, 0x1745D174, 
  0x15555555, 0x13B13B13, 0x12492492, 0x11111111, 0x0FFFFFFF, 0x0F0F0F0F, 
  0x0E38E38E, 0x0D79435E, 0x0CCCCCCC, 0x0C30C30C, 0x0BA2E8BA, 0x0B21642C,
};

// GROUP MUST BE 27 OR LESS!
#define GROUP 13

__device__ __forceinline__ uint32_t copyCountsAndIndexes(
        uint32_t countsAndIndexesOffset, uint4* sortedCountsAndIndexes, uint32_t bucket, uint32_t points,
        int WINDOWS_NUM,
        uint32_t* count12 = nullptr,
        uint32_t* index12 = nullptr
) {
  uint32_t count = 0;
  uint4    load;

  unsigned offset = 0;

#if COUNT_AND_INDEX_NUM != 4
    for (int i = 0; i < 3; ++i) {
      load=sortedCountsAndIndexes[i];
      load.y=load.y<<2;
      load.w=load.w<<2;

      if (i*2+0 >= WINDOWS_NUM) break;
      count += load.x;
      store_shared_u2(countsAndIndexesOffset + offset, make_uint2(load.x, load.y));
      offset += 256;
//      MY_ASSERT(load.y < points*WINDOWS_NUM*4);

      if (i*2+1 >= WINDOWS_NUM) break;
      count += load.z;
      store_shared_u2(countsAndIndexesOffset + offset, make_uint2(load.z, load.w));
      offset += 256;
//      MY_ASSERT(load.w < points*WINDOWS_NUM*4);
  }
#else
    uint32_t cur_count;
    uint32_t cur_index;

    uint32_t* cntAndInd = (uint32_t*)sortedCountsAndIndexes;
    unsigned mask = ~((1<<28) - 1);
//    uint32_t indexes[5];
    for (int i = 0; i < 5; ++i) {
        uint32_t a = cntAndInd[i*2+0];
        uint32_t b = cntAndInd[i*2+1];

        cur_count = ((a&mask)>>28) | ((b&mask)>>24);
        count += cur_count;
//        uint32_t cur_index = a & (~mask);
        cur_index = (cntAndInd[i] & (~mask)) * 4;
        store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count, cur_index));
	MY_ASSERT(count < 256);
	MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);
        offset += 256;
    }

    cur_count = cntAndInd[11]&0xFF;
    cur_index = (cntAndInd[5] & (~mask))*4; 
    store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count,  cur_index));
    offset += 256;
    count += cur_count;
    MY_ASSERT(count < 256);
    MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);

    cur_count = (cntAndInd[11]&0xFF00)>>8; 
    cur_index = (cntAndInd[6] & (~mask))*4; 
    store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count, cur_index));
    offset += 256;
    count += cur_count;
    MY_ASSERT(count < 256);
    MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);

    cur_count = cntAndInd[12]&0xFF; 
    cur_index = (cntAndInd[7] & (~mask))*4;
    store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count, cur_index));
    offset += 256;
    count += cur_count;
    MY_ASSERT(count < 256);
    MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);


    cur_count = (cntAndInd[12]&0xFF00)>>8;
    cur_index = (cntAndInd[8] & (~mask))*4; 
    store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count, cur_index));
    offset += 256;
    count += cur_count; 
    MY_ASSERT(count < 256);
    MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);

    cur_count = cntAndInd[13]&0xFF;
    cur_index = (cntAndInd[9] & (~mask))*4; 
    store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count, cur_index));
    offset += 256;
    count += cur_count;
    MY_ASSERT(count < 256);
    MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);

    cur_count = (cntAndInd[13]&0xFF00)>>8;
    cur_index = (cntAndInd[10] & (~mask))*4; 
    store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count, cur_index));
    offset += 256;
    count += cur_count;
    MY_ASSERT(count < 256);
    MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);

    if (WINDOWS_NUM == 12) {
        cur_count = (cntAndInd[15]);
	cur_index = (cntAndInd[14] & (~mask))*4; 
        store_shared_u2(countsAndIndexesOffset + offset, make_uint2(cur_count, cur_index));
        if (count12 != nullptr) *count12 = cur_count;
        if (index12 != nullptr) *index12 = cur_index;
        offset += 256;
        count += cur_count;
	MY_ASSERT(cur_index < points * WINDOWS_NUM * 4);
    }
#endif

  return count;
}

__device__ __forceinline__ void copyPointIndexes(
        uint32_t total_seq, uint32_t& sequence, uint32_t countsAndIndexesOffset,
        uint32_t pointIndexOffset, void* pointIndexes, uint32_t bucket, uint32_t points,
        int WINDOWS_NUM,
        uint2 *pCountAndIndex = nullptr
) {
  uint32_t remaining, shift, available; 
  uint2    countAndIndex;
  uint4    quad;

  remaining=GROUP;
  available=0;
  countAndIndex.x=0;
  while(remaining>0) {
    // when we enter here, available is zero
    if(countAndIndex.x==0 && sequence==total_seq)
      break;

    if(countAndIndex.x==0) {
      if (pCountAndIndex)
          countAndIndex = *pCountAndIndex;
      else
          countAndIndex=load_shared_u2(countsAndIndexesOffset + sequence);
      sequence+=256;
      shift=countAndIndex.y & 0x0F;
      countAndIndex.y=countAndIndex.y & 0xFFFFFFF0;

      MY_ASSERT(countAndIndex.y % 4 == 0);
      MY_ASSERT(countAndIndex.y < points * WINDOWS_NUM * 4);

      quad=*(uint4*)byteOffset(pointIndexes, countAndIndex.y);
      shift=shift>>2;
      available=umin(countAndIndex.x, 4-shift);
      countAndIndex.y+=(shift + available)<<2;
      quad.x=(shift>=2) ? quad.z : quad.x;
      quad.y=(shift>=2) ? quad.w : quad.y;
      shift=shift & 0x01;
    }
    else {
        MY_ASSERT(countAndIndex.y < points * WINDOWS_NUM * 4);
      quad=*(uint4*)byteOffset(pointIndexes, countAndIndex.y);
        available=umin(countAndIndex.x, 4);
      countAndIndex.y+=available<<2;
      shift=0;
    }      
    countAndIndex.x-=available;
      
    while(remaining>0 && available>0) {
      quad.x=(shift>0) ? quad.y : quad.x;
      quad.y=(shift>0) ? quad.z : quad.y;
      quad.z=(shift>0) ? quad.w : quad.z;
      store_shared_u32(pointIndexOffset, quad.x);
      pointIndexOffset+=4;
      MY_ASSERT(pointIndexOffset < 90*1024);
      available--;
      remaining--;
      shift=1;
    }
  }
  __syncwarp(0xFFFFFFFF);
  countAndIndex.x+=available;
  countAndIndex.y-=available<<2;
  if(countAndIndex.x>0) {
    sequence-=256;
    if (pCountAndIndex)
        *pCountAndIndex = countAndIndex;
    else
        store_shared_u2(countsAndIndexesOffset + sequence, countAndIndex);
  }
}

__device__ __forceinline__ void prefetch(uint32_t storeOffset, uint32_t pointIndex, void* pointsPtr, uint32_t points, int WINDOWS_NUM) {
  uint32_t loadIndex, loadIndex0, loadIndex1, oddEven=threadIdx.x & 0x01;
  void*    p0;
  void*    p1;
  
  #if defined(SMALL) 
    loadIndex=(pointIndex & 0xFFFF) | ((pointIndex & 0x7C000000) >> 10);
  #else
    loadIndex=pointIndex & 0x7FFFFFFF;
  #endif

    MY_ASSERT(loadIndex < points * WINDOWS_NUM * 4);

  loadIndex0=loadIndex;
  loadIndex1=__shfl_xor_sync(0xFFFFFFFF, loadIndex, 1);
  if(oddEven!=0) {
    storeOffset-=80;
    loadIndex0=loadIndex1;
    loadIndex1=loadIndex;
  }
  
  p0=byteOffset(pointsPtr, loadIndex0, 96);
  p1=byteOffset(pointsPtr, loadIndex1, 96);
  
  shared_async_copy_u4(storeOffset+0, byteOffset(p0, 0 + oddEven*16));
  shared_async_copy_u4(storeOffset+32, byteOffset(p0, 32 + oddEven*16));
  shared_async_copy_u4(storeOffset+64, byteOffset(p0, 64 + oddEven*16));
  shared_async_copy_u4(storeOffset+96, byteOffset(p1, 0 + oddEven*16));
  shared_async_copy_u4(storeOffset+128, byteOffset(p1, 32 + oddEven*16));
  shared_async_copy_u4(storeOffset+160, byteOffset(p1, 64 + oddEven*16));
  shared_async_copy_commit();
}

__device__ __forceinline__ void prefetchWait() {
  shared_async_copy_wait();
}

template <bool ZERO_BUCKETS>
__launch_bounds__(384)
__global__ void computeBucketSums(
        uint32_t batch, uint32_t points, void *histogramPtr, void* bucketsPtr, void* pointsPtr,
        void* sortedTriplePtr, void* pointIndexesPtr, void* atomicsPtr, int WINDOWS_NUM, int BITS_PER_WINDOW,
        bool is381
        ) {
  typedef BLS12377::G1Montgomery           Field;
  typedef PointXY<Field>                   PointXY;
  typedef PointXYZZ<Field>                 PointXYZZ;

//  constexpr msm::field fd = msm::field();
  uint32_t* histogram=(uint32_t*)histogramPtr;

  uint32_t         globalTID=blockIdx.x*blockDim.x+threadIdx.x;
  uint32_t         warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t*        atomics=((uint32_t*)atomicsPtr);
  uint32_t*        sortedBuckets=(uint32_t*)sortedTriplePtr;
  uint4*           sortedCountsAndIndexes=(uint4*)(sortedBuckets + NBUCKETS*2 + 32);
  uint4*           pointIndexes=(uint4*)pointIndexesPtr;
  uint32_t         overflowCount = atomics[2];
  uint32_t         globalWarp = blockIdx.x*blockDim.x/32 + warp;

  MY_ASSERT(blockDim.x % 32 == 0);
  uint32_t         next, current, bucket, count, sequence, pointIndex;
  uint32_t         countsAndIndexesOffset, pointIndexesOffset, pointsOffset;
  
  // shared memory offsets:
  //       0 -  1535     1536         constants required by field routines (has been remove)
  //    1536 - 19967    18432         counts and indexes for the 384 threads (48 bytes per thread)
  //   19968 - end     GROUP*4*384    point indexes
  //    xxxx - yyyyy    36864         point data
  
  
//  copyToShared((uint4*)SHMData);

    int count_and_index_perwarp = WINDOWS_NUM * 8 * 32;
    int warps = blockDim.x/32;
  countsAndIndexesOffset = warp *count_and_index_perwarp + warpThread*8;
  pointIndexesOffset     = warps*count_and_index_perwarp + threadIdx.x*GROUP*4;
  pointsOffset           = warps*count_and_index_perwarp + blockDim.x *GROUP*4 + threadIdx.x*96;

  auto compute_overflow_bucket_sums = [&]( ) {
      bucket = sortedBuckets[current];
//    if (bucket >= BUCKETS_NUM*NBUCKETS) {
//        printf("next: %u, bucket: %u\n", next, bucket);
//    }
      MY_ASSERT(bucket < BUCKETS_NUM * NBUCKETS);

      uint32_t count12;
      uint32_t index12;
      uint32_t countIndex = countsAndIndexesOffset;
      if (warpThread == 0)
          count = copyCountsAndIndexes(countIndex, sortedCountsAndIndexes + current * COUNT_AND_INDEX_NUM, bucket,
                                       points, WINDOWS_NUM, &count12, &index12);

      count   = __shfl_sync(0xFFFFFFFF, count, 0);
      count12 = __shfl_sync(0xFFFFFFFF, count12, 0);
      index12 = __shfl_sync(0xFFFFFFFF, index12, 0);
      countIndex = __shfl_sync(0xFFFFFFFF, countIndex, 0);

      uint32_t count0 = count - count12;
      MY_ASSERT(count > 1023);
      MY_ASSERT(count0 < 256);

      msm::point_xyzz *bucket_ptr = (msm::point_xyzz *)bucketsPtr + bucket;
      msm::point_xyzz storeAcc = ZERO_BUCKETS
                            ? msm::point_at_infinity(is381)
                            : memory::load<msm::point_xyzz, memory::ld_modifier::cs>(bucket_ptr);

      msm::point_xyzz localAcc = msm::point_at_infinity(is381);

      msm::point_xyzz *acc;
      if (warpThread == 0)
          acc = &storeAcc;
      else
          acc = &localAcc;

      uint32_t endSeq = 11 * 8 * 32;
      sequence = 0;
      if (warpThread > 0) {
          sequence = 11 * 8 * 32;
          endSeq   = 12 * 8 * 32;
      }

      uint32_t cur_count = count0;
      uint2    countAndIndex = {0};
      if (warpThread > 0) {
//          countAndIndex=load_shared_u2(countIndex + sequence);
          countAndIndex.x = cur_count;
          countAndIndex.y = index12 + (warpThread - 1) * cur_count * 4;
          MY_ASSERT(cur_count*31 < count12);
      }

      while(__any_sync(0xFFFFFFFF, cur_count>0)) {
          copyPointIndexes(endSeq, sequence, countIndex, pointIndexesOffset, pointIndexes, bucket, points, WINDOWS_NUM, warpThread > 0?&countAndIndex: nullptr);
          __syncwarp(0xFFFFFFFF);

          MY_ASSERT(cur_count > 0);
          pointIndex=(cur_count==0) ? 0 : load_shared_u32(pointIndexesOffset);
          prefetch(pointsOffset, pointIndex, pointsPtr, points, WINDOWS_NUM);

#pragma unroll 1
          for(int i=1;__any_sync(0xFFFFFFFF, cur_count>0) && i<=GROUP;i++) {
              prefetchWait();
              __syncwarp(0xFFFFFFFF);
              __align__(16) PointXY point;
              point.loadShared(pointsOffset);
              msm::point_affine base = reinterpret_cast<msm::point_affine &>(point);

              if((pointIndex & 0x80000000)!=0)
                base = msm::point_neg(base, is381);

              __syncwarp(0xFFFFFFFF);

              pointIndex=(cur_count==0) ? 0 : load_shared_u32(pointIndexesOffset + i*4);
              if(i<GROUP)
                  prefetch(pointsOffset, pointIndex, pointsPtr, points, WINDOWS_NUM);
              __syncwarp(0xFFFFFFFF);

              MY_ASSERT(cur_count > 0);

              *acc = msm::curve_add(*acc, base, is381);
              cur_count--;
          }
      }

//      if (bucket == test_bucketnum) {
//          localAcc.to_affine().print("local affine0: ");
//      }

      count12 -= count0*31;
      uint32_t offset = count0*31 * 4;

//      countAndIndex=load_shared_u2(countIndex + sequence);
      countAndIndex.x = 0;
      countAndIndex.y = index12 + offset;

      acc = &localAcc;
      for (uint32_t c = warpThread * GROUP*2; c < (count12+GROUP*2*32)/(GROUP*2*32)*(GROUP*2*32); c += GROUP*2 * 32) {
          uint2    countAndIndex2 = countAndIndex;
          if (c < count12)
            countAndIndex2.x = min(GROUP*2, count12-(c));
          countAndIndex2.y = countAndIndex.y +  c * 4;
          cur_count = countAndIndex2.x;

          sequence = 11 * 8 * 32;
          endSeq   = 12 * 8 * 32;

          while(__any_sync(0xFFFFFFFF, cur_count>0)) {
              copyPointIndexes(endSeq, sequence, countIndex, pointIndexesOffset, pointIndexes, bucket, points, WINDOWS_NUM, &countAndIndex2);
              __syncwarp(0xFFFFFFFF);
              pointIndex=(cur_count==0) ? 0 : load_shared_u32(pointIndexesOffset);
              prefetch(pointsOffset, pointIndex, pointsPtr, points, WINDOWS_NUM);

#pragma unroll 1
              for(int i=1;__any_sync(0xFFFFFFFF, cur_count>0) && i<=GROUP;i++) {
                  prefetchWait();
                  __syncwarp(0xFFFFFFFF);
                  __align__(16) PointXY point;
                  point.loadShared(pointsOffset);
                  msm::point_affine base = reinterpret_cast<msm::point_affine &>(point);

                  if((pointIndex & 0x80000000)!=0)
                    base = msm::point_neg(base, is381);

                  __syncwarp(0xFFFFFFFF);
                  pointIndex=(cur_count==0) ? 0 : load_shared_u32(pointIndexesOffset + i*4);
                  if(i<GROUP)
                      prefetch(pointsOffset, pointIndex, pointsPtr, points, WINDOWS_NUM);
                  __syncwarp(0xFFFFFFFF);
                  if (cur_count == 0)
                      continue;

                  *acc = msm::curve_add(*acc, base, is381);
                  cur_count--;
              }
          }
      }

      for (int sumCount = 16; sumCount > 0; sumCount /= 2) {
          PointXYZZ point = *(PointXYZZ*)acc;
          point.warpShuffle(warpThread<sumCount? warpThread + sumCount : warpThread);
          msm::point_xyzz otherPoint = *(msm::point_xyzz *)&point;

          if (warpThread < sumCount)
              *acc = msm::curve_add(*acc, otherPoint, is381);
      }

      if (warpThread == 0) {

          storeAcc = msm::curve_add(storeAcc, *acc, is381);
          memory::store<msm::point_xyzz, memory::st_modifier::cs>(bucket_ptr, storeAcc);
      }
  };

    if (globalTID == 0)
        // printf("overcount: %u\n", overflowCount);
//    MY_ASSERT(overflowCount == 32);
    if (globalWarp < overflowCount) {
//        if(warpThread==0)
//            current=atomicAdd(atomics+8, 1);
//
//        current=__shfl_sync(0xFFFFFFFF, current, 0);
        current = globalWarp;
        if (current >= overflowCount)
            printf("current: %u, over: %u, warp: %u\n", current, overflowCount, globalWarp);
        MY_ASSERT(current < overflowCount);
        compute_overflow_bucket_sums();
    }

  while(true) {
    if(warpThread==0)
      next=atomicAdd(atomics+2, 32);

    next=__shfl_sync(0xFFFFFFFF, next, 0);

    if(next>= NBUCKETS * BUCKETS_NUM) {
      int32_t warps=gridDim.x*blockDim.x>>5;
      
      if(next>= NBUCKETS * BUCKETS_NUM + (warps - 1) * 32)
        atomics[0]=0;  
      break;
    }
    next=next + warpThread;
    bool noBucket = false;
    if (next < BUCKETS_NUM*NBUCKETS) {
        bucket=sortedBuckets[next];
//        if (bucket >= BUCKETS_NUM*NBUCKETS) {
//            printf("next: %u, bucket: %u\n", next, bucket);
//        }
//        MY_ASSERT(next < NBUCKETS*BUCKETS_NUM);
        MY_ASSERT(bucket < BUCKETS_NUM*NBUCKETS);
        count=copyCountsAndIndexes(countsAndIndexesOffset, sortedCountsAndIndexes + next*COUNT_AND_INDEX_NUM, bucket, points, WINDOWS_NUM);
    } else {
        noBucket = true;
        count = 0;
    }

    msm::point_xyzz *bucket_ptr = (msm::point_xyzz *)bucketsPtr + bucket;
    msm::point_xyzz acc = ZERO_BUCKETS
                              ? acc = msm::point_at_infinity(is381)
                              : acc = memory::load<msm::point_xyzz, memory::ld_modifier::cs>(bucket_ptr);

//    printf("count: %d\n", count);
    sequence=0;
    uint2    countAndIndex = {0};

    while(__any_sync(0xFFFFFFFF, count>0)) {
      copyPointIndexes(count_and_index_perwarp, sequence, countsAndIndexesOffset, pointIndexesOffset, pointIndexes, bucket, points, WINDOWS_NUM, noBucket?&countAndIndex: nullptr);
      __syncwarp(0xFFFFFFFF);

      pointIndex=(count==0) ? 0 : load_shared_u32(pointIndexesOffset);
      prefetch(pointsOffset, pointIndex, pointsPtr, points, WINDOWS_NUM);

#pragma unroll 1
      for(int i=1;__any_sync(0xFFFFFFFF, count>0) && i<=GROUP;i++) {
        prefetchWait();
        __syncwarp(0xFFFFFFFF);
        __align__(16) PointXY point;
        point.loadShared(pointsOffset);
        msm::point_affine base = reinterpret_cast<msm::point_affine &>(point);

        if((pointIndex & 0x80000000)!=0)
          base = msm::point_neg(base, is381);

        __syncwarp(0xFFFFFFFF);

          pointIndex=(count==0) ? 0 : load_shared_u32(pointIndexesOffset + i*4);
        if(i<GROUP)
          prefetch(pointsOffset, pointIndex, pointsPtr, points, WINDOWS_NUM);
        __syncwarp(0xFFFFFFFF);
        if (count == 0)
          continue;
        acc = msm::curve_add(acc, base, is381);
        count--;
      }
    }
    if (!noBucket)
        memory::store<msm::point_xyzz, memory::st_modifier::cs>(bucket_ptr, acc);
  }
}
