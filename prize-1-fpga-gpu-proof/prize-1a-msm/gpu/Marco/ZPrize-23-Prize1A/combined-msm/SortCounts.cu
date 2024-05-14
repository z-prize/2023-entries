/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/
#include <cassert>

__global__ void histogramPrefixSumKernel(void* histogramPtr, void* unsortedTriplePtr, int WINDOWS_NUM, int BITS_PER_WINDOW) {
  uint32_t  globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=gridDim.x*blockDim.x;
  uint32_t* histogram=(uint32_t*)histogramPtr;
  uint32_t* counts=(uint32_t*)unsortedTriplePtr;
  uint32_t  count, localSum;
  uint32_t  i;
  
  __shared__ uint32_t sharedHistogram[1024];
  __shared__ uint32_t warpTotals[32];
  
  // must launch with 1024 threads
  
  sharedHistogram[threadIdx.x]=0;

  __syncthreads();
    
  #pragma unroll 1
  for(i=globalTID;i<NBUCKETS;i+=globalStride) {
    count=0;
    #pragma unroll
    for(int32_t j=0;j<=(WINDOWS_NUM-1)*NBUCKETS;j+=BUCKETS_NUM*NBUCKETS)
      count+=counts[j + i];

    count=umin(count, 1023);
    atomicAdd(&sharedHistogram[1023-count], 1);
  }
  
  #pragma unroll 1
  for(;i<BUCKETS_NUM*NBUCKETS;i+=globalStride) {
    count=0;
    #pragma unroll
    for(int32_t j=0;j<=8*NBUCKETS;j+=BUCKETS_NUM*NBUCKETS)
      count+=counts[j + i];
    count=umin(count, 1023);
    atomicAdd(&sharedHistogram[1023-count], 1);
  }
  
  __syncthreads();

  count=sharedHistogram[threadIdx.x];
  localSum=multiwarpPrefixSum(warpTotals, count, 32);
  atomicAdd(&histogram[threadIdx.x], localSum-count);
}

__launch_bounds__(1024)
__global__ void sortCountsKernel(void* sortedTriplePtr, void* histogramPtr, void* unsortedTriplePtr, int WINDOWS_NUM, int BITS_PER_WINDOW) {
  uint32_t  globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=gridDim.x*blockDim.x;
  uint32_t  warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F, warps=blockDim.x>>5;
  uint32_t* histogram=(uint32_t*)histogramPtr;
  uint32_t  counts_[3] = {0};
  assert(sizeof(counts_) >= WINDOWS_NUM);
  uint8_t *counts = (uint8_t*)counts_;
  uint32_t count12 = 0;
  uint32_t  indexes[12];
  assert(WINDOWS_NUM <= 12);

  uint32_t  count, bin, binCount, writeIndex, mask, thread, localWriteIndex, localBin, localBucket;
  bool      processed;

  // input pointers
  uint32_t* unsortedCounts=(uint32_t*)unsortedTriplePtr;
  uint32_t* unsortedIndexes=((uint32_t*)unsortedTriplePtr) + NBUCKETS*WINDOWS_NUM;

  // output pointers
  uint32_t* sortedBuckets=(uint32_t*)sortedTriplePtr;
  uint4*    sortedCountsAndIndexes=(uint4*)(sortedBuckets + NBUCKETS*2 + 32);
  //uint32_t* sortedCounts=((uint32_t*)sortedTriplePtr) + NBUCKETS*2 + 32;
  //uint32_t* sortedIndexes=((uint32_t*)sortedTriplePtr) + NBUCKETS*14 + 32*7;
  
  extern __shared__ uint32_t shmem[];

  constexpr int BATCH_SIZE = 5;
//  constexpr int
  uint32_t* binCounts=shmem;                         // 1*256 (words)
  uint32_t* buckets=shmem+256;                       // BATCH_SIZE*256 (words)
  uint4*    countsAndIndexes=(uint4*)(shmem+(BATCH_SIZE+1)*256);  // BATCH_SIZE*12*256

  if(globalTID<384) {
    // 32 empty entries
    if(globalTID<32)
      sortedBuckets[NBUCKETS*2 + globalTID]=NBUCKETS*2 + globalTID;

    // 32*12 words: counts and indexes
    sortedBuckets[NBUCKETS*26 + globalTID + 32]=0;
  }

  for(int32_t i=threadIdx.x;i<256;i+=blockDim.x) 
    binCounts[i]=0;
  
  for(int32_t i=threadIdx.x;i<BATCH_SIZE*256;i+=blockDim.x)
    buckets[i]=0xFFFFFFFF;

  __syncthreads();

  auto compress_counts_and_indexes = [&](void* _savePtr, uint32_t bucket) {
      uint32_t *savePtr = (uint32_t*)_savePtr;

      savePtr[0] = indexes[0] | ((((uint32_t)counts[0]>>0)&0xF) << 28);
      savePtr[1] = indexes[1] | ((((uint32_t)counts[0]>>4)&0xF) << 28);
      savePtr[2] = indexes[2] | ((((uint32_t)counts[1]>>0)&0xF) << 28);
      savePtr[3] = indexes[3] | ((((uint32_t)counts[1]>>4)&0xF) << 28);
      savePtr[4] = indexes[4] | ((((uint32_t)counts[2]>>0)&0xF) << 28);
      savePtr[5] = indexes[5] | ((((uint32_t)counts[2]>>4)&0xF) << 28);
      savePtr[6] = indexes[6] | ((((uint32_t)counts[3]>>0)&0xF) << 28);
      savePtr[7] = indexes[7] | ((((uint32_t)counts[3]>>4)&0xF) << 28);
      savePtr[8] = indexes[8] | ((((uint32_t)counts[4]>>0)&0xF) << 28);
      savePtr[9] = indexes[9] | ((((uint32_t)counts[4]>>4)&0xF) << 28);
      savePtr[10] = indexes[10]; 

      savePtr[11] = counts[5] | ((uint32_t)counts[6]<<8);
      savePtr[12] = counts[7] | ((uint32_t)counts[8]<<8) ;
      savePtr[13] = counts[9] | ((uint32_t)counts[10]<<8);
      if (WINDOWS_NUM == 12) {
          savePtr[14] = indexes[11];
          savePtr[15] = count12;
      }

    };

#pragma unroll 1
  for(uint32_t bucket=globalTID;bucket<BUCKETS_NUM*NBUCKETS;bucket+=globalStride) {
      // collect the data
    if(bucket<NBUCKETS) {
      count=0;
      #pragma unroll
      for(int32_t i=0; i < WINDOWS_NUM; i++) {
        uint32_t cnt = unsortedCounts[NBUCKETS*BUCKETS_NUM*i + bucket];
        if (i == 11) {
            count12 = cnt;
        } else {
            MY_ASSERT(cnt < 256);
            counts[i] = cnt;
        }
        indexes[i]=unsortedIndexes[NBUCKETS*BUCKETS_NUM*i + bucket];
        MY_ASSERT(indexes[i] < POINTS * WINDOWS_NUM);

        count+=cnt;
      }
    }
#if COUNT_AND_INDEX_NUM != 4
    else {
      count=0;
      #pragma unroll
      for(int32_t i=0;i<5;i++) {
        counts[i]=unsortedCounts[NBUCKETS*2*i + bucket];
        indexes[i]=unsortedIndexes[NBUCKETS*2*i + bucket];
        count+=counts[i];

          counts[6+i] = counts[i];
          if (count == UINT32_MAX)
              printf("count: %d\n", counts[6+i]);
      }
      counts[5]=0;
      indexes[5]=0;
    }
#endif

    processed=count>255;

    // if we have a lot of points in the coalesced bucket, do special one-off processing
    if(processed) {
      bin=umin(count, 1023);
      writeIndex=atomicAdd(&histogram[1023-bin], 1);
      MY_ASSERT(writeIndex < NBUCKETS*BUCKETS_NUM);
      sortedBuckets[writeIndex]=bucket;

#if COUNT_AND_INDEX_NUM != 4
#pragma unroll
      for(int i=0;i<COUNT_AND_INDEX_NUM;i++) {
          unsigned idx0 = min(i*2+0, WINDOWS_NUM);
          unsigned idx1 = min(i*2+1, WINDOWS_NUM);
          MY_ASSERT(indexes[idx0] < POINTS*WINDOWS_NUM);
          MY_ASSERT(indexes[idx1] < POINTS*WINDOWS_NUM);
          sortedCountsAndIndexes[writeIndex*COUNT_AND_INDEX_NUM + i]=make_uint4(counts[idx0], indexes[idx0], counts[idx1], indexes[idx1]);
      }
#else
        compress_counts_and_indexes(&sortedCountsAndIndexes[writeIndex*4], bucket);
#endif
    }
    
    // we don't have so many points in the coalesced bucket, use sh mem processing
    bin=count;
    binCount=0;
    while(!__all_sync(0xFFFFFFFF, processed)) {
      if(!processed) {
        binCount=atomicAdd(&binCounts[bin], 1);
        if(binCount<BATCH_SIZE) {

#if COUNT_AND_INDEX_NUM != 4
            #pragma unroll
            for(int i=0;i<COUNT_AND_INDEX_NUM;i++) {
                unsigned idx0 = min(i*2+0, WINDOWS_NUM);
                unsigned idx1 = min(i*2+1, WINDOWS_NUM);
                MY_ASSERT(indexes[idx0] < POINTS*WINDOWS_NUM);
                MY_ASSERT(indexes[idx1] < POINTS*WINDOWS_NUM);

                countsAndIndexes[bin*BATCH_SIZE*COUNT_AND_INDEX_NUM + binCount*COUNT_AND_INDEX_NUM + i]=make_uint4(counts[idx0], indexes[idx0], counts[idx1], indexes[idx1]);
            }
#else

            compress_counts_and_indexes(&countsAndIndexes[bin*BATCH_SIZE*4 + binCount*4 + 0], bucket);
            MY_ASSERT(bin*BATCH_SIZE*4 + binCount*4 + 3 < 96*1024/16);
#endif

            buckets[bin*BATCH_SIZE + binCount]=bucket;
            MY_ASSERT(bin*BATCH_SIZE + binCount < BATCH_SIZE*256);
            processed=true;
        }
      }

      if(binCount==BATCH_SIZE-1)
        writeIndex=atomicAdd(&histogram[1023-bin], BATCH_SIZE);
      while(true) {
        mask=__ballot_sync(0xFFFFFFFF, binCount==BATCH_SIZE-1);
        if(mask==0)
          break;
        thread=31-__clz(mask);
        localBin=__shfl_sync(0xFFFFFFFF, bin, thread);
        localWriteIndex=__shfl_sync(0xFFFFFFFF, writeIndex, thread);
        if(warpThread<BATCH_SIZE) {
          localBucket=atomicExch(&buckets[localBin*BATCH_SIZE + warpThread], 0xFFFFFFFF);
          while(localBucket==0xFFFFFFFF)
            localBucket=atomicExch(&buckets[localBin*BATCH_SIZE + warpThread], 0xFFFFFFFF);
          sortedBuckets[localWriteIndex + warpThread]=localBucket;
            MY_ASSERT(localBin*BATCH_SIZE + warpThread < BATCH_SIZE*256);
            MY_ASSERT(localWriteIndex + warpThread < NBUCKETS*BUCKETS_NUM);
            MY_ASSERT(localBucket < NBUCKETS*BUCKETS_NUM);

        }
        __syncwarp(0xFFFFFFFF);
        if(warpThread<COUNT_AND_INDEX_NUM*BATCH_SIZE) {
            sortedCountsAndIndexes[localWriteIndex*COUNT_AND_INDEX_NUM + warpThread]=countsAndIndexes[localBin*BATCH_SIZE*COUNT_AND_INDEX_NUM + warpThread];
            MY_ASSERT(localBin*BATCH_SIZE*COUNT_AND_INDEX_NUM + warpThread < 96*1024/16);

            MY_ASSERT(localWriteIndex*COUNT_AND_INDEX_NUM+COUNT_AND_INDEX_NUM*BATCH_SIZE-1 < NBUCKETS*BUCKETS_NUM*COUNT_AND_INDEX_NUM);
            MY_ASSERT(COUNT_AND_INDEX_NUM == COUNT_AND_INDEX_NUM || COUNT_AND_INDEX_NUM == 4);
            if (COUNT_AND_INDEX_NUM*BATCH_SIZE > 32 && warpThread < COUNT_AND_INDEX_NUM*BATCH_SIZE-32) {
                sortedCountsAndIndexes[localWriteIndex*COUNT_AND_INDEX_NUM + 32 + warpThread]=countsAndIndexes[localBin*BATCH_SIZE*COUNT_AND_INDEX_NUM + 32 + warpThread];
            }
        }
        __syncwarp(0xFFFFFFFF);
        binCounts[localBin]=0;
        binCount=(thread==warpThread) ? 0 : binCount;
      }
    }
  }
  
  __syncthreads();

  for(int32_t i=warp;i<256;i+=warps) {
    binCount=binCounts[i];
    if(binCount>0) {
      if(warpThread==0)
        writeIndex=atomicAdd(&histogram[1023-i], binCount);
      writeIndex=__shfl_sync(0xFFFFFFFF, writeIndex, 0);
      if(warpThread<binCount) {
          sortedBuckets[writeIndex + warpThread] = buckets[i * BATCH_SIZE + warpThread];
          MY_ASSERT(writeIndex + warpThread < NBUCKETS*BUCKETS_NUM);
          MY_ASSERT(i * BATCH_SIZE + warpThread < BATCH_SIZE*256);
      }
      if(warpThread<binCount*COUNT_AND_INDEX_NUM) {
          sortedCountsAndIndexes[writeIndex*COUNT_AND_INDEX_NUM + warpThread]=countsAndIndexes[i*BATCH_SIZE*COUNT_AND_INDEX_NUM + warpThread];
          MY_ASSERT(i*BATCH_SIZE*COUNT_AND_INDEX_NUM + warpThread < 96*1024/16);
      }
    }
  }
}
