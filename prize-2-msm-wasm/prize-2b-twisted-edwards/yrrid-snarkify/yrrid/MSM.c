/***

Copyright (c) 2023-2024, Yrrid Software, Inc. and Snarkify Inc.  All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

#include <stdint.h>
#include <stdbool.h>

#define MAX_N                    1048576     // 2^20
#define MAX_WORKERS              16
#define MAX_ACCUMULATE_BUCKETS   64*8192     // 16 windows of 32K   (16 bit window -> signed digits, means each window is 32K)
#define MAX_REDUCE_BUCKETS       1024
#define MAX_WINDOWS              18
#define SUM_BUCKETS_OFFSET       MAX_ACCUMULATE_BUCKETS
#define SOS_BUCKETS_OFFSET       (MAX_ACCUMULATE_BUCKETS+512)

#include "SIMD.h"
#include "Types.h"
#include "FieldPair.h"
#include "EC.h"

I128     sharedResult[4];
I128     sharedSourcePoints[MAX_N*4];
I128     sharedPoints[MAX_N*7];
I128     sharedScalars[MAX_N*2];
I128     sharedBuckets[(MAX_ACCUMULATE_BUCKETS + MAX_REDUCE_BUCKETS)*10];

// We are going to do an upfront sort of the scalars, to generate list of points to add to each bucket.
// This is done with two sorting steps.  First we sort all additions into bins, where each bin consists of
// 64 consecutive buckets.  We then sort the bins.

uint32_t sharedBucketAdditions[MAX_N*MAX_WINDOWS];                // Each 32-bit value represents a point that needs to be added to a bucket
uint32_t sharedBucketCounts[MAX_ACCUMULATE_BUCKETS];              // Number of points in each bucket
uint32_t sharedBucketBases[MAX_ACCUMULATE_BUCKETS];               // Start offset (in sharedBucketAdditions) for each bucket

uint32_t sharedBinAdditions[MAX_N*MAX_WINDOWS];                   // Here we track points being added to bins of 64 buckets and the bucket within the bin
uint32_t sharedBinCounts[8193];                                   // Number of points in each bin of 64 buckets.
uint32_t sharedBinBases[8193];                                    // Start offset (in sharedBinAdditions) for each bin of 64 buckets.

uint32_t sharedThreadStarts[MAX_WORKERS*1024];
uint32_t sharedThreadBinCounts[MAX_WORKERS*8193];
uint32_t sharedThreadBinBases[MAX_WORKERS*8193];

uint32_t __attribute__ ((aligned (16)))    sharedPointCount;
uint32_t __attribute__ ((aligned (16)))    sharedWorkerIndex;
uint32_t __attribute__ ((aligned (16)))    sharedScalarIndex;
uint32_t __attribute__ ((aligned (16)))    sharedBinIndex;
uint32_t __attribute__ ((aligned (16)))    sharedMontgomeryIndex;
uint32_t __attribute__ ((aligned (16)))    sharedAccumulationIndex;
uint32_t __attribute__ ((aligned (16)))    sharedReductionIndex;

int32_t   __attribute__ ((aligned (16)))   sharedNotification;
Barrier_t __attribute__ ((aligned (16)))   sharedBarriers[10];

#include "Atomic.c"
#include "Barrier.c"
#include "FieldInv.c"
#include "FieldPair.c"
#include "Split.c"

#include "EC.c"

#if defined(WASM)

void* memset(void* address, int byte, unsigned long length) {
  uint8_t* byteAddress=(uint8_t*)address;

  for(int i=0;i<length;i++)
    byteAddress[i]=byte;
  return address;
}

void* memcpy(void* dest, const void* src, unsigned long  count) {
  uint8_t* byteDest=(uint8_t*)dest;
  uint8_t* byteSrc=(uint8_t*)src;

  for(int i=0;i<count;i++)
    byteDest[i]=byteSrc[i];
  return byteDest;
}

uint32_t* resultAddress() {
  return (uint32_t*)&sharedResult[0];
}

uint32_t* pointsAddress() {
  return (uint32_t*)&sharedSourcePoints[0];
}

uint32_t* scalarsAddress() {
  return (uint32_t*)&sharedScalars[0];
}

#endif

// We are going to do a two phase binning sort.  In the first phase, we will bin 64 Pippinger
// buckets into each bin.

uint32_t sort14CountBins(uint32_t workerID, I128* scalars, uint32_t scalarCount) {
  uint32_t  buckets[18], bucket, noSign, nextStart=0, i;
  uint32_t* threadStarts=&sharedThreadStarts[workerID*1024];
  uint32_t* threadCounts=&sharedThreadBinCounts[workerID*8193];

  if(workerID>=MAX_WORKERS) return 0;

  while(true) {
    i=atomic_add(&sharedScalarIndex, 1024);
    if(i>=scalarCount) break;
    threadStarts[nextStart++]=i;
    for(int j=i;j<i+1024;j++) {
      if(j>=scalarCount) break;
      modSplit14(buckets, scalars+j*2);
      for(int k=0;k<18;k++) {
        bucket=buckets[k];
        if(bucket==0xFFFFFFFF)
          continue;
        noSign=bucket & 0x7FFFFFFF;
        threadCounts[noSign>>6]++;
      }
    }
  }
  return nextStart;
}

uint32_t sort16CountBins(uint32_t workerID, I128* scalars, uint32_t scalarCount) {
  uint32_t  buckets[16], bucket, noSign, nextStart=0, i;
  uint32_t* threadStarts=&sharedThreadStarts[workerID*1024];
  uint32_t* threadCounts=&sharedThreadBinCounts[workerID*8193];

  if(workerID>=MAX_WORKERS) return 0;

  while(true) {
    i=atomic_add(&sharedScalarIndex, 1024);
    if(i>=scalarCount) break;
    threadStarts[nextStart++]=i;
    for(int j=i;j<i+1024;j++) {
      if(j>=scalarCount) break;
      split16(buckets, scalars+j*2);
      for(int k=0;k<16;k++) {
        bucket=buckets[k];
        if(bucket==0xFFFFFFFF)
          continue;
        noSign=bucket & 0x7FFFFFFF;
        threadCounts[noSign>>6]++;
      }
    }
  }
  return nextStart;
}

void sortCountPrefixScan(uint32_t workerCount, uint32_t binCount) {
  uint32_t next=0;

  if(workerCount>MAX_WORKERS)
    workerCount=MAX_WORKERS;

  // Set sharedThreadBinBases to the exclusive prefix sum of sharedThreadBinCounts.
  for(int j=0;j<binCount;j++) {
    sharedBinBases[j]=next;
    for(int k=0;k<workerCount;k++) {
      sharedThreadBinBases[k*8193 + j]=next;
      next+=sharedThreadBinCounts[k*8193 + j];
    }
    sharedBinCounts[j]=next-sharedBinBases[j];
  }
  sharedBinBases[binCount]=next;
}

void sort14PartitionBins(uint32_t workerID, I128* scalars, uint32_t scalarCount, uint32_t startCount) {
  uint32_t  i, buckets[18], bucket, noSign, sign;
  uint32_t* threadStarts=&sharedThreadStarts[workerID*1024];
  uint32_t* threadBases=&sharedThreadBinBases[workerID*8193];

  if(workerID>=MAX_WORKERS) return;

  for(int s=0;s<startCount;s++) {
    i=threadStarts[s];
    for(int j=i;j<i+1024;j++) {
      if(j>=scalarCount) break;
      split14(buckets, scalars+j*2);
      for(int k=0;k<18;k++) {
        bucket=buckets[k];
        if(bucket==0xFFFFFFFF)
          continue;
        noSign=bucket & 0x7FFFFFFF;
        sign=bucket & 0x80000000;
        sharedBinAdditions[threadBases[noSign>>6]++] = j + ((bucket & 0x3F)<<21) + (sign>>11);
      }
    }
  }
}

void sort16PartitionBins(uint32_t workerID, I128* scalars, uint32_t scalarCount, uint32_t startCount) {
  uint32_t  i, buckets[16], bucket, noSign, sign;
  uint32_t* threadStarts=&sharedThreadStarts[workerID*1024];
  uint32_t* threadBases=&sharedThreadBinBases[workerID*8193];

  if(workerID>=MAX_WORKERS) return;

  for(int s=0;s<startCount;s++) {
    i=threadStarts[s];
    for(int j=i;j<i+1024;j++) {
      if(j>=scalarCount) break;
      split16(buckets, scalars+j*2);
      for(int k=0;k<16;k++) {
        bucket=buckets[k];
        if(bucket==0xFFFFFFFF)
          continue;
        noSign=bucket & 0x7FFFFFFF;
        sign=bucket & 0x80000000;
        sharedBinAdditions[threadBases[noSign>>6]++] = j + ((bucket & 0x3F)<<21) + (sign>>11);
      }
    }
  }
}

void sortBin(uint32_t bin, uint32_t left, uint32_t right) {
  uint32_t* localBucketCounts = &sharedBucketCounts[64*bin];
  uint32_t* localBucketBases = &sharedBucketBases[64*bin];
  uint32_t  localBases[64];
  uint32_t  bucket;

  // Use bin sorting again to sort the group into individual buckets.  We could write a more complex version of this
  // algorithm that worked in-place (and save 72MB of memory).  But for the sake of simplicity we do out of place sorting.

  // zero out the counts
  for(int i=0;i<64;i++)
    localBucketCounts[i]=0;

  // count the number in each of the 64 sub-bins
  for(int i=left;i<right;i++)
    localBucketCounts[sharedBinAdditions[i]>>21]++;

  // generate a prefix sum
  localBases[0]=left;
  localBucketBases[0]=left;
  for(int i=1;i<64;i++) {
    localBases[i]=localBases[i-1] + localBucketCounts[i-1];
    localBucketBases[i]=localBases[i];
  }

  // sort the data
  for(int i=left;i<right;i++) {
    bucket=sharedBinAdditions[i]>>21;
    sharedBucketAdditions[localBases[bucket]++]=sharedBinAdditions[i] & 0x1FFFFF;
  }
}

void sortBins(uint32_t workerID, uint32_t binCount) {
  uint32_t i;

  while(true) {
    i=atomic_add(&sharedBinIndex, 4);
    for(int j=i;j<i+4;j++) {
      if(j>=binCount) return;
      sortBin(j, sharedBinBases[j], sharedBinBases[j+1]);
    }
  }
}

void processPoints(uint32_t workerID, uint32_t pointCount) {
  PointPairXYT_t pair;
  FieldPair_t    rSquared;
  uint32_t       i;

  rSquared.v[0]=D2I(f64x2_splat(490307797410143.0));              // rSquared mod P
  rSquared.v[1]=D2I(f64x2_splat(1586333332603263.0));
  rSquared.v[2]=D2I(f64x2_splat(585170321528478.0));
  rSquared.v[3]=D2I(f64x2_splat(1638276889425352.0));
  rSquared.v[4]=D2I(f64x2_splat(87054710614137.0));

  if((pointCount & 0x01)!=0) {
    sharedSourcePoints[pointCount*4+0]=i64x2_splat(0L);
    sharedSourcePoints[pointCount*4+1]=i64x2_splat(0L);
    sharedSourcePoints[pointCount*4+2]=i64x2_make(0xC396FFFFFFFFFFFAL, 0xE60136071FFFFFF9L);
    sharedSourcePoints[pointCount*4+3]=i64x2_make(0xBBC63149D6B1DFF7L, 0x0FFB9FC862F41FF9L);
  }

  while(true) {
    i=atomic_add(&sharedMontgomeryIndex, 64);
    if(i>=pointCount)
      break;
    for(int j=i;j<i+64;j+=2) {
      if(j>=pointCount)
        break;
      fieldPairLoad(&pair.x, &sharedSourcePoints[j*4+0], &sharedSourcePoints[j*4+4]);
      fieldPairLoad(&pair.y, &sharedSourcePoints[j*4+2], &sharedSourcePoints[j*4+6]);

      fieldPairConvert(&pair.x, &pair.x);
      fieldPairConvert(&pair.y, &pair.y);
      fieldPairMul(&pair.x, &pair.x, &rSquared);
      fieldPairMul(&pair.y, &pair.y, &rSquared);
      fieldPairReduceResolve(&pair.x, &pair.x);
      fieldPairReduceResolve(&pair.y, &pair.y);

      fieldPairConvertMul(&pair.t, &pair.x, &pair.y);
      fieldPairFullReduceResolve(&pair.t, &pair.t);

      pointPairXYT_store(&sharedPoints[j*7+0], &sharedPoints[j*7+7], &pair);
    }
  }
}

uint32_t nextBucket(uint32_t i, uint32_t* last, I128* halfAcc, int32_t* bucket, uint32_t* count, uint32_t* base, uint32_t bucketCount) {
  uint32_t point, localCount, localBase, stop;
  bool     negate;

  localCount=sharedBucketCounts[i];
  while(localCount<=1 || i==*last) {
    if(i==*last) {
      i=atomic_add(&sharedAccumulationIndex, 512);
      stop=i+512;
      *last=(stop>bucketCount) ? bucketCount : stop;
      if(i>=bucketCount) {
        *bucket=-1;
        return bucketCount;
      }
      localCount=sharedBucketCounts[i];
    }
    else {
      if(localCount==0)
        pointXYTZ_zero(&sharedBuckets[i*10]);
      else {
        point=sharedBucketAdditions[sharedBucketBases[i]];
        negate=(point>>20)!=0;
        point=point & 0xFFFFF;
        pointXYTZ_prepare(&sharedBuckets[i*10], &sharedPoints[point*7], negate);
      }
      localCount=sharedBucketCounts[++i];
    }
  }
  localBase=sharedBucketBases[i];
  point=sharedBucketAdditions[localBase];
  negate=(point>>20)!=0;
  point=point & 0xFFFFF;
  pointXYTZ_prepare(halfAcc, &sharedPoints[point*7], negate);
  *bucket=i++;
  *count=localCount-1;
  *base=localBase+1;
  return i;
}

void accumulateBuckets(uint32_t workerID, uint32_t bucketCount) {
  PointPairXYTZ_t acc;
  PointPairXYT_t  pt;
  uint32_t        count, countL=0, countH=0, last=0;
  int32_t         bucketL, bucketH;
  uint32_t        baseL, baseH;
  uint32_t        pointL, pointH;
  bool            negateL, negateH;
  I128            ptL[10], ptH[10];
  int             i=0;

  while(true) {
    if(countL==0)
      i=nextBucket(i, &last, ptL, &bucketL, &countL, &baseL, bucketCount);
    if(countH==0)
      i=nextBucket(i, &last, ptH, &bucketH, &countH, &baseH, bucketCount);
    if((bucketL | bucketH)==-1)
      break;
    pointPairXYTZ_loadXYTZ(&acc, ptL, ptH);
    count=countL<=countH ? countL : countH;
    for(int k=0;k<count;k++) {
      pointL=sharedBucketAdditions[baseL++];
      negateL=(pointL>>20)!=0;
      pointL=pointL & 0xFFFFF;
      pointH=sharedBucketAdditions[baseH++];
      negateH=(pointH>>20)!=0;
      pointH=pointH & 0xFFFFF;
      pointXYT_prepare(ptL, &sharedPoints[pointL*7], negateL);
      pointXYT_prepare(ptH, &sharedPoints[pointH*7], negateH);
      pointPairXYT_load(&pt, ptL, ptH);
      pointPairXYTZ_accumulateXYT(&acc, &pt);
    }
    countL-=count;
    countH-=count;
    pointPairXYTZ_store(ptL, ptH, &acc);
    if(countL==0) {
      for(int j=0;j<10;j++)
        sharedBuckets[bucketL*10+j]=ptL[j];
    }
    if(countH==0) {
      for(int j=0;j<10;j++)
        sharedBuckets[bucketH*10+j]=ptH[j];
    }
  }

  if(bucketL!=-1 || bucketH!=-1) {
    if(bucketH!=-1) {
      bucketL=bucketH;
      countL=countH;
      baseL=baseH;
      for(int j=0;j<10;j++)
        ptL[j]=ptH[j];
    }

    pointXYTZ_zero(ptH);
    pointPairXYTZ_loadXYTZ(&acc, ptL, ptL);
    for(int k=0;k<countL;k++) {
      pointL=sharedBucketAdditions[baseL++];
      negateL=(pointL>>20)!=0;
      pointL=pointL & 0xFFFFF;
      pointXYT_prepare(ptL, &sharedPoints[pointL*7], negateL);
      pointPairXYT_load(&pt, ptL, ptH);
      pointPairXYTZ_accumulateXYT(&acc, &pt);
    }
    pointPairXYTZ_store(&sharedBuckets[bucketL*10], ptH, &acc);
  }
}

void reduceBuckets(uint32_t bucketCount) {
  PointPairXYTZ_t acc;           // sos in L, sum in H
  PointPairXYTZ_t pt;
  I128            zero[10];
  uint32_t        group, offset;

  pointXYTZ_zero(zero);
  while(true) {
    group=atomic_add(&sharedReductionIndex, 1);
    offset=group*1024;
    if(offset>=bucketCount)
      return;
    pointPairXYTZ_loadXYTZ(&acc, zero, &sharedBuckets[(offset+1023)*10]);
    for(int i=1022;i>=0;i--) {
      pointPairXYTZ_loadH_XYTZ(&pt, &acc, &sharedBuckets[(offset+i)*10]);
      pointPairXYTZ_accumulateXYTZ(&acc, &pt);
    }
    pointPairXYTZ_loadH_XYTZ(&pt, &acc, zero);
    pointPairXYTZ_accumulateXYTZ(&acc, &pt);
    pointPairXYTZ_store(&sharedBuckets[(SOS_BUCKETS_OFFSET+group)*10], &sharedBuckets[(SUM_BUCKETS_OFFSET+group)*10], &acc);
  }
}

void reduceWindows(uint32_t windowCount, uint32_t reductionsPerWindow, uint32_t windowBits) {
  PointPairXYTZ_t acc;           // sos in L, sum in H
  PointPairXYTZ_t pt;
  FieldPair_t     inv, xy, one;
  I128            zero[10];
  uint32_t        group, offset;
  I128            ws[MAX_WINDOWS*10], sum[10];

  pointXYTZ_zero(zero);
  for(int w=0;w<windowCount;w++) {
    pointPairXYTZ_loadXYTZ(&acc, zero, &sharedBuckets[(SUM_BUCKETS_OFFSET + w*reductionsPerWindow + reductionsPerWindow-1)*10]);
    for(int i=reductionsPerWindow-2;i>=1;i--) {
      pointPairXYTZ_loadH_XYTZ(&pt, &acc, &sharedBuckets[(SUM_BUCKETS_OFFSET + w*reductionsPerWindow + i)*10]);
      pointPairXYTZ_accumulateXYTZ(&acc, &pt);
    }
    pointPairXYTZ_loadH_XYTZ(&pt, &acc, zero);
    pointPairXYTZ_accumulateXYTZ(&acc, &pt);
    pointPairXYTZ_store(&ws[w*10], sum, &acc);
  }
  for(int w=0;w<windowCount;w+=2) {
    pointPairXYTZ_loadXYTZ(&acc, &ws[w*10], &ws[w*10+10]);
    for(int i=0;i<10;i++)
      pointPairXYTZ_accumulateXYTZ_unified(&acc, &acc);
    for(int i=0;i<reductionsPerWindow;i++) {
      pointPairXYTZ_loadXYTZ(&pt, &sharedBuckets[(SOS_BUCKETS_OFFSET + w*reductionsPerWindow + i)*10],
                                  &sharedBuckets[(SOS_BUCKETS_OFFSET + (w+1)*reductionsPerWindow + i)*10]);
      pointPairXYTZ_accumulateXYTZ(&acc, &pt);
    }
    pointPairXYTZ_store(&ws[w*10], &ws[(w+1)*10], &acc);
  }

  pointPairXYTZ_loadXYTZ(&acc, &ws[(windowCount-1)*10], zero);
  for(int w=windowCount-2;w>=0;w--) {
    for(int i=0;i<windowBits;i++)
      pointPairXYTZ_accumulateXYTZ_unified(&acc, &acc);
    pointPairXYTZ_loadXYTZ(&pt, &ws[w*10], zero);
    pointPairXYTZ_accumulateXYTZ(&acc, &pt);
  }

  fieldPairInvL(&inv, &acc.z);

  fieldPairExtractLL(&xy, &acc.x, &acc.y);
  fieldPairResolveConvertMul(&xy, &xy, &inv);

  one.v[0]=i64x2_splat(0x1L);
  one.v[1]=i64x2_splat(0x0L);
  one.v[2]=i64x2_splat(0x0L);
  one.v[3]=i64x2_splat(0x0L);
  one.v[4]=i64x2_splat(0x0L);

  fieldPairResolveConvertMul(&xy, &xy, &one);
  fieldPairFullReduceResolve(&xy, &xy);

  fieldPairStore(&sharedResult[0], &sharedResult[2], &xy);
}

void initializeCounters(uint32_t pointCount) {
  sharedPointCount=pointCount;
  sharedWorkerIndex=0;
  sharedScalarIndex=0;
  sharedBinIndex=0;
  sharedMontgomeryIndex=0;
  sharedAccumulationIndex=0;
  sharedReductionIndex=0;

  for(int i=0;i<10;i++) {
    sharedBarriers[i].lock=0;
    sharedBarriers[i].workerCount=0;
    sharedBarriers[i].checkedIn=0;
  }

  for(int i=0;i<MAX_WORKERS*8193;i++)
    sharedThreadBinCounts[i]=0;
}

void computeMSM(uint32_t bits) {
  uint32_t workerID, starts;
  uint32_t windowCount, bucketCount, reductionsPerWindow;

  if(bits==14) {
    windowCount=18;
    bucketCount=2304;
    reductionsPerWindow=8;
  }
  else if(bits==16) {
    windowCount=16;
    bucketCount=8192;
    reductionsPerWindow=32;
  }
  else
    return;

  workerID=atomic_add(&sharedWorkerIndex, 1);

  /* START OF SORTING */

    //sorting phase 1
    if(bits==14)
      starts=sort14CountBins(workerID, sharedScalars, sharedPointCount);
    else if(bits==16)
      starts=sort16CountBins(workerID, sharedScalars, sharedPointCount);

    barrier(0, workerID);

    //sorting phase 2
    if(workerID==0)
      sortCountPrefixScan(sharedWorkerIndex, bucketCount);

    barrier(1, workerID);

    //sorting phase 3

    if(bits==14)
      sort14PartitionBins(workerID, sharedScalars, sharedPointCount, starts);
    else if(bits==16)
      sort16PartitionBins(workerID, sharedScalars, sharedPointCount, starts);

    barrier(2, workerID);

    //sorting phase 4
    sortBins(workerID, bucketCount);

  /* END OF SORTING */

  barrier(3, workerID);

  processPoints(workerID, sharedPointCount);

  barrier(4, workerID);

  accumulateBuckets(workerID, 64*bucketCount);

  barrier(5, workerID);

  reduceBuckets(64*bucketCount);
  // reduceBuckets(64*8192);

  barrier(6, workerID);

  if(workerID==0)
    reduceWindows(windowCount, reductionsPerWindow, bits);

  barrier(7, workerID);
}

#if defined(X86)

#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include "Reader.c"

uint64_t clockMS() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    uint64_t milliseconds = ((uint64_t)te.tv_sec)*1000L + ((uint64_t)te.tv_usec)/1000L; // calculate milliseconds
    return milliseconds;
}

int main(int argc, const char** argv) {
  uint32_t  count, bits=14;
  I128*     points;
  I128*     scalars;
  uint64_t  timer;

  if(argc<2) {
    fprintf(stderr, "Usage: %s <count>\n", argv[0]);
    fprintf(stderr, "Reads <count> XY point values from points.hex, and <count> scalars from scalars.hex\n");
    fprintf(stderr, "and computes and prints out the MSM result.\n");
    exit(1);
  }

  count=atoi(argv[1]);
  if(argc==3)
    bits=atoi(argv[2]);

  FILE* pointsFile=fopen("points.hex", "r");
  FILE* scalarsFile=fopen("scalars.hex", "r");

  // Load the points and scalars
  for(int i=0;i<count;i++) {
    parseHex((uint8_t*)(sharedSourcePoints+i*4), pointsFile, 32);
    parseHex((uint8_t*)(sharedSourcePoints+i*4+2), pointsFile, 32);
    parseHex((uint8_t*)(sharedScalars+i*2), scalarsFile, 32);
  }
  printf("Parse complete\n");

  initializeCounters(count);
  timer=-clockMS();
  #pragma omp parallel
  {
    computeMSM(bits);
  }
  timer+=clockMS();


  printf("%016lX%016lX%016lX%016lX\n", i64x2_extract_h(sharedResult[1]), i64x2_extract_l(sharedResult[1]),
                                       i64x2_extract_h(sharedResult[0]), i64x2_extract_l(sharedResult[0]));
  printf("%016lX%016lX%016lX%016lX\n", i64x2_extract_h(sharedResult[3]), i64x2_extract_l(sharedResult[3]),
                                       i64x2_extract_h(sharedResult[2]), i64x2_extract_l(sharedResult[2]));

  printf("Workers=%d\n", sharedWorkerIndex);
  printf("Running time: %ld MS\n", timer);

}

#endif
