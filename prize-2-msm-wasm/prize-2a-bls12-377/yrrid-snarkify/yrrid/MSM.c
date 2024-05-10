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

#define BATCH_COUNT              1024        // not less than 128
#define BATCH_POINT_COUNT        256         // not less than 128
#define BATCH_RETRY_COUNT        1024
#define BATCH_UNLOCK_COUNT       4096        // should be bigger than a batch

#define MAX_N                    1048576     // 2^20
#define MAX_WORKERS              20
#define MAX_ACCUMULATE_BUCKETS   262144      // 8 windows of 32K   (16 bit window -> signed digits, means each window is 32K)
#define MAX_REDUCE_BUCKETS       16384
#define WINDOWS                  8

#include "SIMD.h"
#include "Types.h"
#include "Atomic.c"
#include "Field.h"
#include "Field.c"
#include "FieldMul.c"
#include "FieldInv.c"
#include "LambdaQR.c"
#include "Curve.c"

I128        sharedResult[12];
I128        sharedPoints[MAX_N*6];
I128        sharedScalars[MAX_N*2];
I128        sharedBuckets[MAX_ACCUMULATE_BUCKETS*6 + MAX_REDUCE_BUCKETS*6];
PointXYZZ_t sharedWindows[WINDOWS];
Batch_t     privateBatches[MAX_WORKERS];

// try to make sure these counters are on separate cache lines
uint32_t __attribute__ ((aligned (16)))    sharedPointCount;
uint32_t __attribute__ ((aligned (16)))    sharedWorkerIndex;
uint32_t __attribute__ ((aligned (16)))    sharedInitializeBucketIndex;
uint32_t __attribute__ ((aligned (16)))    sharedAccumulatePointIndex;
uint32_t __attribute__ ((aligned (16)))    sharedReduceBucketIndex;
uint32_t __attribute__ ((aligned (16)))    sharedReduceWindowIndex;
int32_t  __attribute__ ((aligned (16)))    sharedNotification;
Barrier_t __attribute__ ((aligned (16)))   sharedBarriers[6];

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

bool packedFieldMSB(I128* value) {
  return (i64x2_extract_h(value[2]) & 0x8000000000000000L)!=0;
}

bool fieldMSB(Field_t* field) {
  return (i64x2_extract_h(field->v3) & 0x800000000000L)!=0;
}

void initializeBuckets(uint32_t workerID) {
  PackedPointXY_t zeroBucket;
  uint32_t        index, base, i, j;
  I128            zero, highSet;

  if(workerID>=MAX_WORKERS) return;

  zero=i64x2_splat(0L);
  highSet=i64x2_make(0L, 0x8000000000000000L);

  while(true) {
    index=atomic_add(&sharedInitializeBucketIndex, 1024);
    if(index>=MAX_ACCUMULATE_BUCKETS+MAX_REDUCE_BUCKETS)
      break;
    base=index*6;
    for(i=0;i<6144;i+=6) {
      if(index<MAX_ACCUMULATE_BUCKETS) {
        sharedBuckets[base+i+0]=zero;
        sharedBuckets[base+i+1]=zero;
        sharedBuckets[base+i+2]=highSet;   // bucket is zero
        sharedBuckets[base+i+3]=zero;
        sharedBuckets[base+i+4]=zero;
        sharedBuckets[base+i+5]=highSet;   // bucket is locked
      }
      else if(index<MAX_ACCUMULATE_BUCKETS+MAX_REDUCE_BUCKETS) {
        sharedBuckets[base+i+0]=zero;
        sharedBuckets[base+i+1]=zero;
        sharedBuckets[base+i+2]=highSet;   // bucket is zero
        sharedBuckets[base+i+3]=zero;
        sharedBuckets[base+i+4]=zero;
        sharedBuckets[base+i+5]=zero;      // no lock flag for reduce
      }
      else
        break;
      index++;
    }
  }
}

void initializeBatch(uint32_t workerID) {
  Batch_t* batch;
  uint32_t next;

  if(workerID>=MAX_WORKERS) return;

  batch=&privateBatches[workerID];
  for(int i=0;i<BATCH_POINT_COUNT;i++) {
    next=(i==BATCH_POINT_COUNT-1) ? 0xFFFF : i+1;
    batch->points[i].nextIndex=next;
    batch->points[i].referenceCount=0;
  }
  for(int i=0;i<BATCH_COUNT;i++)
    batch->buckets[i].specialProcessing=0;
}

void split16(uint32_t* bucket, I128 scalar) {
  uint64_t l=i64x2_extract_l(scalar), h=i64x2_extract_h(scalar);
  uint64_t mask=0xFFFFL;
  uint32_t add, zeroAdd, pos, neg, shifted;
  bool     zero;

  bucket[0]=l & mask;
  bucket[1]=(l>>16) & mask;
  bucket[2]=(l>>32) & mask;
  bucket[3]=l>>48;
  bucket[4]=h & mask;
  bucket[5]=(h>>16) & mask;
  bucket[6]=(h>>32) & mask;
  bucket[7]=h>>48;

  add=0;
  shifted=0;
  for(int i=0;i<8;i++) {
    bucket[i]=bucket[i] + add;
    pos=shifted + bucket[i] - 1;
    neg=shifted + (bucket[i] ^ 0x8000FFFF);
    add=bucket[i]>>15;
    zeroAdd=bucket[i]>>16;
    zero=(bucket[i] & 0xFFFF)==0;

    bucket[i]=zero ? 0xFFFFFFFF :  (add!=0) ? neg : pos;
    shifted=shifted + 0x8000;
    add=add - zeroAdd;
  }
}

#define LOCAL_COUNT 8

void runBatch(BatchPoint_t* batchPoints, BatchBucket_t* batchBuckets, uint32_t bucketCount) {
  BatchPoint_t* batchPoint;
  Field_t*      xa;
  Field_t*      xb;
  Field_t*      ya;
  Field_t*      yb;
  Field_t       evenList, oddList, paired, zero, temp, twoN, threeN;
  Field_t       la, lb, lsa, lsb, xra, xrb;
  uint32_t      variety, pointIndex;
  bool          specialProcessing=false;

  /************************************************************************************************************
   *  This routine runs a batch of Affine EC adds.  For performance reasons the special cases, point+Infinity,
   *  Infinity+point, and Infinity+Infinity are handled by the caller.
   *
   *  Phases:
   *    1.  Write batchBuckets[i].invert -> set of Fields to invert
   *    2.  Construct the invertChain for batched modular inversion
   *    3.  Run a single modular inversion
   *    4.  Process the invertChain in reverse
   *    3.  Using the inverses, complete the EC additions
   *    4.  Process any special cases, e.g., EC double X+X or zeroing, due to X-X
   *
   *************************************************************************************************************/

  fieldSet2N(&twoN);
  fieldSet3N(&threeN);
  for(int i=0;i<bucketCount;i++) {
    pointIndex=batchBuckets[i].pointIndex;
    batchPoint=&batchPoints[pointIndex>>2];
    xa=(pointIndex & 0x02)==0 ? &batchPoint->point.x : &batchPoint->endomorphism.x;

    fieldSub(&batchBuckets[i].invert, &batchBuckets[i].bucket.x, xa);
    fieldAdd(&batchBuckets[i].invert, &batchBuckets[i].invert, &twoN);

    if(fieldMSB(&batchBuckets[i].bucket.x) || fieldIsZero(&batchBuckets[i].invert)) {
      specialProcessing=true;

      ya=(pointIndex & 0x01)==0 ? &batchPoint->point.y : &batchPoint->endomorphism.y;
      fieldAdd(&batchBuckets[i].invert, &batchBuckets[i].bucket.y, ya);

      if(fieldMSB(&batchBuckets[i].bucket.x)) {
        batchBuckets[i].specialProcessing=1;
        fieldSetOne(&batchBuckets[i].invert);
      }
      else if(fieldIsZero(&batchBuckets[i].invert)) {
        // zero the bucket case
        batchBuckets[i].specialProcessing=2;
        fieldSetOne(&batchBuckets[i].invert);
      }
      else {
        // double case
        batchBuckets[i].specialProcessing=3;
        fieldPartialResolve(&batchBuckets[i].invert);
      }
    }
  }

  // compute the inverse chain
  fieldSetZero(&zero);
  if(bucketCount>=2) {
    evenList=batchBuckets[0].invert;
    oddList=batchBuckets[1].invert;
    for(int i=3;i<bucketCount;i+=2) {
      batchBuckets[i-3].invertChain=evenList;
      batchBuckets[i-2].invertChain=oddList;
      fieldMul(&evenList, &oddList, &evenList, &batchBuckets[i-1].invert, &oddList, &batchBuckets[i].invert);
    }
    if((bucketCount & 0x01)!=0) {
      batchBuckets[bucketCount-3].invertChain=evenList;
      fieldMul(&evenList, &zero, &evenList, &batchBuckets[bucketCount-1].invert, &zero, &zero);
    }
  }
  else {
    fieldSetOne(&evenList);
    fieldSetOne(&oddList);
    if(bucketCount>0)
      evenList=batchBuckets[0].invert;
  }

  // invert evenList and oddList
  fieldMul(&paired, &zero, &evenList, &oddList, &zero, &zero);
  fieldInv(&paired, &paired);
  fieldMul(&evenList, &oddList, &paired, &oddList, &paired, &evenList);

  // run the inversion chain
  if(bucketCount>=3 && (bucketCount & 0x01)!=0) {
    int i=bucketCount-1;

    fieldMul(&evenList, &batchBuckets[i].invertChain, &evenList, &batchBuckets[i].invert, &evenList, &batchBuckets[i-2].invertChain);
  }

  for(int i=bucketCount & 0xFFFFFFFE;i>=4;i-=2) {
    fieldMul(&oddList, &batchBuckets[i-1].invertChain, &oddList, &batchBuckets[i-1].invert, &oddList, &batchBuckets[i-3].invertChain);
    fieldMul(&evenList, &batchBuckets[i-2].invertChain, &evenList, &batchBuckets[i-2].invert, &evenList, &batchBuckets[i-4].invertChain);
  }
  batchBuckets[1].invertChain=oddList;
  batchBuckets[0].invertChain=evenList;

  // using the inverses, do the EC adds
  for(int i=1;i<bucketCount;i+=2) {
    pointIndex=batchBuckets[i-1].pointIndex;
    batchPoint=&batchPoints[pointIndex>>2];
    xa=(pointIndex & 0x02)==0 ? &batchPoint->point.x : &batchPoint->endomorphism.x;     // x2
    ya=(pointIndex & 0x01)==0 ? &batchPoint->point.y : &batchPoint->endomorphism.y;     // y2
    fieldSub(&xra, &batchBuckets[i-1].bucket.y, ya);
    fieldAdd(&xra, &xra, &twoN);

    pointIndex=batchBuckets[i].pointIndex;
    batchPoint=&batchPoints[pointIndex>>2];
    xb=(pointIndex & 0x02)==0 ? &batchPoint->point.x : &batchPoint->endomorphism.x;     // x2
    yb=(pointIndex & 0x01)==0 ? &batchPoint->point.y : &batchPoint->endomorphism.y;     // y2

    fieldSub(&xrb, &batchBuckets[i].bucket.y, yb);
    fieldAdd(&xrb, &xrb, &twoN);

    fieldMul(&la, &lb, &xra, &batchBuckets[i-1].invertChain, &xrb, &batchBuckets[i].invertChain);
    fieldMul(&lsa, &lsb, &la, &la, &lb, &lb);

    fieldAdd(&xra, xa, &batchBuckets[i-1].bucket.x);
    fieldSub(&xra, &lsa, &xra);
    fieldAdd(&batchBuckets[i-1].bucket.x, &xra, &threeN);
    fieldReduce(&batchBuckets[i-1].bucket.x);
    fieldSub(&xra, xa, &xra);
    fieldAdd(&xra, &xra, &twoN);

    fieldAdd(&xrb, xb, &batchBuckets[i].bucket.x);
    fieldSub(&xrb, &lsb, &xrb);
    fieldAdd(&batchBuckets[i].bucket.x, &xrb, &threeN);
    fieldReduce(&batchBuckets[i].bucket.x);
    fieldSub(&xrb, xb, &xrb);
    fieldAdd(&xrb, &xrb, &twoN);

    fieldMul(&xra, &xrb, &xra, &la, &xrb, &lb);

    fieldSub(&xra, &xra, ya);
    fieldAdd(&batchBuckets[i-1].bucket.y, &xra, &twoN);

    fieldSub(&xrb, &xrb, yb);
    fieldAdd(&batchBuckets[i].bucket.y, &xrb, &twoN);
  }

  if((bucketCount & 0x01)!=0) {
    int i=bucketCount-1;

    pointIndex=batchBuckets[i].pointIndex;
    batchPoint=&batchPoints[pointIndex>>2];
    xa=(pointIndex & 0x02)==0 ? &batchPoint->point.x : &batchPoint->endomorphism.x;     // x2
    ya=(pointIndex & 0x01)==0 ? &batchPoint->point.y : &batchPoint->endomorphism.y;     // y2
    fieldSub(&xra, &batchBuckets[i].bucket.y, ya);
    fieldAdd(&xra, &xra, &twoN);

    fieldMul(&la, &zero, &xra, &batchBuckets[i].invertChain, &xra, &zero);
    fieldMul(&lsa, &zero, &la, &la, &la, &zero);

    fieldAdd(&xra, xa, &batchBuckets[i].bucket.x);
    fieldSub(&xra, &lsa, &xra);
    fieldAdd(&batchBuckets[i].bucket.x, &xra, &threeN);
    fieldReduce(&batchBuckets[i].bucket.x);
    fieldSub(&xra, xa, &xra);
    fieldAdd(&xra, &xra, &twoN);

    fieldMul(&xra, &zero, &la, &xra, &la, &zero);

    fieldSub(&xra, &xra, ya);
    fieldAdd(&batchBuckets[i].bucket.y, &xra, &twoN);
  }

  // handle any special cases
  if(specialProcessing) {
    for(int i=0;i<bucketCount;i++) {
      if(batchBuckets[i].specialProcessing==0)
        continue;
      if(batchBuckets[i].specialProcessing==1) {
        pointIndex=batchBuckets[i].pointIndex;
        batchPoint=&batchPoints[pointIndex>>2];
        xa=(pointIndex & 0x02)==0 ? &batchPoint->point.x : &batchPoint->endomorphism.x;     // x2
        ya=(pointIndex & 0x01)==0 ? &batchPoint->point.y : &batchPoint->endomorphism.y;     // y2
        batchBuckets[i].bucket.x=*xa;
        batchBuckets[i].bucket.y=*ya;
      }
      else if(batchBuckets[i].specialProcessing==2) {
        batchBuckets[i].bucket.x.v0=i64x2_splat(0L);
        batchBuckets[i].bucket.x.v1=i64x2_splat(0L);
        batchBuckets[i].bucket.x.v2=i64x2_splat(0L);
        batchBuckets[i].bucket.x.v3=i64x2_make(0L, 0x800000000000L);
      }
      else if(batchBuckets[i].specialProcessing==3) {
        pointIndex=batchBuckets[i].pointIndex;
        batchPoint=&batchPoints[pointIndex>>2];
        xa=(pointIndex & 0x02)==0 ? &batchPoint->point.x : &batchPoint->endomorphism.x;     // x2
        ya=(pointIndex & 0x01)==0 ? &batchPoint->point.y : &batchPoint->endomorphism.y;     // y2

        fieldMul(&xra, &zero, xa, xa, xa, &zero);

        fieldAdd(&xrb, &xra, &xra);
        fieldAdd(&xra, &xra, &xrb);

        fieldMul(&la, &zero, &xra, &batchBuckets[i].invertChain, &xra, &zero);
        fieldMul(&lsa, &zero, &la, &la, &la, &zero);

        fieldAdd(&xra, xa, xa);
        fieldSub(&xra, &lsa, &xra);
        fieldAdd(&batchBuckets[i].bucket.x, &xra, &threeN);
        fieldReduce(&batchBuckets[i].bucket.x);
        fieldSub(&xra, xa, &xra);
        fieldAdd(&xra, &xra, &twoN);

        fieldMul(&xra, &zero, &la, &xra, &la, &zero);

        fieldSub(&xra, &xra, ya);
        fieldAdd(&batchBuckets[i].bucket.y, &xra, &twoN);
      }
      batchBuckets[i].specialProcessing=0;
    }
  }
}

void accumulatePoints(uint32_t workerID) {
  BatchPoint_t*  batchPoints;
  BatchBucket_t* batchBuckets;
  BatchRetry_t*  retryListA;
  BatchRetry_t*  retryListB;
  uint32_t*      batchUnlocks;
  BatchRetry_t*  swap;

  uint32_t       localIndexes[LOCAL_COUNT];
  PointXY_t      localPoints[LOCAL_COUNT];
  I128           localScalars[LOCAL_COUNT*2];

  uint32_t       localNext, localCount, retryCount, head, pointCount, nextRetry, nextBucket, referenceCount, nextUnlock, bucketIndex, pointIndex, variety;
  Field_t        zero, phi, twoN;
  I128*          bucketPtr;
  Field_t*       x;
  Field_t*       y;

  if(workerID>=MAX_WORKERS) return;

  batchBuckets=privateBatches[workerID].buckets;
  batchPoints=privateBatches[workerID].points;
  retryListA=privateBatches[workerID].retries;
  retryListB=retryListA + BATCH_RETRY_COUNT;
  batchUnlocks=privateBatches[workerID].unlocks;

  localNext=0;
  retryCount=0;
  pointCount=0;
  head=0;

  fieldSetZero(&zero);
  fieldSetPhi(&phi);
  fieldSet2N(&twoN);
  while(true) {
    if(localNext>=sharedPointCount && retryCount==0)
      break;

    nextBucket=0;
    nextRetry=0;
    nextUnlock=0;

    // process the retry list
    for(int i=0;i<retryCount;i++) {
      pointIndex=retryListA[i].pointIndex>>2;
      variety=retryListA[i].pointIndex & 0x03;
      bucketIndex=retryListA[i].bucketIndex;
      bucketPtr=&sharedBuckets[bucketIndex*6];
      if(nextBucket<BATCH_COUNT && atomic_lock_bucket(bucketPtr)) {
        batchBuckets[nextBucket].pointIndex=retryListA[i].pointIndex;
        batchBuckets[nextBucket].bucketIndex=bucketIndex;
        pointXYLoad(&batchBuckets[nextBucket].bucket, bucketPtr);
        if(packedFieldMSB(bucketPtr)) {
          // the bucket is zero, do a copy, schedule an unlock
          x=((variety & 0x02)==0) ? &batchPoints[pointIndex].point.x : &batchPoints[pointIndex].endomorphism.x;
          y=((variety & 0x01)==0) ? &batchPoints[pointIndex].point.y : &batchPoints[pointIndex].endomorphism.y;
          fieldStore(bucketPtr+0, x);
          fieldStore(bucketPtr+3, y);
          batchUnlocks[nextUnlock++]=bucketIndex;
          // if the reference count hits zero, put the batchPoint on the free list
          if(--batchPoints[pointIndex].referenceCount==0) {
            batchPoints[pointIndex].nextIndex=head;
            head=pointIndex;
            pointCount--;
          }
        }
        else
          nextBucket++;
      }
      else
        retryListB[nextRetry++]=retryListA[i];
    }

    // swap retryListA and retryListB
    swap=retryListA; retryListA=retryListB; retryListB=swap;

    while(nextBucket<BATCH_COUNT && pointCount<BATCH_POINT_COUNT-LOCAL_COUNT && nextRetry<BATCH_RETRY_COUNT-LOCAL_COUNT*16 && nextUnlock<BATCH_UNLOCK_COUNT-LOCAL_COUNT*16) {
      // while we have space to fetch more points
      localNext=atomic_add(&sharedAccumulatePointIndex, LOCAL_COUNT);
      if(localNext>=sharedPointCount)
        break;

      // load LOCAL_COUNT points into batchPoints[]
      for(localCount=0;localCount<LOCAL_COUNT;localCount++) {
        if(localNext>=sharedPointCount)
          break;
        fieldLoad(&localPoints[localCount].x, &sharedPoints[localNext*6+0]);
        fieldLoad(&localPoints[localCount].y, &sharedPoints[localNext*6+3]);
        localScalars[localCount*2+0]=sharedScalars[localNext*2+0];
        localScalars[localCount*2+1]=sharedScalars[localNext*2+1];
        localIndexes[localCount]=head;
        head=batchPoints[head].nextIndex;
        localNext++;
      }

      // convert them to Montgomery form
      for(int i=0;i<localCount;i++)
        pointXYToMontgomery(&batchPoints[localIndexes[i]].point, &localPoints[i]);

      // compute the two endomorphisms for the points, in pairs of points
      for(int i=1;i<localCount;i+=2) {
        fieldMul(&batchPoints[localIndexes[i-1]].endomorphism.x, &batchPoints[localIndexes[i]].endomorphism.x,
                 &phi, &batchPoints[localIndexes[i-1]].point.x, &phi, &batchPoints[localIndexes[i]].point.x);
        fieldSub(&batchPoints[localIndexes[i-1]].endomorphism.y, &twoN, &batchPoints[localIndexes[i-1]].point.y);
        fieldSub(&batchPoints[localIndexes[i]].endomorphism.y, &twoN, &batchPoints[localIndexes[i]].point.y);
      }

      // localCount was odd... do the last one
      if((localCount & 0x01)!=0) {
        fieldMul(&batchPoints[localIndexes[localCount-1]].endomorphism.x, &zero, &phi, &batchPoints[localIndexes[localCount-1]].point.x, &phi, &zero);
        fieldSub(&batchPoints[localIndexes[localCount-1]].endomorphism.y, &twoN, &batchPoints[localIndexes[localCount-1]].point.y);
      }

      // for each of the corresponding scalar, break it into q & r, then slice
      for(int i=0;i<localCount;i++) {
        uint32_t q[4], r[4];
        uint32_t buckets[8];

        lambdaQR(q, r, (uint32_t*)&localScalars[i*2]);

        pointIndex=localIndexes[i];
        referenceCount=0;
        x=&batchPoints[pointIndex].point.x;

        // process r, then q
        for(int qr=0;qr<4;qr+=2) {
          split16(buckets, i32x4_make(r[0], r[1], r[2], r[3]));
          // go through 8 slices
          for(int j=0;j<8;j++) {
            if(buckets[j]==0xFFFFFFFF)
              continue;
            variety=buckets[j]>>31;
            bucketIndex=buckets[j] & 0xFFFFFF;
            bucketPtr=&sharedBuckets[bucketIndex*6];
            if(nextBucket<BATCH_COUNT && atomic_lock_bucket(bucketPtr)) {
              pointXYLoad(&batchBuckets[nextBucket].bucket, bucketPtr);
              if(packedFieldMSB(bucketPtr)) {
                // bucket is empty, do a copy, schedule an unlock
                y=(variety==0) ? &batchPoints[pointIndex].point.y : &batchPoints[pointIndex].endomorphism.y;
                fieldStore(bucketPtr+0, x);
                fieldStore(bucketPtr+3, y);
                batchUnlocks[nextUnlock++]=bucketIndex;
              }
              else {
                // add it to the batch
                batchBuckets[nextBucket].pointIndex=(pointIndex<<2) + qr + variety;
                batchBuckets[nextBucket++].bucketIndex=bucketIndex;
                referenceCount++;
              }
            }
            else {
              // the batch was full, or the bucket was locked, add it to the retry list
              retryListA[nextRetry].pointIndex=(pointIndex<<2) + qr + variety;
              retryListA[nextRetry++].bucketIndex=bucketIndex;
              referenceCount++;
            }
          }
          // finished processing r, now process q
          r[0]=q[0]; r[1]=q[1]; r[2]=q[2]; r[3]=q[3];
          x=&batchPoints[pointIndex].endomorphism.x;
        }

        // if the referenceCount is greater than zero, the leave the point, else put it on the free list
        if(referenceCount>0) {
          batchPoints[pointIndex].referenceCount=referenceCount;
          pointCount++;
        }
        else {
          batchPoints[pointIndex].nextIndex=head;
          head=pointIndex;
        }
      }
    }
    retryCount=nextRetry;

    //printf("Worker=%d unlock=%d points=%d buckets=%d retry=%d\n", workerID, nextUnlock, pointCount, nextBucket, retryCount);

    // unlock copied buckets
    if(nextUnlock!=0) {
      thread_fence();
      for(int i=0;i<nextUnlock;i++)
        atomic_unlock_bucket(&sharedBuckets[batchUnlocks[i]*6]);
    }

    // run the batch
    runBatch(batchPoints, batchBuckets, nextBucket);

    // write the results back to the sharedBuckets
    for(int i=nextBucket-1;i>=0;i--) {
      bucketIndex=batchBuckets[i].bucketIndex;
      bucketPtr=&sharedBuckets[bucketIndex*6];
      fieldStore(bucketPtr+0, &batchBuckets[i].bucket.x);
      fieldStore(bucketPtr+3, &batchBuckets[i].bucket.y);
    }

    // write the buckets back
    thread_fence();
    for(int i=0;i<nextBucket;i++) {
      bucketIndex=batchBuckets[i].bucketIndex;
      pointIndex=batchBuckets[i].pointIndex>>2;
      if(--batchPoints[pointIndex].referenceCount==0) {
        batchPoints[pointIndex].nextIndex=head;
        head=pointIndex;
        pointCount--;
      }
      atomic_unlock_bucket(&sharedBuckets[bucketIndex*6]);
    }
  }
}

#define SUM_BASE 262144
#define SOS_BASE 270336

void reduceBuckets(uint32_t workerID, uint32_t msmBucketCount) {
  BatchBucket_t* batchBuckets;
  BatchPoint_t*  batchPoints;
  I128*          sourceBuckets;
  I128*          sumBuckets;
  I128*          sosBuckets;
  I128           mask64=i64x2_make(0xFFFFFFFFFFFFFFFFL, 0x7FFFFFFFFFFFFFFFL);
  uint32_t       index, bucketIndex;

  if(workerID>=MAX_WORKERS) return;

  batchBuckets=privateBatches[workerID].buckets;
  batchPoints=privateBatches[workerID].points;

  while(true) {
    index=atomic_add(&sharedReduceBucketIndex, 32*64);
    if(index>=msmBucketCount)
      return;
    sourceBuckets=&sharedBuckets[index*6];
    index=index>>5;
    sumBuckets=&sharedBuckets[(SUM_BASE + index)*6];
    sosBuckets=&sharedBuckets[(SOS_BASE + index)*6];

    // clear all of the lock flags
    for(int i=0;i<32*64;i++)
      sourceBuckets[i*6+5]=i64x2_and(sourceBuckets[i*6+5], mask64);

    for(int j=0;j<64;j++) {
      sumBuckets[j*6+0]=sourceBuckets[(j*32+31)*6+0];
      sumBuckets[j*6+1]=sourceBuckets[(j*32+31)*6+1];
      sumBuckets[j*6+2]=sourceBuckets[(j*32+31)*6+2];
      sumBuckets[j*6+3]=sourceBuckets[(j*32+31)*6+3];
      sumBuckets[j*6+4]=sourceBuckets[(j*32+31)*6+4];
      sumBuckets[j*6+5]=sourceBuckets[(j*32+31)*6+5];
    }

    for(int i=31;i>=0;i--) {
      int nextIndex=0;

      for(int j=0;j<64;j++) {
        if(packedFieldMSB(&sumBuckets[j*6]))
          continue;

        fieldLoad(&batchPoints[nextIndex].point.x, &sumBuckets[j*6+0]);
        fieldLoad(&batchPoints[nextIndex].point.y, &sumBuckets[j*6+3]);
        fieldReduce(&batchPoints[nextIndex].point.y);

        fieldLoad(&batchBuckets[nextIndex].bucket.x, &sosBuckets[j*6+0]);
        fieldLoad(&batchBuckets[nextIndex].bucket.y, &sosBuckets[j*6+3]);
        batchBuckets[nextIndex].bucketIndex=SOS_BASE + index + j;
        batchBuckets[nextIndex].pointIndex=nextIndex<<2;
        nextIndex++;
      }

      if(i>=1) {
        for(int j=0;j<64;j++) {
          if(packedFieldMSB(&sourceBuckets[(j*32+i-1)*6]))
            continue;

          fieldLoad(&batchPoints[nextIndex].point.x, &sourceBuckets[(j*32+i-1)*6+0]);
          fieldLoad(&batchPoints[nextIndex].point.y, &sourceBuckets[(j*32+i-1)*6+3]);
          fieldReduce(&batchPoints[nextIndex].point.y);

          fieldLoad(&batchBuckets[nextIndex].bucket.x, &sumBuckets[j*6+0]);
          fieldLoad(&batchBuckets[nextIndex].bucket.y, &sumBuckets[j*6+3]);
          batchBuckets[nextIndex].bucketIndex=SUM_BASE + index + j;
          batchBuckets[nextIndex].pointIndex=nextIndex<<2;
          nextIndex++;
        }
      }

      runBatch(batchPoints, batchBuckets, nextIndex);

      for(int j=0;j<nextIndex;j++) {
        bucketIndex=batchBuckets[j].bucketIndex;
        fieldStore(&sharedBuckets[bucketIndex*6+0], &batchBuckets[j].bucket.x);
        fieldStore(&sharedBuckets[bucketIndex*6+3], &batchBuckets[j].bucket.y);
      }
    }
  }
}

void reduceWindows(uint32_t workerID) {
  PointXYZZ_t  windowSum, sum;
  PointXY_t    point;
  uint32_t     index;
  I128*        sumBuckets;
  I128*        sosBuckets;

  if(workerID>=MAX_WORKERS) return;

  while(true) {
    index=atomic_add(&sharedReduceWindowIndex, 1);
    if(index>=WINDOWS) return;

    pointXYZZInitialize(&sum);
    pointXYZZInitialize(&windowSum);

    sumBuckets=&sharedBuckets[(SUM_BASE + index*1024)*6];
    sosBuckets=&sharedBuckets[(SOS_BASE + index*1024)*6];

    for(int i=1023;i>=1;i--) {
      if(!packedFieldMSB(&sumBuckets[i*6])) {
        fieldLoad(&point.x, &sumBuckets[i*6+0]);
        fieldLoad(&point.y, &sumBuckets[i*6+3]);
        fieldReduce(&point.y);
        pointXYZZAccumulateXY(&sum, &point);
      }
      pointXYZZAccumulateXYZZ(&windowSum, &sum);
    }

    for(int i=0;i<5;i++)
      pointXYZZDoubleXYZZ(&windowSum, &windowSum);

    for(int i=0;i<1024;i++) {
      if(packedFieldMSB(&sosBuckets[i*6]))
        continue;
      fieldLoad(&point.x, &sosBuckets[i*6+0]);
      fieldLoad(&point.y, &sosBuckets[i*6+3]);
      fieldReduce(&point.y);
      pointXYZZAccumulateXY(&windowSum, &point);
    }
    sharedWindows[index]=windowSum;
  }
}

#if defined(WASM)

void barrier(uint32_t barrier, uint32_t workerID) {
  uint32_t prior, workers;
  bool     done=false;

  while(true) {
    thread_fence();
    prior=atomic_or(&sharedBarriers[barrier].lock, 0x01);
    if(prior==0) {
      if(sharedBarriers[barrier].workerCount==0) {
        sharedBarriers[barrier].workerCount=sharedWorkerIndex;
        sharedBarriers[barrier].checkedIn=1;
        done=sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount;
        break;
      }
      else {
        if(workerID<sharedBarriers[barrier].workerCount)
          sharedBarriers[barrier].checkedIn++;
        done=sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount;
        break;
      }
    }
  }
  thread_fence();
  atomic_and(&sharedBarriers[barrier].lock, 0xFFFFFFFE);

  if(done) {
    __builtin_wasm_memory_atomic_notify(&sharedNotification, 0xFFFF);                // wake all threads
  }
  else {
    while(true) {
      __builtin_wasm_memory_atomic_wait32(&sharedNotification, 0, 500000);           // 500 usec
      thread_fence();
      if(sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount)
        break;
    }
  }
}

#endif

#if defined(X86)

#include <stdio.h>
#include <unistd.h>

void barrier(uint32_t barrier, uint32_t workerID) {
  uint32_t prior, workers;

  while(true) {
    thread_fence();
    prior=atomic_or(&sharedBarriers[barrier].lock, 0x01);
    if(prior==0) {
      if(sharedBarriers[barrier].workerCount==0) {
        sharedBarriers[barrier].workerCount=sharedWorkerIndex;
        sharedBarriers[barrier].checkedIn=1;
        break;
      }
      else {
        if(workerID<sharedBarriers[barrier].workerCount)
          sharedBarriers[barrier].checkedIn++;
        break;
      }
    }
  }
  thread_fence();
  atomic_and(&sharedBarriers[barrier].lock, 0xFFFFFFFE);
  while(true) {
    thread_fence();
    if(sharedBarriers[barrier].checkedIn>=sharedBarriers[barrier].workerCount)
      break;
    usleep(500);
  }
}

#endif

uint32_t* resultAddress() {
  return (uint32_t*)&sharedResult[0];
}

uint32_t* pointsAddress() {
  return (uint32_t*)&sharedPoints[0];
}

uint32_t* scalarsAddress() {
  return (uint32_t*)&sharedScalars[0];
}

void initializeCounters(uint32_t pointCount) {
  sharedPointCount=pointCount;
  sharedWorkerIndex=0;
  sharedInitializeBucketIndex=0;
  sharedAccumulatePointIndex=0;
  sharedReduceBucketIndex=0;
  sharedReduceWindowIndex=0;
  sharedNotification=0;
  for(int i=0;i<6;i++) {
    sharedBarriers[i].lock=0;
    sharedBarriers[i].workerCount=0;
    sharedBarriers[i].checkedIn=0;
  }
}

void computeMSM() {
  uint32_t workerID;

  workerID=atomic_add(&sharedWorkerIndex, 1);

  initializeBuckets(workerID);

  barrier(0, workerID);

  initializeBatch(workerID);

  barrier(1, workerID);

  accumulatePoints(workerID);

  barrier(2, workerID);

  reduceBuckets(workerID, 262144);

  barrier(3, workerID);

  reduceWindows(workerID);

  barrier(4, workerID);

  if(workerID==0) {
    PointXYZZ_t msm;
    PointXY_t   result;

    msm=sharedWindows[7];
    for(int i=6;i>=0;i--) {
      for(int j=0;j<16;j++)
        pointXYZZDoubleXYZZ(&msm, &msm);
      pointXYZZAccumulateXYZZ(&msm, &sharedWindows[i]);
    }

    pointXYZZNormalize(&result, &msm);
    pointXYFromMontgomery(&result, &result);
    fieldStore(sharedResult+0, &result.x);
    fieldStore(sharedResult+3, &result.y);
  }

  barrier(5, workerID);
}

#if defined(X86)

#include "Reader.c"
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>

void dump(Field_t* field) {
  Field_t          resolved=*field;
  ExpandedField_t* ef;

  fieldResolve(&resolved);
  ef=(ExpandedField_t*)&resolved;
  printf("%012lX%012lX%012lX%012lX", ef->f7, ef->f6, ef->f5, ef->f4);
  printf("%012lX%012lX%012lX%012lX\n", ef->f3, ef->f2, ef->f1, ef->f0);
}

uint64_t clockMS() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    uint64_t milliseconds = ((uint64_t)te.tv_sec)*1000L + ((uint64_t)te.tv_usec)/1000L; // calculate milliseconds
    return milliseconds;
}

int main(int argc, const char** argv) {
  uint32_t  count;
  I128*     points;
  I128*     scalars;
  uint64_t  timer;

  if(argc!=2) {
    fprintf(stderr, "Usage: %s <count>\n", argv[0]);
    fprintf(stderr, "Reads <count> XY point values from points.hex, and <count> scalars from scalars.hex\n");
    fprintf(stderr, "and computes and prints out the MSM result.\n");
    exit(1);
  }

  count=atoi(argv[1]);

  FILE* pointsFile=fopen("points.hex", "r");
  FILE* scalarsFile=fopen("scalars.hex", "r");

  // Load the points and scalars
  for(int i=0;i<count;i++) {
    parseHex((uint8_t*)(sharedPoints+i*6), pointsFile, 48);
    parseHex((uint8_t*)(sharedPoints+i*6+3), pointsFile, 48);
    parseHex((uint8_t*)(sharedScalars+i*2), scalarsFile, 32);
  }

printf("Parse complete\n");

  initializeCounters(count);

  timer=-clockMS();
  #pragma omp parallel
  {
    computeMSM();
  }

  timer+=clockMS();

  printf("Workers=%d\n", sharedWorkerIndex);
  printf("Running time: %ld MS\n", timer);

  PointXY_t xy;

  fieldLoad(&xy.x, sharedResult+0);
  fieldLoad(&xy.y, sharedResult+3);
  dump(&xy.x);
  dump(&xy.y);
}

#endif
