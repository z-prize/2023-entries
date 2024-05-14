/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>
#include "core/core.h"
#include "CollisionQueue.cpp"

#include <map>

#define WINDOW_BITS    14
#define WINDOWS        19
#define CURVE          BLS12381
#define INVALID_BUCKET 0xFFFF
#define DELAY          53*14

typedef struct {
  uint16_t            bucketIndex;
  uint32_t            pointIndex;
  AddSubtractEnum     addSubtract;
  PointXY<CURVE,true> point;
} ECOperation;

typedef struct {
  uint16_t bucketIndex[19];
} BucketIndexes;

typedef CollisionQueue<ECOperation, 1<<WINDOW_BITS-1, 1024/6, INVALID_BUCKET> Queue;

void signedDigits(ECOperation* operations, typename CURVE::OrderField scalar, PointXY<CURVE,true> point, uint32_t pointIndex) {
  operations[0].bucketIndex=scalar.l[0] & 0x3FFF;
  operations[1].bucketIndex=(scalar.l[0]>>14) & 0x3FFF;
  operations[2].bucketIndex=(scalar.l[0]>>28) & 0x3FFF;
  operations[3].bucketIndex=(scalar.l[0]>>42) & 0x3FFF;
  operations[4].bucketIndex=(scalar.l[0]>>56) + (scalar.l[1]<<8) & 0x3FFF;
  operations[5].bucketIndex=(scalar.l[1]>>6) & 0x3FFF;
  operations[6].bucketIndex=(scalar.l[1]>>20) & 0x3FFF;
  operations[7].bucketIndex=(scalar.l[1]>>34) & 0x3FFF;
  operations[8].bucketIndex=(scalar.l[1]>>48) & 0x3FFF;
  operations[9].bucketIndex=(scalar.l[1]>>62) + (scalar.l[2]<<2) & 0x3FFF;
  operations[10].bucketIndex=(scalar.l[2]>>12) & 0x3FFF;
  operations[11].bucketIndex=(scalar.l[2]>>26) & 0x3FFF;
  operations[12].bucketIndex=(scalar.l[2]>>40) & 0x3FFF;
  operations[13].bucketIndex=(scalar.l[2]>>54) + (scalar.l[3]<<10) & 0x3FFF;
  operations[14].bucketIndex=(scalar.l[3]>>4) & 0x3FFF;
  operations[15].bucketIndex=(scalar.l[3]>>18) & 0x3FFF;
  operations[16].bucketIndex=(scalar.l[3]>>32) & 0x3FFF;
  operations[17].bucketIndex=(scalar.l[3]>>46) & 0x3FFF;
  operations[18].bucketIndex=(scalar.l[3]>>60);

  for(int i=0;i<19;i++) {
    operations[i].pointIndex=pointIndex;
    operations[i].addSubtract=operationAdd;
    operations[i].point=point;
    if(operations[i].bucketIndex>=8192) {
      operations[i].bucketIndex=16384-operations[i].bucketIndex;
      operations[i+1].bucketIndex++;
      operations[i].addSubtract=operationSubtract;
    }
    if(operations[i].bucketIndex==0)
      operations[i].bucketIndex=INVALID_BUCKET;
    else
      operations[i].bucketIndex--;
  }
  if(operations[18].bucketIndex!=INVALID_BUCKET)
    operations[18].bucketIndex=operations[18].bucketIndex + (pointIndex<<3) & 0x3FFF;
}

bool emptyQueues(Queue* queues) {
  for(int i=0;i<WINDOWS;i++)
    if(!queues[i].queueEmpty())
      return false;
  return true;
}

bool refusingScalars(Queue* queues) {
  for(int i=0;i<WINDOWS;i++)
    if(queues[i].queueFull())
      return true;
  return false;
}

void run(uint32_t n) {
  Queue*                      queues=new Queue[WINDOWS];
  uint32_t                    currentN=0;
  uint32_t                    metaCycle=0;
  ECOperation                 operations[WINDOWS];
  CURVE::OrderField           scalar;
  PointXY<CURVE,true>         point;
  BucketIndexes               bucketIndexes;
  std::map<int,BucketIndexes> unlocks;

  while(true) {
    if(currentN==n && emptyQueues(queues))
      break;

    if(unlocks.find(metaCycle)!=unlocks.end()) {
      bucketIndexes=unlocks[metaCycle];
      unlocks.erase(metaCycle);
      for(int i=0;i<WINDOWS;i++)
        queues[i].unlock(bucketIndexes.bucketIndex[i]);
    }

    if(currentN>=n || refusingScalars(queues)) {
      scalar=FF<CURVE::OrderField,false>::zero();
      point=PointXY<CURVE,true>();
    }
    else {
      // get the next scalar and point
      scalar=Random::randomField<CURVE::OrderField>();
      point=PointXY<CURVE,true>::generator();
      currentN++;
    }
    signedDigits(operations, scalar, point, metaCycle);

    for(int i=0;i<WINDOWS;i++) {
      operations[i]=queues[i].run(operations[i]);
      bucketIndexes.bucketIndex[i]=operations[i].bucketIndex;
    }

    unlocks[metaCycle+DELAY]=bucketIndexes;
    metaCycle++;
  }
  printf("Iterations: %d\n", metaCycle);

  delete[] queues;
}

int main() {
  run(1000000);
}
