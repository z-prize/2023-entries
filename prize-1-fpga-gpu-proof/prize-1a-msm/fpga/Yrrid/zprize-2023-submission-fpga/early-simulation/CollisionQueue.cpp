/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include <stdio.h>
#include <stdint.h>

#define CE_SELECT_1          0
#define CE_SELECT_2          1
#define CE_SELECT_NA         2       // internal only, CE_SELECT just requires a single bit

#define OUTPUT_NOTHING       0
#define OUTPUT_INPUT         1
#define OUTPUT_CE            2

#define ENQUEUE_NOTHING      0
#define ENQUEUE_INPUT        1
#define ENQUEUE_CE           2
#define ENQUEUE_LAST         3

#define LOCK_NOTHING         0
#define LOCK_INPUT           1
#define LOCK_CE              2

#define CE_NO_UPDATE         0
#define CE_UPDATE_INVALIDATE 1
#define CE_UPDATE_LAST       2

#define LAST_NO_UPDATE       0
#define LAST_UPDATE_DEQUEUE  1

typedef struct {
  uint32_t ce;
  uint32_t output;
  uint32_t enqueue;
  uint32_t lock;
  uint32_t ceUpdate;
  uint32_t lastUpdate;
} CollisionQueueAction_t;

#include "CollisionQueueActiontable.c"

#if 0

CollisionQueueAction_t controlAction(uint32_t actionIndex) {
  CollisionQueueAction_t action;
  bool                   inputValid, ce1Valid, ce2Valid, lastValid;
  bool                   inputMatchesCE1, inputMatchesCE2, inputLocked, lastLocked, lastBucketMatch;

  // code to generate collisionQueueActionTable

  inputValid=(actionIndex & 0x01)!=0;
  ce1Valid=(actionIndex & 0x02)!=0;
  ce2Valid=(actionIndex & 0x04)!=0;
  lastValid=(actionIndex & 0x08)!=0;
  inputMatchesCE1=(actionIndex & 0x10)!=0;
  inputMatchesCE2=(actionIndex & 0x20)!=0;
  inputLocked=(actionIndex & 0x40)!=0;
  lastLocked=(actionIndex & 0x80)!=0;
  lastBucketMatch=(actionIndex & 0x100)!=0;

  if(inputValid && !inputLocked) {
    action.output=OUTPUT_INPUT;
    action.ce=CE_SELECT_NA;
    action.enqueue=ENQUEUE_NOTHING;
    action.lock=LOCK_INPUT;
    if(inputMatchesCE1) {
      action.ce=CE_SELECT_1;
      action.enqueue=ENQUEUE_CE;
    }
    else if(inputMatchesCE2) {
      action.ce=CE_SELECT_2;
      action.enqueue=ENQUEUE_CE;
    }
  }
  else {
    action.output=OUTPUT_NOTHING;
    action.ce=CE_SELECT_NA;
    action.enqueue=inputValid ? ENQUEUE_INPUT : ENQUEUE_NOTHING;
    action.lock=LOCK_NOTHING;
    if(ce1Valid) {
      action.output=OUTPUT_CE;
      action.ce=CE_SELECT_1;
      action.lock=LOCK_CE;
    }
    else if(ce2Valid) {
      action.output=OUTPUT_CE;
      action.ce=CE_SELECT_2;
      action.lock=LOCK_CE;
    }
  }

  action.lastUpdate=LAST_NO_UPDATE;
  action.ceUpdate=CE_NO_UPDATE;
  if(action.ce==CE_SELECT_NA) {
    if(!ce1Valid && lastValid && !lastLocked && !lastBucketMatch) {
      action.ce=CE_SELECT_1;
      action.ceUpdate=CE_UPDATE_LAST;
      action.lastUpdate=LAST_UPDATE_DEQUEUE;
    }
    else if(!ce2Valid && lastValid && !lastLocked && !lastBucketMatch) {
      action.ce=CE_SELECT_2;
      action.ceUpdate=CE_UPDATE_LAST;
      action.lastUpdate=LAST_UPDATE_DEQUEUE;
    }
  }
  else {
    if(lastValid && !lastLocked && !lastBucketMatch) {
      action.ceUpdate=CE_UPDATE_LAST;
      action.lastUpdate=LAST_UPDATE_DEQUEUE;
    }
    else
      action.ceUpdate=CE_UPDATE_INVALIDATE;
  }

  if(!lastValid)
    action.lastUpdate=LAST_UPDATE_DEQUEUE;
  else if(action.enqueue==ENQUEUE_NOTHING && (lastLocked || lastBucketMatch) && (!ce1Valid || !ce2Valid)) {
    action.enqueue=ENQUEUE_LAST;
    action.lastUpdate=LAST_UPDATE_DEQUEUE;
  }

  return action;
}

#endif

template<typename Addition, uint32_t BUCKETS, uint32_t QUEUE_SIZE, uint16_t INVALID_TAG>
class CollisionQueue {
  public:
  bool     locks[BUCKETS];

  Addition cacheEntry1;
  Addition cacheEntry2;
  Addition lastDequeue;
  Addition nextEnqueue;
  Addition queue[QUEUE_SIZE];

  uint8_t    queueHead;
  uint8_t    queueTail;
  uint8_t    queueCount;

  int        OUTPUT, CE, LOCK, QUEUEACTION, LASTUPDATE, CEUPDATE;

  CollisionQueue() {
    cacheEntry1.bucketIndex=INVALID_TAG;
    cacheEntry2.bucketIndex=INVALID_TAG;
    lastDequeue.bucketIndex=INVALID_TAG;
    nextEnqueue.bucketIndex=INVALID_TAG;
    for(int i=0;i<BUCKETS;i++)
      locks[i]=false;
    queueHead=0;
    queueTail=0;
    queueCount=0;
  }

  ~CollisionQueue() {
  }

  uint32_t lockedCount() {
    uint32_t count=0;

    for(int i=0;i<8192;i++)
      if(locks[i])
        count++;
    return count;
  }

  uint8_t cacheCount() {
    uint8_t count=0;

    if(cacheEntry1.bucketIndex!=INVALID_TAG)
      count++;
    if(cacheEntry2.bucketIndex!=INVALID_TAG)
      count++;
    if(lastDequeue.bucketIndex!=INVALID_TAG)
      count++;
    return count;
  }

  uint8_t queued() {
    return cacheCount() + queueCount;
  }

  bool queueFull() {
    return cacheCount() + queueCount==QUEUE_SIZE;
  }

  bool queueEmpty() {
    return cacheCount() + queueCount==0;
  }

  void unlock(uint64_t bucketIndex) {
    if(bucketIndex!=INVALID_TAG)
      locks[bucketIndex]=false;
  }

  Addition run(Addition addition) {
    CollisionQueueAction_t action;
    bool                   inputValid, ce1Valid, ce2Valid, lastValid;
    bool                   inputMatchesCE1, inputMatchesCE2, inputLocked, lastLocked, lastBucketMatch;
    uint32_t               actionIndex;
    Addition               result;

    // note:  lastBucketMatch checks dequeued against ec1, ec2 and the input bucket.

    // inputValid is lsb, lastBucketMatch is msb
    inputValid=addition.bucketIndex!=INVALID_TAG;
    ce1Valid=cacheEntry1.bucketIndex!=INVALID_TAG;
    ce2Valid=cacheEntry2.bucketIndex!=INVALID_TAG;
    lastValid=lastDequeue.bucketIndex!=INVALID_TAG;
    inputMatchesCE1=addition.bucketIndex!=INVALID_TAG && addition.bucketIndex==cacheEntry1.bucketIndex;
    inputMatchesCE2=addition.bucketIndex!=INVALID_TAG && addition.bucketIndex==cacheEntry2.bucketIndex;
    inputLocked=addition.bucketIndex!=INVALID_TAG && locks[addition.bucketIndex];
    lastLocked=lastDequeue.bucketIndex!=INVALID_TAG && locks[lastDequeue.bucketIndex];
    lastBucketMatch=lastDequeue.bucketIndex!=INVALID_TAG && (lastDequeue.bucketIndex==addition.bucketIndex || lastDequeue.bucketIndex==cacheEntry1.bucketIndex || lastDequeue.bucketIndex==cacheEntry2.bucketIndex);

    actionIndex=(inputValid ? 0x01 : 0) + (ce1Valid ? 0x02 : 0) + (ce2Valid ? 0x04 : 0) + (lastValid ? 0x08 : 0) +
                (inputMatchesCE1 ? 0x10 : 0) + (inputMatchesCE2 ? 0x20 : 0) + (inputLocked ? 0x40 : 0) +
                (lastLocked ? 0x80 : 0) + (lastBucketMatch ? 0x100 : 0);

    // We can use the table or the code.  Here we use the table.
    // action=controlAction(actionIndex);
    action=collisionQueueActionTable[actionIndex];

    // decode the action
    if(action.output==OUTPUT_INPUT)
      result=addition;
    else if(action.output==OUTPUT_CE && action.ce==CE_SELECT_1)
      result=cacheEntry1;
    else if(action.output==OUTPUT_CE && action.ce==CE_SELECT_2)
      result=cacheEntry2;
    else if(action.output==OUTPUT_NOTHING)
      result.bucketIndex=INVALID_TAG;

    nextEnqueue.bucketIndex=INVALID_TAG;
    if(action.enqueue==ENQUEUE_INPUT)
      nextEnqueue=addition;
    else if(action.enqueue==ENQUEUE_CE && action.ce==CE_SELECT_1)
      nextEnqueue=cacheEntry1;
    else if(action.enqueue==ENQUEUE_CE && action.ce==CE_SELECT_2)
      nextEnqueue=cacheEntry2;
    else if(action.enqueue==ENQUEUE_LAST)
      nextEnqueue=lastDequeue;

    if(action.lock==LOCK_INPUT)
      locks[addition.bucketIndex]=true;
    else if(action.lock==LOCK_CE && action.ce==CE_SELECT_1)
      locks[cacheEntry1.bucketIndex]=true;
    else if(action.lock==LOCK_CE && action.ce==CE_SELECT_2)
      locks[cacheEntry2.bucketIndex]=true;

    if(action.ceUpdate==CE_UPDATE_LAST && action.ce==CE_SELECT_1)
      cacheEntry1=lastDequeue;
    else if(action.ceUpdate==CE_UPDATE_LAST && action.ce==CE_SELECT_2)
      cacheEntry2=lastDequeue;
    else if(action.ceUpdate==CE_UPDATE_INVALIDATE && action.ce==CE_SELECT_1)
      cacheEntry1.bucketIndex=INVALID_TAG;
    else if(action.ceUpdate==CE_UPDATE_INVALIDATE && action.ce==CE_SELECT_2)
      cacheEntry2.bucketIndex=INVALID_TAG;

    if(action.lastUpdate==LAST_UPDATE_DEQUEUE) {
      if(queueCount>0) {
        lastDequeue=queue[queueTail];
        queueTail=(queueTail==QUEUE_SIZE-1) ? 0 : queueTail+1;
        queueCount--;
      }
      else
        lastDequeue.bucketIndex=INVALID_TAG;
    }

    if(nextEnqueue.bucketIndex!=INVALID_TAG) {
      queue[queueHead]=nextEnqueue;
      queueHead=(queueHead==QUEUE_SIZE-1) ? 0 : queueHead+1;
      queueCount++;
    }

    return result;
  }
};
