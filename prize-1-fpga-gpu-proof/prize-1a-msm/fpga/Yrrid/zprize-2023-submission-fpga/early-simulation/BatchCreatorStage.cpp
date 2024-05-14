/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

#include "CollisionQueue.cpp"

enum {
  heartBeatRest=0,
  heartBeatSend6StoreA=1,
  heartBeatSend6StoreB=2,
  heartBeatSend7ABC=3,
  heartBeatSend7AB=4,
};

typedef struct {
  uint16_t            bucketIndex;
  uint32_t            pointIndex;
  AddSubtractEnum     addSubtract;
  PointXY<CURVE,true> point;
} ECOperation;

typedef CollisionQueue<ECOperation, 1<<WINDOW_BITS-1, 1024/6, INVALID_BUCKET> Queue;

void signedDigits(ECOperation* operations, typename CURVE::OrderField scalar, PointXY<CURVE,true> point, uint32_t pointIndex) {
  // hard coded for 14-bit windows
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
    operations[18].bucketIndex=operations[18].bucketIndex + (pointIndex<<3) & 0x1FFF;
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

void initializeHeartBeatTable(uint32_t* table) {
  // Establish the 6, 6, 7 heartbeat that drives the batch creator
  for(int i=0;i<336;i++) {
    if(i%19==0) {
      // in these 6 cycles we will send 6 groups of 3, and store a window 18 (A)
      table[i]=heartBeatSend6StoreA;
    }
    else if(i%19==6) {
      if(i==329) {
        // the last cycle is an oddball - in the last 7 cycles, we send 6 groups of 3,
        // then send windows 18 A and B
        table[i]=heartBeatSend7AB;
      }
      else {
        // in these 6 cycles we will send 6 groups of 3, and store window 18 (B)
        table[i]=heartBeatSend6StoreB;
      }
    }
    else if(i%19==12 && i<335) {
      // in these 7 cycles, we will send 6 groups of 3, then send window 18 A, B, and C
      table[i]=heartBeatSend7ABC;
    }
    else
      table[i]=heartBeatRest;
  }
}

class BatchCreatorStage {
  public:
  uint32_t                    heartBeatTable[336];

  uint32_t                    msmCount;
  uint32_t                    pointIndex;
  bool                        needsBatchSlot;
  uint32_t                    batchStartCycle;
  uint32_t                    batchSlot;
  uint32_t                    columnIndex;

  Queue*                      queues;
  uint32_t                    metaCycle=0;
  ECOperation                 operations[WINDOWS];
  CURVE::OrderField           scalar;
  PointXY<CURVE,true>         point;

  UpdateEntry                 w18A;
  UpdateEntry                 w18B;
  UpdateEntry                 w18C;

  BatchCreatorStage(uint32_t count) {
    msmCount=count;
    pointIndex=0;
    needsBatchSlot=true;
    batchSlot=0;
    batchStartCycle=0;
    columnIndex=0;

    initializeHeartBeatTable(heartBeatTable);

    queues=new Queue[WINDOWS];
  }

  bool run(Simulator& simulator, uint32_t cycle) {
    CURVE::OrderField   scalar;
    PointXY<CURVE,true> point;
    uint32_t            heartBeat;
    uint32_t            currentPointIndex;
    UpdateEntry         e1, e2, e3;

    if(needsBatchSlot) {
      if(pointIndex>=msmCount && emptyQueues(queues))
        return simulator.allSlotsEmpty() && simulator.postOfficeEmpty();
      if(!simulator.isSlotEmpty(batchSlot))
        return false;
      simulator.markSlotBusy(batchSlot);
      needsBatchSlot=false;
      batchStartCycle=cycle;
      printf("Batch start: %d\n", batchStartCycle);
    }

    if(cycle-batchStartCycle==335) {
      needsBatchSlot=true;
      batchSlot=simulator.nextSlot(batchSlot);
      simulator.postWork(cycle+2, stageDelay);
      return false;
    }

    // 6, 6, 7 heartbeat
    heartBeat=heartBeatTable[cycle-batchStartCycle];
    if(heartBeat==heartBeatRest)
      return false;

    currentPointIndex=pointIndex;
    if(pointIndex>=msmCount || refusingScalars(queues)) {
      scalar=FF<CURVE::OrderField,false>::zero();
      point=PointXY<CURVE,true>();
    }
    else {
      simulator.getNextScalarAndPoint(scalar, point);
      pointIndex++;
    }
    signedDigits(operations, scalar, point, currentPointIndex);

    for(int i=0;i<WINDOWS;i++) {
      // each queue controller decides if a collision occurred, and what to run instead
      operations[i]=queues[i].run(operations[i]);
    }

    for(int i=0;i<6;i++) {
      UpdateEntry e0, e1, e2;

      e1=UpdateEntry::batchCreator(i+0, operations[i+0].bucketIndex, operations[i+0].pointIndex, operations[i+0].addSubtract, operations[i+0].point);
      e2=UpdateEntry::batchCreator(i+6, operations[i+6].bucketIndex, operations[i+6].pointIndex, operations[i+6].addSubtract, operations[i+6].point);
      e3=UpdateEntry::batchCreator(i+12, operations[i+12].bucketIndex, operations[i+12].pointIndex, operations[i+12].addSubtract, operations[i+12].point);

      // launch the update, 10 cycles of latency
      simulator.postRowUpdate(cycle+10+i, UpdateRow::batchCreator(batchSlot, cycle-batchStartCycle+i, e1, e2, e3));

      // launch the bucket reads, 100 cycles of latency
      simulator.readBuckets(cycle+100+i, batchSlot, cycle-batchStartCycle+i, e1, e2, e3);
    }

    if(heartBeat==heartBeatSend6StoreA)
      w18A=UpdateEntry::batchCreator(18, operations[18].bucketIndex, operations[18].pointIndex, operations[18].addSubtract, operations[18].point);
    else if(heartBeat==heartBeatSend6StoreB || heartBeat==heartBeatSend7AB)
      w18B=UpdateEntry::batchCreator(18, operations[18].bucketIndex, operations[18].pointIndex, operations[18].addSubtract, operations[18].point);
    else if(heartBeat==heartBeatSend7ABC)
      w18C=UpdateEntry::batchCreator(18, operations[18].bucketIndex, operations[18].pointIndex, operations[18].addSubtract, operations[18].point);

    if(heartBeat==heartBeatSend7AB || heartBeat==heartBeatSend7ABC) {
      if(heartBeat==heartBeatSend7AB)
        w18C=UpdateEntry::batchCreator(18, INVALID_BUCKET, 0, operationAdd, PointXY<CURVE,true>());

      simulator.postRowUpdate(cycle+16, UpdateRow::batchCreator(batchSlot, cycle-batchStartCycle+6, w18A, w18B, w18C));

      simulator.readBuckets(cycle+106, batchSlot, cycle-batchStartCycle+6, w18A, w18B, w18C);
    }
    return false;
  }

  void unlock(const UpdateRow& row) {
    for(int i=0;i<3;i++) {
      if(row.entries[i].bucketIndex!=INVALID_BUCKET)
        queues[row.entries[i].windowIndex].unlock(row.entries[i].bucketIndex);
    }
  }
};
