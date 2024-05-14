/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class ECStage {
  public:
  bool             needsWork;
  uint32_t         startCycle;
  uint32_t         batchSlot;
  CURVE::MainField recycle[3][42];
  uint32_t         recycleLocation;

  typedef FF<CURVE::MainField,true> FF;

  ECStage() {
    needsWork=true;
    batchSlot=0;
  }

  void run(Simulator& simulator, uint32_t cycle) {
    uint32_t            location;
    bool                infinityOut;
    CURVE::MainField    dxInv, dx, dy, lambda, xOut, yOut;
    PointXY<CURVE,true> pt1, pt2, ecResult;
    UpdateEntry         entries[3];
    UpdateRow           row;

    if(needsWork && simulator.acceptWork(stageEC)) {
      startCycle=cycle;
      needsWork=false;
      for(int i=0;i<3;i++)
        for(int j=0;j<42;j++)
          recycle[i][j]=simulator.dUntree[i][batchSlot*CYCLES + j];
      recycleLocation=0;
    }
    if(needsWork)
      return;

    location=batchSlot*CYCLES +  cycle - startCycle;
    for(int i=0;i<3;i++) {
      dxInv=FF::mul(recycle[i][recycleLocation], simulator.ecInverse[i][location]);

      pt1=simulator.ecBucket[i][location];
      pt2=simulator.ecPoint[i][location];
      if(simulator.ecAddSubtract[i][location]==operationSubtract)
        pt2=pt2.negate();

      dx=FF::sub(pt1.x, pt2.x);
      dy=FF::sub(pt1.y, pt2.y);
      if(pt1.infinity) {
        ecResult=pt2;
        dx=FF::one();
      }
      else if(pt2.infinity) {
        ecResult=pt1;
        dx=FF::one();
      }
      else if(FF::isZero(dx)) {
        if(FF::isZero(dy))
          ecResult=PointXY<CURVE,true>();
        else {
          printf("OH SHIT!  DOUBLING EVENT!\n");
          exit(1);
        }
        dx=FF::one();
      }
      else {
        lambda=FF::mul(dy, dxInv);
        xOut=FF::sub(FF::sqr(lambda), FF::add(pt1.x, pt2.x));
        yOut=FF::sub(FF::mul(lambda, FF::sub(pt2.x, xOut)), pt2.y);
        ecResult=PointXY<CURVE,true>(xOut, yOut);
      }
      recycle[i][recycleLocation]=FF::mul(recycle[i][recycleLocation], dx);
      entries[i]=UpdateEntry::writeBucket(simulator.ecWindowIndex[i][location], simulator.ecBucketIndex[i][location], ecResult);
    }

    simulator.postRowUpdate(cycle + 100, UpdateRow::writeBuckets(entries[0], entries[1], entries[2]));
    simulator.postUnlock(cycle + 150, UpdateRow::writeBuckets(entries[0], entries[1], entries[2]));

    recycleLocation=(recycleLocation==41) ? 0 : recycleLocation++;

    if(cycle-startCycle==335) {
      needsWork=true;
      simulator.markSlotEmpty(batchSlot);
      batchSlot=simulator.nextSlot(batchSlot);
    }
  }
};
