/***

Copyright (c) 2023-2024, Yrrid Software, Inc. All rights reserved.
Dual licensed under the MIT License or the Apache License, Version 2.0.
See LICENSE for details.

Author(s):  Niall Emmart

***/

class DMultiplierStage {
  public:
  bool             needsWork;
  uint32_t         startCycle;
  uint32_t         batchSlot;
  CURVE::MainField recycle[3][42];
  uint32_t         recycleLocation;

  typedef FF<CURVE::MainField,true> FF;

  DMultiplierStage() {
    needsWork=true;
    batchSlot=0;
  }

  void run(Simulator& simulator, uint32_t cycle) {
    uint32_t            location, batchLocation;
    CURVE::MainField    dx;
    UpdateEntry         entries[3];
    PointXY<CURVE,true> addPoint;
    PointXY<CURVE,true> resultBucket;
    UpdateRow           row;

    if(needsWork && simulator.acceptWork(stageDMultiplier)) {
      startCycle=cycle;
      needsWork=false;
      for(int i=0;i<3;i++)
        for(int j=0;j<42;j++)
          recycle[i][j]=FF::one();
      recycleLocation=0;
    }
    if(needsWork)
      return;

    location=335 - (cycle-startCycle);
    batchLocation=batchSlot*CYCLES + location;
    for(int i=0;i<3;i++) {
      dx=FF::sub(simulator.dMultiplierBucketX[i][batchLocation], simulator.dMultiplierPointX[i][batchLocation]);
      if(simulator.dMultiplierPointInfinity[i][batchLocation] || simulator.dMultiplierBucketInfinity[i][batchLocation] || FF::isZero(dx))
        dx=FF::one();

      entries[i]=UpdateEntry::dMultiplier(recycle[i][recycleLocation]);
      recycle[i][recycleLocation]=FF::mul(recycle[i][recycleLocation], dx);
    }
    simulator.postRowUpdate(cycle+16, UpdateRow::dMultiplierInverses(batchSlot, location, entries[0], entries[1], entries[2]));

    if(cycle-startCycle>=294) {
      // last 42 cycles
      for(int i=0;i<3;i++)
        entries[i]=UpdateEntry::dTreeMultiplier(recycle[i][recycleLocation]);
      simulator.postRowUpdate(cycle+5, UpdateRow::dMultiplierTree(batchSlot, location, entries[0], entries[1], entries[2]));
    }

    recycleLocation=(recycleLocation==41) ? 0 : recycleLocation++;

    if(cycle-startCycle==335) {
      needsWork=true;
      simulator.postWork(cycle+100, stageDTreeMultiplier);
      batchSlot=simulator.nextSlot(batchSlot);
    }
  }
};

